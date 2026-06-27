// Implementation of the size-dispatched gather kernel and its extern "C"
// wrappers.
//
// Two entry points share the same size-templated kernel:
//
//  * gather_indexed: element-indexed (caller passes int64 element offsets).
//    The kernel computes byte addresses as ``base + indices[i] * sizeof(T)``
//    per element. This is the hot path used for every contiguous gather --
//    contiguous means the address math is uniform, and pushing it out to the
//    Python level would only force an 8-byte-per-element materialization that
//    we'd consume once and discard.
//
//  * gather_bytes: byte-offset (caller passes pre-computed int64 byte
//    addresses). Used for the multi-record case where addresses are
//    non-uniform (record-header skips, per-record column offsets) and the
//    Python-side searchsorted work amortizes the materialization.
//
// Both are templated on the element *type* (uint8_t/16/32/64_t) so the
// compiler can vectorize the typed loads/stores. The byte-pointer surface at
// the extern "C" boundary keeps Cython binding code simple but the inner loop
// is typed for performance.
//
// Performance levers:
//
//  * OpenMP parallel for. Loop body is independent across i; static schedule
//    keeps overhead minimal.
//  * Software prefetching. ``COLSTORE_PREFETCH`` (defined in the header
//    as ``__builtin_prefetch`` on GCC/Clang, ``_mm_prefetch`` on MSVC)
//    issues a memory hint a few iterations ahead, hiding L3/DRAM miss
//    latency for scattered loads.
//  * ``COLSTORE_RESTRICT`` pointer qualifiers (``__restrict__`` on
//    GCC/Clang, ``__restrict`` on MSVC). Compiler can assume base,
//    indices, and output do not alias, enabling vectorization of the
//    store stream.
//  * Typed loads/stores inside the templated loop. Source loads go through
//    ``load_unaligned`` -- a fixed-size memcpy, the standards-correct
//    unaligned load, compiling to a plain mov on x86 -- and output stores
//    are direct ``T`` stores into the library-allocated, aligned NumPy
//    buffer.

// sched_setaffinity / cpu_set_t (used by colstore_bind_threads_to_cpus) need
// _GNU_SOURCE defined before any system header. GNU dialects define it on the
// command line already; this guard covers strict -std=c++NN builds. Must
// precede the first include.
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "colstore/gather.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace colstore {

// Resolve OpenMP thread count for ``n_indices`` indices under a caller cap.
// The kernel is memory-bound (latency, not bandwidth -- see rule 2), so two
// rules:
//   1. Below PARALLEL_THRESHOLD the fork/join cost dwarfs the work -> serial.
//   2. Above it, scale roughly one thread per ELEMENTS_PER_THREAD elements,
//      bounded by the caller ``cap``. The gather is memory-latency-bound and
//      saturates well below core count, so for large gathers the cap (a
//      topology-derived default, refined per host by colstore.autotune) is
//      the binding limit, not the work term.
// ``cap`` <= 0 means "use the OpenMP maximum" (no colstore-imposed limit).
std::ptrdiff_t resolve_thread_count(std::ptrdiff_t n_indices, int cap) {
#ifdef _OPENMP
  if (n_indices < PARALLEL_THRESHOLD) {
    return 1;
  }
  const int omp_max = omp_get_max_threads();
  int effective_cap = (cap > 0) ? std::min(cap, omp_max) : omp_max;
  if (effective_cap < 1) {
    effective_cap = 1;
  }
  // Work-proportional thread count: ~one thread per ELEMENTS_PER_THREAD
  // elements, rounded up, with a floor of 2 threads. Everything reaching
  // this point is past PARALLEL_THRESHOLD (the early return above), so it
  // benefits from at least two threads; floor division would silently
  // force mid-sized gathers back to serial and dead-code the threshold.
  // Measured scaling flattens well below core count (the gather is
  // memory-latency-bound), so for large gathers the caller cap -- not this
  // work term -- is the binding limit. The cap's default is a topology proxy
  // (config._default_gather_thread_cap) that colstore.autotune refines per
  // host by picking the measured saturation knee.
  std::ptrdiff_t by_work =
      (n_indices + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
  std::ptrdiff_t threads =
      std::min<std::ptrdiff_t>(effective_cap, std::max<std::ptrdiff_t>(2, by_work));
  return std::max<std::ptrdiff_t>(1, threads);
#else
  (void)n_indices;
  (void)cap;
  return 1;
#endif
}

// Element-indexed gather: ``output[i] = base_as_T[indices[i]]``.
//
// The caller passes byte pointers; the kernel types the inner loop itself:
// the output is reinterpreted as T* (library-allocated, aligned), while
// source loads go through load_unaligned. T is one of the unsigned integer
// types (uint8_t/16_t/32_t/64_t); the bytes copied are agnostic to the
// user-facing dtype kind.
namespace {

// Alignment-safe typed load from the file mmap. Record bodies are packed
// with no inter-column padding, so a column's start is naturally aligned
// only if every preceding column's byte count happens to be a multiple of
// its alignment -- e.g. an odd-length int8 column followed by a float64
// column produces 8-byte loads at odd addresses. Dereferencing a misaligned
// T* is undefined behavior in C++ (even though x86 tolerates it); a
// fixed-size memcpy is the standards-correct unaligned load and compiles to
// the identical mov on x86. Outputs are library-allocated NumPy arrays and
// therefore aligned; only source loads need this.
template <typename T>
inline T load_unaligned(const std::uint8_t* address) {
  T value;
  std::memcpy(&value, address, sizeof(T));
  return value;
}

// --- Element store policy: typed fast path + generic memcpy fallback -------
//
// Every gather kernel writes one element per index through a Store, so the
// element width is a policy rather than a hard-coded type. A column whose
// itemsize is 1/2/4/8 selects TypedStore<T>, whose copy is the same
// ``reinterpret_cast<T*>(out)[i] = load_unaligned<T>(src)`` the kernels always
// used -- itemsize() is constexpr, so the emitted hot loop is unchanged. A
// fixed-width column of any other size (wide strings such as ``<U3`` = 12
// bytes or ``S10`` = 10 bytes) selects GenericStore, whose copy is a
// runtime-sized memcpy. dispatch_store maps an itemsize to the right Store at
// the extern "C" boundary, once per call and outside the hot loop; only
// itemsize <= 0 is rejected (returns -1).
template <typename T>
struct TypedStore {
  static constexpr std::int64_t itemsize() { return static_cast<std::int64_t>(sizeof(T)); }
  inline void copy(std::uint8_t* out, std::ptrdiff_t i, const std::uint8_t* src) const {
    reinterpret_cast<T*>(out)[i] = load_unaligned<T>(src);
  }
};

struct GenericStore {
  std::int64_t isz;
  inline std::int64_t itemsize() const { return isz; }
  inline void copy(std::uint8_t* out, std::ptrdiff_t i, const std::uint8_t* src) const {
    std::memcpy(out + i * isz, src, static_cast<std::size_t>(isz));
  }
};

// itemsize -> Store dispatch. The generic lambda is instantiated once per
// Store type, so each kernel is implicitly instantiated for the typed widths
// plus GenericStore; the only runtime cost is one switch per call. Returns 0
// for any itemsize > 0 (1/2/4/8 typed, all others generic) and -1 only for
// itemsize <= 0; the extern "C" entries forward that status and the Cython
// layer raises on -1.
template <typename F>
inline int dispatch_store(int itemsize, F&& f) {
  switch (itemsize) {
    case 1:
      f(TypedStore<std::uint8_t>{});
      return 0;
    case 2:
      f(TypedStore<std::uint16_t>{});
      return 0;
    case 4:
      f(TypedStore<std::uint32_t>{});
      return 0;
    case 8:
      f(TypedStore<std::uint64_t>{});
      return 0;
    default:
      if (itemsize > 0) {
        f(GenericStore{static_cast<std::int64_t>(itemsize)});
        return 0;
      }
      return -1;
  }
}

// --- Reciprocal division by a per-gather constant --------------------------
//
// The uniform-record kernels recover a record index as ``idx / rows_per_record``
// for every element. ``rows_per_record`` is invariant across the whole gather,
// so a runtime 64-bit integer division (tens of cycles on current x86-64) is
// pure waste: it loses to the generic route's branch-predicted binary search
// over a cache-resident record table. Precomputing a magic reciprocal once
// turns each division into a 64x64 high-multiply plus two shifts
// (Granlund-Montgomery, the libdivide branchfull form). The divisor is a
// positive row count and the dividend a non-negative row index, so unsigned
// arithmetic is exact for the full int64 index range.
//
// The fast path needs a 128-bit intermediate (for the one-time magic and the
// per-element high-multiply), available as __int128 on 64-bit GCC/Clang -- the
// deployment toolchain and the platforms where this gather actually runs hot.
// Where it isn't (notably MSVC, which has neither __int128 nor __builtin_clzll),
// fall back to a plain runtime division: correct everywhere, and the reciprocal
// speedup is simply not applied there.
#if defined(__SIZEOF_INT128__)

struct UniformDivisor {
  std::uint64_t magic = 0;
  std::uint32_t shift = 0;
  bool add = false;
  bool shift_only = false;  // divisor is a power of two (incl. 1)
};

inline UniformDivisor make_uniform_divisor(std::uint64_t d) {
  UniformDivisor r{};
  const std::uint32_t floor_log2 = 63u - static_cast<std::uint32_t>(__builtin_clzll(d));
  if ((d & (d - 1)) == 0) {
    r.shift_only = true;
    r.shift = floor_log2;
    return r;
  }
  const __uint128_t num = static_cast<__uint128_t>(1) << (64 + floor_log2);
  std::uint64_t proposed_m = static_cast<std::uint64_t>(num / d);
  const std::uint64_t rem =
      static_cast<std::uint64_t>(num - static_cast<__uint128_t>(proposed_m) * d);
  const std::uint64_t e = d - rem;
  if (e < (static_cast<std::uint64_t>(1) << floor_log2)) {
    r.add = false;
  } else {
    proposed_m += proposed_m;
    const std::uint64_t twice_rem = rem + rem;
    if (twice_rem >= d || twice_rem < rem) {
      proposed_m += 1;
    }
    r.add = true;
  }
  r.shift = floor_log2;
  r.magic = proposed_m + 1;
  return r;
}

inline std::uint64_t uniform_divide(std::uint64_t n, const UniformDivisor& d) {
  if (d.shift_only) {
    return n >> d.shift;
  }
  const std::uint64_t q =
      static_cast<std::uint64_t>((static_cast<__uint128_t>(d.magic) * n) >> 64);
  if (d.add) {
    const std::uint64_t t = ((n - q) >> 1) + q;
    return t >> d.shift;
  }
  return q >> d.shift;
}

#else  // portable fallback (e.g. MSVC): a plain runtime division.

struct UniformDivisor {
  std::uint64_t divisor = 1;
};

inline UniformDivisor make_uniform_divisor(std::uint64_t d) { return UniformDivisor{d}; }

inline std::uint64_t uniform_divide(std::uint64_t n, const UniformDivisor& d) {
  return n / d.divisor;
}

#endif  // __SIZEOF_INT128__

// --- Policy-based gather core ----------------------------------------------
//
// gather_core holds the parallel + prefetch + store skeleton shared by the
// scatter-family kernels; a Policy supplies the per-element byte offset.
//
// Policies are STATELESS: offset/prefetch_offset are static functions, and
// the per-call state (index arrays, record metadata, scalars) threads
// through gather_core as an individual-parameter pack rather than a struct.
// Two reasons, both measured:
//   1. restrict survives: qualifiers on function parameters carry through
//      GCC inlining; qualifiers on struct members do not, so a struct-state
//      policy forced per-element metadata reloads (the dst store could
//      alias the state).
//   2. register allocation: the OpenMP outliner captures each parameter as
//      its own field and hoists it into a register before the loop, exactly
//      like the hand-written kernels. An aggregate policy was captured
//      behind the context pointer, costing one register (a per-iteration
//      col_prefix_bytes reload plus a stack spill in the withbins kernel,
//      measured 1-5% slower on instruction-bound kernels).
// With this shape the emitted loop bodies are instruction-identical to the
// hand-written kernels.
//
// offset(i, ...) may have a side effect (the bins-recording policy writes
// bins[i]). The prefetch hint uses prefetch_offset(i, ...) when the policy
// defines one and falls back to offset otherwise, so pure policies define
// only offset. A policy whose offset HAS a side effect MUST define a pure
// prefetch_offset: the look-ahead element can belong to another thread's
// chunk, so a write there would race.

template <typename Policy, typename Void, typename... State>
struct has_prefetch_offset : std::false_type {};

template <typename Policy, typename... State>
struct has_prefetch_offset<
    Policy,
    std::void_t<decltype(Policy::prefetch_offset(std::declval<std::ptrdiff_t>(),
                                                 std::declval<std::int64_t>(),
                                                 std::declval<State>()...))>,
    State...> : std::true_type {};

template <typename Policy, typename... State>
inline constexpr bool has_prefetch_offset_v =
    has_prefetch_offset<Policy, void, State...>::value;

template <typename Store, typename Policy, typename... State>
inline void gather_core(const std::uint8_t* COLSTORE_RESTRICT base,
                        std::uint8_t* COLSTORE_RESTRICT output,
                        std::ptrdiff_t n_indices, const Store& store, int thread_cap,
                        std::ptrdiff_t prefetch_distance, State... state) {
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::ptrdiff_t i = 0; i < n_indices; ++i) {
    if (prefetch_distance > 0 && i + prefetch_distance < n_indices) {
      if constexpr (has_prefetch_offset_v<Policy, State...>) {
        COLSTORE_PREFETCH(
            base + Policy::prefetch_offset(i + prefetch_distance, store.itemsize(), state...));
      } else {
        COLSTORE_PREFETCH(base + Policy::offset(i + prefetch_distance, store.itemsize(), state...));
      }
    }
    store.copy(output, i, base + Policy::offset(i, store.itemsize(), state...));
  }
}

struct IndexedPolicy {
  static inline std::ptrdiff_t offset(std::ptrdiff_t i, std::int64_t itemsize,
                                      const std::int64_t* COLSTORE_RESTRICT indices) {
    return indices[i] * static_cast<std::ptrdiff_t>(itemsize);
  }
};

// Byte-offset gather: ``output[i]`` is the element at ``base + byte_offsets[i]``.
//
// For the multi-record reader: addresses are non-uniform and pre-computed at
// the Python level (record-header skips, per-record column offsets). Offsets
// need not be element-aligned; the store's copy is alignment-safe.
// itemsize is unused here (offsets are already in bytes); the parameter exists
// so every policy's offset has the uniform (i, itemsize, state...) signature
// and gather_entry can take them interchangeably.
struct BytesPolicy {
  static inline std::ptrdiff_t offset(std::ptrdiff_t i, std::int64_t itemsize,
                                      const std::int64_t* COLSTORE_RESTRICT byte_offsets) {
    (void)itemsize;
    return byte_offsets[i];
  }
};

// Branchless "largest r with rsr[r] <= idx" over the R+1 cumulative row
// boundaries (sorted ascending). Equivalent to
//   (std::upper_bound(rsr, rsr + len, idx) - rsr) - 1
// but the conditional pointer advance compiles to a cmov, so it has no
// data-dependent branch to mispredict. That matters because the gather calls
// this once per element with unpredictable ``idx``; a branchy binary search
// mispredicts ~50% of comparisons and is several times slower in practice.
// ``rsr`` is tiny (8*(R+1) bytes) and stays cache-resident, so the search is
// a handful of L1 loads plus cmovs.
inline std::int64_t bin_record(const std::int64_t* rsr, std::int64_t len,
                               std::int64_t idx) {
  const std::int64_t* basep = rsr;
  std::int64_t n = len;
  while (n > 1) {
    const std::int64_t half = n >> 1;
    const std::int64_t* mid = basep + half;
    basep = (*mid <= idx) ? mid : basep;
    n -= half;
  }
  return basep - rsr;
}

// Store-dispatched entry for a gather_core kernel: one statement per extern
// "C" wrapper. The state pack must match Policy::offset's parameters after the
// element index and itemsize (enforced by the static_assert).
template <typename Policy, typename... State>
inline int gather_entry(int itemsize, const std::uint8_t* base,
                        std::uint8_t* output, std::ptrdiff_t n, int thread_cap,
                        std::ptrdiff_t prefetch_distance, State... state) {
  static_assert(
      std::is_invocable_r_v<std::ptrdiff_t, decltype(&Policy::offset),
                            std::ptrdiff_t, std::int64_t, State...>,
      "state pack does not match Policy::offset(i, itemsize, ...)");
  return dispatch_store(itemsize, [&](auto store) {
    gather_core<decltype(store), Policy>(base, output, n, store, thread_cap,
                                         prefetch_distance, state...);
  });
}

// Fused multi-file fancy gather (see header for the addressing contract).
//
// Its own driver rather than gather_core: each segment lives in a different
// mmap, so the address is an absolute pointer (segment_base[s] is whole, not
// an offset from one base) -- forming a cross-mmap pointer by arithmetic from a
// single base would be undefined. Everything else is reused: bin_record (the
// segment table is the record table one level up), load_unaligned for the
// alignment-safe typed load, and the resolve_thread_count / OpenMP skeleton.
// The prefetch recomputes the segment for the look-ahead index, like the
// fused multi-record kernel; the search is cheap against the DRAM latency it
// hides.
template <typename Store>
inline void gather_segment_typed(const std::int64_t* COLSTORE_RESTRICT indices,
                                   std::uint8_t* COLSTORE_RESTRICT output,
                                   std::ptrdiff_t n_indices,
                                   const std::int64_t* COLSTORE_RESTRICT segment_starts_rows,
                                   const std::int64_t* COLSTORE_RESTRICT segment_base,
                                   std::int64_t n_segments, const Store& store, int thread_cap,
                                   std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = store.itemsize();
  const std::int64_t len = n_segments + 1;
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::ptrdiff_t i = 0; i < n_indices; ++i) {
    if (prefetch_distance > 0 && i + prefetch_distance < n_indices) {
      const std::int64_t pidx = indices[i + prefetch_distance];
      const std::int64_t ps = bin_record(segment_starts_rows, len, pidx);
      COLSTORE_PREFETCH(reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(
          segment_base[ps] + pidx * itemsize)));
    }
    const std::int64_t idx = indices[i];
    const std::int64_t s = bin_record(segment_starts_rows, len, idx);
    const std::int64_t addr = segment_base[s] + idx * itemsize;
    store.copy(output, i,
               reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(addr)));
  }
}

// Bin-recording multi-file gather: gather_segment_typed plus bins[i] = s.
// The prefetch search does NOT write bins (the look-ahead element may belong to
// another thread's chunk -- a write there would race), matching the single-file
// bins policy.
template <typename Store>
inline void gather_segment_bins_typed(const std::int64_t* COLSTORE_RESTRICT indices,
                                        std::uint8_t* COLSTORE_RESTRICT output,
                                        std::int32_t* COLSTORE_RESTRICT bins,
                                        std::ptrdiff_t n_indices,
                                        const std::int64_t* COLSTORE_RESTRICT segment_starts_rows,
                                        const std::int64_t* COLSTORE_RESTRICT segment_base,
                                        std::int64_t n_segments, const Store& store, int thread_cap,
                                        std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = store.itemsize();
  const std::int64_t len = n_segments + 1;
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::ptrdiff_t i = 0; i < n_indices; ++i) {
    if (prefetch_distance > 0 && i + prefetch_distance < n_indices) {
      const std::int64_t pidx = indices[i + prefetch_distance];
      const std::int64_t ps = bin_record(segment_starts_rows, len, pidx);
      COLSTORE_PREFETCH(reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(
          segment_base[ps] + pidx * itemsize)));
    }
    const std::int64_t idx = indices[i];
    const std::int64_t s = bin_record(segment_starts_rows, len, idx);
    bins[i] = static_cast<std::int32_t>(s);
    const std::int64_t addr = segment_base[s] + idx * itemsize;
    store.copy(output, i,
               reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(addr)));
  }
}

// Bins-provided multi-file gather: the segment is a sequential int32 read, no
// search; the prefetch look-ahead reads its bin too. ``segment_base`` is this
// column's bases.
template <typename Store>
inline void gather_segment_withbins_typed(const std::int64_t* COLSTORE_RESTRICT indices,
                                            std::uint8_t* COLSTORE_RESTRICT output,
                                            const std::int32_t* COLSTORE_RESTRICT bins,
                                            std::ptrdiff_t n_indices,
                                            const std::int64_t* COLSTORE_RESTRICT segment_base,
                                            const Store& store, int thread_cap,
                                            std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = store.itemsize();
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::ptrdiff_t i = 0; i < n_indices; ++i) {
    if (prefetch_distance > 0 && i + prefetch_distance < n_indices) {
      const std::int64_t ps = bins[i + prefetch_distance];
      COLSTORE_PREFETCH(reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(
          segment_base[ps] + indices[i + prefetch_distance] * itemsize)));
    }
    const std::int64_t s = bins[i];
    const std::int64_t addr = segment_base[s] + indices[i] * itemsize;
    store.copy(output, i,
               reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(addr)));
  }
}

// Uniform-grid multi-file gather: the searching gather_segment_typed specialized
// for a segment table whose every segment holds the same row count (the global
// last may be partial). The per-index segment is then ``s = idx / rows_per_segment``
// in closed form -- one magic-reciprocal multiply, no binary search -- and the
// load is the same ``segment_base[s] + idx * itemsize`` (the bases stay an array,
// since multi-file segments live in different mmaps). The divide is exact for
// every in-range index; the clamp guards only the degenerate idx == total.
template <typename Store>
inline void gather_segment_uniform_typed(const std::int64_t* COLSTORE_RESTRICT indices,
                                          std::uint8_t* COLSTORE_RESTRICT output,
                                          std::ptrdiff_t n_indices,
                                          std::int64_t rows_per_segment,
                                          const std::int64_t* COLSTORE_RESTRICT segment_base,
                                          std::int64_t n_segments, const Store& store,
                                          int thread_cap, std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = store.itemsize();
  const UniformDivisor div = make_uniform_divisor(static_cast<std::uint64_t>(rows_per_segment));
  const std::int64_t last = n_segments - 1;
  const auto segment_of = [&](std::int64_t idx) -> std::int64_t {
    const std::int64_t s =
        static_cast<std::int64_t>(uniform_divide(static_cast<std::uint64_t>(idx), div));
    return s < last ? s : last;
  };
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::ptrdiff_t i = 0; i < n_indices; ++i) {
    if (prefetch_distance > 0 && i + prefetch_distance < n_indices) {
      const std::int64_t pidx = indices[i + prefetch_distance];
      COLSTORE_PREFETCH(reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(
          segment_base[segment_of(pidx)] + pidx * itemsize)));
    }
    const std::int64_t idx = indices[i];
    const std::int64_t addr = segment_base[segment_of(idx)] + idx * itemsize;
    store.copy(output, i,
               reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(addr)));
  }
}

// Bin-recording uniform-grid gather: gather_segment_uniform_typed plus bins[i] = s,
// so the trailing columns of a multi-column read reuse the segment via
// gather_segment_withbins (a sequential int32 read, no divide). The prefetch does
// NOT write bins -- the look-ahead element may belong to another thread's chunk.
template <typename Store>
inline void gather_segment_uniform_bins_typed(const std::int64_t* COLSTORE_RESTRICT indices,
                                              std::uint8_t* COLSTORE_RESTRICT output,
                                              std::int32_t* COLSTORE_RESTRICT bins,
                                              std::ptrdiff_t n_indices,
                                              std::int64_t rows_per_segment,
                                              const std::int64_t* COLSTORE_RESTRICT segment_base,
                                              std::int64_t n_segments, const Store& store,
                                              int thread_cap, std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = store.itemsize();
  const UniformDivisor div = make_uniform_divisor(static_cast<std::uint64_t>(rows_per_segment));
  const std::int64_t last = n_segments - 1;
  const auto segment_of = [&](std::int64_t idx) -> std::int64_t {
    const std::int64_t s =
        static_cast<std::int64_t>(uniform_divide(static_cast<std::uint64_t>(idx), div));
    return s < last ? s : last;
  };
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::ptrdiff_t i = 0; i < n_indices; ++i) {
    if (prefetch_distance > 0 && i + prefetch_distance < n_indices) {
      const std::int64_t pidx = indices[i + prefetch_distance];
      COLSTORE_PREFETCH(reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(
          segment_base[segment_of(pidx)] + pidx * itemsize)));
    }
    const std::int64_t idx = indices[i];
    const std::int64_t s = segment_of(idx);
    bins[i] = static_cast<std::int32_t>(s);
    const std::int64_t addr = segment_base[s] + idx * itemsize;
    store.copy(output, i,
               reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(addr)));
  }
}

// Sorted multi-file gather: a monotonic segment cursor instead of a per-index
// search, the fast path for a non-decreasing ``indices``. Each thread takes a
// contiguous chunk, binary-searches its first segment, then walks the cursor
// forward -- O(K + n_segments) total comparisons versus O(K log n_segments),
// with sequential within-segment access. ``segment_base`` is already each
// segment's folded absolute base, so the steady state is one compare, one
// multiply-add, one load. The look-ahead prefetch is issued only when the
// look-ahead index still lies in the current segment.
//
// The kernel is order-robust by construction, NOT contract: a step that moves
// *backward* (which can happen only when the caller's sortedness claim is
// wrong) re-locates the segment by binary search before the load, so the
// address always satisfies ``segment_starts_rows[s] <= idx < [s+1]`` and stays
// in bounds. Correct output for any index order, the cursor untouched for a
// genuinely sorted one -- so an inaccurate sortedness hint can only cost speed
// (the re-search), never memory safety. (In-range indices remain a separate
// caller guarantee, enforced upstream by the view layer.)
template <typename Store>
void gather_segment_sorted_typed(const std::int64_t* COLSTORE_RESTRICT indices,
                                   std::uint8_t* COLSTORE_RESTRICT output,
                                   std::ptrdiff_t n_indices,
                                   const std::int64_t* COLSTORE_RESTRICT segment_starts_rows,
                                   const std::int64_t* COLSTORE_RESTRICT segment_base,
                                   std::int64_t n_segments, const Store& store, int thread_cap,
                                   std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = store.itemsize();
  const std::int64_t len = n_segments + 1;
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);

  const auto walk_range = [&](std::ptrdiff_t lo, std::ptrdiff_t hi) {
    if (lo >= hi) {
      return;
    }
    std::int64_t s = bin_record(segment_starts_rows, len, indices[lo]);
    std::int64_t next_boundary = segment_starts_rows[s + 1];
    std::int64_t seg_base = segment_base[s];
    for (std::ptrdiff_t i = lo; i < hi; ++i) {
      const std::int64_t idx = indices[i];
      if (idx >= next_boundary) {
        do {
          ++s;
        } while (idx >= segment_starts_rows[s + 1]);
        next_boundary = segment_starts_rows[s + 1];
        seg_base = segment_base[s];
      } else if (idx < segment_starts_rows[s]) {
        // Backward step: indices were not non-decreasing. Re-locate the
        // segment so the read stays in bounds (the memory-safety guard).
        s = bin_record(segment_starts_rows, len, idx);
        next_boundary = segment_starts_rows[s + 1];
        seg_base = segment_base[s];
      }
      if (prefetch_distance > 0 && i + prefetch_distance < hi) {
        const std::int64_t j = indices[i + prefetch_distance];
        if (j < next_boundary) {
          COLSTORE_PREFETCH(reinterpret_cast<const std::uint8_t*>(
              static_cast<std::uintptr_t>(seg_base + j * itemsize)));
        }
      }
      store.copy(output, i,
                 reinterpret_cast<const std::uint8_t*>(
                     static_cast<std::uintptr_t>(seg_base + idx * itemsize)));
    }
  };

#ifdef _OPENMP
  if (n_threads > 1) {
#pragma omp parallel num_threads(static_cast<int>(n_threads))
    {
      const std::ptrdiff_t worker_count = omp_get_num_threads();
      const std::ptrdiff_t worker_id = omp_get_thread_num();
      const std::ptrdiff_t chunk = (n_indices + worker_count - 1) / worker_count;
      const std::ptrdiff_t lo = worker_id * chunk;
      const std::ptrdiff_t hi = std::min(n_indices, lo + chunk);
      walk_range(lo, hi);
    }
    return;
  }
#else
  (void)n_threads;
#endif
  walk_range(0, n_indices);
}

}  // namespace

// Contiguous multi-record range copy. See header for the addressing contract.
//
// The range [start, stop) is split at record boundaries; each overlapping
// record contributes one contiguous memcpy. The record holding ``start`` is
// found by binary search over ``record_starts_rows`` (R+1 cumulative counts);
// from there we walk forward, because successive records are adjacent in the
// global row space. ``write_pos`` tracks the packed output position in rows.
//
// This is deliberately serial: the work is a sequence of memcpys, and a single
// core saturates memory bandwidth on a large contiguous copy. The kernel
// exists to eliminate per-record host overhead (one np.frombuffer
// construction and one slice assignment per record in the Python fallback),
// which dominates when the range spans many small records.
void copy_multirecord_range(const std::uint8_t* COLSTORE_RESTRICT base,
                            std::uint8_t* COLSTORE_RESTRICT output,
                            std::int64_t start,
                            std::int64_t stop,
                            const std::int64_t* COLSTORE_RESTRICT record_starts_rows,
                            const std::int64_t* COLSTORE_RESTRICT record_starts_bytes,
                            const std::int64_t* COLSTORE_RESTRICT n_rows_per_record,
                            std::int64_t n_records,
                            std::int64_t col_prefix_bytes,
                            std::int64_t itemsize) {
  if (stop <= start || n_records <= 0) {
    return;
  }
  // Largest r with record_starts_rows[r] <= start. ``record_starts_rows`` is
  // sorted ascending with R+1 entries; upper_bound gives the first entry
  // strictly greater than start, so the record index is one before it.
  const std::int64_t* rows_end = record_starts_rows + n_records + 1;
  const std::int64_t* it =
      std::upper_bound(record_starts_rows, rows_end, start);
  std::int64_t r = (it - record_starts_rows) - 1;
  if (r < 0) {
    r = 0;  // start clamps to row 0 if it preceded the first record
  }

  std::int64_t write_pos = 0;  // packed output position, in rows
  for (; r < n_records; ++r) {
    const std::int64_t rec_row_start = record_starts_rows[r];
    if (rec_row_start >= stop) {
      break;  // this record begins past the requested range
    }
    const std::int64_t rec_n = n_rows_per_record[r];
    const std::int64_t rec_row_end = rec_row_start + rec_n;
    const std::int64_t within_lo =
        (start > rec_row_start ? start : rec_row_start) - rec_row_start;
    const std::int64_t within_hi =
        (stop < rec_row_end ? stop : rec_row_end) - rec_row_start;
    const std::int64_t count = within_hi - within_lo;
    if (count <= 0) {
      continue;
    }
    const std::int64_t src_off = record_starts_bytes[r] +
                                 col_prefix_bytes * rec_n +
                                 within_lo * itemsize;
    std::memcpy(output + write_pos * itemsize, base + src_off,
                static_cast<std::size_t>(count * itemsize));
    write_pos += count;
  }
}

// Strided range gather: a linear record walk over the arithmetic row stream
// ``start + i*step``. Structurally the sorted kernel with the index-array
// loads deleted -- the row is synthesized in a register, so the per-element
// steady state is one or two compares, one multiply-add, and the data load.
// The record cursor moves monotonically -- forward for ``step > 0``,
// backward for ``step < 0`` -- so each thread's total cursor movement is
// bounded by R regardless of step size. The same-record prefetch gate
// mirrors the sorted kernel's, with the look-ahead row formed
// arithmetically instead of loaded.
template <typename Store>
void gather_multirecord_strided_typed(const std::uint8_t* COLSTORE_RESTRICT base,
                                      std::uint8_t* COLSTORE_RESTRICT output,
                                      std::int64_t start,
                                      std::int64_t step,
                                      std::ptrdiff_t n_out,
                                      const std::int64_t* COLSTORE_RESTRICT record_starts_rows,
                                      const std::int64_t* COLSTORE_RESTRICT record_starts_bytes,
                                      const std::int64_t* COLSTORE_RESTRICT n_rows_per_record,
                                      std::int64_t n_records,
                                      std::int64_t col_prefix_bytes,
                                      const Store& store,
                                      int thread_cap,
                                      std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = store.itemsize();
  const std::int64_t len = n_records + 1;  // entries in record_starts_rows
  const std::ptrdiff_t n_threads = resolve_thread_count(n_out, thread_cap);
  const std::int64_t pf_jump = step * static_cast<std::int64_t>(prefetch_distance);

  const auto walk_range = [&](std::ptrdiff_t lo, std::ptrdiff_t hi) {
    if (lo >= hi) {
      return;
    }
    std::int64_t idx = start + static_cast<std::int64_t>(lo) * step;
    std::int64_t r = bin_record(record_starts_rows, len, idx);
    // Per-record state, recomputed only at record boundaries. Both boundary
    // tests are kept in one loop body (rather than templating on the walk
    // direction): for any fixed step sign exactly one of them can ever fire,
    // so the other is a perfectly predicted not-taken branch.
    std::int64_t rec_lo = record_starts_rows[r];
    std::int64_t next_boundary = record_starts_rows[r + 1];
    std::int64_t record_base = record_starts_bytes[r] +
                               col_prefix_bytes * n_rows_per_record[r] -
                               rec_lo * itemsize;
    for (std::ptrdiff_t i = lo; i < hi; ++i, idx += step) {
      if (idx >= next_boundary) {
        do {
          ++r;
        } while (idx >= record_starts_rows[r + 1]);
        rec_lo = record_starts_rows[r];
        next_boundary = record_starts_rows[r + 1];
        record_base = record_starts_bytes[r] +
                      col_prefix_bytes * n_rows_per_record[r] -
                      rec_lo * itemsize;
      } else if (idx < rec_lo) {
        do {
          --r;
        } while (idx < record_starts_rows[r]);
        rec_lo = record_starts_rows[r];
        next_boundary = record_starts_rows[r + 1];
        record_base = record_starts_bytes[r] +
                      col_prefix_bytes * n_rows_per_record[r] -
                      rec_lo * itemsize;
      }
      if (prefetch_distance > 0 && i + prefetch_distance < hi) {
        const std::int64_t j = idx + pf_jump;
        if (j >= rec_lo && j < next_boundary) {
          COLSTORE_PREFETCH(base + record_base + j * itemsize);
        }
      }
      store.copy(output, i, base + record_base + idx * itemsize);
    }
  };

#ifdef _OPENMP
  if (n_threads > 1) {
#pragma omp parallel num_threads(static_cast<int>(n_threads))
    {
      const std::ptrdiff_t worker_count = omp_get_num_threads();
      const std::ptrdiff_t worker_id = omp_get_thread_num();
      const std::ptrdiff_t chunk = (n_out + worker_count - 1) / worker_count;
      const std::ptrdiff_t lo = worker_id * chunk;
      const std::ptrdiff_t hi = std::min(n_out, lo + chunk);
      walk_range(lo, hi);
    }
    return;
  }
#endif
  walk_range(0, n_out);
}

// Uniform-record fancy gather: the fused unsorted gather with the binary
// search replaced by a constant-divisor reciprocal multiply and the three
// per-element metadata loads replaced by an affine formula. For a full record r,
//   offset(idx) = first_body_offset + r*stride + col_prefix*U
//                 + (idx - r*U)*itemsize
//               = full_base + r*per_record_step + idx*itemsize
// with full_base and per_record_step loop constants; the (possibly
// partial) final record collapses to a single precomputed base behind one
// guard, which is the only data-dependent branch and is taken only for
// indices in the last record. The prefetch look-ahead reuses the same
// arithmetic, so -- unlike the irregular kernels -- it needs no
// same-record gating: every valid index yields a valid address.
template <typename Store>
void gather_multirecord_uniform_typed(const std::uint8_t* COLSTORE_RESTRICT base,
                                      const std::int64_t* COLSTORE_RESTRICT indices,
                                      std::uint8_t* COLSTORE_RESTRICT output,
                                      std::ptrdiff_t n_indices,
                                      std::int64_t rows_per_record,
                                      std::int64_t record_stride_bytes,
                                      std::int64_t first_body_offset,
                                      std::int64_t n_records,
                                      std::int64_t last_record_rows,
                                      std::int64_t col_prefix_bytes,
                                      const Store& store,
                                      int thread_cap,
                                      std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = store.itemsize();
  const std::int64_t full_base = first_body_offset + col_prefix_bytes * rows_per_record;
  const std::int64_t per_record_step = record_stride_bytes - rows_per_record * itemsize;
  const std::int64_t last_first_row = (n_records - 1) * rows_per_record;
  const std::int64_t last_base = first_body_offset +
                                 (n_records - 1) * record_stride_bytes +
                                 col_prefix_bytes * last_record_rows -
                                 last_first_row * itemsize;
  const UniformDivisor rpr_div = make_uniform_divisor(static_cast<std::uint64_t>(rows_per_record));
  const auto offset_of = [&](std::int64_t idx) -> std::int64_t {
    if (idx >= last_first_row) {
      return last_base + idx * itemsize;
    }
    const std::int64_t r =
        static_cast<std::int64_t>(uniform_divide(static_cast<std::uint64_t>(idx), rpr_div));
    return full_base + r * per_record_step + idx * itemsize;
  };
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::ptrdiff_t i = 0; i < n_indices; ++i) {
    if (prefetch_distance > 0 && i + prefetch_distance < n_indices) {
      COLSTORE_PREFETCH(base + offset_of(indices[i + prefetch_distance]));
    }
    store.copy(output, i, base + offset_of(indices[i]));
  }
}

// Multi-column uniform pair. The bins variant pays the reciprocal divide
// once per element across the whole read; the withbins variant's per-element
// work is a sequential int32 read, one compare, one multiply-add, and the
// load -- no divide, no search, no per-record metadata. This dominates both
// the generic bins route (search -> reciprocal divide for the first column;
// three metadata loads -> affine math for the rest) and per-column
// arithmetic binning (reciprocal divide x C -> x 1).
template <typename Store>
void gather_multirecord_uniform_bins_typed(const std::uint8_t* COLSTORE_RESTRICT base,
                                           const std::int64_t* COLSTORE_RESTRICT indices,
                                           std::uint8_t* COLSTORE_RESTRICT output,
                                           std::int32_t* COLSTORE_RESTRICT bins,
                                           std::ptrdiff_t n_indices,
                                           std::int64_t rows_per_record,
                                           std::int64_t record_stride_bytes,
                                           std::int64_t first_body_offset,
                                           std::int64_t n_records,
                                           std::int64_t last_record_rows,
                                           std::int64_t col_prefix_bytes,
                                           const Store& store,
                                           int thread_cap,
                                           std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = store.itemsize();
  const std::int64_t full_base = first_body_offset + col_prefix_bytes * rows_per_record;
  const std::int64_t per_record_step = record_stride_bytes - rows_per_record * itemsize;
  const std::int64_t last_first_row = (n_records - 1) * rows_per_record;
  const std::int64_t last_base = first_body_offset +
                                 (n_records - 1) * record_stride_bytes +
                                 col_prefix_bytes * last_record_rows -
                                 last_first_row * itemsize;
  const UniformDivisor rpr_div = make_uniform_divisor(static_cast<std::uint64_t>(rows_per_record));
  const auto offset_of = [&](std::int64_t idx) -> std::int64_t {
    if (idx >= last_first_row) {
      return last_base + idx * itemsize;
    }
    const std::int64_t r =
        static_cast<std::int64_t>(uniform_divide(static_cast<std::uint64_t>(idx), rpr_div));
    return full_base + r * per_record_step + idx * itemsize;
  };
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::ptrdiff_t i = 0; i < n_indices; ++i) {
    if (prefetch_distance > 0 && i + prefetch_distance < n_indices) {
      COLSTORE_PREFETCH(base + offset_of(indices[i + prefetch_distance]));
    }
    const std::int64_t idx = indices[i];
    std::int64_t r;
    std::int64_t off;
    if (idx >= last_first_row) {
      r = n_records - 1;
      off = last_base + idx * itemsize;
    } else {
      r = static_cast<std::int64_t>(uniform_divide(static_cast<std::uint64_t>(idx), rpr_div));
      off = full_base + r * per_record_step + idx * itemsize;
    }
    bins[i] = static_cast<std::int32_t>(r);
    store.copy(output, i, base + off);
  }
}

template <typename Store>
void gather_multirecord_uniform_withbins_typed(const std::uint8_t* COLSTORE_RESTRICT base,
                                               const std::int64_t* COLSTORE_RESTRICT indices,
                                               std::uint8_t* COLSTORE_RESTRICT output,
                                               const std::int32_t* COLSTORE_RESTRICT bins,
                                               std::ptrdiff_t n_indices,
                                               std::int64_t rows_per_record,
                                               std::int64_t record_stride_bytes,
                                               std::int64_t first_body_offset,
                                               std::int64_t n_records,
                                               std::int64_t last_record_rows,
                                               std::int64_t col_prefix_bytes,
                                               const Store& store,
                                               int thread_cap,
                                               std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = store.itemsize();
  const std::int64_t full_base = first_body_offset + col_prefix_bytes * rows_per_record;
  const std::int64_t per_record_step = record_stride_bytes - rows_per_record * itemsize;
  const std::int64_t last_record = n_records - 1;
  const std::int64_t last_base = first_body_offset +
                                 last_record * record_stride_bytes +
                                 col_prefix_bytes * last_record_rows -
                                 last_record * rows_per_record * itemsize;
  const auto offset_of = [&](std::int64_t idx, std::int64_t r) -> std::int64_t {
    if (r == last_record) {
      return last_base + idx * itemsize;
    }
    return full_base + r * per_record_step + idx * itemsize;
  };
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::ptrdiff_t i = 0; i < n_indices; ++i) {
    if (prefetch_distance > 0 && i + prefetch_distance < n_indices) {
      const std::ptrdiff_t j = i + prefetch_distance;
      COLSTORE_PREFETCH(base + offset_of(indices[j], bins[j]));
    }
    store.copy(output, i, base + offset_of(indices[i], bins[i]));
  }
}

// Boolean-mask-native gather. Pass 1 counts each thread chunk's selected
// rows (a vectorizable byte sum) so output offsets are exact without
// synchronization; pass 2 walks each chunk's rows with the sorted walk's
// record-cursor machinery, skipping unselected spans 8 mask bytes at a
// time (a uint64 of zeros), extending runs of set bits 8 bytes at a time
// (a uint64 of 0x01s), and serving runs of >= 32 bytes with one memcpy --
// run detection costs nothing extra here because the mask byte is the
// datum being scanned anyway.
namespace {
constexpr std::uint64_t MASK_ALL_ONES = 0x0101010101010101ULL;

inline std::int64_t count_mask_range(const std::uint8_t* mask, std::int64_t lo,
                                     std::int64_t hi) {
  std::int64_t total = 0;
  for (std::int64_t i = lo; i < hi; ++i) {
    total += mask[i];
  }
  return total;
}
}  // namespace

// Boolean-mask-native gather over a segment table -- the one mask kernel for
// both single-file (one record = one segment) and multi-file (one record of one
// file = one segment) reads. ``segment_base[s]`` is the folded absolute base, so
// row ``i`` in segment ``s`` reads at ``segment_base[s] + i * itemsize``. Pass 1
// counts each row-chunk's selected rows for lock-free output offsets; pass 2
// walks each chunk with a monotonic segment cursor, word-at-a-time (skip all-zero
// words, memcpy all-ones runs, branchless-compact mixed words). The chunk-to-
// thread mapping uses an explicit parallel-for over all n_threads chunks (not
// omp_get_thread_num), so every chunk's output region is filled even if the team
// is granted fewer threads than requested.
template <typename Store>
int gather_segment_mask_typed(const std::uint8_t* COLSTORE_RESTRICT mask,
                                std::uint8_t* COLSTORE_RESTRICT output,
                                std::int64_t n_rows,
                                std::ptrdiff_t n_out,
                                const std::int64_t* COLSTORE_RESTRICT segment_starts_rows,
                                const std::int64_t* COLSTORE_RESTRICT segment_base,
                                std::int64_t n_segments,
                                const Store& store,
                                int thread_cap,
                                std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = store.itemsize();
  const std::int64_t len = n_segments + 1;
  const std::ptrdiff_t n_threads = resolve_thread_count(n_rows, thread_cap);
  const std::int64_t chunk = (n_rows + n_threads - 1) / n_threads;

  // Pass 1: per-chunk selected-row counts -> exclusive prefix offsets. The
  // chunk size here is reused verbatim by pass 2 so the offsets and the gather
  // partition agree.
  std::vector<std::int64_t> offsets(static_cast<std::size_t>(n_threads) + 1, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#endif
  for (std::ptrdiff_t t = 0; t < n_threads; ++t) {
    const std::int64_t lo = static_cast<std::int64_t>(t) * chunk;
    const std::int64_t hi = std::min(n_rows, lo + chunk);
    offsets[static_cast<std::size_t>(t) + 1] = (lo < hi) ? count_mask_range(mask, lo, hi) : 0;
  }
  for (std::ptrdiff_t t = 0; t < n_threads; ++t) {
    offsets[static_cast<std::size_t>(t) + 1] += offsets[static_cast<std::size_t>(t)];
  }
  if (offsets[static_cast<std::size_t>(n_threads)] != static_cast<std::int64_t>(n_out)) {
    return 1;  // caller-sized output disagrees with the mask; write nothing
  }

  const auto seg_addr = [](std::int64_t base, std::int64_t byte_offset) {
    return reinterpret_cast<const std::uint8_t*>(
        static_cast<std::uintptr_t>(base + byte_offset));
  };

  // Pass 2: each chunk seeds a segment cursor (one bin_record) and walks its
  // rows monotonically; consecutive global rows in different segments live in
  // different mmaps, so seg_end clips both the copy-run and the branchless
  // window to the current segment. The over-store guard (out + 8 <= quota)
  // keeps the branchless form inside the thread's own output region.
  const auto walk_range = [&](std::int64_t lo, std::int64_t hi, std::int64_t out,
                              std::int64_t quota) {
    if (lo >= hi) {
      return;
    }
    std::int64_t s = bin_record(segment_starts_rows, len, lo);
    std::int64_t i = lo;
    while (i < hi) {
      while (i >= segment_starts_rows[s + 1]) {
        ++s;
      }
      const std::int64_t seg_end = std::min(hi, segment_starts_rows[s + 1]);
      const std::int64_t seg_base = segment_base[s];
      while (i < seg_end) {
        if (i + 8 <= seg_end) {
          const std::uint64_t word = load_unaligned<std::uint64_t>(mask + i);
          if (word == 0) {
            i += 8;
            continue;
          }
          if (word == MASK_ALL_ONES) {
            std::int64_t j = i + 8;
            while (j + 8 <= seg_end &&
                   load_unaligned<std::uint64_t>(mask + j) == MASK_ALL_ONES) {
              j += 8;
            }
            std::memcpy(output + out * itemsize, seg_addr(seg_base, i * itemsize),
                        static_cast<std::size_t>((j - i) * itemsize));
            out += j - i;
            i = j;
            continue;
          }
          if (out + 8 <= quota) {
            for (int b = 0; b < 8; ++b) {
              store.copy(output, out, seg_addr(seg_base, (i + b) * itemsize));
              out += mask[i + b];
            }
            i += 8;
            continue;
          }
        }
        if (mask[i]) {
          store.copy(output, out, seg_addr(seg_base, i * itemsize));
          ++out;
        }
        ++i;
      }
    }
    (void)prefetch_distance;  // linear walk: hardware prefetch covers it
  };

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#endif
  for (std::ptrdiff_t t = 0; t < n_threads; ++t) {
    const std::int64_t lo = static_cast<std::int64_t>(t) * chunk;
    const std::int64_t hi = std::min(n_rows, lo + chunk);
    walk_range(lo, hi, offsets[static_cast<std::size_t>(t)],
               offsets[static_cast<std::size_t>(t) + 1]);
  }
  return 0;
}

}  // namespace colstore

extern "C" {

// Space-separated names of the optimization toggles this build compiled
// with (see CMakeLists COLSTORE_TOGGLES). Empty when none are enabled.
// Lets benchmarks/tests report exactly what they are measuring without a
// per-flag accessor.
const char* colstore_build_flags(void) {
  return
      "";
}

// The thread count a kernel would use for n_indices elements under a caller
// cap (cap <= 0 means the OpenMP maximum). Exposed so benchmarks and tests
// can report the operating point without replicating the formula.
long long colstore_resolve_thread_count(long long n_indices, int cap) {
  return static_cast<long long>(
      colstore::resolve_thread_count(static_cast<std::ptrdiff_t>(n_indices), cap));
}

int colstore_gather_indexed(const std::uint8_t* base,
                               const std::int64_t* indices,
                               std::uint8_t* output,
                               std::ptrdiff_t n, int itemsize, int thread_cap,
                               std::ptrdiff_t prefetch_distance) {
  return colstore::gather_entry<colstore::IndexedPolicy>(
      itemsize, base, output, n, thread_cap, prefetch_distance, indices);
}

int colstore_gather_bytes(const std::uint8_t* base,
                             const std::int64_t* byte_offsets,
                             std::uint8_t* output,
                             std::ptrdiff_t n, int itemsize, int thread_cap,
                               std::ptrdiff_t prefetch_distance) {
  return colstore::gather_entry<colstore::BytesPolicy>(
      itemsize, base, output, n, thread_cap, prefetch_distance, byte_offsets);
}

void colstore_copy_multirecord_range(const std::uint8_t* base,
                                     std::uint8_t* output,
                                     std::int64_t start,
                                     std::int64_t stop,
                                     const std::int64_t* record_starts_rows,
                                     const std::int64_t* record_starts_bytes,
                                     const std::int64_t* n_rows_per_record,
                                     std::int64_t n_records,
                                     std::int64_t col_prefix_bytes,
                                     std::int64_t itemsize) {
  colstore::copy_multirecord_range(base, output, start, stop,
                                   record_starts_rows, record_starts_bytes,
                                   n_rows_per_record, n_records,
                                   col_prefix_bytes, itemsize);
}

int colstore_gather_multirecord_strided(
    const std::uint8_t* base, std::uint8_t* output,
    std::int64_t start, std::int64_t step, std::ptrdiff_t n_out,
    const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int itemsize, int thread_cap,
    std::ptrdiff_t prefetch_distance) {
  return colstore::dispatch_store(itemsize, [&](auto store) {
    colstore::gather_multirecord_strided_typed(base, output, start, step, n_out, record_starts_rows, record_starts_bytes, n_rows_per_record, n_records, col_prefix_bytes, store, thread_cap, prefetch_distance);
  });
}

int colstore_gather_multirecord_uniform(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::ptrdiff_t n, std::int64_t rows_per_record, std::int64_t record_stride_bytes,
    std::int64_t first_body_offset, std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int itemsize, int thread_cap, std::ptrdiff_t prefetch_distance) {
  return colstore::dispatch_store(itemsize, [&](auto store) {
    colstore::gather_multirecord_uniform_typed(base, indices, output, n, rows_per_record, record_stride_bytes, first_body_offset, n_records, last_record_rows, col_prefix_bytes, store, thread_cap, prefetch_distance);
  });
}

int colstore_gather_multirecord_uniform_bins(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::int32_t* bins, std::ptrdiff_t n, std::int64_t rows_per_record,
    std::int64_t record_stride_bytes, std::int64_t first_body_offset,
    std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int itemsize, int thread_cap, std::ptrdiff_t prefetch_distance) {
  return colstore::dispatch_store(itemsize, [&](auto store) {
    colstore::gather_multirecord_uniform_bins_typed(base, indices, output, bins, n, rows_per_record, record_stride_bytes, first_body_offset, n_records, last_record_rows, col_prefix_bytes, store, thread_cap, prefetch_distance);
  });
}

int colstore_gather_multirecord_uniform_withbins(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, std::int64_t rows_per_record,
    std::int64_t record_stride_bytes, std::int64_t first_body_offset,
    std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int itemsize, int thread_cap, std::ptrdiff_t prefetch_distance) {
  return colstore::dispatch_store(itemsize, [&](auto store) {
    colstore::gather_multirecord_uniform_withbins_typed(base, indices, output, bins, n, rows_per_record, record_stride_bytes, first_body_offset, n_records, last_record_rows, col_prefix_bytes, store, thread_cap, prefetch_distance);
  });
}

int colstore_gather_segment(const std::int64_t* indices,
                              std::uint8_t* output, std::ptrdiff_t n,
                              const std::int64_t* segment_starts_rows,
                              const std::int64_t* segment_base,
                              std::int64_t n_segments, int itemsize, int thread_cap,
                              std::ptrdiff_t prefetch_distance) {
  return colstore::dispatch_store(itemsize, [&](auto store) {
    colstore::gather_segment_typed(indices, output, n, segment_starts_rows, segment_base,
                                     n_segments, store, thread_cap, prefetch_distance);
  });
}

int colstore_gather_segment_bins(const std::int64_t* indices, std::uint8_t* output,
                                   std::int32_t* bins, std::ptrdiff_t n,
                                   const std::int64_t* segment_starts_rows,
                                   const std::int64_t* segment_base, std::int64_t n_segments,
                                   int itemsize, int thread_cap, std::ptrdiff_t prefetch_distance) {
  return colstore::dispatch_store(itemsize, [&](auto store) {
    colstore::gather_segment_bins_typed(indices, output, bins, n, segment_starts_rows,
                                          segment_base, n_segments, store, thread_cap,
                                          prefetch_distance);
  });
}

int colstore_gather_segment_withbins(const std::int64_t* indices, std::uint8_t* output,
                                       const std::int32_t* bins, std::ptrdiff_t n,
                                       const std::int64_t* segment_base, int itemsize,
                                       int thread_cap, std::ptrdiff_t prefetch_distance) {
  return colstore::dispatch_store(itemsize, [&](auto store) {
    colstore::gather_segment_withbins_typed(indices, output, bins, n, segment_base, store,
                                              thread_cap, prefetch_distance);
  });
}

int colstore_gather_segment_sorted(const std::int64_t* indices, std::uint8_t* output,
                                     std::ptrdiff_t n, const std::int64_t* segment_starts_rows,
                                     const std::int64_t* segment_base, std::int64_t n_segments,
                                     int itemsize, int thread_cap,
                                     std::ptrdiff_t prefetch_distance) {
  return colstore::dispatch_store(itemsize, [&](auto store) {
    colstore::gather_segment_sorted_typed(indices, output, n, segment_starts_rows,
                                            segment_base, n_segments, store, thread_cap,
                                            prefetch_distance);
  });
}

int colstore_gather_segment_uniform(const std::int64_t* indices, std::uint8_t* output,
                                      std::ptrdiff_t n, std::int64_t rows_per_segment,
                                      const std::int64_t* segment_base, std::int64_t n_segments,
                                      int itemsize, int thread_cap,
                                      std::ptrdiff_t prefetch_distance) {
  return colstore::dispatch_store(itemsize, [&](auto store) {
    colstore::gather_segment_uniform_typed(indices, output, n, rows_per_segment, segment_base,
                                             n_segments, store, thread_cap, prefetch_distance);
  });
}

int colstore_gather_segment_uniform_bins(const std::int64_t* indices, std::uint8_t* output,
                                           std::int32_t* bins, std::ptrdiff_t n,
                                           std::int64_t rows_per_segment,
                                           const std::int64_t* segment_base,
                                           std::int64_t n_segments, int itemsize, int thread_cap,
                                           std::ptrdiff_t prefetch_distance) {
  return colstore::dispatch_store(itemsize, [&](auto store) {
    colstore::gather_segment_uniform_bins_typed(indices, output, bins, n, rows_per_segment,
                                                  segment_base, n_segments, store, thread_cap,
                                                  prefetch_distance);
  });
}

void colstore_parallel_copy_runs(std::uint8_t* output, const std::int64_t* src_addrs,
                                 const std::int64_t* dst_offsets,
                                 const std::int64_t* byte_lengths,
                                 std::int64_t n_runs, int thread_cap) {
  if (n_runs <= 0) {
    return;
  }
  // Prefix sum of the run lengths: prefix[r] is run r's first byte in the runs'
  // logical concatenation. Threads partition that logical span, so the work
  // balances over bytes regardless of run sizes. The output positions
  // (dst_offsets) are independent of the prefix, so the runs may leave gaps --
  // a region a non-viewable file fills separately.
  std::vector<std::int64_t> prefix(static_cast<std::size_t>(n_runs) + 1);
  prefix[0] = 0;
  for (std::int64_t r = 0; r < n_runs; ++r) {
    prefix[r + 1] = prefix[r] + byte_lengths[r];
  }
  const std::int64_t total = prefix[n_runs];
  if (total <= 0) {
    return;
  }
  const std::ptrdiff_t n_threads =
      colstore::resolve_thread_count(static_cast<std::ptrdiff_t>(total), thread_cap);

  // Copy logical bytes [lo, hi), mapping each to its run via the prefix and
  // writing at the run's output position. upper_bound finds the run holding
  // byte lo; ++run advances across run boundaries.
  const auto copy_logical_range = [&](std::int64_t lo, std::int64_t hi) {
    std::int64_t run = std::upper_bound(prefix.begin(), prefix.end(), lo) - prefix.begin() - 1;
    std::int64_t p = lo;
    while (p < hi) {
      const std::int64_t end = std::min(hi, prefix[run + 1]);
      const std::int64_t within = p - prefix[run];
      std::memcpy(output + dst_offsets[run] + within,
                  reinterpret_cast<const std::uint8_t*>(
                      static_cast<std::uintptr_t>(src_addrs[run] + within)),
                  static_cast<std::size_t>(end - p));
      p = end;
      ++run;
    }
  };

#ifdef _OPENMP
  if (n_threads > 1) {
#pragma omp parallel num_threads(static_cast<int>(n_threads))
    {
      const std::int64_t nt = omp_get_num_threads();
      const std::int64_t tid = omp_get_thread_num();
      const std::int64_t chunk = (total + nt - 1) / nt;
      const std::int64_t lo = tid * chunk;
      const std::int64_t hi = std::min(total, lo + chunk);
      if (lo < hi) {
        copy_logical_range(lo, hi);
      }
    }
    return;
  }
#else
  (void)n_threads;
#endif
  copy_logical_range(0, total);
}

namespace {

// Copy one field. The switch hands the compiler a compile-time size for the
// common widths, so each lowers to a single (unaligned) load/store instead of a
// runtime memcpy call; packed record fields may be misaligned, so it stays a
// memcpy rather than a typed dereference.
inline void copy_field(std::uint8_t* dst, const std::uint8_t* src, std::int64_t itemsize) {
  switch (itemsize) {
    case 8:
      std::memcpy(dst, src, 8);
      break;
    case 4:
      std::memcpy(dst, src, 4);
      break;
    case 2:
      std::memcpy(dst, src, 2);
      break;
    case 1:
      *dst = *src;
      break;
    default:
      std::memcpy(dst, src, static_cast<std::size_t>(itemsize));
      break;
  }
}

}  // namespace

void colstore_interleave_records(std::uint8_t* output, std::int64_t record_itemsize,
                                 std::int64_t n_rows, const std::int64_t* src_addrs,
                                 const std::int64_t* src_itemsizes,
                                 const std::int64_t* field_offsets,
                                 std::int64_t n_cols, int thread_cap) {
  if (n_rows <= 0 || n_cols <= 0) {
    return;
  }
  const std::ptrdiff_t total = static_cast<std::ptrdiff_t>(n_rows) * record_itemsize;
  const std::ptrdiff_t n_threads = colstore::resolve_thread_count(total, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::int64_t i = 0; i < n_rows; ++i) {
    std::uint8_t* COLSTORE_RESTRICT rec = output + i * record_itemsize;
    for (std::int64_t c = 0; c < n_cols; ++c) {
      const std::int64_t itemsize = src_itemsizes[c];
      copy_field(rec + field_offsets[c],
                 reinterpret_cast<const std::uint8_t*>(
                     static_cast<std::uintptr_t>(src_addrs[c] + i * itemsize)),
                 itemsize);
    }
  }
}

int colstore_gather_segment_mask(const std::uint8_t* mask, std::uint8_t* output,
                                   std::int64_t n_rows, std::ptrdiff_t n_out,
                                   const std::int64_t* segment_starts_rows,
                                   const std::int64_t* segment_base, std::int64_t n_segments,
                                   int itemsize, int thread_cap,
                                   std::ptrdiff_t prefetch_distance) {
  int status = -1;
  colstore::dispatch_store(itemsize, [&](auto store) {
    status = colstore::gather_segment_mask_typed(mask, output, n_rows, n_out,
                                                   segment_starts_rows, segment_base,
                                                   n_segments, store, thread_cap,
                                                   prefetch_distance);
  });
  return status;
}

int colstore_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

int colstore_bind_threads_to_cpus(const int* cpus, int n) {
#if defined(_OPENMP) && defined(__linux__)
  if (n <= 0) {
    return 0;
  }
  std::atomic<int> bound{0};
  // Force a team of exactly ``n`` workers so each index maps to one CPU. The
  // binding sticks because libgomp reuses its pool across parallel regions, so
  // subsequent gather kernels run on these now-pinned threads.
#pragma omp parallel num_threads(n)
  {
    const int t = omp_get_thread_num();
    if (t < n && cpus[t] >= 0) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(cpus[t], &set);
      if (sched_setaffinity(0, sizeof(set), &set) == 0) {
        bound.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }
  return bound.load(std::memory_order_relaxed);
#else
  (void)cpus;
  (void)n;
  return -1;
#endif
}

}  // extern "C"
