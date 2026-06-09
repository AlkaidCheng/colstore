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

#include "colstore/gather.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cstring>
#include <vector>

namespace colstore {

// Resolve OpenMP thread count for ``n_indices`` indices under a caller cap.
// The kernel is memory-bandwidth-bound, so two rules:
//   1. Below PARALLEL_THRESHOLD the fork/join cost dwarfs the work -> serial.
//   2. Above it, scale roughly one thread per ELEMENTS_PER_THREAD elements,
//      clamped to ``cap``. Bandwidth saturates at a small thread count well
//      below core count, so the cap (typically <= 8) is the real limit; the
//      work-proportional term just avoids the full cap for mid-sized gathers.
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
  // The ramp matches the measured scaling knee (~1 thread per 1<<20
  // elements; scaling stays near-linear to the cap on the
  // bandwidth-limited gather).
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

// --- Policy-based gather prototype (toggled by COLSTORE_USE_POLICY_GATHER) ---
//
// gather_core holds the parallel + prefetch + store skeleton shared by the
// single-record kernels; a Policy supplies the per-element byte offset.
// IndexedPolicy and BytesPolicy reproduce the two addressing schemes below.
// The Policy is a by-value functor whose offset() inlines into the loop, so
// each instantiation emits the same machine code as the hand-written kernel
// it replaces -- compile-time polymorphism, no virtual dispatch. This is an
// intermediate prototype: benchmark/check_policy_gather.py compares a build
// with this macro defined against one without it before the rest of the
// scatter family is folded onto gather_core.
template <typename T, typename Policy>
inline void gather_core(const std::uint8_t* COLSTORE_RESTRICT base,
                        std::uint8_t* COLSTORE_RESTRICT output,
                        std::ptrdiff_t n_indices, int thread_cap,
                        std::ptrdiff_t prefetch_distance, Policy policy) {
  T* dst = reinterpret_cast<T*>(output);
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::ptrdiff_t i = 0; i < n_indices; ++i) {
    if (prefetch_distance > 0 && i + prefetch_distance < n_indices) {
      COLSTORE_PREFETCH(base + policy.offset(i + prefetch_distance));
    }
    dst[i] = load_unaligned<T>(base + policy.offset(i));
  }
}

template <typename T>
struct IndexedPolicy {
  const std::int64_t* COLSTORE_RESTRICT indices;
  inline std::ptrdiff_t offset(std::ptrdiff_t i) const {
    return indices[i] * static_cast<std::ptrdiff_t>(sizeof(T));
  }
};

struct BytesPolicy {
  const std::int64_t* COLSTORE_RESTRICT byte_offsets;
  inline std::ptrdiff_t offset(std::ptrdiff_t i) const { return byte_offsets[i]; }
};

}  // namespace

template <typename T>
void gather_indexed_typed(const std::uint8_t* COLSTORE_RESTRICT base,
                          const std::int64_t* COLSTORE_RESTRICT indices,
                          std::uint8_t* COLSTORE_RESTRICT output,
                          std::ptrdiff_t n_indices,
                          int thread_cap,
                          std::ptrdiff_t prefetch_distance) {
  // Byte-based addressing: ``base`` may not be aligned for T (see
  // load_unaligned). The prefetch below only forms addresses, never loads.
#ifdef COLSTORE_USE_POLICY_GATHER
  gather_core<T>(base, output, n_indices, thread_cap, prefetch_distance,
                 IndexedPolicy<T>{indices});
#else
  T* dst = reinterpret_cast<T*>(output);
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::ptrdiff_t i = 0; i < n_indices; ++i) {
    if (prefetch_distance > 0 && i + prefetch_distance < n_indices) {
      COLSTORE_PREFETCH(base + indices[i + prefetch_distance] * static_cast<std::ptrdiff_t>(sizeof(T)));
    }
    dst[i] = load_unaligned<T>(base + indices[i] * static_cast<std::ptrdiff_t>(sizeof(T)));
  }
#endif
}

// Byte-offset gather: ``output[i]`` is the T at ``base + byte_offsets[i]``.
//
// For the multi-record reader: addresses are non-uniform and pre-computed at
// the Python level (record-header skips, per-record column offsets). Offsets
// need not be T-aligned; source loads go through load_unaligned.
template <typename T>
void gather_bytes_typed(const std::uint8_t* COLSTORE_RESTRICT base,
                        const std::int64_t* COLSTORE_RESTRICT byte_offsets,
                        std::uint8_t* COLSTORE_RESTRICT output,
                        std::ptrdiff_t n_indices,
                        int thread_cap,
                        std::ptrdiff_t prefetch_distance) {
#ifdef COLSTORE_USE_POLICY_GATHER
  gather_core<T>(base, output, n_indices, thread_cap, prefetch_distance,
                 BytesPolicy{byte_offsets});
#else
  T* dst = reinterpret_cast<T*>(output);
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::ptrdiff_t i = 0; i < n_indices; ++i) {
    if (prefetch_distance > 0 && i + prefetch_distance < n_indices) {
      COLSTORE_PREFETCH(base + byte_offsets[i + prefetch_distance]);
    }
    dst[i] = load_unaligned<T>(base + byte_offsets[i]);
  }
#endif
}

// Explicit instantiations -- four sizes for each entry point.
template void gather_indexed_typed<std::uint8_t>(const std::uint8_t*,
                                                 const std::int64_t*,
                                                 std::uint8_t*,
                                                 std::ptrdiff_t, int,
                                                 std::ptrdiff_t);
template void gather_indexed_typed<std::uint16_t>(const std::uint8_t*,
                                                  const std::int64_t*,
                                                  std::uint8_t*,
                                                  std::ptrdiff_t, int,
                                                  std::ptrdiff_t);
template void gather_indexed_typed<std::uint32_t>(const std::uint8_t*,
                                                  const std::int64_t*,
                                                  std::uint8_t*,
                                                  std::ptrdiff_t, int,
                                                  std::ptrdiff_t);
template void gather_indexed_typed<std::uint64_t>(const std::uint8_t*,
                                                  const std::int64_t*,
                                                  std::uint8_t*,
                                                  std::ptrdiff_t, int,
                                                  std::ptrdiff_t);
template void gather_bytes_typed<std::uint8_t>(const std::uint8_t*,
                                               const std::int64_t*,
                                               std::uint8_t*,
                                               std::ptrdiff_t, int,
                                               std::ptrdiff_t);
template void gather_bytes_typed<std::uint16_t>(const std::uint8_t*,
                                                const std::int64_t*,
                                                std::uint8_t*,
                                                std::ptrdiff_t, int,
                                                std::ptrdiff_t);
template void gather_bytes_typed<std::uint32_t>(const std::uint8_t*,
                                                const std::int64_t*,
                                                std::uint8_t*,
                                                std::ptrdiff_t, int,
                                                std::ptrdiff_t);
template void gather_bytes_typed<std::uint64_t>(const std::uint8_t*,
                                                const std::int64_t*,
                                                std::uint8_t*,
                                                std::ptrdiff_t, int,
                                                std::ptrdiff_t);

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

namespace {

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

}  // namespace

// Fused multi-record fancy gather. See header for the addressing contract.
//
// One pass over ``indices``: per element, bin to a record with the branchless
// search above, compute the byte address in registers, and load. The binning
// is fused into the load and the loop is OpenMP-parallel across indices.
//
// The software prefetch recomputes the record bin for the look-ahead index;
// the search is cheap relative to the DRAM latency it hides for the scattered
// data load.
template <typename T>
void gather_multirecord_typed(const std::uint8_t* COLSTORE_RESTRICT base,
                              const std::int64_t* COLSTORE_RESTRICT indices,
                              std::uint8_t* COLSTORE_RESTRICT output,
                              std::ptrdiff_t n_indices,
                              const std::int64_t* COLSTORE_RESTRICT record_starts_rows,
                              const std::int64_t* COLSTORE_RESTRICT record_starts_bytes,
                              const std::int64_t* COLSTORE_RESTRICT n_rows_per_record,
                              std::int64_t n_records,
                              std::int64_t col_prefix_bytes,
                              int thread_cap,
                              std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = static_cast<std::int64_t>(sizeof(T));
  const std::int64_t len = n_records + 1;  // entries in record_starts_rows
  T* dst = reinterpret_cast<T*>(output);
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::ptrdiff_t i = 0; i < n_indices; ++i) {
    if (prefetch_distance > 0 && i + prefetch_distance < n_indices) {
      const std::int64_t j = indices[i + prefetch_distance];
      const std::int64_t rj = bin_record(record_starts_rows, len, j);
      const std::int64_t off_j = record_starts_bytes[rj] +
                                 col_prefix_bytes * n_rows_per_record[rj] +
                                 (j - record_starts_rows[rj]) * itemsize;
      COLSTORE_PREFETCH(base + off_j);
    }
    const std::int64_t idx = indices[i];
    const std::int64_t r = bin_record(record_starts_rows, len, idx);
    const std::int64_t off = record_starts_bytes[r] +
                             col_prefix_bytes * n_rows_per_record[r] +
                             (idx - record_starts_rows[r]) * itemsize;
    dst[i] = load_unaligned<T>(base + off);
  }
}

template void gather_multirecord_typed<std::uint8_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::ptrdiff_t,
    const std::int64_t*, const std::int64_t*, const std::int64_t*,
    std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_typed<std::uint16_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::ptrdiff_t,
    const std::int64_t*, const std::int64_t*, const std::int64_t*,
    std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_typed<std::uint32_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::ptrdiff_t,
    const std::int64_t*, const std::int64_t*, const std::int64_t*,
    std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_typed<std::uint64_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::ptrdiff_t,
    const std::int64_t*, const std::int64_t*, const std::int64_t*,
    std::int64_t, std::int64_t, int, std::ptrdiff_t);

// Bin-recording variant: identical addressing to gather_multirecord_typed,
// plus ``bins[i] = r`` so subsequent columns of the same read reuse the
// binning (the dominant cost of this kernel) instead of recomputing it per
// column.
template <typename T>
void gather_multirecord_bins_typed(const std::uint8_t* COLSTORE_RESTRICT base,
                                   const std::int64_t* COLSTORE_RESTRICT indices,
                                   std::uint8_t* COLSTORE_RESTRICT output,
                                   std::int32_t* COLSTORE_RESTRICT bins,
                                   std::ptrdiff_t n_indices,
                                   const std::int64_t* COLSTORE_RESTRICT record_starts_rows,
                                   const std::int64_t* COLSTORE_RESTRICT record_starts_bytes,
                                   const std::int64_t* COLSTORE_RESTRICT n_rows_per_record,
                                   std::int64_t n_records,
                                   std::int64_t col_prefix_bytes,
                                   int thread_cap,
                                   std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = static_cast<std::int64_t>(sizeof(T));
  const std::int64_t len = n_records + 1;
  T* dst = reinterpret_cast<T*>(output);
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::ptrdiff_t i = 0; i < n_indices; ++i) {
    if (prefetch_distance > 0 && i + prefetch_distance < n_indices) {
      const std::int64_t j = indices[i + prefetch_distance];
      const std::int64_t rj = bin_record(record_starts_rows, len, j);
      const std::int64_t off_j = record_starts_bytes[rj] +
                                 col_prefix_bytes * n_rows_per_record[rj] +
                                 (j - record_starts_rows[rj]) * itemsize;
      COLSTORE_PREFETCH(base + off_j);
    }
    const std::int64_t idx = indices[i];
    const std::int64_t r = bin_record(record_starts_rows, len, idx);
    bins[i] = static_cast<std::int32_t>(r);
    const std::int64_t off = record_starts_bytes[r] +
                             col_prefix_bytes * n_rows_per_record[r] +
                             (idx - record_starts_rows[r]) * itemsize;
    dst[i] = load_unaligned<T>(base + off);
  }
}

template void gather_multirecord_bins_typed<std::uint8_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::int32_t*,
    std::ptrdiff_t, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_bins_typed<std::uint16_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::int32_t*,
    std::ptrdiff_t, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_bins_typed<std::uint32_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::int32_t*,
    std::ptrdiff_t, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_bins_typed<std::uint64_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::int32_t*,
    std::ptrdiff_t, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, std::int64_t, std::int64_t, int, std::ptrdiff_t);

// Bins-provided companion: the per-element record bin is a sequential int32
// read instead of a branchless search, and the prefetch look-ahead likewise
// reads ``bins[i + d]`` -- no second search anywhere.
template <typename T>
void gather_multirecord_withbins_typed(const std::uint8_t* COLSTORE_RESTRICT base,
                                       const std::int64_t* COLSTORE_RESTRICT indices,
                                       std::uint8_t* COLSTORE_RESTRICT output,
                                       const std::int32_t* COLSTORE_RESTRICT bins,
                                       std::ptrdiff_t n_indices,
                                       const std::int64_t* COLSTORE_RESTRICT record_starts_rows,
                                       const std::int64_t* COLSTORE_RESTRICT record_starts_bytes,
                                       const std::int64_t* COLSTORE_RESTRICT n_rows_per_record,
                                       std::int64_t col_prefix_bytes,
                                       int thread_cap,
                                       std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = static_cast<std::int64_t>(sizeof(T));
  T* dst = reinterpret_cast<T*>(output);
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::ptrdiff_t i = 0; i < n_indices; ++i) {
    if (prefetch_distance > 0 && i + prefetch_distance < n_indices) {
      const std::int64_t j = indices[i + prefetch_distance];
      const std::int64_t rj = bins[i + prefetch_distance];
      const std::int64_t off_j = record_starts_bytes[rj] +
                                 col_prefix_bytes * n_rows_per_record[rj] +
                                 (j - record_starts_rows[rj]) * itemsize;
      COLSTORE_PREFETCH(base + off_j);
    }
    const std::int64_t idx = indices[i];
    const std::int64_t r = bins[i];
    const std::int64_t off = record_starts_bytes[r] +
                             col_prefix_bytes * n_rows_per_record[r] +
                             (idx - record_starts_rows[r]) * itemsize;
    dst[i] = load_unaligned<T>(base + off);
  }
}

template void gather_multirecord_withbins_typed<std::uint8_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, const std::int32_t*,
    std::ptrdiff_t, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_withbins_typed<std::uint16_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, const std::int32_t*,
    std::ptrdiff_t, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_withbins_typed<std::uint32_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, const std::int32_t*,
    std::ptrdiff_t, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_withbins_typed<std::uint64_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, const std::int32_t*,
    std::ptrdiff_t, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, std::int64_t, int, std::ptrdiff_t);

// Sorted multi-record fancy gather. ``indices`` must be non-decreasing
// (caller-checked). Each thread locates the record of the first index in
// its chunk with one branchless binary search, then walks the record cursor
// forward monotonically -- the walk does O(K + R) total comparisons across
// the whole call, versus O(K log R) searches for the unsorted kernel.
// Offsets are computed in registers; there is no byte_offsets array.
//
// Prefetching: the look-ahead element's record is only known after walking,
// so the prefetch is issued only when the look-ahead index still lies in
// the *current* record (the common case for dense sorted reads, where rows
// per record >> prefetch distance). It is a hint; skipping cross-record
// look-aheads costs nothing but a few unprefetched boundary elements.
template <typename T>
void gather_multirecord_sorted_typed(const std::uint8_t* COLSTORE_RESTRICT base,
                                     const std::int64_t* COLSTORE_RESTRICT indices,
                                     std::uint8_t* COLSTORE_RESTRICT output,
                                     std::ptrdiff_t n_indices,
                                     const std::int64_t* COLSTORE_RESTRICT record_starts_rows,
                                     const std::int64_t* COLSTORE_RESTRICT record_starts_bytes,
                                     const std::int64_t* COLSTORE_RESTRICT n_rows_per_record,
                                     std::int64_t n_records,
                                     std::int64_t col_prefix_bytes,
                                     int thread_cap,
                                     std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = static_cast<std::int64_t>(sizeof(T));
  const std::int64_t len = n_records + 1;  // entries in record_starts_rows
  T* dst = reinterpret_cast<T*>(output);
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);

  const auto walk_range = [&](std::ptrdiff_t lo, std::ptrdiff_t hi) {
    if (lo >= hi) {
      return;
    }
    std::int64_t r = bin_record(record_starts_rows, len, indices[lo]);
    // Per-record state, recomputed only at record boundaries: the inner
    // loop's steady state is one compare, one multiply-add, one load.
    std::int64_t next_boundary = record_starts_rows[r + 1];
    std::int64_t record_base = record_starts_bytes[r] +
                               col_prefix_bytes * n_rows_per_record[r] -
                               record_starts_rows[r] * itemsize;
    for (std::ptrdiff_t i = lo; i < hi; ++i) {
      const std::int64_t idx = indices[i];
      if (idx >= next_boundary) {
        do {
          ++r;
        } while (idx >= record_starts_rows[r + 1]);
        next_boundary = record_starts_rows[r + 1];
        record_base = record_starts_bytes[r] +
                      col_prefix_bytes * n_rows_per_record[r] -
                      record_starts_rows[r] * itemsize;
      }
      if (prefetch_distance > 0 && i + prefetch_distance < hi) {
        const std::int64_t j = indices[i + prefetch_distance];
        if (j < next_boundary) {
          COLSTORE_PREFETCH(base + record_base + j * itemsize);
        }
      }
      dst[i] = load_unaligned<T>(base + record_base + idx * itemsize);
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
#endif
  walk_range(0, n_indices);
}

template void gather_multirecord_sorted_typed<std::uint8_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::ptrdiff_t,
    const std::int64_t*, const std::int64_t*, const std::int64_t*,
    std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_sorted_typed<std::uint16_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::ptrdiff_t,
    const std::int64_t*, const std::int64_t*, const std::int64_t*,
    std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_sorted_typed<std::uint32_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::ptrdiff_t,
    const std::int64_t*, const std::int64_t*, const std::int64_t*,
    std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_sorted_typed<std::uint64_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::ptrdiff_t,
    const std::int64_t*, const std::int64_t*, const std::int64_t*,
    std::int64_t, std::int64_t, int, std::ptrdiff_t);

// Strided range gather: a linear record walk over the arithmetic row stream
// ``start + i*step``. Structurally the sorted kernel with the index-array
// loads deleted -- the row is synthesized in a register, so the per-element
// steady state is one or two compares, one multiply-add, and the data load.
// The record cursor moves monotonically -- forward for ``step > 0``,
// backward for ``step < 0`` -- so each thread's total cursor movement is
// bounded by R regardless of step size. The same-record prefetch gate
// mirrors the sorted kernel's, with the look-ahead row formed
// arithmetically instead of loaded.
template <typename T>
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
                                      int thread_cap,
                                      std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = static_cast<std::int64_t>(sizeof(T));
  const std::int64_t len = n_records + 1;  // entries in record_starts_rows
  T* dst = reinterpret_cast<T*>(output);
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
      dst[i] = load_unaligned<T>(base + record_base + idx * itemsize);
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

template void gather_multirecord_strided_typed<std::uint8_t>(
    const std::uint8_t*, std::uint8_t*, std::int64_t, std::int64_t,
    std::ptrdiff_t, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_strided_typed<std::uint16_t>(
    const std::uint8_t*, std::uint8_t*, std::int64_t, std::int64_t,
    std::ptrdiff_t, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_strided_typed<std::uint32_t>(
    const std::uint8_t*, std::uint8_t*, std::int64_t, std::int64_t,
    std::ptrdiff_t, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_strided_typed<std::uint64_t>(
    const std::uint8_t*, std::uint8_t*, std::int64_t, std::int64_t,
    std::ptrdiff_t, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, std::int64_t, std::int64_t, int, std::ptrdiff_t);

// Uniform-record fancy gather: the fused unsorted gather with the binary
// search replaced by one integer division and the three per-element
// metadata loads replaced by an affine formula. For a full record r,
//   offset(idx) = first_body_offset + r*stride + col_prefix*U
//                 + (idx - r*U)*itemsize
//               = full_base + r*per_record_step + idx*itemsize
// with full_base and per_record_step loop constants; the (possibly
// partial) final record collapses to a single precomputed base behind one
// guard, which is the only data-dependent branch and is taken only for
// indices in the last record. The prefetch look-ahead reuses the same
// arithmetic, so -- unlike the irregular kernels -- it needs no
// same-record gating: every valid index yields a valid address.
template <typename T>
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
                                      int thread_cap,
                                      std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = static_cast<std::int64_t>(sizeof(T));
  T* dst = reinterpret_cast<T*>(output);
  const std::int64_t full_base = first_body_offset + col_prefix_bytes * rows_per_record;
  const std::int64_t per_record_step = record_stride_bytes - rows_per_record * itemsize;
  const std::int64_t last_first_row = (n_records - 1) * rows_per_record;
  const std::int64_t last_base = first_body_offset +
                                 (n_records - 1) * record_stride_bytes +
                                 col_prefix_bytes * last_record_rows -
                                 last_first_row * itemsize;
  const auto offset_of = [&](std::int64_t idx) -> std::int64_t {
    if (idx >= last_first_row) {
      return last_base + idx * itemsize;
    }
    const std::int64_t r = idx / rows_per_record;
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
    dst[i] = load_unaligned<T>(base + offset_of(indices[i]));
  }
}

template void gather_multirecord_uniform_typed<std::uint8_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::ptrdiff_t,
    std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_uniform_typed<std::uint16_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::ptrdiff_t,
    std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_uniform_typed<std::uint32_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::ptrdiff_t,
    std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_uniform_typed<std::uint64_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::ptrdiff_t,
    std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, int, std::ptrdiff_t);

// Multi-column uniform pair. The bins variant pays the division once per
// element across the whole read; the withbins variant's per-element work
// is a sequential int32 read, one compare, one multiply-add, and the load
// -- no division, no search, no per-record metadata. This dominates both
// the generic bins route (search -> division for the first column;
// three metadata loads -> affine math for the rest) and per-column
// arithmetic binning (division x C -> division x 1).
template <typename T>
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
                                           int thread_cap,
                                           std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = static_cast<std::int64_t>(sizeof(T));
  T* dst = reinterpret_cast<T*>(output);
  const std::int64_t full_base = first_body_offset + col_prefix_bytes * rows_per_record;
  const std::int64_t per_record_step = record_stride_bytes - rows_per_record * itemsize;
  const std::int64_t last_first_row = (n_records - 1) * rows_per_record;
  const std::int64_t last_base = first_body_offset +
                                 (n_records - 1) * record_stride_bytes +
                                 col_prefix_bytes * last_record_rows -
                                 last_first_row * itemsize;
  const auto offset_of = [&](std::int64_t idx) -> std::int64_t {
    if (idx >= last_first_row) {
      return last_base + idx * itemsize;
    }
    return full_base + (idx / rows_per_record) * per_record_step + idx * itemsize;
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
      r = idx / rows_per_record;
      off = full_base + r * per_record_step + idx * itemsize;
    }
    bins[i] = static_cast<std::int32_t>(r);
    dst[i] = load_unaligned<T>(base + off);
  }
}

template <typename T>
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
                                               int thread_cap,
                                               std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = static_cast<std::int64_t>(sizeof(T));
  T* dst = reinterpret_cast<T*>(output);
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
    dst[i] = load_unaligned<T>(base + offset_of(indices[i], bins[i]));
  }
}

template void gather_multirecord_uniform_bins_typed<std::uint8_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::int32_t*,
    std::ptrdiff_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_uniform_bins_typed<std::uint16_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::int32_t*,
    std::ptrdiff_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_uniform_bins_typed<std::uint32_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::int32_t*,
    std::ptrdiff_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_uniform_bins_typed<std::uint64_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, std::int32_t*,
    std::ptrdiff_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_uniform_withbins_typed<std::uint8_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, const std::int32_t*,
    std::ptrdiff_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_uniform_withbins_typed<std::uint16_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, const std::int32_t*,
    std::ptrdiff_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_uniform_withbins_typed<std::uint32_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, const std::int32_t*,
    std::ptrdiff_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, std::int64_t, int, std::ptrdiff_t);
template void gather_multirecord_uniform_withbins_typed<std::uint64_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, const std::int32_t*,
    std::ptrdiff_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, std::int64_t, int, std::ptrdiff_t);

// Record-base withbins variant: the per-element steady state is the
// sequential int32 bin read, one record_base load, one multiply-add, and
// the data load -- against the generic withbins kernel's three metadata
// loads and two multiplies. The record_base array is built by the caller
// per column (O(R), vectorized) and folds the column prefix and the
// row-to-byte conversion into a single per-record scalar.
template <typename T>
void gather_multirecord_withbins_rbase_typed(const std::uint8_t* COLSTORE_RESTRICT base,
                                             const std::int64_t* COLSTORE_RESTRICT indices,
                                             std::uint8_t* COLSTORE_RESTRICT output,
                                             const std::int32_t* COLSTORE_RESTRICT bins,
                                             std::ptrdiff_t n_indices,
                                             const std::int64_t* COLSTORE_RESTRICT record_base,
                                             int thread_cap,
                                             std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = static_cast<std::int64_t>(sizeof(T));
  T* dst = reinterpret_cast<T*>(output);
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
      COLSTORE_PREFETCH(base + record_base[bins[j]] + indices[j] * itemsize);
    }
    dst[i] = load_unaligned<T>(base + record_base[bins[i]] + indices[i] * itemsize);
  }
}

template void gather_multirecord_withbins_rbase_typed<std::uint8_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, const std::int32_t*,
    std::ptrdiff_t, const std::int64_t*, int, std::ptrdiff_t);
template void gather_multirecord_withbins_rbase_typed<std::uint16_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, const std::int32_t*,
    std::ptrdiff_t, const std::int64_t*, int, std::ptrdiff_t);
template void gather_multirecord_withbins_rbase_typed<std::uint32_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, const std::int32_t*,
    std::ptrdiff_t, const std::int64_t*, int, std::ptrdiff_t);
template void gather_multirecord_withbins_rbase_typed<std::uint64_t>(
    const std::uint8_t*, const std::int64_t*, std::uint8_t*, const std::int32_t*,
    std::ptrdiff_t, const std::int64_t*, int, std::ptrdiff_t);

// Boolean-mask-native gather. Pass 1 counts each thread chunk's selected
// rows (a vectorizable byte sum) so output offsets are exact without
// synchronization; pass 2 walks each chunk's rows with the sorted walk's
// record-cursor machinery, skipping unselected spans 8 mask bytes at a
// time (a uint64 of zeros), extending runs of set bits 8 bytes at a time
// (a uint64 of 0x01s), and serving runs of >= 32 bytes with one memcpy --
// run detection costs nothing extra here because the mask byte is the
// datum being scanned anyway.
namespace {
constexpr std::int64_t MASK_RUN_COPY_MIN_BYTES = 32;
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

template <typename T>
int gather_multirecord_mask_typed(const std::uint8_t* COLSTORE_RESTRICT base,
                                  const std::uint8_t* COLSTORE_RESTRICT mask,
                                  std::uint8_t* COLSTORE_RESTRICT output,
                                  std::int64_t n_rows,
                                  std::ptrdiff_t n_out,
                                  const std::int64_t* COLSTORE_RESTRICT record_starts_rows,
                                  const std::int64_t* COLSTORE_RESTRICT record_starts_bytes,
                                  const std::int64_t* COLSTORE_RESTRICT n_rows_per_record,
                                  std::int64_t n_records,
                                  std::int64_t col_prefix_bytes,
                                  int thread_cap,
                                  std::ptrdiff_t prefetch_distance) {
  const std::int64_t itemsize = static_cast<std::int64_t>(sizeof(T));
  const std::int64_t len = n_records + 1;
  T* dst = reinterpret_cast<T*>(output);
  const std::ptrdiff_t n_threads = resolve_thread_count(n_rows, thread_cap);
  const std::int64_t chunk = (n_rows + n_threads - 1) / n_threads;

  // Pass 1: per-chunk selected-row counts -> exclusive prefix offsets.
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

  // Pass 2, word-at-a-time. The per-element mask test is a coin flip at
  // mid densities and mispredicts catastrophically; classifying 8 mask
  // bytes at once makes the branches predictable in every density regime:
  // all-zero words skip, runs of all-ones words become one memcpy, and
  // mixed words compact branchlessly -- eight unconditional loads/stores
  // with ``out += mask[i]``, which over-stores into slots that later
  // selected elements overwrite. The over-store is bounded inside the
  // thread's own output region by the quota guard: the branchless form
  // runs only while out + 8 <= quota, and the last few words of a
  // thread's quota take the branchy scalar form (a handful of
  // mispredicts per THREAD, not per element).
  const auto walk_range = [&](std::int64_t lo, std::int64_t hi, std::int64_t out,
                              std::int64_t quota) {
    if (lo >= hi) {
      return;
    }
    std::int64_t r = bin_record(record_starts_rows, len, lo);
    std::int64_t i = lo;
    while (i < hi) {
      while (i >= record_starts_rows[r + 1]) {
        ++r;
      }
      const std::int64_t seg_end = std::min(hi, record_starts_rows[r + 1]);
      const std::int64_t record_base = record_starts_bytes[r] +
                                       col_prefix_bytes * n_rows_per_record[r] -
                                       record_starts_rows[r] * itemsize;
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
            std::memcpy(dst + out, base + record_base + i * itemsize,
                        static_cast<std::size_t>((j - i) * itemsize));
            out += j - i;
            i = j;
            continue;
          }
          if (out + 8 <= quota) {
            // Mixed word, branchless compaction: unconditional load/store,
            // advance by the mask byte. Reads up to 8 in-record elements
            // regardless of selection; transiently stores garbage into
            // slots later overwritten by the true occupants.
            for (int b = 0; b < 8; ++b) {
              dst[out] = load_unaligned<T>(base + record_base + (i + b) * itemsize);
              out += mask[i + b];
            }
            i += 8;
            continue;
          }
        }
        // Branchy scalar form: segment tails shorter than a word, and the
        // final words of this thread's quota (where branchless over-store
        // could cross into the next thread's region).
        if (mask[i]) {
          dst[out] = load_unaligned<T>(base + record_base + i * itemsize);
          ++out;
        }
        ++i;
      }
    }
    (void)prefetch_distance;  // linear walk: hardware prefetch covers it
  };

#ifdef _OPENMP
  if (n_threads > 1) {
#pragma omp parallel num_threads(static_cast<int>(n_threads))
    {
      const std::ptrdiff_t t = omp_get_thread_num();
      const std::int64_t lo = static_cast<std::int64_t>(t) * chunk;
      const std::int64_t hi = std::min(n_rows, lo + chunk);
      walk_range(lo, hi, offsets[static_cast<std::size_t>(t)],
                 offsets[static_cast<std::size_t>(t) + 1]);
    }
    return 0;
  }
#endif
  walk_range(0, n_rows, 0, static_cast<std::int64_t>(n_out));
  return 0;
}

template int gather_multirecord_mask_typed<std::uint8_t>(
    const std::uint8_t*, const std::uint8_t*, std::uint8_t*, std::int64_t,
    std::ptrdiff_t, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, std::int64_t, std::int64_t, int, std::ptrdiff_t);
template int gather_multirecord_mask_typed<std::uint16_t>(
    const std::uint8_t*, const std::uint8_t*, std::uint8_t*, std::int64_t,
    std::ptrdiff_t, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, std::int64_t, std::int64_t, int, std::ptrdiff_t);
template int gather_multirecord_mask_typed<std::uint32_t>(
    const std::uint8_t*, const std::uint8_t*, std::uint8_t*, std::int64_t,
    std::ptrdiff_t, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, std::int64_t, std::int64_t, int, std::ptrdiff_t);
template int gather_multirecord_mask_typed<std::uint64_t>(
    const std::uint8_t*, const std::uint8_t*, std::uint8_t*, std::int64_t,
    std::ptrdiff_t, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, std::int64_t, std::int64_t, int, std::ptrdiff_t);

}  // namespace colstore

extern "C" {

// Space-separated names of the optimization toggles this build compiled
// with (see CMakeLists COLSTORE_TOGGLES). Empty when none are enabled.
// Lets benchmarks/tests report exactly what they are measuring without a
// per-flag accessor.
const char* colstore_build_flags(void) {
  return
#ifdef COLSTORE_USE_POLICY_GATHER
      "COLSTORE_USE_POLICY_GATHER "
#endif
      "";
}

void colstore_gather_indexed_1(const std::uint8_t* base,
                               const std::int64_t* indices,
                               std::uint8_t* output,
                               std::ptrdiff_t n, int thread_cap,
                               std::ptrdiff_t prefetch_distance) {
  colstore::gather_indexed_typed<std::uint8_t>(base, indices, output, n,
                                               thread_cap, prefetch_distance);
}
void colstore_gather_indexed_2(const std::uint8_t* base,
                               const std::int64_t* indices,
                               std::uint8_t* output,
                               std::ptrdiff_t n, int thread_cap,
                               std::ptrdiff_t prefetch_distance) {
  colstore::gather_indexed_typed<std::uint16_t>(base, indices, output, n,
                                                thread_cap, prefetch_distance);
}
void colstore_gather_indexed_4(const std::uint8_t* base,
                               const std::int64_t* indices,
                               std::uint8_t* output,
                               std::ptrdiff_t n, int thread_cap,
                               std::ptrdiff_t prefetch_distance) {
  colstore::gather_indexed_typed<std::uint32_t>(base, indices, output, n,
                                                thread_cap, prefetch_distance);
}
void colstore_gather_indexed_8(const std::uint8_t* base,
                               const std::int64_t* indices,
                               std::uint8_t* output,
                               std::ptrdiff_t n, int thread_cap,
                               std::ptrdiff_t prefetch_distance) {
  colstore::gather_indexed_typed<std::uint64_t>(base, indices, output, n,
                                                thread_cap, prefetch_distance);
}

void colstore_gather_bytes_1(const std::uint8_t* base,
                             const std::int64_t* byte_offsets,
                             std::uint8_t* output,
                             std::ptrdiff_t n, int thread_cap,
                               std::ptrdiff_t prefetch_distance) {
  colstore::gather_bytes_typed<std::uint8_t>(base, byte_offsets, output, n,
                                             thread_cap, prefetch_distance);
}
void colstore_gather_bytes_2(const std::uint8_t* base,
                             const std::int64_t* byte_offsets,
                             std::uint8_t* output,
                             std::ptrdiff_t n, int thread_cap,
                               std::ptrdiff_t prefetch_distance) {
  colstore::gather_bytes_typed<std::uint16_t>(base, byte_offsets, output, n,
                                              thread_cap, prefetch_distance);
}
void colstore_gather_bytes_4(const std::uint8_t* base,
                             const std::int64_t* byte_offsets,
                             std::uint8_t* output,
                             std::ptrdiff_t n, int thread_cap,
                               std::ptrdiff_t prefetch_distance) {
  colstore::gather_bytes_typed<std::uint32_t>(base, byte_offsets, output, n,
                                              thread_cap, prefetch_distance);
}
void colstore_gather_bytes_8(const std::uint8_t* base,
                             const std::int64_t* byte_offsets,
                             std::uint8_t* output,
                             std::ptrdiff_t n, int thread_cap,
                               std::ptrdiff_t prefetch_distance) {
  colstore::gather_bytes_typed<std::uint64_t>(base, byte_offsets, output, n,
                                              thread_cap, prefetch_distance);
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

void colstore_gather_multirecord_1(const std::uint8_t* base,
                                   const std::int64_t* indices,
                                   std::uint8_t* output, std::ptrdiff_t n,
                                   const std::int64_t* record_starts_rows,
                                   const std::int64_t* record_starts_bytes,
                                   const std::int64_t* n_rows_per_record,
                                   std::int64_t n_records,
                                   std::int64_t col_prefix_bytes, int thread_cap,
                               std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_typed<std::uint8_t>(
      base, indices, output, n, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}
void colstore_gather_multirecord_2(const std::uint8_t* base,
                                   const std::int64_t* indices,
                                   std::uint8_t* output, std::ptrdiff_t n,
                                   const std::int64_t* record_starts_rows,
                                   const std::int64_t* record_starts_bytes,
                                   const std::int64_t* n_rows_per_record,
                                   std::int64_t n_records,
                                   std::int64_t col_prefix_bytes, int thread_cap,
                               std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_typed<std::uint16_t>(
      base, indices, output, n, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}
void colstore_gather_multirecord_4(const std::uint8_t* base,
                                   const std::int64_t* indices,
                                   std::uint8_t* output, std::ptrdiff_t n,
                                   const std::int64_t* record_starts_rows,
                                   const std::int64_t* record_starts_bytes,
                                   const std::int64_t* n_rows_per_record,
                                   std::int64_t n_records,
                                   std::int64_t col_prefix_bytes, int thread_cap,
                               std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_typed<std::uint32_t>(
      base, indices, output, n, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}
void colstore_gather_multirecord_8(const std::uint8_t* base,
                                   const std::int64_t* indices,
                                   std::uint8_t* output, std::ptrdiff_t n,
                                   const std::int64_t* record_starts_rows,
                                   const std::int64_t* record_starts_bytes,
                                   const std::int64_t* n_rows_per_record,
                                   std::int64_t n_records,
                                   std::int64_t col_prefix_bytes, int thread_cap,
                               std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_typed<std::uint64_t>(
      base, indices, output, n, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_bins_1(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_bins_typed<std::uint8_t>(
      base, indices, output, bins, n, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_withbins_1(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_withbins_typed<std::uint8_t>(
      base, indices, output, bins, n, record_starts_rows, record_starts_bytes,
      n_rows_per_record, col_prefix_bytes, thread_cap, prefetch_distance);
}
void colstore_gather_multirecord_bins_2(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_bins_typed<std::uint16_t>(
      base, indices, output, bins, n, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_withbins_2(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_withbins_typed<std::uint16_t>(
      base, indices, output, bins, n, record_starts_rows, record_starts_bytes,
      n_rows_per_record, col_prefix_bytes, thread_cap, prefetch_distance);
}
void colstore_gather_multirecord_bins_4(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_bins_typed<std::uint32_t>(
      base, indices, output, bins, n, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_withbins_4(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_withbins_typed<std::uint32_t>(
      base, indices, output, bins, n, record_starts_rows, record_starts_bytes,
      n_rows_per_record, col_prefix_bytes, thread_cap, prefetch_distance);
}
void colstore_gather_multirecord_bins_8(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_bins_typed<std::uint64_t>(
      base, indices, output, bins, n, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_withbins_8(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_withbins_typed<std::uint64_t>(
      base, indices, output, bins, n, record_starts_rows, record_starts_bytes,
      n_rows_per_record, col_prefix_bytes, thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_sorted_1(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_sorted_typed<std::uint8_t>(
      base, indices, output, n, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_sorted_2(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_sorted_typed<std::uint16_t>(
      base, indices, output, n, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_sorted_4(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_sorted_typed<std::uint32_t>(
      base, indices, output, n, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_sorted_8(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_sorted_typed<std::uint64_t>(
      base, indices, output, n, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_strided_1(
    const std::uint8_t* base, std::uint8_t* output,
    std::int64_t start, std::int64_t step, std::ptrdiff_t n_out,
    const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_strided_typed<std::uint8_t>(
      base, output, start, step, n_out, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_strided_2(
    const std::uint8_t* base, std::uint8_t* output,
    std::int64_t start, std::int64_t step, std::ptrdiff_t n_out,
    const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_strided_typed<std::uint16_t>(
      base, output, start, step, n_out, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_strided_4(
    const std::uint8_t* base, std::uint8_t* output,
    std::int64_t start, std::int64_t step, std::ptrdiff_t n_out,
    const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_strided_typed<std::uint32_t>(
      base, output, start, step, n_out, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_strided_8(
    const std::uint8_t* base, std::uint8_t* output,
    std::int64_t start, std::int64_t step, std::ptrdiff_t n_out,
    const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_strided_typed<std::uint64_t>(
      base, output, start, step, n_out, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}


void colstore_gather_multirecord_uniform_1(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::ptrdiff_t n, std::int64_t rows_per_record, std::int64_t record_stride_bytes,
    std::int64_t first_body_offset, std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_uniform_typed<std::uint8_t>(
      base, indices, output, n, rows_per_record, record_stride_bytes,
      first_body_offset, n_records, last_record_rows, col_prefix_bytes,
      thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_uniform_2(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::ptrdiff_t n, std::int64_t rows_per_record, std::int64_t record_stride_bytes,
    std::int64_t first_body_offset, std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_uniform_typed<std::uint16_t>(
      base, indices, output, n, rows_per_record, record_stride_bytes,
      first_body_offset, n_records, last_record_rows, col_prefix_bytes,
      thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_uniform_4(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::ptrdiff_t n, std::int64_t rows_per_record, std::int64_t record_stride_bytes,
    std::int64_t first_body_offset, std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_uniform_typed<std::uint32_t>(
      base, indices, output, n, rows_per_record, record_stride_bytes,
      first_body_offset, n_records, last_record_rows, col_prefix_bytes,
      thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_uniform_8(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::ptrdiff_t n, std::int64_t rows_per_record, std::int64_t record_stride_bytes,
    std::int64_t first_body_offset, std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_uniform_typed<std::uint64_t>(
      base, indices, output, n, rows_per_record, record_stride_bytes,
      first_body_offset, n_records, last_record_rows, col_prefix_bytes,
      thread_cap, prefetch_distance);
}


void colstore_gather_multirecord_uniform_bins_1(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::int32_t* bins, std::ptrdiff_t n, std::int64_t rows_per_record,
    std::int64_t record_stride_bytes, std::int64_t first_body_offset,
    std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_uniform_bins_typed<std::uint8_t>(
      base, indices, output, bins, n, rows_per_record, record_stride_bytes,
      first_body_offset, n_records, last_record_rows, col_prefix_bytes,
      thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_uniform_bins_2(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::int32_t* bins, std::ptrdiff_t n, std::int64_t rows_per_record,
    std::int64_t record_stride_bytes, std::int64_t first_body_offset,
    std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_uniform_bins_typed<std::uint16_t>(
      base, indices, output, bins, n, rows_per_record, record_stride_bytes,
      first_body_offset, n_records, last_record_rows, col_prefix_bytes,
      thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_uniform_bins_4(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::int32_t* bins, std::ptrdiff_t n, std::int64_t rows_per_record,
    std::int64_t record_stride_bytes, std::int64_t first_body_offset,
    std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_uniform_bins_typed<std::uint32_t>(
      base, indices, output, bins, n, rows_per_record, record_stride_bytes,
      first_body_offset, n_records, last_record_rows, col_prefix_bytes,
      thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_uniform_bins_8(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::int32_t* bins, std::ptrdiff_t n, std::int64_t rows_per_record,
    std::int64_t record_stride_bytes, std::int64_t first_body_offset,
    std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_uniform_bins_typed<std::uint64_t>(
      base, indices, output, bins, n, rows_per_record, record_stride_bytes,
      first_body_offset, n_records, last_record_rows, col_prefix_bytes,
      thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_uniform_withbins_1(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, std::int64_t rows_per_record,
    std::int64_t record_stride_bytes, std::int64_t first_body_offset,
    std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_uniform_withbins_typed<std::uint8_t>(
      base, indices, output, bins, n, rows_per_record, record_stride_bytes,
      first_body_offset, n_records, last_record_rows, col_prefix_bytes,
      thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_uniform_withbins_2(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, std::int64_t rows_per_record,
    std::int64_t record_stride_bytes, std::int64_t first_body_offset,
    std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_uniform_withbins_typed<std::uint16_t>(
      base, indices, output, bins, n, rows_per_record, record_stride_bytes,
      first_body_offset, n_records, last_record_rows, col_prefix_bytes,
      thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_uniform_withbins_4(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, std::int64_t rows_per_record,
    std::int64_t record_stride_bytes, std::int64_t first_body_offset,
    std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_uniform_withbins_typed<std::uint32_t>(
      base, indices, output, bins, n, rows_per_record, record_stride_bytes,
      first_body_offset, n_records, last_record_rows, col_prefix_bytes,
      thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_uniform_withbins_8(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, std::int64_t rows_per_record,
    std::int64_t record_stride_bytes, std::int64_t first_body_offset,
    std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_uniform_withbins_typed<std::uint64_t>(
      base, indices, output, bins, n, rows_per_record, record_stride_bytes,
      first_body_offset, n_records, last_record_rows, col_prefix_bytes,
      thread_cap, prefetch_distance);
}


void colstore_gather_multirecord_withbins_rbase_1(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_base,
    int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_withbins_rbase_typed<std::uint8_t>(
      base, indices, output, bins, n, record_base, thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_withbins_rbase_2(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_base,
    int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_withbins_rbase_typed<std::uint16_t>(
      base, indices, output, bins, n, record_base, thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_withbins_rbase_4(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_base,
    int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_withbins_rbase_typed<std::uint32_t>(
      base, indices, output, bins, n, record_base, thread_cap, prefetch_distance);
}

void colstore_gather_multirecord_withbins_rbase_8(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_base,
    int thread_cap, std::ptrdiff_t prefetch_distance) {
  colstore::gather_multirecord_withbins_rbase_typed<std::uint64_t>(
      base, indices, output, bins, n, record_base, thread_cap, prefetch_distance);
}


int colstore_gather_multirecord_mask_1(
    const std::uint8_t* base, const std::uint8_t* mask, std::uint8_t* output,
    std::int64_t n_rows, std::ptrdiff_t n_out, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance) {
  return colstore::gather_multirecord_mask_typed<std::uint8_t>(
      base, mask, output, n_rows, n_out, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}

int colstore_gather_multirecord_mask_2(
    const std::uint8_t* base, const std::uint8_t* mask, std::uint8_t* output,
    std::int64_t n_rows, std::ptrdiff_t n_out, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance) {
  return colstore::gather_multirecord_mask_typed<std::uint16_t>(
      base, mask, output, n_rows, n_out, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}

int colstore_gather_multirecord_mask_4(
    const std::uint8_t* base, const std::uint8_t* mask, std::uint8_t* output,
    std::int64_t n_rows, std::ptrdiff_t n_out, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance) {
  return colstore::gather_multirecord_mask_typed<std::uint32_t>(
      base, mask, output, n_rows, n_out, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}

int colstore_gather_multirecord_mask_8(
    const std::uint8_t* base, const std::uint8_t* mask, std::uint8_t* output,
    std::int64_t n_rows, std::ptrdiff_t n_out, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance) {
  return colstore::gather_multirecord_mask_typed<std::uint64_t>(
      base, mask, output, n_rows, n_out, record_starts_rows, record_starts_bytes,
      n_rows_per_record, n_records, col_prefix_bytes, thread_cap, prefetch_distance);
}

int colstore_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

}  // extern "C"
