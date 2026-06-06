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
//  * Typed dereferences inside the templated loop. ``*(T*)(base + ...)`` is
//    treated by the compiler as a natural-alignment load and emits the same
//    instructions as ``source[i]`` would in the old per-dtype kernel.

#include "colstore/gather.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cstring>

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
  // Work-proportional thread count, ~one thread per ELEMENTS_PER_THREAD
  // elements. The previous form ``n / ELEMENTS_PER_THREAD + 1`` floored every
  // n below ELEMENTS_PER_THREAD (1<<20) to a single thread, silently
  // overriding PARALLEL_THRESHOLD (1<<18): a gather of 256K-1M elements is
  // past the fork/join floor and measured a clean 2x at 2 threads, but the
  // floor division forced it serial. Round up instead, and grant at least 2
  // threads once we are past PARALLEL_THRESHOLD (already guaranteed by the
  // early return above). Anything below PARALLEL_THRESHOLD never reaches here,
  // so small-input behavior is unchanged. Above it, the ramp matches the
  // measured knee (~1 thread per 1<<20 elements; scaling stays near-linear to
  // the cap on the bandwidth-limited gather).
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
// The caller passes byte pointers but the kernel reinterprets them as T*
// for the load/store -- this gives the compiler typed-alignment information
// and produces the same vectorized loop as a direct ``T* output[i] =
// T* source[i]`` body. T is one of the unsigned integer types (uint8_t/16_t/
// 32_t/64_t); the bytes copied are agnostic to the user-facing dtype kind.
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
}

// Byte-offset gather: ``output[i] = *(T*)(base + byte_offsets[i])``.
//
// For the multi-record reader: addresses are non-uniform and pre-computed at
// the Python level (record-header skips, per-record column offsets). The
// kernel reinterprets the loaded bytes as T to give the compiler the same
// typed-alignment information; caller guarantees offsets are T-aligned.
template <typename T>
void gather_bytes_typed(const std::uint8_t* COLSTORE_RESTRICT base,
                        const std::int64_t* COLSTORE_RESTRICT byte_offsets,
                        std::uint8_t* COLSTORE_RESTRICT output,
                        std::ptrdiff_t n_indices,
                        int thread_cap,
                        std::ptrdiff_t prefetch_distance) {
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
// core saturates memory bandwidth on a large contiguous copy. The win over the
// former Python loop is the elimination of per-record interpreter overhead
// (one np.frombuffer construction and one slice assignment per record), which
// dominates when the range spans many small records.
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
// mispredicts ~50% of comparisons and is several times slower in practice
// (measured ~5x at R=1000). ``rsr`` is tiny (8*(R+1) bytes) and stays cache-
// resident, so the search is a handful of L1 loads plus cmovs.
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
// search above, compute the byte address in registers, and load. This replaces
// the NumPy pipeline whose dominant cost is the searchsorted record-binning
// (measured ~75-85% of the unsorted path) plus several K-sized int64
// temporaries. Here the binning is fused into the load and the loop is
// OpenMP-parallel across indices -- searchsorted cannot be threaded, so on a
// multi-core host the binning speeds up with the thread count on top of the
// per-element branchless win.
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
// binning (87-93% of this kernel's cost on the deployment hardware) instead
// of recomputing it per column.
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
// the whole call, versus O(K log R) searches for the unsorted kernel and
// versus the NumPy boundary-partition pipeline this replaces (whose
// per-record host loop is 79-97% of the sorted path at R >= 10^4). Offsets
// are computed in registers; there is no byte_offsets array.
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
    // loop's steady state is one compare, one multiply-add, one load --
    // strictly less per-element work than the byte-offset kernel this
    // replaces (which reads a precomputed offset from memory instead).
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

}  // namespace colstore

extern "C" {

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
int colstore_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

}  // extern "C"
