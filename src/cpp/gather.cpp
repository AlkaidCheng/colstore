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
template <typename T>
void gather_indexed_typed(const std::uint8_t* COLSTORE_RESTRICT base,
                          const std::int64_t* COLSTORE_RESTRICT indices,
                          std::uint8_t* COLSTORE_RESTRICT output,
                          std::ptrdiff_t n_indices,
                          int thread_cap,
                          std::ptrdiff_t prefetch_distance) {
  const T* src = reinterpret_cast<const T*>(base);
  T* dst = reinterpret_cast<T*>(output);
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::ptrdiff_t i = 0; i < n_indices; ++i) {
    if (i + prefetch_distance < n_indices) {
      COLSTORE_PREFETCH(&src[indices[i + prefetch_distance]]);
    }
    dst[i] = src[indices[i]];
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
    if (i + prefetch_distance < n_indices) {
      COLSTORE_PREFETCH(base + byte_offsets[i + prefetch_distance]);
    }
    dst[i] = *reinterpret_cast<const T*>(base + byte_offsets[i]);
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

}  // namespace colstore

extern "C" {

void colstore_gather_indexed_1(const std::uint8_t* base,
                               const std::int64_t* indices,
                               std::uint8_t* output,
                               std::ptrdiff_t n, int thread_cap) {
  colstore::gather_indexed_typed<std::uint8_t>(base, indices, output, n,
                                               thread_cap);
}
void colstore_gather_indexed_2(const std::uint8_t* base,
                               const std::int64_t* indices,
                               std::uint8_t* output,
                               std::ptrdiff_t n, int thread_cap) {
  colstore::gather_indexed_typed<std::uint16_t>(base, indices, output, n,
                                                thread_cap);
}
void colstore_gather_indexed_4(const std::uint8_t* base,
                               const std::int64_t* indices,
                               std::uint8_t* output,
                               std::ptrdiff_t n, int thread_cap) {
  colstore::gather_indexed_typed<std::uint32_t>(base, indices, output, n,
                                                thread_cap);
}
void colstore_gather_indexed_8(const std::uint8_t* base,
                               const std::int64_t* indices,
                               std::uint8_t* output,
                               std::ptrdiff_t n, int thread_cap) {
  colstore::gather_indexed_typed<std::uint64_t>(base, indices, output, n,
                                                thread_cap);
}

void colstore_gather_bytes_1(const std::uint8_t* base,
                             const std::int64_t* byte_offsets,
                             std::uint8_t* output,
                             std::ptrdiff_t n, int thread_cap) {
  colstore::gather_bytes_typed<std::uint8_t>(base, byte_offsets, output, n,
                                             thread_cap);
}
void colstore_gather_bytes_2(const std::uint8_t* base,
                             const std::int64_t* byte_offsets,
                             std::uint8_t* output,
                             std::ptrdiff_t n, int thread_cap) {
  colstore::gather_bytes_typed<std::uint16_t>(base, byte_offsets, output, n,
                                              thread_cap);
}
void colstore_gather_bytes_4(const std::uint8_t* base,
                             const std::int64_t* byte_offsets,
                             std::uint8_t* output,
                             std::ptrdiff_t n, int thread_cap) {
  colstore::gather_bytes_typed<std::uint32_t>(base, byte_offsets, output, n,
                                              thread_cap);
}
void colstore_gather_bytes_8(const std::uint8_t* base,
                             const std::int64_t* byte_offsets,
                             std::uint8_t* output,
                             std::ptrdiff_t n, int thread_cap) {
  colstore::gather_bytes_typed<std::uint64_t>(base, byte_offsets, output, n,
                                              thread_cap);
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

int colstore_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

}  // extern "C"
