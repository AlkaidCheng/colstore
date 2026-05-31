// Implementation of the templated gather kernel and its extern "C" wrappers.
//
// Performance levers used here:
//
//  * OpenMP parallel for. The loop body is fully independent across i, so
//    a static schedule keeps overhead minimal and maps cleanly to the OS
//    thread pool.
//  * Software prefetching. ``__builtin_prefetch`` issues a memory hint a
//    few iterations ahead, which hides part of the L3/DRAM miss latency
//    when ``indices`` are scattered. The prefetch is non-faulting on
//    out-of-range addresses but we still guard with a bounds check to
//    keep TSan happy.
//  * ``__restrict__`` pointer qualifiers. They let the compiler assume
//    source/indices/output do not alias, enabling vectorization of the
//    store stream even though the load stream is scattered.
//
// The kernel is templated by element type, with explicit instantiations
// below for every NumPy fixed-size dtype we expose through the binding.

#include "colstore/gather.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>

namespace colstore {

// Resolve how many OpenMP threads to use for a gather of ``n_indices``
// elements, given a caller-supplied cap.
//
// The kernel is memory-bandwidth-bound, so two rules apply:
//   1. Below PARALLEL_THRESHOLD the OpenMP fork/join + barrier cost dwarfs the
//      actual work, so run serially (return 1).
//   2. Above it, scale roughly one thread per ELEMENTS_PER_THREAD elements,
//      clamped to ``cap``. Bandwidth saturates at a small thread count well
//      below the core count, so the cap (typically <= 8) is the real limit;
//      the work-proportional term just avoids spinning up the full cap for
//      mid-sized gathers.
//
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
  std::ptrdiff_t by_work = n_indices / ELEMENTS_PER_THREAD + 1;
  std::ptrdiff_t threads = std::min<std::ptrdiff_t>(effective_cap, by_work);
  return std::max<std::ptrdiff_t>(1, threads);
#else
  (void)n_indices;
  (void)cap;
  return 1;
#endif
}

template <typename T>
void gather_typed(const T* __restrict__ source,
                  const std::int64_t* __restrict__ indices,
                  T* __restrict__ output,
                  std::ptrdiff_t n_indices,
                  int thread_cap,
                  std::ptrdiff_t prefetch_distance) {
  const std::ptrdiff_t n_threads = resolve_thread_count(n_indices, thread_cap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_threads)) \
    if (n_threads > 1)
#else
  (void)n_threads;
#endif
  for (std::ptrdiff_t i = 0; i < n_indices; ++i) {
    if (i + prefetch_distance < n_indices) {
      // Hint: read-only, low temporal locality (we won't revisit).
      __builtin_prefetch(&source[indices[i + prefetch_distance]], 0, 0);
    }
    output[i] = source[indices[i]];
  }
}

// Explicit instantiations for every dtype we expose. Keeping these in the
// .cpp avoids re-instantiating the template in every translation unit that
// includes the header.
template void gather_typed<float>(const float*, const std::int64_t*, float*,
                                  std::ptrdiff_t, int, std::ptrdiff_t);
template void gather_typed<double>(const double*, const std::int64_t*, double*,
                                   std::ptrdiff_t, int, std::ptrdiff_t);
template void gather_typed<std::int8_t>(const std::int8_t*, const std::int64_t*,
                                        std::int8_t*, std::ptrdiff_t, int, std::ptrdiff_t);
template void gather_typed<std::int16_t>(const std::int16_t*,
                                         const std::int64_t*, std::int16_t*,
                                         std::ptrdiff_t, int, std::ptrdiff_t);
template void gather_typed<std::int32_t>(const std::int32_t*,
                                         const std::int64_t*, std::int32_t*,
                                         std::ptrdiff_t, int, std::ptrdiff_t);
template void gather_typed<std::int64_t>(const std::int64_t*,
                                         const std::int64_t*, std::int64_t*,
                                         std::ptrdiff_t, int, std::ptrdiff_t);
template void gather_typed<std::uint8_t>(const std::uint8_t*,
                                         const std::int64_t*, std::uint8_t*,
                                         std::ptrdiff_t, int, std::ptrdiff_t);
template void gather_typed<std::uint16_t>(const std::uint16_t*,
                                          const std::int64_t*, std::uint16_t*,
                                          std::ptrdiff_t, int, std::ptrdiff_t);
template void gather_typed<std::uint32_t>(const std::uint32_t*,
                                          const std::int64_t*, std::uint32_t*,
                                          std::ptrdiff_t, int, std::ptrdiff_t);
template void gather_typed<std::uint64_t>(const std::uint64_t*,
                                          const std::int64_t*, std::uint64_t*,
                                          std::ptrdiff_t, int, std::ptrdiff_t);

}  // namespace colstore

extern "C" {

void colstore_gather_f32(const float* source, const std::int64_t* indices,
                         float* output, std::ptrdiff_t n, int thread_cap) {
  colstore::gather_typed<float>(source, indices, output, n, thread_cap);
}
void colstore_gather_f64(const double* source, const std::int64_t* indices,
                         double* output, std::ptrdiff_t n, int thread_cap) {
  colstore::gather_typed<double>(source, indices, output, n, thread_cap);
}
void colstore_gather_i8(const std::int8_t* source,
                        const std::int64_t* indices, std::int8_t* output,
                        std::ptrdiff_t n, int thread_cap) {
  colstore::gather_typed<std::int8_t>(source, indices, output, n, thread_cap);
}
void colstore_gather_i16(const std::int16_t* source,
                         const std::int64_t* indices, std::int16_t* output,
                         std::ptrdiff_t n, int thread_cap) {
  colstore::gather_typed<std::int16_t>(source, indices, output, n, thread_cap);
}
void colstore_gather_i32(const std::int32_t* source,
                         const std::int64_t* indices, std::int32_t* output,
                         std::ptrdiff_t n, int thread_cap) {
  colstore::gather_typed<std::int32_t>(source, indices, output, n, thread_cap);
}
void colstore_gather_i64(const std::int64_t* source,
                         const std::int64_t* indices, std::int64_t* output,
                         std::ptrdiff_t n, int thread_cap) {
  colstore::gather_typed<std::int64_t>(source, indices, output, n, thread_cap);
}
void colstore_gather_u8(const std::uint8_t* source,
                        const std::int64_t* indices, std::uint8_t* output,
                        std::ptrdiff_t n, int thread_cap) {
  colstore::gather_typed<std::uint8_t>(source, indices, output, n, thread_cap);
}
void colstore_gather_u16(const std::uint16_t* source,
                         const std::int64_t* indices, std::uint16_t* output,
                         std::ptrdiff_t n, int thread_cap) {
  colstore::gather_typed<std::uint16_t>(source, indices, output, n, thread_cap);
}
void colstore_gather_u32(const std::uint32_t* source,
                         const std::int64_t* indices, std::uint32_t* output,
                         std::ptrdiff_t n, int thread_cap) {
  colstore::gather_typed<std::uint32_t>(source, indices, output, n, thread_cap);
}
void colstore_gather_u64(const std::uint64_t* source,
                         const std::int64_t* indices, std::uint64_t* output,
                         std::ptrdiff_t n, int thread_cap) {
  colstore::gather_typed<std::uint64_t>(source, indices, output, n, thread_cap);
}

int colstore_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

}  // extern "C"
