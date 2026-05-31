// Public header for the colstore gather kernel.
//
// The C++ implementation provides a single templated function
// ``gather_typed<T>`` that copies ``output[i] = source[indices[i]]`` for every
// ``i``. The loop is parallelized with OpenMP (when available) and uses
// software prefetching to keep memory loads in flight on scattered access
// patterns.
//
// ``extern "C"`` wrappers provide dtype-specialized entry points that Cython
// can bind to without needing to instantiate C++ templates on the Cython
// side. Cython's job is reduced to: pick the right wrapper by NumPy dtype,
// hand it raw pointers, release the GIL.

#pragma once

#include <cstddef>
#include <cstdint>

namespace colstore {

// Default prefetch distance in elements. Eight iterations ahead is roughly
// the sweet spot on current x86 hardware for scattered memory loads; the
// caller can override per-call.
constexpr std::ptrdiff_t DEFAULT_PREFETCH_DISTANCE = 8;

// Template declaration. Definitions and explicit instantiations live in
// src/cpp/gather.cpp so the header stays cheap to include.
template <typename T>
void gather_typed(const T* source,
                  const std::int64_t* indices,
                  T* output,
                  std::ptrdiff_t n_indices,
                  std::ptrdiff_t prefetch_distance = DEFAULT_PREFETCH_DISTANCE);

}  // namespace colstore

// C-callable wrappers used by the Cython binding. One per supported NumPy
// dtype. Cython sees only this extern "C" surface, never the C++ template.
extern "C" {

void colstore_gather_f32(const float* source,
                         const std::int64_t* indices,
                         float* output,
                         std::ptrdiff_t n);
void colstore_gather_f64(const double* source,
                         const std::int64_t* indices,
                         double* output,
                         std::ptrdiff_t n);
void colstore_gather_i8(const std::int8_t* source,
                        const std::int64_t* indices,
                        std::int8_t* output,
                        std::ptrdiff_t n);
void colstore_gather_i16(const std::int16_t* source,
                         const std::int64_t* indices,
                         std::int16_t* output,
                         std::ptrdiff_t n);
void colstore_gather_i32(const std::int32_t* source,
                         const std::int64_t* indices,
                         std::int32_t* output,
                         std::ptrdiff_t n);
void colstore_gather_i64(const std::int64_t* source,
                         const std::int64_t* indices,
                         std::int64_t* output,
                         std::ptrdiff_t n);
void colstore_gather_u8(const std::uint8_t* source,
                        const std::int64_t* indices,
                        std::uint8_t* output,
                        std::ptrdiff_t n);
void colstore_gather_u16(const std::uint16_t* source,
                         const std::int64_t* indices,
                         std::uint16_t* output,
                         std::ptrdiff_t n);
void colstore_gather_u32(const std::uint32_t* source,
                         const std::int64_t* indices,
                         std::uint32_t* output,
                         std::ptrdiff_t n);
void colstore_gather_u64(const std::uint64_t* source,
                         const std::int64_t* indices,
                         std::uint64_t* output,
                         std::ptrdiff_t n);

// Returns the OpenMP thread cap that gathers will use, or 1 if OpenMP is
// not compiled in. Exposed for diagnostics.
int colstore_max_threads();

}  // extern "C"
