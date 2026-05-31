# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
# distutils: language = c++
"""Cython binding for the C++ gather kernel.

Exposes a single Python-callable ``gather`` that dispatches by NumPy dtype
to the appropriate ``extern "C"`` wrapper in ``include/colstore/gather.hpp``.
Index arrays must be ``int64``; source and output must share the same
fixed-size dtype.
"""

import numpy as np

cimport numpy as cnp
from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

cnp.import_array()


cdef extern from "colstore/gather.hpp" nogil:
    void colstore_gather_f32(const float*, const int64_t*, float*,
                             ptrdiff_t)
    void colstore_gather_f64(const double*, const int64_t*, double*,
                             ptrdiff_t)
    void colstore_gather_i8(const int8_t*, const int64_t*, int8_t*,
                            ptrdiff_t)
    void colstore_gather_i16(const int16_t*, const int64_t*, int16_t*,
                             ptrdiff_t)
    void colstore_gather_i32(const int32_t*, const int64_t*, int32_t*,
                             ptrdiff_t)
    void colstore_gather_i64(const int64_t*, const int64_t*, int64_t*,
                             ptrdiff_t)
    void colstore_gather_u8(const uint8_t*, const int64_t*, uint8_t*,
                            ptrdiff_t)
    void colstore_gather_u16(const uint16_t*, const int64_t*, uint16_t*,
                             ptrdiff_t)
    void colstore_gather_u32(const uint32_t*, const int64_t*, uint32_t*,
                             ptrdiff_t)
    void colstore_gather_u64(const uint64_t*, const int64_t*, uint64_t*,
                             ptrdiff_t)
    int colstore_max_threads()


def max_threads() -> int:
    """Return OpenMP's max thread count (or 1 if OpenMP is disabled)."""
    return colstore_max_threads()


def gather(cnp.ndarray source, cnp.ndarray indices, cnp.ndarray output):
    """Compute ``output[i] = source[indices[i]]`` via the parallel C++ kernel.

    Parameters
    ----------
    source : numpy.ndarray
        1D array of any supported fixed-size dtype.
    indices : numpy.ndarray
        1D ``int64`` array of positions into ``source``.
    output : numpy.ndarray
        1D array with the same dtype as ``source`` and length matching
        ``indices``. Filled in-place.

    Raises
    ------
    TypeError
        If dtypes mismatch or the source dtype is unsupported.
    ValueError
        If shapes are incompatible.
    """
    if source.ndim != 1 or indices.ndim != 1 or output.ndim != 1:
        raise ValueError("All inputs to gather must be 1D arrays.")
    if indices.dtype != np.int64:
        raise TypeError(
            f"indices must be int64; got {indices.dtype}."
        )
    if source.dtype != output.dtype:
        raise TypeError(
            f"source dtype {source.dtype} does not match output dtype "
            f"{output.dtype}."
        )

    cdef ptrdiff_t n_indices = indices.shape[0]
    if output.shape[0] != n_indices:
        raise ValueError(
            f"output length {output.shape[0]} does not match indices "
            f"length {n_indices}."
        )
    if n_indices == 0:
        return

    cdef str kind_code = source.dtype.kind
    cdef int itemsize = source.dtype.itemsize
    cdef void* source_ptr = cnp.PyArray_DATA(source)
    cdef int64_t* indices_ptr = <int64_t*>cnp.PyArray_DATA(indices)
    cdef void* output_ptr = cnp.PyArray_DATA(output)

    # 'f' floating, 'i' signed int, 'u'/'b' unsigned/bool
    cdef bint dispatched = False
    if kind_code == 'f':
        if itemsize == 4:
            with nogil:
                colstore_gather_f32(<const float*>source_ptr, indices_ptr,
                                    <float*>output_ptr, n_indices)
            dispatched = True
        elif itemsize == 8:
            with nogil:
                colstore_gather_f64(<const double*>source_ptr, indices_ptr,
                                    <double*>output_ptr, n_indices)
            dispatched = True
    elif kind_code == 'i':
        if itemsize == 1:
            with nogil:
                colstore_gather_i8(<const int8_t*>source_ptr, indices_ptr,
                                   <int8_t*>output_ptr, n_indices)
            dispatched = True
        elif itemsize == 2:
            with nogil:
                colstore_gather_i16(<const int16_t*>source_ptr, indices_ptr,
                                    <int16_t*>output_ptr, n_indices)
            dispatched = True
        elif itemsize == 4:
            with nogil:
                colstore_gather_i32(<const int32_t*>source_ptr, indices_ptr,
                                    <int32_t*>output_ptr, n_indices)
            dispatched = True
        elif itemsize == 8:
            with nogil:
                colstore_gather_i64(<const int64_t*>source_ptr, indices_ptr,
                                    <int64_t*>output_ptr, n_indices)
            dispatched = True
    elif kind_code == 'u' or kind_code == 'b':
        if itemsize == 1:
            with nogil:
                colstore_gather_u8(<const uint8_t*>source_ptr, indices_ptr,
                                   <uint8_t*>output_ptr, n_indices)
            dispatched = True
        elif itemsize == 2:
            with nogil:
                colstore_gather_u16(<const uint16_t*>source_ptr, indices_ptr,
                                    <uint16_t*>output_ptr, n_indices)
            dispatched = True
        elif itemsize == 4:
            with nogil:
                colstore_gather_u32(<const uint32_t*>source_ptr, indices_ptr,
                                    <uint32_t*>output_ptr, n_indices)
            dispatched = True
        elif itemsize == 8:
            with nogil:
                colstore_gather_u64(<const uint64_t*>source_ptr, indices_ptr,
                                    <uint64_t*>output_ptr, n_indices)
            dispatched = True

    if not dispatched:
        raise TypeError(
            f"Unsupported source dtype: {source.dtype}. The C++ kernel "
            f"handles float32/64, int8/16/32/64, uint8/16/32/64, and bool."
        )
