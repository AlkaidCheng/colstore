# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
# distutils: language = c++
"""Cython binding for the size-dispatched C++ gather kernel.

The underlying C++ kernel is templated on element size (1/2/4/8 bytes), not
on NumPy dtype kind. A single set of four templates per entry point covers
every fixed-width numeric dtype, plus fixed-width bytes/unicode strings,
datetime64, and timedelta64 -- anything whose itemsize is one of those four
sizes "just works."

Two Python entry points:

* :func:`gather` -- element-indexed (hot path). Caller passes int64 element
  indices; the kernel computes byte addresses internally. No Python-side
  allocation per call; matches the cost of the original per-dtype kernel.

* :func:`gather_bytes` -- byte-offset. Caller passes int64 byte offsets
  directly. Used by the multi-record reader where byte addresses cross
  record boundaries and can't be computed as a simple multiply.
"""

import numpy as np

cimport numpy as cnp
from libc.stdint cimport int64_t, uint8_t

cnp.import_array()


cdef extern from "colstore/gather.hpp" nogil:
    void colstore_gather_indexed_1(const uint8_t*, const int64_t*, uint8_t*,
                                   ptrdiff_t, int)
    void colstore_gather_indexed_2(const uint8_t*, const int64_t*, uint8_t*,
                                   ptrdiff_t, int)
    void colstore_gather_indexed_4(const uint8_t*, const int64_t*, uint8_t*,
                                   ptrdiff_t, int)
    void colstore_gather_indexed_8(const uint8_t*, const int64_t*, uint8_t*,
                                   ptrdiff_t, int)
    void colstore_gather_bytes_1(const uint8_t*, const int64_t*, uint8_t*,
                                 ptrdiff_t, int)
    void colstore_gather_bytes_2(const uint8_t*, const int64_t*, uint8_t*,
                                 ptrdiff_t, int)
    void colstore_gather_bytes_4(const uint8_t*, const int64_t*, uint8_t*,
                                 ptrdiff_t, int)
    void colstore_gather_bytes_8(const uint8_t*, const int64_t*, uint8_t*,
                                 ptrdiff_t, int)
    int colstore_max_threads()


cdef extern from "colstore/gather.hpp" namespace "colstore" nogil:
    ptrdiff_t resolve_thread_count(ptrdiff_t n_indices, int cap)


def max_threads() -> int:
    """Return OpenMP's max thread count (or 1 if OpenMP is disabled)."""
    return colstore_max_threads()


def thread_count_for(Py_ssize_t n_indices, int cap) -> int:
    """Return the thread count the kernel would use for ``n_indices``/``cap``.

    Exposed for tests and diagnostics; mirrors the C++ ``resolve_thread_count``.
    """
    return resolve_thread_count(n_indices, cap)


def gather(cnp.ndarray source, cnp.ndarray indices, cnp.ndarray output,
           int thread_cap=0):
    """Element-indexed gather: ``output[i] = source[indices[i]]``.

    Parameters
    ----------
    source : numpy.ndarray
        1D array with a fixed-size dtype (itemsize 1/2/4/8 bytes).
    indices : numpy.ndarray
        1D ``int64`` array of element positions into ``source``.
    output : numpy.ndarray
        1D array with the same dtype as ``source`` and length matching
        ``indices``. Filled in-place.
    thread_cap : int, optional
        Maximum OpenMP threads to use. ``0`` (default) or any non-positive
        value means the OpenMP maximum; the kernel still drops to a single
        thread for small inputs and scales up to this cap for large ones.

    Raises
    ------
    TypeError
        If dtypes mismatch, ``indices`` is not int64, or the element size is
        not supported.
    ValueError
        If shapes are incompatible.
    """
    if source.ndim != 1 or indices.ndim != 1 or output.ndim != 1:
        raise ValueError("All inputs to gather must be 1D arrays.")
    if indices.dtype != np.int64:
        raise TypeError(f"indices must be int64; got {indices.dtype}.")
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

    cdef int itemsize = source.dtype.itemsize
    cdef const uint8_t* base = <const uint8_t*>cnp.PyArray_DATA(source)
    cdef const int64_t* indices_ptr = <const int64_t*>cnp.PyArray_DATA(indices)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)

    if itemsize == 1:
        with nogil:
            colstore_gather_indexed_1(base, indices_ptr, output_ptr,
                                      n_indices, thread_cap)
    elif itemsize == 2:
        with nogil:
            colstore_gather_indexed_2(base, indices_ptr, output_ptr,
                                      n_indices, thread_cap)
    elif itemsize == 4:
        with nogil:
            colstore_gather_indexed_4(base, indices_ptr, output_ptr,
                                      n_indices, thread_cap)
    elif itemsize == 8:
        with nogil:
            colstore_gather_indexed_8(base, indices_ptr, output_ptr,
                                      n_indices, thread_cap)
    else:
        raise TypeError(
            f"Unsupported element size: {itemsize} bytes. The C++ kernel "
            f"handles 1, 2, 4, and 8 byte elements."
        )


def gather_bytes(cnp.ndarray source, cnp.ndarray byte_offsets,
                 cnp.ndarray output, int thread_cap=0):
    """Byte-offset gather: copy ``itemsize`` bytes from ``source + byte_offsets[i]``.

    Parameters
    ----------
    source : numpy.ndarray
        Source buffer treated as raw bytes (the dtype is ignored; only the
        base pointer matters).
    byte_offsets : numpy.ndarray
        1D ``int64`` array. Each element is a byte offset into ``source``;
        the kernel reads ``output.dtype.itemsize`` bytes starting there.
        The caller guarantees each offset points at an itemsize-aligned
        address (automatic when offsets are computed as ``element_index *
        itemsize``, optionally with fixed per-record headers added).
    output : numpy.ndarray
        1D array determining both the element size and the output dtype.
    thread_cap : int, optional
        Maximum OpenMP threads. ``0`` means the OpenMP maximum.

    Notes
    -----
    This entry point exists for the multi-record reader (PR 2): byte offsets
    there encode record-header skips and per-record column offsets and so
    cannot be reduced to a simple ``index * itemsize``. For the contiguous
    hot path, :func:`gather` is faster because it skips the byte-offset
    array materialization.

    Raises
    ------
    TypeError
        If ``byte_offsets`` is not int64, or the output element size is not
        supported.
    ValueError
        If shapes are incompatible.
    """
    if byte_offsets.ndim != 1 or output.ndim != 1:
        raise ValueError("byte_offsets and output must be 1D arrays.")
    if byte_offsets.dtype != np.int64:
        raise TypeError(f"byte_offsets must be int64; got {byte_offsets.dtype}.")

    cdef ptrdiff_t n_indices = byte_offsets.shape[0]
    if output.shape[0] != n_indices:
        raise ValueError(
            f"output length {output.shape[0]} does not match byte_offsets "
            f"length {n_indices}."
        )
    if n_indices == 0:
        return

    cdef int itemsize = output.dtype.itemsize
    cdef const uint8_t* base = <const uint8_t*>cnp.PyArray_DATA(source)
    cdef const int64_t* offsets_ptr = <const int64_t*>cnp.PyArray_DATA(byte_offsets)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)

    if itemsize == 1:
        with nogil:
            colstore_gather_bytes_1(base, offsets_ptr, output_ptr,
                                    n_indices, thread_cap)
    elif itemsize == 2:
        with nogil:
            colstore_gather_bytes_2(base, offsets_ptr, output_ptr,
                                    n_indices, thread_cap)
    elif itemsize == 4:
        with nogil:
            colstore_gather_bytes_4(base, offsets_ptr, output_ptr,
                                    n_indices, thread_cap)
    elif itemsize == 8:
        with nogil:
            colstore_gather_bytes_8(base, offsets_ptr, output_ptr,
                                    n_indices, thread_cap)
    else:
        raise TypeError(
            f"Unsupported element size: {itemsize} bytes. The C++ kernel "
            f"handles 1, 2, 4, and 8 byte elements."
        )
