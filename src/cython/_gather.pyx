# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
# distutils: language = c++
"""Cython binding for the size-dispatched C++ gather kernel.

The underlying C++ kernel is templated on element size (1/2/4/8 bytes), not
on NumPy dtype kind. A single set of four templates per entry point covers
every fixed-width numeric dtype, plus fixed-width bytes/unicode strings,
datetime64, and timedelta64 -- anything whose itemsize is one of those four
sizes "just works."

Three Python entry points:

* :func:`gather` -- element-indexed (hot path). Caller passes int64 element
  indices; the kernel computes byte addresses internally. No Python-side
  allocation per call.

* :func:`gather_into` -- alias for :func:`gather` retained for callers that
  reach the binding directly (the colstore.kernels dispatcher uses this
  name to make the no-allocation contract explicit at the call site).

* :func:`gather_bytes` -- byte-offset. Caller passes int64 byte offsets
  directly. Used by the multi-record reader (PR 2) where byte addresses
  cross record boundaries.
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
    void colstore_copy_multirecord_range(const uint8_t*, uint8_t*,
                                         int64_t, int64_t,
                                         const int64_t*, const int64_t*,
                                         const int64_t*, int64_t,
                                         int64_t, int64_t)
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
        ``indices``. Filled in-place; allocation is the caller's job.
    thread_cap : int, optional
        Maximum OpenMP threads. ``0`` (default) means the OpenMP maximum;
        the kernel still drops to a single thread for small inputs and
        scales up to this cap for large ones.

    Raises
    ------
    TypeError
        If dtypes mismatch, ``indices`` is not int64, or the element size
        is not supported.
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


def gather_into(cnp.ndarray source, cnp.ndarray indices, cnp.ndarray output,
                int thread_cap=0):
    """Alias for :func:`gather`.

    The name is used by :mod:`colstore.kernels` to make the no-allocation
    contract explicit at the call site -- the kernel never allocates the
    output buffer; the caller does. Functionally identical to :func:`gather`.
    """
    gather(source, indices, output, thread_cap)


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
        address.
    output : numpy.ndarray
        1D array determining both the element size and the output dtype.
    thread_cap : int, optional
        Maximum OpenMP threads. ``0`` means the OpenMP maximum.

    Notes
    -----
    Used by the multi-record reader (PR 2): byte offsets there encode
    record-header skips and per-record column offsets and cannot be reduced
    to a simple ``index * itemsize``. For the contiguous hot path,
    :func:`gather` is faster because it skips the byte-offset array
    materialization.

    Raises
    ------
    TypeError
        If ``byte_offsets`` is not int64, or the output element size is
        not supported.
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


def copy_multirecord_range(cnp.ndarray source, cnp.ndarray output,
                           long long start, long long stop,
                           cnp.ndarray record_starts_rows,
                           cnp.ndarray record_starts_bytes,
                           cnp.ndarray n_rows_per_record,
                           long long col_prefix_bytes, long long itemsize):
    """Copy global rows ``[start, stop)`` of one column from a multi-record file.

    The output is filled with the packed, contiguous rows of the range. Each
    overlapping record contributes one ``memcpy``; record membership is found
    by binary search inside the kernel, so no per-record Python work happens.

    Parameters
    ----------
    source : numpy.ndarray
        Whole-file ``uint8`` mmap; treated as the raw byte base pointer.
    output : numpy.ndarray
        1D destination array of length ``stop - start`` and the column's dtype.
        Filled in-place. Its dtype must be the on-disk dtype in *native* byte
        order -- this is a raw byte copy and does not byteswap.
    start, stop : int
        Half-open global row range. ``stop > start`` is required by the caller;
        an empty range is a no-op.
    record_starts_rows : numpy.ndarray
        1D ``int64`` cumulative row counts, length ``n_records + 1``.
    record_starts_bytes : numpy.ndarray
        1D ``int64`` per-record body byte offsets, length ``n_records``.
    n_rows_per_record : numpy.ndarray
        1D ``int64`` per-record row counts, length ``n_records``.
    col_prefix_bytes : int
        Sum of itemsizes of the columns preceding this one in a record body.
    itemsize : int
        Bytes per element of the column.

    Raises
    ------
    TypeError
        If any index array is not ``int64``.
    ValueError
        If the index arrays are not 1D or have inconsistent lengths.
    """
    if (record_starts_rows.ndim != 1 or record_starts_bytes.ndim != 1
            or n_rows_per_record.ndim != 1 or output.ndim != 1):
        raise ValueError("Index arrays and output must be 1D.")
    if (record_starts_rows.dtype != np.int64
            or record_starts_bytes.dtype != np.int64
            or n_rows_per_record.dtype != np.int64):
        raise TypeError("record index arrays must be int64.")

    cdef long long n_records = record_starts_bytes.shape[0]
    if n_rows_per_record.shape[0] != n_records:
        raise ValueError("n_rows_per_record length must match record count.")
    if record_starts_rows.shape[0] != n_records + 1:
        raise ValueError("record_starts_rows length must be n_records + 1.")
    if stop <= start:
        return

    cdef const uint8_t* base = <const uint8_t*>cnp.PyArray_DATA(source)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef const int64_t* rsr = <const int64_t*>cnp.PyArray_DATA(record_starts_rows)
    cdef const int64_t* rsb = <const int64_t*>cnp.PyArray_DATA(record_starts_bytes)
    cdef const int64_t* nrr = <const int64_t*>cnp.PyArray_DATA(n_rows_per_record)

    with nogil:
        colstore_copy_multirecord_range(base, output_ptr, start, stop,
                                        rsr, rsb, nrr, n_records,
                                        col_prefix_bytes, itemsize)
