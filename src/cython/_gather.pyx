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
from libc.stdint cimport int32_t, int64_t, uint8_t

cnp.import_array()


cdef extern from "colstore/gather.hpp" nogil:
    void colstore_gather_indexed_1(const uint8_t*, const int64_t*, uint8_t*,
                                   ptrdiff_t, int, ptrdiff_t)
    void colstore_gather_indexed_2(const uint8_t*, const int64_t*, uint8_t*,
                                   ptrdiff_t, int, ptrdiff_t)
    void colstore_gather_indexed_4(const uint8_t*, const int64_t*, uint8_t*,
                                   ptrdiff_t, int, ptrdiff_t)
    void colstore_gather_indexed_8(const uint8_t*, const int64_t*, uint8_t*,
                                   ptrdiff_t, int, ptrdiff_t)
    void colstore_gather_bytes_1(const uint8_t*, const int64_t*, uint8_t*,
                                 ptrdiff_t, int, ptrdiff_t)
    void colstore_gather_bytes_2(const uint8_t*, const int64_t*, uint8_t*,
                                 ptrdiff_t, int, ptrdiff_t)
    void colstore_gather_bytes_4(const uint8_t*, const int64_t*, uint8_t*,
                                 ptrdiff_t, int, ptrdiff_t)
    void colstore_gather_bytes_8(const uint8_t*, const int64_t*, uint8_t*,
                                 ptrdiff_t, int, ptrdiff_t)
    void colstore_copy_multirecord_range(const uint8_t*, uint8_t*,
                                         int64_t, int64_t,
                                         const int64_t*, const int64_t*,
                                         const int64_t*, int64_t,
                                         int64_t, int64_t)
    void colstore_gather_multirecord_1(const uint8_t*, const int64_t*, uint8_t*,
                                       ptrdiff_t, const int64_t*, const int64_t*,
                                       const int64_t*, int64_t, int64_t, int, ptrdiff_t)
    void colstore_gather_multirecord_2(const uint8_t*, const int64_t*, uint8_t*,
                                       ptrdiff_t, const int64_t*, const int64_t*,
                                       const int64_t*, int64_t, int64_t, int, ptrdiff_t)
    void colstore_gather_multirecord_4(const uint8_t*, const int64_t*, uint8_t*,
                                       ptrdiff_t, const int64_t*, const int64_t*,
                                       const int64_t*, int64_t, int64_t, int, ptrdiff_t)
    void colstore_gather_multirecord_8(const uint8_t*, const int64_t*, uint8_t*,
                                       ptrdiff_t, const int64_t*, const int64_t*,
                                       const int64_t*, int64_t, int64_t, int, ptrdiff_t)
    void colstore_gather_multirecord_bins_1(
        const uint8_t*, const int64_t*, uint8_t*, int32_t*, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_bins_2(
        const uint8_t*, const int64_t*, uint8_t*, int32_t*, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_bins_4(
        const uint8_t*, const int64_t*, uint8_t*, int32_t*, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_bins_8(
        const uint8_t*, const int64_t*, uint8_t*, int32_t*, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_withbins_1(
        const uint8_t*, const int64_t*, uint8_t*, const int32_t*, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int, ptrdiff_t)
    void colstore_gather_multirecord_withbins_2(
        const uint8_t*, const int64_t*, uint8_t*, const int32_t*, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int, ptrdiff_t)
    void colstore_gather_multirecord_withbins_4(
        const uint8_t*, const int64_t*, uint8_t*, const int32_t*, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int, ptrdiff_t)
    void colstore_gather_multirecord_withbins_8(
        const uint8_t*, const int64_t*, uint8_t*, const int32_t*, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int, ptrdiff_t)
    void colstore_gather_multirecord_sorted_1(
        const uint8_t*, const int64_t*, uint8_t*, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_sorted_2(
        const uint8_t*, const int64_t*, uint8_t*, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_sorted_4(
        const uint8_t*, const int64_t*, uint8_t*, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_sorted_8(
        const uint8_t*, const int64_t*, uint8_t*, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_strided_1(
        const uint8_t*, uint8_t*, int64_t, int64_t, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_strided_2(
        const uint8_t*, uint8_t*, int64_t, int64_t, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_strided_4(
        const uint8_t*, uint8_t*, int64_t, int64_t, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_strided_8(
        const uint8_t*, uint8_t*, int64_t, int64_t, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_uniform_1(
        const uint8_t*, const int64_t*, uint8_t*, ptrdiff_t,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_uniform_2(
        const uint8_t*, const int64_t*, uint8_t*, ptrdiff_t,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_uniform_4(
        const uint8_t*, const int64_t*, uint8_t*, ptrdiff_t,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_uniform_8(
        const uint8_t*, const int64_t*, uint8_t*, ptrdiff_t,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_uniform_bins_1(
        const uint8_t*, const int64_t*, uint8_t*, int32_t*, ptrdiff_t,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_uniform_bins_2(
        const uint8_t*, const int64_t*, uint8_t*, int32_t*, ptrdiff_t,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_uniform_bins_4(
        const uint8_t*, const int64_t*, uint8_t*, int32_t*, ptrdiff_t,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_uniform_bins_8(
        const uint8_t*, const int64_t*, uint8_t*, int32_t*, ptrdiff_t,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_uniform_withbins_1(
        const uint8_t*, const int64_t*, uint8_t*, const int32_t*, ptrdiff_t,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_uniform_withbins_2(
        const uint8_t*, const int64_t*, uint8_t*, const int32_t*, ptrdiff_t,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_uniform_withbins_4(
        const uint8_t*, const int64_t*, uint8_t*, const int32_t*, ptrdiff_t,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_uniform_withbins_8(
        const uint8_t*, const int64_t*, uint8_t*, const int32_t*, ptrdiff_t,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int, ptrdiff_t)
    void colstore_gather_multirecord_withbins_rbase_1(
        const uint8_t*, const int64_t*, uint8_t*, const int32_t*, ptrdiff_t,
        const int64_t*, int, ptrdiff_t)
    void colstore_gather_multirecord_withbins_rbase_2(
        const uint8_t*, const int64_t*, uint8_t*, const int32_t*, ptrdiff_t,
        const int64_t*, int, ptrdiff_t)
    void colstore_gather_multirecord_withbins_rbase_4(
        const uint8_t*, const int64_t*, uint8_t*, const int32_t*, ptrdiff_t,
        const int64_t*, int, ptrdiff_t)
    void colstore_gather_multirecord_withbins_rbase_8(
        const uint8_t*, const int64_t*, uint8_t*, const int32_t*, ptrdiff_t,
        const int64_t*, int, ptrdiff_t)
    int colstore_gather_multirecord_mask_1(
        const uint8_t*, const uint8_t*, uint8_t*, int64_t, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t,
        int, ptrdiff_t)
    int colstore_gather_multirecord_mask_2(
        const uint8_t*, const uint8_t*, uint8_t*, int64_t, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t,
        int, ptrdiff_t)
    int colstore_gather_multirecord_mask_4(
        const uint8_t*, const uint8_t*, uint8_t*, int64_t, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t,
        int, ptrdiff_t)
    int colstore_gather_multirecord_mask_8(
        const uint8_t*, const uint8_t*, uint8_t*, int64_t, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t,
        int, ptrdiff_t)
    int colstore_max_threads()


cdef extern from "colstore/gather.hpp" namespace "colstore" nogil:
    ptrdiff_t resolve_thread_count(ptrdiff_t n_indices, int cap)
    const ptrdiff_t DEFAULT_PREFETCH_DISTANCE


def max_threads() -> int:
    """Return OpenMP's max thread count (or 1 if OpenMP is disabled)."""
    return colstore_max_threads()


def thread_count_for(Py_ssize_t n_indices, int cap) -> int:
    """Return the thread count the kernel would use for ``n_indices``/``cap``.

    Exposed for tests and diagnostics; mirrors the C++ ``resolve_thread_count``.
    """
    return resolve_thread_count(n_indices, cap)


def default_prefetch_distance() -> int:
    """Return the compiled-in default prefetch distance (gather.hpp).

    Exposed so the Python config layer and tests can pin against the single
    authoritative constant instead of duplicating the value.
    """
    return DEFAULT_PREFETCH_DISTANCE



def _require_c_contiguous(pairs):
    """Raise if any (name, array) pair is not C-contiguous.

    Every entry point in this module hands ``PyArray_DATA`` pointers to C++
    kernels that index them as dense arrays. A strided view (``a[::2]``,
    ``a[::-1]``) would be read at the wrong positions -- silently wrong
    values for positive strides, out-of-bounds reads for negative ones --
    so contiguity is a hard requirement, validated here. Callers that may
    hold strided arrays should normalize with ``np.ascontiguousarray``.
    """
    for name, array in pairs:
        if not array.flags.c_contiguous:
            raise ValueError(
                f"{name} must be C-contiguous; pass np.ascontiguousarray({name})."
            )


def gather(cnp.ndarray source, cnp.ndarray indices, cnp.ndarray output,
           int thread_cap=0, Py_ssize_t prefetch_distance=-1):
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
    prefetch_distance : int, optional
        Software-prefetch look-ahead in elements. ``> 0`` prefetches that
        many iterations ahead; ``0`` disables prefetching (useful when the
        source is cache-resident); negative (default) uses the compiled
        default, :func:`default_prefetch_distance`.

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
    _require_c_contiguous((("source", source), ("indices", indices), ("output", output)))
    cdef const uint8_t* base = <const uint8_t*>cnp.PyArray_DATA(source)
    cdef const int64_t* indices_ptr = <const int64_t*>cnp.PyArray_DATA(indices)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)

    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    if itemsize == 1:
        with nogil:
            colstore_gather_indexed_1(base, indices_ptr, output_ptr,
                                      n_indices, thread_cap, pd)
    elif itemsize == 2:
        with nogil:
            colstore_gather_indexed_2(base, indices_ptr, output_ptr,
                                      n_indices, thread_cap, pd)
    elif itemsize == 4:
        with nogil:
            colstore_gather_indexed_4(base, indices_ptr, output_ptr,
                                      n_indices, thread_cap, pd)
    elif itemsize == 8:
        with nogil:
            colstore_gather_indexed_8(base, indices_ptr, output_ptr,
                                      n_indices, thread_cap, pd)
    else:
        raise TypeError(
            f"Unsupported element size: {itemsize} bytes. The C++ kernel "
            f"handles 1, 2, 4, and 8 byte elements."
        )


def gather_into(cnp.ndarray source, cnp.ndarray indices, cnp.ndarray output,
                int thread_cap=0, Py_ssize_t prefetch_distance=-1):
    """Alias for :func:`gather`.

    The name is used by :mod:`colstore.kernels` to make the no-allocation
    contract explicit at the call site -- the kernel never allocates the
    output buffer; the caller does. Functionally identical to :func:`gather`.
    """
    gather(source, indices, output, thread_cap, prefetch_distance)


def gather_bytes(cnp.ndarray source, cnp.ndarray byte_offsets,
                 cnp.ndarray output, int thread_cap=0, Py_ssize_t prefetch_distance=-1):
    """Byte-offset gather: copy ``itemsize`` bytes from ``source + byte_offsets[i]``.

    Parameters
    ----------
    source : numpy.ndarray
        Source buffer treated as raw bytes (the dtype is ignored; only the
        base pointer matters).
    byte_offsets : numpy.ndarray
        1D ``int64`` array. Each element is a byte offset into ``source``;
        the kernel reads ``output.dtype.itemsize`` bytes starting there.
        Offsets need not be itemsize-aligned: source loads are
        alignment-safe (packed record bodies make misaligned columns
        legal).
    output : numpy.ndarray
        1D array determining both the element size and the output dtype.
    thread_cap : int, optional
        Maximum OpenMP threads. ``0`` means the OpenMP maximum.
    prefetch_distance : int, optional
        Software-prefetch look-ahead in elements. ``> 0`` prefetches that
        many iterations ahead; ``0`` disables prefetching (useful when the
        source is cache-resident); negative (default) uses the compiled
        default, :func:`default_prefetch_distance`.

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
    _require_c_contiguous((("source", source), ("byte_offsets", byte_offsets), ("output", output)))
    cdef const uint8_t* base = <const uint8_t*>cnp.PyArray_DATA(source)
    cdef const int64_t* offsets_ptr = <const int64_t*>cnp.PyArray_DATA(byte_offsets)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)

    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    if itemsize == 1:
        with nogil:
            colstore_gather_bytes_1(base, offsets_ptr, output_ptr,
                                    n_indices, thread_cap, pd)
    elif itemsize == 2:
        with nogil:
            colstore_gather_bytes_2(base, offsets_ptr, output_ptr,
                                    n_indices, thread_cap, pd)
    elif itemsize == 4:
        with nogil:
            colstore_gather_bytes_4(base, offsets_ptr, output_ptr,
                                    n_indices, thread_cap, pd)
    elif itemsize == 8:
        with nogil:
            colstore_gather_bytes_8(base, offsets_ptr, output_ptr,
                                    n_indices, thread_cap, pd)
    else:
        raise TypeError(
            f"Unsupported element size: {itemsize} bytes. The C++ kernel "
            f"handles 1, 2, 4, and 8 byte elements."
        )



def gather_multirecord_bins(cnp.ndarray source, cnp.ndarray indices,
                            cnp.ndarray output, cnp.ndarray bins,
                            cnp.ndarray record_starts_rows,
                            cnp.ndarray record_starts_bytes,
                            cnp.ndarray n_rows_per_record,
                            long long col_prefix_bytes, int thread_cap=0,
                            Py_ssize_t prefetch_distance=-1):
    """Fused multi-record gather that also records each index's record bin.

    Identical addressing and output to :func:`gather_multirecord`, plus
    ``bins[i]`` is filled with the record index (``int32``) that
    ``indices[i]`` binned to. The binning -- a branchless binary search per
    element -- measures 87-93% of the fused kernel's cost on the target
    hardware and is identical for every column of a multi-column read, so a
    caller reading C columns computes it once here and reuses it C-1 times
    via :func:`gather_multirecord_withbins`.

    ``bins`` must be 1D ``int32`` of the same length as ``indices``; the
    caller guarantees ``n_records <= 2**31 - 1``. All other parameters and
    errors match :func:`gather_multirecord`.
    """
    if (indices.ndim != 1 or output.ndim != 1 or bins.ndim != 1
            or record_starts_rows.ndim != 1
            or record_starts_bytes.ndim != 1 or n_rows_per_record.ndim != 1):
        raise ValueError("indices, output, bins, and index arrays must be 1D.")
    if indices.dtype != np.int64:
        raise TypeError(f"indices must be int64; got {indices.dtype}.")
    if bins.dtype != np.int32:
        raise TypeError(f"bins must be int32; got {bins.dtype}.")
    if (record_starts_rows.dtype != np.int64
            or record_starts_bytes.dtype != np.int64
            or n_rows_per_record.dtype != np.int64):
        raise TypeError("record index arrays must be int64.")

    cdef ptrdiff_t n = indices.shape[0]
    if output.shape[0] != n or bins.shape[0] != n:
        raise ValueError(
            f"output/bins lengths ({output.shape[0]}/{bins.shape[0]}) must "
            f"match indices length {n}."
        )
    cdef long long n_records = record_starts_bytes.shape[0]
    if n_rows_per_record.shape[0] != n_records:
        raise ValueError("n_rows_per_record length must match record count.")
    if record_starts_rows.shape[0] != n_records + 1:
        raise ValueError("record_starts_rows length must be n_records + 1.")
    if n == 0:
        return

    cdef int itemsize = output.dtype.itemsize
    _require_c_contiguous((("source", source), ("indices", indices), ("output", output), ("bins", bins), ("record_starts_rows", record_starts_rows), ("record_starts_bytes", record_starts_bytes), ("n_rows_per_record", n_rows_per_record)))
    cdef const uint8_t* base = <const uint8_t*>cnp.PyArray_DATA(source)
    cdef const int64_t* indices_ptr = <const int64_t*>cnp.PyArray_DATA(indices)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef int32_t* bins_ptr = <int32_t*>cnp.PyArray_DATA(bins)
    cdef const int64_t* rsr = <const int64_t*>cnp.PyArray_DATA(record_starts_rows)
    cdef const int64_t* rsb = <const int64_t*>cnp.PyArray_DATA(record_starts_bytes)
    cdef const int64_t* nrr = <const int64_t*>cnp.PyArray_DATA(n_rows_per_record)

    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    if itemsize == 1:
        with nogil:
            colstore_gather_multirecord_bins_1(base, indices_ptr, output_ptr,
                                               bins_ptr, n, rsr, rsb, nrr,
                                               n_records, col_prefix_bytes,
                                               thread_cap, pd)
    elif itemsize == 2:
        with nogil:
            colstore_gather_multirecord_bins_2(base, indices_ptr, output_ptr,
                                               bins_ptr, n, rsr, rsb, nrr,
                                               n_records, col_prefix_bytes,
                                               thread_cap, pd)
    elif itemsize == 4:
        with nogil:
            colstore_gather_multirecord_bins_4(base, indices_ptr, output_ptr,
                                               bins_ptr, n, rsr, rsb, nrr,
                                               n_records, col_prefix_bytes,
                                               thread_cap, pd)
    elif itemsize == 8:
        with nogil:
            colstore_gather_multirecord_bins_8(base, indices_ptr, output_ptr,
                                               bins_ptr, n, rsr, rsb, nrr,
                                               n_records, col_prefix_bytes,
                                               thread_cap, pd)
    else:
        raise TypeError(f"unsupported itemsize {itemsize}.")


def gather_multirecord_withbins(cnp.ndarray source, cnp.ndarray indices,
                                cnp.ndarray output, cnp.ndarray bins,
                                cnp.ndarray record_starts_rows,
                                cnp.ndarray record_starts_bytes,
                                cnp.ndarray n_rows_per_record,
                                long long col_prefix_bytes, int thread_cap=0,
                                Py_ssize_t prefetch_distance=-1):
    """Multi-record gather using record bins from :func:`gather_multirecord_bins`.

    ``bins`` must be the array filled by :func:`gather_multirecord_bins` for
    the *same* ``indices`` against the same record layout; each element's
    record is then a sequential ``int32`` read instead of a binary search
    (including for the prefetch look-ahead). Output is identical to
    :func:`gather_multirecord` for the column selected by
    ``col_prefix_bytes``. Other parameters and errors match
    :func:`gather_multirecord`.
    """
    if (indices.ndim != 1 or output.ndim != 1 or bins.ndim != 1
            or record_starts_rows.ndim != 1
            or record_starts_bytes.ndim != 1 or n_rows_per_record.ndim != 1):
        raise ValueError("indices, output, bins, and index arrays must be 1D.")
    if indices.dtype != np.int64:
        raise TypeError(f"indices must be int64; got {indices.dtype}.")
    if bins.dtype != np.int32:
        raise TypeError(f"bins must be int32; got {bins.dtype}.")
    if (record_starts_rows.dtype != np.int64
            or record_starts_bytes.dtype != np.int64
            or n_rows_per_record.dtype != np.int64):
        raise TypeError("record index arrays must be int64.")

    cdef ptrdiff_t n = indices.shape[0]
    if output.shape[0] != n or bins.shape[0] != n:
        raise ValueError(
            f"output/bins lengths ({output.shape[0]}/{bins.shape[0]}) must "
            f"match indices length {n}."
        )
    cdef long long n_records = record_starts_bytes.shape[0]
    if n_rows_per_record.shape[0] != n_records:
        raise ValueError("n_rows_per_record length must match record count.")
    if record_starts_rows.shape[0] != n_records + 1:
        raise ValueError("record_starts_rows length must be n_records + 1.")
    if n == 0:
        return

    cdef int itemsize = output.dtype.itemsize
    _require_c_contiguous((("source", source), ("indices", indices), ("output", output), ("bins", bins), ("record_starts_rows", record_starts_rows), ("record_starts_bytes", record_starts_bytes), ("n_rows_per_record", n_rows_per_record)))
    cdef const uint8_t* base = <const uint8_t*>cnp.PyArray_DATA(source)
    cdef const int64_t* indices_ptr = <const int64_t*>cnp.PyArray_DATA(indices)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef const int32_t* bins_ptr = <const int32_t*>cnp.PyArray_DATA(bins)
    cdef const int64_t* rsr = <const int64_t*>cnp.PyArray_DATA(record_starts_rows)
    cdef const int64_t* rsb = <const int64_t*>cnp.PyArray_DATA(record_starts_bytes)
    cdef const int64_t* nrr = <const int64_t*>cnp.PyArray_DATA(n_rows_per_record)

    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    if itemsize == 1:
        with nogil:
            colstore_gather_multirecord_withbins_1(base, indices_ptr, output_ptr,
                                                   bins_ptr, n, rsr, rsb, nrr,
                                                   col_prefix_bytes, thread_cap, pd)
    elif itemsize == 2:
        with nogil:
            colstore_gather_multirecord_withbins_2(base, indices_ptr, output_ptr,
                                                   bins_ptr, n, rsr, rsb, nrr,
                                                   col_prefix_bytes, thread_cap, pd)
    elif itemsize == 4:
        with nogil:
            colstore_gather_multirecord_withbins_4(base, indices_ptr, output_ptr,
                                                   bins_ptr, n, rsr, rsb, nrr,
                                                   col_prefix_bytes, thread_cap, pd)
    elif itemsize == 8:
        with nogil:
            colstore_gather_multirecord_withbins_8(base, indices_ptr, output_ptr,
                                                   bins_ptr, n, rsr, rsb, nrr,
                                                   col_prefix_bytes, thread_cap, pd)
    else:
        raise TypeError(f"unsupported itemsize {itemsize}.")



def gather_multirecord_sorted(cnp.ndarray source, cnp.ndarray indices,
                              cnp.ndarray output,
                              cnp.ndarray record_starts_rows,
                              cnp.ndarray record_starts_bytes,
                              cnp.ndarray n_rows_per_record,
                              long long col_prefix_bytes, int thread_cap=0,
                              Py_ssize_t prefetch_distance=-1):
    """Sorted multi-record fancy gather via a linear record walk.

    ``indices`` MUST be non-decreasing; the caller is responsible for the
    check (the reader's sortedness test gates the route) and behavior is
    undefined otherwise. Each thread binary-searches the record of the first
    index in its chunk, then advances the record cursor monotonically --
    O(K + R) total, no ``byte_offsets`` array, no per-record host loop.
    Replaces the NumPy boundary-partition pipeline for the sorted
    multi-record path. Parameters and errors otherwise match
    :func:`gather_multirecord`.
    """
    if (indices.ndim != 1 or output.ndim != 1 or record_starts_rows.ndim != 1
            or record_starts_bytes.ndim != 1 or n_rows_per_record.ndim != 1):
        raise ValueError("indices, output, and index arrays must be 1D.")
    if indices.dtype != np.int64:
        raise TypeError(f"indices must be int64; got {indices.dtype}.")
    if (record_starts_rows.dtype != np.int64
            or record_starts_bytes.dtype != np.int64
            or n_rows_per_record.dtype != np.int64):
        raise TypeError("record index arrays must be int64.")

    cdef ptrdiff_t n = indices.shape[0]
    if output.shape[0] != n:
        raise ValueError(
            f"output length {output.shape[0]} does not match indices length {n}."
        )
    cdef long long n_records = record_starts_bytes.shape[0]
    if n_rows_per_record.shape[0] != n_records:
        raise ValueError("n_rows_per_record length must match record count.")
    if record_starts_rows.shape[0] != n_records + 1:
        raise ValueError("record_starts_rows length must be n_records + 1.")
    if n == 0:
        return

    cdef int itemsize = output.dtype.itemsize
    _require_c_contiguous((("source", source), ("indices", indices), ("output", output), ("record_starts_rows", record_starts_rows), ("record_starts_bytes", record_starts_bytes), ("n_rows_per_record", n_rows_per_record)))
    cdef const uint8_t* base = <const uint8_t*>cnp.PyArray_DATA(source)
    cdef const int64_t* indices_ptr = <const int64_t*>cnp.PyArray_DATA(indices)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef const int64_t* rsr = <const int64_t*>cnp.PyArray_DATA(record_starts_rows)
    cdef const int64_t* rsb = <const int64_t*>cnp.PyArray_DATA(record_starts_bytes)
    cdef const int64_t* nrr = <const int64_t*>cnp.PyArray_DATA(n_rows_per_record)

    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    if itemsize == 1:
        with nogil:
            colstore_gather_multirecord_sorted_1(base, indices_ptr, output_ptr, n,
                                                 rsr, rsb, nrr, n_records,
                                                 col_prefix_bytes, thread_cap, pd)
    elif itemsize == 2:
        with nogil:
            colstore_gather_multirecord_sorted_2(base, indices_ptr, output_ptr, n,
                                                 rsr, rsb, nrr, n_records,
                                                 col_prefix_bytes, thread_cap, pd)
    elif itemsize == 4:
        with nogil:
            colstore_gather_multirecord_sorted_4(base, indices_ptr, output_ptr, n,
                                                 rsr, rsb, nrr, n_records,
                                                 col_prefix_bytes, thread_cap, pd)
    elif itemsize == 8:
        with nogil:
            colstore_gather_multirecord_sorted_8(base, indices_ptr, output_ptr, n,
                                                 rsr, rsb, nrr, n_records,
                                                 col_prefix_bytes, thread_cap, pd)
    else:
        raise TypeError(f"unsupported itemsize {itemsize}.")


def gather_multirecord_strided(cnp.ndarray source, cnp.ndarray output,
                               long long start, long long stop, long long step,
                               cnp.ndarray record_starts_rows,
                               cnp.ndarray record_starts_bytes,
                               cnp.ndarray n_rows_per_record,
                               long long col_prefix_bytes, int thread_cap=0,
                               Py_ssize_t prefetch_distance=-1):
    """Strided multi-record range gather: rows ``start, start+step, ...``.

    Reads the rows of Python ``slice(start, stop, step)`` semantics for one
    column without ever materializing an index array -- the row stream is
    synthesized arithmetically inside the kernel, which walks the record
    cursor monotonically (forward for ``step > 0``, backward for
    ``step < 0``). ``step`` must be non-zero and ``output`` must have exactly
    ``len(range(start, stop, step))`` elements. The caller guarantees every
    visited row is in range (the reader derives the triple from
    ``slice.indices``, which clamps) and that the dtype is in native byte
    order (raw typed loads cannot byteswap). Other parameters and errors
    match :func:`gather_multirecord`.
    """
    if (output.ndim != 1 or record_starts_rows.ndim != 1
            or record_starts_bytes.ndim != 1 or n_rows_per_record.ndim != 1):
        raise ValueError("output and index arrays must be 1D.")
    if step == 0:
        raise ValueError("step must be non-zero.")
    if (record_starts_rows.dtype != np.int64
            or record_starts_bytes.dtype != np.int64
            or n_rows_per_record.dtype != np.int64):
        raise TypeError("record index arrays must be int64.")

    cdef ptrdiff_t n = len(range(start, stop, step))
    if output.shape[0] != n:
        raise ValueError(
            f"output length {output.shape[0]} does not match the slice length {n}."
        )
    cdef long long n_records = record_starts_bytes.shape[0]
    if n_rows_per_record.shape[0] != n_records:
        raise ValueError("n_rows_per_record length must match record count.")
    if record_starts_rows.shape[0] != n_records + 1:
        raise ValueError("record_starts_rows length must be n_records + 1.")
    if n == 0:
        return

    cdef int itemsize = output.dtype.itemsize
    _require_c_contiguous((("source", source), ("output", output), ("record_starts_rows", record_starts_rows), ("record_starts_bytes", record_starts_bytes), ("n_rows_per_record", n_rows_per_record)))
    cdef const uint8_t* base = <const uint8_t*>cnp.PyArray_DATA(source)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef const int64_t* rsr = <const int64_t*>cnp.PyArray_DATA(record_starts_rows)
    cdef const int64_t* rsb = <const int64_t*>cnp.PyArray_DATA(record_starts_bytes)
    cdef const int64_t* nrr = <const int64_t*>cnp.PyArray_DATA(n_rows_per_record)

    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    if itemsize == 1:
        with nogil:
            colstore_gather_multirecord_strided_1(base, output_ptr, start, step, n,
                                                  rsr, rsb, nrr, n_records,
                                                  col_prefix_bytes, thread_cap, pd)
    elif itemsize == 2:
        with nogil:
            colstore_gather_multirecord_strided_2(base, output_ptr, start, step, n,
                                                  rsr, rsb, nrr, n_records,
                                                  col_prefix_bytes, thread_cap, pd)
    elif itemsize == 4:
        with nogil:
            colstore_gather_multirecord_strided_4(base, output_ptr, start, step, n,
                                                  rsr, rsb, nrr, n_records,
                                                  col_prefix_bytes, thread_cap, pd)
    elif itemsize == 8:
        with nogil:
            colstore_gather_multirecord_strided_8(base, output_ptr, start, step, n,
                                                  rsr, rsb, nrr, n_records,
                                                  col_prefix_bytes, thread_cap, pd)
    else:
        raise TypeError(f"unsupported itemsize {itemsize}.")


def gather_multirecord_uniform(cnp.ndarray source, cnp.ndarray indices,
                               cnp.ndarray output,
                               long long rows_per_record,
                               long long record_stride_bytes,
                               long long first_body_offset,
                               long long n_records,
                               long long last_record_rows,
                               long long col_prefix_bytes, int thread_cap=0,
                               Py_ssize_t prefetch_distance=-1):
    """Unsorted fancy gather over a uniform-record file: arithmetic binning.

    Specialization of :func:`gather_multirecord` for files whose records all
    have ``rows_per_record`` rows (the final record may be partial, with
    ``last_record_rows`` rows) and whose record bodies sit at a constant
    byte stride. The record bin is one integer division and the address one
    affine formula -- no binary search, no per-record metadata loads. The
    caller detects the layout and guarantees its invariants (see the C++
    header) plus native byte order; this entry validates only the scalar
    sanity conditions and the array contracts shared by every kernel.
    """
    if indices.ndim != 1 or output.ndim != 1:
        raise ValueError("indices and output must be 1D.")
    if indices.dtype != np.int64:
        raise TypeError(f"indices must be int64; got {indices.dtype}.")
    if rows_per_record <= 0:
        raise ValueError("rows_per_record must be positive.")
    if n_records <= 0:
        raise ValueError("n_records must be positive.")
    if last_record_rows <= 0 or last_record_rows > rows_per_record:
        raise ValueError("last_record_rows must be in [1, rows_per_record].")

    cdef ptrdiff_t n = indices.shape[0]
    if output.shape[0] != n:
        raise ValueError(
            f"output length {output.shape[0]} does not match indices length {n}."
        )
    if n == 0:
        return

    cdef int itemsize = output.dtype.itemsize
    _require_c_contiguous((("source", source), ("indices", indices), ("output", output)))
    cdef const uint8_t* base = <const uint8_t*>cnp.PyArray_DATA(source)
    cdef const int64_t* indices_ptr = <const int64_t*>cnp.PyArray_DATA(indices)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)

    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    if itemsize == 1:
        with nogil:
            colstore_gather_multirecord_uniform_1(base, indices_ptr, output_ptr, n,
                                                  rows_per_record, record_stride_bytes,
                                                  first_body_offset, n_records,
                                                  last_record_rows, col_prefix_bytes,
                                                  thread_cap, pd)
    elif itemsize == 2:
        with nogil:
            colstore_gather_multirecord_uniform_2(base, indices_ptr, output_ptr, n,
                                                  rows_per_record, record_stride_bytes,
                                                  first_body_offset, n_records,
                                                  last_record_rows, col_prefix_bytes,
                                                  thread_cap, pd)
    elif itemsize == 4:
        with nogil:
            colstore_gather_multirecord_uniform_4(base, indices_ptr, output_ptr, n,
                                                  rows_per_record, record_stride_bytes,
                                                  first_body_offset, n_records,
                                                  last_record_rows, col_prefix_bytes,
                                                  thread_cap, pd)
    elif itemsize == 8:
        with nogil:
            colstore_gather_multirecord_uniform_8(base, indices_ptr, output_ptr, n,
                                                  rows_per_record, record_stride_bytes,
                                                  first_body_offset, n_records,
                                                  last_record_rows, col_prefix_bytes,
                                                  thread_cap, pd)
    else:
        raise TypeError(f"unsupported itemsize {itemsize}.")


def gather_multirecord_uniform_bins(cnp.ndarray source, cnp.ndarray indices,
                                    cnp.ndarray output, cnp.ndarray bins,
                                    long long rows_per_record,
                                    long long record_stride_bytes,
                                    long long first_body_offset,
                                    long long n_records,
                                    long long last_record_rows,
                                    long long col_prefix_bytes, int thread_cap=0,
                                    Py_ssize_t prefetch_distance=-1):
    """Uniform-record gather that also records each index's record bin.

    First-column kernel of the uniform multi-column route: the record is
    computed arithmetically (one division, partial-tail guard) and written
    to ``bins`` (int32) so subsequent columns can read it instead of
    re-dividing. Requires ``n_records <= INT32_MAX`` (the caller guards).
    Layout invariants and other parameters match
    :func:`gather_multirecord_uniform`.
    """
    if indices.ndim != 1 or output.ndim != 1 or bins.ndim != 1:
        raise ValueError("indices, output, and bins must be 1D.")
    if indices.dtype != np.int64:
        raise TypeError(f"indices must be int64; got {indices.dtype}.")
    if bins.dtype != np.int32:
        raise TypeError(f"bins must be int32; got {bins.dtype}.")
    if rows_per_record <= 0:
        raise ValueError("rows_per_record must be positive.")
    if n_records <= 0:
        raise ValueError("n_records must be positive.")
    if last_record_rows <= 0 or last_record_rows > rows_per_record:
        raise ValueError("last_record_rows must be in [1, rows_per_record].")

    cdef ptrdiff_t n = indices.shape[0]
    if output.shape[0] != n or bins.shape[0] != n:
        raise ValueError("output and bins lengths must match indices length.")
    if n == 0:
        return

    cdef int itemsize = output.dtype.itemsize
    _require_c_contiguous((("source", source), ("indices", indices), ("output", output), ("bins", bins)))
    cdef const uint8_t* base = <const uint8_t*>cnp.PyArray_DATA(source)
    cdef const int64_t* indices_ptr = <const int64_t*>cnp.PyArray_DATA(indices)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef int32_t* bins_ptr = <int32_t*>cnp.PyArray_DATA(bins)

    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    if itemsize == 1:
        with nogil:
            colstore_gather_multirecord_uniform_bins_1(
                base, indices_ptr, output_ptr, bins_ptr, n, rows_per_record,
                record_stride_bytes, first_body_offset, n_records,
                last_record_rows, col_prefix_bytes, thread_cap, pd)
    elif itemsize == 2:
        with nogil:
            colstore_gather_multirecord_uniform_bins_2(
                base, indices_ptr, output_ptr, bins_ptr, n, rows_per_record,
                record_stride_bytes, first_body_offset, n_records,
                last_record_rows, col_prefix_bytes, thread_cap, pd)
    elif itemsize == 4:
        with nogil:
            colstore_gather_multirecord_uniform_bins_4(
                base, indices_ptr, output_ptr, bins_ptr, n, rows_per_record,
                record_stride_bytes, first_body_offset, n_records,
                last_record_rows, col_prefix_bytes, thread_cap, pd)
    elif itemsize == 8:
        with nogil:
            colstore_gather_multirecord_uniform_bins_8(
                base, indices_ptr, output_ptr, bins_ptr, n, rows_per_record,
                record_stride_bytes, first_body_offset, n_records,
                last_record_rows, col_prefix_bytes, thread_cap, pd)
    else:
        raise TypeError(f"unsupported itemsize {itemsize}.")


def gather_multirecord_uniform_withbins(cnp.ndarray source, cnp.ndarray indices,
                                        cnp.ndarray output, cnp.ndarray bins,
                                        long long rows_per_record,
                                        long long record_stride_bytes,
                                        long long first_body_offset,
                                        long long n_records,
                                        long long last_record_rows,
                                        long long col_prefix_bytes, int thread_cap=0,
                                        Py_ssize_t prefetch_distance=-1):
    """Uniform-record gather using bins from :func:`gather_multirecord_uniform_bins`.

    Per element: a sequential int32 bin read, the partial-tail compare, one
    multiply-add, and the load -- no division, no search, no per-record
    metadata. ``bins`` must come from the bins kernel run on the same
    ``indices`` and layout; behavior is undefined otherwise. Other
    parameters and errors match :func:`gather_multirecord_uniform_bins`.
    """
    if indices.ndim != 1 or output.ndim != 1 or bins.ndim != 1:
        raise ValueError("indices, output, and bins must be 1D.")
    if indices.dtype != np.int64:
        raise TypeError(f"indices must be int64; got {indices.dtype}.")
    if bins.dtype != np.int32:
        raise TypeError(f"bins must be int32; got {bins.dtype}.")
    if rows_per_record <= 0:
        raise ValueError("rows_per_record must be positive.")
    if n_records <= 0:
        raise ValueError("n_records must be positive.")
    if last_record_rows <= 0 or last_record_rows > rows_per_record:
        raise ValueError("last_record_rows must be in [1, rows_per_record].")

    cdef ptrdiff_t n = indices.shape[0]
    if output.shape[0] != n or bins.shape[0] != n:
        raise ValueError("output and bins lengths must match indices length.")
    if n == 0:
        return

    cdef int itemsize = output.dtype.itemsize
    _require_c_contiguous((("source", source), ("indices", indices), ("output", output), ("bins", bins)))
    cdef const uint8_t* base = <const uint8_t*>cnp.PyArray_DATA(source)
    cdef const int64_t* indices_ptr = <const int64_t*>cnp.PyArray_DATA(indices)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef const int32_t* bins_ptr = <const int32_t*>cnp.PyArray_DATA(bins)

    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    if itemsize == 1:
        with nogil:
            colstore_gather_multirecord_uniform_withbins_1(
                base, indices_ptr, output_ptr, bins_ptr, n, rows_per_record,
                record_stride_bytes, first_body_offset, n_records,
                last_record_rows, col_prefix_bytes, thread_cap, pd)
    elif itemsize == 2:
        with nogil:
            colstore_gather_multirecord_uniform_withbins_2(
                base, indices_ptr, output_ptr, bins_ptr, n, rows_per_record,
                record_stride_bytes, first_body_offset, n_records,
                last_record_rows, col_prefix_bytes, thread_cap, pd)
    elif itemsize == 4:
        with nogil:
            colstore_gather_multirecord_uniform_withbins_4(
                base, indices_ptr, output_ptr, bins_ptr, n, rows_per_record,
                record_stride_bytes, first_body_offset, n_records,
                last_record_rows, col_prefix_bytes, thread_cap, pd)
    elif itemsize == 8:
        with nogil:
            colstore_gather_multirecord_uniform_withbins_8(
                base, indices_ptr, output_ptr, bins_ptr, n, rows_per_record,
                record_stride_bytes, first_body_offset, n_records,
                last_record_rows, col_prefix_bytes, thread_cap, pd)
    else:
        raise TypeError(f"unsupported itemsize {itemsize}.")


def gather_multirecord_withbins_rbase(cnp.ndarray source, cnp.ndarray indices,
                                      cnp.ndarray output, cnp.ndarray bins,
                                      cnp.ndarray record_base, int thread_cap=0,
                                      Py_ssize_t prefetch_distance=-1):
    """Bins-fed gather with per-record byte bases precomputed by the caller.

    ``record_base[r] = record_starts_bytes[r] + col_prefix * n_rows_per_record[r]
    - record_starts_rows[r] * itemsize`` for THIS column's prefix and
    itemsize, one entry per record; the per-element address is then
    ``record_base[bins[i]] + indices[i] * itemsize`` -- one metadata load
    instead of the three the generic withbins kernel pays. ``bins`` must
    come from :func:`gather_multirecord_bins` (or the uniform bins kernel)
    on the same ``indices``; record membership and base correctness are the
    caller's contract and are not re-validated. Other parameters and errors
    match :func:`gather_multirecord_withbins`.
    """
    if indices.ndim != 1 or output.ndim != 1 or bins.ndim != 1 or record_base.ndim != 1:
        raise ValueError("indices, output, bins, and record_base must be 1D.")
    if indices.dtype != np.int64:
        raise TypeError(f"indices must be int64; got {indices.dtype}.")
    if bins.dtype != np.int32:
        raise TypeError(f"bins must be int32; got {bins.dtype}.")
    if record_base.dtype != np.int64:
        raise TypeError(f"record_base must be int64; got {record_base.dtype}.")

    cdef ptrdiff_t n = indices.shape[0]
    if output.shape[0] != n or bins.shape[0] != n:
        raise ValueError("output and bins lengths must match indices length.")
    if record_base.shape[0] <= 0:
        raise ValueError("record_base must be non-empty.")
    if n == 0:
        return

    cdef int itemsize = output.dtype.itemsize
    _require_c_contiguous((("source", source), ("indices", indices), ("output", output), ("bins", bins), ("record_base", record_base)))
    cdef const uint8_t* base = <const uint8_t*>cnp.PyArray_DATA(source)
    cdef const int64_t* indices_ptr = <const int64_t*>cnp.PyArray_DATA(indices)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef const int32_t* bins_ptr = <const int32_t*>cnp.PyArray_DATA(bins)
    cdef const int64_t* rbase = <const int64_t*>cnp.PyArray_DATA(record_base)

    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    if itemsize == 1:
        with nogil:
            colstore_gather_multirecord_withbins_rbase_1(base, indices_ptr, output_ptr,
                                                         bins_ptr, n, rbase, thread_cap, pd)
    elif itemsize == 2:
        with nogil:
            colstore_gather_multirecord_withbins_rbase_2(base, indices_ptr, output_ptr,
                                                         bins_ptr, n, rbase, thread_cap, pd)
    elif itemsize == 4:
        with nogil:
            colstore_gather_multirecord_withbins_rbase_4(base, indices_ptr, output_ptr,
                                                         bins_ptr, n, rbase, thread_cap, pd)
    elif itemsize == 8:
        with nogil:
            colstore_gather_multirecord_withbins_rbase_8(base, indices_ptr, output_ptr,
                                                         bins_ptr, n, rbase, thread_cap, pd)
    else:
        raise TypeError(f"unsupported itemsize {itemsize}.")


def gather_multirecord_mask(cnp.ndarray source, cnp.ndarray mask,
                            cnp.ndarray output,
                            cnp.ndarray record_starts_rows,
                            cnp.ndarray record_starts_bytes,
                            cnp.ndarray n_rows_per_record,
                            long long col_prefix_bytes, int thread_cap=0,
                            Py_ssize_t prefetch_distance=-1):
    """Boolean-mask-native multi-record gather: no index materialization.

    ``mask`` is a numpy bool array with one entry per row of the store
    (length must equal ``record_starts_rows[-1]``); selected rows are
    gathered in row order. ``output`` must be sized to exactly
    ``np.count_nonzero(mask)`` -- the kernel re-counts internally to fix
    per-thread offsets and raises if the caller's size disagrees, writing
    nothing in that case. Native byte order required; other parameters and
    errors match :func:`gather_multirecord_sorted`.
    """
    if mask.ndim != 1 or output.ndim != 1:
        raise ValueError("mask and output must be 1D.")
    if mask.dtype != np.bool_:
        raise TypeError(f"mask must be bool; got {mask.dtype}.")
    if (record_starts_rows.dtype != np.int64
            or record_starts_bytes.dtype != np.int64
            or n_rows_per_record.dtype != np.int64):
        raise TypeError("record index arrays must be int64.")
    cdef long long n_records = record_starts_bytes.shape[0]
    if n_rows_per_record.shape[0] != n_records:
        raise ValueError("n_rows_per_record length must match record count.")
    if record_starts_rows.shape[0] != n_records + 1:
        raise ValueError("record_starts_rows length must be n_records + 1.")

    _require_c_contiguous((("source", source), ("mask", mask), ("output", output), ("record_starts_rows", record_starts_rows), ("record_starts_bytes", record_starts_bytes), ("n_rows_per_record", n_rows_per_record)))
    cdef const int64_t* rsr = <const int64_t*>cnp.PyArray_DATA(record_starts_rows)
    cdef const int64_t* rsb = <const int64_t*>cnp.PyArray_DATA(record_starts_bytes)
    cdef const int64_t* nrr = <const int64_t*>cnp.PyArray_DATA(n_rows_per_record)
    cdef int64_t n_rows = rsr[n_records]
    if mask.shape[0] != n_rows:
        raise ValueError(
            f"mask length {mask.shape[0]} does not match the store's row count {n_rows}."
        )
    cdef ptrdiff_t n_out = output.shape[0]
    if n_rows == 0 or n_out == 0:
        # Nothing to write; still verify the contract for nonzero masks.
        if n_out == 0 and mask.shape[0] != 0 and bool(np.any(mask)):
            raise ValueError("output length does not match the mask's selected count.")
        return

    cdef int itemsize = output.dtype.itemsize
    cdef const uint8_t* base = <const uint8_t*>cnp.PyArray_DATA(source)
    cdef const uint8_t* mask_ptr = <const uint8_t*>cnp.PyArray_DATA(mask)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    cdef int status = -1
    if itemsize == 1:
        with nogil:
            status = colstore_gather_multirecord_mask_1(base, mask_ptr, output_ptr,
                                                        n_rows, n_out, rsr, rsb, nrr,
                                                        n_records, col_prefix_bytes,
                                                        thread_cap, pd)
    elif itemsize == 2:
        with nogil:
            status = colstore_gather_multirecord_mask_2(base, mask_ptr, output_ptr,
                                                        n_rows, n_out, rsr, rsb, nrr,
                                                        n_records, col_prefix_bytes,
                                                        thread_cap, pd)
    elif itemsize == 4:
        with nogil:
            status = colstore_gather_multirecord_mask_4(base, mask_ptr, output_ptr,
                                                        n_rows, n_out, rsr, rsb, nrr,
                                                        n_records, col_prefix_bytes,
                                                        thread_cap, pd)
    elif itemsize == 8:
        with nogil:
            status = colstore_gather_multirecord_mask_8(base, mask_ptr, output_ptr,
                                                        n_rows, n_out, rsr, rsb, nrr,
                                                        n_records, col_prefix_bytes,
                                                        thread_cap, pd)
    else:
        raise TypeError(f"unsupported itemsize {itemsize}.")
    if status != 0:
        raise ValueError("output length does not match the mask's selected count.")


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

    _require_c_contiguous((("source", source), ("output", output), ("record_starts_rows", record_starts_rows), ("record_starts_bytes", record_starts_bytes), ("n_rows_per_record", n_rows_per_record)))
    cdef const uint8_t* base = <const uint8_t*>cnp.PyArray_DATA(source)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef const int64_t* rsr = <const int64_t*>cnp.PyArray_DATA(record_starts_rows)
    cdef const int64_t* rsb = <const int64_t*>cnp.PyArray_DATA(record_starts_bytes)
    cdef const int64_t* nrr = <const int64_t*>cnp.PyArray_DATA(n_rows_per_record)

    with nogil:
        colstore_copy_multirecord_range(base, output_ptr, start, stop,
                                        rsr, rsb, nrr, n_records,
                                        col_prefix_bytes, itemsize)


def gather_multirecord(cnp.ndarray source, cnp.ndarray indices,
                       cnp.ndarray output,
                       cnp.ndarray record_starts_rows,
                       cnp.ndarray record_starts_bytes,
                       cnp.ndarray n_rows_per_record,
                       long long col_prefix_bytes, int thread_cap=0,
                       Py_ssize_t prefetch_distance=-1):
    """Fused multi-record fancy gather: ``output[i] = column_value(indices[i])``.

    For each (arbitrary, unsorted) index the record is located by a branchless
    binary search over ``record_starts_rows`` inside the kernel, the byte
    address is computed in registers, and the element is loaded -- one pass, no
    ``byte_offsets`` array, no per-index NumPy temporaries. Replaces the NumPy
    searchsorted pipeline for the unsorted multi-record path.

    Parameters
    ----------
    source : numpy.ndarray
        Whole-file ``uint8`` mmap; the raw byte base pointer.
    indices : numpy.ndarray
        1D ``int64`` element indices into the logical (global) row space. Each
        must lie in ``[0, total_rows)``; the caller (view layer) guarantees
        this. Need not be sorted.
    output : numpy.ndarray
        1D destination of length ``len(indices)`` and the column's dtype, in
        *native* byte order (the kernel does a raw typed load and cannot
        byteswap). Filled in-place.
    record_starts_rows : numpy.ndarray
        1D ``int64`` cumulative row counts, length ``n_records + 1``.
    record_starts_bytes : numpy.ndarray
        1D ``int64`` per-record body byte offsets, length ``n_records``.
    n_rows_per_record : numpy.ndarray
        1D ``int64`` per-record row counts, length ``n_records``.
    col_prefix_bytes : int
        Summed itemsize of the columns preceding this one in a record body.
    thread_cap : int, optional
        Maximum OpenMP threads; ``0`` means the OpenMP maximum. The kernel
        runs serially below its internal parallel threshold.
    prefetch_distance : int, optional
        Software-prefetch look-ahead in elements. ``> 0`` prefetches that
        many iterations ahead; ``0`` disables prefetching (useful when the
        source is cache-resident); negative (default) uses the compiled
        default, :func:`default_prefetch_distance`.

    Raises
    ------
    TypeError
        If ``indices`` or any index array is not ``int64``, or the element
        size is unsupported.
    ValueError
        If shapes are 1D-inconsistent or lengths disagree.
    """
    if (indices.ndim != 1 or output.ndim != 1 or record_starts_rows.ndim != 1
            or record_starts_bytes.ndim != 1 or n_rows_per_record.ndim != 1):
        raise ValueError("indices, output, and index arrays must be 1D.")
    if indices.dtype != np.int64:
        raise TypeError(f"indices must be int64; got {indices.dtype}.")
    if (record_starts_rows.dtype != np.int64
            or record_starts_bytes.dtype != np.int64
            or n_rows_per_record.dtype != np.int64):
        raise TypeError("record index arrays must be int64.")

    cdef ptrdiff_t n = indices.shape[0]
    if output.shape[0] != n:
        raise ValueError(
            f"output length {output.shape[0]} does not match indices length {n}."
        )
    cdef long long n_records = record_starts_bytes.shape[0]
    if n_rows_per_record.shape[0] != n_records:
        raise ValueError("n_rows_per_record length must match record count.")
    if record_starts_rows.shape[0] != n_records + 1:
        raise ValueError("record_starts_rows length must be n_records + 1.")
    if n == 0:
        return

    cdef int itemsize = output.dtype.itemsize
    _require_c_contiguous((("source", source), ("indices", indices), ("output", output), ("record_starts_rows", record_starts_rows), ("record_starts_bytes", record_starts_bytes), ("n_rows_per_record", n_rows_per_record)))
    cdef const uint8_t* base = <const uint8_t*>cnp.PyArray_DATA(source)
    cdef const int64_t* indices_ptr = <const int64_t*>cnp.PyArray_DATA(indices)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef const int64_t* rsr = <const int64_t*>cnp.PyArray_DATA(record_starts_rows)
    cdef const int64_t* rsb = <const int64_t*>cnp.PyArray_DATA(record_starts_bytes)
    cdef const int64_t* nrr = <const int64_t*>cnp.PyArray_DATA(n_rows_per_record)

    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    if itemsize == 1:
        with nogil:
            colstore_gather_multirecord_1(base, indices_ptr, output_ptr, n,
                                          rsr, rsb, nrr, n_records,
                                          col_prefix_bytes, thread_cap, pd)
    elif itemsize == 2:
        with nogil:
            colstore_gather_multirecord_2(base, indices_ptr, output_ptr, n,
                                          rsr, rsb, nrr, n_records,
                                          col_prefix_bytes, thread_cap, pd)
    elif itemsize == 4:
        with nogil:
            colstore_gather_multirecord_4(base, indices_ptr, output_ptr, n,
                                          rsr, rsb, nrr, n_records,
                                          col_prefix_bytes, thread_cap, pd)
    elif itemsize == 8:
        with nogil:
            colstore_gather_multirecord_8(base, indices_ptr, output_ptr, n,
                                          rsr, rsb, nrr, n_records,
                                          col_prefix_bytes, thread_cap, pd)
    else:
        raise TypeError(
            f"Unsupported element size: {itemsize} bytes. The C++ kernel "
            f"handles 1, 2, 4, and 8 byte elements."
        )
