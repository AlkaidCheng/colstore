# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
# distutils: language = c++
"""Cython bindings for the size-dispatched C++ gather kernels.

The C++ kernels are templated on element size (1/2/4/8 bytes), not on NumPy
dtype kind: one set of four instantiations per kernel covers every
fixed-width numeric dtype, plus fixed-width bytes/unicode strings,
datetime64, and timedelta64 -- anything whose itemsize is one of those four
sizes.

Each ``def`` below wraps one kernel: the single-record pair
(:func:`gather` / :func:`gather_bytes`), the multi-record range, strided,
sorted, and unsorted kernels, the bin-reuse and uniform-layout families,
and the boolean-mask kernel. Docstrings state the caller contract;
kernel design lives in ``include/colstore/gather.hpp`` and the
measurements behind each kernel in ``docs/optimization_series.md``.

Contracts shared by every entry point:

* Arrays handed to the kernels as pointers must be C-contiguous
  (validated here; the view layer normalizes fancy selectors with
  ``np.ascontiguousarray``).
* Output buffers are caller-allocated and filled in-place; no entry
  point allocates.
* ``thread_cap`` (int, default 0): maximum OpenMP threads; ``0`` means
  the OpenMP maximum. Kernels run serially below their internal
  parallel threshold.
* ``prefetch_distance`` (int, default -1): software-prefetch look-ahead
  in elements. ``> 0`` prefetches that many iterations ahead; ``0``
  disables prefetching (useful when the source is cache-resident);
  negative uses the compiled default, :func:`default_prefetch_distance`.
"""

import os

import numpy as np

cimport numpy as cnp
from libc.stdint cimport int32_t, int64_t, uint8_t, uint32_t

cnp.import_array()


cdef extern from "colstore/gather.hpp" nogil:
    const char* colstore_build_flags()
    long long colstore_resolve_thread_count(long long, int)
    int colstore_gather_indexed(const uint8_t*, const int64_t*, uint8_t*,
                                   ptrdiff_t, int, int, ptrdiff_t)
    int colstore_gather_bytes(const uint8_t*, const int64_t*, uint8_t*,
                                 ptrdiff_t, int, int, ptrdiff_t)
    void colstore_copy_multirecord_range(const uint8_t*, uint8_t*,
                                         int64_t, int64_t,
                                         const int64_t*, const int64_t*,
                                         const int64_t*, int64_t,
                                         int64_t, int64_t)
    int colstore_gather_segment(const int64_t*, uint8_t*, ptrdiff_t,
                                  const int64_t*, const int64_t*, int64_t,
                                  int, int, ptrdiff_t)
    int colstore_gather_segment_bins(const int64_t*, uint8_t*, int32_t*, ptrdiff_t,
                                       const int64_t*, const int64_t*, int64_t,
                                       int, int, ptrdiff_t)
    int colstore_gather_segment_withbins(const int64_t*, uint8_t*, const int32_t*, ptrdiff_t,
                                           const int64_t*, int, int, ptrdiff_t)
    int colstore_gather_segment_sorted(const int64_t*, uint8_t*, ptrdiff_t,
                                         const int64_t*, const int64_t*, int64_t,
                                         int, int, ptrdiff_t)
    int colstore_gather_segment_uniform(const int64_t*, uint8_t*, ptrdiff_t,
                                          int64_t, const int64_t*, int64_t,
                                          int, int, ptrdiff_t)
    int colstore_gather_segment_uniform_bins(const int64_t*, uint8_t*, int32_t*, ptrdiff_t,
                                               int64_t, const int64_t*, int64_t,
                                               int, int, ptrdiff_t)
    int colstore_gather_segment_mask(const uint8_t*, uint8_t*, int64_t, ptrdiff_t,
                                       const int64_t*, const int64_t*, int64_t,
                                       int, int, ptrdiff_t)
    void colstore_parallel_copy_runs(uint8_t*, const int64_t*, const int64_t*,
                                     const int64_t*, int64_t, int)
    void colstore_interleave_records(uint8_t*, int64_t, int64_t, const int64_t*,
                                     const int64_t*, const int64_t*, int64_t, int)
    int colstore_gather_multirecord_strided(
        const uint8_t*, uint8_t*, int64_t, int64_t, ptrdiff_t,
        const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t,
        int, int, ptrdiff_t)
    int colstore_gather_multirecord_uniform(
        const uint8_t*, const int64_t*, uint8_t*, ptrdiff_t,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int, int, ptrdiff_t)
    int colstore_gather_multirecord_uniform_bins(
        const uint8_t*, const int64_t*, uint8_t*, int32_t*, ptrdiff_t,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int, int, ptrdiff_t)
    int colstore_gather_multirecord_uniform_withbins(
        const uint8_t*, const int64_t*, uint8_t*, const int32_t*, ptrdiff_t,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int, int, ptrdiff_t)
    int colstore_max_threads()
    int colstore_bind_threads_to_cpus(const int*, int)


cdef extern from "colstore/gather.hpp" namespace "colstore" nogil:
    ptrdiff_t resolve_thread_count(ptrdiff_t n_indices, int cap)
    const ptrdiff_t DEFAULT_PREFETCH_DISTANCE


cdef extern from "colstore/record_index.hpp" nogil:
    int colstore_read_record_index(const char*, int64_t, int64_t, int64_t, int64_t,
                                   int64_t*, int64_t*, int64_t*,
                                   int64_t*, int64_t*, int64_t*,
                                   uint32_t*, uint32_t*)


def max_threads() -> int:
    """Return OpenMP's max thread count (or 1 if OpenMP is disabled)."""
    return colstore_max_threads()


def bind_threads_to_cpus(cnp.ndarray cpus) -> int:
    """Pin libgomp's worker pool: worker ``t`` -> ``cpus[t]`` (one CPU each).

    ``cpus`` is a 1D array of CPU ids (cast to C ``int``); its length is the
    number of workers to pin. A negative entry leaves that worker unpinned.
    Returns the number of workers pinned, 0 for an empty request, or -1 where
    unsupported (non-Linux, or built without OpenMP). Best-effort: a worker
    whose ``sched_setaffinity`` fails is simply not counted.
    """
    if cpus.ndim != 1:
        raise ValueError("cpus must be a 1D array.")
    cdef cnp.ndarray c = np.ascontiguousarray(cpus, dtype=np.intc)
    cdef int n = c.shape[0]
    if n == 0:
        return 0
    cdef const int* ptr = <const int*>cnp.PyArray_DATA(c)
    cdef int bound
    with nogil:
        bound = colstore_bind_threads_to_cpus(ptr, n)
    return bound


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


cdef inline void _require_1d(tuple arrays, str message) except *:
    """Raise ValueError(message) unless every array is 1D."""
    cdef cnp.ndarray arr
    for arr in arrays:
        if arr.ndim != 1:
            raise ValueError(message)


cdef inline void _require_int64(cnp.ndarray arr, str name) except *:
    if arr.dtype != np.int64:
        raise TypeError(f"{name} must be int64; got {arr.dtype}.")


cdef inline void _require_int32(cnp.ndarray arr, str name) except *:
    if arr.dtype != np.int32:
        raise TypeError(f"{name} must be int32; got {arr.dtype}.")


cdef inline void _require_output_len(cnp.ndarray output, Py_ssize_t n, str other) except *:
    if output.shape[0] != n:
        raise ValueError(f"output length {output.shape[0]} does not match {other} length {n}.")


cdef inline void _require_record_arrays(
    cnp.ndarray record_starts_rows,
    cnp.ndarray record_starts_bytes,
    cnp.ndarray n_rows_per_record,
) except *:
    if (record_starts_rows.dtype != np.int64
            or record_starts_bytes.dtype != np.int64
            or n_rows_per_record.dtype != np.int64):
        raise TypeError("record index arrays must be int64.")


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
    thread_cap, prefetch_distance : int, optional
        Shared kernel parameters; see the module docstring.

    Raises
    ------
    TypeError
        If dtypes mismatch, ``indices`` is not int64, or the element size
        is not supported.
    ValueError
        If shapes are incompatible.
    """
    _require_1d((source, indices, output), "All inputs to gather must be 1D arrays.")
    _require_int64(indices, "indices")
    if source.dtype != output.dtype:
        raise TypeError(
            f"source dtype {source.dtype} does not match output dtype "
            f"{output.dtype}."
        )

    cdef ptrdiff_t n_indices = indices.shape[0]
    _require_output_len(output, n_indices, "indices")
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
    cdef int status
    with nogil:
        status = colstore_gather_indexed(base, indices_ptr, output_ptr, n_indices, itemsize, thread_cap, pd)
    if status != 0:
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
    thread_cap, prefetch_distance : int, optional
        Shared kernel parameters; see the module docstring.

    Notes
    -----
    Used by the multi-record reader where byte offsets encode record-header
    skips and per-record column offsets and cannot be reduced to a simple
    ``index * itemsize``. For the contiguous hot path, :func:`gather` is
    faster because it skips the byte-offset array materialization.

    Raises
    ------
    TypeError
        If ``byte_offsets`` is not int64, or the output element size is
        not supported.
    ValueError
        If shapes are incompatible.
    """
    _require_1d((byte_offsets, output), "byte_offsets and output must be 1D arrays.")
    _require_int64(byte_offsets, "byte_offsets")

    cdef ptrdiff_t n_indices = byte_offsets.shape[0]
    _require_output_len(output, n_indices, "byte_offsets")
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
    cdef int status
    with nogil:
        status = colstore_gather_bytes(base, offsets_ptr, output_ptr, n_indices, itemsize, thread_cap, pd)
    if status != 0:
        raise TypeError(
            f"Unsupported element size: {itemsize} bytes. The C++ kernel "
            f"handles 1, 2, 4, and 8 byte elements."
        )



def gather_multirecord_strided(cnp.ndarray source, cnp.ndarray output,
                               long long start, long long stop, long long step,
                               cnp.ndarray record_starts_rows,
                               cnp.ndarray record_starts_bytes,
                               cnp.ndarray n_rows_per_record,
                               long long col_prefix_bytes, int thread_cap=0,
                               Py_ssize_t prefetch_distance=-1):
    """Strided multi-record range gather: rows ``start, start+step, ...``.

    Reads the rows of Python ``slice(start, stop, step)`` semantics for one
    column without materializing an index array. ``step`` must be non-zero
    and ``output`` must have exactly ``len(range(start, stop, step))``
    elements. The caller guarantees every visited row is in range (the
    reader derives the triple from ``slice.indices``, which clamps) and
    that the dtype is in native byte order (raw typed loads cannot
    byteswap). ``record_starts_rows`` has ``n_records + 1`` entries and the
    other two index arrays have ``n_records``; an unsupported ``itemsize``
    raises :class:`TypeError`.
    """
    _require_1d((output, record_starts_rows, record_starts_bytes, n_rows_per_record), "output and index arrays must be 1D.")
    if step == 0:
        raise ValueError("step must be non-zero.")
    _require_record_arrays(record_starts_rows, record_starts_bytes, n_rows_per_record)

    cdef ptrdiff_t n = len(range(start, stop, step))
    _require_output_len(output, n, "the slice")
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
    cdef int status
    with nogil:
        status = colstore_gather_multirecord_strided(base, output_ptr, start, step, n, rsr, rsb, nrr, n_records, col_prefix_bytes, itemsize, thread_cap, pd)
    if status != 0:
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

    Specialization of the general fancy gather (:func:`gather_segment`) for
    files whose records all have ``rows_per_record`` rows (the final record
    may be partial, with ``last_record_rows`` rows) and whose record bodies
    sit at a constant byte stride. The caller detects the layout and guarantees its
    invariants (see ``gather.hpp``) plus native byte order; this entry
    validates only the scalar sanity conditions and the array contracts
    shared by every kernel.
    """
    _require_1d((indices, output), "indices and output must be 1D.")
    _require_int64(indices, "indices")
    if rows_per_record <= 0:
        raise ValueError("rows_per_record must be positive.")
    if n_records <= 0:
        raise ValueError("n_records must be positive.")
    if last_record_rows <= 0 or last_record_rows > rows_per_record:
        raise ValueError("last_record_rows must be in [1, rows_per_record].")

    cdef ptrdiff_t n = indices.shape[0]
    _require_output_len(output, n, "indices")
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
    cdef int status
    with nogil:
        status = colstore_gather_multirecord_uniform(base, indices_ptr, output_ptr, n, rows_per_record, record_stride_bytes, first_body_offset, n_records, last_record_rows, col_prefix_bytes, itemsize, thread_cap, pd)
    if status != 0:
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
    _require_1d((indices, output, bins), "indices, output, and bins must be 1D.")
    _require_int64(indices, "indices")
    _require_int32(bins, "bins")
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
    cdef int status
    with nogil:
        status = colstore_gather_multirecord_uniform_bins(base, indices_ptr, output_ptr, bins_ptr, n, rows_per_record, record_stride_bytes, first_body_offset, n_records, last_record_rows, col_prefix_bytes, itemsize, thread_cap, pd)
    if status != 0:
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

    ``bins`` must come from :func:`gather_multirecord_uniform_bins` run on
    the same ``indices`` and layout; behavior is undefined otherwise. Other
    parameters and errors match :func:`gather_multirecord_uniform_bins`.
    """
    _require_1d((indices, output, bins), "indices, output, and bins must be 1D.")
    _require_int64(indices, "indices")
    _require_int32(bins, "bins")
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
    cdef int status
    with nogil:
        status = colstore_gather_multirecord_uniform_withbins(base, indices_ptr, output_ptr, bins_ptr, n, rows_per_record, record_stride_bytes, first_body_offset, n_records, last_record_rows, col_prefix_bytes, itemsize, thread_cap, pd)
    if status != 0:
        raise TypeError(f"unsupported itemsize {itemsize}.")


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
    _require_1d((record_starts_rows, record_starts_bytes, n_rows_per_record, output), "Index arrays and output must be 1D.")
    _require_record_arrays(record_starts_rows, record_starts_bytes, n_rows_per_record)

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


def gather_segment(cnp.ndarray indices, cnp.ndarray output,
                     cnp.ndarray segment_starts_rows,
                     cnp.ndarray segment_base,
                     int thread_cap=0, Py_ssize_t prefetch_distance=-1):
    """Fused multi-file fancy gather: ``output[i] = value(indices[i])``.

    The multi-record fused gather one level up: several files form one global
    row space, decomposed into segments (one record of one file each). Each
    index is binned to its segment by the same branchless search the
    multi-record kernel uses for records, and read at an absolute address --
    no source base pointer, because segments live in different mmaps.

    Parameters
    ----------
    indices : numpy.ndarray
        1D ``int64`` global row indices, each in
        ``[0, segment_starts_rows[-1])`` (the view layer guarantees this).
        Need not be sorted.
    output : numpy.ndarray
        1D destination of length ``len(indices)`` and the column's dtype, in
        *native* byte order (the kernel does a raw typed load and cannot
        byteswap). Filled in-place, contiguous in requested order.
    segment_starts_rows : numpy.ndarray
        1D ``int64`` cumulative global row counts, length ``n_segments + 1``.
    segment_base : numpy.ndarray
        1D ``int64`` per-segment absolute byte addresses, length
        ``n_segments``: global row ``idx`` of segment ``s`` is at
        ``segment_base[s] + idx * itemsize``. The caller folds each file's
        mmap base, record body and column offset, and the segment's global
        start row into this value.
    thread_cap, prefetch_distance : int, optional
        Shared kernel parameters; see the module docstring.

    Raises
    ------
    TypeError
        If ``indices`` or any segment array is not ``int64``, or the element
        size is unsupported.
    ValueError
        If shapes are 1D-inconsistent or lengths disagree.
    """
    _require_1d((indices, output, segment_starts_rows, segment_base), "indices, output, and segment arrays must be 1D.")
    _require_int64(indices, "indices")
    _require_int64(segment_starts_rows, "segment_starts_rows")
    _require_int64(segment_base, "segment_base")

    cdef ptrdiff_t n = indices.shape[0]
    _require_output_len(output, n, "indices")
    cdef long long n_segments = segment_base.shape[0]
    if segment_starts_rows.shape[0] != n_segments + 1:
        raise ValueError("segment_starts_rows length must be n_segments + 1.")
    if n == 0:
        return

    cdef int itemsize = output.dtype.itemsize
    _require_c_contiguous((("indices", indices), ("output", output), ("segment_starts_rows", segment_starts_rows), ("segment_base", segment_base)))
    cdef const int64_t* indices_ptr = <const int64_t*>cnp.PyArray_DATA(indices)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef const int64_t* ssr = <const int64_t*>cnp.PyArray_DATA(segment_starts_rows)
    cdef const int64_t* sb = <const int64_t*>cnp.PyArray_DATA(segment_base)

    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    cdef int status
    with nogil:
        status = colstore_gather_segment(indices_ptr, output_ptr, n, ssr, sb, n_segments, itemsize, thread_cap, pd)
    if status != 0:
        raise TypeError(
            f"Unsupported element size: {itemsize} bytes. The C++ kernel "
            f"handles 1, 2, 4, and 8 byte elements."
        )


def gather_segment_uniform(cnp.ndarray indices, cnp.ndarray output,
                             long long rows_per_segment, cnp.ndarray segment_base,
                             int thread_cap=0, Py_ssize_t prefetch_distance=-1):
    """Uniform-grid multi-file fancy gather: ``s = idx / rows_per_segment``.

    Specialization of :func:`gather_segment` for a segment table whose every
    segment holds ``rows_per_segment`` rows (the global-last may be partial).
    The per-index segment is a magic-reciprocal division instead of the
    branchless binary search; the address is the same
    ``segment_base[s] + idx * itemsize``, so no ``segment_starts_rows`` array is
    needed. The caller (the dataset detects the grid) guarantees every index in
    ``[0, n_segments * rows_per_segment)`` and native byte order.
    """
    _require_1d((indices, output, segment_base), "indices, output, and segment_base must be 1D.")
    _require_int64(indices, "indices")
    _require_int64(segment_base, "segment_base")
    if rows_per_segment <= 0:
        raise ValueError("rows_per_segment must be positive.")

    cdef ptrdiff_t n = indices.shape[0]
    _require_output_len(output, n, "indices")
    cdef long long n_segments = segment_base.shape[0]
    if n == 0:
        return

    cdef int itemsize = output.dtype.itemsize
    _require_c_contiguous((("indices", indices), ("output", output), ("segment_base", segment_base)))
    cdef const int64_t* indices_ptr = <const int64_t*>cnp.PyArray_DATA(indices)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef const int64_t* sb = <const int64_t*>cnp.PyArray_DATA(segment_base)

    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    cdef int status
    with nogil:
        status = colstore_gather_segment_uniform(indices_ptr, output_ptr, n, rows_per_segment, sb, n_segments, itemsize, thread_cap, pd)
    if status != 0:
        raise TypeError(
            f"Unsupported element size: {itemsize} bytes. The C++ kernel "
            f"handles 1, 2, 4, and 8 byte elements."
        )


def gather_segment_uniform_bins(cnp.ndarray indices, cnp.ndarray output, cnp.ndarray bins,
                                  long long rows_per_segment, cnp.ndarray segment_base,
                                  int thread_cap=0, Py_ssize_t prefetch_distance=-1):
    """Uniform-grid multi-file gather that also records each index's segment bin.

    Identical addressing and output to :func:`gather_segment_uniform`, plus
    ``bins[i]`` is filled with the segment index (``int32``) that ``indices[i]``
    binned to -- a multi-column read computes the grid division once here and
    reuses it via :func:`gather_segment_withbins`. ``bins`` must be 1D ``int32``
    of the same length as ``indices``; the caller guarantees
    ``n_segments <= 2**31 - 1``.
    """
    _require_1d((indices, output, bins, segment_base), "indices, output, bins, and segment_base must be 1D.")
    _require_int64(indices, "indices")
    _require_int32(bins, "bins")
    _require_int64(segment_base, "segment_base")
    if rows_per_segment <= 0:
        raise ValueError("rows_per_segment must be positive.")

    cdef ptrdiff_t n = indices.shape[0]
    if output.shape[0] != n or bins.shape[0] != n:
        raise ValueError(
            f"output/bins lengths ({output.shape[0]}/{bins.shape[0]}) must "
            f"match indices length {n}."
        )
    cdef long long n_segments = segment_base.shape[0]
    if n == 0:
        return

    cdef int itemsize = output.dtype.itemsize
    _require_c_contiguous((("indices", indices), ("output", output), ("bins", bins), ("segment_base", segment_base)))
    cdef const int64_t* indices_ptr = <const int64_t*>cnp.PyArray_DATA(indices)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef int32_t* bins_ptr = <int32_t*>cnp.PyArray_DATA(bins)
    cdef const int64_t* sb = <const int64_t*>cnp.PyArray_DATA(segment_base)

    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    cdef int status
    with nogil:
        status = colstore_gather_segment_uniform_bins(indices_ptr, output_ptr, bins_ptr, n, rows_per_segment, sb, n_segments, itemsize, thread_cap, pd)
    if status != 0:
        raise TypeError(
            f"Unsupported element size: {itemsize} bytes. The C++ kernel "
            f"handles 1, 2, 4, and 8 byte elements."
        )


def gather_segment_sorted(cnp.ndarray indices, cnp.ndarray output,
                            cnp.ndarray segment_starts_rows,
                            cnp.ndarray segment_base,
                            int thread_cap=0, Py_ssize_t prefetch_distance=-1):
    """Sorted multi-file fancy gather: a monotonic segment cursor, no search.

    Identical output to :func:`gather_segment` but requires ``indices``
    non-decreasing (the caller proves it). The cursor advances forward through
    the segments as the indices climb, so the per-index cost drops to one
    boundary compare and the within-segment access is sequential. A cursor walk
    has nothing to amortize across columns, so a multi-column sorted read calls
    this per column rather than the bins pair. All other parameters and errors
    match :func:`gather_segment`.
    """
    _require_1d((indices, output, segment_starts_rows, segment_base), "indices, output, and segment arrays must be 1D.")
    _require_int64(indices, "indices")
    _require_int64(segment_starts_rows, "segment_starts_rows")
    _require_int64(segment_base, "segment_base")

    cdef ptrdiff_t n = indices.shape[0]
    _require_output_len(output, n, "indices")
    cdef long long n_segments = segment_base.shape[0]
    if segment_starts_rows.shape[0] != n_segments + 1:
        raise ValueError("segment_starts_rows length must be n_segments + 1.")
    if n == 0:
        return

    cdef int itemsize = output.dtype.itemsize
    _require_c_contiguous((("indices", indices), ("output", output), ("segment_starts_rows", segment_starts_rows), ("segment_base", segment_base)))
    cdef const int64_t* indices_ptr = <const int64_t*>cnp.PyArray_DATA(indices)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef const int64_t* ssr = <const int64_t*>cnp.PyArray_DATA(segment_starts_rows)
    cdef const int64_t* sb = <const int64_t*>cnp.PyArray_DATA(segment_base)

    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    cdef int status
    with nogil:
        status = colstore_gather_segment_sorted(indices_ptr, output_ptr, n, ssr, sb, n_segments, itemsize, thread_cap, pd)
    if status != 0:
        raise TypeError(
            f"Unsupported element size: {itemsize} bytes. The C++ kernel "
            f"handles 1, 2, 4, and 8 byte elements."
        )


def gather_segment_mask(cnp.ndarray mask, cnp.ndarray output,
                          cnp.ndarray segment_starts_rows, cnp.ndarray segment_base,
                          int thread_cap=0, Py_ssize_t prefetch_distance=-1):
    """Boolean-mask-native gather over a segment table, single- or multi-file: no index array.

    ``mask`` is a numpy bool array with one entry per global row (length must
    equal ``segment_starts_rows[-1]``); selected rows are gathered in ascending
    global-row order. ``output`` must be sized to exactly
    ``np.count_nonzero(mask)`` -- the kernel re-counts internally to fix
    per-thread offsets and raises if the caller's size disagrees, writing
    nothing in that case. The segment arrays are :func:`gather_segment`'s.
    Native byte order required; other parameters match :func:`gather_segment`.
    """
    _require_1d((mask, output, segment_starts_rows, segment_base),
                "mask, output, and segment arrays must be 1D.")
    if mask.dtype != np.bool_:
        raise TypeError(f"mask must be bool; got {mask.dtype}.")
    _require_int64(segment_starts_rows, "segment_starts_rows")
    _require_int64(segment_base, "segment_base")

    cdef long long n_segments = segment_base.shape[0]
    if segment_starts_rows.shape[0] != n_segments + 1:
        raise ValueError("segment_starts_rows length must be n_segments + 1.")
    _require_c_contiguous((("mask", mask), ("output", output), ("segment_starts_rows", segment_starts_rows), ("segment_base", segment_base)))
    cdef const int64_t* ssr = <const int64_t*>cnp.PyArray_DATA(segment_starts_rows)
    cdef int64_t n_rows = ssr[n_segments]
    if mask.shape[0] != n_rows:
        raise ValueError(
            f"mask length {mask.shape[0]} does not match the dataset's row count {n_rows}."
        )
    cdef ptrdiff_t n_out = output.shape[0]
    if n_rows == 0 or n_out == 0:
        if n_out == 0 and mask.shape[0] != 0 and bool(np.any(mask)):
            raise ValueError("output length does not match the mask's selected count.")
        return

    cdef int itemsize = output.dtype.itemsize
    cdef const uint8_t* mask_ptr = <const uint8_t*>cnp.PyArray_DATA(mask)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef const int64_t* sb = <const int64_t*>cnp.PyArray_DATA(segment_base)
    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    cdef int status
    with nogil:
        status = colstore_gather_segment_mask(mask_ptr, output_ptr, n_rows, n_out, ssr, sb, n_segments, itemsize, thread_cap, pd)
    if status == -1:
        raise TypeError(f"unsupported itemsize {itemsize}.")
    if status != 0:
        raise ValueError("output length does not match the mask's selected count.")


def parallel_copy_runs(cnp.ndarray output, cnp.ndarray src_addrs,
                       cnp.ndarray dst_offsets, cnp.ndarray byte_lengths,
                       int thread_cap=0):
    """Parallel byte copy of contiguous runs into ``output``.

    Run ``r`` copies ``byte_lengths[r]`` bytes from absolute address
    ``src_addrs[r]`` to ``output`` byte offset ``dst_offsets[r]``; the runs
    tile ``output`` in ascending order. The total byte span is split across
    threads, so a few large runs (a multi-file whole/forward-slice read) still
    parallelize. The caller keeps the source buffers alive across the call.
    """
    _require_1d((src_addrs, dst_offsets, byte_lengths), "run arrays must be 1D.")
    _require_int64(src_addrs, "src_addrs")
    _require_int64(dst_offsets, "dst_offsets")
    _require_int64(byte_lengths, "byte_lengths")
    cdef long long n_runs = src_addrs.shape[0]
    if dst_offsets.shape[0] != n_runs or byte_lengths.shape[0] != n_runs:
        raise ValueError("src_addrs, dst_offsets, and byte_lengths must be equal length.")
    if n_runs == 0:
        return
    _require_c_contiguous((("output", output), ("src_addrs", src_addrs),
                           ("dst_offsets", dst_offsets), ("byte_lengths", byte_lengths)))
    cdef uint8_t* out_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef const int64_t* sa = <const int64_t*>cnp.PyArray_DATA(src_addrs)
    cdef const int64_t* do = <const int64_t*>cnp.PyArray_DATA(dst_offsets)
    cdef const int64_t* bl = <const int64_t*>cnp.PyArray_DATA(byte_lengths)
    with nogil:
        colstore_parallel_copy_runs(out_ptr, sa, do, bl, n_runs, thread_cap)


def interleave_records(cnp.ndarray output, long long record_itemsize, long long n_rows,
                       cnp.ndarray src_addrs, cnp.ndarray src_itemsizes,
                       cnp.ndarray field_offsets, int thread_cap=0):
    """Interleave columns into a record array (the SoA -> AoS transpose).

    Row ``i`` of ``output`` (record stride ``record_itemsize`` bytes) gets each
    column's element ``i`` at its field offset; column ``c``'s element ``i`` is
    ``src_itemsizes[c]`` bytes at ``src_addrs[c] + i * src_itemsizes[c]`` (each
    column contiguous, an absolute address so it may be a memmap). The work is
    split over row ranges. The caller keeps the source buffers alive across the
    call and guarantees native byte order (a raw field copy cannot byteswap).
    """
    _require_1d((src_addrs, src_itemsizes, field_offsets), "column arrays must be 1D.")
    _require_int64(src_addrs, "src_addrs")
    _require_int64(src_itemsizes, "src_itemsizes")
    _require_int64(field_offsets, "field_offsets")
    cdef long long n_cols = src_addrs.shape[0]
    if src_itemsizes.shape[0] != n_cols or field_offsets.shape[0] != n_cols:
        raise ValueError("src_addrs, src_itemsizes, and field_offsets must be equal length.")
    if n_rows == 0 or n_cols == 0:
        return
    _require_c_contiguous((("output", output), ("src_addrs", src_addrs),
                           ("src_itemsizes", src_itemsizes), ("field_offsets", field_offsets)))
    cdef uint8_t* out_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef const int64_t* sa = <const int64_t*>cnp.PyArray_DATA(src_addrs)
    cdef const int64_t* si = <const int64_t*>cnp.PyArray_DATA(src_itemsizes)
    cdef const int64_t* fo = <const int64_t*>cnp.PyArray_DATA(field_offsets)
    with nogil:
        colstore_interleave_records(out_ptr, record_itemsize, n_rows, sa, si, fo,
                                    n_cols, thread_cap)


def gather_segment_bins(cnp.ndarray indices, cnp.ndarray output, cnp.ndarray bins,
                          cnp.ndarray segment_starts_rows, cnp.ndarray segment_base,
                          int thread_cap=0, Py_ssize_t prefetch_distance=-1):
    """Fused multi-file gather that also records each index's segment bin.

    Identical addressing and output to :func:`gather_segment`, plus ``bins[i]``
    is filled with the segment index (``int32``) that ``indices[i]`` binned to.
    The segment is column-independent, so a multi-column read computes it once
    here and reuses it via :func:`gather_segment_withbins`. ``bins`` must be
    1D ``int32`` of the same length as ``indices``; the caller guarantees
    ``n_segments <= 2**31 - 1``. ``segment_base`` is this column's bases.
    """
    _require_1d((indices, output, bins, segment_starts_rows, segment_base), "indices, output, bins, and segment arrays must be 1D.")
    _require_int64(indices, "indices")
    _require_int32(bins, "bins")
    _require_int64(segment_starts_rows, "segment_starts_rows")
    _require_int64(segment_base, "segment_base")

    cdef ptrdiff_t n = indices.shape[0]
    if output.shape[0] != n or bins.shape[0] != n:
        raise ValueError(
            f"output/bins lengths ({output.shape[0]}/{bins.shape[0]}) must "
            f"match indices length {n}."
        )
    cdef long long n_segments = segment_base.shape[0]
    if segment_starts_rows.shape[0] != n_segments + 1:
        raise ValueError("segment_starts_rows length must be n_segments + 1.")
    if n == 0:
        return

    cdef int itemsize = output.dtype.itemsize
    _require_c_contiguous((("indices", indices), ("output", output), ("bins", bins), ("segment_starts_rows", segment_starts_rows), ("segment_base", segment_base)))
    cdef const int64_t* indices_ptr = <const int64_t*>cnp.PyArray_DATA(indices)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef int32_t* bins_ptr = <int32_t*>cnp.PyArray_DATA(bins)
    cdef const int64_t* ssr = <const int64_t*>cnp.PyArray_DATA(segment_starts_rows)
    cdef const int64_t* sb = <const int64_t*>cnp.PyArray_DATA(segment_base)

    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    cdef int status
    with nogil:
        status = colstore_gather_segment_bins(indices_ptr, output_ptr, bins_ptr, n, ssr, sb, n_segments, itemsize, thread_cap, pd)
    if status != 0:
        raise TypeError(
            f"Unsupported element size: {itemsize} bytes. The C++ kernel "
            f"handles 1, 2, 4, and 8 byte elements."
        )


def gather_segment_withbins(cnp.ndarray indices, cnp.ndarray output, cnp.ndarray bins,
                              cnp.ndarray segment_base,
                              int thread_cap=0, Py_ssize_t prefetch_distance=-1):
    """Gather one column reusing segment bins from :func:`gather_segment_bins`.

    The segment is the sequential ``int32`` read ``bins[i]`` rather than a
    search; the address is ``segment_base[bins[i]] + indices[i] * itemsize`` with
    this column's ``segment_base``. ``bins`` must be 1D ``int32`` of the same
    length as ``indices`` (as filled by :func:`gather_segment_bins`).
    """
    _require_1d((indices, output, bins, segment_base), "indices, output, bins, and segment_base must be 1D.")
    _require_int64(indices, "indices")
    _require_int32(bins, "bins")
    _require_int64(segment_base, "segment_base")

    cdef ptrdiff_t n = indices.shape[0]
    if output.shape[0] != n or bins.shape[0] != n:
        raise ValueError(
            f"output/bins lengths ({output.shape[0]}/{bins.shape[0]}) must "
            f"match indices length {n}."
        )
    if n == 0:
        return

    cdef int itemsize = output.dtype.itemsize
    _require_c_contiguous((("indices", indices), ("output", output), ("bins", bins), ("segment_base", segment_base)))
    cdef const int64_t* indices_ptr = <const int64_t*>cnp.PyArray_DATA(indices)
    cdef uint8_t* output_ptr = <uint8_t*>cnp.PyArray_DATA(output)
    cdef const int32_t* bins_ptr = <const int32_t*>cnp.PyArray_DATA(bins)
    cdef const int64_t* sb = <const int64_t*>cnp.PyArray_DATA(segment_base)

    cdef ptrdiff_t pd = (
        DEFAULT_PREFETCH_DISTANCE if prefetch_distance < 0 else prefetch_distance
    )
    cdef int status
    with nogil:
        status = colstore_gather_segment_withbins(indices_ptr, output_ptr, bins_ptr, n, sb, itemsize, thread_cap, pd)
    if status != 0:
        raise TypeError(
            f"Unsupported element size: {itemsize} bytes. The C++ kernel "
            f"handles 1, 2, 4, and 8 byte elements."
        )

def build_flags() -> set:
    """Names of the optimization toggles this extension was compiled with."""
    cdef bytes raw = colstore_build_flags()
    return set(raw.decode("ascii").split())


def resolve_thread_count(n_indices: int, thread_cap: int = 0) -> int:
    """Thread count a kernel would use for ``n_indices`` elements.

    Mirrors the C++ resolution exactly (serial below the parallel threshold,
    then ~one thread per 1<<20 elements, clamped to ``thread_cap``;
    ``thread_cap <= 0`` means the OpenMP maximum).
    """
    return colstore_resolve_thread_count(n_indices, thread_cap)


def read_record_index(path, int64_t data_offset, int64_t n_records, int64_t itemsize_sum,
                       int64_t read_chunk=1 << 20):
    """Build the per-record index by walking the record headers natively.

    Reads each 32-byte record header at its position in ``path`` (skipping the
    bodies), validates magic / sequential index / CRC32, and returns three
    ``int64`` arrays: cumulative row counts ``(n_records + 1,)``, body byte
    offsets ``(n_records,)``, and per-record row counts ``(n_records,)``.

    Parameters
    ----------
    path :
        File to read; accepts ``str``, ``bytes``, or ``os.PathLike``.
    data_offset :
        Byte offset of the first record header.
    n_records :
        Number of records to walk.
    itemsize_sum :
        Sum of the column itemsizes; a record body occupies
        ``align_up(n_rows * itemsize_sum, 8)`` bytes.
    read_chunk :
        Size in bytes of the reused sliding read buffer (default 1 MiB).
        Larger values amortize the syscall count across more records;
        ``32`` reads each header in isolation.

    Raises
    ------
    FormatError
        On a corrupt or truncated record header (bad magic, mismatched record
        index, CRC mismatch, or a file shorter than its records imply).
    """
    cdef cnp.ndarray record_starts_rows = np.empty(n_records + 1, dtype=np.int64)
    cdef cnp.ndarray record_starts_bytes = np.empty(n_records, dtype=np.int64)
    cdef cnp.ndarray n_rows_per_record = np.empty(n_records, dtype=np.int64)
    cdef bytes path_bytes = os.fsencode(path)
    cdef const char* path_c = path_bytes
    cdef int64_t err_offset = 0
    cdef int64_t err_record = 0
    cdef int64_t err_stored = 0
    cdef uint32_t err_crc_stored = 0
    cdef uint32_t err_crc_actual = 0
    cdef int64_t* rows_ptr = <int64_t*>cnp.PyArray_DATA(record_starts_rows)
    cdef int64_t* bytes_ptr = <int64_t*>cnp.PyArray_DATA(record_starts_bytes)
    cdef int64_t* nrows_ptr = <int64_t*>cnp.PyArray_DATA(n_rows_per_record)
    cdef int status
    with nogil:
        status = colstore_read_record_index(
            path_c, data_offset, n_records, itemsize_sum, read_chunk,
            rows_ptr, bytes_ptr, nrows_ptr,
            &err_offset, &err_record, &err_stored,
            &err_crc_stored, &err_crc_actual,
        )
    if status != 0:
        from colstore.format import FormatError

        if status == -2:
            raise FormatError(
                f"Truncated record header at offset {err_offset}: "
                f"expected 32 bytes, got {err_stored}."
            )
        if status == -3:
            raise FormatError(
                f"Bad record magic at offset {err_offset}: expected b'REC\\x01'."
            )
        if status == -4:
            raise FormatError(
                f"Record index mismatch at offset {err_offset}: manifest expects "
                f"record {err_record}, header says {err_stored}."
            )
        if status == -5:
            raise FormatError(
                f"Record {err_record} header CRC mismatch "
                f"(stored {err_crc_stored}, computed {err_crc_actual})."
            )
        if status == -6:
            raise FormatError(
                f"File is truncated: last record body ends at offset {err_offset} "
                f"but file is only {err_stored} bytes."
            )
        raise FormatError(f"Could not read record index (status {status}).")
    return record_starts_rows, record_starts_bytes, n_rows_per_record
