"""Python wrapper layer over the compiled ``colstore._gather`` kernels.

Wraps the gather kernels (C++/Cython, with a NumPy fallback) plus
record-index build, parallel copy-runs, and record interleave.
When the compiled extension is not built, :func:`gather` falls back to NumPy
with a one-time warning; NumPy is always available and is used for ``slice``
and full-column reads where fancy indexing isn't required.
"""

import os
import warnings
from typing import Any

import numpy as np

from . import config

try:
    from . import _gather as _cpp_module  # type: ignore[attr-defined]

    _CPP_AVAILABLE = True
except ImportError as exc:
    _CPP_IMPORT_ERROR = exc
    _CPP_AVAILABLE = False


def cpp_available() -> bool:
    """Return whether the compiled C++ gather extension is importable."""
    return _CPP_AVAILABLE


def max_threads() -> int:
    """Return the gather kernel's max thread count (1 without OpenMP)."""
    if _CPP_AVAILABLE:
        # The compiled module has no stubs; cast its return through int().
        return int(_cpp_module.max_threads())
    return 1


def read_record_index(
    path: str | os.PathLike[str],
    data_offset: int,
    n_records: int,
    itemsizes: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the per-record index with the C++ kernel.

    Returns ``(record_starts_rows, record_starts_bytes, n_rows_per_record)`` as
    int64 arrays. Requires the compiled extension, so callers gate on
    :func:`cpp_available`; ``colstore.format.read_record_index`` falls back to a
    pure-Python walk when it is not built. Raises
    :class:`colstore.format.FormatError` on a corrupt or truncated record header.
    """
    # The compiled module has no stubs; bind through a typed local.
    result: tuple[np.ndarray, np.ndarray, np.ndarray] = _cpp_module.read_record_index(
        os.fspath(path), int(data_offset), int(n_records), int(sum(itemsizes))
    )
    return result


def parallel_copy_runs(
    output: np.ndarray,
    src_addrs: np.ndarray,
    dst_offsets: np.ndarray,
    byte_lengths: np.ndarray,
    thread_cap: int,
) -> None:
    """Copy byte runs into ``output`` via the C++ kernel.

    Run ``r`` copies ``byte_lengths[r]`` bytes from absolute address
    ``src_addrs[r]`` to ``output`` byte offset ``dst_offsets[r]``, splitting the
    runs' total bytes across threads. The caller ensures the extension is built
    (:func:`cpp_available`) and keeps the source buffers alive across the call.
    """
    _cpp_module.parallel_copy_runs(output, src_addrs, dst_offsets, byte_lengths, thread_cap)


def interleave_records(
    output: np.ndarray,
    record_itemsize: int,
    n_rows: int,
    src_addrs: np.ndarray,
    src_itemsizes: np.ndarray,
    field_offsets: np.ndarray,
    thread_cap: int,
) -> None:
    """Interleave columns into a record array via the C++ kernel (SoA -> AoS).

    Row ``i`` of ``output`` gets each column's element ``i`` at its field offset;
    column ``c``'s element ``i`` is ``src_itemsizes[c]`` bytes at
    ``src_addrs[c] + i * src_itemsizes[c]``. The caller ensures the extension is
    built (:func:`cpp_available`), keeps the source buffers alive across the
    call, and guarantees native byte order.
    """
    _cpp_module.interleave_records(
        output, record_itemsize, n_rows, src_addrs, src_itemsizes, field_offsets, thread_cap
    )


def interleave_record_array(
    column_names: list[str], sources: list[np.ndarray], record_dtype: np.dtype[Any]
) -> np.ndarray:
    """Assemble per-column ``sources`` into one record array (assumes >= 1 source).

    Uses the parallel :func:`interleave_records` kernel when the extension is
    built -- writing the record array row-major, once -- else a column-major
    per-field assignment. Each source must be C-contiguous with a dtype equal to
    its field in ``record_dtype`` (the kernel copies raw bytes, so source and
    field byte order match by construction); ``sources`` aligns with
    ``column_names``. Shared by the reader's gather path and the editing frame.
    """
    n_records = sources[0].shape[0]
    record_array: np.ndarray = np.empty(n_records, dtype=record_dtype)
    if n_records == 0 or not cpp_available():
        for name, source in zip(column_names, sources, strict=True):
            record_array[name] = source
        return record_array
    fields = record_dtype.fields
    assert fields is not None  # a structured dtype always has fields
    interleave_records(
        record_array,
        record_dtype.itemsize,
        n_records,
        np.array([s.ctypes.data for s in sources], dtype=np.int64),
        np.array([s.dtype.itemsize for s in sources], dtype=np.int64),
        np.array([fields[name][1] for name in column_names], dtype=np.int64),
        config.get_gather_thread_cap(),
    )
    return record_array


def _is_native_order(array: np.ndarray) -> bool:
    """Return whether `array` is in the host's native byte order."""
    byteorder = array.dtype.byteorder
    if byteorder in ("=", "|"):
        return True
    return (byteorder == "<") == bool(np.little_endian)


def gather(
    source: np.ndarray,
    indices: np.ndarray,
    dtype: np.dtype,
    *,
    backend: str = "cpp",
    thread_cap: int | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``source[indices]`` as an owning ndarray using the chosen backend.

    Parameters
    ----------
    source : numpy.ndarray
        Source 1D array (typically a ``np.memmap`` view).
    indices : numpy.ndarray
        Integer index array, ``int64``.
    dtype : numpy.dtype
        Output dtype; native byte order.
    backend : str, optional
        ``"cpp"`` (default) or ``"numpy"``. Falls back to
        NumPy with a warning if the requested backend is unavailable, and
        silently for dtype kinds or byte orders the compiled kernels
        cannot handle. The C++ kernel decides serial vs parallel execution
        from input size and the configured thread cap.
    thread_cap : int or None, optional
        Per-call OpenMP thread cap for the C++ kernel; ``None`` (default)
        uses :func:`colstore.config.get_gather_thread_cap`. Callers running
        several gathers concurrently pass a reduced cap so outer threads x
        inner OpenMP threads does not oversubscribe the cores.
    out : numpy.ndarray or None, optional
        A contiguous buffer of length ``len(indices)`` and dtype ``dtype`` to fill
        in place instead of allocating -- a reuse hint for streaming batches. The
        compiled/JIT kernels fill it directly; the NumPy fallback copies into it. It
        is returned as-is.

    Returns
    -------
    numpy.ndarray
        Owning 1D array of length ``len(indices)``.
    """
    # The compiled/JIT kernels do raw element copies and assume native byte
    # order and a fixed set of numeric kinds. Anything else uses NumPy, which
    # handles byte-swapping and exotic dtypes correctly.
    kernel_compatible = dtype.kind in ("f", "i", "u", "b") and _is_native_order(source)
    effective_cap = config.resolve_gather_thread_cap(thread_cap)

    if backend == "cpp":
        if _CPP_AVAILABLE and kernel_compatible:
            # The C++ kernel beats numpy.take per-thread by 2-4x even at one
            # thread (numpy re-validates every index on top of work we already
            # did at the Python layer), and pulls further ahead in parallel.
            # So when the kernel is compatible we always use it; the kernel's
            # own resolve_thread_count picks the right number of threads
            # (1 below the parallel threshold, scaling up from there).
            output = out if out is not None else np.empty(indices.shape[0], dtype=dtype)
            # The O(K) sortedness check is only paid in "auto" mode, where it
            # is one of the two signals classifying the access regime; an
            # explicit setting skips it (resolve is a passthrough then).
            if config.get_prefetch_distance() == "auto":
                indices_sorted = indices.shape[0] > 1 and bool(np.all(indices[1:] >= indices[:-1]))
            else:
                indices_sorted = False
            prefetch = config.resolve_prefetch_distance(source.nbytes, indices_sorted)
            _cpp_module.gather_into(source, indices, output, effective_cap, prefetch)
            return output
        if not _CPP_AVAILABLE:
            warnings.warn(
                "Requested 'cpp' backend but the compiled extension is not "
                "available; falling back to NumPy. Rebuild the package to "
                "enable the C++ kernel.",
                RuntimeWarning,
                stacklevel=2,
            )
        # dtypes outside the kernel's supported kinds (e.g. datetime64, fixed-
        # width strings) or non-native byte order fall through to NumPy.
        backend = "numpy"

    if backend != "numpy":
        raise ValueError(f"Unknown gather backend {backend!r}; expected 'cpp' or 'numpy'.")
    result = np.asarray(source[indices], dtype=dtype)
    if out is not None:
        out[:] = result
        return out
    return result
