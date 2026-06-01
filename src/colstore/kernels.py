"""Gather backends: C++/Cython (default), NumPy, and optional Numba.

The C++ backend lives in the compiled extension ``colstore._gather``;
when the extension is not built, the dispatcher falls back to NumPy with
a one-time warning. NumPy is always available and is used for ``slice``
and full-column reads where fancy indexing isn't required.
"""

import warnings

import numpy as np

from . import config

try:
    from . import _gather as _cpp_module  # type: ignore[attr-defined]

    _CPP_AVAILABLE = True
except ImportError as exc:
    _CPP_IMPORT_ERROR = exc
    _CPP_AVAILABLE = False

try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True

    @njit(parallel=True, cache=True, boundscheck=False, fastmath=True)  # type: ignore[untyped-decorator]
    def _numba_gather_kernel(source: np.ndarray, indices: np.ndarray, output: np.ndarray) -> None:
        for i in prange(indices.shape[0]):
            output[i] = source[indices[i]]

except ImportError:
    _NUMBA_AVAILABLE = False


def cpp_available() -> bool:
    """Return whether the compiled C++ gather extension is importable."""
    return _CPP_AVAILABLE


def numba_available() -> bool:
    """Return whether Numba is importable in this environment."""
    return _NUMBA_AVAILABLE


def max_threads() -> int:
    """Return the gather kernel's max thread count (1 without OpenMP)."""
    if _CPP_AVAILABLE:
        # The compiled module has no stubs; cast its return through int().
        return int(_cpp_module.max_threads())
    return 1


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
        ``"cpp"`` (default), ``"numpy"``, or ``"numba"``. Falls back to NumPy
        with a warning if the requested backend isn't available, and silently
        for dtype kinds or byte orders the compiled kernels can't handle.
    thread_cap : int or None, optional
        Per-call OpenMP thread cap for the C++ kernel. ``None`` (default) uses
        the package-wide :func:`colstore.config.get_gather_thread_cap`. Callers
        that run several gathers concurrently (e.g. multi-column reads) pass a
        reduced cap so the product of outer threads and inner OpenMP threads
        does not oversubscribe the cores.

    Returns
    -------
    numpy.ndarray
        Owning 1D array of length ``len(indices)``.
    """
    # The compiled/JIT kernels do raw element copies and assume native byte
    # order and a fixed set of numeric kinds. Anything else uses NumPy, which
    # handles byte-swapping and exotic dtypes correctly.
    kernel_compatible = dtype.kind in ("f", "i", "u", "b") and _is_native_order(source)
    effective_cap = config.get_gather_thread_cap() if thread_cap is None else max(1, thread_cap)

    if backend == "cpp":
        if _CPP_AVAILABLE and kernel_compatible:
            # The C++ kernel beats numpy.take per-thread by 2-4x even at one
            # thread (numpy re-validates every index on top of work we already
            # did at the Python layer), and pulls further ahead in parallel.
            # So when the kernel is compatible we always use it; the kernel's
            # own resolve_thread_count picks the right number of threads
            # (1 below the parallel threshold, scaling up from there).
            output = np.empty(indices.shape[0], dtype=dtype)
            _cpp_module.gather_into(source, indices, output, effective_cap)
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

    if backend == "numba":
        if _NUMBA_AVAILABLE and kernel_compatible:
            output = np.empty(indices.shape[0], dtype=dtype)
            _numba_gather_kernel(source, indices, output)
            return output
        if not _NUMBA_AVAILABLE:
            warnings.warn(
                "Requested 'numba' backend but Numba is not installed; falling back to NumPy.",
                RuntimeWarning,
                stacklevel=2,
            )
        backend = "numpy"

    if backend != "numpy":
        raise ValueError(
            f"Unknown gather backend {backend!r}; " f"expected 'cpp', 'numpy', or 'numba'."
        )
    return np.asarray(source[indices], dtype=dtype)
