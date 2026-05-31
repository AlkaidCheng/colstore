"""Gather backends: C++/Cython (default), NumPy, and optional Numba.

The C++ backend lives in the compiled extension ``colstore._gather``;
when the extension is not built, the dispatcher falls back to NumPy with
a one-time warning. NumPy is always available and is used for ``slice``
and full-column reads where fancy indexing isn't required.
"""

import warnings

import numpy as np

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


def gather(
    source: np.ndarray,
    indices: np.ndarray,
    dtype: np.dtype,
    *,
    backend: str = "cpp",
) -> np.ndarray:
    """Return ``source[indices]`` as an owning ndarray using the chosen backend.

    Parameters
    ----------
    source : numpy.ndarray
        Source 1D array (typically a ``np.memmap`` view).
    indices : numpy.ndarray
        Integer index array, ``int64``.
    dtype : numpy.dtype
        Output dtype; must match ``source.dtype``.
    backend : str, optional
        ``"cpp"`` (default), ``"numpy"``, or ``"numba"``. Falls back to NumPy
        with a warning if the requested backend isn't available.

    Returns
    -------
    numpy.ndarray
        Owning 1D array of length ``len(indices)``.
    """
    if backend == "cpp":
        if _CPP_AVAILABLE:
            output = np.empty(indices.shape[0], dtype=dtype)
            _cpp_module.gather(source, indices, output)
            return output
        warnings.warn(
            "Requested 'cpp' backend but the compiled extension is not "
            "available; falling back to NumPy. Rebuild the package to "
            "enable the C++ kernel.",
            RuntimeWarning,
            stacklevel=2,
        )
        backend = "numpy"

    if backend == "numba":
        if _NUMBA_AVAILABLE:
            output = np.empty(indices.shape[0], dtype=dtype)
            _numba_gather_kernel(source, indices, output)
            return output
        warnings.warn(
            "Requested 'numba' backend but Numba is not installed; " "falling back to NumPy.",
            RuntimeWarning,
            stacklevel=2,
        )
        backend = "numpy"

    if backend != "numpy":
        raise ValueError(
            f"Unknown gather backend {backend!r}; " f"expected 'cpp', 'numpy', or 'numba'."
        )
    return np.asarray(source[indices], dtype=dtype)
