"""Memory-mapped columnar binary format for fast random-access I/O.

``colstore`` provides a compact on-disk container for tabular, fixed-size
numeric data. A ``.cstore`` file stores each column as a contiguous block of
raw bytes; reads use ``numpy.memmap`` plus a parallel C++ gather kernel, so
process memory stays bounded by the size of the materialized output even
when the file on disk is much larger.

The public surface centers on:

* :func:`open` / :func:`store` / :func:`create` / :func:`recreate` /
  :func:`update` -- module-level entry points for reading and writing.
* :func:`compact` -- collapse a multi-record file into a single-record file.
* :func:`info` / :func:`schema` -- introspect a file without reading bodies.
* :class:`ColStoreReader` -- the underlying reader class; indexes return
  lazy views (:class:`ColumnView` for a single column,
  :class:`TableView` for multi-column).
* :class:`ColStoreWriter` -- the underlying streaming writer class.

Package-wide defaults (thread count, ``madvise`` hint, gather backend) live
in :mod:`colstore.config` and can be changed at runtime.
"""

import os as _os


def use_passive_openmp_wait() -> bool:
    """Opt in to ``OMP_WAIT_POLICY=passive`` for OpenMP threads. Returns success.

    The gather kernel runs short, bursty parallel regions interleaved with
    Python. OpenMP's default *active* wait makes idle threads busy-spin between
    regions, which can burn cores. ``passive`` makes them sleep instead.

    This is **opt-in and not called automatically**, because ``OMP_WAIT_POLICY``
    is process-global: it affects every OpenMP runtime in the process (NumPy,
    numba, PyTorch, SciPy, ...), not just colstore. With the per-call thread cap
    in place, colstore's own spinning is already bounded to a handful of
    threads, so most users will not need this.

    Like all OpenMP/BLAS environment variables, it only takes effect if set
    **before** the OpenMP runtime initializes (i.e. before the first import of
    NumPy or the compiled extension). Call it at the very top of your program,
    before importing colstore or NumPy, for it to apply. Returns ``True`` if the
    variable was set, ``False`` if it was already set (and therefore left
    untouched).
    """
    if "OMP_WAIT_POLICY" in _os.environ:
        return False
    _os.environ["OMP_WAIT_POLICY"] = "passive"
    return True


from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("colstore")
except PackageNotFoundError:  # source checkout without an installed dist
    __version__ = "0.0.0+unknown"

# If this machine has been calibrated before, apply the cached gather thread
# cap now. Otherwise the static hardware-derived default from `config` stands.
# Calibration itself never runs implicitly; the user calls `calibrate()` or
# `ensure_calibrated()` explicitly.
from .api import (
    ColStoreInfo,
    compact,
    create,
    info,
    open,
    recreate,
    schema,
    store,
    update,
)
from .autotune import (
    apply_cached_cap_if_present,
    calibrate,
    ensure_calibrated,
)
from .config import (
    get_default_backend,
    get_default_madvise,
    get_gather_thread_cap,
    get_max_workers,
    set_default_backend,
    set_default_madvise,
    set_gather_thread_cap,
    set_max_workers,
)
from .format import FILE_EXTENSION, FormatError
from .kernels import cpp_available, max_threads, numba_available
from .reader import ColStoreReader
from .view import ColumnView, TableView
from .writer import ColStoreWriter

apply_cached_cap_if_present()

__all__ = [
    "FILE_EXTENSION",
    "ColStoreInfo",
    "ColStoreReader",
    "ColStoreWriter",
    "ColumnView",
    "FormatError",
    "TableView",
    "__version__",
    "calibrate",
    "compact",
    "cpp_available",
    "create",
    "ensure_calibrated",
    "get_default_backend",
    "get_default_madvise",
    "get_gather_thread_cap",
    "get_max_workers",
    "info",
    "max_threads",
    "numba_available",
    "open",
    "recreate",
    "schema",
    "set_default_backend",
    "set_default_madvise",
    "set_gather_thread_cap",
    "set_max_workers",
    "store",
    "update",
    "use_passive_openmp_wait",
]
