"""Memory-mapped columnar binary format for fast random-access I/O.

``colstore`` provides a compact on-disk container for tabular, fixed-size
numeric data. A ``.cstore`` file stores each column as a contiguous block of
raw bytes preceded by a small JSON manifest. Reads use ``numpy.memmap`` plus
a parallel C++ gather kernel, so process memory stays bounded by the size of
the materialized output even when the file on disk is much larger.

The package centers on:

* :class:`ColStore` — opens a ``.cstore`` file and exposes NumPy/pandas-style
  indexing returning lazy views.
* :class:`ColumnView` — lazy single-column view produced by ``ds['col']``;
  materializes with :meth:`ColumnView.to_array`.
* :class:`TableView` — lazy multi-column view; materializes with
  :meth:`TableView.to_dict`, :meth:`TableView.to_record`, or
  :meth:`TableView.to_dataframe`.

Package-wide defaults (thread count, ``madvise`` hint, gather backend) live
in :mod:`colstore.config` and can be changed at runtime.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("colstore")
except PackageNotFoundError:  # source checkout without an installed dist
    __version__ = "0.0.0+unknown"

from .config import (
    get_default_backend,
    get_default_madvise,
    get_max_workers,
    set_default_backend,
    set_default_madvise,
    set_max_workers,
)
from .format import FILE_EXTENSION, FormatError
from .kernels import cpp_available, max_threads, numba_available
from .store import ColStore
from .view import ColumnView, TableView

__all__ = [
    "FILE_EXTENSION",
    "ColStore",
    "ColumnView",
    "FormatError",
    "TableView",
    "__version__",
    "cpp_available",
    "get_default_backend",
    "get_default_madvise",
    "get_max_workers",
    "max_threads",
    "numba_available",
    "set_default_backend",
    "set_default_madvise",
    "set_max_workers",
]
