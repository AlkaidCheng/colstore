"""Memory-mapped columnar binary format for fast random-access I/O.

``colstore`` provides a compact on-disk container for tabular, fixed-size
numeric data. A ``.cstore`` file stores each column as a contiguous block of
raw bytes; reads use ``numpy.memmap`` plus a parallel C++ gather kernel, so
process memory stays bounded by the materialized output even when the file
is much larger.

Public surface:

* :func:`open` / :func:`store` / :func:`create` / :func:`recreate` /
  :func:`update` -- module-level entry points for reading and writing.
* :func:`compact` -- collapse a multi-record file into a single-record file.
* :func:`info` / :func:`schema` -- introspect a file without reading bodies.
* :class:`ColStoreReader` / :class:`ColStoreWriter` -- the underlying
  classes; reader indexing returns lazy views (:class:`ColumnView`,
  :class:`TableView`).

Package-wide defaults (thread count, ``madvise`` hint, gather backend) live
in :mod:`colstore.config` and can be changed at runtime.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("colstore")
except _PackageNotFoundError:  # source checkout without an installed dist
    __version__ = "0.0.0+unknown"

# If this machine has been calibrated before, apply the cached gather thread
# cap now. Otherwise the static hardware-derived default from `config` stands.
# Calibration itself never runs implicitly; the user calls `calibrate()` or
# `ensure_calibrated()` explicitly.
from . import autotune, interop, profiling, testing
from ._query import QueryError, col
from .api import (
    ColStoreInfo,
    compact,
    concat,
    convert,
    create,
    from_feather,
    from_hdf,
    from_json,
    from_npz,
    from_parquet,
    info,
    open,
    recreate,
    saveas,
    schema,
    store,
    update,
)
from .autotune import (
    calibrate,
    ensure_calibrated,
)
from .config import (
    get_convert_auto_workers,
    get_default_backend,
    get_default_madvise,
    get_gather_thread_cap,
    get_max_workers,
    set_convert_auto_workers,
    set_default_backend,
    set_default_madvise,
    set_gather_thread_cap,
    set_max_workers,
    use_passive_openmp_wait,
)
from .dataset import ColStoreDataset
from .format import FormatError
from .frame import ColStoreFrame
from .interop.root import from_root, to_root
from .kernels import cpp_available, max_threads
from .reader import ColStoreReader
from .shards import Appender, append, appender
from .view import ColumnView, TableView
from .writer import ColStoreWriter

autotune.apply_cached_cap()

__all__ = [
    "Appender",
    "ColStoreDataset",
    "ColStoreFrame",
    "ColStoreInfo",
    "ColStoreReader",
    "ColStoreWriter",
    "ColumnView",
    "FormatError",
    "QueryError",
    "TableView",
    "__version__",
    "append",
    "appender",
    "calibrate",
    "col",
    "compact",
    "concat",
    "convert",
    "cpp_available",
    "create",
    "ensure_calibrated",
    "from_feather",
    "from_hdf",
    "from_json",
    "from_npz",
    "from_parquet",
    "from_root",
    "get_convert_auto_workers",
    "get_default_backend",
    "get_default_madvise",
    "get_gather_thread_cap",
    "get_max_workers",
    "info",
    "interop",
    "max_threads",
    "open",
    "profiling",
    "recreate",
    "saveas",
    "schema",
    "set_convert_auto_workers",
    "set_default_backend",
    "set_default_madvise",
    "set_gather_thread_cap",
    "set_max_workers",
    "store",
    "testing",
    "to_root",
    "update",
    "use_passive_openmp_wait",
]
