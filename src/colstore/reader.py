"""The ColStoreReader class: a memory-mapped, columnar, randomly-accessible store.

A ``ColStoreReader`` opens a ``.cstore`` file and exposes its columns through a
NumPy/pandas-like indexing API that returns lazy view objects. Single-string
column selection yields a :class:`ColumnView`; every other shape yields a
:class:`TableView`. The package is positioned as an **I/O library for a
custom binary format**: write a tabular dataset once via
:func:`colstore.store` (one-shot) or :class:`ColStoreWriter` (streaming), then
load arbitrary row/column subsets from disk with bounded process memory.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import mmap
import os
import warnings
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from numpy.typing import NDArray

from . import _numa, config, format, kernels
from .view import ColumnView, TableView

if TYPE_CHECKING:
    import pandas as pd

_MADVISE_FLAGS: dict[str, int] = {
    "normal": getattr(mmap, "MADV_NORMAL", 0),
    "sequential": getattr(mmap, "MADV_SEQUENTIAL", 2),
    "random": getattr(mmap, "MADV_RANDOM", 1),
    "willneed": getattr(mmap, "MADV_WILLNEED", 3),
    "dontneed": getattr(mmap, "MADV_DONTNEED", 4),
}

_USE_DEFAULT_MADVISE = "__default__"


def _dtype_is_native(dtype: np.dtype[Any]) -> bool:
    """Return whether ``dtype`` is in the host's native byte order.

    Single-byte and string-of-bytes kinds carry byteorder ``"|"`` (not
    applicable) and are always safe for a raw byte copy. Multi-byte numeric
    dtypes are native when their byteorder is ``"="`` or matches the host.
    The native-only paths (raw memcpy range copy, the typed gather kernels)
    use this to decide whether a byte-level copy preserves values; non-native
    dtypes fall back to NumPy, which byteswaps during the copy.
    """
    byteorder = dtype.byteorder
    if byteorder in ("=", "|"):
        return True
    return (byteorder == "<") == bool(np.little_endian)


# ---- Parallel contiguous copy ------------------------------------------
#
# A contiguous column read is fundamentally one memcpy from the memmap into
# an owning ndarray. NumPy's copy is single-threaded; on modern multi-core
# systems with multi-channel memory, one core can't saturate the bus, so
# splitting the copy across threads is 2-3x faster on a big read. On a
# memory-bandwidth-bound machine where one core already saturates, the
# threadpool overhead is wasted and the constants below keep us on the
# single-thread path.
#
# These thresholds are intentionally conservative: any read below
# _PARALLEL_COPY_MIN_BYTES (~16 MiB) goes single-threaded, and each
# threadpool worker gets at least _PARALLEL_COPY_BYTES_PER_THREAD of work
# so the ~1 ms threadpool fork cost is amortized over a few ms of memcpy.

_PARALLEL_COPY_MIN_BYTES = 16 * 1024 * 1024
_PARALLEL_COPY_BYTES_PER_THREAD = 16 * 1024 * 1024


def _parallel_contiguous_copy(
    source: NDArray[Any],
    dst_dtype: np.dtype[Any],
    *,
    thread_cap: int,
) -> NDArray[Any]:
    """Copy a contiguous numpy view into a new owning ndarray, optionally in parallel.

    Falls back to a single ``np.array`` copy when:

      * ``thread_cap <= 1`` (caller doesn't want parallelism, e.g. because
        the column threadpool is already saturating cores), OR
      * ``source.nbytes < _PARALLEL_COPY_MIN_BYTES`` (work is too small to
        amortize threadpool fork), OR
      * the per-thread share would be below
        ``_PARALLEL_COPY_BYTES_PER_THREAD`` after dividing by ``thread_cap``.

    Otherwise it spawns up to ``thread_cap`` threads, each doing a slice
    assignment into a preallocated output. NumPy releases the GIL during
    the bulk memcpy of slice assignment, so the chunks actually run
    concurrently on multi-core systems.
    """
    n_bytes = source.nbytes
    if thread_cap <= 1 or n_bytes < _PARALLEL_COPY_MIN_BYTES:
        return np.array(source, dtype=dst_dtype, copy=True)
    n_threads = min(thread_cap, max(1, n_bytes // _PARALLEL_COPY_BYTES_PER_THREAD))
    if n_threads <= 1:
        return np.array(source, dtype=dst_dtype, copy=True)
    n_rows = source.shape[0]
    out: NDArray[Any] = np.empty(n_rows, dtype=dst_dtype)
    chunk = (n_rows + n_threads - 1) // n_threads

    def copy_chunk(start: int, end: int) -> None:
        out[start:end] = source[start:end]

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [
            executor.submit(copy_chunk, i * chunk, min((i + 1) * chunk, n_rows))
            for i in range(n_threads)
        ]
        for future in futures:
            future.result()
    return out


# ---- DataFrame construction --------------------------------------------
#
# ``pd.DataFrame(dict_of_arrays)`` groups columns by dtype and copies
# them into one 2D ``Block`` per dtype group ("consolidation"). On a 1 GB
# / 50-column store of all-float64 data, that's a 1 GB extra allocation
# plus memcpy on top of the dict already owning a 1 GB copy of the data
# -- it dominates ``frame()`` (~700 ms vs ~70 ms for ``dict()`` on the
# same data, measured on 256-core hardware). The optimized path
# constructs the ``BlockManager`` with one ``Block`` per column instead,
# sharing memory with the input arrays and skipping the consolidation
# copy.


def _make_dataframe_no_consolidate(columns: dict[str, NDArray[Any]]) -> pd.DataFrame:
    """Build a pandas DataFrame from a column dict without dtype-block consolidation.

    The returned DataFrame is "fragmented" relative to one built by
    ``pd.DataFrame(columns)``: it has one ``Block`` per column rather
    than one ``Block`` per dtype group. The two are functionally
    identical through the public DataFrame API; pandas consolidates on
    demand for operations that benefit from it. For the
    read-once-and-pass-along workload this method targets, paying the
    consolidation cost eagerly is wasted work.

    Uses the pandas private API ``create_block_manager_from_column_arrays``
    plus ``DataFrame._from_mgr`` (both stable in pandas 2.0+). On any of:

      * ``ImportError`` -- a symbol is gone (e.g. ``_from_mgr`` not
        present on pandas 1.x, or the ``managers`` submodule moved);
      * ``AttributeError`` -- a classmethod or attribute is gone; or
      * ``TypeError`` -- the call signature has shifted (a keyword was
        renamed or removed, ``refs`` semantics changed, etc.);

    the helper falls back to ``pd.DataFrame(columns)`` and emits a
    one-shot ``UserWarning`` so the regression is visible without
    breaking user code. The fallback is functionally identical but
    pays the consolidation copy, so on whole-store materialization
    it's roughly an order of magnitude slower.

    ``ValueError`` is intentionally NOT caught: it almost certainly
    signals a data-validation problem (mismatched shapes, etc.) that
    the fallback path would surface in the same way, and catching it
    would mask the original error.
    """
    import pandas as pd

    if not columns:
        return pd.DataFrame(columns)

    try:
        from pandas import Index, RangeIndex
        from pandas.core.internals.managers import (
            create_block_manager_from_column_arrays,
        )

        arrays = list(columns.values())
        n_rows = arrays[0].shape[0]
        block_manager = create_block_manager_from_column_arrays(
            arrays,
            axes=[Index(list(columns)), RangeIndex(n_rows)],
            consolidate=False,
            refs=[None] * len(arrays),
        )
        return pd.DataFrame._from_mgr(block_manager, axes=block_manager.axes)
    except (ImportError, AttributeError, TypeError) as exc:
        warnings.warn(
            f"colstore.frame() optimized construction unavailable on this "
            f"pandas ({pd.__version__}); falling back to pd.DataFrame(dict). "
            f"The result is functionally identical but slower for whole-store "
            f"materialization. This usually indicates a pandas internal API "
            f"change. Cause: {type(exc).__name__}: {exc}",
            stacklevel=2,
        )
        return pd.DataFrame(columns)


class ColStoreReader:
    """Memory-mapped columnar store with lazy, NumPy-style indexing.

    Opening a store reads its header, creates one ``np.memmap`` per column,
    and applies any requested kernel hints. Reads are performed through
    ``__getitem__``, which returns a lazy view: either a :class:`ColumnView`
    (single-column) or a :class:`TableView` (multi-column). The view
    materializes when one of its ``array`` / ``dict`` / ``recarray`` /
    ``frame`` methods is called.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.cstore`` file produced by :func:`colstore.store` or
        :class:`ColStoreWriter`.
    madvise : str or None, optional
        Kernel access-pattern hint applied to every column memmap. One of
        ``"normal"``, ``"sequential"``, ``"random"``, ``"willneed"``,
        ``"dontneed"``, or ``None``. Defaults to the package-wide setting.
    mlock : bool, optional
        If ``True``, attempt to lock every column's pages in RAM. Failures
        emit a warning rather than raising. Defaults to ``False``.
    backend : str or None, optional
        Gather backend used for fancy-index reads (``"cpp"``, ``"numpy"``,
        or ``"numba"``). ``None`` uses the package-wide default.
    max_workers : int or None, optional
        Override the package-wide thread-pool size for multi-column reads.
        ``None`` uses the global setting (physical core count by default).

    Examples
    --------
    >>> ds = colstore.store(df, "data.cstore")
    >>> ds['price']                      # ColumnView -> array()
    >>> ds[100:200, ['price', 'qty']]    # TableView -> dict / recarray / frame
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        madvise: str | None = _USE_DEFAULT_MADVISE,
        mlock: bool = False,
        backend: str | None = None,
        max_workers: int | None = None,
    ) -> None:
        self._path = Path(path)
        self._manifest, data_offset = format.read_header(self._path)
        self._closed = False

        self._n_rows = int(self._manifest["committed_rows"])
        n_records = int(self._manifest["n_records"])
        self._is_multi_record = n_records > 1

        # Walk the record headers. This validates every record (magic, index,
        # CRC) and naturally catches truncation past the file header by
        # reading past EOF. We do it unconditionally -- the cost is one 32B
        # read per record, microseconds for any plausible record count, and
        # it gives both the single-record and multi-record paths a uniformly
        # validated file before any data read.
        columns_meta = self._manifest["columns"]
        itemsizes = [np.dtype(col["dtype"]).itemsize for col in columns_meta]
        record_starts_rows, record_starts_bytes, n_rows_per_record = format.read_record_index(
            self._path, data_offset, n_records, itemsizes
        )

        # Column dtypes are recorded uniformly so introspection (`dtypes`,
        # `columns`) works the same way in both modes.
        self._column_dtypes: dict[str, np.dtype[Any]] = {
            col["name"]: np.dtype(col["dtype"]) for col in columns_meta
        }

        # ---- Resolve layout: two cases ----
        #
        # (1) Single record (n_records == 1, the post-compaction common case):
        #     build per-column memmaps anchored at the validated record body
        #     offset; fancy-index gathers run against the per-column memmap
        #     just like on a fully contiguous file.
        #
        # (2) Multiple records: per-column data is no longer contiguous; mmap
        #     the whole file as bytes and keep the per-record index for the
        #     reader. Reads compute byte addresses via ``np.searchsorted`` and
        #     feed them to ``_gather.gather_bytes``.
        if self._is_multi_record:
            self._layout: format.ColumnLayout = {}  # not meaningful in this mode
            self._record_starts_rows = record_starts_rows
            self._record_starts_bytes = record_starts_bytes
            self._n_rows_per_record = n_rows_per_record
            self._column_prefix_bytes: dict[str, np.int64] = {}
            prefix = 0
            for col, size in zip(columns_meta, itemsizes, strict=True):
                self._column_prefix_bytes[col["name"]] = np.int64(prefix)
                prefix += size
            # mmap the whole file as bytes; this is the kernel's base pointer
            # for byte-offset gathers.
            self._file_mmap = np.memmap(self._path, dtype=np.uint8, mode="r")
            self._memmaps: dict[str, np.memmap[Any, np.dtype[Any]]] = {}
        else:
            # Single-record fast path. The body starts at the offset the
            # record walk produced (== data_offset + 32, but validated).
            self._layout = format.build_column_layout(
                self._manifest, int(record_starts_bytes[0]), self._n_rows
            )
            self._memmaps = {
                name: np.memmap(
                    self._path,
                    dtype=column_dtype,
                    mode="r",
                    offset=column_offset,
                    shape=(self._n_rows,),
                )
                for name, (column_offset, column_dtype) in self._layout.items()
            }

        if madvise == _USE_DEFAULT_MADVISE:
            madvise = config.get_default_madvise()
        # NUMA policy is applied *before* madvise and any data access so
        # that the policy is in place when pages first fault in. mbind
        # without MPOL_MF_MOVE doesn't migrate already-populated pages.
        self._apply_numa_policy()
        if madvise is not None:
            self._apply_madvise(madvise)
        if mlock:
            self._apply_mlock()

        self._backend = backend or config.get_default_backend()
        self._max_workers_override = max_workers

    # ---- Read-only properties ------------------------------------------

    @property
    def path(self) -> Path:
        """Filesystem path the store was opened from."""
        return self._path

    @property
    def n_rows(self) -> int:
        """Number of rows in every column."""
        return self._n_rows

    @property
    def columns(self) -> list[str]:
        """Column names in on-disk order."""
        return list(self._column_dtypes)

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        """Map of column name to NumPy dtype, in the host's native byte order."""
        return {name: dtype.newbyteorder("=") for name, dtype in self._column_dtypes.items()}

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_rows, n_columns)`` tuple, mirroring ``DataFrame.shape``."""
        return self.n_rows, len(self._column_dtypes)

    @property
    def backend(self) -> str:
        """Effective gather backend on this instance."""
        return self._backend

    @property
    def max_workers(self) -> int:
        """Effective thread-pool size for multi-column reads on this instance."""
        if self._max_workers_override is not None:
            return self._max_workers_override
        return config.get_max_workers()

    # ---- Container protocol --------------------------------------------

    def __len__(self) -> int:
        return self.n_rows

    def __contains__(self, column_name: object) -> bool:
        return isinstance(column_name, str) and column_name in self._column_dtypes

    def __iter__(self) -> Iterator[str]:
        return iter(self.columns)

    @overload
    def __getitem__(self, key: str) -> ColumnView: ...
    @overload
    def __getitem__(self, key: tuple[Any, str]) -> ColumnView: ...
    @overload
    def __getitem__(self, key: int | slice | list[Any] | NDArray[Any]) -> TableView: ...
    @overload
    def __getitem__(self, key: tuple[Any, list[str] | tuple[str, ...]]) -> TableView: ...

    def __getitem__(self, key: Any) -> ColumnView | TableView:
        row_part, column_names, is_single_column = self._parse_key(key)
        if is_single_column:
            return ColumnView(self, row_part, column_names[0])
        return TableView(self, row_part, column_names)

    def __repr__(self) -> str:
        column_preview = self.columns[:5]
        suffix = "..." if len(self._column_dtypes) > len(column_preview) else ""
        return (
            f"ColStoreReader(path={self._path.name!r}, "
            f"shape={self.shape}, columns={column_preview}{suffix})"
        )

    # ---- Lifecycle -----------------------------------------------------

    def close(self) -> None:
        """Release all memmaps. Subsequent reads will fail."""
        if self._closed:
            return
        for memmap_view in self._memmaps.values():
            del memmap_view
        self._memmaps.clear()
        # The multi-record path holds a single whole-file mmap instead of
        # per-column memmaps; drop it too if present.
        if self._is_multi_record and hasattr(self, "_file_mmap"):
            del self._file_mmap
        self._closed = True

    def __enter__(self) -> ColStoreReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # ---- Indexing helpers ----------------------------------------------

    def _parse_key(self, key: Any) -> tuple[Any, list[str], bool]:
        """Split a ``__getitem__`` key into row part, column names, singular flag."""
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError(f"Expected at most 2 elements in indexing tuple; got {len(key)}.")
            row_part, column_part = key
        elif self._looks_like_column_spec(key):
            row_part, column_part = None, key
        else:
            row_part, column_part = key, None

        if column_part is None:
            column_names = list(self._column_dtypes)
            is_single_column = False
        elif isinstance(column_part, str):
            column_names = [column_part]
            is_single_column = True
        elif isinstance(column_part, (list, tuple)):
            column_names = list(column_part)
            is_single_column = False
        else:
            raise IndexError(
                f"Column selector must be a string or list/tuple of strings; "
                f"got {type(column_part).__name__}."
            )

        unknown = [name for name in column_names if name not in self._column_dtypes]
        if unknown:
            raise KeyError(f"Unknown column(s): {unknown}. Available columns: {self.columns}")
        return row_part, column_names, is_single_column

    @staticmethod
    def _looks_like_column_spec(value: Any) -> bool:
        """Heuristic distinguishing column specs from row specs in a single key."""
        if isinstance(value, str):
            return True
        if isinstance(value, (list, tuple)) and value:
            return all(isinstance(item, str) for item in value)
        return False

    # ---- madvise / mlock -----------------------------------------------

    def _apply_madvise(self, advice: str) -> None:
        if advice not in _MADVISE_FLAGS:
            raise ValueError(
                f"Invalid madvise value {advice!r}; expected one of {sorted(_MADVISE_FLAGS)}."
            )
        flag = _MADVISE_FLAGS[advice]
        if self._is_multi_record:
            with contextlib.suppress(AttributeError, OSError):
                self._file_mmap._mmap.madvise(flag)  # type: ignore[attr-defined]
            return
        for memmap_view in self._memmaps.values():
            with contextlib.suppress(AttributeError, OSError):
                memmap_view._mmap.madvise(flag)  # type: ignore[attr-defined]

    def _apply_numa_policy(self) -> None:
        """Apply the configured NUMA policy to all file-backed memmaps.

        Called once at open time, before any access faults pages in.
        The default ``"auto"`` policy applies ``MPOL_INTERLEAVE`` on
        multi-node Linux and is a no-op everywhere else.

        ``"local"`` is the opt-out for low-concurrency workloads where
        forced interleaving costs more than it saves (e.g. a single
        consumer thread reading 1 GB ends up doing remote loads for
        7/8 of the pages on an 8-node host).
        """
        policy = config.get_numa_policy()
        if policy == "local":
            return
        if not _numa.is_available():
            return
        if self._is_multi_record:
            _numa.apply_interleave_to_memmap(self._file_mmap)
            return
        for memmap_view in self._memmaps.values():
            _numa.apply_interleave_to_memmap(memmap_view)

    def _apply_mlock(self) -> None:
        libc_name = ctypes.util.find_library("c")
        if libc_name is None:
            warnings.warn("mlock requested but libc could not be located.", stacklevel=2)
            return
        libc = ctypes.CDLL(libc_name, use_errno=True)

        def lock_view(view: Any) -> None:
            address = view.ctypes.data
            length = view.nbytes
            if libc.mlock(ctypes.c_void_p(address), ctypes.c_size_t(length)) != 0:
                errno = ctypes.get_errno()
                warnings.warn(
                    f"mlock failed (errno={errno}); pages may be paged out "
                    f"under memory pressure.",
                    stacklevel=2,
                )

        if self._is_multi_record:
            lock_view(self._file_mmap)
            return
        for memmap_view in self._memmaps.values():
            lock_view(memmap_view)

    # ---- Gather (called by views) --------------------------------------

    def _gather_one(
        self, column_name: str, row_indexer: Any, thread_cap: int | None = None
    ) -> NDArray[Any]:
        """Read one column with the given row selector; return owning ndarray.

        Output arrays are always native byte order, even though the on-disk
        column is stored little-endian. On a little-endian host this is a
        no-op; on a big-endian host NumPy converts during the copy/gather.

        ``thread_cap`` overrides the per-call thread cap used by both the
        fancy-index gather (OpenMP) and the contiguous parallel copy
        (Python threadpool). ``None`` uses the package default.
        :meth:`_gather_many` passes a divided budget here so concurrent
        column reads do not oversubscribe.
        """
        if self._closed:
            raise ValueError("ColStoreReader is closed.")
        if self._is_multi_record:
            return self._gather_one_multi_record(column_name, row_indexer, thread_cap)
        # Single-record fast path: one per-column memmap; the read is a
        # simple slice / copy / kernel gather against that memmap.
        source = self._memmaps[column_name]
        disk_dtype = self._layout[column_name][1]
        native_dtype = disk_dtype.newbyteorder("=")
        effective_cap = thread_cap if thread_cap is not None else config.get_gather_thread_cap()
        # ``np.array(..., copy=True)`` is typed to return ``NDArray[Any]``;
        # the older ``np.asarray(x).copy()`` chain returns ``Any`` under
        # current numpy stubs, hence the explicit constructor calls.
        if row_indexer is None:
            return _parallel_contiguous_copy(source, native_dtype, thread_cap=effective_cap)
        if isinstance(row_indexer, int):
            return np.atleast_1d(np.array(source[row_indexer], dtype=native_dtype, copy=True))
        if isinstance(row_indexer, slice):
            start, stop, step = row_indexer.indices(self._n_rows)
            if step == 1:
                return _parallel_contiguous_copy(
                    source[start:stop], native_dtype, thread_cap=effective_cap
                )
            # Strided slice: cheap rebuild via numpy is fine; not worth
            # parallelizing the non-contiguous case.
            return np.array(source[row_indexer], dtype=native_dtype, copy=True)
        # Integer ndarray (fancy index): dispatch to chosen backend.
        return kernels.gather(
            source, row_indexer, native_dtype, backend=self._backend, thread_cap=thread_cap
        )

    # ---- Multi-record read path -----------------------------------------

    def _gather_one_multi_record(
        self, column_name: str, row_indexer: Any, thread_cap: int | None
    ) -> NDArray[Any]:
        """Read one column from a file with multiple records.

        Per-pattern dispatch:

        * **Slice** (``None`` / ``slice`` / contiguous range): per-record
          contiguous memcpy via ``np.frombuffer``. Avoids the gather kernel
          and the byte-offset materialization entirely. Measured ~48x faster
          than the generic path on a slice spanning 10 records (0.07 ms vs
          3.4 ms).

        * **Sorted fancy index**: boundary-based partition. ``np.searchsorted``
          on the *record-row boundaries* against the indices array (O(R log K))
          replaces searchsorted on the *indices* against the boundaries
          (O(K log R)). For K=200K, R=100 this saves ~1.3 ms on the
          ~4.5 ms sorted path. Same kernel work afterward.

        * **Unsorted fancy index**: searchsorted + byte-offset gather. This
          is the generic path; the searchsorted is unavoidable when indices
          can land anywhere across records, and it dominates the cost.
          Argsort + sorted-path + reindex is *slower* than this (measured)
          because argsort on K int64 costs more than searchsorted does. The
          escape valve is :func:`colstore.compact` -- collapse to a single
          record and the fast path kicks in.

        For an integer scalar selector, the result is a length-1 ndarray
        matching the contiguous path's ``atleast_1d`` semantics.
        """
        from . import _gather as _cpp_module  # type: ignore[attr-defined]

        disk_dtype = self._column_dtypes[column_name]
        native_dtype = disk_dtype.newbyteorder("=")
        itemsize = disk_dtype.itemsize
        col_prefix = int(self._column_prefix_bytes[column_name])

        # ---- Slice / None / scalar ints all map to a contiguous row range.
        # Handle them with one path that avoids the gather kernel entirely.
        if row_indexer is None:
            return self._read_contiguous_range_multi_record(
                0, self._n_rows, disk_dtype, native_dtype, col_prefix, itemsize
            )
        if isinstance(row_indexer, int):
            # Folding negative indices was already done by the view layer.
            return self._read_contiguous_range_multi_record(
                row_indexer, row_indexer + 1, disk_dtype, native_dtype, col_prefix, itemsize
            )
        if isinstance(row_indexer, slice):
            start, stop, step = row_indexer.indices(self._n_rows)
            if step == 1:
                # The hot case. step != 1 falls through to the fancy-index
                # path; non-unit-step slices are rare and not worth special-
                # casing further (they'd need per-record arange-style picks).
                return self._read_contiguous_range_multi_record(
                    start, stop, disk_dtype, native_dtype, col_prefix, itemsize
                )
            indices = np.arange(start, stop, step, dtype=np.int64)
        else:
            # int ndarray -- already validated and made int64 by the view layer.
            indices = np.asarray(row_indexer, dtype=np.int64)

        n = indices.shape[0]
        if n == 0:
            return np.empty(0, dtype=native_dtype)

        # ---- Fancy-index path. Choose how to bin indices to records.
        record_starts_rows = self._record_starts_rows
        record_starts_bytes = self._record_starts_bytes
        n_rows_per_record = self._n_rows_per_record
        # The raw byte-offset kernel (gather_bytes) copies on-disk bytes
        # verbatim, and the disk is always little-endian -- so its
        # destination must be typed with the DISK dtype, not the native one,
        # or a big-endian host would misinterpret every value. The
        # ``astype(native, copy=False)`` at the return is a no-op on
        # little-endian hosts (the dtypes compare equal) and a byteswapping
        # copy on big-endian ones. The fused native kernel branch is only
        # taken when disk == native, where the distinction vanishes.
        output = np.empty(n, dtype=disk_dtype)
        effective_cap = config.get_gather_thread_cap() if thread_cap is None else max(1, thread_cap)

        # Sortedness check is O(K) but ~100x faster than a searchsorted at
        # K=200K, so the early exit is essentially free in the unsorted case.
        if n > 1 and bool(np.all(indices[1:] >= indices[:-1])):
            # Sorted path. Two-part optimization:
            #
            # (1) Boundary-based partition. ``np.searchsorted`` on the
            #     *record-row boundaries* against the indices array
            #     (O(R log K)) replaces searchsorted on the *indices* against
            #     the boundaries (O(K log R)).
            #
            # (2) Per-record byte_offset arithmetic. For each record-bucket
            #     [lo, hi), all entries share the same record body offset,
            #     so:
            #
            #         byte_offsets[i] = rsb[r] + col_prefix * nrr[r]
            #                                  + (indices[i] - rsr[r]) * itemsize
            #                         = base_for_record_r + indices[i] * itemsize
            #
            #     where base_for_record_r = rsb[r] + col_prefix*nrr[r]
            #                               - rsr[r]*itemsize is a scalar.
            #
            #     This is much cheaper than the generic vectorized form,
            #     which would allocate four K-sized int64 temporaries
            #     (record_id, within_record, two intermediate gathers). At
            #     K=1M the generic form costs ~11ms in allocator/cache
            #     traffic alone; the per-record loop costs ~1ms.
            #
            # This path already avoids the big temporaries and runs ~4-11x
            # faster than the unsorted path (measured), so it is left as-is;
            # the fused native kernel below targets the unsorted case.
            crossings = np.searchsorted(indices, record_starts_rows, side="left")
            byte_offsets = np.empty(n, dtype=np.int64)
            for r in range(crossings.shape[0] - 1):
                lo = int(crossings[r])
                hi = int(crossings[r + 1])
                if hi == lo:
                    continue
                base = (
                    int(record_starts_bytes[r])
                    + col_prefix * int(n_rows_per_record[r])
                    - int(record_starts_rows[r]) * itemsize
                )
                np.multiply(indices[lo:hi], itemsize, out=byte_offsets[lo:hi])
                np.add(byte_offsets[lo:hi], base, out=byte_offsets[lo:hi])
            _cpp_module.gather_bytes(
                self._file_mmap,
                byte_offsets,
                output,
                effective_cap,
                config.resolve_prefetch_distance(self._file_mmap.nbytes, indices_sorted=True),
            )
        elif _dtype_is_native(disk_dtype):
            # Unsorted (or n == 1), native byte order: fused native gather.
            # The kernel bins each index to its record with a branchless
            # binary search and loads in one pass -- no searchsorted, no
            # byte_offsets array. The searchsorted-based binning was measured
            # at ~75-85% of this path's cost; folding it into the kernel (and
            # parallelizing it, which searchsorted cannot be) is the win.
            _cpp_module.gather_multirecord(
                self._file_mmap,
                indices,
                output,
                record_starts_rows,
                record_starts_bytes,
                n_rows_per_record,
                int(col_prefix),
                effective_cap,
                config.resolve_prefetch_distance(self._file_mmap.nbytes, indices_sorted=False),
            )
        else:
            # Unsorted, non-native byte order (big-endian host). The fused
            # kernel does a raw typed load and cannot byteswap, so fall back to
            # the NumPy searchsorted pipeline feeding the raw gather_bytes
            # kernel -- identical to the pre-Stage-2 behavior on this host.
            record_id = np.searchsorted(record_starts_rows, indices, side="right") - 1
            within_record = indices - record_starts_rows[record_id]
            byte_offsets = (
                record_starts_bytes[record_id]
                + col_prefix * n_rows_per_record[record_id]
                + within_record * itemsize
            )
            _cpp_module.gather_bytes(
                self._file_mmap,
                byte_offsets,
                output,
                effective_cap,
                config.resolve_prefetch_distance(self._file_mmap.nbytes, indices_sorted=False),
            )

        return output.astype(native_dtype, copy=False)

    def _read_contiguous_range_multi_record(
        self,
        start: int,
        stop: int,
        disk_dtype: np.dtype[Any],
        native_dtype: np.dtype[Any],
        col_prefix: int,
        itemsize: int,
    ) -> NDArray[Any]:
        """Read rows ``[start, stop)`` for one column across records via memcpy.

        Replaces the gather kernel for contiguous-range reads. A range
        spanning R' records is served by R' contiguous memory copies from
        the file mmap, plus an O(log R) search to locate the first
        overlapping record. No per-element work.

        When the C++ extension is available and the on-disk dtype is in native
        byte order, the per-record copy loop runs entirely in C++
        (:func:`_gather.copy_multirecord_range`): one ``memcpy`` per record,
        record membership found by binary search in the kernel, zero per-record
        Python/NumPy overhead. Non-native dtypes (which need a byteswap during
        the copy) and the no-extension case fall back to the NumPy loop in
        :meth:`_copy_multirecord_range_python`.
        """
        n = stop - start
        output: NDArray[Any] = np.empty(n, dtype=native_dtype)
        if n == 0:
            return output

        if kernels.cpp_available() and _dtype_is_native(disk_dtype):
            from . import _gather as _cpp_module  # type: ignore[attr-defined]

            _cpp_module.copy_multirecord_range(
                self._file_mmap,
                output,
                int(start),
                int(stop),
                self._record_starts_rows,
                self._record_starts_bytes,
                self._n_rows_per_record,
                int(col_prefix),
                int(itemsize),
            )
            return output

        return self._copy_multirecord_range_python(
            start, stop, disk_dtype, col_prefix, itemsize, output
        )

    def _copy_multirecord_range_python(
        self,
        start: int,
        stop: int,
        disk_dtype: np.dtype[Any],
        col_prefix: int,
        itemsize: int,
        output: NDArray[Any],
    ) -> NDArray[Any]:
        """NumPy fallback for :meth:`_read_contiguous_range_multi_record`.

        Used when the C++ extension is unavailable, or when the on-disk dtype
        is non-native (the per-record ``np.frombuffer`` view is byteswapped
        during assignment into the native-order ``output``). ``output`` is
        preallocated by the caller with length ``stop - start``.
        """
        record_starts_rows = self._record_starts_rows
        first_record = int(np.searchsorted(record_starts_rows, start, side="right") - 1)
        # stop-1 is the last row index actually read; same searchsorted finds
        # which record holds it.
        last_record = int(np.searchsorted(record_starts_rows, stop - 1, side="right") - 1)

        # Precompute per-record bounds in vectorized numpy. The Python loop
        # below only does the copy itself -- avoiding per-iter numpy-scalar
        # arithmetic and the implicit int() coercion that np.frombuffer would
        # otherwise do on each call. At R'=100 overlapping records this saves
        # ~10% over recomputing bounds in the loop.
        rec_slice = slice(first_record, last_record + 1)
        rec_row_starts = record_starts_rows[rec_slice]
        rec_n_rows = self._n_rows_per_record[rec_slice]
        rec_body_starts = self._record_starts_bytes[rec_slice]
        # Clip [start, stop) against each record's row range to get the
        # number of rows we read from each, and the byte offset of the first
        # one within that record.
        within_los = np.maximum(start, rec_row_starts) - rec_row_starts
        within_his = np.minimum(stop, rec_row_starts + rec_n_rows) - rec_row_starts
        counts = within_his - within_los  # >0 for every overlapping record
        byte_offsets = rec_body_starts + col_prefix * rec_n_rows + within_los * itemsize
        write_starts = np.empty(counts.shape[0] + 1, dtype=np.int64)
        write_starts[0] = 0
        np.cumsum(counts, out=write_starts[1:])

        # Convert to Python ints once, outside the loop. np.frombuffer takes
        # ints for offset/count; passing numpy scalars works but coerces on
        # every call.
        counts_list = counts.tolist()
        byte_offsets_list = byte_offsets.tolist()
        write_starts_list = write_starts.tolist()
        for i in range(counts.shape[0]):
            count = counts_list[i]
            view = np.frombuffer(
                self._file_mmap, dtype=disk_dtype, count=count, offset=byte_offsets_list[i]
            )
            output[write_starts_list[i] : write_starts_list[i] + count] = view

        return output

    def _gather_many(self, column_names: list[str], row_indexer: Any) -> dict[str, NDArray[Any]]:
        """Read multiple columns in parallel; return ordered dict of owning arrays.

        Multi-column **unsorted fancy** reads of a multi-record store take the
        bin-reuse route first (see :meth:`_gather_many_bin_reuse`): the
        per-index record binning is 87-93% of the fused gather kernel's cost
        on the target hardware and is identical for every column of the read,
        so it is computed once and reused -- measured 1.9-2.5x at realistic
        thread counts, growing with the column count, and it also cuts total
        CPU work (one binning pass instead of C).

        Otherwise, columns are read concurrently on a thread pool, and each
        column's C++ gather may itself use OpenMP threads. To keep the product
        of the two from oversubscribing the cores, the per-column OpenMP cap
        is divided by the number of columns running concurrently. With many
        columns this drives each kernel to a single thread, so parallelism
        comes from the column pool (the regime where that is most efficient);
        with few columns each kernel still gets a meaningful share of the cap.
        """
        bin_reuse = self._gather_many_bin_reuse(column_names, row_indexer)
        if bin_reuse is not None:
            return bin_reuse
        workers = self.max_workers
        if workers <= 1 or len(column_names) <= 1:
            return {name: self._gather_one(name, row_indexer) for name in column_names}
        n_workers = min(workers, len(column_names))
        per_column_cap = max(1, config.get_gather_thread_cap() // n_workers)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                name: executor.submit(self._gather_one, name, row_indexer, per_column_cap)
                for name in column_names
            }
            return {name: futures[name].result() for name in column_names}

    def _gather_many_bin_reuse(
        self, column_names: list[str], row_indexer: Any
    ) -> dict[str, NDArray[Any]] | None:
        """Bin-reuse route for multi-column unsorted fancy reads, or ``None``.

        Taken when the store is multi-record, the selector is a fancy index
        array that is unsorted, and at least two requested columns are
        native-byte-order (the bins kernels do raw typed loads). The first
        native column runs ``gather_multirecord_bins`` (the Stage-2 kernel
        plus an ``int32`` bins output); the rest run
        ``gather_multirecord_withbins``, where the per-element record is a
        sequential bins read instead of a branchless binary search. Columns
        run sequentially, each at the full thread cap, OpenMP-parallel over
        indices: measured on the target hardware this beats both the
        column-pool shape and a fully fused C-column kernel (whose per-thread
        C concurrent load streams collapse aggregate throughput at realistic
        thread counts). The sortedness check and prefetch resolution are also
        amortized across the read instead of per column.

        Sorted selectors stay on the per-column boundary-partition path
        (already load-bound; nothing to amortize), and non-native columns of
        a mixed read fall back to :meth:`_gather_one` individually.
        """
        if not self._is_multi_record or len(column_names) <= 1:
            return None
        if not isinstance(row_indexer, np.ndarray):
            return None
        indices = np.asarray(row_indexer, dtype=np.int64)
        n = indices.shape[0]
        if n <= 1:
            return None
        if bool(np.all(indices[1:] >= indices[:-1])):
            return None  # sorted: per-column path is already load-bound
        n_records = int(self._record_starts_bytes.shape[0])
        if n_records > np.iinfo(np.int32).max:
            return None  # bins are int32; unreachable in practice, cheap guard
        native_names = [
            name for name in column_names if _dtype_is_native(self._column_dtypes[name])
        ]
        if len(native_names) <= 1:
            return None

        from . import _gather as _cpp_module  # type: ignore[attr-defined]

        effective_cap = config.get_gather_thread_cap()
        prefetch = config.resolve_prefetch_distance(self._file_mmap.nbytes, indices_sorted=False)
        bins = np.empty(n, dtype=np.int32)
        gathered: dict[str, NDArray[Any]] = {}
        for position, name in enumerate(native_names):
            output = np.empty(n, dtype=self._column_dtypes[name].newbyteorder("="))
            kernel = (
                _cpp_module.gather_multirecord_bins
                if position == 0
                else _cpp_module.gather_multirecord_withbins
            )
            kernel(
                self._file_mmap,
                indices,
                output,
                bins,
                self._record_starts_rows,
                self._record_starts_bytes,
                self._n_rows_per_record,
                int(self._column_prefix_bytes[name]),
                effective_cap,
                prefetch,
            )
            gathered[name] = output
        return {
            name: gathered[name] if name in gathered else self._gather_one(name, row_indexer)
            for name in column_names
        }

    # ---- Whole-store materialization shortcuts -------------------------
    #
    # Equivalent to ``self[:].dict()`` / ``.recarray()`` / ``.frame()`` but
    # skip the intermediate ``TableView`` construction. The common
    # "open and convert" idiom -- ``colstore.open(path).dict()`` -- doesn't
    # need a row indexer or the lazy-view machinery, and these methods
    # make that path direct. Placed at the bottom of the class so the
    # method ``dict`` does not shadow the builtin ``dict`` in the
    # annotation scope of earlier methods (mypy resolves annotations in
    # declaration order against the class namespace).

    def dict(self) -> dict[str, NDArray[Any]]:
        """Materialize the whole store as a dict mapping column name to ndarray.

        Returns
        -------
        dict[str, numpy.ndarray]
            Owning arrays in on-disk column order; each column's stored
            dtype is preserved (native byte order).
        """
        if self._closed:
            raise ValueError("ColStoreReader is closed.")
        return self._gather_many(list(self._column_dtypes), None)

    def recarray(self) -> NDArray[Any]:
        """Materialize the whole store as a structured ndarray.

        Returns
        -------
        numpy.ndarray
            Structured 1D array with one field per column. ``result[name]``
            returns the column.
        """
        column_data = self.dict()
        record_dtype = np.dtype([(name, column_data[name].dtype) for name in self._column_dtypes])
        record_array = np.empty(self._n_rows, dtype=record_dtype)
        for name in self._column_dtypes:
            record_array[name] = column_data[name]
        return record_array

    def frame(self) -> pd.DataFrame:
        """Materialize the whole store as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            Columns in on-disk order with their stored dtypes preserved.
            The frame skips dtype-block consolidation (one ``Block`` per
            column) -- see :func:`_make_dataframe_no_consolidate` for
            rationale and details.
        """
        return _make_dataframe_no_consolidate(self.dict())
