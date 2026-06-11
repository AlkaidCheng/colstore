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


# Sampled-rejection sortedness test: probe this many evenly spaced adjacent
# pairs before committing to the full O(K) pass. Sampling is skipped below
# the size threshold, where the full pass is cheaper than the sampler's
# fixed overhead (threshold set conservatively above the measured
# crossover; see docs/optimization_series.md).
_SORTEDNESS_SAMPLE_FRACTIONS = np.linspace(0.0, 1.0, 16)
_SORTEDNESS_SAMPLE_MIN_SIZE = 32768

# Record-base precompute gate for the irregular multi-column route: build a
# per-column record_base array (an O(R) vectorized pass plus an R-element
# allocation) only when the read is large enough to amortize it -- the
# kernel-side saving is per element, so the ratio of indices to records is
# the deciding quantity. Below the gate the generic withbins kernel runs
# unchanged. The constant doubles as the benchmark's baseline seam.
_RBASE_MIN_INDICES_PER_RECORD = 1.0


def _indices_are_sorted(indices: NDArray[np.int64]) -> bool:
    """Non-decreasing test with a cheap sampled rejection pass first.

    Semantics are identical to ``bool(np.all(indices[1:] >= indices[:-1]))``,
    including ``True`` for lengths 0 and 1. The sampling pass is used only to
    *reject* sortedness, never to prove it -- any sampled descent makes the
    array definitely unsorted, while an all-ascending sample still falls
    through to the full pass. Correctness is therefore unconditional; the
    sampling only changes the cost split between sorted and unsorted
    selectors (see docs/optimization_series.md).
    """
    n = indices.shape[0]
    if n <= 1:
        return True
    if n >= _SORTEDNESS_SAMPLE_MIN_SIZE:
        positions = (_SORTEDNESS_SAMPLE_FRACTIONS * (n - 2)).astype(np.int64)
        if bool(np.any(indices[positions + 1] < indices[positions])):
            return False
    return bool(np.all(indices[1:] >= indices[:-1]))


# ---- Parallel contiguous copy ------------------------------------------
#
# A contiguous column read is one memcpy from the memmap into an owning
# ndarray. NumPy's copy is single-threaded; where one core cannot saturate
# the memory bus, splitting the copy across threads wins on big reads. The
# conservative thresholds below keep small reads single-threaded: any read
# below _PARALLEL_COPY_MIN_BYTES goes single-threaded, and each worker gets
# at least _PARALLEL_COPY_BYTES_PER_THREAD of work so the threadpool fork
# cost is amortized.

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


def _make_dataframe_no_consolidate(columns: dict[str, NDArray[Any]]) -> pd.DataFrame:
    """Build a pandas DataFrame from a column dict without dtype-block consolidation.

    The result is "fragmented" relative to ``pd.DataFrame(columns)``: one
    ``Block`` per column rather than per dtype group. The two are
    functionally identical through the public DataFrame API (pandas
    consolidates on demand); for the read-once-and-pass-along workload
    this targets, eager consolidation is wasted work.

    Uses the pandas private API ``create_block_manager_from_column_arrays``
    plus ``DataFrame._from_mgr`` (both stable in pandas 2.0+). On
    ``ImportError`` (symbol gone or moved), ``AttributeError`` (classmethod
    or attribute gone), or ``TypeError`` (call signature shifted), the
    helper falls back to ``pd.DataFrame(columns)`` -- functionally
    identical but roughly an order of magnitude slower on whole-store
    materialization -- and emits a one-shot ``UserWarning`` so the
    regression is visible without breaking user code. ``ValueError`` is
    intentionally NOT caught: it almost certainly signals a
    data-validation problem the fallback would surface identically, and
    catching it would mask the original error.
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
        or ``"numba"``). ``None`` uses the package-wide default. Applies to
        single-record stores; multi-record stores require the compiled C++
        extension and always use it for fancy-index reads.
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
            # Uniform-record layout (all records the same row count, constant
            # body stride, final record possibly partial), detected lazily on
            # first fancy read and cached -- see _uniform_record_layout.
            self._uniform_layout_cache: tuple[int, int, int, int] | None = None
            self._uniform_layout_known = False
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
        """Effective gather backend on this instance.

        Governs single-record fancy-index reads; multi-record fancy reads
        always use the C++ extension (see the ``backend`` parameter note).
        """
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

        Output arrays are always native byte order (the on-disk column is
        little-endian; big-endian hosts convert during the copy/gather).
        ``thread_cap`` overrides the per-call cap for both the fancy-index
        gather (OpenMP) and the contiguous parallel copy (Python
        threadpool); ``None`` uses the package default, and
        :meth:`_gather_many` passes a divided budget so concurrent column
        reads do not oversubscribe.
        """
        if self._closed:
            raise ValueError("ColStoreReader is closed.")
        if (
            isinstance(row_indexer, np.ndarray)
            and row_indexer.dtype == np.bool_
            and not self._is_multi_record
        ):
            # Single-record stores keep the flatnonzero -> fancy path: the
            # backend parameter's documented contract governs single-record
            # fancy reads, and lowering here preserves it exactly.
            row_indexer = np.flatnonzero(row_indexer)
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

    # ---- Zero-copy read path --------------------------------------------

    def _view_one(self, column_name: str, row_indexer: Any) -> NDArray[Any]:
        """Zero-copy read of one column, or raise if a copy is unavoidable.

        Returns a READ-ONLY ndarray view backed by the column's open
        memmap; no bytes are copied. Supported exactly when the store is
        single-record (``colstore.compact`` produces these), the dtype is
        in native byte order (a view cannot byteswap), and the selector is
        ``None``, an int, or a slice of any step. Fancy/boolean selectors
        require a gather and therefore a copy; multi-record logical
        columns are interleaved on disk and cannot be viewed contiguously.

        Lifetime: the view holds a reference to the underlying mapping
        (via ``.base``), so it remains valid after :meth:`close` -- at the
        cost of keeping the file mapped until the last view is
        garbage-collected.
        """
        if self._closed:
            raise ValueError("ColStoreReader is closed.")
        if self._is_multi_record:
            raise ValueError(
                "Zero-copy reads require a single-record store: multi-record logical "
                "columns are interleaved on disk. Collapse the store with "
                "colstore.compact() first, or use copy=True."
            )
        disk_dtype = self._layout[column_name][1]
        if not _dtype_is_native(disk_dtype):
            raise ValueError(
                f"Zero-copy reads require native byte order; column {column_name!r} "
                f"has dtype {disk_dtype} (a view cannot byteswap). Use copy=True."
            )
        if isinstance(row_indexer, np.ndarray):
            raise ValueError(
                "Zero-copy reads support only contiguous/strided selectors (None, int, "
                "or slice); fancy and boolean selectors require a gather, which copies. "
                "Use copy=True."
            )
        source = self._memmaps[column_name]
        if row_indexer is None:
            selected = source[:]
        elif isinstance(row_indexer, int):
            # Length-1 view, matching the copying path's atleast_1d shape.
            selected = source[row_indexer : row_indexer + 1]
        elif isinstance(row_indexer, slice):
            selected = source[row_indexer]
        else:  # pragma: no cover - the view layer normalizes to the above
            raise TypeError(f"Unsupported row indexer for zero-copy read: {row_indexer!r}")
        # Re-class the np.memmap slice as a plain ndarray view: same buffer,
        # same read-only flags (the memmap is mode="r"), but without the
        # memmap subclass surface; ``.base`` keeps the mapping alive.
        return selected.view(np.ndarray)

    def _view_many(self, column_names: list[str], row_indexer: Any) -> dict[str, NDArray[Any]]:
        """Zero-copy :meth:`_view_one` over multiple columns (all-or-nothing)."""
        return {name: self._view_one(name, row_indexer) for name in column_names}

    # ---- Multi-record read path -----------------------------------------

    def _detect_uniform_record_layout(self) -> tuple[int, int, int, int] | None:
        """Detect a uniform multi-record layout, or ``None``.

        Uniform means: every record except possibly the last has the same
        row count, the last record is no larger, and the record bodies sit
        at a constant byte stride (which the packed format implies for equal
        row counts, but is verified numerically rather than derived from
        format internals). Returns ``(rows_per_record, record_stride_bytes,
        first_body_offset, last_record_rows)`` for the arithmetic-binning
        kernel, whose record bin is ``idx // rows_per_record`` -- exact for
        every index precisely under these conditions. O(R) numpy passes,
        run once per reader (see :meth:`_uniform_record_layout`).
        """
        nrr = self._n_rows_per_record
        rsb = self._record_starts_bytes
        rows = int(nrr[0])
        if rows <= 0:
            return None
        if not bool(np.all(nrr[:-1] == rows)):
            return None
        last_rows = int(nrr[-1])
        if last_rows > rows or last_rows <= 0:
            return None
        stride = int(rsb[1] - rsb[0])
        if not bool(np.all(np.diff(rsb) == stride)):
            return None
        return rows, stride, int(rsb[0]), last_rows

    def _uniform_record_layout(self) -> tuple[int, int, int, int] | None:
        """Cached :meth:`_detect_uniform_record_layout`."""
        if not self._uniform_layout_known:
            self._uniform_layout_cache = self._detect_uniform_record_layout()
            self._uniform_layout_known = True
        return self._uniform_layout_cache

    def _gather_one_multi_record(
        self, column_name: str, row_indexer: Any, thread_cap: int | None
    ) -> NDArray[Any]:
        """Read one column from a file with multiple records.

        Per-pattern dispatch (measurements and rejected alternatives in
        docs/optimization_series.md):

        * **Boolean mask** at/above the density gate, native dtype:
          mask-native kernel (``gather_multirecord_mask``). Below the gate
          or non-native: lower to ``np.flatnonzero`` and continue below.

        * **Slice** (``None`` / ``slice`` / contiguous range): per-record
          contiguous memcpy; no gather kernel, no byte-offset
          materialization.

        * **Strided slice** (``step != 1``, native dtype): native strided
          walk kernel (``gather_multirecord_strided``). Non-native dtypes
          fall back to ``np.arange`` + the fancy path below.

        * **Sorted fancy index**: native linear-walk kernel
          (``gather_multirecord_sorted``); the NumPy boundary-partition
          pipeline survives only as the non-native (big-endian host)
          fallback.

        * **Unsorted fancy index**: fused native kernel (arithmetic binning
          on uniform layouts, branchless search otherwise); the
          searchsorted + ``gather_bytes`` pipeline survives only as the
          non-native fallback. The escape valve for unsorted reads is
          :func:`colstore.compact` -- collapse to a single record and the
          single-record fast path applies.

        For an integer scalar selector, the result is a length-1 ndarray
        matching the contiguous path's ``atleast_1d`` semantics.
        """
        from . import _gather as _cpp_module  # type: ignore[attr-defined]

        disk_dtype = self._column_dtypes[column_name]
        native_dtype = disk_dtype.newbyteorder("=")
        itemsize = disk_dtype.itemsize
        col_prefix = int(self._column_prefix_bytes[column_name])

        # ---- Boolean mask: mask-native kernel where it pays, else lower to
        # indices and continue into the fancy paths below.
        if isinstance(row_indexer, np.ndarray) and row_indexer.dtype == np.bool_:
            mask = row_indexer
            selected = int(np.count_nonzero(mask))
            if (
                _dtype_is_native(disk_dtype)
                and self._n_rows > 0
                and selected / self._n_rows >= config.resolve_mask_density_gate()
            ):
                output = np.empty(selected, dtype=disk_dtype)
                if selected:
                    effective_cap = (
                        config.get_gather_thread_cap() if thread_cap is None else max(1, thread_cap)
                    )
                    _cpp_module.gather_multirecord_mask(
                        self._file_mmap,
                        mask,
                        output,
                        self._record_starts_rows,
                        self._record_starts_bytes,
                        self._n_rows_per_record,
                        col_prefix,
                        effective_cap,
                        config.resolve_prefetch_distance(
                            self._file_mmap.nbytes, indices_sorted=True
                        ),
                    )
                return output.astype(native_dtype, copy=False)
            row_indexer = np.flatnonzero(mask)

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
                # The hot case: one memcpy per overlapping record.
                return self._read_contiguous_range_multi_record(
                    start, stop, disk_dtype, native_dtype, col_prefix, itemsize
                )
            if _dtype_is_native(disk_dtype):
                # Non-unit step, native byte order: strided walk kernel; no
                # index array, no sortedness pass (negative steps included).
                return self._read_strided_range_multi_record(
                    start, stop, step, disk_dtype, native_dtype, col_prefix, thread_cap
                )
            # Non-native (big-endian host) fallback: materialize the indices
            # and take the generic fancy path below.
            indices = np.arange(start, stop, step, dtype=np.int64)
        else:
            # int ndarray -- already validated, made int64, and made
            # C-contiguous by the view layer; ascontiguousarray here is a
            # free no-op backstop for internal callers (the kernels require
            # contiguity and validate it).
            indices = np.ascontiguousarray(row_indexer, dtype=np.int64)

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

        # Sortedness gate: sampled rejection short-circuits the full O(K)
        # pass for random unsorted selectors; sorted selectors still get the
        # full proof (see _indices_are_sorted). The check stays serial, so
        # its share of the read grows with the kernel's thread count.
        if n > 1 and _indices_are_sorted(indices):
            if _dtype_is_native(disk_dtype):
                # Native byte order: linear-walk kernel (see gather.hpp).
                _cpp_module.gather_multirecord_sorted(
                    self._file_mmap,
                    indices,
                    output,
                    record_starts_rows,
                    record_starts_bytes,
                    n_rows_per_record,
                    int(col_prefix),
                    effective_cap,
                    config.resolve_prefetch_distance(self._file_mmap.nbytes, indices_sorted=True),
                )
                return output.astype(native_dtype, copy=False)
            # Non-native (big-endian host) fallback: NumPy boundary-partition
            # pipeline. ``np.searchsorted`` partitions on the *record-row
            # boundaries* against the indices (O(R log K), not O(K log R));
            # within each record-bucket [lo, hi) all entries share one
            # record body, so
            #
            #     byte_offsets[i] = rsb[r] + col_prefix * nrr[r]
            #                              + (indices[i] - rsr[r]) * itemsize
            #                     = base_for_record_r + indices[i] * itemsize
            #
            # with base_for_record_r = rsb[r] + col_prefix*nrr[r]
            # - rsr[r]*itemsize a scalar, avoiding K-sized int64 temporaries.
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
            # Uniform-record layouts get arithmetic binning
            # (gather_multirecord_uniform); irregular layouts get the
            # branchless-search kernel (see gather.hpp).
            uniform = self._uniform_record_layout()
            if uniform is not None:
                rows_per_record, record_stride, first_body, last_rows = uniform
                _cpp_module.gather_multirecord_uniform(
                    self._file_mmap,
                    indices,
                    output,
                    rows_per_record,
                    record_stride,
                    first_body,
                    int(self._record_starts_bytes.shape[0]),
                    last_rows,
                    int(col_prefix),
                    effective_cap,
                    config.resolve_prefetch_distance(self._file_mmap.nbytes, indices_sorted=False),
                )
            else:
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
            # kernel does a raw typed load and cannot byteswap, so fall back
            # to the NumPy searchsorted pipeline feeding the raw gather_bytes
            # kernel.
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

    def _read_strided_range_multi_record(
        self,
        start: int,
        stop: int,
        step: int,
        disk_dtype: np.dtype[Any],
        native_dtype: np.dtype[Any],
        col_prefix: int,
        thread_cap: int | None,
    ) -> NDArray[Any]:
        """Read rows ``start, start+step, ...`` (slice semantics) for one column.

        Serves multi-record slices with ``step != 1`` via the native strided
        walk kernel, with no index array and no sortedness check. Requires a
        native-byte-order disk dtype (the caller gates; the kernel does raw
        typed loads).

        Prefetch is resolved with ``indices_sorted=True`` for both step
        directions: the calibrated regimes classify the *access stream*, and a
        strided walk is monotone (ascending or descending) with the same
        record-local linearity the sorted gather has.
        """
        from . import _gather as _cpp_module  # type: ignore[attr-defined]

        n = len(range(start, stop, step))
        output = np.empty(n, dtype=disk_dtype)
        if n == 0:
            return output.astype(native_dtype, copy=False)
        effective_cap = config.get_gather_thread_cap() if thread_cap is None else max(1, thread_cap)
        _cpp_module.gather_multirecord_strided(
            self._file_mmap,
            output,
            start,
            stop,
            step,
            self._record_starts_rows,
            self._record_starts_bytes,
            self._n_rows_per_record,
            int(col_prefix),
            effective_cap,
            config.resolve_prefetch_distance(self._file_mmap.nbytes, indices_sorted=True),
        )
        return output.astype(native_dtype, copy=False)

    def _gather_many(self, column_names: list[str], row_indexer: Any) -> dict[str, NDArray[Any]]:
        """Read multiple columns in parallel; return ordered dict of owning arrays.

        Multi-column **boolean-mask** reads try the mask-native route first
        (:meth:`_gather_many_mask`); multi-column **unsorted fancy** reads
        of a multi-record store take the bin-reuse route
        (:meth:`_gather_many_bin_reuse`), which computes the per-index
        record binning once and reuses it for every column.

        Otherwise, columns are read concurrently on a thread pool, and each
        column's C++ gather may itself use OpenMP threads. To keep the product
        of the two from oversubscribing the cores, the per-column OpenMP cap
        is divided by the number of columns running concurrently. With many
        columns this drives each kernel to a single thread, so parallelism
        comes from the column pool (the regime where that is most efficient);
        with few columns each kernel still gets a meaningful share of the cap.
        """
        mask_route = self._gather_many_mask(column_names, row_indexer)
        if mask_route is not None:
            return mask_route
        if isinstance(row_indexer, np.ndarray) and row_indexer.dtype == np.bool_:
            # Mask route declined (single-record store, sparse mask, or
            # too few native columns): lower once and use the fancy paths.
            row_indexer = np.flatnonzero(row_indexer)
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

    def _gather_many_mask(
        self, column_names: list[str], row_indexer: Any
    ) -> dict[str, NDArray[Any]] | None:
        """Mask-native route for multi-column boolean-mask reads, or ``None``.

        Taken when the store is multi-record, the selector is a boolean
        mask at or above the density gate
        (``config.resolve_mask_density_gate``: per-host calibrated, or the
        compiled default), and at least two requested columns are
        native-byte-order. Each native column runs the mask kernel
        (sequential columns at the full OMP cap, like the bins route); the
        selected count is computed once and no index array is built for the
        native columns. Non-native columns (and the decline cases) fall
        back to lowered indices in the caller.
        """
        if not (isinstance(row_indexer, np.ndarray) and row_indexer.dtype == np.bool_):
            return None
        if not self._is_multi_record or self._n_rows == 0:
            return None
        mask = row_indexer
        selected = int(np.count_nonzero(mask))
        if selected / self._n_rows < config.resolve_mask_density_gate():
            return None
        native_names = [
            name for name in column_names if _dtype_is_native(self._column_dtypes[name])
        ]
        if len(native_names) < 2:
            return None
        from . import _gather as _cpp_module  # type: ignore[attr-defined]

        effective_cap = config.get_gather_thread_cap()
        prefetch = config.resolve_prefetch_distance(self._file_mmap.nbytes, indices_sorted=True)
        gathered: dict[str, NDArray[Any]] = {}
        for name in native_names:
            disk_dtype = self._column_dtypes[name]
            output = np.empty(selected, dtype=disk_dtype)
            if selected:
                _cpp_module.gather_multirecord_mask(
                    self._file_mmap,
                    mask,
                    output,
                    self._record_starts_rows,
                    self._record_starts_bytes,
                    self._n_rows_per_record,
                    int(self._column_prefix_bytes[name]),
                    effective_cap,
                    prefetch,
                )
            gathered[name] = output.astype(disk_dtype.newbyteorder("="), copy=False)
        if len(gathered) < len(column_names):
            indices = np.flatnonzero(mask)
            for name in column_names:
                if name not in gathered:
                    gathered[name] = self._gather_one(name, indices)
        return {name: gathered[name] for name in column_names}

    def _gather_many_bin_reuse(
        self, column_names: list[str], row_indexer: Any
    ) -> dict[str, NDArray[Any]] | None:
        """Bin-reuse route for multi-column unsorted fancy reads, or ``None``.

        Taken when the store is multi-record, the selector is a fancy index
        array that is unsorted, and at least two requested columns are
        native-byte-order (the bins kernels do raw typed loads). The first
        native column runs ``gather_multirecord_bins``; the rest reuse the
        bins via the withbins kernels. Columns run sequentially, each at the
        full thread cap, OpenMP-parallel over indices -- the shape that won
        on the deployment hardware over both the column-pool shape and a
        fully fused C-column kernel (see docs/optimization_series.md). The
        sortedness check and prefetch resolution are amortized across the
        read instead of per column.

        Sorted selectors decline the route and run the native sorted walk
        kernel per column: the walk's record binning is a cursor advance
        rather than a per-element search, so there is nothing to amortize
        across columns. Non-native columns of a mixed read fall back to
        :meth:`_gather_one` individually.

        In the parallel regime the route also declines when the concurrent
        column-pool fallback would field strictly more parallel threads: the
        gather is bandwidth-bound, so the fallback's extra memory-level
        parallelism beats the work this route saves. See the gate below.
        """
        if not self._is_multi_record or len(column_names) <= 1:
            return None
        if not isinstance(row_indexer, np.ndarray):
            return None
        indices = np.ascontiguousarray(row_indexer, dtype=np.int64)
        n = indices.shape[0]
        if n <= 1:
            return None
        if _indices_are_sorted(indices):
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
        # Parallel-regime routing gate. The bins kernels do less total work
        # than a per-column gather (the record search is computed once and
        # reused), but this route runs the columns sequentially, leaning on a
        # single kernel's intra-column OpenMP for all its parallelism. Thread
        # assignment is work-proportional (resolve_thread_count), so at a
        # moderate index count one kernel claims only a fraction of the cap,
        # while the concurrent column-pool fallback fields one resolved width
        # per column at once -- more independent memory streams on a read that
        # is bandwidth-bound, not compute-bound. When the fallback would field
        # strictly more parallel threads than this route, its extra memory-level
        # parallelism outweighs the work saved here, so decline and let the
        # caller take it. Gated on the parallel regime only: in the serial
        # regime every resolved width is one, the comparison is moot, and the
        # existing routing contracts (all below PARALLEL_THRESHOLD) are
        # untouched.
        sequential_width = _cpp_module.resolve_thread_count(n, effective_cap)
        if sequential_width > 1:
            n_workers = min(self.max_workers, len(column_names))
            per_column_cap = max(1, effective_cap // n_workers)
            concurrent_width = min(
                _cpp_module.max_threads(),
                n_workers * _cpp_module.resolve_thread_count(n, per_column_cap),
            )
            if concurrent_width > sequential_width:
                return None
        prefetch = config.resolve_prefetch_distance(self._file_mmap.nbytes, indices_sorted=False)
        gathered: dict[str, NDArray[Any]] = {}
        uniform = self._uniform_record_layout()
        if uniform is not None:
            # Uniform-record file: same shape as the generic bins route
            # (first column fills bins, the rest read them), with arithmetic
            # binning in place of the binary search.
            rows_per_record, record_stride, first_body, last_rows = uniform
            bins = np.empty(n, dtype=np.int32)
            for position, name in enumerate(native_names):
                output = np.empty(n, dtype=self._column_dtypes[name].newbyteorder("="))
                uniform_kernel = (
                    _cpp_module.gather_multirecord_uniform_bins
                    if position == 0
                    else _cpp_module.gather_multirecord_uniform_withbins
                )
                uniform_kernel(
                    self._file_mmap,
                    indices,
                    output,
                    bins,
                    rows_per_record,
                    record_stride,
                    first_body,
                    n_records,
                    last_rows,
                    int(self._column_prefix_bytes[name]),
                    effective_cap,
                    prefetch,
                )
                gathered[name] = output
            return {
                name: gathered[name] if name in gathered else self._gather_one(name, row_indexer)
                for name in column_names
            }
        bins = np.empty(n, dtype=np.int32)
        # Record-base precompute (irregular files): when the read is large
        # enough to amortize the O(R) per-column record_base build (see the
        # gate constant), columns after the first use the rbase kernel;
        # below the gate the generic withbins kernel runs unchanged.
        use_record_base = n >= n_records * _RBASE_MIN_INDICES_PER_RECORD
        rsr_records = self._record_starts_rows[:-1]
        for position, name in enumerate(native_names):
            output = np.empty(n, dtype=self._column_dtypes[name].newbyteorder("="))
            if position == 0:
                _cpp_module.gather_multirecord_bins(
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
            elif use_record_base:
                itemsize = output.dtype.itemsize
                record_base = (
                    self._record_starts_bytes
                    + int(self._column_prefix_bytes[name]) * self._n_rows_per_record
                    - rsr_records * itemsize
                )
                _cpp_module.gather_multirecord_withbins_rbase(
                    self._file_mmap,
                    indices,
                    output,
                    bins,
                    record_base,
                    effective_cap,
                    prefetch,
                )
            else:
                _cpp_module.gather_multirecord_withbins(
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

    def dict(self, copy: bool = True) -> dict[str, NDArray[Any]]:
        """Materialize the whole store as a dict mapping column name to ndarray.

        Parameters
        ----------
        copy : bool, optional
            ``True`` (default): owning arrays. ``False``: READ-ONLY
            zero-copy views backed by the open memmaps; supported only on
            single-record stores with native-byte-order dtypes, raising
            ``ValueError`` otherwise. Views stay valid after :meth:`close`
            (they pin the mapping until garbage-collected).

        Returns
        -------
        dict[str, numpy.ndarray]
            Arrays in on-disk column order, stored dtypes preserved
            (native byte order).
        """
        if self._closed:
            raise ValueError("ColStoreReader is closed.")
        if not copy:
            return self._view_many(list(self._column_dtypes), None)
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
