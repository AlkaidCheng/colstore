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

from . import config, format, kernels
from .view import ColumnView, TableView

if TYPE_CHECKING:
    pass

_MADVISE_FLAGS: dict[str, int] = {
    "normal": getattr(mmap, "MADV_NORMAL", 0),
    "sequential": getattr(mmap, "MADV_SEQUENTIAL", 2),
    "random": getattr(mmap, "MADV_RANDOM", 1),
    "willneed": getattr(mmap, "MADV_WILLNEED", 3),
    "dontneed": getattr(mmap, "MADV_DONTNEED", 4),
}

_USE_DEFAULT_MADVISE = "__default__"


class ColStoreReader:
    """Memory-mapped columnar store with lazy, NumPy-style indexing.

    Opening a store reads its header, creates one ``np.memmap`` per column,
    and applies any requested kernel hints. Reads are performed through
    ``__getitem__``, which returns a lazy view: either a :class:`ColumnView`
    (single-column) or a :class:`TableView` (multi-column). The view
    materializes when one of its ``to_*`` methods is called.

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
    >>> ds['price']                      # ColumnView -> to_array()
    >>> ds[100:200, ['price', 'qty']]    # TableView -> to_dict / to_record / to_dataframe
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
                raise IndexError(
                    f"Expected at most 2 elements in indexing tuple; " f"got {len(key)}."
                )
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
            raise KeyError(f"Unknown column(s): {unknown}. " f"Available columns: {self.columns}")
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
                f"Invalid madvise value {advice!r}; " f"expected one of {sorted(_MADVISE_FLAGS)}."
            )
        flag = _MADVISE_FLAGS[advice]
        if self._is_multi_record:
            with contextlib.suppress(AttributeError, OSError):
                self._file_mmap._mmap.madvise(flag)  # type: ignore[attr-defined]
            return
        for memmap_view in self._memmaps.values():
            with contextlib.suppress(AttributeError, OSError):
                memmap_view._mmap.madvise(flag)  # type: ignore[attr-defined]

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

        ``thread_cap`` overrides the per-call OpenMP thread cap for the fancy-
        index path; ``None`` uses the package default. :meth:`_gather_many`
        passes a divided budget here so concurrent column reads do not
        oversubscribe.
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
        # ``np.array(..., copy=True)`` is typed to return ``NDArray[Any]``;
        # the older ``np.asarray(x).copy()`` chain returns ``Any`` under
        # current numpy stubs, hence the explicit constructor calls.
        if row_indexer is None:
            return np.array(source, dtype=native_dtype, copy=True)
        if isinstance(row_indexer, int):
            return np.atleast_1d(np.array(source[row_indexer], dtype=native_dtype, copy=True))
        if isinstance(row_indexer, slice):
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
        else:
            # Unsorted (or n == 1). The generic searchsorted is O(K log R)
            # and the byte_offset materialization allocates several K-sized
            # int64 temporaries -- both are significant at large K, and
            # we've measured no cheaper alternative (argsort + sorted-path
            # is strictly worse because argsort dominates).
            record_id = np.searchsorted(record_starts_rows, indices, side="right") - 1
            within_record = indices - record_starts_rows[record_id]
            byte_offsets = (
                record_starts_bytes[record_id]
                + col_prefix * n_rows_per_record[record_id]
                + within_record * itemsize
            )

        output = np.empty(n, dtype=native_dtype)
        effective_cap = config.get_gather_thread_cap() if thread_cap is None else max(1, thread_cap)
        _cpp_module.gather_bytes(self._file_mmap, byte_offsets, output, effective_cap)
        return output

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
        the file mmap, plus two O(log R) searchsorted calls to locate the
        first/last overlapping records. No per-element work.

        The output is always native byte order; if the disk dtype is
        non-native the per-record view from ``np.frombuffer`` is byteswapped
        during the assignment into ``output``.
        """
        n = stop - start
        output = np.empty(n, dtype=native_dtype)
        if n == 0:
            return output

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

        Columns are read concurrently on a thread pool, and each column's C++
        gather may itself use OpenMP threads. To keep the product of the two
        from oversubscribing the cores, the per-column OpenMP cap is divided by
        the number of columns running concurrently. With many columns this
        drives each kernel to a single thread, so parallelism comes from the
        column pool (the regime where that is most efficient); with few columns
        each kernel still gets a meaningful share of the cap.
        """
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
