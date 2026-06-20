"""Shared read interface for the single-file and multi-file readers.

:class:`_ReaderBase` factors out the column and row indexing surface that
:class:`~colstore.reader.ColStoreReader` and
:class:`~colstore.dataset.ColStoreDataset` present identically, so the two
cannot drift apart. Concrete subclasses supply the storage primitives -- the
``n_rows`` count, the ordered ``_column_dtypes`` mapping, and the four-method
gather/view seam that the lazy views call -- and everything expressible purely
in terms of those lives here.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from numpy.typing import NDArray

from . import config, kernels
from .view import ColumnView, TableView

if TYPE_CHECKING:
    from pathlib import Path

    from .frame import ColStoreFrame


class _ReaderBase(abc.ABC):
    """Column and row indexing shared by every colstore reader.

    Subclasses must provide ``_column_dtypes`` (column name -> on-disk dtype,
    in stored order), an :attr:`n_rows` property, and the materialization seam
    (:meth:`_gather_one`, :meth:`_gather_many`, :meth:`_view_one`,
    :meth:`_view_many`) that :class:`~colstore.view.ColumnView` and
    :class:`~colstore.view.TableView` call to realize a selection.
    """

    # Column name -> on-disk dtype (native or not), in stored order. The
    # concrete reader assigns this in its constructor.
    _column_dtypes: dict[str, np.dtype[Any]]

    # ---- Storage seam (implemented by subclasses) ----------------------

    @property
    @abc.abstractmethod
    def n_rows(self) -> int:
        """Total number of logical rows visible through this reader."""

    @abc.abstractmethod
    def _gather_one(
        self, column_name: str, row_indexer: Any, thread_cap: int | None = None
    ) -> NDArray[Any]:
        """Copying read of one column for an already-normalized row selector."""

    @abc.abstractmethod
    def _gather_many(self, column_names: list[str], row_indexer: Any) -> dict[str, NDArray[Any]]:
        """Copying read of several columns for an already-normalized selector."""

    @abc.abstractmethod
    def _view_one(self, column_name: str, row_indexer: Any) -> NDArray[Any]:
        """Zero-copy read of one column, or raise when a copy is unavoidable."""

    @abc.abstractmethod
    def _view_many(self, column_names: list[str], row_indexer: Any) -> dict[str, NDArray[Any]]:
        """Zero-copy read of several columns, or raise when a copy is unavoidable."""

    def _gather_slice_into(
        self, out: NDArray[Any], column_name: str, start: int, stop: int
    ) -> None:
        """Fill ``out`` with rows ``[start, stop)`` of one column.

        The generic path materializes the forward slice and copies it in. The
        multi-file reader overrides this to write each file's portion of ``out``
        directly, sparing the intermediate array.
        """
        out[:] = self._gather_one(column_name, slice(start, stop))

    @abc.abstractmethod
    def _column_disk_runs(self, column_name: str) -> list[tuple[Path, int, int]]:
        """On-disk byte runs of one column, in global row order.

        Returns ``(path, file_offset, n_bytes)`` triples whose concatenation is
        the column's contiguous on-disk image -- the file-coordinate basis for a
        raw passthrough merge copy, which writes those bytes straight to the
        destination instead of materializing them. A single-record file yields
        one run; a multi-record or multi-file source yields one per record, in
        order.

        Implementations raise ``ValueError`` when a raw byte copy would not
        preserve values (a non-native on-disk dtype, which cannot be byteswapped
        by a copy); the merge-copy caller treats that as "not a pure merge" and
        falls back to the materializing write.
        """

    # ---- Column metadata -----------------------------------------------

    @property
    def columns(self) -> list[str]:
        """Column names in stored order."""
        return list(self._column_dtypes)

    @property
    def dtypes(self) -> dict[str, np.dtype[Any]]:
        """Column dtypes in native byte order, in stored order."""
        return {name: self._native_dtype(name) for name in self._column_dtypes}

    def _native_dtype(self, column_name: str) -> np.dtype[Any]:
        """One column's dtype in native byte order.

        The on-disk column is little-endian; a big-endian host reads it as a
        non-native dtype that the gather/copy converts. Callers needing a single
        column's native dtype use this instead of indexing :attr:`dtypes`, which
        rebuilds the whole mapping on every access -- doing that once per column
        is quadratic in the column count. Raises ``KeyError`` for an unknown name.
        """
        native: np.dtype[Any] = self._column_dtypes[column_name].newbyteorder("=")
        return native

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_rows, n_columns)``."""
        return self.n_rows, len(self._column_dtypes)

    # ---- Whole-store materializer --------------------------------------

    def recarray(self) -> NDArray[Any]:
        """Materialize the whole store as a structured (record) ndarray.

        One field per column, in stored order; ``result[name]`` is the column.
        See :meth:`_build_recarray` for how the columns are interleaved.
        """
        names = list(self._column_dtypes)
        if not names or self.n_rows == 0:
            record_dtype = np.dtype([(name, self._native_dtype(name)) for name in names])
            return np.empty(self.n_rows, dtype=record_dtype)
        return self._build_recarray(None, names)

    def _build_recarray(self, row_indexer: Any, column_names: list[str]) -> NDArray[Any]:
        """Interleave a row selection of the named columns into a record array.

        Shared by :meth:`recarray` (whole store) and ``TableView.recarray`` (a
        row/column subset). When the gather extension is built, each column's
        contiguous native source -- a zero-copy view where one exists, else a
        materialized gather -- is interleaved into the record layout by the
        parallel ``interleave_records`` kernel (row-major, so the record array is
        written once rather than once per column); a single-record native whole
        read needs no intermediate column arrays at all. Without the extension it
        falls back to the column-major assignment from a materialized column dict.
        Assumes at least one column.
        """
        record_dtype = np.dtype([(name, self._native_dtype(name)) for name in column_names])
        if not kernels.cpp_available():
            column_data = self._gather_many(column_names, row_indexer)
            record_array: NDArray[Any] = np.empty(
                column_data[column_names[0]].shape[0], dtype=record_dtype
            )
            for name in column_names:
                record_array[name] = column_data[name]
            return record_array

        sources = [self._contiguous_native_source(name, row_indexer) for name in column_names]
        n_records = sources[0].shape[0]
        record_array = np.empty(n_records, dtype=record_dtype)
        if n_records == 0:
            return record_array
        fields = record_dtype.fields
        assert fields is not None  # a structured dtype always has fields
        kernels.interleave_records(
            record_array,
            record_dtype.itemsize,
            n_records,
            np.array([source.ctypes.data for source in sources], dtype=np.int64),
            np.array([source.dtype.itemsize for source in sources], dtype=np.int64),
            np.array([fields[name][1] for name in column_names], dtype=np.int64),
            config.get_gather_thread_cap(),
        )
        return record_array

    def _contiguous_native_source(self, column_name: str, row_indexer: Any) -> NDArray[Any]:
        """A contiguous, native-order array for one column over the row selection.

        A zero-copy memmap view when the store can give a contiguous one
        (single-record, native dtype, contiguous selector), otherwise a
        materialized gather -- which byteswaps a non-native dtype, stitches a
        multi-record or multi-file column, realizes a strided (step > 1) view,
        and resolves a fancy or boolean selection. Always contiguous and native,
        so the kernel's raw field copy is exact.
        """
        try:
            view = self._view_one(column_name, row_indexer)
        except ValueError:
            return self._gather_one(column_name, row_indexer)
        if view.flags["C_CONTIGUOUS"]:
            return view
        return self._gather_one(column_name, row_indexer)

    # ---- Mapping protocol over column names ----------------------------

    def __len__(self) -> int:
        return self.n_rows

    def __contains__(self, column_name: object) -> bool:
        return isinstance(column_name, str) and column_name in self._column_dtypes

    def __iter__(self) -> Iterator[str]:
        return iter(self._column_dtypes)

    # ---- Indexing ------------------------------------------------------

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

    def edit(self) -> ColStoreFrame:
        """Open a deferred editing frame over this store's columns.

        Returns a :class:`~colstore.frame.ColStoreFrame` seeded with this store's
        columns as native-passthrough leaves. Column updates, additions,
        removals, renames, and elementwise transforms are deferred and written to
        a new file by :meth:`~colstore.frame.ColStoreFrame.write`; this store is
        not modified. Over a multi-file dataset the leaves read through the
        dataset's gather seam, so a written result is the combined, transformed
        data in one file.
        """
        from .frame import ColStoreFrame

        return ColStoreFrame(self)
