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

from .view import ColumnView, TableView

if TYPE_CHECKING:
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

    # ---- Column metadata -----------------------------------------------

    @property
    def columns(self) -> list[str]:
        """Column names in stored order."""
        return list(self._column_dtypes)

    @property
    def dtypes(self) -> dict[str, np.dtype[Any]]:
        """Column dtypes in native byte order, in stored order."""
        return {name: dtype.newbyteorder("=") for name, dtype in self._column_dtypes.items()}

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_rows, n_columns)``."""
        return self.n_rows, len(self._column_dtypes)

    # ---- Mapping protocol over column names ----------------------------

    def __len__(self) -> int:
        return self.n_rows

    def __contains__(self, column_name: object) -> bool:
        return isinstance(column_name, str) and column_name in self._column_dtypes

    def __iter__(self) -> Iterator[str]:
        return iter(self.columns)

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
