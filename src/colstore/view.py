"""Lazy views returned by ``ColStore.__getitem__``.

Two concrete classes implement the public surface:

* :class:`ColumnView` — produced by ``ds['col']`` or ``ds[rows, 'col']``.
  Supports only :meth:`to_array`.
* :class:`TableView` — produced by every other indexing pattern.
  Supports :meth:`to_dict`, :meth:`to_record`, and :meth:`to_dataframe`.

Both share a tiny base class for the row-indexer normalization logic; that
base is internal and not part of the public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from .reader import ColStore

_RowIndexer = int | slice | np.ndarray | None


class _BaseView:
    """Shared row-indexer plumbing for the two view types.

    Not part of the public API: users should construct views via
    ``ColStore.__getitem__`` and consume them through the concrete
    subclasses.
    """

    __slots__ = ("_row_part", "_store")

    def __init__(self, store: ColStore, row_part: Any) -> None:
        self._store = store
        self._row_part = row_part

    def _resolve_row_indexer(self) -> _RowIndexer:
        """Normalize the user's row selector into None / int / slice / int-array.

        All integer selectors are validated against ``n_rows`` here, and
        negative positions are folded to their non-negative equivalents. This
        makes every gather backend (NumPy, C++, Numba) agree on bounds and
        wraparound semantics, instead of leaving the unchecked kernels to read
        out of bounds on a memmap.
        """
        row = self._row_part
        if row is None:
            return None
        if isinstance(row, (int, np.integer)):
            return self._normalize_scalar(int(row))
        if isinstance(row, slice):
            return row
        row_array = np.asarray(row)
        if row_array.ndim == 0:
            # A 0-d array (e.g. ``np.array(5)``) behaves like a scalar int, not
            # a length-1 fancy index; route it to the scalar path so the result
            # shape matches ``ds[5]``.
            if row_array.dtype == bool:
                raise IndexError("0-d boolean index is not supported.")
            if row_array.dtype.kind not in ("i", "u"):
                raise IndexError(
                    f"Row index must be integer or boolean; got dtype {row_array.dtype}."
                )
            return self._normalize_scalar(int(row_array))
        if row_array.dtype == bool:
            if row_array.shape[0] != self._store.n_rows:
                raise IndexError(
                    f"Boolean mask length {row_array.shape[0]} does not match "
                    f"n_rows {self._store.n_rows}."
                )
            return np.flatnonzero(row_array)
        if row_array.dtype.kind not in ("i", "u"):
            raise IndexError(
                f"Row index array must be integer or boolean; got dtype {row_array.dtype}."
            )
        return self._validate_fancy_index(row_array.astype(np.int64, copy=False))

    def _normalize_scalar(self, position: int) -> int:
        """Fold a negative scalar row index and bounds-check it."""
        n_rows = self._store.n_rows
        if position < 0:
            position += n_rows
        if not 0 <= position < n_rows:
            raise IndexError(f"Row index {position} out of bounds for n_rows {n_rows}.")
        return position

    def _validate_fancy_index(self, indices: np.ndarray) -> np.ndarray:
        """Fold negative indices and bounds-check the whole array in one pass."""
        n_rows = self._store.n_rows
        if indices.size == 0:
            return indices
        if (indices < 0).any():
            indices = np.where(indices < 0, indices + n_rows, indices)
        if indices.min() < 0 or indices.max() >= n_rows:
            raise IndexError(f"Row index out of bounds for n_rows {n_rows}.")
        return indices

    @staticmethod
    def _summarize_row_part(row_part: Any) -> str:
        if isinstance(row_part, np.ndarray):
            return f"<ndarray shape={row_part.shape} dtype={row_part.dtype}>"
        return repr(row_part)


class ColumnView(_BaseView):
    """Lazy view of a single column produced by indexing with a string name.

    Materializes to a 1D ``numpy.ndarray`` via :meth:`to_array`. No other
    materialization method is available; calling :meth:`to_dict`,
    :meth:`to_record`, or :meth:`to_dataframe` here would not make sense and
    those methods are intentionally absent from this class.
    """

    __slots__ = ("_column_name",)

    def __init__(
        self,
        store: ColStore,
        row_part: Any,
        column_name: str,
    ) -> None:
        super().__init__(store, row_part)
        self._column_name = column_name

    def __repr__(self) -> str:
        return (
            f"ColumnView(column={self._column_name!r}, "
            f"rows={self._summarize_row_part(self._row_part)}, lazy=True)"
        )

    @property
    def column(self) -> str:
        """Name of the column selected by this view."""
        return self._column_name

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype of the selected column."""
        return self._store.dtypes[self._column_name]

    def to_array(self) -> np.ndarray:
        """Materialize as a 1D owning ndarray.

        Returns
        -------
        numpy.ndarray
            Owning 1D array of the selected rows in the column's stored
            dtype. Safe to use after the source store is closed.
        """
        return self._store._gather_one(self._column_name, self._resolve_row_indexer())


class TableView(_BaseView):
    """Lazy view of multiple columns produced by any non-string indexing.

    Materializes through one of :meth:`to_dict`, :meth:`to_record`, or
    :meth:`to_dataframe`. There is intentionally no ``to_array`` method —
    multiple columns generally have different dtypes and cannot be packed
    into a single homogeneous ndarray.
    """

    __slots__ = ("_column_names",)

    def __init__(
        self,
        store: ColStore,
        row_part: Any,
        column_names: list[str],
    ) -> None:
        super().__init__(store, row_part)
        self._column_names = column_names

    def __repr__(self) -> str:
        return (
            f"TableView(columns={self._column_names!r}, "
            f"rows={self._summarize_row_part(self._row_part)}, lazy=True)"
        )

    @property
    def columns(self) -> list[str]:
        """Names of the columns selected by this view, in selection order."""
        return list(self._column_names)

    @property
    def n_columns(self) -> int:
        """Number of columns selected by this view."""
        return len(self._column_names)

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        """Per-column NumPy dtypes."""
        return {name: self._store.dtypes[name] for name in self._column_names}

    def to_dict(self) -> dict[str, np.ndarray]:
        """Materialize as a dict mapping column name to 1D ndarray.

        Returns
        -------
        dict[str, numpy.ndarray]
            Owning arrays in selection order; each column's stored dtype is
            preserved.
        """
        return self._store._gather_many(self._column_names, self._resolve_row_indexer())

    def to_record(self) -> np.ndarray:
        """Materialize as a structured (record) ndarray with one field per column.

        Returns
        -------
        numpy.ndarray
            Structured 1D array. ``result[name]`` returns the column.
        """
        column_data = self.to_dict()
        n_records = next(iter(column_data.values())).shape[0]
        record_dtype = np.dtype([(name, column_data[name].dtype) for name in self._column_names])
        record_array = np.empty(n_records, dtype=record_dtype)
        for name in self._column_names:
            record_array[name] = column_data[name]
        return record_array

    def to_dataframe(self) -> pd.DataFrame:
        """Materialize as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            Columns are in selection order with their stored dtypes preserved.
        """
        import pandas as pd

        return pd.DataFrame(self.to_dict())
