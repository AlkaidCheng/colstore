"""Lazy views returned by ``ColStoreReader.__getitem__``.

Two concrete classes implement the public surface:

* :class:`ColumnView` — produced by ``ds['col']`` or ``ds[rows, 'col']``.
  Supports only :meth:`array`.
* :class:`TableView` — produced by every other indexing pattern.
  Supports :meth:`dict`, :meth:`recarray`, and :meth:`frame`.

Both share a tiny base class for the row-indexer normalization logic; that
base is internal and not part of the public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from ._base import _ReaderBase

_RowIndexer = int | slice | np.ndarray | None


class _BaseView:
    """Shared row-indexer plumbing for the two view types.

    Not part of the public API: users should construct views via
    ``ColStoreReader.__getitem__`` and consume them through the concrete
    subclasses.
    """

    __slots__ = ("_row_part", "_store")

    def __init__(self, store: _ReaderBase, row_part: Any) -> None:
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
            # Pass the mask through as-is: the reader routes multi-record
            # native-dtype reads to the mask-native kernel (1 byte/row of
            # selector traffic instead of materializing 8-byte indices) and
            # falls back to np.flatnonzero + the fancy paths everywhere
            # else, including all single-record reads (where the backend
            # parameter's contract applies to the resulting fancy read).
            return np.ascontiguousarray(row_array)
        if row_array.dtype.kind not in ("i", "u"):
            raise IndexError(
                f"Row index array must be integer or boolean; got dtype {row_array.dtype}."
            )
        # ascontiguousarray, not astype(copy=False): the latter preserves
        # strides when the dtype is already int64, and a strided index view
        # (e.g. ``rows[::2]`` or ``rows[::-1]``) would reach the C++ kernels,
        # which interpret the array as a contiguous int64 pointer -- wrong
        # values for positive strides, out-of-bounds reads for negative
        # ones. No-op (no copy) for arrays that are already contiguous.
        return self._validate_fancy_index(np.ascontiguousarray(row_array, dtype=np.int64))

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

    Materializes to a 1D ``numpy.ndarray`` via :meth:`array`. No other
    materialization method is available; calling :meth:`dict`,
    :meth:`recarray`, or :meth:`frame` here would not make sense and
    those methods are intentionally absent from this class.
    """

    __slots__ = ("_column_name",)

    def __init__(
        self,
        store: _ReaderBase,
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

    def array(self, copy: bool = True) -> np.ndarray:
        """Materialize as a 1D ndarray.

        Parameters
        ----------
        copy : bool, optional
            ``True`` (default): an owning array, safe to mutate and to use
            after the store is closed. ``False``: a READ-ONLY zero-copy
            view backed by the store's open memmap, supported exactly when
            the store is single-record, the column's dtype is in native
            byte order, and the row selector is ``None``, an int, or a
            slice; anything else raises ``ValueError`` rather than
            silently copying. The view holds a reference to the mapping,
            so it stays valid after the store is closed -- at the cost of
            keeping the file mapped until the view is garbage-collected.

        Returns
        -------
        numpy.ndarray
            1D array of the selected rows in the column's stored dtype.
        """
        row_indexer = self._resolve_row_indexer()
        if not copy:
            return self._store._view_one(self._column_name, row_indexer)
        return self._store._gather_one(self._column_name, row_indexer)


class TableView(_BaseView):
    """Lazy view of multiple columns produced by any non-string indexing.

    Materializes through one of :meth:`dict`, :meth:`recarray`, or
    :meth:`frame`. There is intentionally no ``array`` method —
    multiple columns generally have different dtypes and cannot be packed
    into a single homogeneous ndarray.
    """

    __slots__ = ("_column_names",)

    def __init__(
        self,
        store: _ReaderBase,
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

    def dict(self, copy: bool = True) -> dict[str, np.ndarray]:
        """Materialize as a dict mapping column name to 1D ndarray.

        Parameters
        ----------
        copy : bool, optional
            ``True`` (default): owning arrays. ``False``: READ-ONLY
            zero-copy views backed by the store's open memmaps,
            all-or-nothing -- see :meth:`ColumnView.array` for the exact
            support conditions and lifetime semantics. ``recarray`` and
            ``frame`` have no zero-copy form (both repack by construction).

        Returns
        -------
        dict[str, numpy.ndarray]
            Arrays in selection order; each column's stored dtype is
            preserved.
        """
        row_indexer = self._resolve_row_indexer()
        if not copy:
            return self._store._view_many(self._column_names, row_indexer)
        return self._store._gather_many(self._column_names, row_indexer)

    def recarray(self) -> np.ndarray:
        """Materialize as a structured (record) ndarray with one field per column.

        Returns
        -------
        numpy.ndarray
            Structured 1D array. ``result[name]`` returns the column.
        """
        column_data = self.dict()
        n_records = next(iter(column_data.values())).shape[0]
        record_dtype = np.dtype([(name, column_data[name].dtype) for name in self._column_names])
        record_array = np.empty(n_records, dtype=record_dtype)
        for name in self._column_names:
            record_array[name] = column_data[name]
        return record_array

    def frame(self) -> pd.DataFrame:
        """Materialize as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            Columns are in selection order with their stored dtypes preserved.
            The frame skips dtype-block consolidation (one ``Block`` per
            column) -- see
            :func:`colstore.reader._make_dataframe_no_consolidate` for
            rationale and details.
        """
        from .reader import _make_dataframe_no_consolidate

        return _make_dataframe_no_consolidate(self.dict())
