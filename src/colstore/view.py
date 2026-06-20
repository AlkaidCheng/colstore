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

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from . import config
from ._query import _Expr, evaluate_mask
from ._render import Preview, render_lazy_card

if TYPE_CHECKING:
    import pandas as pd

    from ._base import _ReaderBase

_RowIndexer = int | slice | np.ndarray | None


def resolve_preview_n(n: int | None, available_rows: int, row_itemsize: int) -> int:
    """Resolve a preview row count, warning first if the preview would be large.

    ``None`` takes ``config.get_preview_rows()``. If the rows actually shown
    (bounded by ``available_rows``) times ``row_itemsize`` would exceed the
    configured preview memory limit, a ``UserWarning`` is emitted before the
    caller materializes anything.
    """
    rows = config.get_preview_rows() if n is None else n
    limit = config.get_preview_memory_limit()
    if limit:
        estimated = min(max(0, rows), available_rows) * row_itemsize
        if estimated > limit:
            warnings.warn(
                f"this preview would materialize about {estimated / 1e6:.0f} MB; pass a "
                f"smaller n, or raise colstore.config.set_preview_memory_limit().",
                stacklevel=3,
            )
    return rows


def row_width(store: _ReaderBase, columns: list[str]) -> int:
    """Bytes per row across ``columns`` -- the basis for the preview-size estimate."""
    return sum(store._native_dtype(c).itemsize for c in columns)


def preview_index(indexer: _RowIndexer, n_rows: int) -> list[int]:
    """Store row positions for a concrete row ``indexer`` -- the preview index column."""
    if isinstance(indexer, (int, np.integer)):
        return [int(indexer)]
    if indexer is None:
        return list(range(n_rows))
    if isinstance(indexer, slice):
        return list(range(*indexer.indices(n_rows)))
    if indexer.dtype == bool:
        return [int(i) for i in np.flatnonzero(indexer)]
    return [int(i) for i in indexer]


def build_preview(
    label: str,
    total_rows: int | None,
    store: _ReaderBase,
    columns: list[str],
    indexer: _RowIndexer,
) -> Preview:
    """Materialize a concrete row ``indexer`` into a dual-repr ``Preview``.

    Shared by ``TableView`` and the reader/dataset; ``ColumnView`` builds its own
    single-column ``Preview`` from a 1-D array.
    """
    rec = store._build_recarray(indexer, columns)
    return Preview(rec, columns, preview_index(indexer, store.n_rows), total_rows, label)


def validate_columns(available: list[str], names: list[str]) -> None:
    """Raise ``KeyError`` for any name not in ``available`` (matching ``__getitem__``)."""
    available_set = set(available)
    unknown = [n for n in names if n not in available_set]
    if unknown:
        raise KeyError(f"Unknown column(s): {unknown}. Available columns: {available}")


def resolve_select(available: list[str], names: tuple[str, ...]) -> list[str]:
    """Columns to keep for ``select(*names)``: validated, deduplicated-checked, given order."""
    if not names:
        raise ValueError("select() requires at least one column name.")
    chosen = list(names)
    validate_columns(available, chosen)
    if len(set(chosen)) != len(chosen):
        dups = sorted({n for n in chosen if chosen.count(n) > 1})
        raise ValueError(f"Duplicate column(s) in select(): {dups}")
    return chosen


def resolve_drop(available: list[str], names: tuple[str, ...]) -> list[str]:
    """Columns to keep for ``drop(*names)``: ``available`` minus ``names``, in stored order."""
    validate_columns(available, list(names))
    dropped = set(names)
    return [c for c in available if c not in dropped]


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
        """Normalize the row selector into None / int / slice / int-array / mask.

        All integer selectors are validated against ``n_rows`` here, and
        negative positions are folded to their non-negative equivalents. This
        makes every gather backend (NumPy, C++, Numba) agree on bounds and
        wraparound semantics, instead of leaving the unchecked kernels to read
        out of bounds on a memmap. A ``col()`` / ``query`` predicate is evaluated
        against the store first -- reading the columns it references -- into a
        boolean mask.
        """
        row = self._row_part
        if isinstance(row, _Expr):
            row = evaluate_mask(row, self._store._read_query_column, self._store.n_rows)
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
        """Fold negative indices and bounds-check the array against ``n_rows``.

        ``min()`` is taken first and the negative-folding ``np.where`` runs only
        when it is negative, so the common all-non-negative selector skips a full
        ``(indices < 0).any()`` scan and its boolean temporary, leaving just the
        two min/max bounds reductions. ``min() < 0`` is exactly
        ``(indices < 0).any()``, and an index below ``-n_rows`` stays negative
        after folding -- which the post-fold ``lo < 0`` check still rejects -- so
        the result is identical to folding before computing the minimum.
        """
        n_rows = self._store.n_rows
        if indices.size == 0:
            return indices
        lo = indices.min()
        if lo < 0:
            indices = np.where(indices < 0, indices + n_rows, indices)
            lo = indices.min()
        if lo < 0 or indices.max() >= n_rows:
            raise IndexError(f"Row index out of bounds for n_rows {n_rows}.")
        return indices

    @staticmethod
    def _summarize_row_part(row_part: Any) -> str:
        if isinstance(row_part, np.ndarray):
            return f"<ndarray shape={row_part.shape} dtype={row_part.dtype}>"
        return repr(row_part)

    def _head_rows(self, n: int) -> _RowIndexer:
        """A row indexer for the first ``n`` rows of this view's selection."""
        n = max(0, n)
        indexer = self._resolve_row_indexer()
        if isinstance(indexer, (int, np.integer)):
            return indexer
        if indexer is None:
            return slice(0, n)
        if isinstance(indexer, slice):
            chosen = range(*indexer.indices(self._store.n_rows))[:n]
            return np.fromiter(chosen, dtype=np.int64, count=len(chosen))
        selected = np.flatnonzero(indexer) if indexer.dtype == bool else indexer
        return selected[:n]

    def _tail_rows(self, n: int) -> _RowIndexer:
        """A row indexer for the last ``n`` rows of this view's selection."""
        n = max(0, n)
        n_rows = self._store.n_rows
        indexer = self._resolve_row_indexer()
        if isinstance(indexer, (int, np.integer)):
            return indexer
        if indexer is None:
            return slice(max(0, n_rows - n), n_rows)
        if isinstance(indexer, slice):
            full = range(*indexer.indices(n_rows))
            chosen = full[max(0, len(full) - n) :]
            return np.fromiter(chosen, dtype=np.int64, count=len(chosen))
        selected = np.flatnonzero(indexer) if indexer.dtype == bool else indexer
        return selected[max(0, len(selected) - n) :]

    def _row_width(self) -> int:
        """Bytes per row across this view's selected column(s)."""
        raise NotImplementedError

    def _preview_n(self, n: int | None) -> int:
        """Resolve a preview row count for this view, warning if it would be large."""
        return resolve_preview_n(n, self._store.n_rows, self._row_width())


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
        return self._store._native_dtype(self._column_name)

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

    def evaluate(self) -> ColumnView:
        """Resolve the (lazy) row selection now; return a view over the concrete rows.

        For a ``col()`` / ``query`` selection this reads the predicate columns and
        computes the boolean row mask once, returning an equivalent view whose
        rows are already resolved -- so a later ``.array()`` does not recompute the
        selection. The column itself is not materialized.
        """
        return ColumnView(self._store, self._resolve_row_indexer(), self._column_name)

    def head(self, n: int | None = None) -> Preview:
        """First ``n`` values of the column as a previewable peek (default config rows)."""
        return self._preview(self._head_rows(self._preview_n(n)))

    def tail(self, n: int | None = None) -> Preview:
        """Last ``n`` values of the selected column, as a previewable peek."""
        return self._preview(self._tail_rows(self._preview_n(n)))

    def _row_width(self) -> int:
        return self._store._native_dtype(self._column_name).itemsize

    def _preview(self, indexer: _RowIndexer) -> Preview:
        """A single-column ``Preview`` over a concrete row ``indexer``."""
        values = ColumnView(self._store, indexer, self._column_name).array()
        index = preview_index(indexer, self._store.n_rows)
        return Preview(values, [self._column_name], index, None, "ColumnView")

    def _repr_html_(self) -> str | None:
        """Rich Jupyter display: a one-column preview table, or a lazy card.

        A concrete selection previews the first values; a column still under a
        ``col()`` / ``query`` predicate shows a lazy card rather than evaluating
        the predicate to fill it. Returns ``None`` only if the preview can't be
        built (Jupyter then uses the text repr).
        """
        if isinstance(self._row_part, _Expr):
            return render_lazy_card(f"ColumnView({self._column_name!r})", [self._column_name])
        try:
            return self._preview(self._head_rows(self._preview_n(None)))._repr_html_()
        except Exception:
            return None


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

    def select(self, *columns: str) -> TableView:
        """Narrow this view to the named ``columns`` (in the given order), same rows.

        Unknown names raise ``KeyError``; a single name still yields a
        (one-column) ``TableView``. The row selection -- including a lazy
        ``col()`` / ``query`` predicate -- is preserved.
        """
        return TableView(self._store, self._row_part, resolve_select(self._column_names, columns))

    def drop(self, *columns: str) -> TableView:
        """Drop the named ``columns``, keeping the rest in stored order and the same rows."""
        return TableView(self._store, self._row_part, resolve_drop(self._column_names, columns))

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        """Per-column NumPy dtypes."""
        return {name: self._store._native_dtype(name) for name in self._column_names}

    def dict(self, copy: bool = True) -> dict[str, np.ndarray]:
        """Materialize as a dict mapping column name to 1D ndarray.

        Parameters
        ----------
        copy : bool, optional
            ``True`` (default): owning arrays. ``False``: READ-ONLY
            zero-copy views backed by the store's open memmaps,
            all-or-nothing -- see :meth:`ColumnView.array` for the exact
            support conditions and lifetime semantics. ``recarray`` has no
            zero-copy form (it repacks by construction); ``frame`` accepts
            ``copy`` and forwards it here, so ``frame(copy=False)`` aliases
            the same views.

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
        return self._store._build_recarray(self._resolve_row_indexer(), self._column_names)

    def frame(self, copy: bool = True) -> pd.DataFrame:
        """Materialize as a pandas DataFrame.

        Parameters
        ----------
        copy : bool, optional
            ``True`` (default): owning columns. ``False``: a READ-ONLY
            DataFrame whose columns are zero-copy views over the open
            memmaps, forwarding the same all-or-nothing conditions and
            lifetime semantics as :meth:`dict` (raising rather than copying
            when any column cannot be viewed). The per-column block
            construction already shares memory with its input arrays, so the
            frame aliases the mapping with no extra copy.

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

        return _make_dataframe_no_consolidate(self.dict(copy=copy))

    def evaluate(self) -> TableView:
        """Resolve the (lazy) row selection now; return a view over the concrete rows.

        For a ``col()`` / ``query`` selection this reads the predicate columns and
        computes the boolean row mask once, returning an equivalent view whose
        rows are already resolved -- so a later ``.frame()`` / ``.dict()`` /
        ``.recarray()`` (or the head/preview helpers) does not recompute the
        selection. The selected columns are not materialized.
        """
        return TableView(self._store, self._resolve_row_indexer(), self._column_names)

    def head(self, n: int | None = None) -> Preview:
        """First ``n`` rows of the selection as a previewable peek (default config rows)."""
        return self._preview(self._head_rows(self._preview_n(n)))

    def tail(self, n: int | None = None) -> Preview:
        """Last ``n`` rows of the selection, as a previewable peek."""
        return self._preview(self._tail_rows(self._preview_n(n)))

    def _row_width(self) -> int:
        return row_width(self._store, self._column_names)

    def _preview(self, indexer: _RowIndexer) -> Preview:
        """A multi-column ``Preview`` over a concrete row ``indexer``."""
        return build_preview("TableView", None, self._store, self._column_names, indexer)

    def _repr_html_(self) -> str | None:
        """Rich Jupyter display: a small preview table, or a lazy card.

        A view whose row selection is already concrete (whole store, slice, mask,
        or an evaluated predicate) shows the preview. A view still carrying a
        ``col()`` / ``query`` predicate shows a lazy card instead -- previewing it
        would have to read the predicate columns, which a repr must not trigger;
        call ``.head()`` or ``.evaluate()`` to opt in. The full row count is
        omitted (it too is lazy). Returns ``None`` (text repr) if the preview
        can't be built.
        """
        if isinstance(self._row_part, _Expr):
            return render_lazy_card("TableView", self._column_names)
        try:
            return self._preview(self._head_rows(self._preview_n(None)))._repr_html_()
        except Exception:
            return None
