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
from numpy.lib.mixins import NDArrayOperatorsMixin

from . import config
from ._pandas import _make_dataframe_no_consolidate
from ._query import _Expr
from ._render import Preview, render_lazy_card, render_lazy_card_text
from .frame import ColStoreFrame, ColumnReductions
from .interop import InteropMixin

if TYPE_CHECKING:
    import pandas as pd

    from ._base import _ReaderBase

RowIndexer = int | slice | np.ndarray | None


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


def preview_index(indexer: RowIndexer, n_rows: int) -> list[int]:
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
    indexer: RowIndexer,
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


def edit_row_selection(indexer: RowIndexer, n_rows: int) -> np.ndarray | None:
    """Normalize a resolved row indexer into a frame's row selection.

    ``None`` -- and a slice spanning every row -- means "all rows", which the
    frame keeps as its unfiltered streaming-write path. An int or sub-range slice
    becomes an explicit int64 array of the chosen source rows; an integer fancy
    index is kept as int64 indices, and a boolean mask is passed through for the
    frame to store by density (kept when dense, lowered to indices when sparse).
    """
    if indexer is None:
        return None
    if isinstance(indexer, (int, np.integer)):
        return np.array([int(indexer)], dtype=np.int64)
    if isinstance(indexer, slice):
        start, stop, step = indexer.indices(n_rows)
        if step == 1 and start == 0 and stop == n_rows:
            return None
        return np.arange(start, stop, step, dtype=np.int64)
    array = np.asarray(indexer)
    if array.dtype == bool:
        # Pass the mask through; the frame (_normalize_base_rows) keeps it only when it
        # is dense enough to gather through the mask-native kernel (as ds[mask] does),
        # and lowers a sparse one to indices once. A where() composes onto indices.
        return np.ascontiguousarray(array)
    return array.astype(np.int64, copy=False)


def _is_column_key(key: Any) -> bool:
    """Whether a table index selects columns -- a name, or a non-empty list/tuple of names.

    Distinguishes ``tv['a']`` / ``tv[['a', 'b']]`` (column projection) from a row
    re-selection (a slice, integer array, mask, or an empty list, all routed to rows).
    """
    if isinstance(key, str):
        return True
    if isinstance(key, (list, tuple)) and len(key) > 0:
        return all(isinstance(name, str) for name in key)
    return False


def compose_slices(start: int, step: int, length: int, key: slice) -> slice:
    """Compose ``key`` (a slice over a length-``length`` view) onto the view's own
    ``range(start, _, step)`` store rows, analytically -- a slice of a slice stays a
    slice, with no index array materialized.
    """
    sub_start, sub_stop, sub_step = key.indices(length)
    count = len(range(sub_start, sub_stop, sub_step))
    if count == 0:
        return slice(0, 0)
    new_step = step * sub_step
    new_start = start + sub_start * step
    new_stop = new_start + count * new_step
    # A reverse run that passes row 0 needs the open-ended sentinel: a negative
    # numeric stop would otherwise be read as an offset from the end.
    if new_step < 0 and new_stop < 0:
        return slice(new_start, None, new_step)
    return slice(new_start, new_stop, new_step)


def key_to_view_indices(key: Any, length: int) -> int | np.ndarray:
    """Normalize a row re-selection ``key`` into indices in ``[0, length)`` of a view.

    Returns a scalar ``int`` for a scalar selector, else an ``int64`` array (a boolean
    mask must match ``length``). Negatives fold against ``length``; non-integer
    selectors raise. Bounds are the view's own row count, not the store's.
    """
    if isinstance(key, (int, np.integer)):
        position = int(key)
        if position < 0:
            position += length
        if not 0 <= position < length:
            raise IndexError(f"Row index {key} out of bounds for the view's {length} rows.")
        return position
    if isinstance(key, slice):
        return np.arange(*key.indices(length), dtype=np.int64)
    array = np.asarray(key)
    if array.dtype == bool:
        if array.shape[0] != length:
            raise IndexError(
                f"Boolean mask length {array.shape[0]} does not match the view's {length} rows."
            )
        return np.flatnonzero(array)
    if array.size == 0:
        return np.empty(0, dtype=np.int64)
    if array.dtype.kind not in ("i", "u"):
        raise IndexError(f"Row index must be integer or boolean; got dtype {array.dtype}.")
    array = array.astype(np.int64, copy=False)
    if array.min() < 0:
        array = np.where(array < 0, array + length, array)
    if array.min() < 0 or array.max() >= length:
        raise IndexError(f"Row index out of bounds for the view's {length} rows.")
    return array


class _BaseView(InteropMixin):
    """Shared row-indexer plumbing for the two view types.

    Not part of the public API: users should construct views via
    ``ColStoreReader.__getitem__`` and consume them through the concrete
    subclasses.
    """

    __slots__ = ("_row_part", "_store")

    def __init__(self, store: _ReaderBase, row_part: Any) -> None:
        self._store = store
        self._row_part = row_part

    def _resolve_row_indexer(self) -> RowIndexer:
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
            row = self._store._evaluate_query_mask(row)
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
            if row_array.size == 0:
                # A bare empty index (e.g. ``ds[[]]``) is float64 by NumPy's default
                # dtype; it selects no rows, so treat it as an empty integer fancy
                # index rather than rejecting it on the placeholder dtype.
                return np.empty(0, dtype=np.int64)
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
        """Fold negative indices and bounds-check the array against ``n_rows``."""
        n_rows = self._store.n_rows
        if indices.size == 0:
            return indices
        # min() first: the negative-folding np.where runs only when some index is
        # negative, so the common all-non-negative selector skips a full
        # (indices < 0).any() scan and its boolean temporary. min() < 0 is exactly
        # (indices < 0).any(), and an index below -n_rows stays negative after
        # folding (the post-fold lo < 0 check still rejects it), so the result is
        # identical to folding before computing the minimum.
        lo = indices.min()
        if lo < 0:
            indices = np.where(indices < 0, indices + n_rows, indices)
            lo = indices.min()
        if lo < 0 or indices.max() >= n_rows:
            raise IndexError(f"Row index out of bounds for n_rows {n_rows}.")
        return indices

    def _compose_rows(self, key: Any) -> Any:
        """Compose a row re-selection ``key`` onto this view's current rows.

        ``view[key]`` selects ``key`` from the rows the view already covers, so it
        reads the same store rows as ``ds[<view rows>[key], <view cols>]``. Returns a
        store-relative selector (int / slice / int64 array) for a new view; a concrete
        view composes with no I/O (only a ``col()`` / ``query`` view resolves first).
        """
        current = self._resolve_row_indexer()
        if current is None:
            return key  # whole store -- the key already addresses store rows
        if isinstance(current, slice):
            start, stop, step = current.indices(self._store.n_rows)
            length = len(range(start, stop, step))
            if isinstance(key, slice):
                return compose_slices(start, step, length, key)
            sub = key_to_view_indices(key, length)
            return start + sub * step
        if isinstance(current, (int, np.integer)):
            positions = np.array([int(current)], dtype=np.int64)
        else:
            positions = np.flatnonzero(current) if current.dtype == bool else current
        sub = key_to_view_indices(key, positions.shape[0])
        result = positions[sub]
        return int(result) if isinstance(sub, int) else result

    def count(self) -> int:
        """Number of rows this view selects -- a scalar.

        Resolves a ``col()`` / ``query`` predicate (reading only the columns it
        references); a concrete row selection is counted without any I/O.
        """
        indexer = self._resolve_row_indexer()
        if indexer is None:
            return self._store.n_rows
        if isinstance(indexer, (int, np.integer)):
            return 1
        if isinstance(indexer, slice):
            return len(range(*indexer.indices(self._store.n_rows)))
        if indexer.dtype == bool:
            return int(indexer.sum())
        return int(indexer.shape[0])

    def _preview(self, indexer: RowIndexer) -> Preview:
        """A ``Preview`` over a concrete row ``indexer`` -- provided by each view type."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """A formatted preview table (pandas-style), matching the notebook display.

        A view still carrying a ``col()`` / ``query`` predicate shows a lazy card
        rather than a table -- a repr must not read the predicate columns to fill
        one; call :meth:`head` or :meth:`evaluate` to opt in.
        """
        label = type(self).__name__
        if isinstance(self._row_part, _Expr):
            return render_lazy_card_text(label, self._edit_columns())
        try:
            return repr(self._preview(self._head_rows(self._preview_n(None))))
        except Exception:
            n = len(self._edit_columns())
            return f"<{label}: {n} column{'' if n == 1 else 's'}>"

    def _head_rows(self, n: int) -> RowIndexer:
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

    def _tail_rows(self, n: int) -> RowIndexer:
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

    def _edit_columns(self) -> list[str]:
        """The column names this view seeds an editing frame with."""
        raise NotImplementedError

    def edit(self) -> ColStoreFrame:
        """Open a deferred editing frame over this view's columns and rows.

        Seeds a :class:`~colstore.frame.ColStoreFrame` with this view's columns as
        native leaves and carries over its row selection. A concrete selector --
        int, slice, fancy index, or boolean mask -- becomes the frame's fixed row
        set; a lazy ``col()`` / ``query`` predicate is carried as a pending
        :meth:`~colstore.frame.ColStoreFrame.where`, resolved (like the rest of the
        graph) only when the frame is materialized. So ``ds[pred].edit()`` matches
        ``ds.edit().where(pred)`` -- it stays lazy and the cut shows in
        :meth:`~colstore.frame.ColStoreFrame.report`; call :meth:`evaluate` before
        :meth:`edit` to resolve the predicate to a fixed row set first. The source
        store is not modified; :meth:`~colstore.frame.ColStoreFrame.write` produces a
        new file.
        """
        columns = self._edit_columns()
        row = self._row_part
        if isinstance(row, _Expr):
            return ColStoreFrame(self._store, columns, predicate=row)
        rows = edit_row_selection(self._resolve_row_indexer(), self._store.n_rows)
        return ColStoreFrame(self._store, columns, rows)


class ColumnView(NDArrayOperatorsMixin, ColumnReductions, _BaseView):
    """Lazy view of a single column produced by indexing with a string name.

    Materializes to a 1D ``numpy.ndarray`` via :meth:`array`. No other
    materialization method is available; calling :meth:`dict`,
    :meth:`recarray`, or :meth:`frame` here would not make sense and
    those methods are intentionally absent from this class.

    A column view is an eager read surface, so the elementwise operators and
    NumPy ufuncs it inherits compute immediately: ``ds[name] * 2``,
    ``ds['a'] + ds['b']``, ``ds[name] > 0``, and ``np.log(ds[name])`` each gather
    the selected rows and return a plain ``ndarray``. To build a *deferred*
    transform that composes without reading, edit the store into a frame
    (``reader.edit()``); to select rows by a column predicate, use
    :func:`~colstore.col` / :meth:`~colstore.ColStoreReader.query`.
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

    def __getitem__(self, key: Any) -> ColumnView:
        """Narrow this column's rows: ``ds['x'][:100]`` reads the first 100 ``x``.

        Composes ``key`` (a slice, integer, integer array, or boolean mask) onto the
        view's current row selection, the same as ``ds[<view rows>[key], 'x']`` -- so
        ``ds['x'][:100]`` equals ``ds[:100, 'x']``. To read the values, call
        :meth:`array` (or ``np.asarray``).
        """
        return ColumnView(self._store, self._compose_rows(key), self._column_name)

    @property
    def column(self) -> str:
        """Name of the column selected by this view."""
        return self._column_name

    def _edit_columns(self) -> list[str]:
        return [self._column_name]

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

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """Apply a NumPy ufunc eagerly by materializing the column(s).

        Backs the inherited operators (``*``, ``+``, ``>``, ...) and direct ufunc
        calls (``np.log(ds[name])``): each column-view operand is gathered over its
        selected rows and the ufunc runs on the resulting arrays, returning a plain
        ``ndarray``. Reductions and other ufunc methods are declined so they fall
        back to NumPy -- a column reduces through its :meth:`sum` / :meth:`mean`
        terminals, not here.
        """
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        operands = [x.array() if isinstance(x, ColumnView) else x for x in inputs]
        return ufunc(*operands, **kwargs)

    def _reduction_frame(self) -> ColStoreFrame:
        # A reader column reduces through its editing frame (the streaming engine).
        return self.edit()

    def _reduction_name(self) -> str:
        return self._column_name

    def evaluate(self) -> ColumnView:
        """Resolve the (lazy) row selection now; return a view over the concrete rows.

        For a ``col()`` / ``query`` selection this reads the predicate columns and
        computes the boolean row mask once, returning an equivalent view whose
        rows are already resolved -- so a later ``.array()`` does not recompute the
        selection. The column itself is not materialized.
        """
        return ColumnView(self._store, self._resolve_row_indexer(), self._column_name)

    def _interop_target(self) -> tuple[_ReaderBase, list[str], Any, bool]:
        """This single column and its resolved rows -- the export seam (see InteropMixin)."""
        return self._store, [self._column_name], self._resolve_row_indexer(), True

    def __arrow_c_array__(self, requested_schema: Any = None) -> Any:
        """Arrow C array interface: lets any Arrow consumer ingest the column.

        For example ``pyarrow.array(view)`` or ``polars.from_arrow(view)``. A
        column split across segments is concatenated into one array here (a copy);
        :meth:`~colstore.interop.base.InteropMixin.__arrow_c_stream__` keeps it
        zero-copy.
        """
        from .interop.arrow import to_c_array

        return to_c_array(self.arrow(), requested_schema)

    def head(self, n: int | None = None) -> Preview:
        """First ``n`` values of the column as a previewable peek (default config rows)."""
        return self._preview(self._head_rows(self._preview_n(n)))

    def tail(self, n: int | None = None) -> Preview:
        """Last ``n`` values of the selected column, as a previewable peek."""
        return self._preview(self._tail_rows(self._preview_n(n)))

    def _row_width(self) -> int:
        return self._store._native_dtype(self._column_name).itemsize

    def _preview(self, indexer: RowIndexer) -> Preview:
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


class _ColumnTable:
    """Column access shared by the store-backed, row-selected tables.

    A reader/dataset and a :class:`TableView` are both multi-column tables over a
    store and a row selection, so they reach their columns the same way:
    ``table[name]`` projects one column (a :class:`ColumnView`), ``table[[names]]``
    narrows to a sub-table (a :class:`TableView`), and :meth:`array` reads one column
    to a 1-D array. The concrete class supplies its column names and how it builds a
    column view / sub-table over its own rows; this keeps that surface identical
    wherever a table comes from.
    """

    __slots__ = ()

    @property
    def columns(self) -> list[str]:
        """Names of this table's columns, in selection order."""
        raise NotImplementedError

    def _column_view(self, name: str) -> ColumnView:
        raise NotImplementedError

    def _sub_table(self, names: list[str]) -> TableView:
        raise NotImplementedError

    def __getitem__(self, key: Any) -> ColumnView | TableView:
        """``table[name]`` → a column view; ``table[[names]]`` → a narrowed table."""
        if isinstance(key, str):
            validate_columns(self.columns, [key])
            return self._column_view(key)
        if isinstance(key, (list, tuple)):
            return self._sub_table(resolve_select(self.columns, tuple(key)))
        raise TypeError(
            f"index a table by a column name (→ a column) or a list of names (→ a "
            f"sub-table); got {type(key).__name__}. To select rows, index the reader "
            "(ds[rows, cols]) or filter with where() / query()."
        )

    def array(self, name: str, copy: bool = True) -> np.ndarray:
        """Materialize one column as a 1-D array -- the shortcut for ``table[name].array()``."""
        return self._column_view(name).array(copy=copy)


class TableView(_ColumnTable, _BaseView):
    """View of multiple columns produced by any non-string indexing.

    Materializes through :meth:`array` (one column, by name), :meth:`dict`,
    :meth:`recarray`, or :meth:`frame`. There is no no-argument ``array()`` --
    several columns generally have different dtypes and cannot be packed into a
    single homogeneous ndarray; read one column by name (``view['col']`` /
    ``view.array('col')``) or use :meth:`recarray` for all of them. Indexing by a
    name projects a column (``view['col']`` → a :class:`ColumnView`) and by a list of
    names a sub-table (``view[['a', 'b']]``, the same as :meth:`select`); a row
    selector (``view[:100]``, ``view[idx]``, ``view[mask]``) narrows the rows, composed
    onto the view's selection, and ``view[rows, cols]`` does both.
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

    def __getitem__(self, key: Any) -> ColumnView | TableView:
        """Narrow this table by columns or rows.

        ``view['col']`` / ``view[['a', 'b']]`` project columns (as before); a row
        selector -- ``view[:100]``, ``view[idx]``, ``view[mask]`` -- narrows the rows,
        composed onto the view's current selection; and ``view[rows, cols]`` does both.
        So ``ds[:1000, cols][:10]`` equals ``ds[:10, cols]``.
        """
        if _is_column_key(key):
            return super().__getitem__(key)  # _ColumnTable: project columns
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError(f"Expected at most 2 elements in indexing tuple; got {len(key)}.")
            rows, cols = key
            composed = self._compose_rows(rows)
            if isinstance(cols, str):
                validate_columns(self._column_names, [cols])
                return ColumnView(self._store, composed, cols)
            if isinstance(cols, (list, tuple)):
                chosen = resolve_select(self._column_names, tuple(cols))
                return TableView(self._store, composed, chosen)
            raise IndexError(
                f"Column selector must be a string or list of strings; got {type(cols).__name__}."
            )
        return TableView(self._store, self._compose_rows(key), self._column_names)

    @property
    def columns(self) -> list[str]:
        """Names of the columns selected by this view, in selection order."""
        return list(self._column_names)

    def _column_view(self, name: str) -> ColumnView:
        return ColumnView(self._store, self._row_part, name)

    def _sub_table(self, names: list[str]) -> TableView:
        return TableView(self._store, self._row_part, names)

    def _edit_columns(self) -> list[str]:
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
            DataFrame whose columns are zero-copy views, forwarding the same
            conditions and lifetime as :meth:`dict` (raising rather than copying
            when any column cannot be viewed).

        Returns
        -------
        pandas.DataFrame
            Columns are in selection order with their stored dtypes preserved.
        """
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

    def _interop_target(self) -> tuple[_ReaderBase, list[str], Any, bool]:
        """These columns and their resolved rows -- the export seam (see InteropMixin)."""
        return self._store, self._column_names, self._resolve_row_indexer(), False

    def head(self, n: int | None = None) -> Preview:
        """First ``n`` rows of the selection as a previewable peek (default config rows)."""
        return self._preview(self._head_rows(self._preview_n(n)))

    def tail(self, n: int | None = None) -> Preview:
        """Last ``n`` rows of the selection, as a previewable peek."""
        return self._preview(self._tail_rows(self._preview_n(n)))

    def _row_width(self) -> int:
        return row_width(self._store, self._column_names)

    def _preview(self, indexer: RowIndexer) -> Preview:
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
