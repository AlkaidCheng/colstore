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

from . import kernels
from ._pandas import _make_dataframe_no_consolidate
from ._query import _Expr, parse_query, validate_predicate
from ._render import Preview
from .view import (
    ColumnView,
    TableView,
    build_preview,
    resolve_drop,
    resolve_preview_n,
    resolve_select,
    row_width,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

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

    @abc.abstractmethod
    def _check_open(self) -> None:
        """Raise ``ValueError`` if the store is closed."""

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

    # ---- Column selection ----------------------------------------------

    def select(self, *columns: str) -> TableView:
        """Lazy view of the named ``columns`` (in the given order), all rows.

        Names are validated immediately; unknown names raise ``KeyError``. A
        single name still yields a (one-column) ``TableView`` -- use ``ds[name]``
        for a 1-D ``ColumnView``.
        """
        return TableView(self, None, resolve_select(self.columns, columns))

    def drop(self, *columns: str) -> TableView:
        """Lazy view of all columns except the named ``columns``, all rows."""
        return TableView(self, None, resolve_drop(self.columns, columns))

    # ---- Peeking -------------------------------------------------------

    def head(self, n: int | None = None) -> Preview:
        """First ``n`` rows of the store as a previewable peek (default config rows)."""
        rows = self._preview_n(n)
        return self._preview(None, slice(0, max(0, rows)))

    def tail(self, n: int | None = None) -> Preview:
        """Last ``n`` rows of the store, as a previewable peek."""
        rows = self._preview_n(n)
        return self._preview(None, slice(max(0, self.n_rows - rows), self.n_rows))

    def _repr_html_(self) -> str | None:
        """Rich Jupyter display: a head preview under a shape caption.

        Returns ``None`` -- so Jupyter uses the text repr -- only if the preview
        can't be built.
        """
        try:
            return self._preview(self.n_rows, slice(0, max(0, self._preview_n(None))))._repr_html_()
        except Exception:
            return None

    def _preview_n(self, n: int | None) -> int:
        """Resolve a whole-store preview row count, warning if it would be large."""
        return resolve_preview_n(n, self.n_rows, row_width(self, self.columns))

    def _preview(self, total_rows: int | None, rows: slice) -> Preview:
        """A whole-store preview over a concrete ``rows`` slice."""
        return build_preview(type(self).__name__, total_rows, self, self.columns, rows)

    # ---- Materializers -------------------------------------------------

    def recarray(self) -> NDArray[Any]:
        """Materialize the whole store as a structured (record) ndarray.

        One field per column, in stored order; ``result[name]`` is the column.
        See :meth:`_build_recarray` for how the columns are interleaved.
        """
        self._check_open()
        names = self.columns
        if not names or self.n_rows == 0:
            record_dtype = np.dtype([(name, self._native_dtype(name)) for name in names])
            return np.empty(self.n_rows, dtype=record_dtype)
        return self._build_recarray(None, names)

    def array(self, name: str, copy: bool = True) -> NDArray[Any]:
        """Materialize one column as a 1-D array -- the shortcut for ``self[name].array()``.

        Parameters
        ----------
        name : str
            Column to read.
        copy : bool, optional
            ``True`` (default): an owning array. ``False``: a READ-ONLY zero-copy
            view backed by the open memmap, under the same conditions and lifetime
            as :meth:`~colstore.view.ColumnView.array` (raising rather than copying
            when a view cannot be given).

        Returns
        -------
        numpy.ndarray
            1-D array of the column in its stored dtype.
        """
        self._check_open()
        return self[name].array(copy=copy)

    def _build_recarray(self, row_indexer: Any, column_names: list[str]) -> NDArray[Any]:
        """Interleave a row selection of the named columns into a record array.

        Shared by :meth:`recarray` (whole store) and ``TableView.recarray`` (a
        row/column subset). With the gather extension, each column's contiguous
        native source -- a zero-copy view where one exists, else a materialized
        gather -- feeds the parallel ``interleave_records`` kernel; without it, a
        parallel multi-column gather is assembled per field. The assembly itself
        lives in :func:`colstore.kernels.interleave_record_array`. Assumes at
        least one column.
        """
        record_dtype = np.dtype([(name, self._native_dtype(name)) for name in column_names])
        if kernels.cpp_available():
            sources = [self._contiguous_native_source(name, row_indexer) for name in column_names]
        else:
            column_data = self._gather_many(column_names, row_indexer)
            sources = [column_data[name] for name in column_names]
        return kernels.interleave_record_array(column_names, sources, record_dtype)

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
    def __getitem__(self, key: int | slice | list[Any] | NDArray[Any] | _Expr) -> TableView: ...
    @overload
    def __getitem__(self, key: tuple[Any, list[str] | tuple[str, ...]]) -> TableView: ...

    def __getitem__(self, key: Any) -> ColumnView | TableView:
        row_part, column_names, is_single_column = self._parse_key(key)
        if isinstance(row_part, _Expr):
            validate_predicate(row_part, frozenset(self._column_dtypes), self._query_probe)
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
            column_names = self.columns
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

    def query(
        self,
        expression: str | _Expr,
        *,
        columns: list[str] | tuple[str, ...] | None = None,
        params: dict[str, Any] | None = None,
        lazy: bool = True,
    ) -> TableView:
        """Select the rows matching a predicate; returns a lazy view by default.

        ``expression`` is either a pandas-style condition string -- e.g.
        ``"energy > 100 and -2.5 < eta < 2.5"`` -- or a :func:`~colstore.col`
        expression (``(col("pt") > 30) & col("ok")``). The predicate is parsed and
        validated eagerly (an unknown column, an unsupported construct, or a
        non-boolean condition raises :class:`~colstore.QueryError` immediately,
        reading no data), but evaluation is deferred: the result is a lazy
        :class:`~colstore.view.TableView` whose row selection is computed -- and
        whose selected columns are materialized -- only when you call
        :meth:`~colstore.view.TableView.evaluate`, ``.frame()`` / ``.dict()`` /
        ``.recarray()`` / ``.array()``, or another consumer. Pass ``lazy=False``
        to resolve the row mask immediately (equivalent to calling
        :meth:`~colstore.view.TableView.evaluate` on the result).

        The string grammar is a strict whitelist evaluated without ``eval``:
        column names, numeric / string / bool literals, comparisons (including
        chained ``a < x < b``), the boolean operators (``and`` / ``or`` / ``not``
        and ``& | ~`` -- parenthesize the bitwise forms, which bind tighter than
        comparison), arithmetic, and ``in`` / ``not in`` membership. ``@name``
        resolves from ``params`` (``query("pt > @cut", params={"cut": 30})``);
        the calling frame is never inspected.

        Parameters
        ----------
        expression : str or column expression
            The predicate. Must reduce to a per-row boolean condition.
        columns : list of str, optional
            Project the result to these columns; the default keeps all columns.
        params : dict, optional
            Values for ``@name`` references (string ``expression`` only).
        lazy : bool, optional
            ``True`` (default) returns a lazy view; ``False`` resolves the row
            mask now and returns a view over it.

        Returns
        -------
        TableView
            A lazy view of the matching rows (and selected columns).
        """
        predicate = (
            expression
            if isinstance(expression, _Expr)
            else parse_query(expression, frozenset(self.columns), params)
        )
        view = self[predicate] if columns is None else self[predicate, list(columns)]
        return view if lazy else view.evaluate()

    def where(self, condition: _Expr) -> TableView:
        """Select rows where a :func:`~colstore.col` expression is true (lazy).

        ``ds.where(col("pt") > 30)`` is the explicit form of
        ``ds[col("pt") > 30]``: a lazy :class:`~colstore.view.TableView` of the
        matching rows. Combine conditions with ``& | ~``.
        """
        return self[condition]

    def _read_query_column(self, name: str) -> NDArray[Any]:
        """Read one whole column as an array, for predicate evaluation."""
        return self[name].array()

    def _query_probe(self, name: str) -> NDArray[Any]:
        """An empty typed array for one column, for the data-free query dtype probe."""
        return np.empty(0, dtype=self._native_dtype(name))

    # ---- Whole-store materialization shortcuts -------------------------
    #
    # ``dict`` / ``frame`` are kept at the bottom of the class so the method
    # named ``dict`` does not shadow the builtin ``dict`` in the annotation
    # scope of earlier methods (mypy resolves annotations in declaration order
    # against the class namespace). ``recarray`` / ``array`` sit with the other
    # materializers above -- their names collide with nothing.

    def dict(self, copy: bool = True) -> dict[str, NDArray[Any]]:
        """Materialize the whole store as a dict mapping column name to ndarray.

        Parameters
        ----------
        copy : bool, optional
            ``True`` (default): owning arrays. ``False``: READ-ONLY zero-copy
            views over the store; supported only on single-record stores with
            native-byte-order dtypes, raising ``ValueError`` otherwise. The views
            stay valid after :meth:`close`.

        Returns
        -------
        dict[str, numpy.ndarray]
            Arrays in on-disk column order, stored dtypes preserved (native byte
            order).
        """
        self._check_open()
        names = self.columns
        return self._view_many(names, None) if not copy else self._gather_many(names, None)

    def frame(self, copy: bool = True) -> pd.DataFrame:
        """Materialize the whole store as a pandas DataFrame.

        Columns are in on-disk order with their stored dtypes preserved.
        ``copy=False`` returns a READ-ONLY frame whose columns are zero-copy views,
        under the same conditions and lifetime as :meth:`dict` (raising rather than
        copying when any column cannot be viewed).
        """
        return _make_dataframe_no_consolidate(self.dict(copy=copy))
