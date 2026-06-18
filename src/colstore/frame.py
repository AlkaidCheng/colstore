"""Deferred column expression graph for the lazy editing layer.

This module is the computation core that the column-editing frame builds on.
Nothing here reads or writes a column body; it records *what* to compute so the
caller can evaluate it later, one row range at a time.

An :class:`Expr` is a node in a per-column computation graph. Leaf nodes name a
data source -- :class:`NativeColumn` (a column in an open store, read by range),
:class:`MemoryColumn` (an in-memory array), or :class:`ConstColumn` (a scalar
broadcast to any length). :class:`UFunc` nodes apply an elementwise NumPy ufunc
to other nodes. Python operators and whitelisted NumPy ufuncs invoked on an
``Expr`` return new ``Expr`` nodes instead of computing, so a whole-column
transformation such as ``(x + y) * 2`` is captured as a graph::

    UFunc(multiply, [UFunc(add, [x, y]), 2])

Only *row-independent elementwise* operations are representable: each output row
depends solely on the input row at the same position. That is exactly what makes
range-at-a-time evaluation correct, so range-coupling operations -- reductions,
accumulations, sorts -- are rejected when the graph is built, not silently
mis-evaluated. Reductions and accumulations arrive through ``__array_ufunc__``
with a ``method`` other than ``"__call__"`` and are refused there; ufuncs outside
:data:`_ALLOWED_UFUNCS` are refused too.

Evaluation (:func:`evaluate`) walks the graph for a half-open row range against a
caller-supplied ``memo``. The memo is keyed on each node's structural key, so a
subexpression that appears more than once -- whether repeated within one column
or shared across several output columns evaluated under the same memo -- is read
or computed once. The structural key also makes the zero-length dtype probe in
:func:`result_dtype` free: evaluating any node over an empty range runs the whole
graph on empty typed arrays, yielding the output dtype with no data read and no
value-dependent surprises.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ._base import _ReaderBase
    from .reader import ColStoreReader

__all__ = [
    "ColStoreFrame",
    "ConstColumn",
    "Expr",
    "MemoryColumn",
    "NativeColumn",
    "UFunc",
    "as_expr",
    "declared_length",
    "evaluate",
    "result_dtype",
    "validate_length",
]

# Elementwise ufuncs allowed inside a column expression. Membership is the
# contract for "row-independent": every entry maps each output element to the
# input element(s) at the same position. Reductions/accumulations are rejected
# separately (by ufunc method), so only the elementwise *call* of these is ever
# reached. Extend deliberately -- anything added here must preserve the
# one-row-in, one-row-out property that range-at-a-time evaluation relies on.
_ALLOWED_UFUNCS: frozenset[np.ufunc] = frozenset(
    {
        np.add,
        np.subtract,
        np.multiply,
        np.true_divide,
        np.floor_divide,
        np.mod,
        np.power,
        np.negative,
        np.positive,
        np.absolute,
        np.greater,
        np.greater_equal,
        np.less,
        np.less_equal,
        np.equal,
        np.not_equal,
        np.sin,
        np.cos,
        np.tan,
        np.arcsin,
        np.arccos,
        np.arctan,
        np.exp,
        np.log,
        np.log2,
        np.log10,
        np.sqrt,
        np.cbrt,
        np.square,
        np.rint,
        np.floor,
        np.ceil,
        np.trunc,
        np.maximum,
        np.minimum,
        np.logical_and,
        np.logical_or,
        np.logical_not,
        np.bitwise_and,
        np.bitwise_or,
        np.bitwise_xor,
        np.invert,
    }
)

# Python/NumPy scalar operand types accepted alongside expressions.
_SCALAR_TYPES = (int, float, bool, complex)


def _is_scalar_operand(value: Any) -> bool:
    """Return whether ``value`` is a scalar usable directly inside an expression.

    Accepts Python numeric scalars, NumPy scalars (``np.generic``), and 0-d
    arrays. A 0-d array carries a concrete dtype and broadcasts like a scalar,
    so it follows the same promotion rules as the equivalent NumPy scalar.
    """
    if isinstance(value, _SCALAR_TYPES):
        return True
    if isinstance(value, np.generic):
        return True
    return isinstance(value, np.ndarray) and value.ndim == 0


def _check_operand(value: Any) -> None:
    """Reject operands that cannot appear inside a column expression.

    Expressions and scalars pass. A 1-D-or-higher ``ndarray`` is rejected with a
    pointed message: a raw array has no place in the structural key and no row
    alignment guarantee, so the caller must attach it as a column first and
    reference that column instead.
    """
    if isinstance(value, Expr) or _is_scalar_operand(value):
        return
    if isinstance(value, np.ndarray):
        raise TypeError(
            "cannot use a raw ndarray inside a column expression; attach it as a "
            "column first and reference that column."
        )
    raise TypeError(f"unsupported operand for a column expression: {type(value).__name__!r}.")


class Expr:
    """A node in a deferred column computation graph.

    Operators and whitelisted NumPy ufuncs called on an ``Expr`` build new
    ``Expr`` nodes rather than computing; materialization is deferred to
    :func:`evaluate`. Instances are intentionally not hashable and have no truth
    value -- ``==`` and ``<`` build comparison nodes, and a node is not a
    boolean -- so they cannot be used as dict keys or in ``if`` tests by mistake;
    :attr:`_key` carries the structural identity used for memoization instead.
    """

    __slots__ = ()

    # Structural identity: a plain hashable tuple set by every subclass. Two
    # nodes with equal keys denote the same computation over the same sources,
    # which is what lets the evaluator's memo deduplicate them.
    _key: tuple[Any, ...]

    # Make ``ndarray <op> expr`` defer to us rather than treating the expr as an
    # array-like operand.
    __array_priority__ = 1_000_000

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Expr:
        if method != "__call__":
            raise TypeError(
                f"{ufunc.__name__}.{method} is a reduction or accumulation; column "
                "expressions support only elementwise (row-independent) operations."
            )
        if kwargs.get("out") is not None:
            raise TypeError("out= is not supported when building a column expression.")
        if ufunc not in _ALLOWED_UFUNCS:
            raise TypeError(f"ufunc {ufunc.__name__!r} is not allowed in a column expression.")
        for value in inputs:
            _check_operand(value)
        return UFunc(ufunc, inputs)

    def __array__(self, dtype: Any = None) -> NDArray[Any]:
        raise TypeError(
            "a column expression cannot be converted to an array directly; attach "
            "it as a column and read it back, or call evaluate() over a row range."
        )

    def __bool__(self) -> bool:
        raise TypeError(
            "the truth value of a column expression is ambiguous; build the "
            "comparison and attach it as a column instead of testing it."
        )

    def _binop(self, other: Any, ufunc: np.ufunc, *, reflected: bool = False) -> Expr:
        _check_operand(other)
        inputs = (other, self) if reflected else (self, other)
        return UFunc(ufunc, inputs)

    def _unop(self, ufunc: np.ufunc) -> Expr:
        return UFunc(ufunc, (self,))

    # -- arithmetic --
    def __add__(self, other: Any) -> Expr:
        return self._binop(other, np.add)

    def __radd__(self, other: Any) -> Expr:
        return self._binop(other, np.add, reflected=True)

    def __sub__(self, other: Any) -> Expr:
        return self._binop(other, np.subtract)

    def __rsub__(self, other: Any) -> Expr:
        return self._binop(other, np.subtract, reflected=True)

    def __mul__(self, other: Any) -> Expr:
        return self._binop(other, np.multiply)

    def __rmul__(self, other: Any) -> Expr:
        return self._binop(other, np.multiply, reflected=True)

    def __truediv__(self, other: Any) -> Expr:
        return self._binop(other, np.true_divide)

    def __rtruediv__(self, other: Any) -> Expr:
        return self._binop(other, np.true_divide, reflected=True)

    def __floordiv__(self, other: Any) -> Expr:
        return self._binop(other, np.floor_divide)

    def __rfloordiv__(self, other: Any) -> Expr:
        return self._binop(other, np.floor_divide, reflected=True)

    def __mod__(self, other: Any) -> Expr:
        return self._binop(other, np.mod)

    def __rmod__(self, other: Any) -> Expr:
        return self._binop(other, np.mod, reflected=True)

    def __pow__(self, other: Any) -> Expr:
        return self._binop(other, np.power)

    def __rpow__(self, other: Any) -> Expr:
        return self._binop(other, np.power, reflected=True)

    def __neg__(self) -> Expr:
        return self._unop(np.negative)

    def __pos__(self) -> Expr:
        return self._unop(np.positive)

    def __abs__(self) -> Expr:
        return self._unop(np.absolute)

    # -- comparisons --
    def __gt__(self, other: Any) -> Expr:
        return self._binop(other, np.greater)

    def __ge__(self, other: Any) -> Expr:
        return self._binop(other, np.greater_equal)

    def __lt__(self, other: Any) -> Expr:
        return self._binop(other, np.less)

    def __le__(self, other: Any) -> Expr:
        return self._binop(other, np.less_equal)

    def __eq__(self, other: Any) -> Expr:  # type: ignore[override]
        return self._binop(other, np.equal)

    def __ne__(self, other: Any) -> Expr:  # type: ignore[override]
        return self._binop(other, np.not_equal)

    # -- bitwise / boolean-mask combinators --
    def __and__(self, other: Any) -> Expr:
        return self._binop(other, np.bitwise_and)

    def __rand__(self, other: Any) -> Expr:
        return self._binop(other, np.bitwise_and, reflected=True)

    def __or__(self, other: Any) -> Expr:
        return self._binop(other, np.bitwise_or)

    def __ror__(self, other: Any) -> Expr:
        return self._binop(other, np.bitwise_or, reflected=True)

    def __xor__(self, other: Any) -> Expr:
        return self._binop(other, np.bitwise_xor)

    def __rxor__(self, other: Any) -> Expr:
        return self._binop(other, np.bitwise_xor, reflected=True)

    def __invert__(self) -> Expr:
        return self._unop(np.invert)

    # -- composed helpers for non-ufunc NumPy functions --
    def round(self, decimals: int = 0) -> Expr:
        """Round to ``decimals`` places, expressed with whitelisted ufuncs.

        ``np.round`` is not a ufunc, so rather than route it through a separate
        protocol it is expanded into ``rint`` (and a scale/unscale pair when
        ``decimals`` is nonzero), matching ``numpy.round``'s own definition.
        """
        if decimals == 0:
            return self._unop(np.rint)
        factor = 10.0**decimals
        scaled = UFunc(np.multiply, (self, factor))
        rounded = UFunc(np.rint, (scaled,))
        return UFunc(np.true_divide, (rounded, factor))

    def clip(self, low: Any = None, high: Any = None) -> Expr:
        """Clamp to ``[low, high]`` using ``maximum``/``minimum``; either bound
        may be ``None`` to leave that side unbounded."""
        node: Expr = self
        if low is not None:
            _check_operand(low)
            node = UFunc(np.maximum, (node, low))
        if high is not None:
            _check_operand(high)
            node = UFunc(np.minimum, (node, high))
        return node

    def compute(self, n_rows: int | None = None) -> NDArray[Any]:
        """Eagerly materialize this column as a NumPy array.

        Evaluates the expression over its full length with a fresh memo --
        useful for inspecting a column mid-edit without writing a file.
        ``n_rows`` is needed only for an all-constant expression, whose length
        is otherwise indeterminate; for any expression with a sized leaf the
        length is inferred.
        """
        length = declared_length(self) if n_rows is None else n_rows
        if length is None:
            raise ValueError("cannot infer the length of an all-constant column; pass n_rows.")
        return evaluate(self, 0, length, {})


class _Leaf(Expr):
    """A graph leaf: a named data source materialized by row range.

    Concrete leaves know their output dtype without reading data, report a row
    length (or ``None`` for length-agnostic constants), and read a half-open row
    range. The zero-length read is required to be data-free so the dtype probe
    never touches a backing store.
    """

    __slots__ = ()

    @property
    def dtype(self) -> np.dtype[Any]:
        raise NotImplementedError

    def _length(self) -> int | None:
        raise NotImplementedError

    def _read(self, start: int, stop: int) -> NDArray[Any]:
        raise NotImplementedError


class NativeColumn(_Leaf):
    """A leaf backed by a column in an open store, read as a contiguous range.

    Holds the store and the column's *raw on-disk name* -- never a logical
    rename target -- so a transformation referencing this column survives any
    later renaming of the schema key it is bound to. The dtype is taken from the
    store's manifest (native byte order) without reading data, and a zero-length
    read returns an empty typed array rather than calling the store, so the dtype
    probe stays data-free.
    """

    __slots__ = ("_dtype", "_key", "_name", "_store")

    def __init__(self, store: _ReaderBase, name: str) -> None:
        if name not in store.dtypes:
            raise KeyError(f"column {name!r} is not in the store; have {list(store.dtypes)}.")
        self._store = store
        self._name = name
        self._dtype = store.dtypes[name]
        self._key = ("native", id(store), name)

    @property
    def name(self) -> str:
        """Raw on-disk column name this leaf reads."""
        return self._name

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def _length(self) -> int | None:
        return self._store.n_rows

    def _read(self, start: int, stop: int) -> NDArray[Any]:
        if stop <= start:
            return np.empty(0, dtype=self._dtype)
        return self._store._gather_one(self._name, slice(start, stop))

    def _fill_into(self, out: NDArray[Any], start: int, stop: int) -> None:
        """Write rows ``[start, stop)`` of the backing column straight into ``out``.

        The read counterpart to :meth:`_read` that fills the caller's array (e.g.
        a region of a memory-mapped output) instead of allocating a fresh one, so
        a passthrough column reaches its destination in a single copy from the
        source rather than via an intermediate array.
        """
        if stop > start:
            self._store._gather_slice_into(out, self._name, start, stop)


class MemoryColumn(_Leaf):
    """A leaf backed by an in-memory 1-D array.

    By default the array is held by reference, so a later in-place mutation is
    reflected when the column is materialized; pass ``copy=True`` to snapshot it
    at attach time. Identity (not contents) forms the structural key, so a
    snapshot is a distinct source from the array it was copied from.
    """

    __slots__ = ("_array", "_key")

    def __init__(self, array: NDArray[Any], copy: bool = False) -> None:
        arr = np.array(array, copy=True) if copy else np.asarray(array)
        if arr.ndim != 1:
            raise ValueError(f"a column array must be 1-D; got shape {arr.shape}.")
        self._array = arr
        self._key = ("memory", id(self._array))

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._array.dtype

    def _length(self) -> int | None:
        return int(self._array.shape[0])

    def _read(self, start: int, stop: int) -> NDArray[Any]:
        return self._array[start:stop]


class ConstColumn(_Leaf):
    """A leaf that broadcast-fills a scalar to whatever row range is requested.

    Length-agnostic by construction (``_length`` is ``None``): it adopts the
    length of whatever it is combined with, and on its own fills the frame's row
    count. The dtype is the scalar's, or an explicit override.
    """

    __slots__ = ("_dtype", "_key", "_value")

    def __init__(self, value: Any, dtype: Any = None) -> None:
        if not _is_scalar_operand(value):
            raise TypeError(
                f"a constant column value must be a scalar; got {type(value).__name__!r}."
            )
        self._value = value
        self._dtype = np.dtype(dtype) if dtype is not None else np.asarray(value).dtype
        self._key = ("const", repr(value), str(self._dtype))

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def _length(self) -> int | None:
        return None

    def _read(self, start: int, stop: int) -> NDArray[Any]:
        return np.full(stop - start, self._value, dtype=self._dtype)


class UFunc(Expr):
    """An internal node applying an elementwise NumPy ufunc to its inputs.

    Inputs are a mix of child :class:`Expr` nodes and scalars (scalars are kept
    verbatim and passed straight to the ufunc at evaluation). Operand order is
    preserved so non-commutative operations such as subtraction are faithful.
    """

    __slots__ = ("_inputs", "_key", "_ufunc")

    def __init__(self, ufunc: np.ufunc, inputs: tuple[Any, ...]) -> None:
        self._ufunc = ufunc
        self._inputs = tuple(inputs)
        self._key = (
            "ufunc",
            ufunc.__name__,
            tuple(
                child._key if isinstance(child, Expr) else ("scalar", repr(child))
                for child in self._inputs
            ),
        )


def evaluate(
    node: Expr,
    start: int,
    stop: int,
    memo: dict[tuple[Any, ...], NDArray[Any]],
) -> NDArray[Any]:
    """Materialize ``node`` over the half-open row range ``[start, stop)``.

    ``memo`` maps a node's structural key to its already-computed result for this
    range; pass the *same* memo across every column of one batch so a shared
    subexpression is read or computed once, and a *fresh* memo per batch so each
    batch's working set is released. A zero-length range (``start == stop``) runs
    the whole graph on empty typed arrays without touching any backing store,
    which is how :func:`result_dtype` recovers the output dtype for free.
    """
    cached = memo.get(node._key)
    if cached is not None:
        return cached
    if isinstance(node, _Leaf):
        result = node._read(start, stop)
    elif isinstance(node, UFunc):
        args = [
            evaluate(child, start, stop, memo) if isinstance(child, Expr) else child
            for child in node._inputs
        ]
        result = node._ufunc(*args)
    else:  # pragma: no cover - the Expr hierarchy is closed to _Leaf and UFunc
        raise TypeError(f"cannot evaluate node of type {type(node).__name__!r}.")
    memo[node._key] = result
    return result


def result_dtype(node: Expr) -> np.dtype[Any]:
    """Output dtype of ``node`` via a zero-length evaluation; reads no data."""
    return evaluate(node, 0, 0, {}).dtype


def _iter_leaves(node: Expr) -> Iterator[_Leaf]:
    """Yield the leaf nodes reachable from ``node`` (with repetition)."""
    if isinstance(node, _Leaf):
        yield node
    elif isinstance(node, UFunc):
        for child in node._inputs:
            if isinstance(child, Expr):
                yield from _iter_leaves(child)


def fusible_passthroughs(specs: dict[str, Expr]) -> dict[str, NativeColumn]:
    """Output columns a streaming sink can fill straight from their source.

    A column qualifies when its expression is a bare :class:`NativeColumn` (a
    plain passthrough, no transform) whose source leaf is referenced by no other
    column. The second condition preserves the evaluator's shared-subexpression
    reuse: a native read feeding another column too is left on the memoized path
    so it is read once, not once per consumer.
    """
    counts = Counter(leaf._key for spec in specs.values() for leaf in _iter_leaves(spec))
    return {
        name: spec
        for name, spec in specs.items()
        if isinstance(spec, NativeColumn) and counts[spec._key] == 1
    }


def declared_length(node: Expr) -> int | None:
    """Common row length of ``node``'s sized leaves, or ``None`` if it has none.

    Constant leaves are length-agnostic and excluded. If the remaining leaves
    disagree the expression is internally inconsistent and a ``ValueError`` is
    raised; an all-constant expression returns ``None`` (it adopts whatever
    length it is later required to fill).
    """
    lengths = {length for leaf in _iter_leaves(node) if (length := leaf._length()) is not None}
    if not lengths:
        return None
    if len(lengths) > 1:
        raise ValueError(f"column expression mixes leaves of different lengths: {sorted(lengths)}.")
    return next(iter(lengths))


def validate_length(node: Expr, n_rows: int) -> None:
    """Raise if ``node`` cannot produce exactly ``n_rows`` rows.

    A length-agnostic expression (all-constant leaves) is accepted: it
    broadcast-fills ``n_rows``. A sized expression must match ``n_rows`` exactly.
    A length-1 array does *not* broadcast here -- only true scalars do -- so a
    length-1 column against a wider frame is rejected, matching pandas. NumPy
    would not catch this on its own: it raises only at evaluation, silently
    length-1-broadcasts, and is blind to length under the zero-length dtype
    probe, so length is checked here, eagerly, before any data is read.
    """
    length = declared_length(node)
    if length is None:
        return
    if length != n_rows:
        raise ValueError(f"column length {length} does not match the frame's row count {n_rows}.")


def as_expr(value: Any, *, copy: bool = False) -> Expr:
    """Coerce a user-supplied column value into an :class:`Expr`.

    An ``Expr`` passes through. An ``ndarray`` (or array-like) becomes a
    :class:`MemoryColumn` -- held by reference unless ``copy=True``. A scalar (or
    0-d array) becomes a :class:`ConstColumn`. This is the bridge from
    assignment syntax to the graph; native columns are constructed by the frame
    from its source store, not here.
    """
    if isinstance(value, Expr):
        return value
    if _is_scalar_operand(value):
        return ConstColumn(value)
    array = np.asarray(value)
    if array.ndim == 0:
        return ConstColumn(array)
    return MemoryColumn(array, copy=copy)


class ColStoreFrame:
    """A mutable, deferred editing view over an opened store's columns.

    Created by :meth:`edit` on an opened reader or dataset. A frame holds an
    ordered mapping of output column name to an expression (:class:`Expr`);
    opening one seeds it with a native-passthrough leaf per source column.
    Indexing returns the expression for a column, which composes with operators
    and whitelisted NumPy ufuncs to build transformations; assignment, deletion,
    and renaming edit the mapping. Nothing is read or written until :meth:`write`,
    which streams the result to a new file and returns a reader for it. The source
    store is never modified.

    Assignment holds arrays by reference; pass ``copy=True`` to :meth:`assign`
    to snapshot them instead. A column length is checked eagerly on assignment
    against the frame's fixed row count.
    """

    __slots__ = ("_columns", "_n_rows", "_store")

    def __init__(self, store: _ReaderBase) -> None:
        self._store = store
        self._n_rows = store.n_rows
        self._columns: dict[str, Expr] = {name: NativeColumn(store, name) for name in store.columns}

    @property
    def n_rows(self) -> int:
        """Fixed row count this frame writes, inherited from the source store."""
        return self._n_rows

    @property
    def columns(self) -> list[str]:
        """Output column names, in order."""
        return list(self._columns)

    def __len__(self) -> int:
        return len(self._columns)

    def __iter__(self) -> Iterator[str]:
        return iter(self._columns)

    def __contains__(self, name: object) -> bool:
        return name in self._columns

    def __getitem__(self, name: str) -> Expr:
        try:
            return self._columns[name]
        except KeyError:
            raise KeyError(
                f"column {name!r} is not in the frame; have {list(self._columns)}."
            ) from None

    def __setitem__(self, name: str, value: Any) -> None:
        self._columns[name] = self._coerce(value, copy=False)

    def __delitem__(self, name: str) -> None:
        try:
            del self._columns[name]
        except KeyError:
            raise KeyError(
                f"column {name!r} is not in the frame; have {list(self._columns)}."
            ) from None

    def _coerce(self, value: Any, *, copy: bool) -> Expr:
        expr = as_expr(value, copy=copy)
        validate_length(expr, self._n_rows)
        return expr

    def assign(self, *, copy: bool = False, **new_columns: Any) -> ColStoreFrame:
        """Add or replace columns from keyword arguments; returns ``self``.

        Each value may be an expression, array, or scalar. Arrays are held by
        reference unless ``copy=True``. (A column literally named ``copy`` must
        be set with ``frame[name] = ...`` instead.)
        """
        coerced = {name: self._coerce(value, copy=copy) for name, value in new_columns.items()}
        self._columns.update(coerced)
        return self

    def with_columns(self, *, copy: bool = False, **new_columns: Any) -> ColStoreFrame:
        """Alias for :meth:`assign`, under a polars-style name."""
        return self.assign(copy=copy, **new_columns)

    def drop(self, *names: str) -> ColStoreFrame:
        """Remove one or more columns; returns ``self``."""
        for name in names:
            del self[name]
        return self

    def rename(self, columns: dict[str, str]) -> ColStoreFrame:
        """Rename columns, resolving all mappings simultaneously; returns ``self``.

        A swap such as ``{"a": "b", "b": "a"}`` exchanges the two names in one
        step. Renames change output names only -- the underlying data, and any
        expression already referencing a column, are unaffected. Raises if a
        source name is missing or the result would contain duplicate names.
        """
        missing = [src for src in columns if src not in self._columns]
        if missing:
            raise KeyError(f"cannot rename columns that are not in the frame: {missing}.")
        renamed: dict[str, Expr] = {}
        for name, expr in self._columns.items():
            target = columns.get(name, name)
            if target in renamed:
                raise ValueError(f"rename produces a duplicate column name {target!r}.")
            renamed[target] = expr
        self._columns = renamed
        return self

    def compute(self, name: str) -> NDArray[Any]:
        """Eagerly materialize one column as an array, without writing a file."""
        return self[name].compute(self._n_rows)

    def write(
        self, path: str | os.PathLike[str], *, memory_budget: int | None = None
    ) -> ColStoreReader:
        """Stream the edited columns to a new ``.cstore`` and return a reader.

        Evaluates every column one row range at a time into a new file (see
        :func:`colstore.format.write_dataset_streaming`); peak memory is bounded
        by ``memory_budget`` (bytes; ``None`` uses the configured default). The
        source store is not modified. Writing a frame with no columns is an
        error.
        """
        from .api import open as open_store
        from .format import write_dataset_streaming

        write_dataset_streaming(self._columns, self._n_rows, path, memory_budget=memory_budget)
        return open_store(path)
