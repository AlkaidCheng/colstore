"""Deferred column expression graph for the lazy editing layer.

This module is the computation core that the column-editing frame builds on.
Nothing here reads or writes a column body; it records *what* to compute so the
caller can evaluate it later, one row range at a time.

An :class:`Expr` is a node in a per-column computation graph. Leaf nodes name a
data source -- :class:`NativeColumn` (a column in an open store, read by range),
:class:`MemoryColumn` (an in-memory array), or :class:`ConstColumn` (a scalar
broadcast to any length). :class:`UFunc` nodes apply an elementwise NumPy ufunc
to other nodes. Python operators and elementwise NumPy ufuncs invoked on an
``Expr`` return new ``Expr`` nodes instead of computing, so a whole-column
transformation such as ``(x + y) * 2`` is captured as a graph::

    UFunc(multiply, [UFunc(add, [x, y]), 2])

Only *row-independent elementwise* operations are representable: each output row
depends solely on the input row at the same position. That is exactly what makes
range-at-a-time evaluation correct, so range-coupling operations -- reductions,
accumulations, sorts -- are rejected when the graph is built, not silently
mis-evaluated. Reductions and accumulations arrive through ``__array_ufunc__``
with a ``method`` other than ``"__call__"`` and are refused there. An elementwise
call of a ufunc is admitted only when it maps one input row to one output row;
multi-output ufuncs (``numpy.modf``) and generalized ufuncs that mix rows across a
core dimension (``numpy.matmul``) are refused by their ``nout`` and ``signature``.

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
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from . import config, kernels
from ._pandas import _make_dataframe_no_consolidate
from ._query import QueryError, _Expr, parse_query
from ._render import render_table_html, render_table_text
from ._sizes import resolve_batch_rows

if TYPE_CHECKING:
    import pandas as pd

    from ._base import _ReaderBase
    from .reader import ColStoreReader

__all__ = [
    "ColStoreFrame",
    "ConstColumn",
    "CutInfo",
    "CutflowReport",
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
    if isinstance(value, _Expr):
        raise TypeError(
            "cannot use a col() expression as an operand of a frame (cf[...]) expression; "
            "build the expression entirely with col(), or replace the col() reference with "
            "the matching cf[...] column."
        )
    if isinstance(value, np.ndarray):
        raise TypeError(
            "cannot use a raw ndarray inside a column expression; attach it as a "
            "column first and reference that column."
        )
    raise TypeError(f"unsupported operand for a column expression: {type(value).__name__!r}.")


class Expr:
    """A node in a deferred column computation graph.

    Operators and elementwise NumPy ufuncs called on an ``Expr`` build new
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
        if ufunc.nout != 1:
            raise TypeError(
                f"ufunc {ufunc.__name__!r} returns multiple outputs, which a column "
                "expression cannot represent; use single-output operations."
            )
        if ufunc.signature is not None:
            raise TypeError(
                f"ufunc {ufunc.__name__!r} is a generalized ufunc that mixes rows across "
                "a core dimension; column expressions support only elementwise "
                "(row-independent) operations."
            )
        for value in inputs:
            _check_operand(value)
        return UFunc(ufunc, inputs)

    def __array_function__(
        self, func: Any, types: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Expr:
        builder = _ARRAY_FUNCTIONS.get(func)
        if builder is None:
            raise TypeError(
                f"numpy.{getattr(func, '__name__', func)} is not supported in a column "
                "expression; build it from elementwise operations instead."
            )
        return builder(*args, **kwargs)

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
        """Round to ``decimals`` places, expressed with elementwise ufuncs.

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

    def astype(self, dtype: Any) -> Expr:
        """Cast this column to ``dtype`` (anything ``numpy.dtype`` accepts), NumPy
        ``astype`` semantics."""
        return Cast(self, np.dtype(dtype))

    def where(self, cond: Any, other: Any = np.nan) -> Expr:
        """Keep this column where ``cond`` is true, else ``other``.

        ``cond`` is a boolean column expression and ``other`` a column or scalar;
        equivalent to ``numpy.where(cond, self, other)``.
        """
        return Where(cond, self, other)

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
        return evaluate(self, slice(0, length), {})


# A row selector: a slice (a contiguous half-open range), a 1-D integer index array
# naming the chosen rows in order, or a 1-D boolean mask over the source rows. A base
# mask is kept (see _normalize_base_rows) only when it will gather mask-natively -- a
# multi-record single-file store, or a multi-file dataset, at/above its density gate
# (default 0.0); a single-record store or a below-gate selection lowers it to indices at
# construction, and a where() predicate composes onto indices. The three forms are uniform
# under _row_count.
Rows = slice | np.ndarray

# A per-batch evaluation cache: a node's structural key -> its computed array, so a shared
# subexpression is read or computed once across the columns of one batch. Aliased at module
# scope because the ``dict`` builtin is shadowed by the frame's ``dict()`` terminal in the
# class body, where it cannot be used directly as a parameter annotation.
Memo = dict[tuple[Any, ...], NDArray[Any]]

# Per-column reuse buffers for copy=False iter_batches: output name -> a buffer the gather
# fills in place each batch, or ``None`` once a store is found not to honor the out= hint (a
# multi-file boundary read), so later batches of that column skip the dead allocation. A name
# absent from the dict has not been probed yet. Aliased like Memo (the shadowed ``dict``).
Buffers = dict[str, NDArray[Any] | None]


def _row_count(rows: Rows) -> int:
    """Rows a selector picks: a slice's span, a boolean mask's popcount, or an index
    array's length."""
    if isinstance(rows, slice):
        return max(0, int(rows.stop or 0) - int(rows.start or 0))
    if rows.dtype == bool:
        return int(np.count_nonzero(rows))
    return len(rows)


def _normalize_base_rows(
    rows: NDArray[Any] | None, n_rows: int, store: _ReaderBase
) -> NDArray[Any] | None:
    """Compact a base boolean mask, keeping it only when it will gather mask-natively.

    A single mask-native kernel (``gather_segment_mask``) gathers a 1-byte/row mask over
    a per-column segment table without materializing an index array; it serves both a
    multi-record single-file store (``_is_multi_record``) and a multi-file dataset. It
    keeps the mask only when its selected fraction is at or above the relevant density
    gate (default 0.0: kept at every density, since the kernel word-skips unselected runs
    and wins even on a sparse mask). A below-gate mask, and a contiguous single-record
    store, lower to indices once here, since the gather would otherwise repeat
    ``flatnonzero`` per column for no gain. Non-mask selectors (``None`` or an index
    array) pass through unchanged.
    """
    if rows is None or rows.dtype != bool:
        return rows
    selected = int(np.count_nonzero(rows))
    if getattr(store, "_is_multi_record", False):
        if selected >= n_rows * config.resolve_mask_density_gate():
            return rows
        return np.flatnonzero(rows)
    keeps_mask = getattr(store, "_keeps_boolean_mask", None)
    if keeps_mask is not None and keeps_mask(selected, n_rows):
        return rows
    return np.flatnonzero(rows)


def _wide_sum(batch: NDArray[Any]) -> Any:
    """Sum a batch in a wide accumulator so the total is independent of the batch size.

    Floating inputs accumulate in float64 (complex in complex128); integer inputs already
    widen to int64 in NumPy. Without this a float32 column would round differently at each
    batch boundary, making the reduction's value depend on the memory budget.
    """
    kind = batch.dtype.kind
    if kind == "f":
        return batch.sum(dtype=np.float64)
    if kind == "c":
        return batch.sum(dtype=np.complex128)
    return batch.sum()


# A reduction folds each materialized batch right after reading it, so a batch that spills
# last-level cache costs a second DRAM pass over the same data. When a column must be
# materialized to fold (a gather, a transform, or a non-native dtype) it is chunked to this
# size to keep the batch cache-resident; a stored column that can be viewed zero-copy skips
# chunking and folds in one pass. Kept small and fixed (not a user budget): a fold's
# per-batch overhead is negligible, so cache-residency dominates.
_REDUCTION_CHUNK_BYTES = 8 * 1024 * 1024


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

    def _read(self, rows: Rows) -> NDArray[Any]:
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
        try:
            self._dtype = store._native_dtype(name)
        except KeyError:
            raise KeyError(f"column {name!r} is not in the store; have {store.columns}.") from None
        self._store = store
        self._name = name
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

    def _read(self, rows: Rows) -> NDArray[Any]:
        if _row_count(rows) == 0:
            return np.empty(0, dtype=self._dtype)
        return self._store._gather_one(self._name, rows)

    def _fill_into(self, out: NDArray[Any], start: int, stop: int) -> None:
        """Write rows ``[start, stop)`` of the backing column straight into ``out``.

        The read counterpart to :meth:`_read` that fills the caller's array (e.g.
        a region of a memory-mapped output) instead of allocating a fresh one, so
        a passthrough column reaches its destination in a single copy from the
        source rather than via an intermediate array.
        """
        if stop > start:
            self._store._gather_slice_into(out, self._name, start, stop)

    def _disk_runs(self) -> list[tuple[Any, int, int]]:
        """Source ``(path, file_offset, n_bytes)`` runs for a raw merge copy.

        Defers to the backing store's :meth:`~colstore._base._ReaderBase.
        _column_disk_runs`, so a no-transform merge copies the column's bytes
        straight from the source file(s). Raises ``ValueError`` (propagated from
        the store) when the dtype cannot be raw-copied, which the caller treats
        as "not a pure merge".
        """
        return self._store._column_disk_runs(self._name)


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

    def _read(self, rows: Rows) -> NDArray[Any]:
        return self._array[rows]


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

    def _read(self, rows: Rows) -> NDArray[Any]:
        return np.full(_row_count(rows), self._value, dtype=self._dtype)


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


class Cast(Expr):
    """An internal node casting its input to a target dtype (NumPy ``astype``).

    Elementwise and row-independent like the ufunc nodes, so it evaluates one row
    range at a time. The structural key folds in the dtype, so two casts of the
    same input to the same dtype are computed once.
    """

    __slots__ = ("_dtype", "_input", "_key")

    def __init__(self, node: Expr, dtype: np.dtype[Any]) -> None:
        self._input = node
        self._dtype = dtype
        self._key = ("cast", dtype.str, node._key)


class Isin(Expr):
    """An internal node testing membership of its input in a fixed set (``np.isin``).

    Elementwise and row-independent like :class:`Cast`, so it evaluates one row
    selection at a time. The structural key folds in the test values, so two equal
    membership tests of the same input are computed once.
    """

    __slots__ = ("_key", "_target", "_values")

    def __init__(self, target: Expr, values: Any) -> None:
        self._target = target
        self._values = np.asarray(values)
        self._key = ("isin", target._key, self._values.dtype.str, self._values.tobytes())


class Where(Expr):
    """An internal node selecting elementwise between two inputs by a condition (``numpy.where``).

    Row-independent like the ufunc nodes -- each output row is ``a`` or ``b`` at that
    row, chosen by ``cond`` at the same row -- so it evaluates one row range at a time.
    Built when ``numpy.where`` is called on a column expression, or via
    :meth:`Expr.where`. Each of ``cond`` / ``a`` / ``b`` is an :class:`Expr` or a scalar.
    """

    __slots__ = ("_a", "_b", "_cond", "_key")

    def __init__(self, cond: Any, a: Any, b: Any) -> None:
        for value in (cond, a, b):
            _check_operand(value)
        if not any(isinstance(value, Expr) for value in (cond, a, b)):
            raise TypeError("numpy.where on a column expression needs at least one column operand.")
        self._cond = cond
        self._a = a
        self._b = b
        self._key = (
            "where",
            *(v._key if isinstance(v, Expr) else ("scalar", repr(v)) for v in (cond, a, b)),
        )


class Apply(Expr):
    """A node applying an opaque user function to its input columns, batch by batch.

    The function receives each input column as a NumPy array -- the whole batch,
    treated as the whole array -- and returns a 1-D array of the same length. Any
    NumPy is allowed, since the function is run rather than traced; it is the escape
    hatch for a column that cannot be written as a NumPy expression over ``cf[...]``.
    The output dtype is ``out_dtype`` when given, else inferred once at build time by
    running the function on empty inputs (no store data is read, though the function is
    executed). An inferred dtype must hold for every batch: a batch whose result dtype
    differs raises rather than being silently cast, so a function whose output dtype
    depends on the data must pass ``out_dtype`` -- which is then authoritative, each
    batch cast to it. ``out_dtype`` is also required when the function cannot run on a
    zero-length input. Every batch is checked to be a 1-D numeric array of the batch's
    length.

    Like every node, the function must be elementwise for a batched terminal
    (:meth:`~ColStoreFrame.write` / :meth:`~ColStoreFrame.iter_batches`) to match a
    single-pass result: batching only bounds memory, so each batch is the whole array
    to the function, and one that mixes rows sees each batch on its own.
    """

    __slots__ = ("_declared", "_dtype", "_func", "_inputs", "_key")

    def __init__(
        self, func: Callable[..., Any], inputs: tuple[Expr, ...], out_dtype: Any = None
    ) -> None:
        self._func = func
        self._inputs = inputs
        self._declared = out_dtype is not None
        self._dtype = _apply_output_dtype(func, inputs, out_dtype)
        self._key = ("apply", id(func), self._dtype.str, tuple(inp._key for inp in inputs))


def _apply_output_dtype(
    func: Callable[..., Any], inputs: tuple[Expr, ...], out_dtype: Any
) -> np.dtype[Any]:
    """Resolve an :class:`Apply` node's output dtype, reading no data.

    Returns ``out_dtype`` when given. Otherwise runs ``func`` once on empty arrays of
    the inputs' dtypes -- NumPy resolves the output dtype shape-independently, so this
    touches no store and is exact for NumPy-composed functions. Raises when the
    function cannot run on empty input (pass ``out_dtype``) or yields a non-1-D or
    unsupported (object / structured) result.
    """
    if out_dtype is not None:
        declared: np.dtype[Any] = np.dtype(out_dtype)
        return declared
    sample = [np.empty(0, result_dtype(inp)) for inp in inputs]
    try:
        probed: NDArray[Any] = np.asarray(func(*sample))
    except Exception as exc:
        raise TypeError(
            f"running apply()'s function on empty inputs to infer its dtype failed "
            f"({type(exc).__name__}: {exc}); if it cannot accept a zero-length input, "
            "pass out_dtype=, otherwise this is a bug in the function."
        ) from exc
    if probed.ndim != 1 or probed.dtype.kind in "OV":
        raise TypeError(
            f"apply()'s function gave an unsupported result on empty inputs (dtype "
            f"{probed.dtype}, ndim {probed.ndim}); return a 1-D numeric array, or pass out_dtype=."
        )
    return probed.dtype


def _np_where(condition: Any, *branches: Any) -> Expr:
    """``numpy.where(cond, x, y)`` builder for the column-expression dispatch table."""
    if len(branches) != 2:
        raise TypeError(
            "numpy.where on a column expression requires both branches: "
            "np.where(cond, x, y); the single-argument index form mixes rows and is "
            "not supported."
        )
    return Where(condition, branches[0], branches[1])


def _np_clip(a: Any, a_min: Any = None, a_max: Any = None, **kwargs: Any) -> Expr:
    """``numpy.clip`` builder; routes to :meth:`Expr.clip` on the clipped column.

    Accepts the bounds under either ``a_min`` / ``a_max`` or ``min`` / ``max`` (both
    spellings NumPy's ``clip`` allows); ``out=`` and any other keyword are rejected.
    """
    if not isinstance(a, Expr):
        raise TypeError("numpy.clip needs the clipped value to be a column expression.")
    if kwargs.get("out") is not None:
        raise TypeError("out= is not supported when building a column expression.")
    unsupported = set(kwargs) - {"min", "max", "out"}
    if unsupported:
        raise TypeError(f"unsupported argument(s) to numpy.clip: {sorted(unsupported)}.")
    low = a_min if a_min is not None else kwargs.get("min")
    high = a_max if a_max is not None else kwargs.get("max")
    return a.clip(low, high)


# Non-ufunc NumPy functions admitted in a column expression, each mapped to the
# node it builds. Anything outside this table raises in ``Expr.__array_function__``.
_ARRAY_FUNCTIONS: dict[Any, Callable[..., Expr]] = {np.where: _np_where, np.clip: _np_clip}


def evaluate(
    node: Expr,
    rows: Rows,
    memo: dict[tuple[Any, ...], NDArray[Any]],
) -> NDArray[Any]:
    """Materialize ``node`` over a row selection ``rows`` (a slice or index array).

    A contiguous ``slice`` reads a row range; an index array reads exactly those
    rows (e.g. a filtered selection). ``memo`` maps a node's structural key to its
    already-computed result for this selection; pass the *same* memo across every
    column of one batch so a shared subexpression is read or computed once, and a
    *fresh* memo per batch so each batch's working set is released. A zero-length
    selection runs the whole graph on empty typed arrays without touching any
    backing store, which is how :func:`result_dtype` recovers the dtype for free.
    """
    cached = memo.get(node._key)
    if cached is not None:
        return cached
    if isinstance(node, _Leaf):
        result = node._read(rows)
    elif isinstance(node, UFunc):
        args = [
            evaluate(child, rows, memo) if isinstance(child, Expr) else child
            for child in node._inputs
        ]
        result = node._ufunc(*args)
    elif isinstance(node, Cast):
        result = evaluate(node._input, rows, memo).astype(node._dtype)
    elif isinstance(node, Isin):
        result = np.isin(evaluate(node._target, rows, memo), node._values)
    elif isinstance(node, Where):
        cond = evaluate(node._cond, rows, memo) if isinstance(node._cond, Expr) else node._cond
        a = evaluate(node._a, rows, memo) if isinstance(node._a, Expr) else node._a
        b = evaluate(node._b, rows, memo) if isinstance(node._b, Expr) else node._b
        result = np.where(cond, a, b)
    elif isinstance(node, Apply):
        n = _row_count(rows)
        if n == 0:
            result = np.empty(0, node._dtype)  # pinned dtype; never re-run func on empty
        else:
            out = np.asarray(node._func(*(evaluate(inp, rows, memo) for inp in node._inputs)))
            if out.ndim != 1 or out.shape[0] != n:
                raise ValueError(
                    f"apply()'s function must return a 1-D array of length {n}; "
                    f"got shape {out.shape}."
                )
            if out.dtype.kind in "OV":
                raise TypeError(
                    f"apply()'s function returned an unsupported dtype {out.dtype}; "
                    "return a numeric array, or pass a concrete out_dtype=."
                )
            if out.dtype == node._dtype:
                result = out
            elif node._declared:
                result = out.astype(node._dtype)  # the declared out_dtype is authoritative
            else:
                raise ValueError(
                    f"apply()'s function returned dtype {out.dtype} on a batch but "
                    f"{node._dtype} was inferred from empty inputs -- its output dtype "
                    "depends on the data; pass out_dtype= to set the column dtype."
                )
    else:  # pragma: no cover - the Expr hierarchy is closed to its node types
        raise TypeError(f"cannot evaluate node of type {type(node).__name__!r}.")
    memo[node._key] = result
    return result


def result_dtype(node: Expr) -> np.dtype[Any]:
    """Output dtype of ``node`` via a zero-length evaluation; reads no data."""
    return evaluate(node, slice(0, 0), {}).dtype


def _iter_leaves(node: Expr) -> Iterator[_Leaf]:
    """Yield the leaf nodes reachable from ``node`` (with repetition)."""
    if isinstance(node, _Leaf):
        yield node
    elif isinstance(node, UFunc):
        for child in node._inputs:
            if isinstance(child, Expr):
                yield from _iter_leaves(child)
    elif isinstance(node, Cast):
        yield from _iter_leaves(node._input)
    elif isinstance(node, Isin):
        yield from _iter_leaves(node._target)
    elif isinstance(node, Where):
        for child in (node._cond, node._a, node._b):
            if isinstance(child, Expr):
                yield from _iter_leaves(child)
    elif isinstance(node, Apply):
        for inp in node._inputs:
            yield from _iter_leaves(inp)


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
    length-1 column against a wider frame is rejected. NumPy
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


class _QueryValueBuilder:
    """Rebuilds a ``col()`` predicate expression (:mod:`colstore._query`) as a frame
    value graph, so one ``col()`` serves both the filter and the assign contexts.

    ``column`` resolves a name against the frame's columns -- so a ``col()``
    reference picks up a column derived in the frame, not only a stored one --
    while ``op``, ``isin``, and ``where`` build the matching graph nodes.
    """

    __slots__ = ("_resolve",)

    def __init__(self, resolve: Callable[[str], Expr]) -> None:
        self._resolve = resolve

    def column(self, name: str) -> Expr:
        return self._resolve(name)

    def op(self, ufunc: np.ufunc, operands: list[Any]) -> Expr:
        return UFunc(ufunc, tuple(operands))

    def isin(self, target: Expr, values: Any) -> Expr:
        return Isin(target, values)

    def where(self, cond: Any, a: Any, b: Any) -> Expr:
        return Where(cond, a, b)


class _Cut(NamedTuple):
    """One ``where`` / ``filter`` predicate in a frame's pipeline, with label and weight."""

    label: str | None
    node: Expr
    weight: Expr | None


class CutInfo(NamedTuple):
    """One row of a :meth:`ColStoreFrame.report` cutflow.

    ``entering`` is the number of rows reaching the cut (survivors of every earlier
    cut) and ``passing`` how many satisfy it; ``efficiency`` is their ratio. When a
    weight is in effect, ``weighted_entering`` / ``weighted_passing`` are the summed
    weights over those same rows (else ``None``) and ``weighted_efficiency`` their ratio.
    """

    label: str
    entering: int
    passing: int
    weighted_entering: float | None = None
    weighted_passing: float | None = None

    @property
    def efficiency(self) -> float:
        """Fraction of entering rows that pass this cut (``0.0`` if none entered)."""
        return self.passing / self.entering if self.entering else 0.0

    @property
    def weighted_efficiency(self) -> float | None:
        """Weighted pass fraction, or ``None`` when no weight was in effect."""
        entering, passing = self.weighted_entering, self.weighted_passing
        if entering is None or passing is None:
            return None
        return passing / entering if entering else 0.0


class CutflowReport:
    """The cutflow of a frame's ``where`` / ``filter`` pipeline: per-cut survivor counts.

    Returned by :meth:`ColStoreFrame.report`. Iterate it for :class:`CutInfo` rows in
    cut order, index it by position or by label, and print it for a table of each cut's
    entering / passing counts and efficiency -- raw, weighted, or both, per ``show``.
    """

    __slots__ = ("_cuts", "_show")

    def __init__(self, cuts: list[CutInfo], show: str | None = None) -> None:
        self._cuts = cuts
        self._show = show

    def __len__(self) -> int:
        return len(self._cuts)

    def __iter__(self) -> Iterator[CutInfo]:
        return iter(self._cuts)

    def __getitem__(self, key: int | str) -> CutInfo:
        if isinstance(key, str):
            for cut in self._cuts:
                if cut.label == key:
                    return cut
            raise KeyError(f"no cut labeled {key!r}; have {[c.label for c in self._cuts]}.")
        return self._cuts[key]

    def records(self) -> list[dict[str, Any]]:
        """The cutflow as a list of per-cut dicts -- one dict per cut, for saving.

        Each carries ``label``, the raw ``entering`` / ``passing`` / ``efficiency``, and
        the ``weighted_*`` counterparts (``None`` when no weight was in effect), so the
        report round-trips to JSON, CSV, or any tabular sink. Every dict has the same
        keys, weighted or not.
        """
        return [
            {
                "label": c.label,
                "entering": c.entering,
                "passing": c.passing,
                "efficiency": c.efficiency,
                "weighted_entering": c.weighted_entering,
                "weighted_passing": c.weighted_passing,
                "weighted_efficiency": c.weighted_efficiency,
            }
            for c in self._cuts
        ]

    def _cells(self) -> tuple[list[str], list[list[str]]]:
        """The header row and one preformatted string row per cut, honoring ``show``."""
        weighted = any(c.weighted_entering is not None for c in self._cuts)
        show = self._show or ("both" if weighted else "raw")
        raw = show in ("raw", "both")
        wt = show in ("weighted", "both") and weighted
        if not raw and not wt:  # weighted asked for but none in effect
            raw = True
        headers = ["cut"]
        if raw:
            headers += ["entering", "passing", "eff"]
        if wt:
            headers += ["wt_entering", "wt_passing", "wt_eff"]
        rows: list[list[str]] = []
        for c in self._cuts:
            row = [c.label]
            if raw:
                row += [str(c.entering), str(c.passing), f"{c.efficiency:.2%}"]
            if wt:
                eff = c.weighted_efficiency
                row += [
                    "-" if c.weighted_entering is None else f"{c.weighted_entering:.6g}",
                    "-" if c.weighted_passing is None else f"{c.weighted_passing:.6g}",
                    "-" if eff is None else f"{eff:.2%}",
                ]
            rows.append(row)
        return headers, rows

    def _caption(self) -> str:
        n = len(self._cuts)
        weighted = any(c.weighted_entering is not None for c in self._cuts)
        return f"{'weighted ' if weighted else ''}cutflow ({n} cut{'' if n == 1 else 's'})"

    def __repr__(self) -> str:
        if not self._cuts:
            return "CutflowReport(no cuts)"
        return render_table_text(*self._cells())

    def _repr_html_(self) -> str:
        if not self._cuts:
            return (
                '<div style="font-family:ui-monospace,monospace;font-size:90%;'
                'color:#57606a">CutflowReport: no cuts</div>'
            )
        headers, rows = self._cells()
        return render_table_html(self._caption(), headers, rows)


class ColStoreFrame:
    """A deferred editing view over an opened store's columns.

    Created by :meth:`edit` on an opened reader or dataset. A frame holds an
    ordered mapping of output column name to an expression (:class:`Expr`);
    opening one seeds it with a native-passthrough leaf per source column.
    Indexing by name returns the expression for a column, which composes with
    operators and elementwise NumPy ufuncs to build transformations; a frame does not
    slice or index rows or columns (that is the reader's role). The edit methods
    (``assign`` / ``with_columns`` / ``drop`` / ``rename`` / ``astype`` /
    ``select``) return a new frame by default -- pass ``inplace=True`` to edit
    this one -- so edits
    branch cheaply off a shared base; ``frame[name] = ...`` and ``del`` always
    edit in place. Nothing is read until you materialize:
    :meth:`array` / :meth:`dict` / :meth:`recarray` evaluate columns into memory,
    and :meth:`write` streams the result to a new file and returns a reader for it.
    The source store is never modified.

    A frame may also carry a row selection -- a concrete index set (e.g. from
    ``ds[idx].edit()``) and/or pending :meth:`where` predicates -- applied to
    every column alike when the frame is materialized.

    Assignment adds a column as an expression over the frame's columns, which
    co-filters with any selection. An external array may be attached only on an
    unfiltered frame, where its length is checked against the source row count;
    it is held by reference unless ``copy=True`` is passed to :meth:`assign`.
    """

    __slots__ = ("_columns", "_n_rows", "_predicates", "_rows", "_store")

    # The source store, or ``None`` for a materialized in-memory frame (one whose
    # columns are concrete arrays, e.g. a batch from :meth:`iter_batches`); such a
    # frame never reads a store, so the field is only ever along for the ride.
    _store: _ReaderBase | None

    def __init__(
        self,
        store: _ReaderBase,
        columns: list[str] | None = None,
        rows: NDArray[Any] | None = None,
        predicate: _Expr | None = None,
    ) -> None:
        self._store = store
        self._n_rows = store.n_rows
        self._rows = _normalize_base_rows(rows, self._n_rows, store)
        self._predicates: tuple[_Cut, ...] = ()
        names = store.columns if columns is None else columns
        self._columns: dict[str, Expr] = {name: NativeColumn(store, name) for name in names}
        if predicate is not None:
            self._carry_reader_predicate(predicate)

    @property
    def n_rows(self) -> int:
        """Number of rows the frame selects: the source count, or the size of a
        concrete selection (e.g. from ``ds[idx].edit()``).

        A pending :meth:`where` predicate is evaluated on access -- an O(n) scan --
        so the read is cheap only when the selection is already concrete or absent.
        """
        return self._row_count()

    def _row_count(self) -> int:
        # n_rows must never throw, so count via the predicate mask's popcount -- it never
        # materializes the (potentially huge) row index a terminal would.
        if not self._predicates:
            return self._n_rows if self._rows is None else _row_count(self._rows)
        _, mask = self._composed_mask()
        return int(np.count_nonzero(mask))

    def _apply_cuts(
        self, base_rows: Rows, *, count: bool = False
    ) -> tuple[NDArray[Any] | None, list[CutInfo]]:
        """AND every ``where`` / ``filter`` predicate mask over ``base_rows``.

        Returns the combined boolean mask (``None`` only when there are no
        predicates) and, when ``count`` is set, a :class:`CutInfo` per cut tracking how
        many rows -- and, where a weight is in effect, how much summed weight -- enter and
        pass it. One memo is shared so a subexpression used by several cuts (or a cut and
        its weight) runs once.
        """
        memo: dict[tuple[Any, ...], NDArray[Any]] = {}
        mask: NDArray[Any] | None = None
        cuts: list[CutInfo] = []
        entering = (
            base_rows.stop - base_rows.start if isinstance(base_rows, slice) else len(base_rows)
        )
        for i, cut in enumerate(self._predicates):
            evaluated = np.asarray(evaluate(cut.node, base_rows, memo))
            entering_mask = mask  # cumulative survivors before this cut (None = all rows)
            mask = evaluated if mask is None else (mask & evaluated)
            if count:
                passing = int(np.count_nonzero(mask))
                w_entering: float | None = None
                w_passing: float | None = None
                if cut.weight is not None:
                    weights = np.asarray(evaluate(cut.weight, base_rows, memo))
                    w_entering = float(
                        weights.sum() if entering_mask is None else weights[entering_mask].sum()
                    )
                    w_passing = float(weights[mask].sum())
                cuts.append(CutInfo(cut.label or f"#{i}", entering, passing, w_entering, w_passing))
                entering = passing
        return mask, cuts

    def _base_indices(self) -> NDArray[Any] | None:
        """:attr:`_rows` with a kept boolean mask lowered to its int64 positions.

        The mask form survives only for an *un-composed* selection (so a terminal can
        gather it mask-natively); a pending predicate composes onto a positional index
        set -- ``flatnonzero(_rows)[passing]`` -- so the mask lowers to indices here
        before the cut is applied.
        """
        base = self._rows
        if base is not None and base.dtype == bool:
            return np.flatnonzero(base)
        return base

    def _composed_mask(self) -> tuple[NDArray[Any] | None, NDArray[Any]]:
        """The base selection (a kept mask lowered to indices) and the AND-of-predicates
        mask over it. Callers either count it (``count_nonzero``, for :attr:`n_rows`) or
        materialize the surviving indices (``flatnonzero``, for a terminal)."""
        base = self._base_indices()
        base_rows: Rows = slice(0, self._n_rows) if base is None else base
        mask, _ = self._apply_cuts(base_rows)
        assert mask is not None  # _predicates is non-empty here
        return base, mask

    def _resolve_selection(self) -> NDArray[Any] | None:
        """The concrete row selection: the base rows narrowed by every pending
        predicate (``None`` = all rows). An un-composed boolean mask is returned as-is
        so a terminal can gather it mask-natively; a predicate composes onto the lowered
        indices. Evaluating the predicates is the deferred work :meth:`where` records."""
        if not self._predicates:
            return self._rows
        base, mask = self._composed_mask()
        return np.flatnonzero(mask) if base is None else base[mask]

    def report(self, show: str | None = None) -> CutflowReport:
        """Cutflow of the ``where`` / ``filter`` pipeline: per-cut survivor counts.

        An action that evaluates each cut in order and returns a :class:`CutflowReport`
        of how many rows enter and pass it (named by :meth:`where`'s ``label``, else by
        position), plus the summed weight over those rows when a cut carries a ``where``
        ``weight``. ``show`` picks what the report prints -- ``"raw"``, ``"weighted"``, or
        ``"both"`` (the default shows both when any weight is in effect, else raw). A
        frame with no cuts gives an empty report; a concrete base selection
        (``ds[idx].edit()``) is the first cut's entering count.
        """
        if show not in (None, "raw", "weighted", "both"):
            raise ValueError(f"show must be 'raw', 'weighted', 'both', or None; got {show!r}.")
        if not self._predicates:
            return CutflowReport([], show)
        base = self._base_indices()
        base_rows: Rows = slice(0, self._n_rows) if base is None else base
        _, cuts = self._apply_cuts(base_rows, count=True)
        return CutflowReport(cuts, show)

    def _resolve_rows(self) -> Rows:
        """The selector the in-memory terminals (:meth:`dict` / :meth:`array` /
        :meth:`recarray`) evaluate over: the full source range, or the resolved
        selection. A kept boolean mask is passed straight through, so a one-shot gather
        over the whole selection lets the reader's density gate route a dense mask to the
        mask-native kernel rather than paying ``flatnonzero`` and the fancy path."""
        selection = self._resolve_selection()
        return slice(0, self._n_rows) if selection is None else selection

    def _resolve_index_selection(self) -> NDArray[Any] | None:
        """The selection as ``None`` or an int64 index array, for the batched paths that
        slice and count it (:meth:`write` / :meth:`iter_batches` / the reductions). A kept
        boolean mask is lowered to its positions: a per-batch ``selection[start:stop]``
        over a mask would slice mask positions, not selected source rows."""
        selection = self._resolve_selection()
        if selection is not None and selection.dtype == bool:
            return np.flatnonzero(selection)
        return selection

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

    def __getitem__(self, key: Any) -> Expr:
        """The named column's expression -- for building transforms, not data access.

        ``frame["a"]`` returns column ``a``'s lazy expression, which composes with
        operators and ufuncs (``frame["a"] * 2``, ``frame.assign(x=frame["a"] +
        frame["b"])``). A frame does **not** slice or index rows or columns: filter with
        :meth:`where`, project columns with :meth:`select`, and for positional or fancy
        indexing :meth:`write` the frame to a ``.cstore`` and index the returned reader,
        where the data is contiguous. (The reader / dataset is for viewing; the frame is
        for filtering and editing.)
        """
        if not isinstance(key, str):
            raise TypeError(
                f"a frame indexes only a column by name (returning its expression); got "
                f"{type(key).__name__}. It does not slice or index rows or columns -- filter "
                f"with where(), pick columns with select(), and for positional or fancy indexing "
                f"write() the frame and index the returned reader."
            )
        try:
            return self._columns[key]
        except KeyError:
            raise KeyError(
                f"column {key!r} is not in the frame; have {list(self._columns)}."
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

    def _resolve_column(self, name: str) -> Expr:
        try:
            return self._columns[name]
        except KeyError:
            raise KeyError(
                f"col({name!r}) is not a column of this frame; have {list(self._columns)}."
            ) from None

    def _coerce(self, value: Any, *, copy: bool) -> Expr:
        if isinstance(value, _Expr):
            value = value._emit(_QueryValueBuilder(self._resolve_column))
        expr = as_expr(value, copy=copy)
        if isinstance(expr, MemoryColumn) and (self._rows is not None or self._predicates):
            raise ValueError(
                "cannot attach a raw array to a frame that already has a row selection "
                "(its length would be ambiguous against the selection); attach external "
                "arrays at the base -- before any where() or index selection -- or pass a "
                "col() expression, which derives from the frame and co-filters."
            )
        validate_length(expr, self._n_rows)
        return expr

    def _resolve_column_arg(self, value: Any, error: str) -> Expr:
        """Resolve a column name, ``col()`` expression, or frame ``Expr`` to a frame ``Expr``.

        Raises ``TypeError(error)`` for any other type.
        """
        if isinstance(value, str):
            return self._resolve_column(value)
        if isinstance(value, _Expr):
            emitted: Expr = value._emit(_QueryValueBuilder(self._resolve_column))
            return emitted
        if isinstance(value, Expr):
            return value
        raise TypeError(error)

    def _resolve_value_column(self, value: Any) -> Expr:
        """Resolve an :meth:`apply` input -- a column name, a :func:`~colstore.col`
        expression, or a frame expression -- to a frame :class:`Expr`."""
        return self._resolve_column_arg(
            value,
            f"apply() columns must be column names or expressions; got "
            f"{type(value).__name__}. Bake any constants into the function instead.",
        )

    def apply(self, func: Callable[..., Any], *cols: Any, out_dtype: Any = None) -> Expr:
        """Derive a column by running ``func`` on the named columns' arrays.

        ``func`` receives each of ``cols`` -- column names, :func:`~colstore.col`
        expressions, or frame expressions -- as a NumPy array (the whole batch) and
        returns a 1-D array of the same length: the escape hatch for a column that
        cannot be written as a NumPy expression over ``cf[...]``. Any NumPy is allowed
        (the function is run, not traced). The result dtype is inferred by running
        ``func`` once on empty inputs (reads no store data, but does execute ``func``,
        so an effectful one should be pure) -- pass ``out_dtype`` when ``func`` cannot
        run on a zero-length input, when its output dtype depends on the data, or to
        skip that build-time call. The returned expression composes and is attached with
        ``cf[name] = ...`` or :meth:`assign`.

        Because batching only bounds memory, each batch is the whole array to ``func``;
        a function that mixes rows (a running total, a sort) sees each batch on its own,
        while elementwise (per-row) functions -- the common case -- are unaffected.
        """
        if not cols:
            raise TypeError("apply() needs at least one column.")
        inputs = tuple(self._resolve_value_column(c) for c in cols)
        return Apply(func, inputs, out_dtype)

    def copy(self) -> ColStoreFrame:
        """An independent copy of the frame -- a cheap branch point.

        The expression nodes are immutable and shared; only the column mapping is
        duplicated, so editing the copy never affects this frame. This is the
        default result of every edit method below (see ``inplace``).
        """
        clone = ColStoreFrame.__new__(ColStoreFrame)
        clone._store = self._store
        clone._n_rows = self._n_rows
        clone._rows = self._rows
        clone._predicates = self._predicates
        clone._columns = dict(self._columns)
        return clone

    @classmethod
    def _materialized(cls, columns: dict[str, NDArray[Any]], n_rows: int) -> ColStoreFrame:
        """A store-detached frame whose columns are concrete in-memory arrays.

        Each column becomes a :class:`MemoryColumn` over the given array, so the
        result is a full frame -- same terminals (``dict`` / ``recarray`` /
        ``write``) and edits -- but reads from memory rather than a source store.
        :meth:`iter_batches` builds one per batch so the caller gets a frame to
        convert to any format, not a fixed array type.
        """
        frame = cls.__new__(cls)
        frame._store = None
        frame._n_rows = n_rows
        frame._rows = None
        frame._predicates = ()
        frame._columns = {name: MemoryColumn(array) for name, array in columns.items()}
        return frame

    def assign(
        self, *, copy: bool = False, inplace: bool = False, **new_columns: Any
    ) -> ColStoreFrame:
        """Add or replace columns from keyword arguments.

        Returns a new frame, leaving this one unchanged, unless ``inplace=True``.
        Each value may be an expression, array, or scalar; arrays are held by
        reference unless ``copy=True``. (Columns literally named ``copy`` or
        ``inplace`` must be set with ``frame[name] = ...`` instead.)
        """
        coerced = {name: self._coerce(value, copy=copy) for name, value in new_columns.items()}
        frame = self if inplace else self.copy()
        frame._columns.update(coerced)
        return frame

    def with_columns(
        self, *, copy: bool = False, inplace: bool = False, **new_columns: Any
    ) -> ColStoreFrame:
        """Add or replace columns from keyword arguments; an alias for :meth:`assign`."""
        return self.assign(copy=copy, inplace=inplace, **new_columns)

    def drop(self, *names: str, inplace: bool = False) -> ColStoreFrame:
        """Remove one or more columns.

        Returns a new frame unless ``inplace=True``. Raises ``KeyError`` for a
        name that is not in the frame, before changing anything.
        """
        missing = [name for name in names if name not in self._columns]
        if missing:
            raise KeyError(f"cannot drop columns that are not in the frame: {missing}.")
        frame = self if inplace else self.copy()
        for name in names:
            del frame._columns[name]
        return frame

    def rename(self, columns: dict[str, str], inplace: bool = False) -> ColStoreFrame:
        """Rename columns, resolving all mappings simultaneously.

        Returns a new frame unless ``inplace=True``. A swap such as
        ``{"a": "b", "b": "a"}`` exchanges the two names in one step. Renames
        change output names only -- the underlying data, and any expression
        already referencing a column, are unaffected. Raises if a source name is
        missing or the result would contain duplicate names.
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
        frame = self if inplace else self.copy()
        frame._columns = renamed
        return frame

    def astype(self, dtypes: dict[str, Any], inplace: bool = False) -> ColStoreFrame:
        """Cast the named columns to new dtypes (deferred).

        Returns a new frame unless ``inplace=True``. Each value is anything
        ``numpy.dtype`` accepts; the cast is lazy -- evaluated by ``array`` /
        ``dict`` / ``recarray`` / ``write``. Raises ``KeyError`` for an unknown
        name and validates every dtype before changing anything.
        """
        missing = [name for name in dtypes if name not in self._columns]
        if missing:
            raise KeyError(f"cannot cast columns that are not in the frame: {missing}.")
        resolved = {name: np.dtype(dtype) for name, dtype in dtypes.items()}
        frame = self if inplace else self.copy()
        for name, dtype in resolved.items():
            frame._columns[name] = Cast(frame._columns[name], dtype)
        return frame

    def select(self, *names: str, inplace: bool = False) -> ColStoreFrame:
        """Project the frame to ``names`` in the given order, dropping the rest.

        Returns a new frame by default; pass ``inplace=True`` to edit this one.
        An unknown name raises ``KeyError`` and a repeated name ``ValueError``.
        Only the column mapping is narrowed -- no data is read, and the row
        selection is preserved.
        """
        missing = [name for name in names if name not in self._columns]
        if missing:
            raise KeyError(f"cannot select columns that are not in the frame: {missing}.")
        repeated = sorted(name for name, count in Counter(names).items() if count > 1)
        if repeated:
            raise ValueError(f"duplicate column(s) in select(): {repeated}.")
        frame = self if inplace else self.copy()
        frame._columns = {name: frame._columns[name] for name in names}
        return frame

    def _compile_predicate(
        self,
        predicate: str | _Expr,
        params: dict[str, Any] | None,
        *,
        resolve: Callable[[str], Expr] | None = None,
        universe: frozenset[str] | None = None,
    ) -> Expr:
        """Validate a ``col()`` / query-string predicate against a column universe
        and compile it to a boolean frame-expression node -- reading no data.

        By default names resolve in order against the frame's own columns -- a
        column derived earlier is a valid target, a dropped or renamed-away one is
        not. The :meth:`~colstore.view.TableView.edit` seam passes a store-backed
        ``resolve`` and the store's columns as ``universe`` instead, so a carried
        reader predicate keeps the reader's resolution scope. Raises
        :class:`~colstore.QueryError` for an unknown column or a non-boolean condition.
        """
        resolve = self._resolve_column if resolve is None else resolve
        universe = frozenset(self._columns) if universe is None else universe
        query = (
            predicate if isinstance(predicate, _Expr) else parse_query(predicate, universe, params)
        )
        missing = sorted(name for name in set(query._columns()) if name not in universe)
        if missing:
            raise QueryError(
                f"the where() predicate references unknown column(s) {missing}; "
                f"have {sorted(universe)}."
            )
        node: Expr = query._emit(_QueryValueBuilder(resolve))
        dtype = result_dtype(node)
        if dtype.kind != "b":
            raise QueryError(f"a where() predicate must be a boolean condition; got dtype {dtype}.")
        return node

    def _carry_reader_predicate(self, predicate: _Expr) -> None:
        """Record a reader predicate from :meth:`~colstore.view.TableView.edit` as a
        pending :meth:`where`.

        The predicate's names resolve against the source store, not only the frame's
        (possibly projected) columns, so a predicate referencing a column the view
        dropped still filters -- matching ``ds[pred, cols]`` reader semantics and
        ``ds.edit().where(pred).select(cols)``. Reads no data; the cut resolves with
        the rest of the graph when the frame is materialized.
        """
        store = self._store
        assert store is not None  # the view.edit() seam always carries a real store
        node = self._compile_predicate(
            predicate,
            None,
            resolve=lambda name: NativeColumn(store, name),
            universe=frozenset(store.columns),
        )
        self._predicates = (*self._predicates, _Cut(None, node, None))

    def _resolve_weight(self, weight: str | _Expr | Expr | None) -> Expr | None:
        """Resolve a ``where`` weight to a numeric column expression (or ``None``).

        A column name resolves against the frame's columns; a ``col()`` expression is
        rebuilt onto the frame's graph. The weight is summed in the :meth:`report`
        cutflow, never used for selection, so it must be numeric.
        """
        if weight is None:
            return None
        expr = self._resolve_column_arg(
            weight,
            f"weight must be a column name, a col() expression, or None; "
            f"got {type(weight).__name__}.",
        )
        if result_dtype(expr).kind not in "iuf":
            raise TypeError("a where() weight must be a numeric column.")
        return expr

    def where(
        self,
        predicate: str | _Expr,
        label: str | None = None,
        *,
        weight: str | _Expr | Expr | None = None,
        params: dict[str, Any] | None = None,
        inplace: bool = False,
    ) -> ColStoreFrame:
        """Keep the rows where ``predicate`` is true; returns a new frame by default.

        **Lazy**: the predicate -- a :func:`~colstore.col` expression or a query string
        (the grammar of :meth:`~colstore.ColStoreReader.query`) -- is validated up front
        (an unknown column or a non-boolean condition raises :class:`~colstore.QueryError`,
        reading no data) but only evaluated when the frame is materialized. Names
        resolve against the frame's columns in order, so a column derived earlier is a
        valid target and a dropped or renamed-away one is not. Successive ``where``
        calls compose (AND); reading :attr:`n_rows`, like any terminal, resolves them.

        Pass ``label`` to name this cut in the :meth:`report` cutflow, and ``weight`` (a
        column name or expression) to sum a per-row weight entering and passing the cut in
        the weighted cutflow. A weight is **sticky** -- it carries to later cuts until
        another is given. ``filter`` is an alias. Pass ``inplace=True`` to edit this frame.
        """
        node = self._compile_predicate(predicate, params)
        cut_weight = self._resolve_weight(weight)
        if cut_weight is None and self._predicates:
            cut_weight = self._predicates[-1].weight  # sticky: carry the most recent weight
        frame = self if inplace else self.copy()
        frame._predicates = (*self._predicates, _Cut(label, node, cut_weight))
        return frame

    def filter(
        self,
        predicate: str | _Expr,
        label: str | None = None,
        *,
        weight: str | _Expr | Expr | None = None,
        params: dict[str, Any] | None = None,
        inplace: bool = False,
    ) -> ColStoreFrame:
        """Alias of :meth:`where` -- keep the rows where ``predicate`` is true."""
        return self.where(predicate, label, weight=weight, params=params, inplace=inplace)

    def _materialize(self, rows: Rows) -> dict[str, NDArray[Any]]:
        """Evaluate every column over ``rows`` into a ``name -> array`` mapping, sharing
        one memo so a subexpression used by several columns is computed once."""
        memo: dict[tuple[Any, ...], NDArray[Any]] = {}
        return {name: evaluate(expr, rows, memo) for name, expr in self._columns.items()}

    def array(self, name: str) -> NDArray[Any]:
        """Materialize one column as a 1-D array over the selected rows; writes no file.

        The single-column counterpart of :meth:`dict` / :meth:`recarray` -- resolves any
        pending :meth:`where` predicate, then evaluates column ``name`` over the selected
        rows. (``frame[name]`` returns the column's expression for building transforms;
        evaluating that directly with ``Expr.compute`` ignores the frame's row selection.)
        """
        return evaluate(self[name], self._resolve_rows(), {})

    def dict(self) -> dict[str, NDArray[Any]]:
        """Compute every column into memory as a ``name -> array`` mapping; writes no file.

        Resolves any pending :meth:`where` predicate, then evaluates each column over
        the selected rows (one shared memo, so a subexpression used by several columns
        runs once). The in-memory analogue of :meth:`write`.
        """
        return self._materialize(self._resolve_rows())

    def recarray(self) -> NDArray[Any]:
        """Compute every column into one structured (record) ndarray; writes no file.

        Columns are interleaved into the record layout by the same parallel
        ``interleave_records`` kernel the reader's :meth:`recarray` uses.
        """
        columns = self.dict()
        if not columns:
            return np.empty(self._row_count(), dtype=np.dtype([]))
        names = list(columns)
        sources = [np.ascontiguousarray(columns[name]) for name in names]
        dtype = np.dtype([(name, src.dtype) for name, src in zip(names, sources, strict=True)])
        return kernels.interleave_record_array(names, sources, dtype)

    def frame(self) -> pd.DataFrame:
        """Compute every column into a pandas DataFrame; writes no file.

        The pandas analogue of :meth:`dict` / :meth:`recarray`: resolves any pending
        :meth:`where` predicate, evaluates each column over the selected rows, and assembles
        them into a DataFrame in column order with the computed dtypes. Requires pandas.
        """
        return _make_dataframe_no_consolidate(self.dict())

    # ---- Reduction terminals (full pass, scalar result) ----------------

    def _stream_column(self, expr: Expr) -> Iterator[NDArray[Any]]:
        """Yield the selected rows of one column expression for a reduction to fold.

        A stored column with no fancy selection over a single-record native store yields one
        read-only zero-copy view, so the fold is a single pass over the source with no copy.
        Otherwise -- a gather (interleaved multi-record or a filtered index), a transform, or
        a non-native dtype -- the column is materialized in cache-resident chunks so the fold
        never re-reads a cache-spilling batch from DRAM; peak memory is one chunk of a single
        column. A fresh memo per chunk releases each batch's working set.
        """
        selection = self._resolve_index_selection()
        if isinstance(expr, NativeColumn) and not isinstance(selection, np.ndarray):
            view_rows: Rows = slice(0, self._n_rows) if selection is None else selection
            try:
                view = expr._store._view_one(expr.name, view_rows)
            except ValueError:
                pass  # a copy is unavoidable (interleaved / non-native) -> chunk below
            else:
                if len(view):
                    yield view
                return
        n = self._n_rows if selection is None else len(selection)
        if n == 0:
            return
        rows_per_batch = max(1, _REDUCTION_CHUNK_BYTES // max(1, result_dtype(expr).itemsize))
        for start in range(0, n, rows_per_batch):
            stop = min(start + rows_per_batch, n)
            rows: Rows = slice(start, stop) if selection is None else selection[start:stop]
            yield evaluate(expr, rows, {})

    def count(self) -> int:
        """Number of selected rows (resolves any pending :meth:`where`)."""
        return self._row_count()

    def _reduction_expr(self, column: Any, op: str, kinds: str, requirement: str) -> Expr:
        """Resolve a reduction's ``column`` and require a supported dtype, else a clear error.

        ``kinds`` is the set of acceptable ``dtype.kind`` characters -- numeric for
        ``sum``/``mean`` (NumPy has no string ``add``), numeric-or-datetime for
        ``min``/``max`` (NumPy's ``minimum``/``maximum`` have no string loop either).
        """
        expr = self._resolve_value_column(column)
        dtype = result_dtype(expr)
        if dtype.kind not in kinds:
            raise TypeError(f"{op}() needs a {requirement} column; got dtype {dtype}.")
        return expr

    def sum(self, column: Any) -> Any:
        """Sum of ``column`` over the selected rows -- a full pass, returning a scalar.

        ``column`` is a name, a :func:`~colstore.col` expression, or a frame expression.
        The column is streamed in bounded-memory batches and the per-batch sums added.
        Integer sums widen to int64; floating sums accumulate in float64 for a stable,
        batch-size-independent result (so a float32 column sums to float64). A NaN in the
        data propagates to the result (NumPy semantics, not NaN-skipping); an empty
        selection sums to zero.
        """
        expr = self._reduction_expr(column, "sum", "biufc", "numeric")
        total: Any = None
        for batch in self._stream_column(expr):
            partial = _wide_sum(batch)
            total = partial if total is None else total + partial
        return total if total is not None else _wide_sum(np.empty(0, result_dtype(expr)))

    def mean(self, column: Any) -> Any:
        """Mean of ``column`` over the selected rows -- a full pass, returning a float.

        Streams the per-batch sum (a float64 accumulator) and the row count, dividing once
        at the end. A NaN in the data propagates; an empty selection gives ``nan``.
        """
        expr = self._reduction_expr(column, "mean", "biufc", "numeric")
        total: Any = None
        n = 0
        for batch in self._stream_column(expr):
            partial = _wide_sum(batch)
            total = partial if total is None else total + partial
            n += len(batch)
        return total / n if n else float("nan")

    def min(self, column: Any) -> Any:
        """Minimum of ``column`` over the selected rows -- a full pass.

        A NaN in the data propagates to the result (NumPy semantics); an empty selection
        returns a float ``nan`` regardless of the column's dtype.
        """
        expr = self._reduction_expr(column, "min", "biufMm", "numeric or datetime")
        acc: Any = None
        for batch in self._stream_column(expr):
            batch_min = batch.min()
            acc = batch_min if acc is None else np.minimum(acc, batch_min)
        return acc if acc is not None else float("nan")

    def max(self, column: Any) -> Any:
        """Maximum of ``column`` over the selected rows -- a full pass.

        A NaN in the data propagates to the result (NumPy semantics); an empty selection
        returns a float ``nan`` regardless of the column's dtype.
        """
        expr = self._reduction_expr(column, "max", "biufMm", "numeric or datetime")
        acc: Any = None
        for batch in self._stream_column(expr):
            batch_max = batch.max()
            acc = batch_max if acc is None else np.maximum(acc, batch_max)
        return acc if acc is not None else float("nan")

    def _batch_column(
        self, name: str, expr: Expr, rows: Rows, memo: Memo, buffers: Buffers, rows_per_batch: int
    ) -> NDArray[Any]:
        """One ``copy=False`` batch column.

        A read-only zero-copy view when the source can give one (a stored column over a
        contiguous range of a single-record native store); else, for a bare stored column, a
        gather into ``buffers[name]`` -- a per-column buffer reused across batches so the
        streaming gather allocates nothing per batch; else (a transform) a freshly computed
        array shared through ``memo``. The view and reused-buffer results are valid only until
        the next batch and must not be held.
        """
        if isinstance(expr, NativeColumn):
            if not isinstance(rows, np.ndarray):
                try:
                    return expr._store._view_one(expr.name, rows)
                except ValueError:
                    pass  # interleaved, non-native, or fancy -> gather into a reused buffer
            if name not in buffers:
                # First gather for this column: probe whether the store fills out= in place. A
                # store that ignores the hint (a multi-file boundary read) returns a fresh array,
                # so remember that and stop allocating a dead buffer for it.
                probe = np.empty(rows_per_batch, result_dtype(expr))
                got = expr._store._gather_one(expr.name, rows, out=probe[: _row_count(rows)])
                buffers[name] = probe if np.shares_memory(got, probe) else None
                return got
            buf = buffers[name]
            if buf is None:
                return expr._store._gather_one(expr.name, rows)  # store ignores out=; gather fresh
            return expr._store._gather_one(expr.name, rows, out=buf[: _row_count(rows)])
        return evaluate(expr, rows, memo)

    def iter_batches(
        self, batch_size: int | str | None = None, *, copy: bool = True
    ) -> Iterator[ColStoreFrame]:
        """Yield the selected rows as materialized frames, bounded in memory.

        The streaming counterpart of :meth:`recarray`: the selection is resolved
        once, then each batch of the selected rows is evaluated with a fresh memo
        into a **materialized, in-memory** :class:`ColStoreFrame` -- its columns
        are concrete arrays detached from the source store -- so peak memory is one
        batch rather than the whole frame, and the caller picks the format
        (:meth:`dict` / :meth:`recarray` / :meth:`write`, or further edits) instead
        of a fixed array type. ``batch_size`` sizes each batch -- ``int`` rows, ``str`` an IEC
        memory budget per batch (e.g. ``"256 MiB"``) converted from the per-row
        byte size, ``None`` the configured default budget. A frame with no columns
        or no selected rows yields nothing.

        ``copy=False`` is a fast path for read-only streaming consumers that finish with each
        batch before drawing the next (accumulating a reduction, writing each batch out, feeding
        a model): a batch column is a **read-only zero-copy view** of the source where available
        (a stored column over a contiguous range of a single-record native store), or a gather
        into a **per-column buffer reused across batches** (an interleaved multi-record column,
        or a filtered/fancy selection) so the gather allocates nothing per batch; a transform
        is freshly computed. Because the views and buffers are reused, a batch is valid only
        until the next is drawn -- do not mutate or hold it. Keep the default ``copy=True`` for
        owning arrays safe to mutate or hold after the store closes.
        """
        names = list(self._columns)
        selection = self._resolve_index_selection()
        n = self._n_rows if selection is None else len(selection)
        if not names or n == 0:
            return
        bytes_per_row = sum(result_dtype(self._columns[name]).itemsize for name in names)
        rows_per_batch = resolve_batch_rows(batch_size, bytes_per_row=bytes_per_row)
        if rows_per_batch is None:
            rows_per_batch = max(1, config.get_default_memory_budget() // bytes_per_row)
        buffers: Buffers = {}  # copy=False: per-column gather buffers, reused across batches
        for start in range(0, n, rows_per_batch):
            stop = min(start + rows_per_batch, n)
            rows: Rows = slice(start, stop) if selection is None else selection[start:stop]
            memo: Memo = {}
            if copy:
                columns = {name: evaluate(self._columns[name], rows, memo) for name in names}
            else:
                columns = {
                    name: self._batch_column(
                        name, self._columns[name], rows, memo, buffers, rows_per_batch
                    )
                    for name in names
                }
            yield ColStoreFrame._materialized(columns, stop - start)

    def write(
        self, path: str | os.PathLike[str], *, memory_budget: int | None = None
    ) -> ColStoreReader:
        """Stream the edited columns to a new ``.cstore`` and return a reader.

        Evaluates every column one batch at a time into the new file (see
        :func:`colstore.format.write_dataset_streaming`); peak memory is bounded by
        ``memory_budget`` (bytes; ``None`` uses the configured default). A frame
        carrying a selection (an index set, or a resolved :meth:`where`) streams its
        selected rows the same way -- each batch gathers the selected source rows
        rather than a contiguous range. The source store is not modified; writing a
        frame with no columns is an error.
        """
        from .api import open as open_store
        from .format import write_dataset_streaming

        selection = self._resolve_index_selection()
        if selection is None:
            write_dataset_streaming(self._columns, self._n_rows, path, memory_budget=memory_budget)
        else:
            write_dataset_streaming(
                self._columns, len(selection), path, memory_budget=memory_budget, rows=selection
            )
        return open_store(path)
