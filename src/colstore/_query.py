"""Lazy column expressions and the predicate-string parser for ``query()``.

``col("pt") > 30`` and the like build a lazy :class:`_Expr` tree that reads
nothing until it is applied to a store; the string form ``query("pt > 30")``
parses to the *same* tree. Evaluating an ``_Expr`` against a store yields a NumPy
array -- a boolean one is a row mask. The string grammar is a strict whitelist
walked over an ``ast`` tree (**never** ``eval``): column references, numeric /
string / bool literals, comparisons (including chained ``a < x < b``), the
boolean operators (``and`` / ``or`` / ``not`` and ``& | ~``), arithmetic, and
membership (``in`` / ``not in``). A ``@name`` token resolves from a caller-
supplied ``params`` mapping; the calling frame is never inspected. Anything else
-- a function call, an attribute, a name that is neither a column nor a parameter
-- is rejected, so an untrusted string can neither execute code nor read beyond
the named columns. ``col()`` expressions cannot express ``and``/``or``/``not``
(Python evaluates those eagerly); use ``& | ~`` and ``.isin(...)`` instead.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterator
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

# ``@name`` is not valid Python, so it is rewritten to a reserved identifier
# before parsing and mapped back to ``params[name]`` while building the tree.
_PARAM_RE = re.compile(r"@([A-Za-z_]\w*)")
_PARAM_PREFIX = "__colstore_param_"

_COMPARE_OPS: dict[type[ast.cmpop], np.ufunc] = {
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
}
_BINOP_OPS: dict[type[ast.operator], np.ufunc] = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.true_divide,
    ast.FloorDiv: np.floor_divide,
    ast.Mod: np.mod,
    ast.Pow: np.power,
    ast.BitAnd: np.bitwise_and,
    ast.BitOr: np.bitwise_or,
    ast.BitXor: np.bitwise_xor,
}
_UNARY_OPS: dict[type[ast.unaryop], np.ufunc] = {
    ast.USub: np.negative,
    ast.UAdd: np.positive,
    ast.Invert: np.invert,
    ast.Not: np.logical_not,
}
# Comparison ufuncs rebuild as ``_Compare`` (which aligns str/bytes operands) rather
# than ``_Op``, consistent with the comparison operators.
_COMPARE_UFUNCS = frozenset(_COMPARE_OPS.values())

# Comparison ufuncs whose truth over a block of values is decided by the block's
# min and max, and how each one reads when the column moves to the right of the
# operator (``scalar < col`` is ``col > scalar``). ``!=`` is absent: a block can
# almost always contain a non-equal value, so it does not prune.
_BOUND_FLIP: dict[np.ufunc, np.ufunc] = {
    np.less: np.greater,
    np.less_equal: np.greater_equal,
    np.greater: np.less,
    np.greater_equal: np.less_equal,
    np.equal: np.equal,
}


def _is_scalar_operand(value: Any) -> bool:
    """Whether ``value`` is a single scalar (not a column expression or array)."""
    return not isinstance(value, _Expr) and np.ndim(value) == 0


ReadColumn = Callable[[str], NDArray[Any]]


class QueryError(ValueError):
    """A query string or column expression is malformed or unsupported."""


def _string_kind(value: Any) -> str:
    """Return ``'S'`` / ``'U'`` if ``value`` is a NumPy bytes/unicode array, else ``''``."""
    dtype = getattr(value, "dtype", None)
    if dtype is not None and dtype.kind in ("S", "U"):
        return str(dtype.kind)
    return ""


def _align_strings(left: Any, right: Any) -> tuple[Any, Any]:
    """Coerce a str/bytes operand to the string kind of the array it compares to.

    NumPy will not compare a fixed-width-bytes column (dtype kind ``'S'``) with a
    Python ``str``, nor a unicode column (``'U'``) with ``bytes``; without this a
    string literal in a query would raise instead of matching. The array side
    sets the kind and the other operand is encoded/decoded to it.
    """
    left_kind = _string_kind(left)
    right_kind = _string_kind(right)
    if not left_kind and not right_kind:
        return left, right
    if left_kind == right_kind:
        return left, right
    kind = left_kind or right_kind
    if left_kind == kind:
        return left, np.asarray(right, dtype=kind)
    return np.asarray(left, dtype=kind), right


def _operand(value: Any, read_column: ReadColumn) -> Any:
    """Evaluate an operand: an ``_Expr`` against the columns, a scalar as itself."""
    return value._evaluate(read_column) if isinstance(value, _Expr) else value


def _operand_columns(value: Any) -> Iterator[str]:
    """Column names a (possibly scalar) operand references."""
    if isinstance(value, _Expr):
        yield from value._columns()


def _emit_operand(operand: Any, builder: Any) -> Any:
    """Rebuild an operand through ``builder``; a scalar passes through unchanged."""
    return operand._emit(builder) if isinstance(operand, _Expr) else operand


def _check_operand(value: Any) -> None:
    """Reject a raw row-length array as an arithmetic or comparison operand.

    A 1-D-or-higher array has no guarantee of matching the column's rows -- and
    would not under a filtered view or a streamed batch -- so it cannot be an
    operand. A scalar, a 0-d array, another column expression, or ``.isin(...)``
    for membership are the supported forms.
    """
    if isinstance(value, np.ndarray) and value.ndim >= 1:
        raise TypeError(
            "cannot use a raw array as an operand in a col() expression; use a scalar, "
            "another col() expression, or .isin(...) for membership."
        )


class _Expr:
    """A node in a lazy column-expression tree built by :func:`col` and operators.

    Operators build new nodes rather than computing; nothing is read until
    :meth:`_evaluate` is called against a store. Instances have no truth value
    and are not hashable -- ``==`` / ``<`` build comparison nodes, so a stray
    ``and`` / ``or`` / ``not`` (which Python would evaluate eagerly) or use as a
    dict key is a mistake, caught here with a pointed message.
    """

    __slots__ = ()
    # Make ``ndarray <op> expr`` / ``scalar <op> expr`` defer to us.
    __array_priority__ = 1_000_000

    def _evaluate(self, read_column: ReadColumn) -> Any:
        raise NotImplementedError

    def _columns(self) -> Iterator[str]:
        return iter(())

    def predicate_bounds(self) -> tuple[str, np.ufunc, Any] | None:
        """Lower this expression to ``(column, ufunc, scalar)`` block-pruning bounds.

        ``None`` for any expression that is not a single ``col(name) <op> scalar``
        comparison; :class:`_Compare` overrides it.
        """
        return None

    def _emit(self, builder: Any) -> Any:
        """Rebuild this expression through ``builder``, a node factory -- a fold
        over the tree. ``builder`` supplies ``column(name)`` / ``op(ufunc,
        operands)`` / ``isin(target, values)`` / ``where(cond, a, b)`` and gets back
        whatever it builds; the frame uses it to splice a ``col()`` expression into
        its value graph.
        """
        raise NotImplementedError

    def __bool__(self) -> NoReturn:
        raise QueryError(
            "a column expression has no truth value; combine conditions with "
            "& | ~ (not and / or / not), and apply with ds[expr] or ds.where(expr)."
        )

    __hash__ = None  # type: ignore[assignment]

    # The ufunc and array-function protocols build the same nodes as the operator
    # methods; only elementwise (single-output, non-generalized) operations are admitted.
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> _Expr:
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
                "a core dimension; column expressions support only elementwise operations."
            )
        if ufunc in _COMPARE_UFUNCS and len(inputs) == 2:
            return _Compare(ufunc, inputs[0], inputs[1])
        return _Op(ufunc, *inputs)

    def __array_function__(
        self, func: Any, types: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> _Expr:
        builder = _ARRAY_FUNCTIONS.get(func)
        if builder is None:
            raise TypeError(
                f"numpy.{getattr(func, '__name__', func)} is not supported in a column "
                "expression; build it from elementwise operations instead."
            )
        return builder(*args, **kwargs)

    # -- comparisons (Python derives the reflected forms, e.g. 30 < col -> col > 30) --
    def __lt__(self, other: Any) -> _Expr:
        return _Compare(np.less, self, other)

    def __le__(self, other: Any) -> _Expr:
        return _Compare(np.less_equal, self, other)

    def __gt__(self, other: Any) -> _Expr:
        return _Compare(np.greater, self, other)

    def __ge__(self, other: Any) -> _Expr:
        return _Compare(np.greater_equal, self, other)

    def __eq__(self, other: Any) -> _Expr:  # type: ignore[override]
        return _Compare(np.equal, self, other)

    def __ne__(self, other: Any) -> _Expr:  # type: ignore[override]
        return _Compare(np.not_equal, self, other)

    # -- boolean combinators (use & | ~) --
    def __and__(self, other: Any) -> _Expr:
        return _Op(np.bitwise_and, self, other)

    def __rand__(self, other: Any) -> _Expr:
        return _Op(np.bitwise_and, other, self)

    def __or__(self, other: Any) -> _Expr:
        return _Op(np.bitwise_or, self, other)

    def __ror__(self, other: Any) -> _Expr:
        return _Op(np.bitwise_or, other, self)

    def __xor__(self, other: Any) -> _Expr:
        return _Op(np.bitwise_xor, self, other)

    def __invert__(self) -> _Expr:
        return _Op(np.invert, self)

    # -- arithmetic (with reflected forms for scalar-on-the-left) --
    def __add__(self, other: Any) -> _Expr:
        return _Op(np.add, self, other)

    def __radd__(self, other: Any) -> _Expr:
        return _Op(np.add, other, self)

    def __sub__(self, other: Any) -> _Expr:
        return _Op(np.subtract, self, other)

    def __rsub__(self, other: Any) -> _Expr:
        return _Op(np.subtract, other, self)

    def __mul__(self, other: Any) -> _Expr:
        return _Op(np.multiply, self, other)

    def __rmul__(self, other: Any) -> _Expr:
        return _Op(np.multiply, other, self)

    def __truediv__(self, other: Any) -> _Expr:
        return _Op(np.true_divide, self, other)

    def __rtruediv__(self, other: Any) -> _Expr:
        return _Op(np.true_divide, other, self)

    def __floordiv__(self, other: Any) -> _Expr:
        return _Op(np.floor_divide, self, other)

    def __rfloordiv__(self, other: Any) -> _Expr:
        return _Op(np.floor_divide, other, self)

    def __mod__(self, other: Any) -> _Expr:
        return _Op(np.mod, self, other)

    def __rmod__(self, other: Any) -> _Expr:
        return _Op(np.mod, other, self)

    def __pow__(self, other: Any) -> _Expr:
        return _Op(np.power, self, other)

    def __rpow__(self, other: Any) -> _Expr:
        return _Op(np.power, other, self)

    def __neg__(self) -> _Expr:
        return _Op(np.negative, self)

    def isin(self, values: Any) -> _Expr:
        """Build a membership test (``in``); ``values`` is any sequence of scalars."""
        return _Isin(self, values)


class _Col(_Expr):
    """A column reference: reads the named column when evaluated."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def _evaluate(self, read_column: ReadColumn) -> Any:
        return read_column(self._name)

    def _columns(self) -> Iterator[str]:
        yield self._name

    def _emit(self, builder: Any) -> Any:
        return builder.column(self._name)


class _Op(_Expr):
    """A NumPy ufunc applied to one or more operands (arithmetic / boolean)."""

    __slots__ = ("_operands", "_ufunc")

    def __init__(self, ufunc: np.ufunc, *operands: Any) -> None:
        for operand in operands:
            _check_operand(operand)
        self._ufunc = ufunc
        self._operands = operands

    def _evaluate(self, read_column: ReadColumn) -> Any:
        return self._ufunc(*(_operand(o, read_column) for o in self._operands))

    def _columns(self) -> Iterator[str]:
        for operand in self._operands:
            yield from _operand_columns(operand)

    def _emit(self, builder: Any) -> Any:
        return builder.op(self._ufunc, [_emit_operand(o, builder) for o in self._operands])


class _Compare(_Expr):
    """A comparison ufunc, with str/bytes operands coerced to the column's kind."""

    __slots__ = ("_left", "_right", "_ufunc")

    def __init__(self, ufunc: np.ufunc, left: Any, right: Any) -> None:
        _check_operand(left)
        _check_operand(right)
        self._ufunc = ufunc
        self._left = left
        self._right = right

    def _evaluate(self, read_column: ReadColumn) -> Any:
        left = _operand(self._left, read_column)
        right = _operand(self._right, read_column)
        left, right = _align_strings(left, right)
        return self._ufunc(left, right)

    def predicate_bounds(self) -> tuple[str, np.ufunc, Any] | None:
        """Lower a ``col(name) <op> scalar`` comparison to ``(name, ufunc, scalar)``.

        Returns ``None`` unless exactly one operand is a bare column reference and
        the other is a scalar, with ``<op>`` one of ``< <= > >= ==`` (the forms a
        block's min/max can decide). A scalar-on-the-left comparison is flipped so
        the column is on the left. Used to skip blocks that cannot match a filter.
        """
        if self._ufunc not in _BOUND_FLIP:
            return None
        if isinstance(self._left, _Col) and _is_scalar_operand(self._right):
            return self._left._name, self._ufunc, self._right
        if isinstance(self._right, _Col) and _is_scalar_operand(self._left):
            return self._right._name, _BOUND_FLIP[self._ufunc], self._left
        return None

    def _columns(self) -> Iterator[str]:
        yield from _operand_columns(self._left)
        yield from _operand_columns(self._right)

    def _emit(self, builder: Any) -> Any:
        return builder.op(
            self._ufunc,
            [_emit_operand(self._left, builder), _emit_operand(self._right, builder)],
        )


def isin_test_values(values: Any) -> NDArray[Any]:
    """Normalize ``isin`` test values to a 1-D array -- a set/frozenset becomes its members.

    NumPy reads a bare ``set`` as a single object element rather than the values it holds,
    so it is expanded to a list first; a list, tuple, or array passes straight through.
    Shared by every ``isin`` surface (the query predicate, the editing-frame node, and the
    eager column view) so they expand membership sets identically.
    """
    if isinstance(values, (set, frozenset)):
        values = list(values)
    return np.asarray(values)


class _Isin(_Expr):
    """A membership test ``target in values`` (``np.isin``)."""

    __slots__ = ("_target", "_values")

    def __init__(self, target: _Expr, values: Any) -> None:
        self._target = target
        self._values = isin_test_values(values)

    def _evaluate(self, read_column: ReadColumn) -> Any:
        target = _operand(self._target, read_column)
        target, members = _align_strings(target, self._values)
        return np.isin(target, members)

    def _columns(self) -> Iterator[str]:
        yield from _operand_columns(self._target)

    def _emit(self, builder: Any) -> Any:
        return builder.isin(_emit_operand(self._target, builder), self._values)


class _Where(_Expr):
    """A ``numpy.where(cond, a, b)`` elementwise choice between two operands."""

    __slots__ = ("_a", "_b", "_cond")

    def __init__(self, cond: Any, a: Any, b: Any) -> None:
        for value in (cond, a, b):
            _check_operand(value)
        self._cond = cond
        self._a = a
        self._b = b

    def _evaluate(self, read_column: ReadColumn) -> Any:
        return np.where(
            _operand(self._cond, read_column),
            _operand(self._a, read_column),
            _operand(self._b, read_column),
        )

    def _columns(self) -> Iterator[str]:
        for operand in (self._cond, self._a, self._b):
            yield from _operand_columns(operand)

    def _emit(self, builder: Any) -> Any:
        return builder.where(
            _emit_operand(self._cond, builder),
            _emit_operand(self._a, builder),
            _emit_operand(self._b, builder),
        )


def _np_where(condition: Any, *branches: Any) -> _Expr:
    """``numpy.where(cond, x, y)`` builder for the column-expression dispatch table."""
    if len(branches) != 2:
        raise TypeError(
            "numpy.where on a column expression requires both branches: "
            "np.where(cond, x, y); the single-argument index form mixes rows and is "
            "not supported."
        )
    return _Where(condition, branches[0], branches[1])


def _np_clip(a: Any, a_min: Any = None, a_max: Any = None, **kwargs: Any) -> _Expr:
    """``numpy.clip`` builder, expressed with ``maximum`` / ``minimum`` ufuncs.

    Accepts the bounds under either ``a_min`` / ``a_max`` or ``min`` / ``max`` (both
    spellings NumPy's ``clip`` allows); ``out=`` and any other keyword are rejected.
    """
    if kwargs.get("out") is not None:
        raise TypeError("out= is not supported when building a column expression.")
    unsupported = set(kwargs) - {"min", "max", "out"}
    if unsupported:
        raise TypeError(f"unsupported argument(s) to numpy.clip: {sorted(unsupported)}.")
    low = a_min if a_min is not None else kwargs.get("min")
    high = a_max if a_max is not None else kwargs.get("max")
    node: Any = a
    if low is not None:
        node = _Op(np.maximum, node, low)
    if high is not None:
        node = _Op(np.minimum, node, high)
    if not isinstance(node, _Expr):
        raise TypeError("numpy.clip needs the clipped value to be a column expression.")
    return node


# Non-ufunc NumPy functions admitted in a column expression, each mapped to the node
# it builds. Anything outside this table raises in ``_Expr.__array_function__``.
_ARRAY_FUNCTIONS: dict[Any, Callable[..., _Expr]] = {np.where: _np_where, np.clip: _np_clip}


def col(name: str) -> _Expr:
    """A lazy reference to a column, for building filter expressions.

    ``col("pt") > 30`` and combinations (``(col("pt") > 30) & col("ok")``,
    ``col("eta").isin([0, 1])``) build a lazy predicate that reads nothing until
    applied with ``ds[expr]`` / :meth:`ColStoreReader.where` / passed to
    :meth:`ColStoreReader.query`. Combine conditions with ``& | ~`` (not
    ``and`` / ``or`` / ``not``), and use ``.isin(...)`` for membership.
    """
    if not isinstance(name, str):
        raise TypeError(f"col() name must be a string; got {type(name).__name__}.")
    return _Col(name)


# ---- String predicate -> the same _Expr tree -------------------------------


def parse_query(expression: str, columns: frozenset[str], params: dict[str, Any] | None) -> _Expr:
    """Parse a predicate string into an :class:`_Expr`, validating columns/params.

    Builds the same tree :func:`col` and operators build, so the string and
    object forms evaluate identically. Raises :class:`QueryError` on a syntax
    error, an unsupported construct, an unknown column name, or an undefined
    ``@param``.
    """
    prepared = _PARAM_RE.sub(lambda match: _PARAM_PREFIX + match.group(1), expression)
    try:
        tree = ast.parse(prepared, mode="eval")
    except SyntaxError as exc:
        raise QueryError(f"could not parse query {expression!r}: {exc.msg}.") from None
    built = _build(tree.body, columns, params or {}, expression)
    if not isinstance(built, _Expr):
        raise QueryError(f"query {expression!r} must reference at least one column.")
    return built


def _fail(source: str, message: str) -> NoReturn:
    raise QueryError(f"in query {source!r}: {message}")


def _build(node: ast.AST, columns: frozenset[str], params: dict[str, Any], source: str) -> Any:
    """Map one whitelisted ``ast`` node to an ``_Expr`` (or a literal scalar)."""
    if isinstance(node, ast.BoolOp):
        reducer = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        built = _build(node.values[0], columns, params, source)
        for value in node.values[1:]:
            built = _Op(reducer, built, _build(value, columns, params, source))
        return built
    if isinstance(node, ast.UnaryOp):
        ufunc = _UNARY_OPS.get(type(node.op))
        if ufunc is None:
            _fail(source, f"unsupported unary operator {type(node.op).__name__}.")
        return _Op(ufunc, _build(node.operand, columns, params, source))
    if isinstance(node, ast.BinOp):
        ufunc = _BINOP_OPS.get(type(node.op))
        if ufunc is None:
            _fail(source, f"unsupported operator {type(node.op).__name__}.")
        return _Op(
            ufunc,
            _build(node.left, columns, params, source),
            _build(node.right, columns, params, source),
        )
    if isinstance(node, ast.Compare):
        return _build_compare(node, columns, params, source)
    if isinstance(node, ast.Name):
        return _build_name(node, columns, params, source)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_build(element, columns, params, source) for element in node.elts]
    _fail(source, f"unsupported expression element {type(node).__name__}.")


def _build_compare(
    node: ast.Compare, columns: frozenset[str], params: dict[str, Any], source: str
) -> _Expr:
    left = _build(node.left, columns, params, source)
    result: _Expr | None = None
    for op, comparator in zip(node.ops, node.comparators, strict=True):
        right = _build(comparator, columns, params, source)
        if isinstance(op, ast.In):
            piece: _Expr = _Isin(_as_expr(left, source), right)
        elif isinstance(op, ast.NotIn):
            piece = _Op(np.logical_not, _Isin(_as_expr(left, source), right))
        else:
            ufunc = _COMPARE_OPS.get(type(op))
            if ufunc is None:
                _fail(source, f"unsupported comparison {type(op).__name__}.")
            piece = _Compare(ufunc, left, right)
        result = piece if result is None else _Op(np.bitwise_and, result, piece)
        left = right
    assert result is not None  # a Compare always has at least one operator
    return result


def _build_name(
    node: ast.Name, columns: frozenset[str], params: dict[str, Any], source: str
) -> Any:
    name = node.id
    if name.startswith(_PARAM_PREFIX):
        key = name[len(_PARAM_PREFIX) :]
        if key not in params:
            _fail(source, f"undefined parameter @{key}; pass params={{{key!r}: ...}}.")
        return params[key]
    if name in columns:
        return _Col(name)
    _fail(source, f"unknown name {name!r}: not a column or an @param.")


def _as_expr(value: Any, source: str) -> _Expr:
    if isinstance(value, _Expr):
        return value
    _fail(source, "the left side of 'in' must reference a column.")


# ---- Validation and evaluation against a store -----------------------------


def validate_predicate(expr: _Expr, columns: frozenset[str], probe: ReadColumn) -> None:
    """Eagerly reject a predicate that is unusable, without reading data.

    Checks every referenced column exists and that the predicate reduces to a
    boolean condition over at least one column, the latter via a 0-row dtype
    probe (``probe(name)`` returns an empty typed array). Raises
    :class:`QueryError`; reads no row data.
    """
    referenced = set(expr._columns())
    missing = sorted(name for name in referenced if name not in columns)
    if missing:
        raise QueryError(f"query references unknown column(s) {missing}; have {sorted(columns)}.")
    if not referenced:
        raise QueryError("a query must reference at least one column.")
    result = np.asarray(expr._evaluate(probe))
    if result.dtype.kind != "b":
        raise QueryError(
            f"a query must be a boolean condition; this one evaluates to dtype {result.dtype}."
        )


def evaluate_mask(expr: _Expr, read_column: ReadColumn, n_rows: int) -> NDArray[np.bool_]:
    """Evaluate a (validated) predicate to a boolean row mask of length ``n_rows``."""
    mask = np.asarray(expr._evaluate(read_column))
    if mask.ndim != 1 or mask.shape[0] != n_rows:
        raise QueryError(
            f"a query must reduce to a per-row condition of length {n_rows}; "
            f"got an array of shape {mask.shape}."
        )
    return mask
