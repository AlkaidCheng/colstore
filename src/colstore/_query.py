"""A safe predicate parser for :meth:`_ReaderBase.query`.

Turns a pandas-style predicate string into a boolean row mask, reading only the
columns it names. The grammar is a strict whitelist walked over an ``ast`` tree
-- never ``eval`` -- so a query expresses exactly: column references, numeric /
string / bool literals, comparisons (including chained ``a < x < b``), the
boolean operators (``and`` / ``or`` / ``not`` and ``& | ~``), arithmetic, and
membership (``in`` / ``not in``). A ``@name`` token resolves from a caller-
supplied ``params`` mapping; the calling frame is never inspected. Any other
construct -- a function call, an attribute, a subscript, a name that is neither a
column nor a parameter -- is rejected, so an untrusted string can neither execute
code nor read beyond the named columns.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

# ``@name`` is not valid Python, so it is rewritten to a reserved identifier
# before parsing and mapped back to ``params[name]`` during evaluation.
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


class QueryError(ValueError):
    """A query string is malformed or uses an unsupported construct."""


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


class _Evaluator:
    """Walks the whitelisted ``ast`` nodes, reading each named column once."""

    def __init__(
        self,
        columns: frozenset[str],
        read_column: Callable[[str], NDArray[Any]],
        params: dict[str, Any],
        source: str,
    ) -> None:
        self._columns = columns
        self._read = read_column
        self._params = params
        self._source = source
        self._cache: dict[str, NDArray[Any]] = {}

    def _fail(self, message: str) -> NoReturn:
        raise QueryError(f"in query {self._source!r}: {message}")

    def visit(self, node: ast.AST) -> Any:
        if isinstance(node, ast.BoolOp):
            return self._bool_op(node)
        if isinstance(node, ast.UnaryOp):
            return self._unary_op(node)
        if isinstance(node, ast.BinOp):
            return self._bin_op(node)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.Name):
            return self._name(node)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.visit(element) for element in node.elts]
        self._fail(f"unsupported expression element {type(node).__name__}.")

    def _name(self, node: ast.Name) -> Any:
        name = node.id
        if name.startswith(_PARAM_PREFIX):
            key = name[len(_PARAM_PREFIX) :]
            if key not in self._params:
                self._fail(f"undefined parameter @{key}; pass params={{{key!r}: ...}}.")
            return self._params[key]
        if name in self._columns:
            cached = self._cache.get(name)
            if cached is None:
                cached = self._read(name)
                self._cache[name] = cached
            return cached
        self._fail(f"unknown name {name!r}: not a column or an @param.")

    def _bool_op(self, node: ast.BoolOp) -> Any:
        reducer = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        result = self.visit(node.values[0])
        for value in node.values[1:]:
            result = reducer(result, self.visit(value))
        return result

    def _unary_op(self, node: ast.UnaryOp) -> Any:
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            self._fail(f"unsupported unary operator {type(node.op).__name__}.")
        return op(self.visit(node.operand))

    def _bin_op(self, node: ast.BinOp) -> Any:
        op = _BINOP_OPS.get(type(node.op))
        if op is None:
            self._fail(f"unsupported operator {type(node.op).__name__}.")
        return op(self.visit(node.left), self.visit(node.right))

    def _compare(self, node: ast.Compare) -> Any:
        # Chained comparisons ``a < x < b`` combine pairwise with logical-and.
        left = self.visit(node.left)
        result: Any = None
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = self.visit(comparator)
            piece = self._compare_one(op, left, right)
            result = piece if result is None else np.logical_and(result, piece)
            left = right
        return result

    def _compare_one(self, op: ast.cmpop, left: Any, right: Any) -> Any:
        if isinstance(op, (ast.In, ast.NotIn)):
            left, members = _align_strings(left, np.asarray(right))
            found = np.isin(left, members)
            return found if isinstance(op, ast.In) else np.logical_not(found)
        compare = _COMPARE_OPS.get(type(op))
        if compare is None:
            self._fail(f"unsupported comparison {type(op).__name__}.")
        left, right = _align_strings(left, right)
        return compare(left, right)


def evaluate_query(
    expression: str,
    columns: frozenset[str],
    read_column: Callable[[str], NDArray[Any]],
    params: dict[str, Any] | None = None,
) -> NDArray[np.bool_]:
    """Evaluate ``expression`` to a boolean row mask over ``columns``.

    ``read_column(name)`` returns one column as a 1-D array (each referenced
    column is read once). ``params`` supplies ``@name`` values. Raises
    :class:`QueryError` on a malformed or unsupported expression, or when the
    result is not boolean.
    """
    prepared = _PARAM_RE.sub(lambda match: _PARAM_PREFIX + match.group(1), expression)
    try:
        tree = ast.parse(prepared, mode="eval")
    except SyntaxError as exc:
        raise QueryError(f"could not parse query {expression!r}: {exc.msg}.") from None
    result = _Evaluator(columns, read_column, params or {}, expression).visit(tree.body)
    mask = np.asarray(result)
    if mask.dtype.kind != "b":
        raise QueryError(
            f"query {expression!r} must be a boolean condition; it evaluated to "
            f"dtype {mask.dtype}."
        )
    return mask
