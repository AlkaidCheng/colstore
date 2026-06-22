"""Tests for the deferred column expression graph (colstore.frame).

Pure-Python and IO-free: the graph, evaluator, dtype probe, CSE memo, and
length validation are exercised against eager NumPy. NativeColumn is driven by a
minimal stub store so its contract (dtype and length without IO, data-free
zero-length read, range read delegated to the store) can be checked without a
built extension or a real file. End-to-end reads against a real reader belong to
the frame-integration layer, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from colstore.frame import (
    Apply,
    ConstColumn,
    Expr,
    MemoryColumn,
    NativeColumn,
    UFunc,
    Where,
    as_expr,
    declared_length,
    evaluate,
    result_dtype,
    validate_length,
)


class _StubStore:
    """Minimal stand-in for ColStoreReader: just enough for NativeColumn.

    Records every ``_gather_one`` call so tests can assert that dtype/length
    queries and the zero-length probe read no data.
    """

    def __init__(self, columns: dict[str, np.ndarray]) -> None:
        self._columns = {name: np.asarray(arr) for name, arr in columns.items()}
        self.n_rows = int(next(iter(self._columns.values())).shape[0])
        self.gather_calls: list[tuple[str, slice]] = []

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        return {name: arr.dtype for name, arr in self._columns.items()}

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    def _native_dtype(self, name: str) -> np.dtype:
        return self._columns[name].dtype.newbyteorder("=")

    def _gather_one(self, name: str, row_indexer: slice) -> np.ndarray:
        self.gather_calls.append((name, row_indexer))
        return np.array(self._columns[name][row_indexer], copy=True)


class _CountingColumn(MemoryColumn):
    """A MemoryColumn that counts how many times it is read."""

    __slots__ = ("reads",)

    def __init__(self, array: np.ndarray) -> None:
        super().__init__(array)
        self.reads = 0

    def _read(self, rows: slice | np.ndarray) -> np.ndarray:
        self.reads += 1
        return super()._read(rows)


@pytest.fixture
def xy() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(256).astype(np.float64)
    y = rng.standard_normal(256).astype(np.float64)
    return x, y


def _full(node: Expr, n: int) -> np.ndarray:
    """Evaluate a node over its whole range with a fresh memo."""
    return evaluate(node, slice(0, n), {})


# -- graph construction --


def test_operators_build_expr_nodes(xy):
    x, y = xy
    a, b = MemoryColumn(x), MemoryColumn(y)
    assert isinstance(a + b, UFunc)
    assert isinstance((a + b) * 2, UFunc)
    assert isinstance(a > 0, UFunc)
    assert isinstance(-a, UFunc)
    assert isinstance(np.log(a), UFunc)


def test_arithmetic_matches_numpy(xy):
    x, y = xy
    a, b = MemoryColumn(x), MemoryColumn(y)
    n = len(x)
    np.testing.assert_array_equal(_full((a + b) * 2, n), (x + y) * 2)
    np.testing.assert_array_equal(_full(a - b, n), x - y)
    np.testing.assert_array_equal(_full(a * b, n), x * y)
    np.testing.assert_array_equal(_full(a / b, n), x / y)
    np.testing.assert_array_equal(_full(a**2, n), x**2)
    np.testing.assert_array_equal(_full(-a, n), -x)
    np.testing.assert_array_equal(_full(abs(a), n), np.abs(x))


def test_numpy_ufuncs_match(xy):
    x, _ = xy
    a = MemoryColumn(np.abs(x) + 0.1)  # keep positive for log/sqrt
    n = len(x)
    np.testing.assert_allclose(_full(np.log(a), n), np.log(np.abs(x) + 0.1))
    np.testing.assert_allclose(_full(np.sqrt(a), n), np.sqrt(np.abs(x) + 0.1))
    np.testing.assert_allclose(_full(np.sin(a), n), np.sin(np.abs(x) + 0.1))


def test_scalar_and_reflected_operands(xy):
    x, _ = xy
    a = MemoryColumn(x)
    n = len(x)
    np.testing.assert_array_equal(_full(2 * a, n), 2 * x)
    np.testing.assert_array_equal(_full(a - 1, n), x - 1)
    np.testing.assert_array_equal(_full(2 - a, n), 2 - x)
    np.testing.assert_array_equal(_full(np.float64(3.0) * a, n), 3.0 * x)


def test_comparison_yields_bool(xy):
    x, _ = xy
    a = MemoryColumn(x)
    out = _full(a > 0, len(x))
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out, x > 0)


def test_partial_range_matches_slice(xy):
    x, y = xy
    a, b = MemoryColumn(x), MemoryColumn(y)
    out = evaluate((a + b) * 2, slice(3, 7), {})
    np.testing.assert_array_equal(out, (x[3:7] + y[3:7]) * 2)


# -- rejected operations --


def test_reductions_rejected(xy):
    x, y = xy
    a, b = MemoryColumn(x), MemoryColumn(y)
    with pytest.raises(TypeError, match="reduction or accumulation"):
        np.add.reduce(a)
    with pytest.raises(TypeError, match="reduction or accumulation"):
        np.maximum.accumulate(a)
    with pytest.raises(TypeError, match="reduction or accumulation"):
        np.add.outer(a, b)


def test_out_kwarg_rejected(xy):
    x, y = xy
    a, b = MemoryColumn(x), MemoryColumn(y)
    with pytest.raises(TypeError, match="out="):
        np.add(a, b, out=np.empty(len(x)))


def test_elementwise_ufuncs_defer(xy):
    x, y = xy
    a, b = MemoryColumn(x), MemoryColumn(y)
    # Any single-output elementwise ufunc builds a node and evaluates as numpy would
    # (hypot / arctan2 / ... -- the kinematics path).
    for node, expected in [
        (np.hypot(a, b), np.hypot(x, y)),
        (np.arctan2(a, b), np.arctan2(x, y)),
        (np.logaddexp(a, b), np.logaddexp(x, y)),
        (np.copysign(a, b), np.copysign(x, y)),
    ]:
        assert isinstance(node, Expr)
        np.testing.assert_allclose(_full(node, len(x)), expected)


def test_multi_output_ufunc_rejected(xy):
    x, _ = xy
    a = MemoryColumn(x)
    with pytest.raises(TypeError, match="multiple outputs"):
        np.modf(a)


def test_np_where_and_clip_defer(xy):
    x, y = xy
    a, b = MemoryColumn(x), MemoryColumn(y)
    np.testing.assert_array_equal(_full(np.where(a > 0, a, b), len(x)), np.where(x > 0, x, y))
    np.testing.assert_array_equal(_full(np.where(a > 0, a, 0.0), len(x)), np.where(x > 0, x, 0.0))
    np.testing.assert_array_equal(_full(np.clip(a, -0.5, 0.5), len(x)), np.clip(x, -0.5, 0.5))


def test_expr_where_method(xy):
    x, y = xy
    a, b = MemoryColumn(x), MemoryColumn(y)
    np.testing.assert_array_equal(_full(a.where(a > 0, b), len(x)), np.where(x > 0, x, y))
    np.testing.assert_array_equal(_full(a.where(a > 0), len(x)), np.where(x > 0, x, np.nan))


def test_unsupported_array_function_rejected(xy):
    x, _ = xy
    a = MemoryColumn(x)
    with pytest.raises(TypeError, match="not supported in a column expression"):
        np.cumsum(a)  # row-mixing, not in the elementwise dispatch table


def test_generalized_ufunc_rejected(xy):
    x, y = xy
    a, b = MemoryColumn(x), MemoryColumn(y)
    with pytest.raises(TypeError, match="generalized ufunc"):
        np.matmul(a, b)  # __call__/nout==1 but mixes rows across a core dimension


def test_np_clip_keyword_and_positional_bounds(xy):
    x, _ = xy
    a = MemoryColumn(x)
    kw = _full(np.clip(a, min=-0.5, max=0.5), len(x))  # keyword bounds
    pos = _full(np.clip(a, -0.5, 0.5), len(x))  # positional bounds
    np.testing.assert_array_equal(kw, pos)
    np.testing.assert_array_equal(kw, np.clip(x, -0.5, 0.5))


def test_np_clip_out_rejected(xy):
    x, _ = xy
    a = MemoryColumn(x)
    with pytest.raises(TypeError, match="out="):
        np.clip(a, -0.5, 0.5, out=np.empty(len(x)))


def test_np_where_single_arg_rejected(xy):
    x, _ = xy
    a = MemoryColumn(x)
    with pytest.raises(TypeError, match="requires both branches"):
        np.where(a > 0)


def test_where_all_scalar_rejected():
    with pytest.raises(TypeError, match="at least one column operand"):
        Where(True, 1, 2)


def test_apply_runs_func_and_infers_dtype_from_empty(xy):
    x, y = xy
    a, b = MemoryColumn(x), MemoryColumn(y)
    node = Apply(lambda u, v: np.hypot(u, v), (a, b))
    assert result_dtype(node) == np.dtype(np.float64)  # empty-probe, reads no data
    np.testing.assert_allclose(_full(node, len(x)), np.hypot(x, y))
    assert declared_length(node) == len(x)  # length comes from the input leaves


def test_apply_out_dtype_overrides_probe(xy):
    x, _ = xy
    a = MemoryColumn(x)
    node = Apply(lambda u: u * 2, (a,), out_dtype="float32")
    assert result_dtype(node) == np.dtype(np.float32)
    assert _full(node, len(x)).dtype == np.dtype(np.float32)


def test_apply_length_changing_func_rejected_at_eval(xy):
    x, _ = xy
    a = MemoryColumn(x)
    node = Apply(lambda u: u[:3], (a,), out_dtype="float64")  # out_dtype skips the build probe
    with pytest.raises(ValueError, match="1-D array of length"):
        _full(node, len(x))


def test_apply_reduction_rejected_at_build(xy):
    x, _ = xy
    a = MemoryColumn(x)
    with pytest.raises(TypeError, match="unsupported result"):
        Apply(lambda u: u.sum(), (a,))  # collapses to 0-d on empty input


def test_apply_empty_hostile_func_requires_out_dtype(xy):
    x, _ = xy
    a = MemoryColumn(x)
    with pytest.raises(TypeError, match="out_dtype"):
        Apply(lambda u: u / u[0], (a,))  # indexes [0]; raises on empty input
    node = Apply(lambda u: u / u[0], (a,), out_dtype="float64")  # ok once declared
    np.testing.assert_allclose(_full(node, len(x)), x / x[0])


def test_apply_value_dependent_dtype_rejected_at_eval(xy):
    x, _ = xy
    a = MemoryColumn(np.arange(len(x), dtype=np.int64))
    # int8 on the empty probe, int64 on the real batch -> a data-dependent dtype.
    node = Apply(lambda u: u.astype(np.int8) if u.size == 0 else u, (a,))
    with pytest.raises(ValueError, match="depends on the data"):
        _full(node, len(x))


def test_apply_declared_out_dtype_coerces_batch(xy):
    x, _ = xy
    a = MemoryColumn(x)
    node = Apply(lambda u: u, (a,), out_dtype="float32")  # returns float64, declared float32
    got = _full(node, len(x))
    assert got.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(got, x.astype(np.float32), rtol=1e-6)


def test_apply_object_return_rejected_at_eval(xy):
    x, _ = xy
    a = MemoryColumn(x)
    # numeric on the empty probe, object on the real batch.
    node = Apply(lambda u: u if u.size == 0 else np.array([object()] * len(u)), (a,))
    with pytest.raises(TypeError, match="unsupported dtype"):
        _full(node, len(x))


def test_raw_ndarray_operand_rejected(xy):
    x, _ = xy
    a = MemoryColumn(x)
    with pytest.raises(TypeError, match="raw ndarray"):
        _ = a + np.arange(len(x))
    with pytest.raises(TypeError, match="raw ndarray"):
        _ = np.arange(len(x)) + a


def test_truth_value_and_array_coercion_guarded(xy):
    x, _ = xy
    a = MemoryColumn(x)
    with pytest.raises(TypeError, match="truth value"):
        bool(a > 0)
    with pytest.raises(TypeError, match="cannot be converted to an array"):
        np.asarray(a)


# -- dtype probe --


def test_result_dtype_follows_numpy_promotion():
    i = MemoryColumn(np.arange(4, dtype=np.int64))
    f32 = MemoryColumn(np.ones(4, dtype=np.float32))
    f64 = MemoryColumn(np.ones(4, dtype=np.float64))
    assert result_dtype(i + i) == np.int64
    assert result_dtype(i / i) == np.float64  # true division promotes to float
    assert result_dtype(i > 0) == np.bool_
    assert result_dtype(f32 + f64) == np.float64
    assert result_dtype(f32 * 2) == np.float32  # weak scalar promotion (NEP 50)


# -- NativeColumn against the stub store (no IO for dtype/length/probe) --


def test_native_column_dtype_and_length_read_no_data():
    x = np.arange(10, dtype=np.int32)
    store = _StubStore({"x": x})
    col = NativeColumn(store, "x")
    assert col.dtype == np.int32
    assert col.name == "x"
    assert col._length() == 10
    assert store.gather_calls == []  # dtype and length touch no data


def test_native_column_dtype_probe_reads_no_data():
    store = _StubStore({"x": np.arange(10, dtype=np.int64)})
    col = NativeColumn(store, "x")
    assert result_dtype(col * 2 + 1) == np.int64
    assert store.gather_calls == []  # the zero-length probe is data-free


def test_native_column_range_read_delegates():
    x = np.arange(10, dtype=np.int64)
    store = _StubStore({"x": x})
    col = NativeColumn(store, "x")
    out = col._read(slice(2, 5))
    np.testing.assert_array_equal(out, x[2:5])
    assert store.gather_calls == [("x", slice(2, 5))]


def test_native_column_zero_length_read_is_data_free():
    store = _StubStore({"x": np.arange(10, dtype=np.int64)})
    col = NativeColumn(store, "x")
    empty = col._read(slice(0, 0))
    assert empty.shape == (0,)
    assert empty.dtype == np.int64
    assert store.gather_calls == []


def test_native_column_unknown_name_raises():
    store = _StubStore({"x": np.arange(4)})
    with pytest.raises(KeyError, match="not in the store"):
        NativeColumn(store, "missing")


# -- MemoryColumn / ConstColumn --


def test_memory_column_reference_vs_copy():
    arr = np.arange(5, dtype=np.float64)
    by_ref = MemoryColumn(arr)
    snapshot = MemoryColumn(arr, copy=True)
    arr[0] = 99.0
    assert _full(by_ref, 5)[0] == 99.0  # reference reflects the later mutation
    assert _full(snapshot, 5)[0] == 0.0  # snapshot froze the original value


def test_memory_column_rejects_non_1d():
    with pytest.raises(ValueError, match="1-D"):
        MemoryColumn(np.zeros((2, 2)))


def test_const_column():
    c = ConstColumn(7)
    assert c._length() is None
    np.testing.assert_array_equal(c._read(slice(0, 3)), np.full(3, 7))
    assert ConstColumn(1.5).dtype == np.float64
    assert ConstColumn(7, dtype=np.int16).dtype == np.int16


def test_as_expr_dispatch():
    e = MemoryColumn(np.arange(3))
    assert as_expr(e) is e  # Expr passes through
    assert isinstance(as_expr(np.arange(3)), MemoryColumn)
    assert isinstance(as_expr([1, 2, 3]), MemoryColumn)  # array-like
    assert isinstance(as_expr(5), ConstColumn)
    assert isinstance(as_expr(np.array(5)), ConstColumn)  # 0-d -> scalar


def test_as_expr_copy_flag():
    arr = np.arange(3, dtype=np.float64)
    held = as_expr(arr)
    snap = as_expr(arr, copy=True)
    arr[0] = 42.0
    assert _full(held, 3)[0] == 42.0
    assert _full(snap, 3)[0] == 0.0


# -- length resolution and validation --


def test_declared_length():
    a = MemoryColumn(np.arange(8))
    b = MemoryColumn(np.arange(8))
    assert declared_length(a + b) == 8
    assert declared_length(a + 1) == 8  # scalar is length-agnostic
    assert declared_length(ConstColumn(0)) is None  # all-constant


def test_declared_length_mismatch_raises():
    a = MemoryColumn(np.arange(8))
    short = MemoryColumn(np.arange(4))
    with pytest.raises(ValueError, match="different lengths"):
        declared_length(a + short)


def test_validate_length():
    validate_length(MemoryColumn(np.arange(8)) * 2, 8)  # exact match: ok
    validate_length(ConstColumn(0), 8)  # constant broadcast-fills: ok
    with pytest.raises(ValueError, match="does not match"):
        validate_length(MemoryColumn(np.arange(4)), 8)
    with pytest.raises(ValueError, match="does not match"):
        validate_length(MemoryColumn(np.arange(1)), 8)  # length-1 does not broadcast


# -- common-subexpression elimination via the shared memo --


def test_cse_structural_dedup_across_columns(xy):
    x, y = xy
    a, b = _CountingColumn(x), _CountingColumn(y)
    n = len(x)
    # Two structurally identical `a + b` subtrees built independently, exactly
    # as writing ds['x'] + ds['y'] twice would produce.
    c1 = (a + b) * 2
    c2 = (a + b) - 1
    memo: dict = {}
    r1 = evaluate(c1, slice(0, n), memo)
    r2 = evaluate(c2, slice(0, n), memo)
    np.testing.assert_array_equal(r1, (x + y) * 2)
    np.testing.assert_array_equal(r2, (x + y) - 1)
    assert a.reads == 1  # the shared a+b (and its leaves) computed once
    assert b.reads == 1


def test_no_cse_across_separate_memos(xy):
    x, _ = xy
    a = _CountingColumn(x)
    evaluate(a + 1, slice(0, len(x)), {})
    evaluate(a + 2, slice(0, len(x)), {})
    assert a.reads == 2  # a fresh memo per batch releases the working set


def test_cse_dedup_within_one_expression(xy):
    x, _ = xy
    a = _CountingColumn(x)
    evaluate(a + a, slice(0, len(x)), {})
    assert a.reads == 1


# -- composed helpers --


def test_round_matches_numpy(xy):
    x, _ = xy
    a = MemoryColumn(x)
    n = len(x)
    np.testing.assert_array_equal(_full(a.round(), n), np.round(x))
    np.testing.assert_allclose(_full(a.round(2), n), np.round(x, 2))


def test_clip_matches_numpy(xy):
    x, _ = xy
    a = MemoryColumn(x)
    n = len(x)
    np.testing.assert_array_equal(_full(a.clip(-0.5, 0.5), n), np.clip(x, -0.5, 0.5))
    np.testing.assert_array_equal(_full(a.clip(low=0.0), n), np.clip(x, 0.0, None))
    np.testing.assert_array_equal(_full(a.clip(high=0.0), n), np.clip(x, None, 0.0))
