"""Tests for the pandas-style terminals on a single column.

A column handle -- a reader/dataset ``ColumnView`` (``ds[name]``) or a frame's
``FrameColumn`` (``frame[name]``) -- offers reductions (``sum`` / ``mean`` /
``min`` / ``max`` / ``count``), 1-D materialization (``array`` / ``np.asarray``),
and, for the whole frame, a structured ``np.asarray(frame)``. The reductions are
shared through one ``ColumnReductions`` mixin and stream over the column's current
row selection, so a filtered handle reduces only its selected rows.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import colstore
from colstore import col
from colstore.frame import ColumnReductions, Expr, FrameColumn, Isin
from colstore.view import ColumnView


def _raw_expr_nodes(node):
    """Yield every Expr node in the tree as stored, without the walkers' unwrapping."""
    yield node
    children = list(getattr(node, "_inputs", ()))
    children += [getattr(node, attr, None) for attr in ("_input", "_target", "_cond", "_a", "_b")]
    for child in children:
        if isinstance(child, Expr):
            yield from _raw_expr_nodes(child)


@pytest.fixture()
def store(tmp_path):
    data = {
        "a": np.arange(10, dtype=np.float64),
        "b": np.arange(10, dtype=np.int64),
        "w": np.full(10, 2.0),
    }
    ds = colstore.store(data, tmp_path / "t.cstore", show_progress=False)
    yield ds
    ds.close()


# ---- reductions on a reader ColumnView (ds[name]) --------------------------


def test_column_view_reductions(store):
    assert store["a"].sum() == 45.0
    assert store["a"].mean() == 4.5
    assert store["a"].min() == 0.0
    assert store["a"].max() == 9.0
    assert store["a"].count() == 10


def test_column_view_reduction_respects_row_selection(store):
    # ds[mask, name] is a ColumnView over a row subset; the reduction sees only it.
    hot = store[col("a") >= 5, "a"]
    assert hot.mean() == 7.0  # mean of 5..9
    assert hot.count() == 5


# ---- reductions on a frame FrameColumn (frame[name]) -----------------------


def test_frame_column_reductions(store):
    df = store.edit()
    assert df["a"].sum() == 45.0
    assert df["a"].mean() == 4.5
    assert df["a"].min() == 0.0
    assert df["a"].max() == 9.0
    assert df["a"].count() == 10


def test_frame_column_reduction_respects_where(store):
    df = store.edit().where("a >= 5")
    assert df["a"].mean() == 7.0
    assert df["a"].count() == 5


def test_reductions_share_one_mixin():
    # The reduction definitions live in exactly one place for both column handles.
    assert issubclass(FrameColumn, ColumnReductions)
    assert issubclass(ColumnView, ColumnReductions)


# ---- materialization: array() and the NumPy array interface ----------------


def test_frame_column_array_respects_selection(store):
    df = store.edit().where("a >= 5")
    assert df["a"].array().tolist() == [5.0, 6.0, 7.0, 8.0, 9.0]
    # frame.array(name) is the same thing.
    assert df.array("a").tolist() == df["a"].array().tolist()


def test_np_asarray_of_frame_column(store):
    df = store.edit()
    # No numpy DeprecationWarning about the __array__ copy keyword.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = np.asarray(df["a"])
    assert out.tolist() == list(range(10))


def test_np_array_of_frame_column_casts_dtype(store):
    df = store.edit()
    out = np.array(df["a"], dtype=np.int64)
    assert out.dtype == np.int64
    assert out.tolist() == list(range(10))


def test_frame_column_array_copy_false_raises(store):
    df = store.edit()
    with pytest.raises(ValueError, match="without copying"):
        np.array(df["a"], copy=False)


def test_np_asarray_of_frame_is_a_record_array(store):
    # np.asarray(frame) yields the data (a structured array), not the column names.
    df = store.edit()
    out = np.asarray(df)
    assert out.dtype.names == ("a", "b", "w")
    assert out["a"].tolist() == list(range(10))


def test_np_array_of_frame_copy_false_raises(store):
    df = store.edit()
    with pytest.raises(ValueError, match="without copying"):
        np.array(df, copy=False)


# ---- numpy reduction protocol routes through the streaming reduction -------


def test_numpy_reduction_dispatches_to_column_view(store):
    # np.mean/np.sum etc. call column.mean(axis=, dtype=, out=) on a materialized
    # ColumnView; they must accept those parameters and reduce the whole column.
    assert np.mean(store["a"]) == 4.5
    assert np.sum(store["a"]) == 45.0
    assert np.min(store["a"]) == 0.0
    assert np.max(store["a"]) == 9.0


def test_numpy_reduction_on_frame_column_is_eager(store):
    # A NumPy reduction on a frame column is an eager terminal over the frame's
    # selection -- the same as the .sum()/.mean() method -- not a lazy node.
    df = store.edit()
    assert np.sum(df["a"]) == 45.0
    assert np.mean(df["a"]) == 4.5
    assert np.min(df["a"]) == 0.0
    assert np.max(df["a"]) == 9.0
    assert np.mean(df.where("a >= 5")["a"]) == 7.0  # reduces only the selected rows


def test_numpy_reduction_on_bare_expression_raises(store):
    # A composed expression carries no frame, so it has no rows to reduce; the error
    # points at the frame-bound forms.
    df = store.edit()
    with pytest.raises(TypeError, match="reduces over rows"):
        np.sum(df["a"] + 2)
    # but elementwise NumPy on a frame column stays lazy
    assert isinstance(np.log(df["a"]), Expr)


def test_reduction_rejects_unsupported_axis(store):
    with pytest.raises(TypeError, match="axis"):
        store["a"].mean(axis=1)


def test_reduction_rejects_silently_ignored_numpy_params(store):
    # A column reduction cannot honor dtype/out/keepdims/where/initial; rather than
    # silently ignore them (and return a result that disregards the request), reject.
    cv = store["a"]
    with pytest.raises(TypeError, match="dtype"):
        cv.sum(dtype=np.float64)
    with pytest.raises(TypeError, match="keepdims"):
        cv.min(keepdims=True)
    with pytest.raises(TypeError, match="where"):
        np.min(cv, where=np.array([True] * 5 + [False] * 5), initial=0.0)


# ---- graph hygiene: a FrameColumn never persists in a stored graph ----------


def test_frame_column_stripped_at_node_construction(store):
    df = store.edit()
    inner_a, inner_b = df._columns["a"], df._columns["b"]
    # Composition strips the frame-bound wrapper down to the wrapped node, in every
    # operand position and through every node constructor, so no frame back-reference
    # is captured. One assertion per node type (UFunc / Cast / Isin / Where).
    assert (df["a"] * 2)._inputs[0] is inner_a  # UFunc via _binop
    assert (df["a"] + df["b"])._inputs == (inner_a, inner_b)  # both operands
    assert (-df["a"])._inputs[0] is inner_a  # UFunc via _unop
    assert np.maximum(df["a"], df["b"])._inputs == (inner_a, inner_b)  # UFunc via __array_ufunc__
    assert df["a"].clip(0, 5)._inputs[0]._inputs[0] is inner_a  # nested UFunc (clip)
    assert df["a"].astype("int32")._input is inner_a  # Cast
    assert Isin(df["a"], [1, 2])._target is inner_a  # Isin (no single-call user builder)
    assert df["a"].where(df["a"] > 5, 0.0)._a is inner_a  # Where (and its condition)


def test_stored_composed_column_holds_no_frame_column(store):
    df = store.edit()
    # A column exercising several node types (UFunc, Cast, Where) -- none may hold a
    # FrameColumn once stored, or the frame back-reference / cycle would be back.
    df2 = df.with_columns(
        x=np.maximum(df["a"] * 2, df["b"]).astype("float64").where(df["a"] > 3, 0.0),
    )
    stored = df2._columns["x"]
    assert all(not isinstance(n, FrameColumn) for n in _raw_expr_nodes(stored))
    assert df2.array("x").tolist()[:5] == [0.0, 0.0, 0.0, 0.0, 8.0]  # max(2i, i) gated by a>3


def test_frame_column_compute_evaluates_through_top_level_unwrap(store):
    # A bare FrameColumn evaluated directly is unwrapped by evaluate()/_iter_leaves.
    df = store.edit()
    assert df["a"].compute().tolist() == list(range(10))


# ---- composition is unchanged: a FrameColumn still builds transforms --------


def test_frame_column_still_composes(store):
    df = store.edit()
    out = df.with_columns(x=df["a"] * 2 + df["b"]).recarray()
    assert out["x"].tolist() == [3 * i for i in range(10)]
