"""Tests for eager elementwise operators on a reader/view column.

A ColumnView is an eager read surface: the operators and NumPy ufuncs it inherits
materialize the selected rows and return a plain ndarray, while reductions stay
scalar terminals and the deferred/expression world remains the frame's.
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import col


@pytest.fixture()
def store(tmp_path):
    ds = colstore.store(
        {"a": np.arange(10, dtype=np.float64), "b": np.arange(10, dtype=np.int64)},
        tmp_path / "t.cstore",
        show_progress=False,
    )
    yield ds
    ds.close()


def test_arithmetic_is_eager_ndarray(store):
    out = store["a"] * 2
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [2 * i for i in range(10)]


def test_reflected_and_unary(store):
    assert (2 * store["a"]).tolist() == [2 * i for i in range(10)]
    assert (-store["a"]).tolist() == [-i for i in range(10)]
    assert (store["a"] + 1).tolist() == [i + 1 for i in range(10)]


def test_two_column_views_combine(store):
    out = store["a"] + store["b"]
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [2 * i for i in range(10)]


def test_comparison_returns_bool_array(store):
    out = store["a"] > 5
    assert out.dtype == bool
    assert out.tolist() == [i > 5 for i in range(10)]


def test_direct_ufunc_call_is_eager(store):
    out = np.sqrt(store["a"])
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, np.sqrt(np.arange(10.0)))


def test_operators_respect_row_selection(store):
    assert (store[2:6, "a"] * 10).tolist() == [20, 30, 40, 50]
    # a column-predicate selection composes too
    hot = store[col("a") >= 5, "a"]
    assert (hot + 1).tolist() == [6, 7, 8, 9, 10]


def test_reductions_stay_scalar_terminals(store):
    # adding operators must not turn np.sum / .sum into arrays
    assert np.sum(store["a"]) == 45.0
    assert store["a"].sum() == 45.0
    assert np.mean(store["a"]) == 4.5
    assert isinstance(store["a"].mean(), (float, np.floating))


def test_array_interface_unchanged(store):
    # np.asarray still materializes; np.sort / np.concatenate via __array__ still work
    assert np.asarray(store["a"]).tolist() == list(range(10))
    np.testing.assert_array_equal(np.sort(store["a"]), np.arange(10.0))
    assert np.concatenate([store["a"], store["a"]]).shape == (20,)
