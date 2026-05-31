"""Tests for ColStore indexing semantics across every documented pattern."""

from __future__ import annotations

import numpy as np
import pytest


def test_single_int_row_indexing(medium_store, medium_frame):
    one_row = medium_store[42].to_dict()
    assert one_row["a"].shape == (1,)
    assert np.isclose(one_row["a"][0], medium_frame["a"].iloc[42])


def test_slice_row_indexing(medium_store, medium_frame):
    rows = medium_store[100:1100].to_dict()
    assert rows["a"].shape == (1000,)
    assert np.allclose(rows["a"], medium_frame["a"].iloc[100:1100].to_numpy())


def test_negative_step_not_supported_but_does_not_crash(medium_store):
    """NumPy's slice semantics propagate through; reversed slice is allowed."""
    result = medium_store[200:100:-1, "a"].to_array()
    assert result.shape == (100,)


def test_single_column_string_selects_all_rows(medium_store, medium_frame):
    full_column = medium_store["b"].to_array()
    assert np.allclose(full_column, medium_frame["b"].to_numpy())


def test_list_of_columns_selects_all_rows(medium_store, medium_frame):
    columns = medium_store[["a", "c"]].to_dict()
    assert np.allclose(columns["a"], medium_frame["a"].to_numpy())
    assert np.array_equal(columns["c"], medium_frame["c"].to_numpy())


def test_slice_with_single_column(medium_store, medium_frame):
    result = medium_store[1000:2000, "a"].to_array()
    assert result.shape == (1000,)
    assert np.allclose(result, medium_frame["a"].iloc[1000:2000].to_numpy())


def test_slice_with_multi_columns(medium_store, medium_frame):
    result = medium_store[1000:1010, ["a", "b"]].to_dict()
    assert np.allclose(result["a"], medium_frame["a"].iloc[1000:1010].to_numpy())
    assert np.allclose(result["b"], medium_frame["b"].iloc[1000:1010].to_numpy())


def test_int_array_with_multi_columns(medium_store, medium_frame):
    indices = np.array([1, 5, 99, 0, 49999])
    result = medium_store[indices, ["a", "c"]].to_dict()
    assert np.allclose(result["a"], medium_frame["a"].iloc[indices].to_numpy())
    assert np.array_equal(result["c"], medium_frame["c"].iloc[indices].to_numpy())


def test_int_list_with_multi_columns(medium_store, medium_frame):
    indices = [1, 5, 99, 0, 49999]
    result = medium_store[indices, ["a", "c"]].to_dict()
    assert np.allclose(result["a"], medium_frame["a"].iloc[indices].to_numpy())


def test_boolean_mask_with_single_column(medium_store, medium_frame):
    rng = np.random.default_rng(0)
    mask = rng.random(medium_store.n_rows) < 0.05
    result = medium_store[mask, "a"].to_array()
    assert result.shape == (mask.sum(),)
    assert np.allclose(result, medium_frame["a"].to_numpy()[mask])


def test_boolean_mask_with_multi_columns(medium_store, medium_frame):
    rng = np.random.default_rng(1)
    mask = rng.random(medium_store.n_rows) < 0.02
    result = medium_store[mask, ["a", "b"]].to_dict()
    for column_name in ["a", "b"]:
        expected = medium_frame[column_name].to_numpy()[mask]
        assert np.allclose(result[column_name], expected)


def test_unsorted_indices_preserve_caller_order(medium_store, medium_frame):
    indices = np.array([100, 5, 99, 0, 49_999, 5])
    result = medium_store[indices, "a"].to_array()
    expected = medium_frame["a"].iloc[indices].to_numpy()
    assert np.allclose(result, expected)


def test_duplicate_indices_repeated_in_output(medium_store, medium_frame):
    indices = np.array([42, 42, 42, 100, 100])
    result = medium_store[indices, "c"].to_array()
    assert result.shape == (5,)
    assert np.array_equal(result, medium_frame["c"].iloc[indices].to_numpy())


def test_unknown_column_raises_keyerror(small_store):
    with pytest.raises(KeyError, match="missing"):
        small_store["missing"]


def test_unknown_column_in_list_raises_keyerror(small_store):
    with pytest.raises(KeyError):
        small_store[["price", "missing"]]


def test_oversized_tuple_raises_indexerror(small_store):
    with pytest.raises(IndexError, match="2 elements"):
        small_store[0, 1, 2]


def test_invalid_column_selector_type_raises_indexerror(small_store):
    with pytest.raises(IndexError, match="string"):
        small_store[100:200, 42]
