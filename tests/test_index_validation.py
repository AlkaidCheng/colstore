"""Tests for backend-agnostic row-index validation.

Every integer selector is bounds-checked and negatives are folded before the
gather backend runs, so the C++/Numba kernels obey the same contract as NumPy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from colstore import ColStore
from colstore.kernels import cpp_available, numba_available

_BACKENDS = ["numpy"]
if cpp_available():
    _BACKENDS.append("cpp")
if numba_available():
    _BACKENDS.append("numba")


@pytest.fixture
def store(tmp_path, request):
    frame = pd.DataFrame(
        {"a": np.arange(100, dtype=np.float64), "c": np.arange(100, dtype=np.int64)}
    )
    opened = ColStore.from_dataframe(
        frame, tmp_path / "idx.cstore", show_progress=False, backend=request.param
    )
    yield opened
    opened.close()


@pytest.mark.parametrize("store", _BACKENDS, indirect=True)
def test_out_of_bounds_fancy_index_raises(store):
    with pytest.raises(IndexError, match="out of bounds"):
        store[np.array([100]), "a"].to_array()


@pytest.mark.parametrize("store", _BACKENDS, indirect=True)
def test_negative_fancy_index_folds(store):
    assert store[np.array([-1, -2]), "c"].to_array().tolist() == [99, 98]


@pytest.mark.parametrize("store", _BACKENDS, indirect=True)
def test_out_of_bounds_scalar_raises(store):
    with pytest.raises(IndexError, match="out of bounds"):
        store[100, "a"].to_array()


@pytest.mark.parametrize("store", _BACKENDS, indirect=True)
def test_negative_scalar_folds(store):
    assert store[-1, "a"].to_array().tolist() == [99.0]


@pytest.mark.parametrize("store", _BACKENDS, indirect=True)
def test_float_index_array_raises(store):
    with pytest.raises(IndexError, match="integer or boolean"):
        store[np.array([1.7, 2.9]), "a"].to_array()


@pytest.mark.parametrize("store", _BACKENDS, indirect=True)
def test_zero_dim_array_is_scalar_shape(store):
    result = store[np.array(5), "a"].to_array()
    assert result.shape == (1,)
    assert result.tolist() == [5.0]


@pytest.mark.parametrize("store", _BACKENDS, indirect=True)
def test_bad_mask_length_raises(store):
    with pytest.raises(IndexError, match="mask length"):
        store[np.zeros(50, dtype=bool), "a"].to_array()
