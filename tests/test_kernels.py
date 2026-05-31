"""Tests for the C++ gather kernel and the dispatcher."""

from __future__ import annotations

import numpy as np
import pytest

from colstore import cpp_available
from colstore.kernels import gather, max_threads

pytestmark = pytest.mark.skipif(
    not cpp_available(),
    reason="C++ gather extension not built; install with `pip install .`",
)


@pytest.mark.parametrize(
    "numpy_dtype",
    [
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ],
)
def test_cpp_gather_matches_numpy_fancy_index_for_every_dtype(numpy_dtype):
    rng = np.random.default_rng(0)
    if np.issubdtype(numpy_dtype, np.floating):
        source = rng.standard_normal(10_000).astype(numpy_dtype)
    elif np.issubdtype(numpy_dtype, np.signedinteger):
        max_value = np.iinfo(numpy_dtype).max // 2
        source = rng.integers(-max_value, max_value, size=10_000, dtype=numpy_dtype)
    else:
        max_value = np.iinfo(numpy_dtype).max // 2
        source = rng.integers(0, max_value, size=10_000, dtype=numpy_dtype)
    indices = np.sort(rng.choice(source.size, size=1_000, replace=False))
    cpp_result = gather(source, indices, source.dtype, backend="cpp")
    numpy_result = source[indices]
    assert cpp_result.dtype == source.dtype
    assert np.array_equal(cpp_result, numpy_result)


def test_cpp_gather_handles_unsorted_indices_with_duplicates():
    rng = np.random.default_rng(1)
    source = rng.standard_normal(5_000).astype(np.float64)
    indices = np.array([4_999, 0, 4_999, 100, 42, 100], dtype=np.int64)
    cpp_result = gather(source, indices, source.dtype, backend="cpp")
    assert np.array_equal(cpp_result, source[indices])


def test_cpp_gather_zero_length_input_returns_empty():
    source = np.zeros(100, dtype=np.float32)
    indices = np.empty(0, dtype=np.int64)
    result = gather(source, indices, source.dtype, backend="cpp")
    assert result.shape == (0,)
    assert result.dtype == np.float32


def test_cpp_gather_rejects_non_int64_indices():
    from colstore import _gather as cpp_module

    source = np.zeros(10, dtype=np.float32)
    indices_wrong = np.array([0, 1, 2], dtype=np.int32)
    output = np.empty(3, dtype=np.float32)
    with pytest.raises(TypeError, match="int64"):
        cpp_module.gather(source, indices_wrong, output)


def test_cpp_gather_rejects_dtype_mismatch():
    from colstore import _gather as cpp_module

    source = np.zeros(10, dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int64)
    output_wrong_dtype = np.empty(3, dtype=np.float64)
    with pytest.raises(TypeError, match="dtype"):
        cpp_module.gather(source, indices, output_wrong_dtype)


def test_max_threads_returns_positive_integer():
    threads = max_threads()
    assert isinstance(threads, int)
    assert threads >= 1


def test_numpy_backend_always_works():
    rng = np.random.default_rng(2)
    source = rng.standard_normal(1_000).astype(np.float32)
    indices = np.sort(rng.choice(1_000, size=100, replace=False))
    cpp_result = gather(source, indices, source.dtype, backend="cpp")
    numpy_result = gather(source, indices, source.dtype, backend="numpy")
    assert np.array_equal(cpp_result, numpy_result)
