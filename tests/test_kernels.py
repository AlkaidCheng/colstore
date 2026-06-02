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


# ---- Size-based kernel: new dtype coverage --------------------------------


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype("|S1"),  # 1-byte fixed bytes -> kernel size 1
        np.dtype("|S2"),  # 2-byte fixed bytes -> kernel size 2
        np.dtype("|S4"),  # 4-byte fixed bytes -> kernel size 4
        np.dtype("|S8"),  # 8-byte fixed bytes -> kernel size 8
        np.dtype("|U1"),  # 1-codepoint unicode = 4 bytes -> size 4
        np.dtype("|U2"),  # 2-codepoint unicode = 8 bytes -> size 8
        np.dtype("datetime64[D]"),  # 8-byte int64 under the hood
        np.dtype("timedelta64[s]"),  # 8-byte int64 under the hood
    ],
)
def test_kernel_handles_fixed_width_non_numeric_dtypes(dtype):
    """The size-dispatched kernel handles any dtype whose itemsize is 1/2/4/8.

    Prior to PR 1 the kernel was templated by NumPy dtype kind ('f'/'i'/'u'/'b'),
    so fixed-width strings, datetime64, and timedelta64 fell through to NumPy
    even though their underlying layout is a fixed-size POD copy. The byte-
    offset kernel doesn't care about kind -- only itemsize -- so these now
    work natively at the kernel level.
    """
    from colstore import _gather  # type: ignore[attr-defined]

    if dtype.kind in ("S", "U"):
        values = np.array(["a", "bc", "de", "fghi"][:4], dtype=dtype)
        # Pad / truncate to dtype's natural length via the cast.
        source = np.tile(values, 4)
    elif dtype.kind == "M":
        source = np.array(["2020-01-01", "2021-06-15", "2022-03-30", "2023-12-31"], dtype=dtype)
        source = np.tile(source, 4)
    else:  # timedelta64
        source = np.arange(16, dtype=dtype)

    indices = np.array([15, 0, 8, 1, 14, 2], dtype=np.int64)
    output = np.empty(indices.shape[0], dtype=dtype)
    _gather.gather(source, indices, output, 0)
    assert np.array_equal(output, source[indices])


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
def test_kernel_rejects_unsupported_itemsize():
    """Itemsizes outside {1,2,4,8} raise a clean error from the kernel."""
    from colstore import _gather  # type: ignore[attr-defined]

    # |S5 is a 5-byte fixed-width string -- size 5 is not in the supported set.
    source = np.array(["hello", "world"], dtype="|S5")
    indices = np.array([1, 0], dtype=np.int64)
    output = np.empty(2, dtype="|S5")
    with pytest.raises(TypeError, match="element size"):
        _gather.gather(source, indices, output, 0)


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
def test_gather_bytes_entry_point_equivalent_to_gather():
    """``gather_bytes`` with offsets = indices * itemsize matches ``gather``.

    Locks in the relationship the multi-record reader (PR 2) relies on: when
    no record-header arithmetic is needed, byte-offset gather degenerates
    cleanly to element-indexed gather.
    """
    from colstore import _gather  # type: ignore[attr-defined]

    rng = np.random.default_rng(0)
    source = rng.standard_normal(10_000).astype(np.float32)
    indices = rng.permutation(10_000)[:500].astype(np.int64)
    byte_offsets = indices * source.dtype.itemsize

    out_element = np.empty(500, dtype=np.float32)
    out_bytes = np.empty(500, dtype=np.float32)
    _gather.gather(source, indices, out_element, 4)
    _gather.gather_bytes(source, byte_offsets, out_bytes, 4)
    assert np.array_equal(out_element, out_bytes)


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
def test_gather_bytes_with_offset_origin():
    """``gather_bytes`` indices can include any per-element byte offset.

    This is the multi-record case in miniature: an extra fixed offset added to
    every byte address (simulating "skip the record header"). Verifies the
    kernel does not silently assume offsets are multiples of itemsize.
    """
    from colstore import _gather  # type: ignore[attr-defined]

    # Build a "file" of two records sharing one float64 column.
    rec0 = np.arange(10, dtype=np.float64)
    rec1 = np.arange(100, 110, dtype=np.float64)
    HEADER = 32  # bytes of (fake) record header before each record's data
    blob = bytearray()
    blob.extend(b"\x00" * HEADER)
    blob.extend(rec0.tobytes())
    blob.extend(b"\x00" * HEADER)
    blob.extend(rec1.tobytes())
    source = np.frombuffer(blob, dtype=np.uint8)

    # Read elements 0 and 5 from rec0, then 2 and 9 from rec1.
    rec0_offset = HEADER
    rec1_offset = HEADER + rec0.nbytes + HEADER
    itemsize = 8
    byte_offsets = np.array(
        [
            rec0_offset + 0 * itemsize,
            rec0_offset + 5 * itemsize,
            rec1_offset + 2 * itemsize,
            rec1_offset + 9 * itemsize,
        ],
        dtype=np.int64,
    )
    output = np.empty(4, dtype=np.float64)
    _gather.gather_bytes(source, byte_offsets, output, 0)
    assert np.array_equal(output, np.array([0.0, 5.0, 102.0, 109.0]))
