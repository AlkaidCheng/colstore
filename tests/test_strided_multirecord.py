"""Tests for the native strided multi-record range gather.

Kernel contract: ``gather_multirecord_strided`` must equal NumPy slicing of
the full column for every supported dtype, both step directions, irregular
record sizes, and every prefetch setting -- the row stream is synthesized
arithmetically, so there is no index array whose contiguity or dtype could
mask a bug. Reader routing: multi-record slices with ``step != 1`` and a
native dtype engage the strided kernel; non-native dtypes keep the
``np.arange`` + fancy-path fallback; results match the fancy path the route
replaces on identical selectors.
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import _gather
from colstore import reader as reader_mod
from colstore.kernels import cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")


def _layout(n_rows_per_record, dtype, col_prefix_rows=0, seed: int = 0):
    """Build a fake single-column multi-record byte layout.

    ``col_prefix_rows`` simulates preceding columns by reserving that many
    bytes per row ahead of the column (prefix = col_prefix_rows bytes/row).
    With the default of 0 the column starts each record body.
    """
    rng = np.random.default_rng(seed)
    itemsize = np.dtype(dtype).itemsize
    nrr = np.asarray(n_rows_per_record, dtype=np.int64)
    n_records = nrr.shape[0]
    rsr = np.zeros(n_records + 1, dtype=np.int64)
    rsr[1:] = np.cumsum(nrr)
    body_sizes = nrr * (itemsize + col_prefix_rows)
    rsb = np.zeros(n_records, dtype=np.int64)
    rsb[1:] = np.cumsum(body_sizes)[:-1]
    total = int(rsr[-1])
    if np.issubdtype(np.dtype(dtype), np.floating):
        column = rng.standard_normal(total).astype(dtype)
    else:
        column = rng.integers(-100, 100, total).astype(dtype)
    buf = np.zeros(int(body_sizes.sum()), dtype=np.uint8)
    for r in range(n_records):
        off = int(rsb[r]) + col_prefix_rows * int(nrr[r])
        rows = column[int(rsr[r]) : int(rsr[r + 1])]
        buf[off : off + rows.nbytes] = rows.view(np.uint8)
    return buf, column, rsr, rsb, nrr, col_prefix_rows


def _slice_rows(total: int):
    """Slice triples (already resolved, as the reader produces them)."""
    return [
        (0, total, 2),
        (1, total, 3),
        (5, total - 5, 7),
        (0, total, total + 10),  # single element via huge step
        (3, 4, 5),  # single element via tight range
        (0, 0, 2),  # empty
        (total - 1, -1, -1),  # full reverse
        (total - 1, -1, -3),
        (total - 2, 4, -7),
        (total // 2, total // 2, -2),  # empty, negative
    ]


def _expected(column, start, stop, step):
    return column[start : (stop if stop >= 0 else None) : step]


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int16, np.int8])
@pytest.mark.parametrize("thread_cap", [1, 4])
def test_strided_kernel_matches_numpy_slicing(dtype, thread_cap):
    buf, column, rsr, rsb, nrr, prefix = _layout([3, 7, 1, 64, 10, 4, 25], dtype, seed=1)
    total = int(rsr[-1])
    for start, stop, step in _slice_rows(total):
        n = len(range(start, stop, step))
        output = np.empty(n, dtype=dtype)
        _gather.gather_multirecord_strided(
            buf, output, start, stop, step, rsr, rsb, nrr, prefix, thread_cap, 0
        )
        assert np.array_equal(output, _expected(column, start, stop, step)), (start, stop, step)


@pytest.mark.parametrize("prefetch", [0, 8, 128])
def test_strided_kernel_prefetch_invariance(prefetch):
    buf, column, rsr, rsb, nrr, prefix = _layout([50] * 16, np.float64, seed=2)
    total = int(rsr[-1])
    for start, stop, step in _slice_rows(total):
        n = len(range(start, stop, step))
        output = np.empty(n, dtype=np.float64)
        _gather.gather_multirecord_strided(
            buf, output, start, stop, step, rsr, rsb, nrr, prefix, 2, prefetch
        )
        assert np.array_equal(output, _expected(column, start, stop, step)), (
            start,
            stop,
            step,
            prefetch,
        )


def test_strided_kernel_column_prefix_addressing():
    # Non-zero column prefix: the column sits behind 3 bytes/row of earlier
    # columns, so record-base arithmetic must include the prefix term.
    buf, column, rsr, rsb, nrr, prefix = _layout([9, 2, 31, 5], np.int32, col_prefix_rows=3)
    total = int(rsr[-1])
    output = np.empty(len(range(1, total, 4)), dtype=np.int32)
    _gather.gather_multirecord_strided(buf, output, 1, total, 4, rsr, rsb, nrr, prefix, 1, 8)
    assert np.array_equal(output, column[1:total:4])


def test_strided_kernel_step_crossing_many_records_both_directions():
    # Step larger than every record: each element lands in a new record, so
    # the cursor-advance loop (not the steady state) carries the walk.
    buf, column, rsr, rsb, nrr, prefix = _layout([4] * 200, np.float64, seed=3)
    total = int(rsr[-1])
    for start, stop, step in [(0, total, 13), (total - 1, -1, -13)]:
        n = len(range(start, stop, step))
        output = np.empty(n, dtype=np.float64)
        _gather.gather_multirecord_strided(
            buf, output, start, stop, step, rsr, rsb, nrr, prefix, 2, 8
        )
        assert np.array_equal(output, _expected(column, start, stop, step))


def test_strided_kernel_validates_inputs():
    buf, _, rsr, rsb, nrr, prefix = _layout([10] * 4, np.float64)
    output = np.empty(5, dtype=np.float64)
    with pytest.raises(ValueError, match="step"):
        _gather.gather_multirecord_strided(buf, output, 0, 10, 0, rsr, rsb, nrr, prefix)
    with pytest.raises(ValueError, match="output length"):
        _gather.gather_multirecord_strided(buf, output, 0, 40, 2, rsr, rsb, nrr, prefix)
    with pytest.raises(TypeError, match="int64"):
        _gather.gather_multirecord_strided(
            buf, np.empty(20, dtype=np.float64), 0, 40, 2, rsr.astype(np.int32), rsb, nrr, prefix
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        strided_rsr = np.repeat(rsr, 2)[::2]
        _gather.gather_multirecord_strided(
            buf, np.empty(20, dtype=np.float64), 0, 40, 2, strided_rsr, rsb, nrr, prefix
        )


# ---- Reader routing ------------------------------------------------------


@pytest.fixture()
def multi_record_store(tmp_path):
    rng = np.random.default_rng(11)
    rows_per_record = [137, 64, 1, 350, 99, 200, 13, 470, 5, 261]
    total = sum(rows_per_record)
    full = {
        "f8": rng.standard_normal(total),
        "f4": rng.standard_normal(total).astype(np.float32),
        "i2": rng.integers(-(2**14), 2**14, total).astype(np.int16),
    }
    path = tmp_path / "strided.cstore"
    offset = 0
    with colstore.create(path) as writer:
        for rows in rows_per_record:
            writer.write({k: v[offset : offset + rows] for k, v in full.items()})
            offset += rows
    return path, full, total


STEPS = [2, 3, 10, -1, -2, -7, 1000, -1000]


@pytest.mark.parametrize("step", STEPS)
def test_reader_strided_slice_matches_ground_truth(multi_record_store, step):
    path, full, total = multi_record_store
    dataset = colstore.open(path)
    try:
        for name, values in full.items():
            result = dataset[::step, name].array()
            assert result.dtype == values.dtype
            assert np.array_equal(result, values[::step]), (name, step)
        offset_result = dataset[5 : total - 5 : step, "f8"].array()
        assert np.array_equal(offset_result, full["f8"][5 : total - 5 : step])
    finally:
        dataset.close()


def test_reader_strided_slice_matches_fancy_path(multi_record_store):
    # The route this kernel replaces: explicit arange selector through the
    # fancy path. Identical selectors must produce identical results.
    path, _full, total = multi_record_store
    dataset = colstore.open(path)
    try:
        for step in (4, -4):
            indices = np.arange(*slice(None, None, step).indices(total), dtype=np.int64)
            assert np.array_equal(
                dataset[::step, "f4"].array(), dataset[indices, "f4"].array()
            ), step
    finally:
        dataset.close()


def test_reader_strided_multi_column_dict(multi_record_store):
    path, full, _ = multi_record_store
    dataset = colstore.open(path)
    try:
        result = dataset[::3, ["f8", "i2"]].dict()
        assert np.array_equal(result["f8"], full["f8"][::3])
        assert np.array_equal(result["i2"], full["i2"][::3])
    finally:
        dataset.close()


def test_reader_routes_strided_slices_to_kernel(multi_record_store, monkeypatch):
    path, _, _ = multi_record_store
    calls = []
    original = _gather.gather_multirecord_strided

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(_gather, "gather_multirecord_strided", spy)
    dataset = colstore.open(path)
    try:
        dataset[::2, "f8"].array()
        assert len(calls) == 1
        dataset[100:1000, "f8"].array()  # unit step: contiguous route, not the kernel
        assert len(calls) == 1
        dataset[::-1, "i2"].array()
        assert len(calls) == 2
    finally:
        dataset.close()


def test_reader_non_native_dtype_falls_back(multi_record_store, monkeypatch):
    # Forcing the native check false must bypass the strided kernel (raw
    # typed loads cannot byteswap) and still return correct, native-order
    # values via the arange + fancy fallback.
    path, full, _ = multi_record_store
    calls = []
    original = _gather.gather_multirecord_strided

    def spy(*args, **kwargs):  # pragma: no cover - must not run
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(_gather, "gather_multirecord_strided", spy)
    monkeypatch.setattr(reader_mod, "_dtype_is_native", lambda dtype: False)
    dataset = colstore.open(path)
    try:
        result = dataset[::5, "f8"].array()
    finally:
        dataset.close()
    monkeypatch.undo()
    assert not calls
    assert result.dtype == np.dtype(np.float64).newbyteorder("=")
    assert np.array_equal(result, full["f8"][::5])


def test_reader_strided_misaligned_columns(tmp_path):
    # Odd-length int8 leading column puts every later column at odd byte
    # addresses inside each record body; the kernel's loads must be
    # alignment-safe (invariant pinned semantically here, UBSan-verified in
    # the compile harness).
    rng = np.random.default_rng(13)
    n_records, rows = 12, 7
    total = n_records * rows
    full = {
        "pad": rng.integers(-100, 100, total).astype(np.int8),
        "f8": rng.standard_normal(total),
        "f4": rng.standard_normal(total).astype(np.float32),
    }
    path = tmp_path / "mis_strided.cstore"
    with colstore.create(path) as writer:
        for r in range(n_records):
            writer.write({k: v[r * rows : (r + 1) * rows] for k, v in full.items()})
    dataset = colstore.open(path)
    try:
        for step in (2, 3, -1, -5):
            for name in ("f8", "f4"):
                assert np.array_equal(dataset[::step, name].array(), full[name][::step]), (
                    name,
                    step,
                )
    finally:
        dataset.close()


@pytest.mark.parametrize("step", [2, -3])
def test_reader_strided_single_record_store_unaffected(tmp_path, step):
    # Single-record stores keep their existing numpy strided-copy path; the
    # multi-record kernel must not be invoked.
    frame = {"a": np.arange(500, dtype=np.float64)}
    path = tmp_path / "single.cstore"
    with colstore.create(path) as writer:
        writer.write(frame)
    dataset = colstore.open(path)
    try:
        assert np.array_equal(dataset[::step, "a"].array(), frame["a"][::step])
    finally:
        dataset.close()
