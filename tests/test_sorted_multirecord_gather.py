"""Tests for the native sorted multi-record fancy gather.

Kernel contract: ``gather_multirecord_sorted`` must equal the unsorted fused
kernel on the same (sorted) inputs for every supported dtype, including
duplicates and record-boundary indices. Reader routing: sorted native fancy
reads engage the walk kernel; unsorted reads and non-native dtypes do not;
results match ground truth and the boundary-partition pipeline it replaces.
The walk requires non-decreasing indices -- the reader's existing sortedness
check gates the route, and these tests never feed it unsorted input.
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import _gather, config
from colstore import reader as reader_mod
from colstore.kernels import cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")


def _layout(n_records: int, rows: int, dtype, seed: int = 0):
    rng = np.random.default_rng(seed)
    itemsize = np.dtype(dtype).itemsize
    nrr = np.full(n_records, rows, dtype=np.int64)
    rsr = np.zeros(n_records + 1, dtype=np.int64)
    rsr[1:] = np.cumsum(nrr)
    rsb = np.arange(n_records, dtype=np.int64) * (rows * itemsize)
    source = rng.integers(-100, 100, n_records * rows).astype(dtype)
    return source.view(np.uint8), source, rsr, rsb, nrr


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8])
@pytest.mark.parametrize("thread_cap", [1, 4])
def test_sorted_kernel_matches_unsorted_kernel(dtype, thread_cap):
    n_records, rows = 64, 100
    buf, source, rsr, rsb, nrr = _layout(n_records, rows, dtype)
    rng = np.random.default_rng(1)
    indices = np.sort(rng.integers(0, n_records * rows, size=5_000).astype(np.int64))

    out_sorted = np.empty(indices.size, dtype=dtype)
    _gather.gather_multirecord_sorted(buf, indices, out_sorted, rsr, rsb, nrr, 0, thread_cap, 0)
    out_reference = np.empty(indices.size, dtype=dtype)
    _gather.gather_multirecord(buf, indices, out_reference, rsr, rsb, nrr, 0, thread_cap, 0)
    assert np.array_equal(out_sorted, out_reference)
    assert np.array_equal(out_sorted, source[indices])


@pytest.mark.parametrize("prefetch", [0, 8, 128])
def test_sorted_kernel_edge_index_patterns(prefetch):
    n_records, rows = 16, 50
    buf, source, rsr, rsb, nrr = _layout(n_records, rows, np.float64, seed=2)
    total = n_records * rows
    patterns = {
        "duplicates_and_boundaries": np.sort(
            np.array([0, 0, rows - 1, rows, rows, total - 1, total - 1], dtype=np.int64)
        ),
        "all_in_one_record": np.sort(
            np.random.default_rng(3).integers(3 * rows, 4 * rows, size=200).astype(np.int64)
        ),
        "one_per_record": (np.arange(n_records, dtype=np.int64) * rows + rows // 2),
        "every_row": np.arange(total, dtype=np.int64),
        "fewer_than_records": np.sort(
            np.random.default_rng(4).integers(0, total, size=5).astype(np.int64)
        ),
        "single_element": np.array([total // 2], dtype=np.int64),
    }
    for name, indices in patterns.items():
        output = np.empty(indices.size, dtype=np.float64)
        _gather.gather_multirecord_sorted(buf, indices, output, rsr, rsb, nrr, 0, 2, prefetch)
        assert np.array_equal(output, source[indices]), (name, prefetch)


def test_sorted_kernel_validates_inputs():
    buf, _, rsr, rsb, nrr = _layout(4, 10, np.float64)
    indices = np.array([0, 5, 39], dtype=np.int64)
    with pytest.raises(TypeError, match="int64"):
        _gather.gather_multirecord_sorted(
            buf, indices.astype(np.int32), np.empty(3), rsr, rsb, nrr, 0
        )
    with pytest.raises(ValueError, match="length"):
        _gather.gather_multirecord_sorted(buf, indices, np.empty(2), rsr, rsb, nrr, 0)


@pytest.fixture()
def mixed_store(tmp_path):
    rng = np.random.default_rng(7)
    total = 50_000
    full = {
        "f8": rng.standard_normal(total),
        "i4": rng.integers(-(2**20), 2**20, total).astype(np.int32),
        "pad": rng.integers(-9, 9, total).astype(np.int8),
    }
    path = tmp_path / "m.cstore"
    with colstore.create(path) as writer:
        for offset in range(0, total, 500):  # 100 records
            writer.write({k: v[offset : offset + 500] for k, v in full.items()})
    return path, full, total


def _spy_sorted_kernel(monkeypatch):
    calls: list[int] = []
    real = _gather.gather_multirecord_sorted

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(_gather, "gather_multirecord_sorted", spy)
    return calls


def test_sorted_reads_route_through_walk_kernel(mixed_store, monkeypatch):
    path, full, total = mixed_store
    calls = _spy_sorted_kernel(monkeypatch)
    dataset = colstore.open(path)
    indices = np.sort(np.random.default_rng(9).integers(0, total, size=20_000).astype(np.int64))
    for name in full:
        assert np.array_equal(dataset[indices, name].array(), full[name][indices]), name
    assert len(calls) == len(full)
    dataset.close()


def test_unsorted_reads_do_not_route(mixed_store, monkeypatch):
    path, full, total = mixed_store
    calls = _spy_sorted_kernel(monkeypatch)
    dataset = colstore.open(path)
    indices = np.random.default_rng(10).integers(0, total, size=5_000).astype(np.int64)
    assert np.array_equal(dataset[indices, "f8"].array(), full["f8"][indices])
    assert calls == []
    dataset.close()


def test_sorted_matches_partition_pipeline_it_replaces(mixed_store, monkeypatch):
    # The boundary-partition pipeline survives as the non-native fallback;
    # forcing it must give identical results to the walk kernel route.
    path, full, total = mixed_store
    indices = np.sort(np.random.default_rng(11).integers(0, total, size=15_000).astype(np.int64))
    dataset = colstore.open(path)
    via_kernel = {name: dataset[indices, name].array() for name in full}
    dataset.close()
    monkeypatch.setattr(reader_mod, "_dtype_is_native", lambda dtype: False)
    dataset = colstore.open(path)
    via_pipeline = {name: dataset[indices, name].array() for name in full}
    dataset.close()
    for name in full:
        assert np.array_equal(via_kernel[name], via_pipeline[name]), name
        assert np.array_equal(via_kernel[name], full[name][indices]), name


def test_sorted_read_on_misaligned_columns(tmp_path):
    # The walk kernel must use alignment-safe loads: an odd-length int8
    # column puts the f8 column at odd addresses (packed record bodies).
    rng = np.random.default_rng(12)
    n_records, rows = 10, 7
    total = n_records * rows
    full = {"pad": rng.integers(0, 9, total).astype(np.int8), "f8": rng.standard_normal(total)}
    path = tmp_path / "mis.cstore"
    with colstore.create(path) as writer:
        for r in range(n_records):
            writer.write({k: v[r * rows : (r + 1) * rows] for k, v in full.items()})
    dataset = colstore.open(path)
    indices = np.sort(rng.integers(0, total, size=40).astype(np.int64))
    assert np.array_equal(dataset[indices, "f8"].array(), full["f8"][indices])
    dataset.close()


def test_sorted_read_respects_thread_cap_config(mixed_store):
    path, full, total = mixed_store
    original = config.get_gather_thread_cap()
    try:
        for cap in (1, 8):
            config.set_gather_thread_cap(cap)
            dataset = colstore.open(path)
            indices = np.sort(
                np.random.default_rng(13).integers(0, total, size=10_000).astype(np.int64)
            )
            assert np.array_equal(dataset[indices, "f8"].array(), full["f8"][indices]), cap
            dataset.close()
    finally:
        config.set_gather_thread_cap(original)
