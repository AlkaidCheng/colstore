"""Tests for the bin-reuse multi-column gather route.

Two layers. Kernel contracts: ``gather_multirecord_bins`` must fill bins
identical to a searchsorted reference while producing the same column output
as ``gather_multirecord``, and ``gather_multirecord_withbins`` fed those bins
must match ``gather_multirecord`` for every supported dtype. Reader routing:
multi-column unsorted fancy reads must engage the route (and only then),
produce results identical to per-column reads, and preserve requested column
order -- including when non-native columns are mixed in and fall back.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import colstore
from colstore import _gather, config
from colstore.kernels import cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")


def _layout(n_records: int, rows: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    nrr = np.full(n_records, rows, dtype=np.int64)
    rsr = np.zeros(n_records + 1, dtype=np.int64)
    rsr[1:] = np.cumsum(nrr)
    rsb = np.arange(n_records, dtype=np.int64) * (rows * 8)
    buf = rng.standard_normal(n_records * rows).view(np.uint8)
    return buf, rsr, rsb, nrr


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8])
def test_bins_kernel_matches_searchsorted_and_plain_kernel(dtype):
    n_records, rows = 64, 100
    rng = np.random.default_rng(1)
    nrr = np.full(n_records, rows, dtype=np.int64)
    rsr = np.zeros(n_records + 1, dtype=np.int64)
    rsr[1:] = np.cumsum(nrr)
    itemsize = np.dtype(dtype).itemsize
    rsb = np.arange(n_records, dtype=np.int64) * (rows * itemsize)
    source = rng.integers(-100, 100, n_records * rows).astype(dtype)
    buf = source.view(np.uint8)
    indices = rng.integers(0, n_records * rows, size=5_000).astype(np.int64)

    out_plain = np.empty(indices.size, dtype=dtype)
    _gather.gather_multirecord(buf, indices, out_plain, rsr, rsb, nrr, 0, 2, 0)

    out_bins = np.empty(indices.size, dtype=dtype)
    bins = np.empty(indices.size, dtype=np.int32)
    _gather.gather_multirecord_bins(buf, indices, out_bins, bins, rsr, rsb, nrr, 0, 2, 0)
    assert np.array_equal(out_bins, out_plain)
    expected_bins = (np.searchsorted(rsr, indices, side="right") - 1).astype(np.int32)
    assert np.array_equal(bins, expected_bins)

    out_with = np.empty(indices.size, dtype=dtype)
    _gather.gather_multirecord_withbins(buf, indices, out_with, bins, rsr, rsb, nrr, 0, 2, 0)
    assert np.array_equal(out_with, out_plain)


def test_bins_kernels_validate_bins_dtype_and_length():
    buf, rsr, rsb, nrr = _layout(4, 10)
    indices = np.array([0, 5, 39], dtype=np.int64)
    out = np.empty(3)
    with pytest.raises(TypeError, match="bins must be int32"):
        _gather.gather_multirecord_bins(
            buf, indices, out, np.empty(3, dtype=np.int64), rsr, rsb, nrr, 0
        )
    with pytest.raises(ValueError, match="lengths"):
        _gather.gather_multirecord_withbins(
            buf, indices, out, np.empty(2, dtype=np.int32), rsr, rsb, nrr, 0
        )


@pytest.fixture()
def mixed_store(tmp_path):
    rng = np.random.default_rng(7)
    full = {
        "f8": rng.standard_normal(40_000),
        "f4": rng.standard_normal(40_000).astype(np.float32),
        "i4": rng.integers(-(2**20), 2**20, 40_000).astype(np.int32),
        "i2": rng.integers(-1000, 1000, 40_000).astype(np.int16),
    }
    path = tmp_path / "m.cstore"
    # Irregular record sizes (one record split unevenly): these tests pin the
    # GENERIC bins route, which uniform-record files no longer take (they
    # route to the arithmetic-binning kernels, covered by
    # tests/test_uniform_multirecord.py).
    boundaries = [0, 500, *range(800, 40_001, 800)]
    with colstore.create(path) as writer:
        for lo, hi in itertools.pairwise(boundaries):
            writer.write({name: col[lo:hi] for name, col in full.items()})
    return path, full


def _spy_withbins(monkeypatch):
    calls: list[int] = []
    real = _gather.gather_multirecord_withbins

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(_gather, "gather_multirecord_withbins", spy)
    return calls


def test_multicolumn_read_routes_and_matches_per_column(mixed_store, monkeypatch):
    path, full = mixed_store
    calls = _spy_withbins(monkeypatch)
    dataset = colstore.open(path)
    indices = np.random.default_rng(9).integers(0, 40_000, size=15_000).astype(np.int64)

    table = dataset[indices, list(full)].dict()
    assert len(calls) == len(full) - 1  # first column binned, rest reused
    assert list(table) == list(full)  # requested order preserved
    for name, column in full.items():
        assert np.array_equal(table[name], column[indices]), name
        assert np.array_equal(
            table[name], dataset[indices, name].array()
        ), f"per-column mismatch: {name}"
    dataset.close()


def test_duplicates_and_reversed_indices(mixed_store, monkeypatch):
    path, full = mixed_store
    dataset = colstore.open(path)
    indices = np.array([39_999, 0, 5, 5, 39_999, 17, 0], dtype=np.int64)
    table = dataset[indices, ["f8", "i4"]].dict()
    assert np.array_equal(table["f8"], full["f8"][indices])
    assert np.array_equal(table["i4"], full["i4"][indices])
    dataset.close()


def test_route_not_taken_for_sorted_single_column_or_slice(mixed_store, monkeypatch):
    path, full = mixed_store
    calls = _spy_withbins(monkeypatch)
    dataset = colstore.open(path)
    indices = np.random.default_rng(2).integers(0, 40_000, size=5_000).astype(np.int64)

    sorted_table = dataset[np.sort(indices), ["f8", "f4"]].dict()
    assert np.array_equal(sorted_table["f8"], full["f8"][np.sort(indices)])
    single = dataset[indices, "f8"].array()
    assert np.array_equal(single, full["f8"][indices])
    sliced = dataset[100:900, ["f8", "f4"]].dict()
    assert np.array_equal(sliced["f8"], full["f8"][100:900])
    assert calls == []  # none of the above may engage the bins kernels
    dataset.close()


def test_route_respects_thread_cap_config(mixed_store):
    path, full = mixed_store
    original = config.get_gather_thread_cap()
    try:
        config.set_gather_thread_cap(1)
        dataset = colstore.open(path)
        indices = np.random.default_rng(4).integers(0, 40_000, size=8_000).astype(np.int64)
        table = dataset[indices, ["f8", "f4", "i4"]].dict()
        for name in ("f8", "f4", "i4"):
            assert np.array_equal(table[name], full[name][indices])
        dataset.close()
    finally:
        config.set_gather_thread_cap(original)


def test_single_record_store_not_routed(tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    full = {"a": rng.standard_normal(10_000), "b": rng.standard_normal(10_000)}
    path = tmp_path / "single.cstore"
    colstore.store(full, path, show_progress=False)  # one-record file
    calls = _spy_withbins(monkeypatch)
    dataset = colstore.open(path)
    indices = rng.integers(0, 10_000, size=3_000).astype(np.int64)
    table = dataset[indices, ["a", "b"]].dict()
    assert calls == []
    assert np.array_equal(table["a"], full["a"][indices])
    dataset.close()
