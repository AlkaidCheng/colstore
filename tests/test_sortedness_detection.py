"""Tests for sampled-rejection sortedness detection.

Helper contract: ``_indices_are_sorted`` must agree with the full check
``bool(np.all(indices[1:] >= indices[:-1]))`` on every input -- sampling is
only ever used to reject sortedness, never to prove it, so an adversarial
nearly-sorted array whose only descent falls between probe positions must
still come back False via the full pass. Reader routing: the gate change is
behavior-preserving -- sorted selectors still reach the sorted walk kernel,
unsorted selectors the fused kernel (single column) and the bin-reuse route
(multi-column), above and below the sampling threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import _gather
from colstore import reader as reader_mod
from colstore.kernels import cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")

THRESHOLD = reader_mod._SORTEDNESS_SAMPLE_MIN_SIZE


def _full_check(indices: np.ndarray) -> bool:
    return bool(np.all(indices[1:] >= indices[:-1]))


def _sample_positions(n: int) -> np.ndarray:
    return (reader_mod._SORTEDNESS_SAMPLE_FRACTIONS * (n - 2)).astype(np.int64)


@pytest.mark.parametrize("n", [0, 1, 2, 100, THRESHOLD - 1, THRESHOLD, THRESHOLD * 4])
def test_helper_agrees_with_full_check_on_battery(n):
    rng = np.random.default_rng(n)
    candidates = [
        np.sort(rng.integers(0, 10**9, n).astype(np.int64)),  # sorted
        rng.integers(0, 10**9, n).astype(np.int64),  # random
        np.full(n, 7, dtype=np.int64),  # all-duplicate (non-decreasing)
        np.arange(n, dtype=np.int64)[::-1].copy(),  # descending
    ]
    for indices in candidates:
        assert reader_mod._indices_are_sorted(indices) == _full_check(indices)


def test_descent_between_probe_positions_is_still_caught():
    # Sorted array with exactly one adjacent descent placed away from every
    # sampled pair: the sampler must pass and the full check must decide.
    n = THRESHOLD * 2
    indices = np.arange(n, dtype=np.int64)
    probed = set(_sample_positions(n).tolist())
    pos = next(p for p in range(n // 3, n) if p not in probed and (p + 1) not in probed)
    indices[pos], indices[pos + 1] = indices[pos + 1], indices[pos]
    assert not _full_check(indices)
    assert reader_mod._indices_are_sorted(indices) is False


def test_descent_at_probe_positions_is_caught_by_sampling():
    n = THRESHOLD * 2
    indices = np.arange(n, dtype=np.int64)
    pos = int(_sample_positions(n)[7])
    indices[pos], indices[pos + 1] = indices[pos + 1], indices[pos]
    assert reader_mod._indices_are_sorted(indices) is False


def test_sorted_above_threshold_returns_true():
    n = THRESHOLD * 8
    indices = np.sort(np.random.default_rng(3).integers(0, 10**12, n).astype(np.int64))
    assert reader_mod._indices_are_sorted(indices) is True


# ---- Reader routing equivalence -----------------------------------------


@pytest.fixture()
def multi_record_store(tmp_path):
    rng = np.random.default_rng(21)
    # Irregular record sizes (first record split unevenly): these tests pin
    # the GENERIC kernels' routing, which uniform-record files no longer
    # take (they route to the arithmetic-binning kernels, covered by
    # tests/test_uniform_multirecord.py). Total 160_000 rows so selectors
    # can exceed THRESHOLD.
    rows_per_record = [1_500, 2_500] + [4_000] * 39
    total = sum(rows_per_record)
    full = {
        "f8": rng.standard_normal(total),
        "i4": rng.integers(-(2**20), 2**20, total).astype(np.int32),
    }
    path = tmp_path / "sortedness.cstore"
    offset = 0
    with colstore.create(path) as writer:
        for rows in rows_per_record:
            writer.write({k: v[offset : offset + rows] for k, v in full.items()})
            offset += rows
    return path, full, total


@pytest.mark.parametrize("size", [100, THRESHOLD * 2])
def test_routing_unchanged_for_sorted_and_unsorted(multi_record_store, monkeypatch, size):
    path, full, total = multi_record_store
    rng = np.random.default_rng(size)
    unsorted_idx = rng.integers(0, total, size).astype(np.int64)
    sorted_idx = np.sort(unsorted_idx)

    routes = []
    for name in ("gather_multirecord_sorted", "gather_multirecord"):
        original = getattr(_gather, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            routes.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(_gather, name, spy)

    dataset = colstore.open(path)
    try:
        assert np.array_equal(dataset[sorted_idx, "f8"].array(), full["f8"][sorted_idx])
        assert routes == ["gather_multirecord_sorted"]
        routes.clear()
        assert np.array_equal(dataset[unsorted_idx, "f8"].array(), full["f8"][unsorted_idx])
        assert routes == ["gather_multirecord"]
    finally:
        dataset.close()


@pytest.mark.parametrize("size", [1_000, THRESHOLD * 2])
def test_bin_reuse_gate_unchanged(multi_record_store, monkeypatch, size):
    path, full, total = multi_record_store
    rng = np.random.default_rng(size + 1)
    unsorted_idx = rng.integers(0, total, size).astype(np.int64)
    sorted_idx = np.sort(unsorted_idx)

    bins_calls = []
    original = _gather.gather_multirecord_bins

    def spy(*args, **kwargs):
        bins_calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_gather, "gather_multirecord_bins", spy)
    dataset = colstore.open(path)
    try:
        result = dataset[unsorted_idx, ["f8", "i4"]].dict()
        assert len(bins_calls) == 1  # unsorted multi-column: bin-reuse route
        assert np.array_equal(result["f8"], full["f8"][unsorted_idx])
        assert np.array_equal(result["i4"], full["i4"][unsorted_idx])
        result = dataset[sorted_idx, ["f8", "i4"]].dict()
        assert len(bins_calls) == 1  # sorted multi-column: per-column path, no bins
        assert np.array_equal(result["f8"], full["f8"][sorted_idx])
    finally:
        dataset.close()


def test_single_element_selector_keeps_fused_route(multi_record_store, monkeypatch):
    # n == 1 historically routes to the fused (unsorted) kernel via the
    # ``n > 1 and`` guard; the helper change must not move it.
    path, full, _ = multi_record_store
    routes = []
    for name in ("gather_multirecord_sorted", "gather_multirecord"):
        original = getattr(_gather, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            routes.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(_gather, name, spy)
    dataset = colstore.open(path)
    try:
        assert dataset[np.array([42], dtype=np.int64), "f8"].array()[0] == full["f8"][42]
        assert routes == ["gather_multirecord"]
    finally:
        dataset.close()
