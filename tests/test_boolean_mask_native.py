"""Tests for the boolean-mask-native read path.

Kernel contract: ``gather_multirecord_mask`` must equal NumPy boolean
indexing of the full column for every supported dtype, mask density,
record shape (including records shorter than the 8-byte mask word), thread
cap, and prefetch setting; the internal count check rejects mis-sized
outputs without writing. Routing: multi-record reads with native dtypes
and mask density at or above the gate take the kernel (single column and
multi-column); sparse masks, single-record stores, and non-native dtypes
lower to ``np.flatnonzero`` and the pre-existing fancy paths, preserving
their semantics (including the backend contract on single-record fancy
reads). Results are identical on both sides of the gate.
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import _gather
from colstore import config as config_mod
from colstore.kernels import cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")

# Routing tests pin an explicit nonzero gate so both sides of the route
# are exercised regardless of the compiled default (0.0: always native)
# or any calibration cache on the dev machine.
GATE = 0.15


def _layout(n_rows_per_record, dtype, col_prefix_rows=0, seed=0):
    rng = np.random.default_rng(seed)
    itemsize = np.dtype(dtype).itemsize
    nrr = np.asarray(n_rows_per_record, dtype=np.int64)
    n_records = nrr.shape[0]
    rsr = np.zeros(n_records + 1, dtype=np.int64)
    rsr[1:] = np.cumsum(nrr)
    body = nrr * (col_prefix_rows + itemsize)
    rsb = np.zeros(n_records, dtype=np.int64)
    rsb[1:] = np.cumsum(body)[:-1]
    total = int(rsr[-1])
    if np.issubdtype(np.dtype(dtype), np.floating):
        column = rng.standard_normal(total).astype(dtype)
    else:
        column = rng.integers(-100, 100, total).astype(dtype)
    buf = np.zeros(int(body.sum()), dtype=np.uint8)
    for r in range(n_records):
        off = int(rsb[r]) + col_prefix_rows * int(nrr[r])
        rows = column[int(rsr[r]) : int(rsr[r + 1])]
        buf[off : off + rows.nbytes] = rows.view(np.uint8)
    return buf, column, rsr, rsb, nrr, col_prefix_rows, total


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int16, np.int8])
@pytest.mark.parametrize("density", [0.9, 0.5, 0.1, 0.01])
@pytest.mark.parametrize("thread_cap", [1, 4])
def test_mask_kernel_matches_boolean_indexing(dtype, density, thread_cap):
    buf, column, rsr, rsb, nrr, prefix, total = _layout([137, 64, 350, 99, 470, 261], dtype, 2)
    mask = np.random.default_rng(1).random(total) < density
    output = np.empty(int(mask.sum()), dtype=dtype)
    _gather.gather_multirecord_mask(buf, mask, output, rsr, rsb, nrr, prefix, thread_cap, 8)
    assert np.array_equal(output, column[mask])


@pytest.mark.parametrize("prefetch", [0, 8, 128])
def test_mask_kernel_edge_masks(prefetch):
    # Records shorter than the 8-byte mask word exercise the scalar form.
    buf, column, rsr, rsb, nrr, prefix, total = _layout([3, 7, 1, 5, 2, 9, 100, 4], np.float64, 3)
    masks = {
        "all_true": np.ones(total, dtype=bool),
        "all_false": np.zeros(total, dtype=bool),
        "single_last_row": np.zeros(total, dtype=bool),
        "head_run_only": np.zeros(total, dtype=bool),
        "alternating": np.tile([True, False], total // 2 + 1)[:total],
    }
    masks["single_last_row"][-1] = True
    masks["head_run_only"][:9] = True
    for name, mask in masks.items():
        output = np.empty(int(mask.sum()), dtype=np.float64)
        _gather.gather_multirecord_mask(buf, mask, output, rsr, rsb, nrr, prefix, 2, prefetch)
        assert np.array_equal(output, column[mask]), (name, prefetch)


def test_mask_kernel_quota_boundaries_threaded():
    # Selected elements clustered at chunk boundaries stress the branchless
    # over-store quota guard: each thread's writes must stay in its region.
    buf, column, rsr, rsb, nrr, prefix, total = _layout([1000] * 64, np.float64, seed=2)
    mask = np.zeros(total, dtype=bool)
    chunk = total // 4
    for boundary in range(chunk, total, chunk):
        mask[boundary - 11 : boundary + 11] = True
    output = np.empty(int(mask.sum()), dtype=np.float64)
    _gather.gather_multirecord_mask(buf, mask, output, rsr, rsb, nrr, prefix, 4, 8)
    assert np.array_equal(output, column[mask])


def test_mask_kernel_count_mismatch_rejected():
    buf, _, rsr, rsb, nrr, prefix, total = _layout([10, 20, 5], np.float64)
    mask = np.zeros(total, dtype=bool)
    mask[::3] = True
    wrong = np.empty(int(mask.sum()) + 1, dtype=np.float64)
    with pytest.raises(ValueError, match="selected count"):
        _gather.gather_multirecord_mask(buf, mask, wrong, rsr, rsb, nrr, prefix)


def test_mask_kernel_validates_inputs():
    buf, _, rsr, rsb, nrr, prefix, total = _layout([10, 20, 5], np.float64)
    mask = np.zeros(total, dtype=bool)
    output = np.empty(0, dtype=np.float64)
    with pytest.raises(TypeError, match="bool"):
        _gather.gather_multirecord_mask(buf, mask.astype(np.uint8), output, rsr, rsb, nrr, prefix)
    with pytest.raises(ValueError, match="row count"):
        _gather.gather_multirecord_mask(buf, mask[:-1], output, rsr, rsb, nrr, prefix)
    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather_multirecord_mask(buf, np.repeat(mask, 2)[::2], output, rsr, rsb, nrr, prefix)


# ---- Reader routing -------------------------------------------------------


def _write_store(tmp_path, rows_per_record, seed=61):
    rng = np.random.default_rng(seed)
    total = sum(rows_per_record)
    full = {
        "f8": rng.standard_normal(total),
        "f4": rng.standard_normal(total).astype(np.float32),
        "i2": rng.integers(-(2**14), 2**14, total).astype(np.int16),
    }
    path = tmp_path / "mask.cstore"
    offset = 0
    with colstore.create(path) as writer:
        for rows in rows_per_record:
            writer.write({k: v[offset : offset + rows] for k, v in full.items()})
            offset += rows
    return path, full, total


@pytest.fixture()
def irregular_store(tmp_path):
    return _write_store(tmp_path, [137, 640, 350, 99, 2000, 13, 470, 2610, 88, 318])


def _spy(monkeypatch, names):
    calls = []
    for name in names:
        original = getattr(_gather, name)

        def wrapper(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(_gather, name, wrapper)
    return calls


SPIED = ["gather_multirecord_mask", "gather_multirecord_sorted", "gather_multirecord_bins"]


def test_dense_mask_routes_to_mask_kernel(irregular_store, monkeypatch):
    path, full, total = irregular_store
    monkeypatch.setattr(config_mod, "_mask_density_gate", GATE)
    calls = _spy(monkeypatch, SPIED)
    mask = np.random.default_rng(62).random(total) < 0.5
    dataset = colstore.open(path)
    try:
        assert np.array_equal(dataset[mask, "f8"].array(), full["f8"][mask])
        assert calls == ["gather_multirecord_mask"]
        calls.clear()
        result = dataset[mask, ["f8", "f4", "i2"]].dict()
        assert calls == ["gather_multirecord_mask"] * 3
        for name in ("f8", "f4", "i2"):
            assert np.array_equal(result[name], full[name][mask]), name
    finally:
        dataset.close()


def test_sparse_mask_lowers_to_indices(irregular_store, monkeypatch):
    path, full, total = irregular_store
    monkeypatch.setattr(config_mod, "_mask_density_gate", GATE)
    calls = _spy(monkeypatch, SPIED)
    mask = np.zeros(total, dtype=bool)
    mask[:: int(2 / GATE)] = True  # density well below the gate
    assert mask.mean() < GATE
    dataset = colstore.open(path)
    try:
        assert np.array_equal(dataset[mask, "f8"].array(), full["f8"][mask])
        assert calls == ["gather_multirecord_sorted"]  # flatnonzero is sorted
    finally:
        dataset.close()


def test_gate_seam_parity(irregular_store, monkeypatch):
    path, full, total = irregular_store
    mask = np.random.default_rng(63).random(total) < 0.4
    dataset = colstore.open(path)
    try:
        via_mask = dataset[mask, ["f8", "i2"]].dict()
        monkeypatch.setattr(config_mod, "_mask_density_gate", 2.0)
        via_indices = dataset[mask, ["f8", "i2"]].dict()
    finally:
        dataset.close()
    for name in ("f8", "i2"):
        assert np.array_equal(via_mask[name], via_indices[name]), name
        assert np.array_equal(via_mask[name], full[name][mask]), name


def test_single_record_masks_keep_fancy_path(tmp_path, monkeypatch):
    rng = np.random.default_rng(64)
    data = {"a": rng.standard_normal(5_000)}
    path = tmp_path / "single.cstore"
    with colstore.create(path) as writer:
        writer.write(data)
    calls = _spy(monkeypatch, ["gather_multirecord_mask"])
    mask = rng.random(5_000) < 0.6
    dataset = colstore.open(path)
    try:
        assert np.array_equal(dataset[mask, "a"].array(), data["a"][mask])
        assert calls == []  # mask lowered to indices; backend contract intact
    finally:
        dataset.close()


def test_mask_misaligned_columns(tmp_path):
    rng = np.random.default_rng(65)
    rows_per_record = [7, 13, 7, 9, 11, 7, 15, 7]
    total = sum(rows_per_record)
    full = {
        "pad": rng.integers(-100, 100, total).astype(np.int8),
        "f8": rng.standard_normal(total),
    }
    path = tmp_path / "mis_mask.cstore"
    offset = 0
    with colstore.create(path) as writer:
        for rows in rows_per_record:
            writer.write({k: v[offset : offset + rows] for k, v in full.items()})
            offset += rows
    mask = rng.random(total) < 0.5
    dataset = colstore.open(path)
    try:
        assert np.array_equal(dataset[mask, "f8"].array(), full["f8"][mask])
    finally:
        dataset.close()


def test_mask_zero_copy_still_rejected(tmp_path):
    rng = np.random.default_rng(66)
    path = tmp_path / "zc.cstore"
    with colstore.create(path) as writer:
        writer.write({"a": rng.standard_normal(100)})
    dataset = colstore.open(path)
    try:
        with pytest.raises(ValueError, match="copy=True"):
            dataset[np.ones(100, dtype=bool), "a"].array(copy=False)
    finally:
        dataset.close()
