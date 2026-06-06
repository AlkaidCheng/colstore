"""Tests for record-base precompute on the irregular multi-column route.

Kernel contract: ``gather_multirecord_withbins_rbase`` fed bins from
``gather_multirecord_bins`` and a record_base array built as
``rsb + col_prefix * nrr - rsr[:-1] * itemsize`` must match
``gather_multirecord_withbins`` on the same inputs for every supported
dtype, irregular record sizes, nonzero prefixes, and misaligned columns.
Routing: irregular multi-column unsorted reads above the size gate run
bins + withbins_rbase for the trailing columns; below the gate the generic
withbins kernel is kept; uniform stores keep the uniform bins pair.
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import _gather
from colstore import reader as reader_mod
from colstore.kernels import cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")


def _irregular_layout(n_rows_per_record, dtype, col_prefix_rows=0, seed=0):
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
    return buf, column, rsr, rsb, nrr, col_prefix_rows


def _record_base(rsr, rsb, nrr, col_prefix, itemsize):
    return rsb + col_prefix * nrr - rsr[:-1] * itemsize


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int16, np.int8])
@pytest.mark.parametrize("thread_cap", [1, 4])
@pytest.mark.parametrize("prefetch", [0, 8])
def test_rbase_kernel_matches_withbins(dtype, thread_cap, prefetch):
    shape = [3, 70, 1, 640, 10, 4, 250, 33]
    buf, column, rsr, rsb, nrr, prefix = _irregular_layout(shape, dtype, col_prefix_rows=2)
    total = int(rsr[-1])
    indices = np.random.default_rng(1).integers(0, total, 4_000).astype(np.int64)
    out_first = np.empty(indices.size, dtype=dtype)
    bins = np.empty(indices.size, dtype=np.int32)
    _gather.gather_multirecord_bins(
        buf, indices, out_first, bins, rsr, rsb, nrr, prefix, thread_cap, prefetch
    )
    rbase = _record_base(rsr, rsb, nrr, prefix, np.dtype(dtype).itemsize)
    out_rbase = np.empty(indices.size, dtype=dtype)
    _gather.gather_multirecord_withbins_rbase(
        buf, indices, out_rbase, bins, rbase, thread_cap, prefetch
    )
    out_withbins = np.empty(indices.size, dtype=dtype)
    _gather.gather_multirecord_withbins(
        buf, indices, out_withbins, bins, rsr, rsb, nrr, prefix, thread_cap, prefetch
    )
    assert np.array_equal(out_rbase, out_withbins)
    assert np.array_equal(out_rbase, column[indices])


def test_rbase_kernel_edge_patterns():
    buf, column, rsr, rsb, nrr, prefix = _irregular_layout([5, 1, 100, 7], np.float64, 3, seed=2)
    total = int(rsr[-1])
    rbase = _record_base(rsr, rsb, nrr, prefix, 8)
    patterns = {
        "boundaries_duplicates": np.array([0, 0, 4, 5, 5, 6, total - 1, total - 1], dtype=np.int64),
        "single": np.array([total // 2], dtype=np.int64),
        "all_rows_shuffled": np.random.default_rng(3).permutation(total).astype(np.int64),
    }
    for name, indices in patterns.items():
        bins = np.empty(indices.size, dtype=np.int32)
        first = np.empty(indices.size, dtype=np.float64)
        _gather.gather_multirecord_bins(buf, indices, first, bins, rsr, rsb, nrr, prefix, 2, 0)
        output = np.empty(indices.size, dtype=np.float64)
        _gather.gather_multirecord_withbins_rbase(buf, indices, output, bins, rbase, 2, 8)
        assert np.array_equal(output, column[indices]), name


def test_rbase_kernel_validates_inputs():
    buf, _, rsr, rsb, nrr, prefix = _irregular_layout([10, 20, 5], np.float64)
    indices = np.array([0, 5, 30], dtype=np.int64)
    output = np.empty(3, dtype=np.float64)
    bins = np.zeros(3, dtype=np.int32)
    rbase = _record_base(rsr, rsb, nrr, prefix, 8)
    with pytest.raises(TypeError, match="int32"):
        _gather.gather_multirecord_withbins_rbase(
            buf, indices, output, bins.astype(np.int64), rbase
        )
    with pytest.raises(TypeError, match="record_base"):
        _gather.gather_multirecord_withbins_rbase(
            buf, indices, output, bins, rbase.astype(np.float64)
        )
    with pytest.raises(ValueError, match="lengths"):
        _gather.gather_multirecord_withbins_rbase(
            buf, indices, np.empty(5, dtype=np.float64), bins, rbase
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather_multirecord_withbins_rbase(
            buf, indices, output, bins, np.repeat(rbase, 2)[::2]
        )


# ---- Reader routing -------------------------------------------------------


def _write_irregular_store(tmp_path, rows_per_record, seed=31):
    rng = np.random.default_rng(seed)
    total = sum(rows_per_record)
    full = {
        "f8": rng.standard_normal(total),
        "f4": rng.standard_normal(total).astype(np.float32),
        "i2": rng.integers(-(2**14), 2**14, total).astype(np.int16),
    }
    path = tmp_path / "rbase.cstore"
    offset = 0
    with colstore.create(path) as writer:
        for rows in rows_per_record:
            writer.write({k: v[offset : offset + rows] for k, v in full.items()})
            offset += rows
    return path, full, total


@pytest.fixture()
def irregular_store(tmp_path):
    shape = [137, 64, 350, 99, 200, 13, 470, 261, 88, 318]
    return _write_irregular_store(tmp_path, shape)


def _spy(monkeypatch, names):
    calls = []
    for name in names:
        original = getattr(_gather, name)

        def wrapper(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(_gather, name, wrapper)
    return calls


KERNELS = [
    "gather_multirecord_bins",
    "gather_multirecord_withbins",
    "gather_multirecord_withbins_rbase",
    "gather_multirecord_uniform_bins",
]


def test_large_read_routes_trailing_columns_to_rbase(irregular_store, monkeypatch):
    path, full, total = irregular_store
    calls = _spy(monkeypatch, KERNELS)
    # n >= n_records * gate: 10 records, 5000 indices -> rbase engaged.
    indices = np.random.default_rng(32).integers(0, total, 5_000).astype(np.int64)
    dataset = colstore.open(path)
    try:
        result = dataset[indices, ["f8", "f4", "i2"]].dict()
        assert calls == [
            "gather_multirecord_bins",
            "gather_multirecord_withbins_rbase",
            "gather_multirecord_withbins_rbase",
        ]
        for name in ("f8", "f4", "i2"):
            assert np.array_equal(result[name], full[name][indices]), name
    finally:
        dataset.close()


def test_small_read_keeps_generic_withbins(tmp_path, monkeypatch):
    # 2000 records, 30 indices: below the gate, generic withbins retained.
    shape = [17, 23] * 1000
    path, full, total = _write_irregular_store(tmp_path, shape, seed=33)
    calls = _spy(monkeypatch, KERNELS)
    indices = np.random.default_rng(34).integers(0, total, 30).astype(np.int64)
    dataset = colstore.open(path)
    try:
        result = dataset[indices, ["f8", "i2"]].dict()
        assert calls == ["gather_multirecord_bins", "gather_multirecord_withbins"]
        assert np.array_equal(result["f8"], full["f8"][indices])
    finally:
        dataset.close()


def test_uniform_store_keeps_uniform_pair(tmp_path, monkeypatch):
    path, full, total = _write_irregular_store(tmp_path, [300] * 12, seed=35)
    calls = _spy(monkeypatch, KERNELS)
    indices = np.random.default_rng(36).integers(0, total, 2_000).astype(np.int64)
    dataset = colstore.open(path)
    try:
        result = dataset[indices, ["f8", "f4"]].dict()
        assert calls == ["gather_multirecord_uniform_bins"]  # withbins variants unspied here
        assert np.array_equal(result["f8"], full["f8"][indices])
    finally:
        dataset.close()


def test_rbase_route_matches_forced_generic_route(irregular_store, monkeypatch):
    path, full, total = irregular_store
    indices = np.random.default_rng(37).integers(0, total, 8_000).astype(np.int64)
    dataset = colstore.open(path)
    try:
        via_rbase = dataset[indices, ["f8", "f4", "i2"]].dict()
    finally:
        dataset.close()
    monkeypatch.setattr(reader_mod, "_RBASE_MIN_INDICES_PER_RECORD", float("inf"))
    dataset = colstore.open(path)
    try:
        via_generic = dataset[indices, ["f8", "f4", "i2"]].dict()
    finally:
        dataset.close()
    for name in ("f8", "f4", "i2"):
        assert np.array_equal(via_rbase[name], via_generic[name]), name
        assert np.array_equal(via_rbase[name], full[name][indices]), name


def test_rbase_misaligned_columns(tmp_path):
    # Odd-length int8 leading column with irregular record sizes: later
    # columns are misaligned and record bases vary non-affinely.
    rng = np.random.default_rng(38)
    rows_per_record = [7, 13, 7, 9, 11, 7, 15, 7]
    total = sum(rows_per_record)
    full = {
        "pad": rng.integers(-100, 100, total).astype(np.int8),
        "f8": rng.standard_normal(total),
        "f4": rng.standard_normal(total).astype(np.float32),
    }
    path = tmp_path / "mis_rbase.cstore"
    offset = 0
    with colstore.create(path) as writer:
        for rows in rows_per_record:
            writer.write({k: v[offset : offset + rows] for k, v in full.items()})
            offset += rows
    indices = np.random.default_rng(39).integers(0, total, 500).astype(np.int64)
    dataset = colstore.open(path)
    try:
        assert dataset._uniform_record_layout() is None
        result = dataset[indices, ["f8", "f4"]].dict()
        for name in ("f8", "f4"):
            assert np.array_equal(result[name], full[name][indices]), name
    finally:
        dataset.close()
