"""Tests for the uniform-record fast path of the unsorted fancy gather.

Kernel contract: ``gather_multirecord_uniform`` must equal the generic fused
kernel (``gather_multirecord``) on every uniform layout it can be routed --
including a partial final record, nonzero column prefixes, duplicates, and
record-boundary indices -- for every supported dtype and prefetch setting.
Detection contract: a layout qualifies exactly when every record but the
last has the same row count, the last is no larger, and the body stride is
constant; anything else returns ``None`` and the generic route is taken.
Routing: uniform stores engage the arithmetic kernel on both the
single-column unsorted path and the multi-column route (where the int32
bins array is skipped entirely); irregular stores keep the bins route.
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import _gather
from colstore.kernels import cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")


def _uniform_layout(n_records, rows, dtype, last_rows=None, col_prefix_rows=0, seed=0):
    """Synthetic single-column uniform byte layout with optional partial tail.

    ``col_prefix_rows`` reserves that many bytes per row ahead of the column
    (simulating preceding columns); the body stride is computed from FULL
    records, as the packed format implies for equal row counts.
    """
    rng = np.random.default_rng(seed)
    itemsize = np.dtype(dtype).itemsize
    last = rows if last_rows is None else last_rows
    per_record_rows = [rows] * (n_records - 1) + [last]
    total = sum(per_record_rows)
    if np.issubdtype(np.dtype(dtype), np.floating):
        column = rng.standard_normal(total).astype(dtype)
    else:
        column = rng.integers(-100, 100, total).astype(dtype)
    stride = rows * (col_prefix_rows + itemsize)
    rsb = np.arange(n_records, dtype=np.int64) * stride
    buf = np.zeros(int(rsb[-1]) + last * (col_prefix_rows + itemsize), dtype=np.uint8)
    nrr = np.asarray(per_record_rows, dtype=np.int64)
    rsr = np.zeros(n_records + 1, dtype=np.int64)
    rsr[1:] = np.cumsum(nrr)
    start_row = 0
    for r, rec_rows in enumerate(per_record_rows):
        off = int(rsb[r]) + col_prefix_rows * rec_rows
        chunk = column[start_row : start_row + rec_rows]
        buf[off : off + chunk.nbytes] = chunk.view(np.uint8)
        start_row += rec_rows
    return buf, column, rsr, rsb, nrr, stride, col_prefix_rows, total


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int16, np.int8])
@pytest.mark.parametrize("last_rows", [None, 1, 37])
@pytest.mark.parametrize("thread_cap", [1, 4])
def test_uniform_kernel_matches_generic_kernel(dtype, last_rows, thread_cap):
    n_records, rows = 64, 100
    buf, column, rsr, rsb, nrr, stride, prefix, total = _uniform_layout(
        n_records, rows, dtype, last_rows=last_rows, seed=1
    )
    indices = np.random.default_rng(2).integers(0, total, 5_000).astype(np.int64)
    out_uniform = np.empty(indices.size, dtype=dtype)
    _gather.gather_multirecord_uniform(
        buf,
        indices,
        out_uniform,
        rows,
        stride,
        0,
        n_records,
        int(nrr[-1]),
        prefix,
        thread_cap,
        0,
    )
    out_generic = np.empty(indices.size, dtype=dtype)
    _gather.gather_multirecord(buf, indices, out_generic, rsr, rsb, nrr, prefix, thread_cap, 0)
    assert np.array_equal(out_uniform, out_generic)
    assert np.array_equal(out_uniform, column[indices])


@pytest.mark.parametrize("prefetch", [0, 8, 128])
def test_uniform_kernel_edge_index_patterns(prefetch):
    n_records, rows, last = 16, 50, 13
    buf, column, _, _, _nrr, stride, prefix, total = _uniform_layout(
        n_records, rows, np.float64, last_rows=last, col_prefix_rows=3, seed=3
    )
    patterns = {
        "boundaries_and_duplicates": np.array(
            [0, 0, rows - 1, rows, total - 1, total - 1, (n_records - 1) * rows],
            dtype=np.int64,
        ),
        "all_in_last_partial_record": np.random.default_rng(4)
        .integers((n_records - 1) * rows, total, 200)
        .astype(np.int64),
        "all_in_one_full_record": np.random.default_rng(5)
        .integers(3 * rows, 4 * rows, 200)
        .astype(np.int64),
        "single_element": np.array([total // 2], dtype=np.int64),
        "every_row_shuffled": np.random.default_rng(6).permutation(total).astype(np.int64),
    }
    for name, indices in patterns.items():
        output = np.empty(indices.size, dtype=np.float64)
        _gather.gather_multirecord_uniform(
            buf, indices, output, rows, stride, 0, n_records, last, prefix, 2, prefetch
        )
        assert np.array_equal(output, column[indices]), (name, prefetch)


def test_uniform_kernel_rows_per_record_one():
    # U == 1: every record is one row; the division degenerates to identity.
    n_records = 200
    buf, column, _, _, _nrr, stride, prefix, total = _uniform_layout(n_records, 1, np.int32, seed=7)
    indices = np.random.default_rng(8).permutation(total).astype(np.int64)
    output = np.empty(indices.size, dtype=np.int32)
    _gather.gather_multirecord_uniform(
        buf, indices, output, 1, stride, 0, n_records, 1, prefix, 1, 8
    )
    assert np.array_equal(output, column[indices])


def test_uniform_kernel_validates_inputs():
    buf, _, _, _, _nrr, stride, prefix, _total = _uniform_layout(4, 10, np.float64)
    indices = np.array([0, 5, 39], dtype=np.int64)
    output = np.empty(3, dtype=np.float64)
    with pytest.raises(ValueError, match="rows_per_record"):
        _gather.gather_multirecord_uniform(buf, indices, output, 0, stride, 0, 4, 10, prefix)
    with pytest.raises(ValueError, match="last_record_rows"):
        _gather.gather_multirecord_uniform(buf, indices, output, 10, stride, 0, 4, 11, prefix)
    with pytest.raises(ValueError, match="output length"):
        _gather.gather_multirecord_uniform(
            buf, indices, np.empty(5, dtype=np.float64), 10, stride, 0, 4, 10, prefix
        )
    with pytest.raises(TypeError, match="int64"):
        _gather.gather_multirecord_uniform(
            buf, indices.astype(np.int32), output, 10, stride, 0, 4, 10, prefix
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather_multirecord_uniform(
            buf, np.repeat(indices, 2)[::2], output, 10, stride, 0, 4, 10, prefix
        )


# ---- Detection + reader routing ------------------------------------------


def _write_store(tmp_path, rows_per_record, seed=11, name="store"):
    rng = np.random.default_rng(seed)
    total = sum(rows_per_record)
    full = {
        "f8": rng.standard_normal(total),
        "f4": rng.standard_normal(total).astype(np.float32),
        "i2": rng.integers(-(2**14), 2**14, total).astype(np.int16),
    }
    path = tmp_path / f"{name}.cstore"
    offset = 0
    with colstore.create(path) as writer:
        for rows in rows_per_record:
            writer.write({k: v[offset : offset + rows] for k, v in full.items()})
            offset += rows
    return path, full, total


@pytest.mark.parametrize("tail", [200, 57])
def test_detection_accepts_uniform_layouts(tmp_path, tail):
    path, _, _ = _write_store(tmp_path, [200] * 9 + [tail])
    dataset = colstore.open(path)
    try:
        layout = dataset._uniform_record_layout()
        assert layout is not None
        rows, stride, first_body, last_rows = layout
        assert rows == 200 and last_rows == tail
        assert stride > 0 and first_body >= 0
    finally:
        dataset.close()


@pytest.mark.parametrize(
    "shape",
    [
        [200] * 5 + [201] + [200] * 4,  # interior record differs
        [200] * 9 + [300],  # last record LARGER than the others
        [100, 200, 200, 200],  # first record differs
    ],
)
def test_detection_rejects_irregular_layouts(tmp_path, shape):
    path, _, _ = _write_store(tmp_path, shape)
    dataset = colstore.open(path)
    try:
        assert dataset._uniform_record_layout() is None
    finally:
        dataset.close()


def _spy(monkeypatch, names):
    calls = []
    for name in names:
        original = getattr(_gather, name)

        def wrapper(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(_gather, name, wrapper)
    return calls


def test_uniform_store_routes_single_column_to_uniform_kernel(tmp_path, monkeypatch):
    path, full, total = _write_store(tmp_path, [500] * 8)
    calls = _spy(monkeypatch, ["gather_multirecord_uniform", "gather_multirecord"])
    indices = np.random.default_rng(12).integers(0, total, 700).astype(np.int64)
    dataset = colstore.open(path)
    try:
        for name, values in full.items():
            assert np.array_equal(dataset[indices, name].array(), values[indices]), name
        assert calls == ["gather_multirecord_uniform"] * 3
    finally:
        dataset.close()


def test_uniform_store_multi_column_uses_uniform_bins_pair(tmp_path, monkeypatch):
    path, full, total = _write_store(tmp_path, [500] * 7 + [123])
    calls = _spy(
        monkeypatch,
        [
            "gather_multirecord_uniform_bins",
            "gather_multirecord_uniform_withbins",
            "gather_multirecord_bins",
            "gather_multirecord_withbins",
        ],
    )
    indices = np.random.default_rng(13).integers(0, total, 900).astype(np.int64)
    dataset = colstore.open(path)
    try:
        result = dataset[indices, ["f8", "f4", "i2"]].dict()
        assert calls == [
            "gather_multirecord_uniform_bins",
            "gather_multirecord_uniform_withbins",
            "gather_multirecord_uniform_withbins",
        ]
        for name in ("f8", "f4", "i2"):
            assert np.array_equal(result[name], full[name][indices]), name
    finally:
        dataset.close()


@pytest.mark.parametrize("last_rows", [None, 13])
@pytest.mark.parametrize("thread_cap", [1, 4])
def test_uniform_bins_pair_matches_generic_pair(last_rows, thread_cap):
    n_records, rows = 32, 100
    buf, column, rsr, rsb, nrr, stride, prefix, total = _uniform_layout(
        n_records, rows, np.float64, last_rows=last_rows, col_prefix_rows=2, seed=21
    )
    indices = np.random.default_rng(22).integers(0, total, 4_000).astype(np.int64)
    last = int(nrr[-1])

    out_u = np.empty(indices.size, dtype=np.float64)
    bins_u = np.empty(indices.size, dtype=np.int32)
    _gather.gather_multirecord_uniform_bins(
        buf, indices, out_u, bins_u, rows, stride, 0, n_records, last, prefix, thread_cap, 0
    )
    out_g = np.empty(indices.size, dtype=np.float64)
    bins_g = np.empty(indices.size, dtype=np.int32)
    _gather.gather_multirecord_bins(
        buf, indices, out_g, bins_g, rsr, rsb, nrr, prefix, thread_cap, 0
    )
    assert np.array_equal(out_u, out_g)
    assert np.array_equal(bins_u, bins_g)
    assert np.array_equal(out_u, column[indices])

    out_w = np.empty(indices.size, dtype=np.float64)
    _gather.gather_multirecord_uniform_withbins(
        buf, indices, out_w, bins_u, rows, stride, 0, n_records, last, prefix, thread_cap, 8
    )
    assert np.array_equal(out_w, column[indices])


def test_uniform_bins_pair_validates_inputs():
    buf, _, _, _, _nrr, stride, prefix, _total = _uniform_layout(4, 10, np.float64)
    indices = np.array([0, 5, 39], dtype=np.int64)
    output = np.empty(3, dtype=np.float64)
    bins = np.empty(3, dtype=np.int32)
    with pytest.raises(TypeError, match="int32"):
        _gather.gather_multirecord_uniform_bins(
            buf, indices, output, bins.astype(np.int64), 10, stride, 0, 4, 10, prefix
        )
    with pytest.raises(ValueError, match="lengths"):
        _gather.gather_multirecord_uniform_withbins(
            buf, indices, output, np.empty(5, dtype=np.int32), 10, stride, 0, 4, 10, prefix
        )


def test_irregular_store_keeps_bins_route(tmp_path, monkeypatch):
    path, full, total = _write_store(tmp_path, [500, 400, 500, 500, 500])
    calls = _spy(
        monkeypatch,
        ["gather_multirecord_uniform", "gather_multirecord_bins", "gather_multirecord_withbins"],
    )
    indices = np.random.default_rng(14).integers(0, total, 900).astype(np.int64)
    dataset = colstore.open(path)
    try:
        result = dataset[indices, ["f8", "i2"]].dict()
        assert calls == ["gather_multirecord_bins", "gather_multirecord_withbins"]
        assert np.array_equal(result["f8"], full["f8"][indices])
    finally:
        dataset.close()


def test_uniform_store_sorted_path_unaffected(tmp_path, monkeypatch):
    path, full, total = _write_store(tmp_path, [500] * 8)
    calls = _spy(monkeypatch, ["gather_multirecord_uniform", "gather_multirecord_sorted"])
    indices = np.sort(np.random.default_rng(15).integers(0, total, 700).astype(np.int64))
    dataset = colstore.open(path)
    try:
        assert np.array_equal(dataset[indices, "f8"].array(), full["f8"][indices])
        assert calls == ["gather_multirecord_sorted"]
    finally:
        dataset.close()


def test_uniform_misaligned_columns(tmp_path):
    # Odd-length int8 leading column: every later column starts at odd byte
    # addresses inside each record body; uniform-kernel loads must be
    # alignment-safe.
    rng = np.random.default_rng(16)
    n_records, rows = 12, 7
    total = n_records * rows
    full = {
        "pad": rng.integers(-100, 100, total).astype(np.int8),
        "f8": rng.standard_normal(total),
        "f4": rng.standard_normal(total).astype(np.float32),
    }
    path = tmp_path / "mis_uniform.cstore"
    with colstore.create(path) as writer:
        for r in range(n_records):
            writer.write({k: v[r * rows : (r + 1) * rows] for k, v in full.items()})
    indices = np.random.default_rng(17).integers(0, total, 300).astype(np.int64)
    dataset = colstore.open(path)
    try:
        assert dataset._uniform_record_layout() is not None
        for name in ("f8", "f4"):
            assert np.array_equal(dataset[indices, name].array(), full[name][indices]), name
    finally:
        dataset.close()


def test_forced_generic_route_matches_uniform_route(tmp_path, monkeypatch):
    # The benchmark's baseline seam: detection forced to None must produce
    # identical results through the generic kernels.
    from colstore import reader as reader_mod

    path, _full, total = _write_store(tmp_path, [400] * 10)
    indices = np.random.default_rng(18).integers(0, total, 1_500).astype(np.int64)
    dataset = colstore.open(path)
    try:
        via_uniform = dataset[indices, ["f8", "f4"]].dict()
    finally:
        dataset.close()
    monkeypatch.setattr(
        reader_mod.ColStoreReader, "_detect_uniform_record_layout", lambda self: None
    )
    dataset = colstore.open(path)
    try:
        assert dataset._uniform_record_layout() is None
        via_generic = dataset[indices, ["f8", "f4"]].dict()
    finally:
        dataset.close()
    for name in ("f8", "f4"):
        assert np.array_equal(via_uniform[name], via_generic[name]), name
