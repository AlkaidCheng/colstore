"""Kernel contracts of the multi-record gather family (direct ``_gather`` calls).

One section per kernel; each must equal its reference (NumPy indexing, or
the generic fused kernel) for every supported dtype, irregular record
shapes, nonzero column prefixes, duplicates, record-boundary indices,
thread caps, and prefetch settings, and must validate its inputs. Reader
routing for these kernels is pinned in ``test_multirecord_routing.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import build_layout, build_uniform_layout

from colstore import _gather
from colstore.kernels import cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")


# ---- Sorted linear-walk kernel ---------------------------------------------


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8])
@pytest.mark.parametrize("thread_cap", [1, 4])
def test_sorted_kernel_matches_unsorted_kernel(dtype, thread_cap):
    n_records, rows = 64, 100
    lay = build_layout([rows] * n_records, dtype)
    rng = np.random.default_rng(1)
    indices = np.sort(rng.integers(0, lay.total, size=5_000).astype(np.int64))

    out_sorted = np.empty(indices.size, dtype=dtype)
    _gather.gather_multirecord_sorted(
        lay.buf, indices, out_sorted, lay.rsr, lay.rsb, lay.nrr, 0, thread_cap, 0
    )
    out_reference = np.empty(indices.size, dtype=dtype)
    _gather.gather_multirecord(
        lay.buf, indices, out_reference, lay.rsr, lay.rsb, lay.nrr, 0, thread_cap, 0
    )
    assert np.array_equal(out_sorted, out_reference)
    assert np.array_equal(out_sorted, lay.column[indices])


@pytest.mark.parametrize("prefetch", [0, 8, 128])
def test_sorted_kernel_edge_index_patterns(prefetch):
    n_records, rows = 16, 50
    lay = build_layout([rows] * n_records, np.float64, seed=2)
    total = lay.total
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
        _gather.gather_multirecord_sorted(
            lay.buf, indices, output, lay.rsr, lay.rsb, lay.nrr, 0, 2, prefetch
        )
        assert np.array_equal(output, lay.column[indices]), (name, prefetch)


def test_sorted_kernel_validates_inputs():
    lay = build_layout([10] * 4, np.float64)
    indices = np.array([0, 5, 39], dtype=np.int64)
    with pytest.raises(TypeError, match="int64"):
        _gather.gather_multirecord_sorted(
            lay.buf, indices.astype(np.int32), np.empty(3), lay.rsr, lay.rsb, lay.nrr, 0
        )
    with pytest.raises(ValueError, match="length"):
        _gather.gather_multirecord_sorted(
            lay.buf, indices, np.empty(2), lay.rsr, lay.rsb, lay.nrr, 0
        )


# ---- Strided range-walk kernel ---------------------------------------------


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
    lay = build_layout([3, 7, 1, 64, 10, 4, 25], dtype, seed=1)
    for start, stop, step in _slice_rows(lay.total):
        n = len(range(start, stop, step))
        output = np.empty(n, dtype=dtype)
        _gather.gather_multirecord_strided(
            lay.buf,
            output,
            start,
            stop,
            step,
            lay.rsr,
            lay.rsb,
            lay.nrr,
            lay.prefix,
            thread_cap,
            0,
        )
        assert np.array_equal(output, _expected(lay.column, start, stop, step)), (
            start,
            stop,
            step,
        )


@pytest.mark.parametrize("prefetch", [0, 8, 128])
def test_strided_kernel_prefetch_invariance(prefetch):
    lay = build_layout([50] * 16, np.float64, seed=2)
    for start, stop, step in _slice_rows(lay.total):
        n = len(range(start, stop, step))
        output = np.empty(n, dtype=np.float64)
        _gather.gather_multirecord_strided(
            lay.buf, output, start, stop, step, lay.rsr, lay.rsb, lay.nrr, lay.prefix, 2, prefetch
        )
        assert np.array_equal(output, _expected(lay.column, start, stop, step)), (
            start,
            stop,
            step,
            prefetch,
        )


def test_strided_kernel_column_prefix_addressing():
    # Non-zero column prefix: the column sits behind 3 bytes/row of earlier
    # columns, so record-base arithmetic must include the prefix term.
    lay = build_layout([9, 2, 31, 5], np.int32, col_prefix_rows=3)
    total = lay.total
    output = np.empty(len(range(1, total, 4)), dtype=np.int32)
    _gather.gather_multirecord_strided(
        lay.buf, output, 1, total, 4, lay.rsr, lay.rsb, lay.nrr, lay.prefix, 1, 8
    )
    assert np.array_equal(output, lay.column[1:total:4])


def test_strided_kernel_step_crossing_many_records_both_directions():
    # Step larger than every record: each element lands in a new record, so
    # the cursor-advance loop (not the steady state) carries the walk.
    lay = build_layout([4] * 200, np.float64, seed=3)
    total = lay.total
    for start, stop, step in [(0, total, 13), (total - 1, -1, -13)]:
        n = len(range(start, stop, step))
        output = np.empty(n, dtype=np.float64)
        _gather.gather_multirecord_strided(
            lay.buf, output, start, stop, step, lay.rsr, lay.rsb, lay.nrr, lay.prefix, 2, 8
        )
        assert np.array_equal(output, _expected(lay.column, start, stop, step))


def test_strided_kernel_validates_inputs():
    lay = build_layout([10] * 4, np.float64)
    output = np.empty(5, dtype=np.float64)
    with pytest.raises(ValueError, match="step"):
        _gather.gather_multirecord_strided(
            lay.buf, output, 0, 10, 0, lay.rsr, lay.rsb, lay.nrr, lay.prefix
        )
    with pytest.raises(ValueError, match="output length"):
        _gather.gather_multirecord_strided(
            lay.buf, output, 0, 40, 2, lay.rsr, lay.rsb, lay.nrr, lay.prefix
        )
    with pytest.raises(TypeError, match="int64"):
        _gather.gather_multirecord_strided(
            lay.buf,
            np.empty(20, dtype=np.float64),
            0,
            40,
            2,
            lay.rsr.astype(np.int32),
            lay.rsb,
            lay.nrr,
            lay.prefix,
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        strided_rsr = np.repeat(lay.rsr, 2)[::2]
        _gather.gather_multirecord_strided(
            lay.buf,
            np.empty(20, dtype=np.float64),
            0,
            40,
            2,
            strided_rsr,
            lay.rsb,
            lay.nrr,
            lay.prefix,
        )


# ---- Uniform-record arithmetic-binning kernels ------------------------------


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int16, np.int8])
@pytest.mark.parametrize("last_rows", [None, 1, 37])
@pytest.mark.parametrize("thread_cap", [1, 4])
def test_uniform_kernel_matches_generic_kernel(dtype, last_rows, thread_cap):
    n_records, rows = 64, 100
    lay = build_uniform_layout(n_records, rows, dtype, last_rows=last_rows, seed=1)
    indices = np.random.default_rng(2).integers(0, lay.total, 5_000).astype(np.int64)
    out_uniform = np.empty(indices.size, dtype=dtype)
    _gather.gather_multirecord_uniform(
        lay.buf,
        indices,
        out_uniform,
        rows,
        lay.stride,
        0,
        n_records,
        int(lay.nrr[-1]),
        lay.prefix,
        thread_cap,
        0,
    )
    out_generic = np.empty(indices.size, dtype=dtype)
    _gather.gather_multirecord(
        lay.buf, indices, out_generic, lay.rsr, lay.rsb, lay.nrr, lay.prefix, thread_cap, 0
    )
    assert np.array_equal(out_uniform, out_generic)
    assert np.array_equal(out_uniform, lay.column[indices])


@pytest.mark.parametrize("prefetch", [0, 8, 128])
def test_uniform_kernel_edge_index_patterns(prefetch):
    n_records, rows, last = 16, 50, 13
    lay = build_uniform_layout(
        n_records, rows, np.float64, last_rows=last, col_prefix_rows=3, seed=3
    )
    total = lay.total
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
            lay.buf, indices, output, rows, lay.stride, 0, n_records, last, lay.prefix, 2, prefetch
        )
        assert np.array_equal(output, lay.column[indices]), (name, prefetch)


def test_uniform_kernel_rows_per_record_one():
    # U == 1: every record is one row; the division degenerates to identity.
    n_records = 200
    lay = build_uniform_layout(n_records, 1, np.int32, seed=7)
    indices = np.random.default_rng(8).permutation(lay.total).astype(np.int64)
    output = np.empty(indices.size, dtype=np.int32)
    _gather.gather_multirecord_uniform(
        lay.buf, indices, output, 1, lay.stride, 0, n_records, 1, lay.prefix, 1, 8
    )
    assert np.array_equal(output, lay.column[indices])


def test_uniform_kernel_validates_inputs():
    lay = build_uniform_layout(4, 10, np.float64)
    indices = np.array([0, 5, 39], dtype=np.int64)
    output = np.empty(3, dtype=np.float64)
    with pytest.raises(ValueError, match="rows_per_record"):
        _gather.gather_multirecord_uniform(
            lay.buf, indices, output, 0, lay.stride, 0, 4, 10, lay.prefix
        )
    with pytest.raises(ValueError, match="last_record_rows"):
        _gather.gather_multirecord_uniform(
            lay.buf, indices, output, 10, lay.stride, 0, 4, 11, lay.prefix
        )
    with pytest.raises(ValueError, match="output length"):
        _gather.gather_multirecord_uniform(
            lay.buf, indices, np.empty(5, dtype=np.float64), 10, lay.stride, 0, 4, 10, lay.prefix
        )
    with pytest.raises(TypeError, match="int64"):
        _gather.gather_multirecord_uniform(
            lay.buf, indices.astype(np.int32), output, 10, lay.stride, 0, 4, 10, lay.prefix
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather_multirecord_uniform(
            lay.buf, np.repeat(indices, 2)[::2], output, 10, lay.stride, 0, 4, 10, lay.prefix
        )


@pytest.mark.parametrize("last_rows", [None, 13])
@pytest.mark.parametrize("thread_cap", [1, 4])
def test_uniform_bins_pair_matches_generic_pair(last_rows, thread_cap):
    n_records, rows = 32, 100
    lay = build_uniform_layout(
        n_records, rows, np.float64, last_rows=last_rows, col_prefix_rows=2, seed=21
    )
    indices = np.random.default_rng(22).integers(0, lay.total, 4_000).astype(np.int64)
    last = int(lay.nrr[-1])

    out_u = np.empty(indices.size, dtype=np.float64)
    bins_u = np.empty(indices.size, dtype=np.int32)
    _gather.gather_multirecord_uniform_bins(
        lay.buf,
        indices,
        out_u,
        bins_u,
        rows,
        lay.stride,
        0,
        n_records,
        last,
        lay.prefix,
        thread_cap,
        0,
    )
    out_g = np.empty(indices.size, dtype=np.float64)
    bins_g = np.empty(indices.size, dtype=np.int32)
    _gather.gather_multirecord_bins(
        lay.buf, indices, out_g, bins_g, lay.rsr, lay.rsb, lay.nrr, lay.prefix, thread_cap, 0
    )
    assert np.array_equal(out_u, out_g)
    assert np.array_equal(bins_u, bins_g)
    assert np.array_equal(out_u, lay.column[indices])

    out_w = np.empty(indices.size, dtype=np.float64)
    _gather.gather_multirecord_uniform_withbins(
        lay.buf,
        indices,
        out_w,
        bins_u,
        rows,
        lay.stride,
        0,
        n_records,
        last,
        lay.prefix,
        thread_cap,
        8,
    )
    assert np.array_equal(out_w, lay.column[indices])


def test_uniform_bins_pair_validates_inputs():
    lay = build_uniform_layout(4, 10, np.float64)
    indices = np.array([0, 5, 39], dtype=np.int64)
    output = np.empty(3, dtype=np.float64)
    bins = np.empty(3, dtype=np.int32)
    with pytest.raises(TypeError, match="int32"):
        _gather.gather_multirecord_uniform_bins(
            lay.buf, indices, output, bins.astype(np.int64), 10, lay.stride, 0, 4, 10, lay.prefix
        )
    with pytest.raises(ValueError, match="lengths"):
        _gather.gather_multirecord_uniform_withbins(
            lay.buf,
            indices,
            output,
            np.empty(5, dtype=np.int32),
            10,
            lay.stride,
            0,
            4,
            10,
            lay.prefix,
        )


# ---- Bin-reuse pair (bins / withbins) ---------------------------------------


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8])
def test_bins_kernel_matches_searchsorted_and_plain_kernel(dtype):
    lay = build_layout([100] * 64, dtype, seed=1)
    indices = np.random.default_rng(1).integers(0, lay.total, size=5_000).astype(np.int64)

    out_plain = np.empty(indices.size, dtype=dtype)
    _gather.gather_multirecord(lay.buf, indices, out_plain, lay.rsr, lay.rsb, lay.nrr, 0, 2, 0)

    out_bins = np.empty(indices.size, dtype=dtype)
    bins = np.empty(indices.size, dtype=np.int32)
    _gather.gather_multirecord_bins(
        lay.buf, indices, out_bins, bins, lay.rsr, lay.rsb, lay.nrr, 0, 2, 0
    )
    assert np.array_equal(out_bins, out_plain)
    expected_bins = (np.searchsorted(lay.rsr, indices, side="right") - 1).astype(np.int32)
    assert np.array_equal(bins, expected_bins)

    out_with = np.empty(indices.size, dtype=dtype)
    _gather.gather_multirecord_withbins(
        lay.buf, indices, out_with, bins, lay.rsr, lay.rsb, lay.nrr, 0, 2, 0
    )
    assert np.array_equal(out_with, out_plain)


def test_bins_kernels_validate_bins_dtype_and_length():
    lay = build_layout([10] * 4, np.float64)
    indices = np.array([0, 5, 39], dtype=np.int64)
    out = np.empty(3)
    with pytest.raises(TypeError, match="bins must be int32"):
        _gather.gather_multirecord_bins(
            lay.buf, indices, out, np.empty(3, dtype=np.int64), lay.rsr, lay.rsb, lay.nrr, 0
        )
    with pytest.raises(ValueError, match="lengths"):
        _gather.gather_multirecord_withbins(
            lay.buf, indices, out, np.empty(2, dtype=np.int32), lay.rsr, lay.rsb, lay.nrr, 0
        )


# ---- Record-base (rbase) variant --------------------------------------------


def _record_base(rsr, rsb, nrr, col_prefix, itemsize):
    return rsb + col_prefix * nrr - rsr[:-1] * itemsize


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int16, np.int8])
@pytest.mark.parametrize("thread_cap", [1, 4])
@pytest.mark.parametrize("prefetch", [0, 8])
def test_rbase_kernel_matches_withbins(dtype, thread_cap, prefetch):
    lay = build_layout([3, 70, 1, 640, 10, 4, 250, 33], dtype, col_prefix_rows=2)
    indices = np.random.default_rng(1).integers(0, lay.total, 4_000).astype(np.int64)
    out_first = np.empty(indices.size, dtype=dtype)
    bins = np.empty(indices.size, dtype=np.int32)
    _gather.gather_multirecord_bins(
        lay.buf,
        indices,
        out_first,
        bins,
        lay.rsr,
        lay.rsb,
        lay.nrr,
        lay.prefix,
        thread_cap,
        prefetch,
    )
    rbase = _record_base(lay.rsr, lay.rsb, lay.nrr, lay.prefix, np.dtype(dtype).itemsize)
    out_rbase = np.empty(indices.size, dtype=dtype)
    _gather.gather_multirecord_withbins_rbase(
        lay.buf, indices, out_rbase, bins, rbase, thread_cap, prefetch
    )
    out_withbins = np.empty(indices.size, dtype=dtype)
    _gather.gather_multirecord_withbins(
        lay.buf,
        indices,
        out_withbins,
        bins,
        lay.rsr,
        lay.rsb,
        lay.nrr,
        lay.prefix,
        thread_cap,
        prefetch,
    )
    assert np.array_equal(out_rbase, out_withbins)
    assert np.array_equal(out_rbase, lay.column[indices])


def test_rbase_kernel_edge_patterns():
    lay = build_layout([5, 1, 100, 7], np.float64, col_prefix_rows=3, seed=2)
    total = lay.total
    rbase = _record_base(lay.rsr, lay.rsb, lay.nrr, lay.prefix, 8)
    patterns = {
        "boundaries_duplicates": np.array([0, 0, 4, 5, 5, 6, total - 1, total - 1], dtype=np.int64),
        "single": np.array([total // 2], dtype=np.int64),
        "all_rows_shuffled": np.random.default_rng(3).permutation(total).astype(np.int64),
    }
    for name, indices in patterns.items():
        bins = np.empty(indices.size, dtype=np.int32)
        first = np.empty(indices.size, dtype=np.float64)
        _gather.gather_multirecord_bins(
            lay.buf, indices, first, bins, lay.rsr, lay.rsb, lay.nrr, lay.prefix, 2, 0
        )
        output = np.empty(indices.size, dtype=np.float64)
        _gather.gather_multirecord_withbins_rbase(lay.buf, indices, output, bins, rbase, 2, 8)
        assert np.array_equal(output, lay.column[indices]), name


def test_rbase_kernel_validates_inputs():
    lay = build_layout([10, 20, 5], np.float64)
    indices = np.array([0, 5, 30], dtype=np.int64)
    output = np.empty(3, dtype=np.float64)
    bins = np.zeros(3, dtype=np.int32)
    rbase = _record_base(lay.rsr, lay.rsb, lay.nrr, lay.prefix, 8)
    with pytest.raises(TypeError, match="int32"):
        _gather.gather_multirecord_withbins_rbase(
            lay.buf, indices, output, bins.astype(np.int64), rbase
        )
    with pytest.raises(TypeError, match="record_base"):
        _gather.gather_multirecord_withbins_rbase(
            lay.buf, indices, output, bins, rbase.astype(np.float64)
        )
    with pytest.raises(ValueError, match="lengths"):
        _gather.gather_multirecord_withbins_rbase(
            lay.buf, indices, np.empty(5, dtype=np.float64), bins, rbase
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather_multirecord_withbins_rbase(
            lay.buf, indices, output, bins, np.repeat(rbase, 2)[::2]
        )


# ---- Boolean-mask-native kernel ---------------------------------------------


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int16, np.int8])
@pytest.mark.parametrize("density", [0.9, 0.5, 0.1, 0.01])
@pytest.mark.parametrize("thread_cap", [1, 4])
def test_mask_kernel_matches_boolean_indexing(dtype, density, thread_cap):
    lay = build_layout([137, 64, 350, 99, 470, 261], dtype, col_prefix_rows=2)
    mask = np.random.default_rng(1).random(lay.total) < density
    output = np.empty(int(mask.sum()), dtype=dtype)
    _gather.gather_multirecord_mask(
        lay.buf, mask, output, lay.rsr, lay.rsb, lay.nrr, lay.prefix, thread_cap, 8
    )
    assert np.array_equal(output, lay.column[mask])


@pytest.mark.parametrize("prefetch", [0, 8, 128])
def test_mask_kernel_edge_masks(prefetch):
    # Records shorter than the 8-byte mask word exercise the scalar form.
    lay = build_layout([3, 7, 1, 5, 2, 9, 100, 4], np.float64, col_prefix_rows=3)
    total = lay.total
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
        _gather.gather_multirecord_mask(
            lay.buf, mask, output, lay.rsr, lay.rsb, lay.nrr, lay.prefix, 2, prefetch
        )
        assert np.array_equal(output, lay.column[mask]), (name, prefetch)


def test_mask_kernel_quota_boundaries_threaded():
    # Selected elements clustered at chunk boundaries stress the branchless
    # over-store quota guard: each thread's writes must stay in its region.
    lay = build_layout([1000] * 64, np.float64, seed=2)
    total = lay.total
    mask = np.zeros(total, dtype=bool)
    chunk = total // 4
    for boundary in range(chunk, total, chunk):
        mask[boundary - 11 : boundary + 11] = True
    output = np.empty(int(mask.sum()), dtype=np.float64)
    _gather.gather_multirecord_mask(
        lay.buf, mask, output, lay.rsr, lay.rsb, lay.nrr, lay.prefix, 4, 8
    )
    assert np.array_equal(output, lay.column[mask])


def test_mask_kernel_count_mismatch_rejected():
    lay = build_layout([10, 20, 5], np.float64)
    mask = np.zeros(lay.total, dtype=bool)
    mask[::3] = True
    wrong = np.empty(int(mask.sum()) + 1, dtype=np.float64)
    with pytest.raises(ValueError, match="selected count"):
        _gather.gather_multirecord_mask(lay.buf, mask, wrong, lay.rsr, lay.rsb, lay.nrr, lay.prefix)


def test_mask_kernel_validates_inputs():
    lay = build_layout([10, 20, 5], np.float64)
    mask = np.zeros(lay.total, dtype=bool)
    output = np.empty(0, dtype=np.float64)
    with pytest.raises(TypeError, match="bool"):
        _gather.gather_multirecord_mask(
            lay.buf, mask.astype(np.uint8), output, lay.rsr, lay.rsb, lay.nrr, lay.prefix
        )
    with pytest.raises(ValueError, match="row count"):
        _gather.gather_multirecord_mask(
            lay.buf, mask[:-1], output, lay.rsr, lay.rsb, lay.nrr, lay.prefix
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather_multirecord_mask(
            lay.buf, np.repeat(mask, 2)[::2], output, lay.rsr, lay.rsb, lay.nrr, lay.prefix
        )


# ---- Contiguity rejection across all pointer-interpreting entries -----------


def test_kernel_entries_reject_strided_arrays():
    # Direct-API backstop: every pointer-interpreting entry must refuse
    # strided arrays rather than misread them.
    lay = build_layout([10] * 4, np.float64)
    total_rows = lay.total
    strided = np.arange(2 * total_rows, dtype=np.int64)[::2]
    valid = np.arange(total_rows, dtype=np.int64)
    out = np.empty(total_rows)
    bins = np.empty(total_rows, dtype=np.int32)

    # Contiguous control: must not raise.
    _gather.gather_multirecord_bins(lay.buf, valid, out, bins, lay.rsr, lay.rsb, lay.nrr, 0)

    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather_multirecord(lay.buf, strided, out, lay.rsr, lay.rsb, lay.nrr, 0)
    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather_multirecord_sorted(lay.buf, strided, out, lay.rsr, lay.rsb, lay.nrr, 0)
    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather_multirecord_bins(lay.buf, strided, out, bins, lay.rsr, lay.rsb, lay.nrr, 0)
    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather_multirecord_withbins(
            lay.buf, valid, out[::-1], bins, lay.rsr, lay.rsb, lay.nrr, 0
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather_bytes(lay.buf, strided, out, 1, 0)
    flat = lay.buf.view(np.float64)
    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather(flat, strided, out, 1, 0)
