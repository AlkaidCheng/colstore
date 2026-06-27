"""Contract of the fused multi-file gather kernel (direct ``_gather`` calls).

A dataset of several files is one global row space decomposed into segments --
one record of one file each -- with ``segment_base[s]`` an absolute byte
address such that global row ``idx`` of segment ``s`` reads at
``segment_base[s] + idx * itemsize``. The kernel must equal NumPy indexing of
the concatenated rows for every supported dtype, across files that each hold
one or more records (segments at byte offsets inside a shared per-file buffer),
including empty files, record/file boundaries, duplicates, reversal, and a
forced thread cap; and it must validate its inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from colstore import _gather
from colstore.kernels import cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")


def build_segments(file_record_rows, dtype, seed=7):
    """Build a synthetic segment table from per-file, per-record row counts.

    ``file_record_rows`` is one entry per file, each a list of that file's
    per-record row counts. Each file is a single contiguous buffer; its records
    are segments at increasing byte offsets within it. ``seed`` varies the data
    (not the layout), so two calls with the same ``file_record_rows`` model two
    columns: identical ``segment_starts_rows``, different ``segment_base`` and
    data. Returns ``(segment_starts_rows, segment_base, oracle, keepalive)``
    where ``oracle`` is the global-row concatenation and ``keepalive`` pins the
    buffers alive.
    """
    itemsize = np.dtype(dtype).itemsize
    rng = np.random.default_rng(seed)
    starts = [0]
    seg_base: list[int] = []
    oracle_blocks = []
    keepalive = []
    g = 0
    for records in file_record_rows:
        total_file_rows = sum(records)
        buf = (rng.integers(0, 10_000, size=max(total_file_rows, 1))).astype(dtype)
        keepalive.append(buf)
        base_addr = buf.ctypes.data
        row_off = 0
        for r in records:
            # Segment covers global rows [g, g + r); its row 0 sits at
            # buffer row row_off, so base = addr(row_off) - g * itemsize folds
            # the within-file offset and the global start out of the address.
            seg_base.append(base_addr + (row_off - g) * itemsize)
            oracle_blocks.append(buf[row_off : row_off + r])
            g += r
            row_off += r
            starts.append(g)
    oracle = np.concatenate(oracle_blocks) if oracle_blocks else np.empty(0, dtype=dtype)
    return (
        np.asarray(starts, dtype=np.int64),
        np.asarray(seg_base, dtype=np.int64),
        oracle,
        keepalive,
    )


def gather(indices, starts, seg_base, dtype, thread_cap=0, prefetch=-1):
    out = np.empty(len(indices), dtype=dtype)
    _gather.gather_segment(
        np.asarray(indices, dtype=np.int64), out, starts, seg_base, thread_cap, prefetch
    )
    return out


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8])
@pytest.mark.parametrize("thread_cap", [1, 4])
@pytest.mark.parametrize("prefetch", [0, 8, 128])
def test_matches_oracle_across_dtypes(dtype, thread_cap, prefetch):
    # Several files, several records each, an empty file in the middle, uneven.
    starts, seg_base, oracle, _keep = build_segments([[30, 0, 20], [], [40], [10, 15]], dtype)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, oracle.size, size=4000).astype(np.int64)
    out = gather(idx, starts, seg_base, dtype, thread_cap, prefetch)
    np.testing.assert_array_equal(out, oracle[idx])


def test_edge_indices_match_oracle():
    dtype = np.float64
    starts, seg_base, oracle, _keep = build_segments([[40], [55], [17], [33]], dtype)
    total = int(oracle.size)
    boundaries = [b for b in starts[1:-1]]  # internal file/segment boundaries
    idx = np.array(
        [0, total - 1, total - 1, *boundaries, *[b - 1 for b in boundaries], 1, 1, 1],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(gather(idx, starts, seg_base, dtype), oracle[idx])
    # Full reversal and an already-sorted run.
    rev = np.arange(total - 1, -1, -1, dtype=np.int64)
    np.testing.assert_array_equal(gather(rev, starts, seg_base, dtype), oracle[rev])
    asc = np.arange(total, dtype=np.int64)
    np.testing.assert_array_equal(gather(asc, starts, seg_base, dtype), oracle)


def test_single_file_single_record():
    dtype = np.int32
    starts, seg_base, oracle, _keep = build_segments([[64]], dtype)
    rng = np.random.default_rng(3)
    idx = rng.integers(0, oracle.size, size=200).astype(np.int64)
    np.testing.assert_array_equal(gather(idx, starts, seg_base, dtype), oracle[idx])


def test_large_selection_matches_oracle():
    # Past PARALLEL_THRESHOLD so the threaded path is taken where the runtime
    # has the cores; output[i] depends only on i, so the result is identical.
    dtype = np.float64
    starts, seg_base, oracle, _keep = build_segments([[200_000], [300_000], [250_000]], dtype)
    rng = np.random.default_rng(5)
    idx = rng.integers(0, oracle.size, size=400_000).astype(np.int64)
    np.testing.assert_array_equal(gather(idx, starts, seg_base, dtype, thread_cap=4), oracle[idx])


def test_empty_indices_is_noop():
    dtype = np.float64
    starts, seg_base, _oracle, _keep = build_segments([[10], [10]], dtype)
    out = np.empty(0, dtype=dtype)
    _gather.gather_segment(np.empty(0, dtype=np.int64), out, starts, seg_base)
    assert out.shape == (0,)


@pytest.mark.parametrize("dtype", ["<U3", "|S5"])
def test_wide_itemsize_gathers(dtype):
    """Elements outside {1,2,4,8} bytes gather via the generic memcpy path."""
    dtype = np.dtype(dtype)
    starts, seg_base, oracle, _keep = build_segments([[8], [8]], dtype)
    idx = np.array([0, 15, 7, 3, 8, 1], dtype=np.int64)
    out = np.empty(len(idx), dtype=dtype)
    _gather.gather_segment(idx, out, starts, seg_base)
    assert np.array_equal(out.view(np.uint8), np.ascontiguousarray(oracle[idx]).view(np.uint8))


def test_segment_starts_length_mismatch_raises():
    starts, seg_base, _oracle, _keep = build_segments([[8], [8]], np.float64)
    idx = np.zeros(4, dtype=np.int64)
    out = np.empty(4, dtype=np.float64)
    with pytest.raises(ValueError, match="n_segments"):
        _gather.gather_segment(idx, out, starts[:-1], seg_base)


def test_non_int64_indices_raises():
    starts, seg_base, _oracle, _keep = build_segments([[8], [8]], np.float64)
    out = np.empty(4, dtype=np.float64)
    with pytest.raises(TypeError):
        _gather.gather_segment(np.zeros(4, dtype=np.int32), out, starts, seg_base)


# ---- Segment-bin reuse across columns --------------------------------------


def test_multifile_bins_reuse_matches_plain_and_oracle():
    # Two columns share one segmentation (same row layout, different data). The
    # bins kernel gathers the first column and records the segment ids; the
    # withbins kernel reuses those ids -- no search -- to gather the second from
    # its own bases. Both must equal their oracle, the bins kernel must equal the
    # plain kernel, and the recorded ids must be the true segment of each index.
    layout = [[40, 30, 30], [50], [0], [20, 25]]  # multi-record, single, empty
    starts_a, base_a, oracle_a, _ka = build_segments(layout, np.float64, seed=7)
    starts_b, base_b, oracle_b, _kb = build_segments(layout, np.float64, seed=11)
    np.testing.assert_array_equal(starts_a, starts_b)  # column-independent layout

    total = int(starts_a[-1])
    rng = np.random.default_rng(3)
    idx = rng.integers(0, total, size=2000, dtype=np.int64)
    idx[::9] = idx[0]  # duplicates

    out_a = np.empty(len(idx), dtype=np.float64)
    bins = np.empty(len(idx), dtype=np.int32)
    _gather.gather_segment_bins(idx, out_a, bins, starts_a, base_a, 0, -1)
    np.testing.assert_array_equal(out_a, oracle_a[idx])
    np.testing.assert_array_equal(out_a, gather(idx, starts_a, base_a, np.float64))
    expected = np.searchsorted(starts_a, idx, side="right").astype(np.int32) - 1
    np.testing.assert_array_equal(bins, expected)

    out_b = np.empty(len(idx), dtype=np.float64)
    _gather.gather_segment_withbins(idx, out_b, bins, base_b, 0, -1)
    np.testing.assert_array_equal(out_b, oracle_b[idx])


def test_multifile_bins_reuse_small_dtype_and_cap():
    # Bin reuse holds for a sub-word dtype and an explicit thread cap.
    layout = [[16, 16], [24]]
    starts, base_a, oracle_a, _ka = build_segments(layout, np.int16, seed=1)
    _starts_b, base_b, oracle_b, _kb = build_segments(layout, np.int16, seed=2)
    idx = np.arange(int(starts[-1]) - 1, -1, -1, dtype=np.int64)  # reversed, exhaustive
    out_a = np.empty(len(idx), dtype=np.int16)
    bins = np.empty(len(idx), dtype=np.int32)
    _gather.gather_segment_bins(idx, out_a, bins, starts, base_a, 2, 4)
    out_b = np.empty(len(idx), dtype=np.int16)
    _gather.gather_segment_withbins(idx, out_b, bins, base_b, 2, 4)
    np.testing.assert_array_equal(out_a, oracle_a[idx])
    np.testing.assert_array_equal(out_b, oracle_b[idx])


def test_multifile_bins_length_mismatch_raises():
    starts, seg_base, _oracle, _keep = build_segments([[8], [8]], np.float64)
    idx = np.zeros(4, dtype=np.int64)
    out = np.empty(4, dtype=np.float64)
    bins = np.empty(3, dtype=np.int32)  # wrong length
    with pytest.raises(ValueError, match="bins"):
        _gather.gather_segment_bins(idx, out, bins, starts, seg_base)


def test_multifile_bins_wrong_dtype_raises():
    starts, seg_base, _oracle, _keep = build_segments([[8], [8]], np.float64)
    idx = np.zeros(4, dtype=np.int64)
    out = np.empty(4, dtype=np.float64)
    bins = np.empty(4, dtype=np.int64)  # must be int32
    with pytest.raises(TypeError, match="bins"):
        _gather.gather_segment_bins(idx, out, bins, starts, seg_base)


# ---- Sorted cursor walk ----------------------------------------------------


@pytest.mark.parametrize("cap", [0, 1, 2, 4])
def test_multifile_sorted_matches_plain_and_oracle(cap):
    # Non-decreasing indices through the cursor walk must equal the searching
    # kernel and the oracle, with duplicates, across many segments (including an
    # empty one), serial and threaded.
    layout = [[40, 30, 30], [50], [0], [20, 25], [60]]
    starts, base, oracle, _keep = build_segments(layout, np.float64, seed=5)
    total = int(starts[-1])
    idx = np.sort(np.random.default_rng(2).integers(0, total, size=3000)).astype(np.int64)
    out = np.empty(len(idx), dtype=np.float64)
    _gather.gather_segment_sorted(idx, out, starts, base, cap, -1)
    np.testing.assert_array_equal(out, oracle[idx])
    np.testing.assert_array_equal(out, gather(idx, starts, base, np.float64, thread_cap=cap))


def test_multifile_sorted_exhaustive_ascending():
    # Every row in order crosses each segment boundary exactly once.
    layout = [[16, 16, 16], [8], [24]]
    starts, base, oracle, _keep = build_segments(layout, np.int32, seed=6)
    idx = np.arange(int(starts[-1]), dtype=np.int64)
    out = np.empty(len(idx), dtype=np.int32)
    _gather.gather_segment_sorted(idx, out, starts, base, 2, 4)
    np.testing.assert_array_equal(out, oracle[idx])


def test_multifile_sorted_duplicates_and_single_segment():
    # Heavy duplicates (the cursor must hold) within a single segment.
    starts, base, oracle, _keep = build_segments([[100]], np.float64, seed=8)
    idx = np.sort(np.array([0, 0, 0, 5, 5, 99, 99, 50], dtype=np.int64))
    out = np.empty(len(idx), dtype=np.float64)
    _gather.gather_segment_sorted(idx, out, starts, base, 0, -1)
    np.testing.assert_array_equal(out, oracle[idx])


@pytest.mark.parametrize("cap", [0, 4])
def test_segment_sorted_is_order_robust_and_in_bounds(cap):
    # The cursor kernel is the fast path for sorted indices but stays correct and
    # in-bounds for ANY order: a backward step re-locates by binary search rather
    # than reading past the segment (a forward-only cursor would go out of bounds
    # on a descending step). This is the memory-safety guard that lets the reader
    # treat its sortedness hint as advisory, never a safety assertion.
    layout = [[40, 30, 30], [50], [0], [20, 25], [60]]
    starts, base, oracle, _keep = build_segments(layout, np.float64, seed=5)
    total = int(starts[-1])
    rng = np.random.default_rng(9)
    patterns = {
        "descending": np.arange(total - 1, -1, -1, dtype=np.int64),
        "random_unsorted": rng.integers(0, total, size=4000).astype(np.int64),
        "shuffled_all": rng.permutation(total).astype(np.int64),
        "adversarial_jumps": np.array(
            [0, total - 1, 1, total - 2, total // 2, 2, total - 1, 0], dtype=np.int64
        ),
    }
    for name, idx in patterns.items():
        out = np.empty(idx.size, dtype=np.float64)
        _gather.gather_segment_sorted(idx, out, starts, base, cap, 8)
        np.testing.assert_array_equal(out, oracle[idx], err_msg=f"{name}/cap={cap}")


# ---- Uniform-grid division-binning -----------------------------------------
# When every segment holds the same row count (the global-last may be partial),
# the per-index binary search collapses to s = idx / rows_per_segment. The kernel
# must equal the searching kernel and the oracle across files/records, partial
# tails, single-byte dtypes, and thread caps; the bins variant must record the
# true segment for cross-column reuse.


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8])
@pytest.mark.parametrize("thread_cap", [1, 4])
@pytest.mark.parametrize("prefetch", [0, 8, 128])
def test_segment_uniform_matches_searching_and_oracle(dtype, thread_cap, prefetch):
    # Uniform grid spanning several files and several records per file.
    rows = 200
    starts, base, oracle, _keep = build_segments([[rows, rows, rows], [rows, rows], [rows]], dtype)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, oracle.size, size=5000).astype(np.int64)
    out_u = np.empty(idx.size, dtype=dtype)
    _gather.gather_segment_uniform(idx, out_u, rows, base, thread_cap, prefetch)
    out_g = np.empty(idx.size, dtype=dtype)
    _gather.gather_segment(idx, out_g, starts, base, thread_cap, prefetch)
    np.testing.assert_array_equal(out_u, out_g)
    np.testing.assert_array_equal(out_u, oracle[idx])


def test_segment_uniform_partial_global_tail():
    # The final segment is smaller than rows_per_segment -- still a uniform grid.
    rows, tail = 256, 73
    _starts, base, oracle, _keep = build_segments(
        [[rows, rows], [rows], [tail]], np.float64, seed=4
    )
    total = int(oracle.size)
    patterns = {
        "boundaries": np.array([0, rows - 1, rows, total - 1, total - 1], dtype=np.int64),
        "all_in_tail": np.random.default_rng(5).integers(3 * rows, total, 300).astype(np.int64),
        "every_row": np.arange(total, dtype=np.int64),
        "single": np.array([total // 2], dtype=np.int64),
    }
    for name, idx in patterns.items():
        out = np.empty(idx.size, dtype=np.float64)
        _gather.gather_segment_uniform(idx, out, rows, base, 2, 8)
        np.testing.assert_array_equal(out, oracle[idx], err_msg=name)


def test_segment_uniform_bins_reuse_matches_oracle():
    # Two columns share one uniform grid: the bins kernel divides once and records
    # the segment; the withbins kernel reuses it for the second column.
    layout = [[64, 64], [64], [64, 64]]  # 5 segments, all 64 rows
    rows = 64
    starts_a, base_a, oracle_a, _ka = build_segments(layout, np.float64, seed=7)
    _starts_b, base_b, oracle_b, _kb = build_segments(layout, np.float64, seed=11)
    rng = np.random.default_rng(3)
    idx = rng.integers(0, int(starts_a[-1]), size=2000, dtype=np.int64)
    idx[::9] = idx[0]  # duplicates

    out_a = np.empty(len(idx), dtype=np.float64)
    bins = np.empty(len(idx), dtype=np.int32)
    _gather.gather_segment_uniform_bins(idx, out_a, bins, rows, base_a, 0, -1)
    np.testing.assert_array_equal(out_a, oracle_a[idx])
    np.testing.assert_array_equal(bins, (idx // rows).astype(np.int32))

    out_b = np.empty(len(idx), dtype=np.float64)
    _gather.gather_segment_withbins(idx, out_b, bins, base_b, 0, -1)
    np.testing.assert_array_equal(out_b, oracle_b[idx])


def test_segment_uniform_validates_inputs():
    _starts, base, _oracle, _keep = build_segments([[8], [8]], np.float64)
    idx = np.zeros(4, dtype=np.int64)
    out = np.empty(4, dtype=np.float64)
    bins = np.empty(4, dtype=np.int32)
    with pytest.raises(ValueError, match="rows_per_segment"):
        _gather.gather_segment_uniform(idx, out, 0, base)
    with pytest.raises(TypeError, match="int64"):
        _gather.gather_segment_uniform(idx.astype(np.int32), out, 8, base)
    with pytest.raises(ValueError, match="length"):
        _gather.gather_segment_uniform(idx, np.empty(2), 8, base)
    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather_segment_uniform(idx, out, 8, np.repeat(base, 2)[::2])
    with pytest.raises(TypeError, match="int32"):
        _gather.gather_segment_uniform_bins(idx, out, bins.astype(np.int64), 8, base)
    with pytest.raises(ValueError, match="lengths"):
        _gather.gather_segment_uniform_bins(idx, out, np.empty(2, dtype=np.int32), 8, base)


# ---- parallel_copy_runs ------------------------------------------------


def _runs(sources, dst_rows, itemsize):
    """Run arrays placing each source array at its destination row offset."""
    src = np.array([s.ctypes.data for s in sources], dtype=np.int64)
    dst = np.array([row * itemsize for row in dst_rows], dtype=np.int64)
    lengths = np.array([s.size * itemsize for s in sources], dtype=np.int64)
    return src, dst, lengths


@pytest.mark.parametrize("cap", [0, 1, 2, 4])
def test_parallel_copy_runs_matches_concatenation(cap):
    rng = np.random.default_rng(11)
    blocks = [rng.integers(0, 1000, size=n).astype(np.float64) for n in (50, 1, 200, 99, 7)]
    offsets = np.cumsum([0, *(b.size for b in blocks)])
    out = np.empty(int(offsets[-1]), dtype=np.float64)
    src, dst, lengths = _runs(blocks, offsets[:-1], 8)
    _gather.parallel_copy_runs(out, src, dst, lengths, cap)
    np.testing.assert_array_equal(out, np.concatenate(blocks))


def test_parallel_copy_runs_leaves_gaps_untouched():
    # Runs need not tile the output: a left-out region keeps its prior contents.
    a = np.arange(10, dtype=np.float64)
    b = np.arange(100, 105, dtype=np.float64)
    out = np.full(20, -1.0, dtype=np.float64)
    src, dst, lengths = _runs([a, b], [0, 15], 8)  # gap at rows [10, 15)
    _gather.parallel_copy_runs(out, src, dst, lengths, 4)
    expected = np.full(20, -1.0)
    expected[0:10] = a
    expected[15:20] = b
    np.testing.assert_array_equal(out, expected)


def test_parallel_copy_runs_large_multithread():
    rng = np.random.default_rng(3)
    blocks = [rng.integers(0, 1 << 20, size=n).astype(np.int32) for n in (2_000_000, 1_500_000)]
    offsets = np.cumsum([0, *(b.size for b in blocks)])
    out = np.empty(int(offsets[-1]), dtype=np.int32)
    src, dst, lengths = _runs(blocks, offsets[:-1], 4)
    _gather.parallel_copy_runs(out, src, dst, lengths, 4)
    np.testing.assert_array_equal(out, np.concatenate(blocks))


def test_parallel_copy_runs_empty_is_noop():
    out = np.array([1.0, 2.0])
    empty = np.empty(0, dtype=np.int64)
    _gather.parallel_copy_runs(out, empty, empty, empty, 0)
    np.testing.assert_array_equal(out, [1.0, 2.0])


def test_parallel_copy_runs_length_mismatch_raises():
    out = np.empty(4, dtype=np.float64)
    src = np.array([out.ctypes.data], dtype=np.int64)
    with pytest.raises(ValueError, match="equal length"):
        _gather.parallel_copy_runs(
            out, src, np.array([0, 0], dtype=np.int64), np.array([32], dtype=np.int64), 0
        )


def test_parallel_copy_runs_non_int64_raises():
    # The dtype guard rejects the int32 address array before any dereference.
    out = np.empty(4, dtype=np.float64)
    bad = np.array([0], dtype=np.int32)
    with pytest.raises(TypeError, match="int64"):
        _gather.parallel_copy_runs(
            out, bad, np.array([0], dtype=np.int64), np.array([32], dtype=np.int64), 0
        )


# ---- interleave_records ------------------------------------------------


def _interleave(out, n_rows, columns, record_dtype, names, cap):
    """Drive the kernel from a list of column arrays into a record array."""
    _gather.interleave_records(
        out,
        record_dtype.itemsize,
        n_rows,
        np.array([c.ctypes.data for c in columns], dtype=np.int64),
        np.array([c.dtype.itemsize for c in columns], dtype=np.int64),
        np.array([record_dtype.fields[name][1] for name in names], dtype=np.int64),
        cap,
    )


@pytest.mark.parametrize("cap", [0, 1, 4])
def test_interleave_records_matches_numpy(cap):
    # Mixed itemsizes exercise every branch of the field-copy switch (8/4/2/1).
    rng = np.random.default_rng(5)
    n = 10_000
    columns = [
        rng.integers(0, 1000, n).astype(np.float64),
        rng.integers(0, 1000, n).astype(np.int32),
        rng.integers(0, 1000, n).astype(np.int16),
        rng.integers(0, 1000, n).astype(np.int8),
        rng.random(n),
    ]
    names = [f"c{i}" for i in range(len(columns))]
    record_dtype = np.dtype([(name, col.dtype) for name, col in zip(names, columns, strict=True)])
    oracle = np.empty(n, dtype=record_dtype)
    for name, col in zip(names, columns, strict=True):
        oracle[name] = col

    out = np.empty(n, dtype=record_dtype)
    _interleave(out, n, columns, record_dtype, names, cap)
    np.testing.assert_array_equal(out, oracle)


def test_interleave_records_empty_is_noop():
    record_dtype = np.dtype([("a", np.float64), ("b", np.int32)])
    out = np.zeros(0, dtype=record_dtype)
    empty = np.empty(0, dtype=np.int64)
    _gather.interleave_records(out, record_dtype.itemsize, 0, empty, empty, empty, 0)
    assert out.shape == (0,)


def test_interleave_records_length_mismatch_raises():
    record_dtype = np.dtype([("a", np.float64)])
    out = np.empty(4, dtype=record_dtype)
    a = np.arange(4, dtype=np.float64)
    with pytest.raises(ValueError, match="equal length"):
        _gather.interleave_records(
            out,
            record_dtype.itemsize,
            4,
            np.array([a.ctypes.data], dtype=np.int64),
            np.array([8, 8], dtype=np.int64),
            np.array([0], dtype=np.int64),
            0,
        )


def test_interleave_records_non_int64_raises():
    record_dtype = np.dtype([("a", np.float64)])
    out = np.empty(4, dtype=record_dtype)
    with pytest.raises(TypeError, match="int64"):
        _gather.interleave_records(
            out,
            record_dtype.itemsize,
            4,
            np.array([0], dtype=np.int32),
            np.array([8], dtype=np.int64),
            np.array([0], dtype=np.int64),
            0,
        )
