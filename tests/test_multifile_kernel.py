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
    _gather.gather_multifile(
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
    _gather.gather_multifile(np.empty(0, dtype=np.int64), out, starts, seg_base)
    assert out.shape == (0,)


def test_unsupported_itemsize_raises():
    starts, seg_base, _oracle, _keep = build_segments([[8], [8]], np.float64)
    idx = np.zeros(4, dtype=np.int64)
    out = np.empty(4, dtype=np.dtype("V3"))  # 3-byte element: unsupported
    with pytest.raises(TypeError, match="element size"):
        _gather.gather_multifile(idx, out, starts, seg_base)


def test_segment_starts_length_mismatch_raises():
    starts, seg_base, _oracle, _keep = build_segments([[8], [8]], np.float64)
    idx = np.zeros(4, dtype=np.int64)
    out = np.empty(4, dtype=np.float64)
    with pytest.raises(ValueError, match="n_segments"):
        _gather.gather_multifile(idx, out, starts[:-1], seg_base)


def test_non_int64_indices_raises():
    starts, seg_base, _oracle, _keep = build_segments([[8], [8]], np.float64)
    out = np.empty(4, dtype=np.float64)
    with pytest.raises(TypeError):
        _gather.gather_multifile(np.zeros(4, dtype=np.int32), out, starts, seg_base)


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
    _gather.gather_multifile_bins(idx, out_a, bins, starts_a, base_a, 0, -1)
    np.testing.assert_array_equal(out_a, oracle_a[idx])
    np.testing.assert_array_equal(out_a, gather(idx, starts_a, base_a, np.float64))
    expected = np.searchsorted(starts_a, idx, side="right").astype(np.int32) - 1
    np.testing.assert_array_equal(bins, expected)

    out_b = np.empty(len(idx), dtype=np.float64)
    _gather.gather_multifile_withbins(idx, out_b, bins, base_b, 0, -1)
    np.testing.assert_array_equal(out_b, oracle_b[idx])


def test_multifile_bins_reuse_small_dtype_and_cap():
    # Bin reuse holds for a sub-word dtype and an explicit thread cap.
    layout = [[16, 16], [24]]
    starts, base_a, oracle_a, _ka = build_segments(layout, np.int16, seed=1)
    _starts_b, base_b, oracle_b, _kb = build_segments(layout, np.int16, seed=2)
    idx = np.arange(int(starts[-1]) - 1, -1, -1, dtype=np.int64)  # reversed, exhaustive
    out_a = np.empty(len(idx), dtype=np.int16)
    bins = np.empty(len(idx), dtype=np.int32)
    _gather.gather_multifile_bins(idx, out_a, bins, starts, base_a, 2, 4)
    out_b = np.empty(len(idx), dtype=np.int16)
    _gather.gather_multifile_withbins(idx, out_b, bins, base_b, 2, 4)
    np.testing.assert_array_equal(out_a, oracle_a[idx])
    np.testing.assert_array_equal(out_b, oracle_b[idx])


def test_multifile_bins_length_mismatch_raises():
    starts, seg_base, _oracle, _keep = build_segments([[8], [8]], np.float64)
    idx = np.zeros(4, dtype=np.int64)
    out = np.empty(4, dtype=np.float64)
    bins = np.empty(3, dtype=np.int32)  # wrong length
    with pytest.raises(ValueError, match="bins"):
        _gather.gather_multifile_bins(idx, out, bins, starts, seg_base)


def test_multifile_bins_wrong_dtype_raises():
    starts, seg_base, _oracle, _keep = build_segments([[8], [8]], np.float64)
    idx = np.zeros(4, dtype=np.int64)
    out = np.empty(4, dtype=np.float64)
    bins = np.empty(4, dtype=np.int64)  # must be int32
    with pytest.raises(TypeError, match="bins"):
        _gather.gather_multifile_bins(idx, out, bins, starts, seg_base)
