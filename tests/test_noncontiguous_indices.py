"""Non-contiguous fancy-selector regression tests.

A strided index view (``rows[::2]``, ``rows[::-1]``) used to reach the C++
kernels unchanged, because ``astype(np.int64, copy=False)`` preserves
strides when the dtype is already int64 -- and the kernels interpret the
array as a contiguous ``int64_t*``. Positive strides read the wrong
positions (silently wrong values); negative strides read out of bounds
(observed segfault). The fix is layered: the view boundary normalizes with
``np.ascontiguousarray`` (no-op for already-contiguous arrays), the
reader's own conversion points do the same defensively, and every Cython
entry point validates contiguity of all pointer-interpreted arrays and
raises rather than misread.
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import _gather
from colstore.kernels import cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")


@pytest.fixture()
def multi_record_store(tmp_path):
    rng = np.random.default_rng(0)
    total = 5_000
    full = {
        "a": rng.standard_normal(total),
        "b": rng.integers(-(2**20), 2**20, total).astype(np.int32),
        "c": rng.standard_normal(total).astype(np.float32),
    }
    path = tmp_path / "m.cstore"
    with colstore.create(path) as writer:
        for offset in range(0, total, 250):  # 20 records
            writer.write({k: v[offset : offset + 250] for k, v in full.items()})
    return path, full, total


def _strided_selectors(total: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    unsorted_base = rng.permutation(total).astype(np.int64)
    return {
        "sorted_step2": np.arange(total, dtype=np.int64)[::2],
        "sorted_step7": np.arange(total, dtype=np.int64)[::7],
        "reversed": np.arange(total, dtype=np.int64)[::-1],  # was a segfault
        "unsorted_step2": unsorted_base[::2],
        "unsorted_negative_step": unsorted_base[::-3],
        "int32_step2": np.arange(total, dtype=np.int32)[::2],
    }


def test_multi_record_strided_selectors(multi_record_store):
    path, full, total = multi_record_store
    dataset = colstore.open(path)
    for name, selector in _strided_selectors(total).items():
        expected = selector.astype(np.int64)
        for column in full:
            got = dataset[selector, column].array()
            assert np.array_equal(got, full[column][expected]), (name, column)
    dataset.close()


def test_multi_record_strided_multicolumn_bin_reuse(multi_record_store):
    path, full, total = multi_record_store
    dataset = colstore.open(path)
    selector = np.random.default_rng(2).permutation(total).astype(np.int64)[::2]
    table = dataset[selector, list(full)].dict()
    for column in full:
        assert np.array_equal(table[column], full[column][selector]), column
    dataset.close()


def test_single_record_strided_selectors(tmp_path):
    rng = np.random.default_rng(3)
    total = 3_000
    full = {"a": rng.standard_normal(total)}
    path = tmp_path / "s.cstore"
    colstore.store(full, path, show_progress=False)
    dataset = colstore.open(path)
    for name, selector in _strided_selectors(total).items():
        expected = selector.astype(np.int64)
        got = dataset[selector, "a"].array()
        assert np.array_equal(got, full["a"][expected]), name
    dataset.close()


def test_assessment_reproducer_verbatim(tmp_path):
    path = tmp_path / "x.cstore"
    a = np.arange(20, dtype=np.int64)
    with colstore.create(path) as writer:
        writer.write({"a": a[:10]})
        writer.write({"a": a[10:]})
    dataset = colstore.open(path)
    idx = np.arange(20, dtype=np.int64)[::2]
    assert not idx.flags.c_contiguous
    assert np.array_equal(dataset[idx, "a"].array(), a[idx])
    dataset.close()


def test_kernel_entries_reject_strided_arrays():
    # Direct-API backstop: every pointer-interpreting entry must refuse
    # strided arrays rather than misread them.
    n_records, rows = 4, 10
    total_rows = n_records * rows
    nrr = np.full(n_records, rows, dtype=np.int64)
    rsr = np.zeros(n_records + 1, dtype=np.int64)
    rsr[1:] = np.cumsum(nrr)
    rsb = np.arange(n_records, dtype=np.int64) * (rows * 8)
    buf = np.zeros(total_rows * 8, dtype=np.uint8)
    strided = np.arange(2 * total_rows, dtype=np.int64)[::2]
    valid = np.arange(total_rows, dtype=np.int64)
    out = np.empty(total_rows)
    bins = np.empty(total_rows, dtype=np.int32)

    # Contiguous control: must not raise.
    _gather.gather_multirecord_bins(buf, valid, out, bins, rsr, rsb, nrr, 0)

    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather_multirecord(buf, strided, out, rsr, rsb, nrr, 0)
    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather_multirecord_sorted(buf, strided, out, rsr, rsb, nrr, 0)
    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather_multirecord_bins(buf, strided, out, bins, rsr, rsb, nrr, 0)
    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather_multirecord_withbins(buf, valid, out[::-1], bins, rsr, rsb, nrr, 0)
    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather_bytes(buf, strided, out, 1, 0)
    flat = buf.view(np.float64)
    with pytest.raises(ValueError, match="C-contiguous"):
        _gather.gather(flat, strided, out, 1, 0)


def test_boolean_mask_selectors_unaffected(multi_record_store):
    path, full, total = multi_record_store
    dataset = colstore.open(path)
    mask = np.zeros(total, dtype=bool)
    mask[::3] = True
    got = dataset[mask, "a"].array()
    assert np.array_equal(got, full["a"][mask])
    dataset.close()
