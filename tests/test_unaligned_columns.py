"""Misaligned-column regression tests for the gather kernels.

Record bodies are packed with no inter-column padding, so a column's start
is naturally aligned only if every preceding column's byte count is a
multiple of its alignment -- an odd-length ``int8`` column followed by a
``float64`` column puts the f8 column at an odd byte address. The C++
kernels must load such sources with alignment-safe ``memcpy`` loads:
dereferencing a misaligned typed pointer is undefined behavior, even where
x86 tolerates it. These tests pin the *semantics* on every read path that
reaches the kernels; the UB itself was verified by compiling the kernels
under ``-fsanitize=alignment`` (pre-fix traps, post-fix clean -- review
artifact, not CI).
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore.kernels import cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")


@pytest.fixture()
def misaligned_store(tmp_path):
    """Multi-record store whose f8/f4/i4 columns start at odd addresses.

    The leading column is int8 with an odd per-record row count (7), so every
    subsequent column's prefix is odd within the record body.
    """
    rng = np.random.default_rng(3)
    n_records, rows = 12, 7
    total = n_records * rows
    full = {
        "pad": rng.integers(-100, 100, total).astype(np.int8),
        "f8": rng.standard_normal(total),
        "f4": rng.standard_normal(total).astype(np.float32),
        "i4": rng.integers(-(2**20), 2**20, total).astype(np.int32),
    }
    path = tmp_path / "mis.cstore"
    with colstore.create(path) as writer:
        for r in range(n_records):
            writer.write({k: v[r * rows : (r + 1) * rows] for k, v in full.items()})
    return path, full, total


def test_misaligned_columns_are_actually_misaligned(misaligned_store):
    path, _, _ = misaligned_store
    dataset = colstore.open(path)
    offset = int(dataset._record_starts_bytes[0]) + int(dataset._column_prefix_bytes["f8"]) * int(
        dataset._n_rows_per_record[0]
    )
    assert offset % 8 != 0, "fixture no longer exercises misalignment"
    dataset.close()


def test_misaligned_unsorted_fancy_single_column(misaligned_store):
    path, full, total = misaligned_store
    dataset = colstore.open(path)
    indices = np.random.default_rng(1).integers(0, total, size=300).astype(np.int64)
    for name in ("f8", "f4", "i4", "pad"):
        assert np.array_equal(dataset[indices, name].array(), full[name][indices]), name
    dataset.close()


def test_misaligned_multicolumn_bin_reuse_route(misaligned_store):
    path, full, total = misaligned_store
    dataset = colstore.open(path)
    indices = np.random.default_rng(2).integers(0, total, size=300).astype(np.int64)
    table = dataset[indices, ["f8", "f4", "i4"]].dict()
    for name, values in table.items():
        assert np.array_equal(values, full[name][indices]), name
    dataset.close()


def test_misaligned_sorted_fancy_and_contiguous(misaligned_store):
    path, full, total = misaligned_store
    dataset = colstore.open(path)
    sorted_indices = np.sort(np.random.default_rng(4).integers(0, total, size=200).astype(np.int64))
    assert np.array_equal(dataset[sorted_indices, "f8"].array(), full["f8"][sorted_indices])
    assert np.array_equal(dataset[5:60, "f8"].array(), full["f8"][5:60])
    dataset.close()


def test_misaligned_single_record_store(tmp_path):
    rng = np.random.default_rng(5)
    full = {"pad": rng.integers(0, 9, 1001).astype(np.int8), "f8": rng.standard_normal(1001)}
    path = tmp_path / "sr.cstore"
    colstore.store(full, path, show_progress=False)
    dataset = colstore.open(path)
    indices = rng.integers(0, 1001, size=257).astype(np.int64)
    assert np.array_equal(dataset[indices, "f8"].array(), full["f8"][indices])
    dataset.close()
