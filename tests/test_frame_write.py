"""Tests for the streaming single-record write sink (write_dataset_streaming).

Exercises the sink end to end: round-trip correctness (including byte-identity
with the eager writer), budget-driven batch sizing, common-subexpression reuse
across columns within a batch, atomic-rename behavior on failure, and the eager
validation that runs before any byte is written. Single-record reads take the
memmap fast path, so read-back needs no compiled extension.
"""

from __future__ import annotations

import hashlib
import math
import os

import numpy as np
import pytest

import colstore
from colstore.config import get_default_memory_budget, set_default_memory_budget
from colstore.format import write_dataset, write_dataset_streaming
from colstore.frame import ConstColumn, MemoryColumn


class _CountingColumn(MemoryColumn):
    """A MemoryColumn that counts data reads (the zero-length probe is ignored)."""

    __slots__ = ("reads",)

    def __init__(self, array: np.ndarray) -> None:
        super().__init__(array)
        self.reads = 0

    def _read(self, start: int, stop: int) -> np.ndarray:
        if stop > start:
            self.reads += 1
        return super()._read(start, stop)


class _BoomColumn(MemoryColumn):
    """A MemoryColumn that raises once a batch at or past ``boom_at`` is read."""

    __slots__ = ("_boom_at",)

    def __init__(self, array: np.ndarray, boom_at: int) -> None:
        super().__init__(array)
        self._boom_at = boom_at

    def _read(self, start: int, stop: int) -> np.ndarray:
        if start >= self._boom_at:
            raise RuntimeError("boom")
        return super()._read(start, stop)


def _read_back(path) -> dict[str, np.ndarray]:
    reader = colstore.open(path)
    try:
        return reader.dict()
    finally:
        reader.close()


# -- round-trip correctness --


def test_round_trip_plain_columns(tmp_path):
    rng = np.random.default_rng(0)
    n = 1000
    cols = {
        "price": rng.standard_normal(n).astype(np.float32),
        "qty": rng.integers(0, 1000, n, dtype=np.int32),
        "vol": rng.standard_normal(n).astype(np.float64),
        "name": np.array(["abc"] * n, dtype="S8"),
    }
    path = tmp_path / "out.cstore"
    write_dataset_streaming({k: MemoryColumn(v) for k, v in cols.items()}, n, path)
    out = _read_back(path)
    assert list(out) == list(cols)
    for name, expected in cols.items():
        np.testing.assert_array_equal(out[name], expected)


def test_byte_identical_to_write_dataset(tmp_path):
    rng = np.random.default_rng(1)
    n = 777
    cols = {
        "price": rng.standard_normal(n).astype(np.float32),
        "qty": rng.integers(0, 1000, n, dtype=np.int32),
        "vol": rng.standard_normal(n).astype(np.float64),
        "name": np.array(["abc"] * n, dtype="S8"),
    }
    eager = tmp_path / "eager.cstore"
    streamed = tmp_path / "streamed.cstore"
    write_dataset(cols, eager, batch_size=None, show_progress=False)
    # A tiny budget forces many batches; the bytes on disk must be identical.
    write_dataset_streaming(
        {k: MemoryColumn(v) for k, v in cols.items()}, n, streamed, memory_budget=2048
    )
    assert hashlib.sha256(eager.read_bytes()).hexdigest() == (
        hashlib.sha256(streamed.read_bytes()).hexdigest()
    )


def test_transform_and_const_columns_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    n = 500
    a = rng.standard_normal(n).astype(np.float64)
    b = rng.standard_normal(n).astype(np.float64)
    ca, cb = MemoryColumn(a), MemoryColumn(b)
    specs = {
        "sum2": (ca + cb) * 2.0,
        "ratio": ca / cb,
        "flag": ConstColumn(np.int8(1)),
    }
    path = tmp_path / "t.cstore"
    write_dataset_streaming(specs, n, path, memory_budget=4096)
    out = _read_back(path)
    np.testing.assert_allclose(out["sum2"], (a + b) * 2.0)
    np.testing.assert_allclose(out["ratio"], a / b)
    np.testing.assert_array_equal(out["flag"], np.ones(n, dtype=np.int8))


def test_big_endian_input_normalized(tmp_path):
    n = 64
    native = np.arange(n, dtype=np.float64)
    big = native.astype(">f8")
    path = tmp_path / "be.cstore"
    write_dataset_streaming({"x": MemoryColumn(big)}, n, path, memory_budget=256)
    out = _read_back(path)
    np.testing.assert_array_equal(out["x"], native)
    # Stored little-endian regardless of input byte order.
    reader = colstore.open(path)
    try:
        assert reader.dtypes["x"].str in ("<f8", "=f8")
    finally:
        reader.close()


# -- budget-driven batch sizing --


def test_small_budget_forces_batches(tmp_path):
    n = 100
    col = _CountingColumn(np.arange(n, dtype=np.float64))  # 8 bytes/row, 1 node
    path = tmp_path / "b.cstore"
    write_dataset_streaming({"x": col}, n, path, memory_budget=8 * 10)  # 10 rows/batch
    assert col.reads == 10
    np.testing.assert_array_equal(_read_back(path)["x"], np.arange(n, dtype=np.float64))


def test_large_budget_is_single_pass(tmp_path):
    n = 100
    col = _CountingColumn(np.arange(n, dtype=np.float64))
    path = tmp_path / "b.cstore"
    write_dataset_streaming({"x": col}, n, path, memory_budget=1 << 30)
    assert col.reads == 1


def test_batch_size_counts_intermediate_nodes(tmp_path):
    # a + b has three distinct nodes (a, b, a+b), all float64 -> 24 bytes/row.
    # Budget 288 => 12 rows/batch => 10 batches over 120 rows. If the sizing
    # counted only the two leaves (16 bytes/row) it would pick 18 rows/batch
    # and read each leaf 7 times, so the read count pins the node accounting.
    n = 120
    a = _CountingColumn(np.arange(n, dtype=np.float64))
    b = _CountingColumn(np.arange(n, dtype=np.float64) + 0.5)
    path = tmp_path / "n.cstore"
    write_dataset_streaming({"s": a + b}, n, path, memory_budget=24 * 12)
    assert a.reads == 10
    assert b.reads == 10


def test_budget_defaults_to_config(tmp_path):
    n = 50
    col = _CountingColumn(np.arange(n, dtype=np.float64))
    path = tmp_path / "d.cstore"
    previous = get_default_memory_budget()
    set_default_memory_budget(8 * 5)  # 5 rows/batch -> 10 batches
    try:
        write_dataset_streaming({"x": col}, n, path)  # no explicit budget
    finally:
        set_default_memory_budget(previous)
    assert col.reads == 10


# -- common-subexpression reuse across columns within a batch --


def test_shared_subexpression_read_once_per_batch(tmp_path):
    n = 100
    a = _CountingColumn(np.arange(n, dtype=np.float64))
    b = _CountingColumn(np.arange(n, dtype=np.float64) + 1.0)
    shared = a + b
    specs = {"c1": shared * 2.0, "c2": shared - 1.0}
    path = tmp_path / "cse.cstore"
    # One batch: the shared a+b (and its leaves) is computed once across both
    # output columns, not once per column.
    write_dataset_streaming(specs, n, path, memory_budget=1 << 30)
    assert a.reads == 1
    assert b.reads == 1
    out = _read_back(path)
    base = np.arange(n, dtype=np.float64) * 2.0 + 1.0  # a + b
    np.testing.assert_allclose(out["c1"], base * 2.0)
    np.testing.assert_allclose(out["c2"], base - 1.0)


# -- atomicity --


def test_failure_leaves_no_destination(tmp_path):
    n = 100
    path = tmp_path / "fail.cstore"
    specs = {"x": _BoomColumn(np.arange(n, dtype=np.float64), boom_at=40)}
    before = set(os.listdir(tmp_path))
    with pytest.raises(RuntimeError, match="boom"):
        write_dataset_streaming(specs, n, path, memory_budget=8 * 10)
    assert not path.exists()
    assert set(os.listdir(tmp_path)) == before  # temp file cleaned up


def test_failure_preserves_existing_destination(tmp_path):
    n = 100
    path = tmp_path / "keep.cstore"
    path.write_bytes(b"sentinel-contents")
    specs = {"x": _BoomColumn(np.arange(n, dtype=np.float64), boom_at=40)}
    with pytest.raises(RuntimeError, match="boom"):
        write_dataset_streaming(specs, n, path, memory_budget=8 * 10)
    assert path.read_bytes() == b"sentinel-contents"
    assert set(os.listdir(tmp_path)) == {"keep.cstore"}  # no stray temp


# -- eager validation, before any byte is written --


def test_length_mismatch_rejected(tmp_path):
    path = tmp_path / "x.cstore"
    with pytest.raises(ValueError, match="does not match"):
        write_dataset_streaming({"x": MemoryColumn(np.arange(3))}, 5, path)
    assert not path.exists()


def test_length_one_rejected(tmp_path):
    path = tmp_path / "x.cstore"
    with pytest.raises(ValueError, match="does not match"):
        write_dataset_streaming({"x": MemoryColumn(np.arange(1))}, 8, path)
    assert not path.exists()


def test_object_dtype_rejected(tmp_path):
    path = tmp_path / "x.cstore"
    objects = np.array([object(), object(), object()], dtype=object)
    with pytest.raises(TypeError, match="object dtype"):
        write_dataset_streaming({"x": MemoryColumn(objects)}, 3, path)
    assert not path.exists()


def test_empty_specs_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty column mapping"):
        write_dataset_streaming({}, 0, tmp_path / "x.cstore")


def test_zero_rows(tmp_path):
    path = tmp_path / "z.cstore"
    write_dataset_streaming({"x": MemoryColumn(np.arange(0, dtype=np.int64))}, 0, path)
    reader = colstore.open(path)
    try:
        assert reader.n_rows == 0
        assert reader.dict()["x"].shape == (0,)
    finally:
        reader.close()


def test_batch_count_matches_ceiling(tmp_path):
    # Cross-check the documented batch count: ceil(n / rows_per_batch).
    n = 95
    rows_per_batch = 10
    col = _CountingColumn(np.arange(n, dtype=np.float64))
    path = tmp_path / "c.cstore"
    write_dataset_streaming({"x": col}, n, path, memory_budget=8 * rows_per_batch)
    assert col.reads == math.ceil(n / rows_per_batch)


def test_fusible_passthroughs_selects_unshared_native_columns(tmp_path):
    # The streaming sink fills a passthrough column straight from its source only
    # when its expression is a bare native column whose leaf no other column
    # reads -- so a native read shared with a transform stays on the memoized
    # path and is read once, not once per consumer.
    from colstore.frame import NativeColumn, fusible_passthroughs

    path = tmp_path / "s.cstore"
    colstore.store(
        {"a": np.arange(5, dtype=np.int64), "b": np.arange(5, dtype=np.float64)},
        path,
        show_progress=False,
    ).close()
    with colstore.open(path) as store:
        a = NativeColumn(store, "a")
        b = NativeColumn(store, "b")
        specs = {"a": a, "a_plus": a + 1, "b": b}  # a feeds a transform; b is plain
        fusible = fusible_passthroughs(specs)
        assert set(fusible) == {"b"}  # a excluded (shared leaf), a_plus excluded (transform)
        assert fusible["b"] is b
