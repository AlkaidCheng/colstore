"""Tests for the append-shard / streaming-concat writer (colstore.shards).

A managed dataset is a directory of ``.cstore`` shards. ``append`` writes one new
shard; ``appender`` streams batches and rolls shards at a size budget; ``open(dir)``
reads them in order. Shards are immutable and committed atomically; one writer
holds the directory lock at a time.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

import colstore
from colstore import col


def _shards(directory):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(str(directory), "*.cstore")))


# ---- one-shot append + directory read --------------------------------------


def test_append_then_open_concatenates_in_order(tmp_path):
    for i in range(3):
        path = colstore.append(tmp_path, {"x": np.arange(i * 10, i * 10 + 10, dtype=np.int64)})
        assert path.name == f"shard_{i:05d}.cstore"
    ds = colstore.open(tmp_path)
    assert ds.n_rows == 30
    np.testing.assert_array_equal(ds.array("x"), np.arange(30))
    ds.close()


def test_open_empty_directory_is_empty_dataset(tmp_path):
    (tmp_path / "ds").mkdir()
    ds = colstore.open(tmp_path / "ds")
    assert ds.n_rows == 0
    ds.close()


def test_append_accepts_a_reader_as_data(tmp_path):
    src = colstore.store({"x": np.arange(5, dtype=np.int64)}, tmp_path / "src.cstore")
    colstore.append(tmp_path / "ds", src)
    src.close()
    ds = colstore.open(tmp_path / "ds")
    np.testing.assert_array_equal(ds.array("x"), np.arange(5))
    ds.close()


# ---- ColStoreDataset over a directory ---------------------------------------


def test_dataset_constructed_from_a_directory(tmp_path):
    for i in range(3):
        colstore.append(tmp_path, {"x": np.arange(i * 10, i * 10 + 10, dtype=np.int64)})
    ds = colstore.ColStoreDataset(tmp_path)  # a directory, not just open()
    assert ds.n_rows == 30
    np.testing.assert_array_equal(ds.array("x"), np.arange(30))
    ds.close()
    # a one-element list naming the directory behaves identically
    ds = colstore.ColStoreDataset([tmp_path])
    assert ds.n_rows == 30
    ds.close()


def test_dataset_from_empty_directory_is_empty(tmp_path):
    (tmp_path / "ds").mkdir()
    ds = colstore.ColStoreDataset(tmp_path / "ds")
    assert ds.n_rows == 0
    ds.close()


def test_dataset_mixes_a_directory_and_a_file(tmp_path):
    shard_dir = tmp_path / "ds"
    colstore.append(shard_dir, {"x": np.arange(10, dtype=np.int64)})
    colstore.append(shard_dir, {"x": np.arange(10, 20, dtype=np.int64)})
    loose = colstore.store({"x": np.arange(20, 25, dtype=np.int64)}, tmp_path / "loose.cstore")
    loose.close()
    ds = colstore.ColStoreDataset([shard_dir, tmp_path / "loose.cstore"])
    assert ds.n_rows == 25
    np.testing.assert_array_equal(ds.array("x"), np.arange(25))  # dir shards then the file
    ds.close()


def test_dataset_append_grows_from_a_directory(tmp_path):
    shard_dir = tmp_path / "ds"
    colstore.append(shard_dir, {"x": np.arange(10, dtype=np.int64)})
    colstore.append(shard_dir, {"x": np.arange(10, 20, dtype=np.int64)})
    ds = colstore.ColStoreDataset()
    ds.append(shard_dir)  # a directory source grows the in-memory dataset
    assert ds.n_rows == 20
    np.testing.assert_array_equal(ds.array("x"), np.arange(20))
    ds.close()


# ---- naming -----------------------------------------------------------------


def test_custom_template_and_literal_name(tmp_path):
    colstore.append(tmp_path, {"x": np.arange(3, dtype=np.int64)}, name="run_{index:03d}.cstore")
    colstore.append(tmp_path, {"x": np.arange(3, dtype=np.int64)}, name="run_{index:03d}.cstore")
    colstore.append(tmp_path, {"x": np.arange(3, dtype=np.int64)}, name="snapshot.cstore")
    assert _shards(tmp_path) == ["run_000.cstore", "run_001.cstore", "snapshot.cstore"]


def test_literal_name_collision_is_rejected(tmp_path):
    colstore.append(tmp_path, {"x": np.arange(3, dtype=np.int64)}, name="only.cstore")
    with pytest.raises(FileExistsError):
        colstore.append(tmp_path, {"x": np.arange(3, dtype=np.int64)}, name="only.cstore")


def test_index_allocation_tolerates_gaps(tmp_path):
    for _ in range(5):
        colstore.append(tmp_path, {"x": np.arange(2, dtype=np.int64)})  # shard_00000..00004
    os.remove(tmp_path / "shard_00002.cstore")  # punch a hole
    path = colstore.append(tmp_path, {"x": np.arange(2, dtype=np.int64)})
    assert path.name == "shard_00005.cstore"  # max + 1, not the count


# ---- atomicity / orphans ----------------------------------------------------


def test_orphan_temp_and_lock_are_not_shards(tmp_path):
    colstore.append(tmp_path, {"x": np.arange(4, dtype=np.int64)})
    (tmp_path / ".shard_00001.cstore.tmp").write_bytes(b"partial junk")  # simulated crash orphan
    ds = colstore.open(tmp_path)  # ignores the temp and the .colstore.lock sentinel
    assert ds.n_rows == 4
    assert all(not p.endswith(".tmp") for p in ds.path) if isinstance(ds.path, list) else True
    ds.close()


# ---- single-writer lock -----------------------------------------------------


def test_directory_lock_rejects_a_second_appender(tmp_path):
    first = colstore.appender(tmp_path)
    try:
        with pytest.raises(OSError, match="lock"):
            colstore.appender(tmp_path)
    finally:
        first.close()
    # released on close -> a new appender succeeds
    colstore.appender(tmp_path).close()


def test_appender_releases_lock_if_schema_read_fails(tmp_path, monkeypatch):
    # A damaged first shard makes the construction-time schema read raise; the
    # directory lock must be released, not leaked, so a later writer can take it.
    import colstore.shards as shards_mod

    colstore.append(tmp_path, {"x": np.arange(3, dtype=np.int64)})

    def _boom(directory):
        raise colstore.FormatError("corrupt first shard")

    monkeypatch.setattr(shards_mod, "_existing_schema", _boom)
    with pytest.raises(colstore.FormatError):
        colstore.appender(tmp_path)
    monkeypatch.undo()  # restore the real schema read
    colstore.appender(tmp_path).close()  # the lock was released, so this succeeds


# ---- schema enforcement -----------------------------------------------------


def test_schema_mismatch_rejected_and_directory_unchanged(tmp_path):
    colstore.append(tmp_path, {"x": np.arange(3, dtype=np.int64)})
    before = _shards(tmp_path)
    with pytest.raises(ValueError):
        colstore.append(tmp_path, {"x": np.arange(3, dtype=np.float64)})  # dtype mismatch
    with pytest.raises(ValueError):
        colstore.append(tmp_path, {"y": np.arange(3, dtype=np.int64)})  # name mismatch
    assert _shards(tmp_path) == before  # no bad shard written


# ---- streaming appender -----------------------------------------------------


def test_appender_rolls_at_row_budget(tmp_path):
    with colstore.appender(tmp_path, shard_size=25) as ap:
        for _ in range(4):  # 4 x 10 = 40 rows
            ap.write({"x": np.arange(10, dtype=np.int64)})
        assert ap.n_shards == 1 and ap.pending_rows == 10  # rolled once at 30 >= 25
    assert len(_shards(tmp_path)) == 2  # remainder flushed on close
    ds = colstore.open(tmp_path)
    assert ds.n_rows == 40
    ds.close()


def test_appender_rolls_at_byte_budget(tmp_path):
    with colstore.appender(tmp_path, shard_size="1 KiB") as ap:
        for _ in range(20):  # 100 int64 = 800 B/batch; 1 KiB -> ~128 rows/shard
            ap.write({"v": np.arange(100, dtype=np.int64)})
    ds = colstore.open(tmp_path)
    assert ds.n_rows == 2000
    np.testing.assert_array_equal(ds.array("v")[:100], np.arange(100))
    ds.close()


def test_appender_none_budget_rolls_only_on_flush(tmp_path):
    with colstore.appender(tmp_path, shard_size=None) as ap:
        ap.write({"x": np.arange(10, dtype=np.int64)})
        ap.write({"x": np.arange(10, 20, dtype=np.int64)})
        assert ap.n_shards == 0 and ap.pending_rows == 20  # no auto-roll
        ap.flush()
        assert ap.n_shards == 1 and ap.pending_rows == 0
    assert len(_shards(tmp_path)) == 1  # nothing left to flush on close


def test_appender_literal_name_with_budget_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="literal"):
        colstore.appender(tmp_path, name="one.cstore", shard_size=5)


def test_appender_literal_name_writes_one_shard(tmp_path):
    with colstore.appender(tmp_path, name="one.cstore") as ap:  # shard_size=None
        ap.write({"x": np.arange(5, dtype=np.int64)})
        ap.write({"x": np.arange(5, 10, dtype=np.int64)})
    assert _shards(tmp_path) == ["one.cstore"]
    ds = colstore.open(tmp_path)
    np.testing.assert_array_equal(ds.array("x"), np.arange(10))
    ds.close()


def test_appender_close_is_idempotent(tmp_path):
    ap = colstore.appender(tmp_path)
    ap.write({"x": np.arange(3, dtype=np.int64)})
    ap.close()
    ap.close()  # no error
    assert ap.closed


def test_appender_mid_stream_schema_mismatch(tmp_path):
    with colstore.appender(tmp_path) as ap:
        ap.write({"x": np.arange(3, dtype=np.int64)})
        with pytest.raises(ValueError):
            ap.write({"x": np.arange(3, dtype=np.float64)})  # rejected before buffering
    ds = colstore.open(tmp_path)  # the good batch survived
    np.testing.assert_array_equal(ds.array("x"), np.arange(3))
    ds.close()


# ---- statistics flow through to the per-shard read skip ---------------------


def test_append_with_statistics_enables_skip(tmp_path):
    # multi-record shards with statistics -> the #214 record skip applies per shard
    with colstore.appender(tmp_path, shard_size=50, statistics=True) as ap:
        for i in range(4):
            ap.write({"key": np.arange(i * 50, i * 50 + 50, dtype=np.int64)})
    ds = colstore.open(tmp_path)
    np.testing.assert_array_equal(ds[col("key") > 180, "key"].array(), np.arange(181, 200))
    ds.close()
