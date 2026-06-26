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


def test_append_accepts_a_file_path_as_data(tmp_path):
    colstore.store({"x": np.arange(8, dtype=np.int64)}, tmp_path / "src.cstore").close()
    path = colstore.append(tmp_path / "ds", tmp_path / "src.cstore")  # a path, not in-memory
    assert path.name == "shard_00000.cstore"
    ds = colstore.open(tmp_path / "ds")
    np.testing.assert_array_equal(ds.array("x"), np.arange(8))
    ds.close()


def test_append_single_file_source_is_a_verbatim_copy(tmp_path):
    # A single-file source is copied byte-for-byte, not read and rewritten.
    src_path = tmp_path / "src.cstore"
    colstore.store({"x": np.arange(200, dtype=np.int64)}, src_path).close()
    shard = colstore.append(tmp_path / "ds", src_path)
    assert shard.read_bytes() == src_path.read_bytes()
    ds = colstore.open(tmp_path / "ds")
    np.testing.assert_array_equal(ds.array("x"), np.arange(200))
    ds.close()


def test_parallel_copy_is_byte_identical(tmp_path, monkeypatch):
    # Force the multi-stream copy path on a small file and check the shard is exact.
    import colstore.shards as shards_mod

    monkeypatch.setattr(shards_mod, "_PARALLEL_COPY_MIN_CHUNK", 256)
    data = {
        "x": np.arange(5000, dtype=np.int64),
        "y": (np.arange(5000) * 0.5).astype(np.float64),
    }
    src_path = tmp_path / "src.cstore"
    colstore.store(data, src_path).close()
    assert src_path.stat().st_size // 256 >= shards_mod._PARALLEL_COPY_MAX_STREAMS  # >1 stream
    shard = colstore.append(tmp_path / "ds", src_path)
    assert shard.read_bytes() == src_path.read_bytes()  # parallel copy is exact
    ds = colstore.open(tmp_path / "ds")
    np.testing.assert_array_equal(ds.array("x"), data["x"])
    np.testing.assert_array_equal(ds.array("y"), data["y"])
    ds.close()


def test_copy_file_below_threshold_is_single_stream(tmp_path):
    # A file under the threshold copies in one stream; still byte-identical.
    import colstore.shards as shards_mod

    src_path = tmp_path / "small.cstore"
    colstore.store({"x": np.arange(50, dtype=np.int64)}, src_path).close()
    assert src_path.stat().st_size < shards_mod._PARALLEL_COPY_MIN_CHUNK
    dst_path = tmp_path / "out.bin"
    shards_mod._copy_file(src_path, dst_path)
    assert dst_path.read_bytes() == src_path.read_bytes()


def test_append_streams_a_multifile_dataset_source(tmp_path):
    # A multi-file dataset source is streamed (not materialized) into one shard.
    a = colstore.store({"x": np.arange(10, dtype=np.int64)}, tmp_path / "a.cstore")
    b = colstore.store({"x": np.arange(10, 20, dtype=np.int64)}, tmp_path / "b.cstore")
    src = a | b  # a two-file ColStoreDataset
    colstore.append(tmp_path / "ds", src)
    a.close()
    b.close()
    ds = colstore.open(tmp_path / "ds")
    assert ds.n_rows == 20
    np.testing.assert_array_equal(ds.array("x"), np.arange(20))
    ds.close()


def test_append_rejects_self_append(tmp_path):
    # Appending the dataset's own directory (or a dataset over its shards) would
    # silently duplicate every existing row; reject it before writing.
    colstore.append(tmp_path, {"x": np.arange(5, dtype=np.int64)})
    colstore.append(tmp_path, {"x": np.arange(5, 10, dtype=np.int64)})
    before = _shards(tmp_path)
    with pytest.raises(ValueError, match="itself"):
        colstore.append(tmp_path, tmp_path)  # the directory itself
    own = colstore.open(tmp_path)
    with pytest.raises(ValueError, match="itself"):
        colstore.append(tmp_path, own)  # a dataset over its own shards
    own.close()
    with pytest.raises(ValueError, match="itself"):
        colstore.append(tmp_path, tmp_path, statistics=True)  # statistics path too
    assert _shards(tmp_path) == before  # nothing duplicated


def test_append_source_validates_schema_before_writing(tmp_path):
    shard_dir = tmp_path / "ds"
    colstore.append(shard_dir, {"x": np.arange(3, dtype=np.int64)})
    before = _shards(shard_dir)
    bad = colstore.store({"x": np.arange(3, dtype=np.float64)}, tmp_path / "bad.cstore")
    with pytest.raises(ValueError, match="schema"):
        colstore.append(shard_dir, bad)  # dtype mismatch, from headers, no shard written
    bad.close()
    assert _shards(shard_dir) == before


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


def test_non_cstore_name_is_rejected(tmp_path):
    # A name without the .cstore extension would be written but never listed by
    # the reader's *.cstore glob, so it is refused at the entry point.
    for bad in ("shard_{index:03d}.bin", "snapshot.dat", "shard_{index}"):
        with pytest.raises(ValueError, match="cstore"):
            colstore.append(tmp_path, {"x": np.arange(3, dtype=np.int64)}, name=bad)
    with pytest.raises(ValueError, match="cstore"):
        colstore.appender(tmp_path, name="run_{index}.parquet")
    assert _shards(tmp_path) == []  # nothing written by a rejected name


# ---- atomicity / orphans ----------------------------------------------------


def test_orphan_temp_and_lock_are_not_shards(tmp_path):
    colstore.append(tmp_path, {"x": np.arange(4, dtype=np.int64)})
    (tmp_path / ".shard_00001.cstore.tmp").write_bytes(b"partial junk")  # simulated crash orphan
    ds = colstore.open(tmp_path)  # ignores the temp and the .colstore.lock sentinel
    assert ds.n_rows == 4
    assert all(not str(p).endswith(".tmp") for p in ds.paths)
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


def test_acquire_lock_closes_fd_on_unexpected_lock_error(tmp_path, monkeypatch):
    # An OSError from the lock syscall that is not contention (e.g. EIO) must not
    # leak the open lock-file descriptor.
    import colstore.shards as shards_mod

    def _raise_eio(fd):
        raise OSError(5, "EIO")

    monkeypatch.setattr(shards_mod._lock, "lock_exclusive_nonblocking", _raise_eio)
    lock_fds = []
    closed = []
    real_open, real_close = os.open, os.close

    def spy_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if str(path).endswith(shards_mod._LOCK_NAME):
            lock_fds.append(fd)
        return fd

    def spy_close(fd):
        closed.append(fd)
        return real_close(fd)

    monkeypatch.setattr(os, "open", spy_open)
    monkeypatch.setattr(os, "close", spy_close)
    with pytest.raises(OSError):
        shards_mod._acquire_directory_lock(tmp_path)
    assert lock_fds and lock_fds[0] in closed  # the lock fd was closed, not leaked


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
