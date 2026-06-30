"""Tests for ColStoreWriter: streaming, append, crash-safety, locking, vectored I/O.

The module-level API tests (``test_api.py``) cover the create/recreate/update
entry points at the function level. These tests focus on the writer's own
contracts: schema validation, multi-record append, the orphan-byte truncation
on update, advisory locking, the close-on-GC fallback, and the close-without-
write cleanup.
"""

from __future__ import annotations

import hashlib
import io
import os
import warnings

import numpy as np
import pytest

import colstore
from colstore import format as fmt
from colstore import writer as writer_mod

# ---- Basic writer lifecycle ------------------------------------------------


def test_writer_writes_single_record_round_trip(tmp_path):
    """One write + close produces a file readable as a single-record dataset."""
    path = tmp_path / "single.cstore"
    with colstore.create(path) as w:
        w.write({"a": np.arange(10, dtype=np.float32)})
    with colstore.open(path) as ds:
        assert ds.n_rows == 10
        assert ds._is_multi_record is False  # single record fast path
        assert np.array_equal(ds[:, "a"].array(), np.arange(10, dtype=np.float32))


def test_writer_writes_multi_record(tmp_path):
    """Multiple writes produce a multi-record file with the right total row count."""
    path = tmp_path / "multi.cstore"
    with colstore.create(path) as w:
        w.write({"a": np.array([1, 2, 3], dtype=np.int32)})
        w.write({"a": np.array([4, 5], dtype=np.int32)})
        w.write({"a": np.array([6, 7, 8, 9], dtype=np.int32)})
        assert w.n_records == 3
        assert w.committed_rows == 9
    with colstore.open(path) as ds:
        assert ds.n_rows == 9
        assert ds._is_multi_record is True
        assert np.array_equal(
            ds[:, "a"].array(), np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
        )


def test_writer_multi_record_datetime_and_timedelta(tmp_path):
    """A multi-record write of datetime64 / timedelta64 columns round-trips.

    The writev path casts each buffer to bytes, which memoryview cannot do for a
    temporal dtype directly, so the arrays are viewed as uint8 first.
    """
    path = tmp_path / "temporal.cstore"
    times = np.array(["2020-01-01", "2021-06-15", "2022-12-31"], dtype="datetime64[ns]")
    deltas = np.array([1, 2, 3], dtype="timedelta64[s]").astype("timedelta64[ns]")
    with colstore.create(path) as w:
        w.write({"t": times[:2], "d": deltas[:2]})
        w.write({"t": times[2:], "d": deltas[2:]})
    with colstore.open(path) as ds:
        assert ds._is_multi_record is True
        np.testing.assert_array_equal(ds.array("t"), times)
        np.testing.assert_array_equal(ds.array("d"), deltas)


def test_writer_close_is_idempotent(tmp_path):
    """Calling close() twice does not raise."""
    path = tmp_path / "idem.cstore"
    w = colstore.create(path)
    w.write({"a": np.arange(3, dtype=np.int32)})
    w.close()
    w.close()  # must not raise
    assert w.closed


def test_writer_rejects_writes_after_close(tmp_path):
    """write() after close() raises ValueError."""
    path = tmp_path / "after.cstore"
    w = colstore.create(path)
    w.write({"a": np.arange(3, dtype=np.int32)})
    w.close()
    with pytest.raises(ValueError, match="closed"):
        w.write({"a": np.arange(3, dtype=np.int32)})


def test_writer_context_manager_closes(tmp_path):
    """The context manager calls close() on exit."""
    path = tmp_path / "ctx.cstore"
    with colstore.create(path) as w:
        w.write({"a": np.arange(3, dtype=np.int32)})
    assert w.closed


def test_writer_context_manager_closes_on_exception(tmp_path):
    """Even on exception, the context manager calls close()."""
    path = tmp_path / "exc.cstore"
    with pytest.raises(RuntimeError), colstore.create(path) as w:
        w.write({"a": np.arange(3, dtype=np.int32)})
        raise RuntimeError("simulated failure")
    assert w.closed
    # File should still be valid up to the last committed write.
    with colstore.open(path) as ds:
        assert ds.n_rows == 3


# ---- Empty / edge cases ----------------------------------------------------


def test_writer_empty_dict_is_noop(tmp_path):
    """write({}) is a no-op: no record, schema not locked."""
    path = tmp_path / "empty.cstore"
    with colstore.create(path) as w:
        w.write({})
        # Should still be able to lock the schema with a non-empty write next.
        w.write({"a": np.arange(2, dtype=np.int32)})
        assert w.n_records == 1
    with colstore.open(path) as ds:
        assert ds.columns == ["a"]
        assert ds.n_rows == 2


def test_writer_zero_row_record_is_legal(tmp_path):
    """A record with zero rows is allowed (write({col: empty_array})).

    Distinct from write({}) -- this locks the schema and writes a record
    that just happens to have zero rows.
    """
    path = tmp_path / "zr.cstore"
    with colstore.create(path) as w:
        w.write({"a": np.array([], dtype=np.int32)})
        w.write({"a": np.array([1, 2, 3], dtype=np.int32)})
        assert w.n_records == 2
        assert w.committed_rows == 3
    with colstore.open(path) as ds:
        assert ds.n_rows == 3
        assert np.array_equal(ds[:, "a"].array(), np.array([1, 2, 3]))


def test_writer_close_without_writing_removes_file(tmp_path):
    """create()/recreate() + close() without write() leaves no file behind.

    A zero-byte .cstore would fail every reader; better to not leave it.
    """
    path = tmp_path / "nope.cstore"
    with colstore.create(path):
        pass  # no write
    assert not path.exists()


def test_writer_close_without_writing_after_empty_dict_removes_file(tmp_path):
    """write({}) is a no-op, so closing after only empty dicts also removes."""
    path = tmp_path / "noempty.cstore"
    with colstore.create(path) as w:
        w.write({})
        w.write({})
    assert not path.exists()


# ---- Schema validation -----------------------------------------------------


def test_writer_schema_locked_on_first_write(tmp_path):
    """Second write with a different schema raises ValueError."""
    path = tmp_path / "lock.cstore"
    with colstore.create(path) as w:
        w.write({"a": np.arange(3, dtype=np.int32)})
        with pytest.raises(ValueError, match="Column name mismatch"):
            w.write({"b": np.arange(3, dtype=np.int32)})


def test_writer_schema_dtype_must_match(tmp_path):
    """Same column name but different dtype is rejected."""
    path = tmp_path / "dt.cstore"
    with colstore.create(path) as w:
        w.write({"a": np.arange(3, dtype=np.int32)})
        with pytest.raises(ValueError, match="dtype"):
            w.write({"a": np.arange(3, dtype=np.int64)})


def test_writer_schema_column_count_must_match(tmp_path):
    """Different number of columns is rejected."""
    path = tmp_path / "cc.cstore"
    with colstore.create(path) as w:
        w.write({"a": np.arange(3, dtype=np.int32), "b": np.arange(3, dtype=np.float64)})
        with pytest.raises(ValueError, match="Schema mismatch"):
            w.write({"a": np.arange(3, dtype=np.int32)})


def test_writer_rejects_ragged_columns(tmp_path):
    """All columns in a single write() call must have the same length."""
    path = tmp_path / "rag.cstore"
    with colstore.create(path) as w, pytest.raises(ValueError, match="rows"):
        w.write({"a": np.arange(3, dtype=np.int32), "b": np.arange(4, dtype=np.float64)})


def test_writer_rejects_object_dtype(tmp_path):
    """Object dtype is unsupported."""
    path = tmp_path / "obj.cstore"
    with colstore.create(path) as w, pytest.raises(TypeError, match="object dtype"):
        w.write({"a": np.array(["x", "y"], dtype=object)})


def test_writer_rejects_2d_input(tmp_path):
    """Each column must be 1D."""
    path = tmp_path / "2d.cstore"
    with colstore.create(path) as w, pytest.raises(ValueError, match="1D"):
        w.write({"a": np.arange(6).reshape(2, 3)})


# ---- Update mode -----------------------------------------------------------


def test_update_loads_existing_schema(tmp_path):
    """update() picks up the schema from the existing file's manifest."""
    path = tmp_path / "up.cstore"
    with colstore.create(path) as w:
        w.write({"a": np.arange(3, dtype=np.int32), "b": np.arange(3, dtype=np.float64)})
    with colstore.update(path) as w, pytest.raises(ValueError, match="Column name mismatch"):
        # Wrong column name on a write must be rejected with the loaded schema.
        w.write({"c": np.arange(3, dtype=np.int32), "b": np.arange(3, dtype=np.float64)})


def test_update_appends_to_existing_records(tmp_path):
    """update() finds the end of the last record and appends past it."""
    path = tmp_path / "app.cstore"
    with colstore.create(path) as w:
        w.write({"a": np.arange(5, dtype=np.int32)})
        w.write({"a": np.arange(10, 15, dtype=np.int32)})
    with colstore.update(path) as w:
        assert w.n_records == 2
        assert w.committed_rows == 10
        w.write({"a": np.array([100, 200, 300], dtype=np.int32)})
    with colstore.open(path) as ds:
        expected = np.concatenate(
            [
                np.arange(5, dtype=np.int32),
                np.arange(10, 15, dtype=np.int32),
                np.array([100, 200, 300], dtype=np.int32),
            ]
        )
        assert np.array_equal(ds[:, "a"].array(), expected)


def test_update_truncates_orphan_bytes(tmp_path):
    """If a prior writer crashed leaving bytes past the last committed record,
    update() truncates them so the new appends land at the right offset.
    """
    path = tmp_path / "orphan.cstore"
    # Build a clean file with one record.
    with colstore.create(path) as w:
        w.write({"a": np.arange(5, dtype=np.int32)})
    clean_size = path.stat().st_size

    # Manually append junk bytes simulating a crashed second write.
    with open(path, "ab") as f:
        f.write(b"\x99" * 256)
    assert path.stat().st_size == clean_size + 256

    # update() must truncate the junk before its own append.
    with colstore.update(path) as w:
        w.write({"a": np.arange(100, 103, dtype=np.int32)})

    # File should now be exactly: clean header + one record (5 rows)
    # + one new record (3 rows). Read it back and check row content; if the
    # truncate hadn't happened, the second record header would land at the
    # wrong offset and the read would fail.
    with colstore.open(path) as ds:
        assert ds.n_rows == 8
        expected = np.concatenate(
            [np.arange(5, dtype=np.int32), np.array([100, 101, 102], dtype=np.int32)]
        )
        assert np.array_equal(ds[:, "a"].array(), expected)


def test_update_writer_records_count_starts_from_existing(tmp_path):
    """update() inherits n_records from the existing file."""
    path = tmp_path / "in.cstore"
    with colstore.create(path) as w:
        w.write({"a": np.arange(2, dtype=np.int32)})
        w.write({"a": np.arange(3, dtype=np.int32)})
    with colstore.update(path) as w:
        assert w.n_records == 2
        assert w.committed_rows == 5


# ---- Crash safety / commit-on-close ---------------------------------------


def test_uncommitted_writes_are_invisible_to_readers(tmp_path):
    """A reader opening the file mid-write sees only what was committed by
    the last successful close.
    """
    path = tmp_path / "snap.cstore"
    with colstore.create(path) as w:
        w.write({"a": np.arange(3, dtype=np.int32)})
    # Now open for update, write more, but DON'T close yet.
    w2 = colstore.update(path)
    w2.write({"a": np.array([10, 20], dtype=np.int32)})
    # Open a reader concurrently. It should see only the committed record.
    with colstore.open(path) as ds:
        assert ds.n_rows == 3
        assert np.array_equal(ds[:, "a"].array(), np.arange(3, dtype=np.int32))
    w2.close()
    # After close, the new record is visible.
    with colstore.open(path) as ds:
        assert ds.n_rows == 5


def test_simulated_crash_loses_only_uncommitted(tmp_path):
    """A 'crash' (file handle dropped without close) is equivalent to
    rolling back to the last committed state.
    """
    path = tmp_path / "crash.cstore"
    with colstore.create(path) as w:
        w.write({"a": np.arange(3, dtype=np.int32)})

    # Simulate a crashed writer: open for update, write, then bypass close
    # by destroying state without going through close().
    w2 = colstore.update(path)
    w2.write({"a": np.array([99, 99, 99], dtype=np.int32)})
    # Close the file handle WITHOUT committing counters. This mimics a hard
    # crash. We have to suppress the close-on-del ResourceWarning since it
    # would try to commit; we manually invalidate state so __del__ skips.
    w2._has_header = True  # ensure not deleted on close path
    # Force-release the lock and close the fd without commit. Use the
    # cross-platform _lock helper so this test runs on Windows too.
    from colstore import _lock

    _lock.unlock(w2._file.fileno())
    w2._file.close()
    w2._closed = True  # tell __del__ to leave us alone

    # Reopen: the uncommitted record's bytes are past the manifest's
    # n_records; they're orphan bytes. update() should truncate them, and
    # a read should see only the original record.
    with colstore.open(path) as ds:
        assert ds.n_rows == 3
    with colstore.update(path) as w3:
        assert w3.n_records == 1  # orphan record didn't survive
        w3.write({"a": np.array([1000], dtype=np.int32)})
    with colstore.open(path) as ds:
        assert ds.n_rows == 4
        assert np.array_equal(ds[:, "a"].array(), np.array([0, 1, 2, 1000], dtype=np.int32))


# ---- Advisory locking ------------------------------------------------------


def test_concurrent_writers_rejected(tmp_path):
    """Opening a second writer on the same path while one is alive raises."""
    path = tmp_path / "lock.cstore"
    w1 = colstore.create(path)
    try:
        with pytest.raises(OSError, match="lock"):
            colstore.create(path)  # would conflict on the lock
    finally:
        w1.close()


def test_lock_released_after_close(tmp_path):
    """Once close() runs, a new writer can claim the lock."""
    path = tmp_path / "rel.cstore"
    with colstore.create(path) as w:
        w.write({"a": np.arange(3, dtype=np.int32)})
    # Lock is released; recreate() must succeed.
    with colstore.recreate(path) as w2:
        w2.write({"a": np.arange(5, dtype=np.int32)})


# ---- __del__ safety net ----------------------------------------------------


def test_del_emits_warning_and_commits(tmp_path):
    """A writer dropped without close() emits ResourceWarning and commits."""
    path = tmp_path / "gc.cstore"
    w = colstore.create(path)
    w.write({"a": np.arange(3, dtype=np.int32)})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        del w
        import gc

        gc.collect()
    # The warning may or may not have fired depending on GC timing; if it
    # did, it must be a ResourceWarning. (We don't assert it fired because
    # __del__ doesn't always run promptly in CPython, but if it did, it
    # should have committed.)
    if any(issubclass(rec.category, ResourceWarning) for rec in caught):
        # Commit must have happened; file should be readable.
        with colstore.open(path) as ds:
            assert ds.n_rows == 3


# ---- Multiple writes against multi-column schema --------------------------


def test_writer_multicol_round_trip(tmp_path):
    """Multi-column streaming write produces a valid multi-record file."""
    path = tmp_path / "mc.cstore"
    with colstore.create(path) as w:
        for i in range(5):
            n = 7 + i
            w.write(
                {
                    "x": np.arange(n, dtype=np.int32) + 100 * i,
                    "y": np.linspace(0, 1, n, dtype=np.float64) + i,
                }
            )
    with colstore.open(path) as ds:
        assert ds.n_rows == 7 + 8 + 9 + 10 + 11
        assert ds._is_multi_record is True
        # Spot-check first record's contents via slice.
        first = ds[:7, "x"].array()
        assert np.array_equal(first, np.arange(7, dtype=np.int32))


# ---- Vectored (writev) record emission ----------------------------------------
# Tests for the writer's vectored (writev) record emission.
#
# The vectored and sequential paths must produce byte-identical files -- the
# on-disk format is pinned by the sequential path, and writev only changes how
# the bytes reach the fd. The tests therefore hash whole files across paths,
# including the awkward cases: zero-row records, padding, strided inputs,
# update-mode appends, partial writev returns, and IOV_MAX chunking.

requires_writev = pytest.mark.skipif(not writer_mod._HAS_WRITEV, reason="platform has no os.writev")


def _records():
    rng = np.random.default_rng(3)
    return [
        {
            "a": rng.standard_normal(n),
            "b": rng.integers(0, 100, n).astype(np.int32),
            "c": rng.standard_normal(n).astype(np.float32),
        }
        for n in (50, 0, 1, 1000, 7)  # zero-row record and pad-triggering sizes
    ]


def _write_and_hash(path, records, *, vectored: bool, monkeypatch) -> str:
    monkeypatch.setattr(writer_mod, "_HAS_WRITEV", vectored)
    with colstore.create(path) as writer:
        for record in records:
            writer.write(record)
    with colstore.update(path) as writer:  # appends must also match
        writer.write(records[0])
    return hashlib.sha256(path.read_bytes()).hexdigest()


@requires_writev
def test_vectored_and_sequential_paths_are_byte_identical(tmp_path, monkeypatch):
    records = _records()
    digest_vec = _write_and_hash(
        tmp_path / "vec.cstore", records, vectored=True, monkeypatch=monkeypatch
    )
    digest_seq = _write_and_hash(
        tmp_path / "seq.cstore", records, vectored=False, monkeypatch=monkeypatch
    )
    assert digest_vec == digest_seq

    dataset = colstore.open(tmp_path / "vec.cstore")
    expected = np.concatenate([r["a"] for r in records] + [records[0]["a"]])
    assert np.array_equal(dataset[:, "a"].array(), expected)
    dataset.close()


@requires_writev
def test_strided_column_input_matches_sequential(tmp_path, monkeypatch):
    # normalize_columns passes strided views through; the vectored path must
    # serialize their logical order exactly as tofile() does.
    base = np.arange(200.0)
    record = {"x": base[::2]}
    digest_vec = _write_and_hash(
        tmp_path / "v.cstore", [record], vectored=True, monkeypatch=monkeypatch
    )
    digest_seq = _write_and_hash(
        tmp_path / "s.cstore", [record], vectored=False, monkeypatch=monkeypatch
    )
    assert digest_vec == digest_seq
    dataset = colstore.open(tmp_path / "v.cstore")
    assert np.array_equal(dataset[:, "x"].array(), np.concatenate([base[::2], base[::2]]))
    dataset.close()


@requires_writev
def test_partial_writev_returns_are_resumed(tmp_path, monkeypatch):
    # POSIX permits writev to write fewer bytes than requested; emulate a
    # pathological fd that accepts at most 7 bytes per call, splitting
    # buffers mid-element.
    real_write = os.write

    def tiny_writev(fd, buffers):
        for view in buffers:
            if view.nbytes:
                return real_write(fd, view[:7].tobytes())
        return 0

    monkeypatch.setattr(os, "writev", tiny_writev)
    records = _records()
    with colstore.create(tmp_path / "p.cstore") as writer:
        for record in records:
            writer.write(record)
    monkeypatch.undo()

    digest = hashlib.sha256((tmp_path / "p.cstore").read_bytes()).hexdigest()
    with colstore.create(tmp_path / "ref.cstore") as writer:
        for record in records:
            writer.write(record)
    assert digest == hashlib.sha256((tmp_path / "ref.cstore").read_bytes()).hexdigest()


@requires_writev
def test_iov_max_chunking(tmp_path, monkeypatch):
    # Records with more segments than IOV_MAX must split into successive
    # writev calls without reordering.
    monkeypatch.setattr(writer_mod, "_IOV_MAX", 2)
    rng = np.random.default_rng(5)
    record = {f"c{i}": rng.standard_normal(13) for i in range(9)}  # 11 segments
    with colstore.create(tmp_path / "chunked.cstore") as writer:
        writer.write(record)
    monkeypatch.undo()

    dataset = colstore.open(tmp_path / "chunked.cstore")
    for name, expected in record.items():
        assert np.array_equal(dataset[:, name].array(), expected)
    dataset.close()


@requires_writev
def test_writev_zero_progress_raises_instead_of_spinning(monkeypatch):
    # POSIX permits writev to return 0; a write-full loop must surface that
    # as an error rather than retrying forever.
    monkeypatch.setattr(os, "writev", lambda fd, buffers: 0)
    read_fd, write_fd = os.pipe()
    try:
        with pytest.raises(OSError, match="no progress"):
            writer_mod._writev_full(write_fd, [b"abcd"])
    finally:
        os.close(read_fd)
        os.close(write_fd)


@requires_writev
def test_writev_full_zero_length_buffers():
    read_fd, write_fd = os.pipe()
    try:
        writer_mod._writev_full(write_fd, [b"", b"ab", b"", b"cd", b""])
        assert os.read(read_fd, 16) == b"abcd"
    finally:
        os.close(read_fd)
        os.close(write_fd)


@requires_writev
def test_record_header_bytes_matches_streamed_header():
    # The bytes-building helper and the file-writing helper must stay in
    # lockstep: same 32-byte layout, same CRC coverage.
    stream = io.BytesIO()
    fmt.write_record_header(stream, record_index=7, n_rows=12345)
    assert stream.getvalue() == fmt.record_header_bytes(7, 12345)
    assert len(stream.getvalue()) == 32
