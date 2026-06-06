"""Tests for the writer's vectored (writev) record emission.

The vectored and sequential paths must produce byte-identical files -- the
on-disk format is pinned by the sequential path, and writev only changes how
the bytes reach the fd. The tests therefore hash whole files across paths,
including the awkward cases: zero-row records, padding, strided inputs,
update-mode appends, partial writev returns, and IOV_MAX chunking.
"""

from __future__ import annotations

import hashlib
import io
import os

import numpy as np
import pytest

import colstore
from colstore import format as fmt
from colstore import writer as writer_mod

pytestmark = pytest.mark.skipif(not writer_mod._HAS_WRITEV, reason="platform has no os.writev")


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


def test_writev_full_zero_length_buffers():
    read_fd, write_fd = os.pipe()
    try:
        writer_mod._writev_full(write_fd, [b"", b"ab", b"", b"cd", b""])
        assert os.read(read_fd, 16) == b"abcd"
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_record_header_bytes_matches_streamed_header():
    # The bytes-building helper and the file-writing helper must stay in
    # lockstep: same 32-byte layout, same CRC coverage.
    stream = io.BytesIO()
    fmt.write_record_header(stream, record_index=7, n_rows=12345)
    assert stream.getvalue() == fmt.record_header_bytes(7, 12345)
    assert len(stream.getvalue()) == 32
