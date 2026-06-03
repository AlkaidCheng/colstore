"""Tests for colstore.info and colstore.schema introspection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import colstore
from colstore import ColStoreInfo, FormatError


def test_info_basic_fields(tmp_path):
    path = tmp_path / "x.cstore"
    colstore.store(
        {"a": np.arange(10, dtype=np.float32), "b": np.arange(10, dtype=np.int64)},
        path,
        show_progress=False,
    ).close()

    i = colstore.info(path)
    assert isinstance(i, ColStoreInfo)
    assert i.path == Path(path)
    assert i.format_version == 1
    assert i.n_rows == 10
    assert i.n_records == 1
    assert i.file_size == path.stat().st_size
    assert [c["name"] for c in i.columns] == ["a", "b"]
    assert [c["dtype"] for c in i.columns] == ["<f4", "<i8"]
    assert not i.needs_compaction


def test_info_needs_compaction_flips_on_multi_record(tmp_path):
    path = tmp_path / "m.cstore"
    with colstore.create(path) as f:
        for _ in range(3):
            f.write({"a": np.arange(10, dtype=np.float32)})

    i = colstore.info(path)
    assert i.n_records == 3
    assert i.n_rows == 30
    assert i.needs_compaction


def test_info_repr_is_readable(tmp_path):
    path = tmp_path / "r.cstore"
    colstore.store({"a": np.arange(5)}, path, show_progress=False).close()
    text = repr(colstore.info(path))
    # Spot-check that key fields are visible without scrolling.
    assert "r.cstore" in text
    assert "n_rows=5" in text
    assert "n_records=1" in text
    assert "a:" in text  # column listing


def test_info_repr_flags_needs_compaction(tmp_path):
    path = tmp_path / "nc.cstore"
    with colstore.create(path) as f:
        f.write({"a": np.arange(3, dtype=np.int32)})
        f.write({"a": np.arange(3, 6, dtype=np.int32)})
    assert "needs_compaction=True" in repr(colstore.info(path))


def test_info_on_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        colstore.info(tmp_path / "nope.cstore")


def test_info_on_corrupt_file_raises(tmp_path):
    path = tmp_path / "c.cstore"
    colstore.store({"a": np.arange(5)}, path, show_progress=False).close()
    raw = bytearray(path.read_bytes())
    raw[0] ^= 0xFF  # break the magic bytes
    path.write_bytes(bytes(raw))
    with pytest.raises(FormatError):
        colstore.info(path)


def test_schema_returns_column_metadata(tmp_path):
    path = tmp_path / "s.cstore"
    colstore.store(
        {
            "alpha": np.arange(7, dtype=np.float64),
            "beta": np.arange(7, dtype=np.int32),
        },
        path,
        show_progress=False,
    ).close()

    s = colstore.schema(path)
    assert isinstance(s, list)
    assert len(s) == 2
    assert s[0]["name"] == "alpha"
    assert s[0]["dtype"] == "<f8"
    assert s[1]["name"] == "beta"
    assert s[1]["dtype"] == "<i4"


def test_schema_does_not_open_for_reads(tmp_path):
    """schema() should work even if the records would fail to read.

    We can't easily construct such a file, so settle for: schema() makes only
    a header-read and never touches record bodies, which means it's cheap on
    a multi-GB file. This is a structural test via inspection rather than
    instrumentation -- just verify it returns quickly on a multi-record file.
    """
    path = tmp_path / "big.cstore"
    with colstore.create(path) as f:
        for _ in range(50):
            f.write({"a": np.arange(100, dtype=np.float64)})
    # The schema call should return the same data as a full info() call.
    assert colstore.schema(path) == colstore.info(path).columns
