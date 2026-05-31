"""Tests for header validation: version, checksum, and truncation."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from colstore import ColStore, FormatError
from colstore import format as fmt


def _write(path, columns):
    fmt.write_dataset(columns, path, batch_size=1000, show_progress=False)


def test_truncated_file_raises(tmp_path):
    path = tmp_path / "t.cstore"
    _write(path, {"x": np.arange(100, dtype=np.float64)})
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(FormatError, match="truncated"):
        ColStore(path)


def test_corrupt_manifest_checksum_raises(tmp_path):
    path = tmp_path / "c.cstore"
    _write(path, {"alpha": np.arange(3, dtype=np.float64)})
    raw = bytearray(path.read_bytes())
    manifest_size = struct.unpack("<Q", raw[8:16])[0]
    manifest = json.loads(raw[16 : 16 + manifest_size])
    # Equal-length edit: change content without resizing the header.
    manifest["columns"][0]["name"] = "ALPHA"
    edited = json.dumps(manifest).encode("utf-8")
    assert len(edited) == manifest_size
    raw[16 : 16 + manifest_size] = edited
    path.write_bytes(bytes(raw))
    with pytest.raises(FormatError, match="checksum"):
        ColStore(path)


def test_unsupported_version_raises(tmp_path):
    path = tmp_path / "v.cstore"
    _write(path, {"x": np.arange(4, dtype=np.float64)})
    manifest, data_offset = fmt.read_header(path)
    manifest["format_version"] = 999
    manifest["manifest_crc32"] = fmt._manifest_checksum(manifest["columns"], manifest["n_rows"])
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    header_size = len(fmt._MAGIC) + fmt._MANIFEST_LEN_SIZE + len(manifest_bytes)
    new_offset = fmt.align_up(header_size)
    column_bytes = path.read_bytes()[data_offset:]
    with open(path, "wb") as handle:
        handle.write(fmt._MAGIC)
        handle.write(struct.pack(fmt._MANIFEST_LEN_FMT, len(manifest_bytes)))
        handle.write(manifest_bytes)
        handle.write(b"\x00" * (new_offset - header_size))
        handle.write(column_bytes)
    with pytest.raises(FormatError, match="format_version"):
        ColStore(path)


def test_checksum_is_written(tmp_path):
    path = tmp_path / "k.cstore"
    _write(path, {"x": np.arange(4, dtype=np.float64)})
    manifest, _ = fmt.read_header(path)
    assert "manifest_crc32" in manifest
