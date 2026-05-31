"""Tests for the on-disk file format."""

from __future__ import annotations

import numpy as np
import pytest

from colstore import FILE_EXTENSION, FormatError
from colstore.format import (
    align_up,
    build_column_layout,
    read_header,
    write_dataset,
)


def test_file_extension_is_cstore():
    assert FILE_EXTENSION == ".cstore"


def test_align_up_rounds_to_alignment():
    assert align_up(0, 64) == 0
    assert align_up(1, 64) == 64
    assert align_up(63, 64) == 64
    assert align_up(64, 64) == 64
    assert align_up(65, 64) == 128


def test_write_then_read_header_roundtrips_metadata(tmp_path):
    path = tmp_path / "case.cstore"
    columns = {
        "x": np.arange(100, dtype=np.float32),
        "y": np.arange(100, dtype=np.int64),
    }
    write_dataset(columns, path, batch_size=50, show_progress=False)
    manifest, data_offset = read_header(path)
    assert manifest["format_version"] == 1
    assert manifest["n_rows"] == 100
    assert [c["name"] for c in manifest["columns"]] == ["x", "y"]
    assert data_offset % 64 == 0


def test_column_layout_offsets_are_contiguous(tmp_path):
    path = tmp_path / "case.cstore"
    columns = {
        "a": np.arange(10, dtype=np.float32),  # 40 bytes
        "b": np.arange(10, dtype=np.int64),    # 80 bytes
    }
    write_dataset(columns, path, batch_size=10, show_progress=False)
    manifest, data_offset = read_header(path)
    layout = build_column_layout(manifest, data_offset)
    a_offset, a_dtype = layout["a"]
    b_offset, b_dtype = layout["b"]
    assert a_offset == data_offset
    assert b_offset == a_offset + 10 * a_dtype.itemsize


def test_read_header_rejects_bad_magic(tmp_path):
    path = tmp_path / "bogus.cstore"
    # An 8-byte non-magic header followed by zero-padding.
    path.write_bytes(b"NOTCSTOR" + b"\x00" * 100)
    with pytest.raises(FormatError):
        read_header(path)


def test_write_rejects_object_dtype(tmp_path):
    path = tmp_path / "objs.cstore"
    with pytest.raises(TypeError, match="object dtype"):
        write_dataset(
            {"o": np.array(["x", "y", "z"], dtype=object)},
            path,
            batch_size=10,
            show_progress=False,
        )


def test_write_rejects_inconsistent_row_counts(tmp_path):
    path = tmp_path / "mismatch.cstore"
    with pytest.raises(ValueError, match="rows"):
        write_dataset(
            {"a": np.zeros(10, np.float32), "b": np.zeros(11, np.float32)},
            path,
            batch_size=10,
            show_progress=False,
        )


def test_write_rejects_empty(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_dataset({}, tmp_path / "empty.cstore",
                      batch_size=10, show_progress=False)


def test_dtype_is_preserved_byte_for_byte(tmp_path):
    """Round-trip every supported NumPy fixed-size dtype."""
    path = tmp_path / "all_dtypes.cstore"
    rng = np.random.default_rng(0)
    columns = {
        "f32": rng.standard_normal(64).astype(np.float32),
        "f64": rng.standard_normal(64).astype(np.float64),
        "i8": rng.integers(-128, 127, size=64, dtype=np.int8),
        "i16": rng.integers(-1000, 1000, size=64, dtype=np.int16),
        "i32": rng.integers(-1_000_000, 1_000_000, size=64, dtype=np.int32),
        "i64": rng.integers(-(2**40), 2**40, size=64, dtype=np.int64),
        "u8": rng.integers(0, 255, size=64, dtype=np.uint8),
        "u16": rng.integers(0, 2**16 - 1, size=64, dtype=np.uint16),
        "u32": rng.integers(0, 2**31, size=64, dtype=np.uint32),
        "u64": rng.integers(0, 2**40, size=64, dtype=np.uint64),
    }
    write_dataset(columns, path, batch_size=32, show_progress=False)
    manifest, data_offset = read_header(path)
    layout = build_column_layout(manifest, data_offset)

    raw_bytes = path.read_bytes()
    for name, expected in columns.items():
        offset, stored_dtype = layout[name]
        assert stored_dtype == expected.dtype
        recovered = np.frombuffer(
            raw_bytes,
            dtype=stored_dtype,
            count=expected.shape[0],
            offset=offset,
        )
        assert np.array_equal(recovered, expected)
