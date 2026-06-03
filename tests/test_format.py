"""Tests for the on-disk file format.

Covers the header (layout, alignment, magic), write-time validation, dtype
preservation and support (numeric, strings, datetime, byte order), header
integrity checks (version, checksum, truncation), reserved manifest keys, and
the no-op batching behaviour of ``batch_size``.
"""

from __future__ import annotations

import json
import struct
import warnings

import numpy as np
import pytest

from colstore import FILE_EXTENSION, ColStore, FormatError
from colstore import format as fmt
from colstore.format import (
    align_up,
    build_column_layout,
    read_header,
    write_dataset,
)
from colstore.kernels import cpp_available, numba_available

_BACKENDS = ["numpy"]
if cpp_available():
    _BACKENDS.append("cpp")
if numba_available():
    _BACKENDS.append("numba")


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
    assert manifest["n_records"] == 1
    assert manifest["committed_rows"] == 100
    assert [c["name"] for c in manifest["columns"]] == ["x", "y"]
    assert data_offset % 64 == 0


def test_column_layout_offsets_are_contiguous(tmp_path):
    path = tmp_path / "case.cstore"
    columns = {
        "a": np.arange(10, dtype=np.float32),  # 40 bytes
        "b": np.arange(10, dtype=np.int64),  # 80 bytes
    }
    write_dataset(columns, path, batch_size=10, show_progress=False)
    manifest, data_offset = read_header(path)
    n_rows = int(manifest["committed_rows"])
    body_offset = data_offset + fmt._RECORD_HEADER_SIZE
    layout = build_column_layout(manifest, body_offset, n_rows)
    a_offset, a_dtype = layout["a"]
    b_offset, _b_dtype = layout["b"]
    # Columns start at the record body offset (data_offset + 32).
    assert a_offset == body_offset
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
        write_dataset({}, tmp_path / "empty.cstore", batch_size=10, show_progress=False)


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
    n_rows = int(manifest["committed_rows"])
    layout = build_column_layout(manifest, data_offset + fmt._RECORD_HEADER_SIZE, n_rows)

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


# ---- Dtype support and byte order ------------------------------------------


def test_write_rejects_unsupported_dtype_kind(tmp_path):
    with pytest.raises(TypeError, match="unsupported dtype kind"):
        write_dataset(
            {"z": np.array([1 + 2j], dtype=np.complex128)},
            tmp_path / "z.cstore",
            batch_size=None,
            show_progress=False,
        )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_fixed_width_bytes_roundtrip(tmp_path, backend):
    columns = {"name": np.array([b"alice", b"bob", b"carol"], dtype="S8")}
    store = ColStore.from_dict(columns, tmp_path / "s.cstore", show_progress=False, backend=backend)
    result = store[np.array([2, 0]), "name"].to_array()
    assert result.tolist() == [b"carol", b"alice"]
    store.close()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_fixed_width_unicode_roundtrip(tmp_path, backend):
    columns = {"label": np.array(["alpha", "beta", "gamma"], dtype="U10")}
    store = ColStore.from_dict(columns, tmp_path / "u.cstore", show_progress=False, backend=backend)
    assert store[1:3, "label"].to_array().tolist() == ["beta", "gamma"]
    # Fancy index exercises the kernel-fallback path for unicode.
    assert store[np.array([2, 0]), "label"].to_array().tolist() == ["gamma", "alpha"]
    store.close()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_datetime64_roundtrip(tmp_path, backend):
    values = np.array(["2020-01-01", "2021-06-15"], dtype="datetime64[ns]")
    # cpp/numba backends must not warn here; they silently fall back to NumPy.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store = ColStore.from_dict(
            {"t": values}, tmp_path / "dt.cstore", show_progress=False, backend=backend
        )
        result = store[np.array([1, 0]), "t"].to_array()
    assert np.array_equal(result, values[[1, 0]])
    store.close()


def test_big_endian_input_stored_little_endian(tmp_path):
    path = tmp_path / "be.cstore"
    write_dataset({"v": np.arange(5, dtype=">i4")}, path, batch_size=None, show_progress=False)
    manifest, _ = read_header(path)
    assert manifest["columns"][0]["dtype"] == "<i4"
    store = ColStore(path, backend="numpy")
    assert store["v"].to_array().tolist() == [0, 1, 2, 3, 4]
    assert store.dtypes["v"].byteorder in ("=", "<", "|")
    store.close()


# ---- Reserved manifest keys ------------------------------------------------


def test_manifest_has_reserved_keys(tmp_path):
    path = tmp_path / "keys.cstore"
    write_dataset({"x": np.arange(4, dtype=np.float64)}, path, batch_size=None, show_progress=False)
    manifest, _ = read_header(path)
    column = manifest["columns"][0]
    assert column["encoding"] == "raw"
    assert column["nullable"] is False


# ---- Header integrity: version, checksum, truncation -----------------------


def test_checksum_is_written(tmp_path):
    path = tmp_path / "k.cstore"
    write_dataset({"x": np.arange(4, dtype=np.float64)}, path, batch_size=None, show_progress=False)
    manifest, _ = read_header(path)
    assert "manifest_crc32" in manifest


def test_truncated_file_raises(tmp_path):
    path = tmp_path / "t.cstore"
    write_dataset(
        {"x": np.arange(100, dtype=np.float64)}, path, batch_size=None, show_progress=False
    )
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(FormatError, match="truncated"):
        ColStore(path)


def test_corrupt_manifest_checksum_raises(tmp_path):
    path = tmp_path / "c.cstore"
    write_dataset(
        {"alpha": np.arange(3, dtype=np.float64)}, path, batch_size=None, show_progress=False
    )
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
    write_dataset({"x": np.arange(4, dtype=np.float64)}, path, batch_size=None, show_progress=False)
    manifest, data_offset = read_header(path)
    manifest["format_version"] = 999
    manifest["manifest_crc32"] = fmt._manifest_checksum(
        manifest["columns"], manifest["n_records"], manifest["committed_rows"]
    )
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    header_size = len(fmt._MAGIC) + fmt._MANIFEST_LEN_SIZE + len(manifest_bytes)
    new_offset = align_up(header_size)
    # Everything past data_offset (record header + body) is preserved verbatim.
    body_bytes = path.read_bytes()[data_offset:]
    with open(path, "wb") as handle:
        handle.write(fmt._MAGIC)
        handle.write(struct.pack(fmt._MANIFEST_LEN_FMT, len(manifest_bytes)))
        handle.write(manifest_bytes)
        handle.write(b"\x00" * (new_offset - header_size))
        handle.write(body_bytes)
    with pytest.raises(FormatError, match="format_version"):
        ColStore(path)


# ---- Batching is a no-op on output bytes -----------------------------------


@pytest.mark.parametrize("batch_size", [None, -1, 0])
def test_unbatched_write_matches_batched(tmp_path, batch_size):
    columns = {"x": np.arange(1000, dtype=np.float32)}
    batched = tmp_path / "batched.cstore"
    unbatched = tmp_path / "unbatched.cstore"
    write_dataset(columns, batched, batch_size=100, show_progress=False)
    write_dataset(columns, unbatched, batch_size=batch_size, show_progress=False)
    assert batched.read_bytes() == unbatched.read_bytes()
