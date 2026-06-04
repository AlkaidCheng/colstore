"""Tests for the on-disk file format.

Covers the header (layout, alignment, magic), write-time validation, dtype
preservation and support (numeric, strings, datetime, byte order), header
integrity checks (version, checksum, truncation), reserved manifest keys, and
the no-op batching behaviour of ``batch_size``.
"""

from __future__ import annotations

import itertools
import json
import struct
import warnings

import numpy as np
import pytest

import colstore
from colstore import FILE_EXTENSION, ColStoreReader, FormatError
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
    store = colstore.store(columns, tmp_path / "s.cstore", show_progress=False, backend=backend)
    result = store[np.array([2, 0]), "name"].array()
    assert result.tolist() == [b"carol", b"alice"]
    store.close()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_fixed_width_unicode_roundtrip(tmp_path, backend):
    columns = {"label": np.array(["alpha", "beta", "gamma"], dtype="U10")}
    store = colstore.store(columns, tmp_path / "u.cstore", show_progress=False, backend=backend)
    assert store[1:3, "label"].array().tolist() == ["beta", "gamma"]
    # Fancy index exercises the kernel-fallback path for unicode.
    assert store[np.array([2, 0]), "label"].array().tolist() == ["gamma", "alpha"]
    store.close()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_datetime64_roundtrip(tmp_path, backend):
    values = np.array(["2020-01-01", "2021-06-15"], dtype="datetime64[ns]")
    # cpp/numba backends must not warn here; they silently fall back to NumPy.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store = colstore.store(
            {"t": values}, tmp_path / "dt.cstore", show_progress=False, backend=backend
        )
        result = store[np.array([1, 0]), "t"].array()
    assert np.array_equal(result, values[[1, 0]])
    store.close()


def test_big_endian_input_stored_little_endian(tmp_path):
    path = tmp_path / "be.cstore"
    write_dataset({"v": np.arange(5, dtype=">i4")}, path, batch_size=None, show_progress=False)
    manifest, _ = read_header(path)
    assert manifest["columns"][0]["dtype"] == "<i4"
    store = ColStoreReader(path, backend="numpy")
    assert store["v"].array().tolist() == [0, 1, 2, 3, 4]
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
        ColStoreReader(path)


def test_corrupt_manifest_checksum_raises(tmp_path):
    path = tmp_path / "c.cstore"
    write_dataset(
        {"alpha": np.arange(3, dtype=np.float64)}, path, batch_size=None, show_progress=False
    )
    raw = bytearray(path.read_bytes())
    # Layout: magic (8) + counters (32) + manifest_len (8) + manifest (...).
    manifest_len_offset = len(fmt._MAGIC) + fmt._COUNTERS_SIZE
    manifest_offset = manifest_len_offset + fmt._MANIFEST_LEN_SIZE
    manifest_size = struct.unpack("<Q", raw[manifest_len_offset:manifest_offset])[0]
    manifest = json.loads(raw[manifest_offset : manifest_offset + manifest_size])
    # Equal-length edit: change content without resizing the manifest.
    manifest["columns"][0]["name"] = "ALPHA"
    edited = json.dumps(manifest).encode("utf-8")
    assert len(edited) == manifest_size
    raw[manifest_offset : manifest_offset + manifest_size] = edited
    path.write_bytes(bytes(raw))
    with pytest.raises(FormatError, match="checksum"):
        ColStoreReader(path)


def test_corrupt_counters_block_raises(tmp_path):
    """The counters block (n_records + committed_rows + CRC) sits at fixed
    offset 8. A bit-flip there must surface as a clean FormatError before
    the manifest is even parsed.
    """
    path = tmp_path / "ct.cstore"
    write_dataset({"x": np.arange(3, dtype=np.float64)}, path, batch_size=None, show_progress=False)
    raw = bytearray(path.read_bytes())
    # Flip a bit inside the n_records field. The counters CRC will mismatch.
    raw[len(fmt._MAGIC)] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(FormatError, match=r"[Cc]ounters"):
        ColStoreReader(path)


def test_unsupported_version_raises(tmp_path):
    path = tmp_path / "v.cstore"
    write_dataset({"x": np.arange(4, dtype=np.float64)}, path, batch_size=None, show_progress=False)
    manifest, data_offset = read_header(path)
    n_records = manifest["n_records"]
    committed_rows = manifest["committed_rows"]
    # Rebuild the JSON manifest with format_version corrupted to an unsupported
    # value; the counters block stays valid (we just want to trip the version
    # check, not the CRC check).
    bad_manifest = {
        "format_version": 999,
        "columns": manifest["columns"],
        "manifest_crc32": fmt._manifest_checksum(manifest["columns"]),
    }
    manifest_bytes = json.dumps(bad_manifest).encode("utf-8")
    header_size = (
        len(fmt._MAGIC) + fmt._COUNTERS_SIZE + fmt._MANIFEST_LEN_SIZE + len(manifest_bytes)
    )
    new_offset = align_up(header_size)
    # Everything past data_offset (record header + body) is preserved verbatim.
    body_bytes = path.read_bytes()[data_offset:]
    with open(path, "wb") as handle:
        handle.write(fmt._MAGIC)
        handle.write(fmt._pack_counters(n_records, committed_rows))
        handle.write(struct.pack(fmt._MANIFEST_LEN_FMT, len(manifest_bytes)))
        handle.write(manifest_bytes)
        handle.write(b"\x00" * (new_offset - header_size))
        handle.write(body_bytes)
    with pytest.raises(FormatError, match="format_version"):
        ColStoreReader(path)


# ---- Batching is a no-op on output bytes -----------------------------------


@pytest.mark.parametrize("batch_size", [None, -1, 0])
def test_unbatched_write_matches_batched(tmp_path, batch_size):
    columns = {"x": np.arange(1000, dtype=np.float32)}
    batched = tmp_path / "batched.cstore"
    unbatched = tmp_path / "unbatched.cstore"
    write_dataset(columns, batched, batch_size=100, show_progress=False)
    write_dataset(columns, unbatched, batch_size=batch_size, show_progress=False)
    assert batched.read_bytes() == unbatched.read_bytes()


# ---- Polymorphic batch_size: bytes per batch ------------------------------


def test_string_batch_size_matches_int_byte_equivalent(tmp_path):
    """A '4 KB' batch should produce the same bytes as the int equivalent."""
    # 1000 float32 values across one column = 4000 bytes total.
    # "4 KB" = 4096 bytes per batch -> single batch per column.
    # Int 1000 with 1 column = 1000 rows per inner step -> single batch per column.
    columns = {"x": np.arange(1000, dtype=np.float32)}
    a = tmp_path / "a.cstore"
    b = tmp_path / "b.cstore"
    write_dataset(columns, a, batch_size="4 KB", show_progress=False)
    write_dataset(columns, b, batch_size=1000, show_progress=False)
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize("batch_size", ["1 KB", "1 KiB", "1024 B", "1024"])
def test_string_batch_size_accepts_various_units(tmp_path, batch_size):
    """All these should mean exactly 1024 bytes per batch."""
    columns = {"x": np.arange(2000, dtype=np.int32)}  # 8 KB total
    path = tmp_path / "x.cstore"
    write_dataset(columns, path, batch_size=batch_size, show_progress=False)
    # Just verify it wrote a valid file; bytes-equivalence to the int form
    # is covered by other tests.
    manifest, _ = read_header(path)
    assert manifest["committed_rows"] == 2000


def test_auto_batch_size_writes_valid_file(tmp_path):
    """The 'auto' default produces a byte-identical file to None (single-pass)."""
    columns = {
        "x": np.arange(500, dtype=np.float64),
        "y": np.arange(500, dtype=np.int32),
    }
    auto = tmp_path / "auto.cstore"
    none = tmp_path / "none.cstore"
    write_dataset(columns, auto, batch_size="auto", show_progress=False)
    write_dataset(columns, none, batch_size=None, show_progress=False)
    # For a tiny dataset, auto falls back to single-pass, so bytes match.
    assert auto.read_bytes() == none.read_bytes()


def test_auto_batch_size_above_threshold_takes_adaptive_path(tmp_path):
    """A dataset above the auto-batching threshold writes correct bytes adaptively.

    Doesn't assert specific batch sizes (those are bandwidth-dependent and
    timing-flaky); just verifies the output matches the single-pass reference.
    """
    # 20 MiB of float64 -> well above the 16 MiB single-pass threshold,
    # but small enough to keep the test fast.
    n_rows = 20 * 1024 * 1024 // 8  # 20 MiB / 8 bytes per float64
    columns = {"x": np.arange(n_rows, dtype=np.float64)}
    adaptive = tmp_path / "adaptive.cstore"
    reference = tmp_path / "reference.cstore"
    write_dataset(columns, adaptive, batch_size="auto", show_progress=False)
    write_dataset(columns, reference, batch_size=None, show_progress=False)
    assert adaptive.read_bytes() == reference.read_bytes()


def test_int_batch_size_divides_across_columns(tmp_path):
    """``batch_size=N`` with C columns means N/C rows per inner step."""
    # 5 columns x 100 rows. batch_size=50 -> rows_per_step=10 per column.
    # Output bytes are independent of step count, so compare to single-pass.
    columns = {f"c{i}": np.arange(100, dtype=np.float64) for i in range(5)}
    batched = tmp_path / "b.cstore"
    one_shot = tmp_path / "o.cstore"
    write_dataset(columns, batched, batch_size=50, show_progress=False)
    write_dataset(columns, one_shot, batch_size=None, show_progress=False)
    assert batched.read_bytes() == one_shot.read_bytes()


def test_string_batch_size_handles_mixed_dtypes(tmp_path):
    """Bytes-per-batch with mixed dtypes still writes correct bytes."""
    columns = {
        "tiny": np.arange(500, dtype=np.int8),  # 500 bytes
        "big": np.arange(500, dtype=np.float64),  # 4000 bytes
    }
    a = tmp_path / "a.cstore"
    b = tmp_path / "b.cstore"
    write_dataset(columns, a, batch_size="256 B", show_progress=False)
    write_dataset(columns, b, batch_size=None, show_progress=False)
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize("bad", [1.5, [100], {"size": 100}, object()])
def test_batch_size_rejects_non_int_non_str(tmp_path, bad):
    columns = {"x": np.arange(10, dtype=np.int32)}
    path = tmp_path / "bad.cstore"
    with pytest.raises(TypeError, match="batch_size"):
        write_dataset(columns, path, batch_size=bad, show_progress=False)


def test_batch_size_rejects_bool(tmp_path):
    """``True`` would silently mean 1 row per step; reject it explicitly."""
    columns = {"x": np.arange(10, dtype=np.int32)}
    path = tmp_path / "bool.cstore"
    with pytest.raises(TypeError, match="batch_size"):
        write_dataset(columns, path, batch_size=True, show_progress=False)


def test_string_batch_size_rejects_unparseable(tmp_path):
    columns = {"x": np.arange(10, dtype=np.int32)}
    path = tmp_path / "bad.cstore"
    with pytest.raises(ValueError, match="Cannot parse byte size"):
        write_dataset(columns, path, batch_size="not a size", show_progress=False)


def test_string_batch_size_rejects_unknown_unit(tmp_path):
    columns = {"x": np.arange(10, dtype=np.int32)}
    path = tmp_path / "bad.cstore"
    with pytest.raises(ValueError, match="Unknown unit"):
        write_dataset(columns, path, batch_size="100 XB", show_progress=False)


# ---- Unit tests for the parser/resolver/formatter internals ---------------


def test_parse_byte_size_basic_units():
    assert fmt._parse_byte_size("100") == 100
    assert fmt._parse_byte_size("100 B") == 100
    assert fmt._parse_byte_size("1 KB") == 1024
    assert fmt._parse_byte_size("1 KiB") == 1024
    assert fmt._parse_byte_size("1 MB") == 1024**2
    assert fmt._parse_byte_size("1 MiB") == 1024**2
    assert fmt._parse_byte_size("1 GB") == 1024**3
    assert fmt._parse_byte_size("1.5 MB") == int(1.5 * 1024**2)


def test_parse_byte_size_whitespace_and_case():
    assert fmt._parse_byte_size("  100mb  ") == 100 * 1024**2
    assert fmt._parse_byte_size("100MIB") == 100 * 1024**2
    assert fmt._parse_byte_size("100m") == 100 * 1024**2


def test_resolve_rows_per_step_returns_per_column_list():
    # 2 columns, different itemsizes, bytes-per-step uses each column's
    # itemsize to compute its rows_per_step.
    rows_per_step = fmt._resolve_rows_per_step(
        "1024 B",
        n_rows=1000,
        n_columns=2,
        total_bytes=12_000,
        column_itemsizes=[1, 8],  # int8 + float64
    )
    # 1024 / 1 = 1024 rows per step for the int8 column
    # 1024 / 8 = 128 rows per step for the float64 column
    assert rows_per_step == [1024, 128]


def test_resolve_rows_per_step_int_divides_by_columns():
    # batch_size=100 with 5 columns -> 20 rows per inner step.
    rows_per_step = fmt._resolve_rows_per_step(
        100,
        n_rows=1000,
        n_columns=5,
        total_bytes=20_000,
        column_itemsizes=[4] * 5,
    )
    assert rows_per_step == [20, 20, 20, 20, 20]


def test_resolve_rows_per_step_zero_rows_returns_single_pass():
    rows_per_step = fmt._resolve_rows_per_step(
        100,
        n_rows=0,
        n_columns=3,
        total_bytes=0,
        column_itemsizes=[4, 4, 4],
    )
    assert rows_per_step == [0, 0, 0]


def test_format_bytes_per_sec_auto_scales():
    assert fmt._format_bytes_per_sec(0) == "0 B/s"
    assert fmt._format_bytes_per_sec(512) == "512 B/s"
    assert fmt._format_bytes_per_sec(2048) == "2.00 KB/s"
    assert fmt._format_bytes_per_sec(1.5 * 1024**2) == "1.50 MB/s"
    assert fmt._format_bytes_per_sec(2.5 * 1024**3) == "2.50 GB/s"
    # Very large bandwidth stays in GB/s (no TB/s tier yet -- not needed).
    assert fmt._format_bytes_per_sec(1024**4).endswith("GB/s")


def test_format_rows_per_sec_auto_scales():
    assert fmt._format_rows_per_sec(0) == "0 rows/s"
    assert fmt._format_rows_per_sec(500) == "500 rows/s"
    assert fmt._format_rows_per_sec(1500) == "1.50 Krows/s"
    assert fmt._format_rows_per_sec(2_500_000) == "2.50 Mrows/s"
    assert fmt._format_rows_per_sec(3.5 * 1_000_000_000) == "3.50 Grows/s"


def test_auto_adaptive_constants_form_a_sensible_ramp():
    """The growth-rate cap means a wildly-wrong first measurement is bounded.

    Even if the first batch's measured bandwidth implies a target batch
    size of 100 GiB (e.g., from OS cache absorbing the probe), the next
    batch can be at most _AUTO_GROWTH_RATE * _AUTO_INITIAL_BYTES. A few
    iterations are needed to ramp up to large steady-state sizes; that's
    the price of robustness.
    """
    # Simulate "bandwidth estimate suggests target = 100 GiB" by checking
    # that the growth cap dominates when target_bytes is huge.
    initial = fmt._AUTO_INITIAL_BYTES
    growth = fmt._AUTO_GROWTH_RATE
    huge_target = 100 * 1024**3  # 100 GiB

    # Apply the same min/max/cap formula used in write_dataset:
    next_size = max(
        fmt._AUTO_MIN_BYTES_PER_BATCH,
        min(huge_target, int(initial * growth)),
    )
    # With initial=1MiB and growth=2, even a 100 GiB target only gives 2 MiB.
    assert next_size == int(initial * growth)
    assert next_size <= initial * 2  # bounded ramp-up


def test_estimate_total_batches_zero_remaining_returns_done():
    assert fmt._estimate_total_batches(batches_done=5, remaining_bytes=0, bytes_per_batch=1024) == 5
    assert (
        fmt._estimate_total_batches(batches_done=5, remaining_bytes=-100, bytes_per_batch=1024) == 5
    )


def test_estimate_total_batches_ceil_division():
    # 10 batches remaining at exactly 1 MiB each
    one_mib = 1024 * 1024
    assert (
        fmt._estimate_total_batches(
            batches_done=3, remaining_bytes=10 * one_mib, bytes_per_batch=one_mib
        )
        == 13
    )
    # 10.5 MiB / 1 MiB per batch -> 11 more batches (ceiling), not 10
    assert (
        fmt._estimate_total_batches(
            batches_done=3, remaining_bytes=10 * one_mib + 1, bytes_per_batch=one_mib
        )
        == 14
    )


def test_estimate_total_batches_zero_bytes_per_batch_falls_back_to_plus_one():
    # Degenerate input: report at least one more batch rather than 0/inf.
    assert fmt._estimate_total_batches(batches_done=7, remaining_bytes=1024, bytes_per_batch=0) == 8


def test_estimate_total_batches_converges_as_batch_size_grows():
    """The whole point: as bytes_per_batch converges to steady-state size,
    the total estimate converges. During ramp-up, the estimate gets
    smaller (because future batches will be bigger than past ones).
    """
    total_bytes = 80 * 1024 * 1024  # 80 MiB
    # Simulated ramp: 1 MiB, 2 MiB, 4 MiB, 8 MiB, 16 MiB, then steady at ~16 MiB.
    one_mib = 1024 * 1024
    ramp = [1, 2, 4, 8, 16, 16, 16]
    body_bytes = 0
    estimates = []
    for batch_idx, mib in enumerate(ramp):
        body_bytes += mib * one_mib
        if body_bytes >= total_bytes:
            break
        next_bytes = mib * 2 * one_mib if mib < 16 else 16 * one_mib
        estimates.append(
            fmt._estimate_total_batches(
                batches_done=batch_idx + 1,
                remaining_bytes=total_bytes - body_bytes,
                bytes_per_batch=next_bytes,
            )
        )
    # Each successive estimate should be <= the previous one as batch
    # size grows (or equal once steady state is reached).
    for a, b in itertools.pairwise(estimates):
        assert b <= a, f"estimate should not grow during ramp: {estimates}"
    # Once batch size stabilizes (last three ramp entries are all 16 MiB),
    # the estimate stabilizes too: the last few values must agree.
    assert (
        estimates[-3:] == [estimates[-1]] * 3
    ), f"estimate should freeze in steady state: {estimates}"


def test_auto_progress_total_is_defined_throughout(tmp_path, monkeypatch):
    """Regression guard: adaptive writes must give tqdm a non-None total.

    The earlier version of adaptive mode passed ``total=None`` to tqdm,
    which rendered as ``204/?`` -- no ETA, no progress fraction. The fix
    seeds the total with a coarse estimate and refines it after each
    batch during the ramp.
    """
    from contextlib import contextmanager

    from colstore.progress import NullProgressBar

    captured_totals: list[int | None] = []

    @contextmanager
    def recording_progress_bar(total, **kwargs):
        captured_totals.append(total)
        # Yield a NullProgressBar so the write loop's progress.total =
        # ... assignments land somewhere harmless. We only care about
        # the initial total passed in.
        yield NullProgressBar()

    monkeypatch.setattr("colstore.format.progress_bar", recording_progress_bar)

    # 20 MiB float64 -> above the 16 MiB adaptive threshold.
    n_rows = 20 * 1024 * 1024 // 8
    columns = {"x": np.arange(n_rows, dtype=np.float64)}
    write_dataset(columns, tmp_path / "x.cstore", batch_size="auto", show_progress=True)

    assert captured_totals, "progress_bar was never invoked"
    initial = captured_totals[0]
    assert initial is not None, "adaptive mode passed total=None to progress_bar"
    assert initial > 0, f"adaptive initial total should be positive, got {initial}"
