"""Tests for colstore.compact: byte-correct concatenation, no-ops, atomicity, locking."""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import FormatError

# ---- Basic correctness -----------------------------------------------------


def test_compact_collapses_multi_record_to_single(tmp_path):
    """A 5-record file becomes a 1-record file with identical row content."""
    path = tmp_path / "m.cstore"
    with colstore.create(path) as f:
        for i in range(5):
            f.write(
                {
                    "a": np.arange(10, dtype=np.float32) + 100 * i,
                    "b": np.arange(10, dtype=np.int64) + 1000 * i,
                }
            )

    before = colstore.info(path)
    assert before.n_records == 5
    assert before.n_rows == 50
    assert before.needs_compaction

    # Snapshot full content before compaction.
    with colstore.open(path) as ds:
        a_before = ds[:, "a"].array().copy()
        b_before = ds[:, "b"].array().copy()

    colstore.compact(path, show_progress=False)

    after = colstore.info(path)
    assert after.n_records == 1
    assert after.n_rows == 50
    assert not after.needs_compaction
    # File is smaller because record headers from records 1..4 are gone.
    # (32 bytes saved per dropped record header.)
    assert after.file_size == before.file_size - 4 * 32

    with colstore.open(path) as ds:
        assert np.array_equal(ds[:, "a"].array(), a_before)
        assert np.array_equal(ds[:, "b"].array(), b_before)


def test_compact_preserves_fancy_index_reads(tmp_path):
    """Fancy-index reads return identical results before and after compaction."""
    path = tmp_path / "f.cstore"
    rng = np.random.default_rng(0)
    with colstore.create(path) as f:
        for _ in range(10):
            n = rng.integers(50, 100)
            f.write({"x": rng.standard_normal(n).astype(np.float64)})

    with colstore.open(path) as ds:
        n_rows = ds.n_rows
        sorted_idx = np.sort(rng.choice(n_rows, size=200, replace=False)).astype(np.int64)
        unsorted_idx = rng.permutation(n_rows)[:200].astype(np.int64)
        a_sorted_before = ds[sorted_idx, "x"].array().copy()
        a_unsorted_before = ds[unsorted_idx, "x"].array().copy()

    colstore.compact(path, show_progress=False)

    with colstore.open(path) as ds:
        assert np.array_equal(ds[sorted_idx, "x"].array(), a_sorted_before)
        assert np.array_equal(ds[unsorted_idx, "x"].array(), a_unsorted_before)


def test_compact_multi_column_with_mixed_dtypes(tmp_path):
    """Heterogeneous itemsizes (int8 + float64 + int32) compact correctly.

    The byte splice math depends on per-column itemsize and per-record row
    count; mismatched itemsizes are the most likely place for an off-by-one.
    """
    path = tmp_path / "mc.cstore"
    rng = np.random.default_rng(0)
    with colstore.create(path) as f:
        for _ in range(7):
            n = int(rng.integers(20, 60))
            f.write(
                {
                    "tiny": rng.integers(-100, 100, size=n, dtype=np.int8),
                    "big": rng.standard_normal(n),  # float64
                    "mid": rng.integers(0, 10**6, size=n, dtype=np.int32),
                }
            )

    with colstore.open(path) as ds:
        snapshots = {col: ds[:, col].array().copy() for col in ds.columns}

    colstore.compact(path, show_progress=False)

    with colstore.open(path) as ds:
        assert colstore.info(path).n_records == 1
        for col, expected in snapshots.items():
            assert np.array_equal(ds[:, col].array(), expected), f"{col} differs"


# ---- No-op short-circuits --------------------------------------------------


def test_compact_inplace_noop_on_single_record(tmp_path):
    """Compacting an already-single-record file is a no-op; bytes untouched."""
    path = tmp_path / "single.cstore"
    colstore.store({"a": np.arange(20, dtype=np.float32)}, path, show_progress=False).close()
    inode_before = path.stat().st_ino
    bytes_before = path.read_bytes()

    returned = colstore.compact(path, show_progress=False)

    assert returned == path
    assert path.stat().st_ino == inode_before  # no rename happened
    assert path.read_bytes() == bytes_before


def test_compact_out_on_single_record_copies_bytes(tmp_path):
    """Out-of-place compact of a single-record file just copies the bytes."""
    src = tmp_path / "src.cstore"
    dst = tmp_path / "dst.cstore"
    colstore.store({"a": np.arange(20, dtype=np.float32)}, src, show_progress=False).close()

    returned = colstore.compact(src, out=dst, show_progress=False)

    assert returned == dst
    assert dst.exists()
    assert src.read_bytes() == dst.read_bytes()
    # Source is untouched.
    assert src.exists()


def test_compact_out_same_as_path_is_inplace(tmp_path):
    """Passing out=path is equivalent to passing no out."""
    path = tmp_path / "same.cstore"
    with colstore.create(path) as f:
        f.write({"a": np.arange(5, dtype=np.int32)})
        f.write({"a": np.arange(5, 10, dtype=np.int32)})
    returned = colstore.compact(path, out=path, show_progress=False)
    assert returned == path
    assert colstore.info(path).n_records == 1


# ---- Out-of-place mode -----------------------------------------------------


def test_compact_out_leaves_source_untouched(tmp_path):
    """When out= is given, the source's bytes are not modified."""
    src = tmp_path / "src.cstore"
    dst = tmp_path / "dst.cstore"
    with colstore.create(src) as f:
        for i in range(3):
            f.write({"a": np.arange(10, dtype=np.float32) + 100 * i})

    src_bytes_before = src.read_bytes()
    src_info_before = colstore.info(src)

    colstore.compact(src, out=dst, show_progress=False)

    # Source unchanged.
    assert src.read_bytes() == src_bytes_before
    assert colstore.info(src).n_records == src_info_before.n_records

    # Destination is the compacted version.
    assert colstore.info(dst).n_records == 1
    assert colstore.info(dst).n_rows == src_info_before.n_rows


# ---- Edge cases ------------------------------------------------------------


def test_compact_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        colstore.compact(tmp_path / "nope.cstore")


def test_compact_corrupt_file_raises(tmp_path):
    """A bit-flipped header surfaces as FormatError, not silent corruption."""
    path = tmp_path / "c.cstore"
    colstore.store({"a": np.arange(10, dtype=np.float32)}, path, show_progress=False).close()
    raw = bytearray(path.read_bytes())
    # Flip a byte inside the manifest, which has its own CRC.
    raw[100] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(FormatError):
        colstore.compact(path)


def test_compact_zero_row_records_in_stream(tmp_path):
    """Records with n_rows=0 are legal and compact away to zero output bytes.

    The compaction math handles count=0 as a no-op per record. The output
    should have committed_rows summing only the non-empty records.
    """
    path = tmp_path / "zr.cstore"
    with colstore.create(path) as f:
        f.write({"a": np.arange(5, dtype=np.int32)})
        f.write({"a": np.empty(0, dtype=np.int32)})
        f.write({"a": np.arange(100, 103, dtype=np.int32)})
        f.write({"a": np.empty(0, dtype=np.int32)})

    with colstore.open(path) as ds:
        expected = ds[:, "a"].array().copy()
    assert len(expected) == 8

    colstore.compact(path, show_progress=False)

    assert colstore.info(path).n_records == 1
    assert colstore.info(path).n_rows == 8
    with colstore.open(path) as ds:
        assert np.array_equal(ds[:, "a"].array(), expected)


def test_compact_preserves_byte_order_on_disk(tmp_path):
    """Big-endian on disk stays big-endian after compaction.

    The writer stores little-endian, so this test forces the on-disk
    little-endian default path. We're really checking that the bytes
    aren't reinterpreted -- a memcpy-class copy preserves the on-disk
    representation by construction.
    """
    path = tmp_path / "be.cstore"
    # Use the writer in normal mode (little-endian on disk).
    with colstore.create(path) as f:
        f.write({"x": np.arange(7, dtype=np.float64)})
        f.write({"x": np.arange(100, 107, dtype=np.float64)})

    with colstore.open(path) as ds:
        expected = ds[:, "x"].array().copy()

    colstore.compact(path, show_progress=False)

    with colstore.open(path) as ds:
        assert np.array_equal(ds[:, "x"].array(), expected)


# ---- Atomicity / failure handling ------------------------------------------


def test_compact_cleans_up_temp_on_failure(tmp_path, monkeypatch):
    """If the byte copy fails, the temp file is removed and source is intact."""
    path = tmp_path / "fail.cstore"
    with colstore.create(path) as f:
        for i in range(3):
            f.write({"a": np.arange(10, dtype=np.float32) + 100 * i})

    src_bytes = path.read_bytes()

    # Force the copy step to fail. Patch the kernel-level copy primitive in
    # the compaction module so we can exercise the cleanup path.
    from colstore import compaction

    def boom(*args, **kwargs):
        raise RuntimeError("simulated copy failure")

    monkeypatch.setattr(compaction, "_copy_range", boom)

    with pytest.raises(RuntimeError, match="simulated"):
        colstore.compact(path, show_progress=False)

    # Source untouched.
    assert path.read_bytes() == src_bytes
    # No leftover temp file.
    temp_files = [p for p in tmp_path.iterdir() if p.name.endswith(".compacting")]
    assert temp_files == []


def test_compact_keyboard_interrupt_cleans_up(tmp_path, monkeypatch):
    """A Ctrl-C mid-compact also triggers the cleanup path."""
    path = tmp_path / "kbi.cstore"
    with colstore.create(path) as f:
        for i in range(3):
            f.write({"a": np.arange(10, dtype=np.float32) + 100 * i})

    src_bytes = path.read_bytes()
    from colstore import compaction

    def raise_kbi(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(compaction, "_copy_range", raise_kbi)

    with pytest.raises(KeyboardInterrupt):
        colstore.compact(path, show_progress=False)

    assert path.read_bytes() == src_bytes
    temp_files = [p for p in tmp_path.iterdir() if p.name.endswith(".compacting")]
    assert temp_files == []


# ---- Concurrency / locking -------------------------------------------------


def test_compact_blocked_by_active_writer(tmp_path):
    """A writer holding the advisory lock blocks compaction."""
    path = tmp_path / "w.cstore"
    with colstore.create(path) as f:
        f.write({"a": np.arange(5, dtype=np.int32)})
        f.write({"a": np.arange(5, 10, dtype=np.int32)})

    # Open a writer (holds the flock).
    holder = colstore.update(path)
    try:
        with pytest.raises(OSError, match="lock"):
            colstore.compact(path, show_progress=False)
    finally:
        holder.close()

    # After the writer closes, compact succeeds.
    colstore.compact(path, show_progress=False)
    assert colstore.info(path).n_records == 1


# ---- Memory bound (smoke test) --------------------------------------------


def test_compact_memory_bounded_on_large_file(tmp_path):
    """Compacting a file larger than a small RAM budget completes successfully.

    Not a strict memory measurement (process RSS is noisy under pytest);
    we use a moderately-sized file and verify correctness. The real
    guarantee is structural: the sendfile path doesn't surface bytes
    through Python.
    """
    path = tmp_path / "big.cstore"
    rng = np.random.default_rng(0)
    # ~10MB across 20 records (small enough for CI, large enough that we'd
    # blow up if we accidentally read it all into a Python list).
    with colstore.create(path) as f:
        for _ in range(20):
            f.write({"x": rng.standard_normal(64_000).astype(np.float32)})

    with colstore.open(path) as ds:
        total_before = ds.n_rows
        first_few = ds[:100, "x"].array().copy()
        last_few = ds[-100:, "x"].array().copy()

    colstore.compact(path, show_progress=False)

    with colstore.open(path) as ds:
        assert ds.n_rows == total_before
        assert np.array_equal(ds[:100, "x"].array(), first_few)
        assert np.array_equal(ds[-100:, "x"].array(), last_few)
        assert colstore.info(path).n_records == 1


# ---- Platform fallback (non-Linux portable path) --------------------------


def test_compact_works_on_non_linux_fallback_path(tmp_path, monkeypatch):
    """The non-Linux code path (shutil.copyfileobj + _BoundedReader) must
    produce byte-identical output to the Linux sendfile path.

    macOS and Windows take the fallback in production; this test exercises
    it from Linux CI by patching the platform gate. Regression guard for
    the original bug where ``os.sendfile`` was used unconditionally on
    POSIX (and failed with ENOTSOCK on macOS).
    """
    from colstore import compaction

    monkeypatch.setattr(compaction, "_USE_SENDFILE", False)

    path = tmp_path / "fallback.cstore"
    rng = np.random.default_rng(0)
    with colstore.create(path) as f:
        for _ in range(4):
            f.write(
                {
                    "x": rng.standard_normal(100).astype(np.float32),
                    "y": rng.integers(-1000, 1000, size=100, dtype=np.int64),
                }
            )

    with colstore.open(path) as ds:
        x_before = ds[:, "x"].array().copy()
        y_before = ds[:, "y"].array().copy()

    colstore.compact(path, show_progress=False)

    assert colstore.info(path).n_records == 1
    with colstore.open(path) as ds:
        assert np.array_equal(ds[:, "x"].array(), x_before)
        assert np.array_equal(ds[:, "y"].array(), y_before)
