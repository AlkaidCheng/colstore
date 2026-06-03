"""Tests for multi-record file reads.

A colstore file is a sequence of records, each prefixed by a 32-byte header
and column-major body padded to 8 bytes. Reads of single-record files take a
fast path (per-column memmaps, contiguous gather); reads of multi-record
files take a slower path that bins indices to records via
``np.searchsorted`` and gathers via byte offsets.

PR 2 lands the multi-record reader without a writer that can produce
multi-record files (the writer comes in PR 3). These tests build files by
hand via :mod:`_format_fixture` and exercise both paths.

Coverage matrix:

* Open & introspect: ``n_rows``, ``columns``, ``dtypes``, ``shape``
* Full-table read (``ds[:]``)
* Slice within / spanning records
* Scalar integer index
* Fancy index: sorted / unsorted / spanning records / single-record subset
* Boolean mask
* Multi-column reads (single record and multi)
* Negative indices fold correctly
* Error paths: corrupt record magic, mismatched record_index, CRC mismatch
* Padding: a body whose raw bytes don't naturally divide by 8 still reads OK
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from _format_fixture import expected_column_values, write_record_file

from colstore import ColStoreReader
from colstore import format as fmt
from colstore.format import FormatError

# ---- Helpers ----------------------------------------------------------------


def _make_records(n_records: int, rng_seed: int = 0) -> list[dict[str, np.ndarray]]:
    """Build ``n_records`` records of mixed dtypes with varying row counts.

    Returns records with three columns:
      * ``i32`` -- int32, the most common case.
      * ``f64`` -- float64, exercises 8-byte itemsize.
      * ``i8``  -- int8, exercises 1-byte itemsize (forces non-aligned bodies
        without 8-byte padding).

    Each record has a different ``n_rows`` so the per-record index isn't
    uniform; this catches off-by-one errors in the cumulative-rows math.
    """
    rng = np.random.default_rng(rng_seed)
    base = 0
    records = []
    for i in range(n_records):
        n = 7 + 3 * i  # 7, 10, 13, 16, ... -- varied, deliberately non-power-of-2
        rec = {
            "i32": np.arange(base, base + n, dtype=np.int32),
            "f64": rng.standard_normal(n).astype(np.float64),
            "i8": np.arange(base % 100, base % 100 + n, dtype=np.int8) % 127,
        }
        records.append(rec)
        base += n
    return records


def _schema() -> list[tuple[str, str]]:
    """The schema used by ``_make_records``."""
    return [("i32", "<i4"), ("f64", "<f8"), ("i8", "<i1")]


# ---- Introspection ----------------------------------------------------------


@pytest.mark.parametrize("n_records", [1, 2, 5])
def test_introspection_matches_logical_data(tmp_path, n_records):
    """``n_rows``, ``columns``, ``dtypes``, ``shape`` reflect aggregated state."""
    path = tmp_path / "introspect.cstore"
    records = _make_records(n_records)
    write_record_file(path, _schema(), records)
    expected_total = sum(len(r["i32"]) for r in records)
    with ColStoreReader(path) as ds:
        assert ds.n_rows == expected_total
        assert ds.columns == ["i32", "f64", "i8"]
        assert ds.shape == (expected_total, 3)
        # Native byte order on this host.
        assert ds.dtypes["i32"] == np.dtype("=i4")
        assert ds.dtypes["f64"] == np.dtype("=f8")


@pytest.mark.parametrize("n_records,expected_multi", [(1, False), (2, True), (5, True)])
def test_dispatches_fast_path_for_single_record(tmp_path, n_records, expected_multi):
    """R==1 takes the single-record fast path; R>1 takes the multi-record path."""
    path = tmp_path / "dispatch.cstore"
    write_record_file(path, _schema(), _make_records(n_records))
    with ColStoreReader(path) as ds:
        assert ds._is_multi_record is expected_multi


# ---- Read shapes: full / slice / scalar / fancy / bool ----------------------


@pytest.mark.parametrize("n_records", [1, 3])
def test_full_table_read_concatenates_records(tmp_path, n_records):
    """``ds[:, name]`` returns the logical concatenation across all records."""
    path = tmp_path / "full.cstore"
    records = _make_records(n_records)
    write_record_file(path, _schema(), records)
    with ColStoreReader(path) as ds:
        for name, _ in _schema():
            expected = expected_column_values(records, name)
            got = ds[:, name].to_array()
            assert np.array_equal(got, expected), f"column {name} differs"


@pytest.mark.parametrize("n_records", [1, 3])
def test_slice_within_and_across_records(tmp_path, n_records):
    """Slice reads agree with numpy slicing on the concatenated truth."""
    path = tmp_path / "slice.cstore"
    records = _make_records(n_records)
    write_record_file(path, _schema(), records)
    total = sum(len(r["i32"]) for r in records)
    with ColStoreReader(path) as ds:
        truth = expected_column_values(records, "i32")
        # Slice strictly inside the first record.
        assert np.array_equal(ds[0:3, "i32"].to_array(), truth[0:3])
        # Slice spanning record boundaries (when R > 1).
        assert np.array_equal(ds[2 : total - 2, "i32"].to_array(), truth[2 : total - 2])
        # Full slice.
        assert np.array_equal(ds[0:total, "i32"].to_array(), truth)
        # Empty slice.
        assert np.array_equal(ds[5:5, "i32"].to_array(), truth[5:5])


@pytest.mark.parametrize("n_records", [1, 3])
def test_scalar_integer_index(tmp_path, n_records):
    """Scalar int indexing returns a length-1 array matching the row."""
    path = tmp_path / "scalar.cstore"
    records = _make_records(n_records)
    write_record_file(path, _schema(), records)
    truth = expected_column_values(records, "f64")
    total = truth.shape[0]
    with ColStoreReader(path) as ds:
        assert np.array_equal(ds[0, "f64"].to_array(), truth[0:1])
        # In the middle (likely crosses a record boundary if R > 1).
        mid = total // 2
        assert np.array_equal(ds[mid, "f64"].to_array(), truth[mid : mid + 1])
        # Last row.
        assert np.array_equal(ds[total - 1, "f64"].to_array(), truth[total - 1 : total])


@pytest.mark.parametrize("n_records", [1, 3])
@pytest.mark.parametrize("pattern", ["sorted", "unsorted", "duplicates", "single_record"])
def test_fancy_index_returns_correct_values(tmp_path, n_records, pattern):
    """Fancy-index reads produce the same values as numpy on the logical array."""
    path = tmp_path / f"fancy_{pattern}.cstore"
    records = _make_records(n_records)
    write_record_file(path, _schema(), records)
    truth = expected_column_values(records, "i32")
    total = truth.shape[0]

    rng = np.random.default_rng(42)
    if pattern == "sorted":
        indices = np.sort(rng.choice(total, size=min(20, total), replace=False))
    elif pattern == "unsorted":
        indices = rng.permutation(total)[:10]
    elif pattern == "duplicates":
        indices = np.array([0, 0, total - 1, total // 2, total // 2, 1])
    else:  # single_record -- entirely within record 0
        first_record_len = len(records[0]["i32"])
        indices = np.arange(min(5, first_record_len))
    indices = indices.astype(np.int64)

    with ColStoreReader(path) as ds:
        got = ds[indices, "i32"].to_array()
        assert np.array_equal(got, truth[indices])


@pytest.mark.parametrize("n_records", [1, 3])
def test_boolean_mask(tmp_path, n_records):
    """Boolean mask reads work on multi-record files just like on single-record."""
    path = tmp_path / "mask.cstore"
    records = _make_records(n_records)
    write_record_file(path, _schema(), records)
    truth = expected_column_values(records, "i32")
    mask = (truth % 3) == 0
    with ColStoreReader(path) as ds:
        assert np.array_equal(ds[mask, "i32"].to_array(), truth[mask])


@pytest.mark.parametrize("n_records", [1, 3])
def test_negative_indices_fold(tmp_path, n_records):
    """Negative fancy-index entries are folded to their positive equivalents."""
    path = tmp_path / "neg.cstore"
    records = _make_records(n_records)
    write_record_file(path, _schema(), records)
    truth = expected_column_values(records, "i32")
    total = truth.shape[0]
    indices = np.array([-1, -2, 0, -total])
    with ColStoreReader(path) as ds:
        assert np.array_equal(ds[indices, "i32"].to_array(), truth[indices])


# ---- Multi-column reads -----------------------------------------------------


@pytest.mark.parametrize("n_records", [1, 3])
def test_multi_column_to_dict(tmp_path, n_records):
    """``ds[indices, [...]].to_dict()`` returns each column correctly."""
    path = tmp_path / "multi.cstore"
    records = _make_records(n_records)
    write_record_file(path, _schema(), records)
    truth = {name: expected_column_values(records, name) for name, _ in _schema()}
    indices = np.array([0, 3, len(truth["i32"]) - 1])
    with ColStoreReader(path) as ds:
        got = ds[indices, ["i32", "f64", "i8"]].to_dict()
        for name in got:
            assert np.array_equal(got[name], truth[name][indices]), f"col {name}"


# ---- Padding ----------------------------------------------------------------


def test_record_body_padding_to_8_bytes(tmp_path):
    """A record whose raw body size is not 8-aligned still reads correctly.

    With schema ``[i32, i8]`` and n_rows=5, raw body = 5*4 + 5*1 = 25 bytes,
    padded up to 32. The padding must be transparent to the reader and the
    next record's header must land 32 bytes after the body start.
    """
    schema = [("a", "<i4"), ("b", "<i1")]
    records = [
        {"a": np.arange(5, dtype=np.int32), "b": np.arange(5, dtype=np.int8)},
        {"a": np.arange(100, 103, dtype=np.int32), "b": np.arange(50, 53, dtype=np.int8)},
    ]
    path = tmp_path / "pad.cstore"
    write_record_file(path, schema, records)
    with ColStoreReader(path) as ds:
        assert ds.n_rows == 8
        # Both records read back intact, demonstrating the second record
        # header was found at the right offset despite the first body padding.
        truth_a = expected_column_values(records, "a")
        truth_b = expected_column_values(records, "b")
        assert np.array_equal(ds[:, "a"].to_array(), truth_a)
        assert np.array_equal(ds[:, "b"].to_array(), truth_b)


# ---- Error paths ------------------------------------------------------------


def _corrupt_record_field(path: Path, record_offset: int, field_offset: int, new_bytes: bytes):
    """Overwrite ``len(new_bytes)`` bytes at ``record_offset + field_offset``."""
    data = bytearray(path.read_bytes())
    start = record_offset + field_offset
    data[start : start + len(new_bytes)] = new_bytes
    path.write_bytes(bytes(data))


def test_corrupt_record_magic_raises(tmp_path):
    """Bad record magic surfaces as FormatError at open time."""
    path = tmp_path / "bad_magic.cstore"
    write_record_file(path, _schema(), _make_records(3))
    # First record starts at data_offset; corrupt its magic (offset 0).
    _, data_offset = fmt.read_header(path)
    _corrupt_record_field(path, data_offset, 0, b"BAD\x00")
    with pytest.raises(FormatError, match="record magic"):
        ColStoreReader(path)


def test_mismatched_record_index_raises(tmp_path):
    """Stored ``record_index`` not matching position surfaces as FormatError."""
    path = tmp_path / "bad_index.cstore"
    write_record_file(path, _schema(), _make_records(3))
    # Second record: corrupt its record_index field (offset 4, 8 bytes).
    # Need to compute the second record's offset: data_offset + 32 + body0_size.
    _, data_offset = fmt.read_header(path)
    schema = _schema()
    itemsizes = [np.dtype(dt).itemsize for _, dt in schema]
    body0_size = fmt.record_body_size(7, itemsizes)  # first record has n_rows=7
    second_offset = data_offset + 32 + body0_size
    # Write 999 instead of 1 at the index slot (preserves magic, breaks index).
    _corrupt_record_field(path, second_offset, 4, struct.pack("<q", 999))
    # CRC will also mismatch; either error message is acceptable -- both are
    # FormatError, both correctly identify a broken header.
    with pytest.raises(FormatError):
        ColStoreReader(path)


def test_corrupt_record_crc_raises(tmp_path):
    """A record header with the right magic but wrong CRC is rejected."""
    path = tmp_path / "bad_crc.cstore"
    write_record_file(path, _schema(), _make_records(2))
    _, data_offset = fmt.read_header(path)
    # CRC is at offset 28 of the header (last 4 bytes).
    _corrupt_record_field(path, data_offset, 28, b"\xff\xff\xff\xff")
    with pytest.raises(FormatError, match="CRC"):
        ColStoreReader(path)


def test_truncated_file_raises(tmp_path):
    """A file shorter than its declared n_records is rejected at open."""
    path = tmp_path / "trunc.cstore"
    write_record_file(path, _schema(), _make_records(3))
    data = path.read_bytes()
    # Truncate well into the second record's body. The manifest says
    # n_records=3 but we leave only enough bytes for ~one record body.
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(FormatError):
        ColStoreReader(path)


# ---- Per-pattern fast paths --------------------------------------------------
#
# Three optimizations on top of the multi-record reader (see
# ColStoreReader._gather_one_multi_record): a contiguous-range path for slices,
# a boundary-based partition for sorted fancy indices, and a fall-through
# searchsorted path for unsorted indices. These tests cover correctness at
# the boundaries where the optimizations diverge from the generic path --
# the underlying parametrized tests above already cover the common cases.


def test_slice_boundary_starting_exactly_on_record_boundary(tmp_path):
    """Slice that starts at record N's first row hits a clean path boundary.

    The contiguous-range path computes the first overlapping record via
    ``searchsorted(record_starts_rows, start, side='right') - 1``. Off-by-one
    here would mis-skip the first record's rows; this test pins that down.
    """
    path = tmp_path / "boundary_start.cstore"
    records = _make_records(4)
    write_record_file(path, _schema(), records)
    truth = expected_column_values(records, "i32")
    # First row of record 1 = total rows in record 0.
    record1_start = len(records[0]["i32"])
    with ColStoreReader(path) as ds:
        got = ds[record1_start:, "i32"].to_array()
        assert np.array_equal(got, truth[record1_start:])


def test_slice_boundary_ending_exactly_on_record_boundary(tmp_path):
    """Slice ``stop`` exactly at a record boundary should not read past it.

    The contiguous path uses ``stop - 1`` for the last-record searchsorted;
    if ``stop`` equals the start of the next record, the math must still
    land on the previous record.
    """
    path = tmp_path / "boundary_stop.cstore"
    records = _make_records(4)
    write_record_file(path, _schema(), records)
    truth = expected_column_values(records, "i32")
    # End-of-record-1 boundary.
    boundary = len(records[0]["i32"]) + len(records[1]["i32"])
    with ColStoreReader(path) as ds:
        got = ds[:boundary, "i32"].to_array()
        assert np.array_equal(got, truth[:boundary])


def test_slice_fully_inside_one_record(tmp_path):
    """Slice contained in a single record exercises the loop's count==n case."""
    path = tmp_path / "inside.cstore"
    records = _make_records(5)
    write_record_file(path, _schema(), records)
    truth = expected_column_values(records, "i32")
    # Pick a slice inside record 2 (rows [r0+r1, r0+r1+r2)).
    r0, r1, r2 = (len(records[i]["i32"]) for i in range(3))
    a, b = r0 + r1 + 1, r0 + r1 + r2 - 1
    with ColStoreReader(path) as ds:
        got = ds[a:b, "i32"].to_array()
        assert np.array_equal(got, truth[a:b])


def test_slice_with_non_unit_step_still_correct(tmp_path):
    """A slice with step != 1 falls through to the fancy-index path.

    The contiguous-range fast path only handles step==1; non-unit steps
    materialize as np.arange and go through the regular gather path. Test
    that this still produces correct results across record boundaries.
    """
    path = tmp_path / "step.cstore"
    records = _make_records(4)
    write_record_file(path, _schema(), records)
    truth = expected_column_values(records, "i32")
    with ColStoreReader(path) as ds:
        got = ds[1:20:3, "i32"].to_array()
        assert np.array_equal(got, truth[1:20:3])
        got = ds[::5, "i32"].to_array()
        assert np.array_equal(got, truth[::5])


def test_sorted_fancy_index_matches_unsorted_fancy_index(tmp_path):
    """Sorted-input path produces the same values as a generic gather.

    The sortedness check redirects sorted-input reads through a boundary-
    based partition (one searchsorted on the indices array per record,
    via np.searchsorted with the broadcast trick). The bypass must yield
    bitwise-identical results to the unsorted path; permuting the indices
    is the cleanest way to compare since the unsorted path is the generic
    one.
    """
    path = tmp_path / "sorted.cstore"
    records = _make_records(5)
    write_record_file(path, _schema(), records)
    truth = expected_column_values(records, "i32")
    rng = np.random.default_rng(123)
    sorted_idx = np.sort(rng.choice(truth.shape[0], size=20, replace=False)).astype(np.int64)
    permutation = rng.permutation(sorted_idx.shape[0])
    unsorted_idx = sorted_idx[permutation]
    with ColStoreReader(path) as ds:
        sorted_out = ds[sorted_idx, "i32"].to_array()
        unsorted_out = ds[unsorted_idx, "i32"].to_array()
        # The two reads produce the same values in different orders.
        assert np.array_equal(sorted_out, truth[sorted_idx])
        assert np.array_equal(unsorted_out, truth[unsorted_idx])
        # Cross-check: unsorting and resorting should agree.
        assert np.array_equal(unsorted_out[np.argsort(permutation)], sorted_out)


def test_sorted_with_duplicates_uses_correct_path(tmp_path):
    """Sorted-with-duplicates is still "sorted" -- diff >= 0.

    The sortedness check is ``indices[1:] >= indices[:-1]`` (non-strict).
    Sorted-with-duplicates triggers the optimized path; verify it still
    returns each duplicate correctly.
    """
    path = tmp_path / "dups.cstore"
    records = _make_records(3)
    write_record_file(path, _schema(), records)
    truth = expected_column_values(records, "i32")
    # Indices: sorted with duplicates spanning record boundaries.
    boundary = len(records[0]["i32"])  # first row of record 1
    idx = np.array([0, 0, 1, boundary - 1, boundary, boundary, boundary + 1], dtype=np.int64)
    assert np.all(idx[1:] >= idx[:-1])  # confirm sortedness predicate
    with ColStoreReader(path) as ds:
        got = ds[idx, "i32"].to_array()
        assert np.array_equal(got, truth[idx])


def test_slice_across_zero_row_record(tmp_path):
    """A zero-row record between two non-empty records is correctly skipped.

    The contiguous-range path iterates over all records in
    ``[first_record, last_record]``. A zero-row record in the middle of that
    range computes count=0, which the vectorized bounds math handles as a
    no-op without any explicit guard. This pins down that behavior.
    """
    path = tmp_path / "zero_rec.cstore"
    records = [
        {
            "i32": np.arange(5, dtype=np.int32),
            "f64": np.arange(5, dtype=np.float64),
            "i8": np.arange(5, dtype=np.int8),
        },
        {
            "i32": np.empty(0, dtype=np.int32),
            "f64": np.empty(0, dtype=np.float64),
            "i8": np.empty(0, dtype=np.int8),
        },
        {
            "i32": np.arange(100, 105, dtype=np.int32),
            "f64": np.arange(100, 105, dtype=np.float64),
            "i8": (np.arange(5, dtype=np.int8) + 50),
        },
    ]
    write_record_file(path, _schema(), records)
    truth = expected_column_values(records, "i32")
    with ColStoreReader(path) as ds:
        # Slice spans the zero-row record.
        got = ds[3:7, "i32"].to_array()
        assert np.array_equal(got, truth[3:7])
        # Full read also has to walk past the zero-row record.
        assert np.array_equal(ds[:, "i32"].to_array(), truth)
