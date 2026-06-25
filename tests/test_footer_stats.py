"""Tests for the optional per-record statistics footer (Stage A: write + load).

The footer records, per record and per column, the column's min/max plus a
``prunable`` flag. It is opt-in (``statistics=True``) and off by default; Stage A
writes it and loads it lazily, but no read path consumes it yet, so these tests
assert the footer round-trips correctly and that data reads are unaffected.
"""

from __future__ import annotations

import numpy as np

import colstore
from colstore import _footer
from colstore.format import write_dataset_streaming
from colstore.frame import MemoryColumn


def _stats(path):
    reader = colstore.open(path)
    try:
        return reader._record_stats, reader.dict()
    finally:
        reader.close()


# ---- opt-in default ---------------------------------------------------------


def test_default_store_writes_no_footer(tmp_path):
    path = tmp_path / "d.cstore"
    colstore.store({"i": np.arange(10, dtype=np.int64)}, path, show_progress=False).close()
    stats, data = _stats(path)
    assert stats is None  # opt-in: no footer unless statistics=True
    np.testing.assert_array_equal(data["i"], np.arange(10))


def test_streaming_write_has_no_footer(tmp_path):
    # The lazy/transform single-record path writes no footer (stats are opt-in on
    # store/create; this path is single-record and coarse).
    arr = np.array([7.0, 2.0, 9.0], dtype=np.float64)
    path = tmp_path / "stream.cstore"
    write_dataset_streaming({"f": MemoryColumn(arr)}, len(arr), path)
    stats, data = _stats(path)
    assert stats is None
    np.testing.assert_array_equal(data["f"], arr)


# ---- single-record (one-shot store, opt-in) --------------------------------


def test_store_single_record_stats(tmp_path):
    cols = {
        "i": np.array([5, 1, 9, 3], dtype=np.int64),
        "u": np.array([10, 40, 20, 30], dtype=np.uint16),
        "f": np.array([2.5, -1.0, 8.0, 0.0], dtype=np.float64),
        "ok": np.array([True, False, True, False]),
        "s": np.array(["bb", "aa", "dd", "cc"]),  # string: not prunable
    }
    path = tmp_path / "a.cstore"
    colstore.store(cols, path, statistics=True, show_progress=False).close()
    stats, data = _stats(path)
    assert stats is not None
    assert int(stats["i"]["min"][0]) == 1 and int(stats["i"]["max"][0]) == 9
    assert int(stats["u"]["min"][0]) == 10 and int(stats["u"]["max"][0]) == 40
    assert float(stats["f"]["min"][0]) == -1.0 and float(stats["f"]["max"][0]) == 8.0
    assert bool(stats["ok"]["min"][0]) is False and bool(stats["ok"]["max"][0]) is True
    assert all(bool(stats[c]["prunable"][0]) for c in ("i", "u", "f", "ok"))
    assert bool(stats["s"]["prunable"][0]) is False  # strings unsupported in v1
    np.testing.assert_array_equal(data["i"], cols["i"])  # data unaffected


def test_nan_float_is_not_prunable(tmp_path):
    path = tmp_path / "nan.cstore"
    colstore.store(
        {"f": np.array([1.0, np.nan, 3.0])}, path, statistics=True, show_progress=False
    ).close()
    stats, _ = _stats(path)
    assert bool(stats["f"]["prunable"][0]) is False


def test_zero_row_is_not_prunable(tmp_path):
    path = tmp_path / "z.cstore"
    colstore.store(
        {"i": np.array([], dtype=np.int64)}, path, statistics=True, show_progress=False
    ).close()
    stats, _ = _stats(path)
    assert bool(stats["i"]["prunable"][0]) is False


def test_datetime_timedelta_stats(tmp_path):
    days = np.array(["2024-01-03", "2024-01-01", "2024-01-09"], dtype="datetime64[D]")
    spans = np.array([5, 2, 8], dtype="timedelta64[s]")
    path = tmp_path / "dt.cstore"
    colstore.store({"d": days, "t": spans}, path, statistics=True, show_progress=False).close()
    stats, _ = _stats(path)
    assert stats["d"]["min"][0] == np.datetime64("2024-01-01")
    assert stats["d"]["max"][0] == np.datetime64("2024-01-09")
    assert stats["t"]["min"][0] == np.timedelta64(2, "s")
    assert bool(stats["d"]["prunable"][0]) and bool(stats["t"]["prunable"][0])


def test_nat_datetime_timedelta_is_not_prunable(tmp_path):
    # NaT views as the int64 sentinel and is not caught by isfinite; it must make
    # the chunk non-prunable, like a NaN float, so Stage B never skips it wrongly.
    days = np.array(["2020-01-01", "NaT", "2021-12-31"], dtype="datetime64[D]")
    spans = np.array([5, "NaT", 8], dtype="timedelta64[s]")
    path = tmp_path / "nat.cstore"
    colstore.store({"d": days, "t": spans}, path, statistics=True, show_progress=False).close()
    stats, _ = _stats(path)
    assert bool(stats["d"]["prunable"][0]) is False
    assert bool(stats["t"]["prunable"][0]) is False


# ---- multi-record (streaming writer, opt-in) -------------------------------


def test_multi_record_per_record_stats(tmp_path):
    path = tmp_path / "multi.cstore"
    with colstore.create(path, statistics=True) as writer:
        writer.write({"v": np.array([5, 1, 9], dtype=np.int64)})  # record 0
        writer.write({"v": np.array([20, 30], dtype=np.int64)})  # record 1
        writer.write({"v": np.array([-4, -1], dtype=np.int64)})  # record 2
    stats, _ = _stats(path)
    assert [int(x) for x in stats["v"]["min"]] == [1, 20, -4]
    assert [int(x) for x in stats["v"]["max"]] == [9, 30, -1]
    assert list(stats["v"]["prunable"]) == [True, True, True]


def test_writer_default_has_no_footer(tmp_path):
    path = tmp_path / "nofoot.cstore"
    with colstore.create(path) as writer:  # statistics defaults to False
        writer.write({"v": np.array([1, 2, 3], dtype=np.int64)})
    stats, _ = _stats(path)
    assert stats is None


def test_update_mode_preserves_and_extends_stats(tmp_path):
    path = tmp_path / "upd.cstore"
    with colstore.create(path, statistics=True) as writer:
        writer.write({"v": np.array([1, 2, 3], dtype=np.int64)})  # record 0: 1..3
        writer.write({"v": np.array([10, 20], dtype=np.int64)})  # record 1: 10..20
    with colstore.update(path, statistics=True) as writer:
        writer.write({"v": np.array([100, 300, 200], dtype=np.int64)})  # record 2: 100..300
    stats, _ = _stats(path)
    assert [int(x) for x in stats["v"]["min"]] == [1, 10, 100]  # old two preserved
    assert [int(x) for x in stats["v"]["max"]] == [3, 20, 300]


# ---- advisory / degradation ------------------------------------------------


def test_no_footer_when_stats_offset_zero(tmp_path):
    path = tmp_path / "nostat.cstore"
    colstore.store(
        {"i": np.arange(10, dtype=np.int64)}, path, statistics=True, show_progress=False
    ).close()
    # Zero the stats_offset to simulate a footer-less file; the counters CRC also
    # covers stats_offset, so rewrite the whole counters block consistently.
    reader = colstore.open(path)
    n_records = reader._manifest["n_records"]
    committed_rows = reader._manifest["committed_rows"]
    reader.close()
    from colstore import format as fmt

    with open(path, "r+b") as f:
        f.seek(fmt._COUNTERS_OFFSET)
        f.write(fmt._pack_counters(n_records, committed_rows, 0))
    stats, data = _stats(path)
    assert stats is None
    np.testing.assert_array_equal(data["i"], np.arange(10))


def test_corrupted_footer_degrades_to_none(tmp_path):
    path = tmp_path / "corrupt.cstore"
    colstore.store(
        {"i": np.arange(10, dtype=np.int64)}, path, statistics=True, show_progress=False
    ).close()
    reader = colstore.open(path)
    stats_offset = reader._stats_offset
    reader.close()
    # Flip a byte inside the footer; its CRC no longer matches, so parsing fails
    # and the reader degrades to "no stats" rather than raising.
    with open(path, "r+b") as f:
        f.seek(stats_offset + 16)
        original = f.read(1)
        f.seek(stats_offset + 16)
        f.write(bytes([original[0] ^ 0xFF]))
    stats, data = _stats(path)
    assert stats is None
    np.testing.assert_array_equal(data["i"], np.arange(10))


# ---- unit round-trip --------------------------------------------------------


def test_footer_serialize_parse_roundtrip():
    columns_meta = [
        {"name": "i", "dtype": "<i8"},
        {"name": "f", "dtype": "<f4"},
        {"name": "s", "dtype": "<U3"},
    ]
    dt_i, dt_f, dt_s = np.dtype("<i8"), np.dtype("<f4"), np.dtype("<U3")
    per_record = [
        {
            "i": _footer.column_stat(dt_i, np.array([3, 1, 2], dtype=dt_i)),
            "f": _footer.column_stat(
                dt_f, np.array([1.0, np.nan], dtype=dt_f)
            ),  # NaN: not prunable
            "s": _footer.column_stat(
                dt_s, np.array(["ab", "cd"], dtype=dt_s)
            ),  # string: not prunable
        },
        {
            "i": _footer.column_stat(dt_i, np.array([10, 9], dtype=dt_i)),
            "f": _footer.column_stat(dt_f, np.array([2.0, 5.0], dtype=dt_f)),
            "s": _footer.column_stat(dt_s, np.array(["zz"], dtype=dt_s)),
        },
    ]
    parsed = _footer.parse_stats(_footer.serialize_stats(columns_meta, per_record))
    assert parsed is not None
    assert [int(x) for x in parsed["i"]["min"]] == [1, 9]
    assert [int(x) for x in parsed["i"]["max"]] == [3, 10]
    assert list(parsed["f"]["prunable"]) == [False, True]
    assert float(parsed["f"]["max"][1]) == 5.0
    assert list(parsed["s"]["prunable"]) == [False, False]
    # a truncated / corrupt buffer parses to None, never raises
    assert _footer.parse_stats(b"not a footer") is None
