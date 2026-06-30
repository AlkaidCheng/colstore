"""Tests for bounded-memory streaming export: ``colstore.convert(.cstore -> foreign,
batch_size=...)`` and ``ds.saveas(..., batch_size=...)``.

Each streamed file is checked against its whole-file (``batch_size=None``) oracle by
re-importing both, the streamed container is checked to actually hold several
batches/row-groups, and the warn-and-whole-file fallback is checked for the targets that
have no appendable path (JSON / NPZ, the pandas HDF5 backend, Feather with write options).
A ``tracemalloc`` check confirms the streamed HDF5 export peaks far below the whole-file one.
"""

from __future__ import annotations

import tracemalloc
import warnings

import numpy as np
import pytest

import colstore


def _warned_whole_file(record: list[warnings.WarningMessage]) -> bool:
    return any(
        issubclass(w.category, RuntimeWarning) and "written whole-file" in str(w.message)
        for w in record
    )


def _sample_store(path, n=4000):
    data = {
        "id": np.arange(n, dtype=np.int64),
        "x": (np.arange(n) * 1.5).astype(np.float32),
        "name": np.array([f"row{i % 7}" for i in range(n)], dtype="U6"),
        "t": (np.datetime64("2020-01-01") + np.arange(n) * np.timedelta64(1, "h")).astype(
            "datetime64[ns]"
        ),
        "d": (np.arange(n) * np.timedelta64(1, "s")).astype("timedelta64[ns]"),
    }
    colstore.store(data, path, show_progress=False).close()
    return data


def _reimport(tmp_path, src, tag):
    """Round-trip a foreign file back to a column dict for comparison."""
    return colstore.convert(src, tmp_path / f"back_{tag}.cstore").dict()


# ---- Parquet ---------------------------------------------------------------


def test_parquet_streamed_export_matches_whole_file(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    data = _sample_store(tmp_path / "src.cstore")
    src = tmp_path / "src.cstore"
    whole = tmp_path / "whole.parquet"
    streamed = tmp_path / "stream.parquet"
    colstore.convert(src, whole)
    colstore.convert(src, streamed, batch_size=500)

    assert pq.ParquetFile(str(streamed)).num_row_groups > 1  # actually streamed
    a, b = _reimport(tmp_path, whole, "pw"), _reimport(tmp_path, streamed, "ps")
    assert set(a) == set(b) == set(data)
    for name in data:
        np.testing.assert_array_equal(a[name], b[name])
        np.testing.assert_array_equal(b[name], data[name])


@pytest.mark.parametrize("ext", ["parquet", "feather"])
@pytest.mark.parametrize(
    "values",
    [
        np.array([0, 1, 2, 3], dtype="datetime64[ns]"),
        np.array([0, 1, 2, 3], dtype="timedelta64[ns]"),
        np.array([0, 1, 2, 3], dtype="datetime64[us]"),
    ],
)
def test_streamed_export_preserves_temporal_nat(tmp_path, ext, values):
    """A NaT in a datetime64/timedelta64 column is a valid value, not a null: it must
    survive a streamed export exactly as the whole-file path keeps it (re-importable)."""
    pytest.importorskip("pyarrow")
    column = values.copy()
    column[2] = np.datetime64("NaT")  # NaT shares one int64 sentinel across M and m
    src = tmp_path / "nat.cstore"
    colstore.store({"t": column}, src, show_progress=False).close()
    ds = colstore.open(src)
    try:
        out = tmp_path / f"nat.{ext}"
        ds.saveas(out, batch_size=2)
    finally:
        ds.close()
    back = colstore.convert(out, tmp_path / "nat_back.cstore").array("t")
    np.testing.assert_array_equal(back.view("i8"), column.view("i8"))


@pytest.mark.parametrize("ext", ["parquet", "feather"])
def test_streamed_export_preserves_embedded_nul_bytes(tmp_path, ext):
    """An 'S' fixed-width-bytes column with embedded NULs must round-trip byte-for-byte;
    pyarrow's default numpy conversion would truncate each value at its first NUL."""
    pytest.importorskip("pyarrow")
    blobs = np.array([b"x\x00y", b"\xff\x00\xfe", b"\x00\x00ab", b"ok"], dtype="S5")
    src = tmp_path / "blobs.cstore"
    colstore.store({"b": blobs}, src, show_progress=False).close()
    ds = colstore.open(src)
    try:
        out = tmp_path / f"blobs.{ext}"
        ds.saveas(out, batch_size=2)
    finally:
        ds.close()
    back = colstore.convert(out, tmp_path / "blobs_back.cstore").array("b")
    np.testing.assert_array_equal(back, blobs)


def test_parquet_export_compression_streams(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    _sample_store(tmp_path / "src.cstore", n=2000)
    out = tmp_path / "c.parquet"
    colstore.convert(tmp_path / "src.cstore", out, batch_size=500, compression="zstd")
    meta = pq.ParquetFile(str(out)).metadata
    assert meta.num_row_groups > 1
    assert meta.row_group(0).column(0).compression == "ZSTD"


# ---- Feather ---------------------------------------------------------------


def test_feather_streamed_export_matches_whole_file(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    data = _sample_store(tmp_path / "src.cstore")
    src = tmp_path / "src.cstore"
    whole, streamed = tmp_path / "whole.feather", tmp_path / "stream.feather"
    colstore.convert(src, whole)
    colstore.convert(src, streamed, batch_size=500)

    assert pa.ipc.open_file(str(streamed)).num_record_batches > 1
    a, b = _reimport(tmp_path, whole, "fw"), _reimport(tmp_path, streamed, "fs")
    for name in data:
        np.testing.assert_array_equal(b[name], data[name])
        np.testing.assert_array_equal(a[name], b[name])


def test_feather_export_with_write_option_falls_back(tmp_path):
    pytest.importorskip("pyarrow")
    _sample_store(tmp_path / "src.cstore", n=1000)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        colstore.convert(
            tmp_path / "src.cstore", tmp_path / "c.feather", batch_size=200, compression="zstd"
        )
    assert _warned_whole_file(record)
    # whole-file fallback still honored the option and produced a readable file
    import pyarrow.feather as feather

    assert feather.read_table(str(tmp_path / "c.feather")).num_rows == 1000


# ---- HDF5 ------------------------------------------------------------------


def test_hdf5_h5py_streamed_export_matches_whole_file(tmp_path):
    pytest.importorskip("h5py")
    data = _sample_store(tmp_path / "src.cstore")
    src = tmp_path / "src.cstore"
    whole, streamed = tmp_path / "whole.h5", tmp_path / "stream.h5"
    colstore.convert(src, whole)
    colstore.convert(src, streamed, batch_size=500)

    a, b = _reimport(tmp_path, whole, "hw"), _reimport(tmp_path, streamed, "hs")
    assert set(a) == set(b) == set(data)
    for name in data:
        np.testing.assert_array_equal(b[name], data[name])
        np.testing.assert_array_equal(a[name], b[name])


def test_hdf5_pandas_backend_export_falls_back(tmp_path):
    pytest.importorskip("h5py")
    pytest.importorskip("tables")
    _sample_store(tmp_path / "src.cstore", n=1000)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        colstore.convert(
            tmp_path / "src.cstore", tmp_path / "p.h5", batch_size=200, backend="pandas"
        )
    assert _warned_whole_file(record)
    back = colstore.convert(tmp_path / "p.h5", tmp_path / "p.cstore")
    assert back.n_rows == 1000


def test_hdf5_streamed_export_is_memory_bounded(tmp_path):
    pytest.importorskip("h5py")
    n = 500_000
    src = tmp_path / "big.cstore"
    colstore.store(
        {f"c{i}": (np.arange(n, dtype=np.float64) + i) for i in range(4)},
        src,
        show_progress=False,
    ).close()
    ds = colstore.open(src)
    try:
        tracemalloc.start()
        ds.saveas(tmp_path / "whole.h5")
        _, whole_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        ds.saveas(tmp_path / "stream.h5", batch_size="4 MiB")
        _, stream_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        ds.close()
    # whole-file gathers every column (~16 MB); streaming holds one ~4 MB batch.
    assert stream_peak < whole_peak / 3


# ---- .cstore -> .cstore ----------------------------------------------------


def test_cstore_streamed_export_matches_whole_file(tmp_path):
    data = _sample_store(tmp_path / "src.cstore")
    src = tmp_path / "src.cstore"
    whole = colstore.convert(src, tmp_path / "whole2.cstore").dict()
    streamed_bytes = colstore.convert(src, tmp_path / "s_bytes.cstore", batch_size="8 KiB").dict()
    streamed_rows = colstore.convert(src, tmp_path / "s_rows.cstore", batch_size=500).dict()
    for name in data:
        np.testing.assert_array_equal(streamed_bytes[name], data[name])
        np.testing.assert_array_equal(streamed_rows[name], whole[name])


def test_cstore_export_rejects_batch_size_with_memory_budget(tmp_path):
    _sample_store(tmp_path / "src.cstore", n=100)
    ds = colstore.open(tmp_path / "src.cstore")
    try:
        with pytest.raises(TypeError, match="not both"):
            ds.saveas(tmp_path / "x.cstore", batch_size=10, memory_budget=1 << 20)
    finally:
        ds.close()


# ---- Selections, empty, and the decline targets ----------------------------


def test_streamed_export_of_a_row_and_column_subset(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    _sample_store(tmp_path / "src.cstore")
    ds = colstore.open(tmp_path / "src.cstore")
    try:
        ds[100:250, ["id", "x"]].saveas(tmp_path / "sub.parquet", batch_size=40)
    finally:
        ds.close()
    table = pq.read_table(str(tmp_path / "sub.parquet"))
    assert table.column_names == ["id", "x"]
    assert table.num_rows == 150
    assert table.column("id").to_pylist()[:3] == [100, 101, 102]


def test_streamed_export_of_an_empty_selection_writes_a_valid_file(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    src = tmp_path / "empty.cstore"
    colstore.store({"a": np.arange(0, dtype=np.int64)}, src, show_progress=False).close()
    out = tmp_path / "empty.parquet"
    colstore.convert(src, out, batch_size=10)
    table = pq.read_table(str(out))
    assert table.num_rows == 0 and table.column_names == ["a"]


def test_json_and_npz_export_warn_and_write_whole(tmp_path):
    _sample_store(tmp_path / "src.cstore", n=50)
    src = tmp_path / "src.cstore"
    for ext in ("json", "npz"):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            colstore.convert(src, tmp_path / f"o.{ext}", batch_size=10)
        assert _warned_whole_file(record), ext
        back = colstore.convert(tmp_path / f"o.{ext}", tmp_path / f"{ext}.cstore")
        np.testing.assert_array_equal(back.array("id"), np.arange(50))
