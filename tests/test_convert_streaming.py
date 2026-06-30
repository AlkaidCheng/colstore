"""Tests for bounded-memory streaming import: ``colstore.convert(..., batch_size=...)``.

Each streamed result is checked against the whole-file (``batch_size=None``) oracle, and
the warn-and-whole-file fallback is checked for the schemas that cannot stream stably
(variable-width strings, nulls, pandas fixed-format HDF5, JSON / NPZ).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import colstore


def _warned_whole_file(record: list[warnings.WarningMessage]) -> bool:
    return any(
        issubclass(w.category, RuntimeWarning) and "converted whole-file" in str(w.message)
        for w in record
    )


# ---- Parquet ---------------------------------------------------------------


def test_parquet_streamed_matches_whole_file(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    n = 4000
    table = pa.table(
        {
            "id": np.arange(n, dtype=np.int64),
            "x": (np.arange(n) * 1.5).astype(np.float32),
            "f": np.arange(n, dtype=np.float64),
        }
    )
    src = tmp_path / "big.parquet"
    pq.write_table(table, str(src), row_group_size=1000)

    whole = colstore.convert(src, tmp_path / "whole.cstore").dict()
    for batch_size in (1000, "16 KiB"):
        out = colstore.convert(src, tmp_path / f"s_{batch_size}.cstore", batch_size=batch_size)
        for name in ("id", "x", "f"):
            np.testing.assert_array_equal(out.array(name), whole[name])
        assert out.dtypes["id"] == np.int64 and out.dtypes["x"] == np.float32
    # compaction: streamed output is single-record by default, multi-record with compact=False
    assert colstore.info(tmp_path / "s_1000.cstore").n_records == 1
    colstore.convert(src, tmp_path / "multi.cstore", batch_size=1000, compact=False)
    assert colstore.info(tmp_path / "multi.cstore").n_records > 1


def test_parquet_streaming_with_dtypes_and_projection(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pa.table(
        {"id": np.arange(2000, dtype=np.int64), "flag": np.arange(2000, dtype=np.int64) % 2}
    )
    src = tmp_path / "p.parquet"
    pq.write_table(table, str(src), row_group_size=500)
    out = colstore.convert(
        src, tmp_path / "p.cstore", batch_size=500, columns=["flag"], dtypes={"flag": "bool"}
    )
    assert out.columns == ["flag"]
    assert out.dtypes["flag"] == np.bool_ and out.array("flag").tolist()[:4] == [
        False,
        True,
        False,
        True,
    ]


def test_parquet_string_column_falls_back_with_warning(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pa.table(
        {"id": np.arange(10, dtype=np.int64), "name": pa.array(["a" * i for i in range(10)])}
    )
    src = tmp_path / "s.parquet"
    pq.write_table(table, str(src))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = colstore.convert(src, tmp_path / "s.cstore", batch_size=2)
    assert _warned_whole_file(record)
    assert out.n_rows == 10 and list(out.array("name"))[3] == "aaa"


def test_parquet_temporal_columns_stream(tmp_path):
    """datetime64 / timedelta64 columns stream to the same result as a whole-file read."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    n = 1500
    times = (np.datetime64("2020-01-01") + np.arange(n) * np.timedelta64(1, "h")).astype(
        "datetime64[ns]"
    )
    deltas = (np.arange(n) * np.timedelta64(1, "s")).astype("timedelta64[ns]")
    src = tmp_path / "t.parquet"
    pq.write_table(pa.table({"t": times, "d": deltas}), str(src), row_group_size=300)
    whole = colstore.convert(src, tmp_path / "tw.cstore").dict()
    out = colstore.convert(src, tmp_path / "ts.cstore", batch_size=300)
    np.testing.assert_array_equal(out.array("t"), whole["t"])
    np.testing.assert_array_equal(out.array("d"), whole["d"])
    assert out.dtypes["t"] == np.dtype("datetime64[ns]")


def test_streaming_rejects_a_null_that_slips_the_gate(tmp_path):
    """A feather field declared non-nullable but holding nulls is rejected, not 0-filled."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.feather as feather

    schema = pa.schema([pa.field("b", pa.int64(), nullable=False)])
    table = pa.Table.from_arrays([pa.array([1, None, 3, None, 5], type=pa.int64())], schema=schema)
    src = tmp_path / "bad.feather"
    feather.write_feather(table, str(src))
    with pytest.raises(TypeError, match="null"):
        colstore.convert(src, tmp_path / "bad.cstore", batch_size=2)


def test_parquet_null_column_falls_back(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pa.table({"v": pa.array([1, None, 3], type=pa.int64())})
    src = tmp_path / "n.parquet"
    pq.write_table(table, str(src))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match="null"):  # whole-file rejects the partial null
            colstore.convert(src, tmp_path / "n.cstore", batch_size=2)
    assert _warned_whole_file(record)


# ---- Feather ---------------------------------------------------------------


def test_feather_nonnullable_streams_nullable_falls_back(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.feather as feather

    n = 2000
    schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
    table = pa.table({"id": np.arange(n, dtype=np.int64)}, schema=schema)
    nn = tmp_path / "nn.feather"
    feather.write_feather(table, str(nn), chunksize=500)
    out = colstore.convert(nn, tmp_path / "nn.cstore", batch_size=500)
    np.testing.assert_array_equal(out.array("id"), np.arange(n))
    assert colstore.info(tmp_path / "nn.cstore").n_records == 1

    nullable = tmp_path / "null.feather"  # default schema is nullable -> conservative fallback
    feather.write_feather(pa.table({"id": np.arange(5, dtype=np.int64)}), str(nullable))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        colstore.convert(nullable, tmp_path / "nb.cstore", batch_size=2)
    assert _warned_whole_file(record)


# ---- HDF5 ------------------------------------------------------------------


def test_hdf5_h5py_streams(tmp_path):
    h5py = pytest.importorskip("h5py")
    n = 3000
    src = tmp_path / "h.h5"
    with h5py.File(src, "w") as handle:
        group = handle.create_group("data")
        group.create_dataset("id", data=np.arange(n, dtype=np.int64))
        group.create_dataset("x", data=(np.arange(n) * 1.5).astype(np.float64))
    whole = colstore.convert(src, tmp_path / "hw.cstore").dict()
    out = colstore.convert(src, tmp_path / "hs.cstore", batch_size=500)
    np.testing.assert_array_equal(out.array("id"), whole["id"])
    np.testing.assert_array_equal(out.array("x"), whole["x"])
    assert colstore.info(tmp_path / "hs.cstore").n_records == 1


def test_hdf5_pandas_table_streams_fixed_falls_back(tmp_path):
    pytest.importorskip("h5py")
    pytest.importorskip("tables")
    import pandas as pd

    frame = pd.DataFrame(
        {"id": np.arange(3000, dtype=np.int64), "v": np.arange(3000, dtype=np.float64)}
    )
    table_path = tmp_path / "table.h5"
    frame.to_hdf(table_path, key="data", format="table", mode="w")
    out = colstore.convert(table_path, tmp_path / "t.cstore", batch_size=500)
    np.testing.assert_array_equal(out.array("id"), np.arange(3000))
    assert colstore.info(tmp_path / "t.cstore").n_records == 1

    fixed_path = tmp_path / "fixed.h5"  # the default to_hdf format has no chunked reader
    frame.head(10).to_hdf(fixed_path, key="data", mode="w")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        fixed = colstore.convert(fixed_path, tmp_path / "f.cstore", batch_size=2)
    assert _warned_whole_file(record) and fixed.n_rows == 10


# ---- JSON / NPZ: accepted but whole-file ------------------------------------


def test_json_and_npz_warn_and_read_whole(tmp_path):
    pytest.importorskip("pandas")
    store = colstore.store(
        {"id": np.arange(5, dtype=np.int64)}, tmp_path / "s.cstore", show_progress=False
    )
    store.saveas(tmp_path / "d.json")
    store.saveas(tmp_path / "d.npz")
    for ext in ("json", "npz"):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            out = colstore.convert(tmp_path / f"d.{ext}", tmp_path / f"{ext}.cstore", batch_size=2)
        assert _warned_whole_file(record), ext
        assert out.array("id").tolist() == [0, 1, 2, 3, 4]
