"""Tests for the from_dataframe / from_dict / from_records factory methods."""

from __future__ import annotations

import numpy as np
import pandas as pd

from colstore import ColStore


def test_from_dataframe_returns_opened_store(tmp_path):
    frame = pd.DataFrame(
        {"x": np.arange(100, dtype=np.float32), "y": np.arange(100, dtype=np.int64)}
    )
    store = ColStore.from_dataframe(frame, tmp_path / "out.cstore", show_progress=False)
    try:
        assert store.n_rows == 100
        assert store.columns == ["x", "y"]
        assert store.dtypes["x"] == np.float32
        assert store.dtypes["y"] == np.int64
    finally:
        store.close()


def test_from_dataframe_preserves_dtypes(tmp_path):
    frame = pd.DataFrame(
        {
            "f32": np.array([1.0, 2.0], dtype=np.float32),
            "i16": np.array([1, 2], dtype=np.int16),
            "u8": np.array([255, 0], dtype=np.uint8),
        }
    )
    with ColStore.from_dataframe(frame, tmp_path / "dtypes.cstore", show_progress=False) as store:
        assert store.dtypes["f32"] == np.float32
        assert store.dtypes["i16"] == np.int16
        assert store.dtypes["u8"] == np.uint8


def test_from_dict_round_trips(tmp_path):
    columns = {
        "x": np.arange(50, dtype=np.float64),
        "y": np.arange(50, dtype=np.int32) * 3,
    }
    with ColStore.from_dict(columns, tmp_path / "out.cstore", show_progress=False) as store:
        roundtripped = store[:].to_dict()
        for column_name in columns:
            assert np.array_equal(roundtripped[column_name], columns[column_name])


def test_from_records_round_trips(tmp_path):
    record_dtype = np.dtype([("price", np.float32), ("count", np.int64)])
    records = np.empty(20, dtype=record_dtype)
    records["price"] = np.arange(20, dtype=np.float32) * 0.5
    records["count"] = np.arange(20, dtype=np.int64) * 7

    with ColStore.from_records(records, tmp_path / "rec.cstore", show_progress=False) as store:
        out = store[:].to_record()
        assert out.dtype.names == ("price", "count")
        assert np.array_equal(out["price"], records["price"])
        assert np.array_equal(out["count"], records["count"])


def test_factory_forwards_open_kwargs(tmp_path):
    columns = {"a": np.zeros(10, dtype=np.float32)}
    store = ColStore.from_dict(
        columns,
        tmp_path / "kw.cstore",
        show_progress=False,
        madvise="random",
        max_workers=1,
    )
    try:
        assert store.max_workers == 1
    finally:
        store.close()


def test_factory_writes_correct_extension_by_convention(tmp_path):
    frame = pd.DataFrame({"a": np.arange(10, dtype=np.float32)})
    target = tmp_path / "with_extension.cstore"
    with ColStore.from_dataframe(frame, target, show_progress=False):
        pass
    assert target.exists()
    assert target.suffix == ".cstore"
