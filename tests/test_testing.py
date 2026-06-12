"""Tests for the public ``colstore.testing`` synthetic-data API."""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import testing


def test_testing_is_namespaced_on_the_package():
    assert colstore.testing is testing
    for name in ("make_columns", "make_store", "uniform_record_rows"):
        assert name in testing.__all__


def test_make_columns_shape_names_and_dtype():
    cols = testing.make_columns(1000, 3, dtype="float32", seed=0)
    assert list(cols) == ["c0", "c1", "c2"]
    for arr in cols.values():
        assert arr.shape == (1000,)
        assert arr.dtype == np.dtype("float32")


def test_make_columns_custom_names():
    cols = testing.make_columns(64, 3, dtype=("f8", "f4", "i2"), names=("f8", "f4", "i2"), seed=1)
    assert list(cols) == ["f8", "f4", "i2"]
    assert cols["f4"].dtype == np.dtype("f4")


def test_make_columns_rejects_bad_names():
    with pytest.raises(ValueError):
        testing.make_columns(10, 2, names=("a",))  # wrong length
    with pytest.raises(ValueError):
        testing.make_columns(10, 2, names=("a", "a"))  # not unique


def test_make_columns_rng_threads_one_stream():
    # Drawing two batches from one rng matches drawing four from a fresh rng
    # of the same seed -- i.e. rng= threads a single stream across calls.
    rng = np.random.default_rng(7)
    first = testing.make_columns(50, 2, rng=rng)
    second = testing.make_columns(50, 2, rng=rng)
    combined = testing.make_columns(50, 4, seed=7)
    assert np.array_equal(first["c0"], combined["c0"])
    assert np.array_equal(first["c1"], combined["c1"])
    assert np.array_equal(second["c0"], combined["c2"])
    assert np.array_equal(second["c1"], combined["c3"])


def test_make_columns_is_reproducible_by_seed():
    a = testing.make_columns(500, 2, dtype="int32", seed=42)
    b = testing.make_columns(500, 2, dtype="int32", seed=42)
    c = testing.make_columns(500, 2, dtype="int32", seed=43)
    for name in a:
        assert np.array_equal(a[name], b[name])
    assert not np.array_equal(a["c0"], c["c0"])


def test_make_columns_mixed_dtypes_cycle_across_columns():
    cols = testing.make_columns(64, 5, dtype=("f8", "f4", "i4", "i2"), seed=1)
    kinds = [cols[f"c{i}"].dtype for i in range(5)]
    assert kinds == [
        np.dtype("f8"),
        np.dtype("f4"),
        np.dtype("i4"),
        np.dtype("i2"),
        np.dtype("f8"),  # cycles back
    ]


def test_make_columns_integer_values_in_range():
    cols = testing.make_columns(10_000, 1, dtype="int16", seed=2)
    info = np.iinfo(np.int16)
    values = cols["c0"]
    assert values.min() >= info.min // 2
    assert values.max() < info.max // 2


@pytest.mark.parametrize("bad", [{"rows": -1, "cols": 1}, {"rows": 10, "cols": 0}])
def test_make_columns_validates(bad):
    with pytest.raises(ValueError):
        testing.make_columns(**bad)


def test_make_columns_rejects_unsupported_dtype():
    with pytest.raises(ValueError):
        testing.make_columns(10, 1, dtype="bool")
    with pytest.raises(ValueError):
        testing.make_columns(10, 1, dtype=[])


def test_uniform_record_rows_splits_and_absorbs_remainder():
    assert testing.uniform_record_rows(100, 4) == [25, 25, 25, 25]
    rows = testing.uniform_record_rows(103, 4)
    assert rows == [25, 25, 25, 28]
    assert sum(rows) == 103


def test_uniform_record_rows_more_records_than_rows_pads_empty():
    rows = testing.uniform_record_rows(3, 5)
    assert sum(rows) == 3
    assert len(rows) == 5
    assert rows.count(0) >= 2


@pytest.mark.parametrize("bad", [(10, 0), (-1, 2)])
def test_uniform_record_rows_validates(bad):
    with pytest.raises(ValueError):
        testing.uniform_record_rows(*bad)


def test_make_store_roundtrips_against_make_columns(tmp_path):
    expected = testing.make_columns(5000, 3, dtype=("f8", "i4", "f4"), seed=7)
    with testing.make_store(
        tmp_path / "s.cstore", rows=5000, cols=3, records=4, dtype=("f8", "i4", "f4"), seed=7
    ) as ds:
        for name, values in expected.items():
            got = ds[:, name].array()
            assert np.array_equal(got, values), name
            assert got.dtype == values.dtype


def test_make_store_single_record_default(tmp_path):
    with testing.make_store(tmp_path / "one.cstore", rows=1000) as ds:
        assert ds.shape[0] == 1000
        assert np.array_equal(ds[:, "c0"].array(), testing.make_columns(1000, 1, seed=0)["c0"])


def test_make_store_explicit_record_split(tmp_path):
    with testing.make_store(
        tmp_path / "rs.cstore", rows=600, cols=2, records=[100, 200, 300], seed=3
    ) as ds:
        assert ds.shape[0] == 600
        expected = testing.make_columns(600, 2, seed=3)
        assert np.array_equal(ds[:, "c1"].array(), expected["c1"])


def test_make_store_rejects_mismatched_split(tmp_path):
    with pytest.raises(ValueError):
        testing.make_store(tmp_path / "bad.cstore", rows=600, records=[100, 200])
    with pytest.raises(ValueError):
        testing.make_store(tmp_path / "bad2.cstore", rows=100, records=[60, -10, 50])


def test_make_store_accepts_str_path(tmp_path):
    path = str(tmp_path / "strpath.cstore")
    with testing.make_store(path, rows=128, cols=2, seed=9) as ds:
        assert ds.shape == (128, 2)


def test_write_columns_writes_given_arrays(tmp_path):
    cols = {
        "a": np.arange(600, dtype=np.float64),
        "b": (np.arange(600) * 2).astype(np.int32),
    }
    with testing.write_columns(tmp_path / "w.cstore", cols, records=4) as ds:
        assert ds.shape[0] == 600
        for name, values in cols.items():
            assert np.array_equal(ds[:, name].array(), values), name


def test_write_columns_explicit_record_split(tmp_path):
    cols = {"x": np.arange(100, dtype=np.float64)}
    with testing.write_columns(tmp_path / "w.cstore", cols, records=[10, 40, 50]) as ds:
        assert ds.shape[0] == 100
        assert np.array_equal(ds[:, "x"].array(), cols["x"])


def test_write_columns_validates():
    with pytest.raises(ValueError):
        testing.write_columns("ignored", {})  # empty
    with pytest.raises(ValueError):
        testing.write_columns("ignored", {"a": np.arange(10), "b": np.arange(11)})  # ragged
    with pytest.raises(ValueError):
        testing.write_columns("ignored", {"a": np.arange(10)}, records=[3, 3])  # sum != 10


def test_make_store_delegates_to_write_columns(tmp_path):
    # make_store == make_columns + write_columns: the store must match a
    # manual write of the same generated columns.
    expected = testing.make_columns(500, 2, dtype=("f8", "i4"), seed=11)
    with (
        testing.make_store(
            tmp_path / "s.cstore", rows=500, cols=2, dtype=("f8", "i4"), seed=11
        ) as a,
        testing.write_columns(tmp_path / "w.cstore", expected) as b,
    ):
        for name in expected:
            assert np.array_equal(a[:, name].array(), b[:, name].array()), name
