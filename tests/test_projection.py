"""Tests for the select() / drop() column-projection helpers."""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import ColStoreDataset, TableView, col


@pytest.fixture()
def store4(tmp_path):
    data = {
        "pt": np.arange(5.0),
        "eta": np.arange(5.0) + 10,
        "phi": np.arange(5.0) + 20,
        "id": np.arange(5, dtype=np.int32),
    }
    path = tmp_path / "s.cstore"
    colstore.store(data, path, show_progress=False).close()
    ds = colstore.open(path)
    yield ds, data
    ds.close()


def test_select_returns_tableview_with_columns(store4):
    ds, _ = store4
    view = ds.select("pt", "eta")
    assert isinstance(view, TableView)
    assert list(view.dict()) == ["pt", "eta"]


def test_select_preserves_given_order(store4):
    ds, _ = store4
    assert list(ds.select("eta", "pt").dict()) == ["eta", "pt"]


def test_select_single_column_is_a_tableview(store4):
    ds, _ = store4
    view = ds.select("pt")
    assert isinstance(view, TableView)
    assert list(view.dict()) == ["pt"]


def test_select_materializes_correct_data(store4):
    ds, data = store4
    out = ds.select("pt", "id").dict()
    assert np.array_equal(out["pt"], data["pt"])
    assert np.array_equal(out["id"], data["id"])


def test_select_recarray_fields_follow_selection(store4):
    ds, _ = store4
    rec = ds.select("eta", "pt").recarray()
    assert rec.dtype.names == ("eta", "pt")


def test_drop_keeps_rest_in_stored_order(store4):
    ds, _ = store4
    assert list(ds.drop("eta").dict()) == ["pt", "phi", "id"]
    assert list(ds.drop("id", "phi").dict()) == ["pt", "eta"]


def test_select_chains_after_query(store4):
    ds, data = store4
    view = ds.query("pt > 2").select("eta")  # filter on pt, project eta
    mask = data["pt"] > 2
    assert list(view.dict()) == ["eta"]
    assert np.array_equal(view.dict()["eta"], data["eta"][mask])


def test_drop_chains_after_col_filter(store4):
    ds, data = store4
    view = ds[col("pt") > 2].drop("id", "phi")
    mask = data["pt"] > 2
    assert list(view.dict()) == ["pt", "eta"]
    assert np.array_equal(view.dict()["pt"], data["pt"][mask])


def test_select_narrows_within_a_view(store4):
    ds, _ = store4
    assert list(ds.select("pt", "eta", "phi").select("eta").dict()) == ["eta"]


def test_select_outside_view_columns_raises(store4):
    ds, _ = store4
    with pytest.raises(KeyError, match="phi"):
        ds.select("pt", "eta").select("phi")


def test_unknown_column_raises_keyerror(store4):
    ds, _ = store4
    with pytest.raises(KeyError, match="Unknown column"):
        ds.select("nope")
    with pytest.raises(KeyError, match="Unknown column"):
        ds.drop("nope")


def test_duplicate_select_raises(store4):
    ds, _ = store4
    with pytest.raises(ValueError, match="Duplicate"):
        ds.select("pt", "pt")


def test_empty_select_raises(store4):
    ds, _ = store4
    with pytest.raises(ValueError, match="at least one"):
        ds.select()


def test_select_drop_on_dataset(tmp_path):
    for i in range(2):
        colstore.store(
            {"a": np.arange(3.0) + i, "b": np.arange(3.0) + i, "c": np.arange(3.0) + i},
            tmp_path / f"m{i}.cstore",
            show_progress=False,
        ).close()
    with colstore.open([str(tmp_path / "m0.cstore"), str(tmp_path / "m1.cstore")]) as ds:
        assert isinstance(ds, ColStoreDataset)
        assert list(ds.select("c", "a").dict()) == ["c", "a"]
        assert list(ds.drop("b").dict()) == ["a", "c"]
