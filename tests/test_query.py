"""Tests for the lazy query() predicate API."""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import ColStoreDataset, QueryError, TableView


@pytest.fixture()
def qstore(tmp_path):
    data = {
        "pt": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        "eta": np.array([-3.0, -1.0, 0.0, 1.0, 3.0]),
        "flag": np.array([0, 1, 0, 1, 0], dtype=np.int32),
        "region": np.array(["SR", "CR", "SR", "VR", "CR"], dtype="S2"),
        "is_sig": np.array([True, False, True, False, True]),
    }
    path = tmp_path / "q.cstore"
    colstore.store(data, path, show_progress=False).close()
    ds = colstore.open(path)
    yield ds, data
    ds.close()


def _pts(view):
    return view.dict()["pt"].tolist()


def test_query_returns_lazy_tableview(qstore):
    ds, _ = qstore
    assert isinstance(ds.query("pt > 25"), TableView)


def test_query_simple_comparison(qstore):
    ds, _ = qstore
    assert _pts(ds.query("pt > 25")) == [30.0, 40.0, 50.0]


def test_query_boolean_and(qstore):
    ds, _ = qstore
    assert _pts(ds.query("pt > 20 and eta < 1")) == [30.0]


def test_query_chained_comparison(qstore):
    ds, _ = qstore
    assert _pts(ds.query("-1 < eta < 1")) == [30.0]  # eta == 0 -> pt 30


def test_query_arithmetic(qstore):
    ds, _ = qstore
    assert _pts(ds.query("pt / 10 >= 3")) == [30.0, 40.0, 50.0]


def test_query_membership(qstore):
    ds, _ = qstore
    assert _pts(ds.query("pt in (10, 50)")) == [10.0, 50.0]
    assert _pts(ds.query("pt not in (10, 50)")) == [20.0, 30.0, 40.0]


def test_query_param(qstore):
    ds, _ = qstore
    assert _pts(ds.query("pt > @cut", params={"cut": 30})) == [40.0, 50.0]


def test_query_string_column(qstore):
    ds, _ = qstore
    assert _pts(ds.query("region == 'SR'")) == [10.0, 30.0]


def test_query_bitwise_with_parens(qstore):
    ds, _ = qstore
    assert _pts(ds.query("(pt > 40) | (eta < -2)")) == [10.0, 50.0]


def test_query_bool_column_predicate(qstore):
    ds, _ = qstore
    assert _pts(ds.query("is_sig")) == [10.0, 30.0, 50.0]
    assert _pts(ds.query("not is_sig")) == [20.0, 40.0]


def test_query_column_projection(qstore):
    ds, _ = qstore
    result = ds.query("pt > 25", columns=["region"])
    assert list(result.dict()) == ["region"]
    assert result.dict()["region"].tolist() == [b"SR", b"VR", b"CR"]


def test_query_materializes_full_dict(qstore):
    ds, data = qstore
    mask = data["pt"] > 25
    out = ds.query("pt > 25").dict()
    for name in data:
        assert np.array_equal(out[name], data[name][mask]), name


def test_query_empty_result(qstore):
    ds, _ = qstore
    assert _pts(ds.query("pt > 1000")) == []


@pytest.mark.parametrize(
    "expr, match",
    [
        ("missing > 5", "unknown name"),
        ("len(pt) > 0", "unsupported expression element"),  # function call
        ("pt.size > 0", "unsupported expression element"),  # attribute access
        ("pt + 1", "boolean condition"),  # non-bool result
        ("pt > @undef", "undefined parameter"),
        ("pt >", "could not parse"),  # syntax error
    ],
)
def test_query_rejects(qstore, expr, match):
    ds, _ = qstore
    with pytest.raises(QueryError, match=match):
        ds.query(expr)


def test_query_on_multifile_dataset(tmp_path):
    for i in range(2):
        colstore.store(
            {"pt": np.array([10.0, 60.0]) + i * 100},
            tmp_path / f"p{i}.cstore",
            show_progress=False,
        ).close()
    ds = colstore.open([str(tmp_path / "p0.cstore"), str(tmp_path / "p1.cstore")])
    assert isinstance(ds, ColStoreDataset)
    assert ds.query("pt > 50").dict()["pt"].tolist() == [60.0, 110.0, 160.0]
    ds.close()
