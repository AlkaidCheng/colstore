"""Tests for the lazy query() predicate API."""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import ColStoreDataset, ColumnView, QueryError, TableView, col
from colstore._query import _Expr


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


# ---- col() expression form -------------------------------------------------


def test_col_comparison(qstore):
    ds, _ = qstore
    assert _pts(ds[col("pt") > 25]) == [30.0, 40.0, 50.0]


def test_col_where_verb(qstore):
    ds, _ = qstore
    assert _pts(ds.where(col("pt") > 25)) == [30.0, 40.0, 50.0]


def test_col_stacking_with_and(qstore):
    ds, _ = qstore
    # pt > 20 -> {30,40,50}; region == 'SR' -> {10,30}; & -> {30}
    assert _pts(ds[(col("pt") > 20) & (col("region") == "SR")]) == [30.0]


def test_col_or_and_invert(qstore):
    ds, _ = qstore
    assert _pts(ds[(col("pt") > 40) | (col("eta") < -2)]) == [10.0, 50.0]
    assert _pts(ds[~col("is_sig")]) == [20.0, 40.0]


def test_col_isin(qstore):
    ds, _ = qstore
    assert _pts(ds[col("pt").isin([10, 50])]) == [10.0, 50.0]


def test_col_reflected_arithmetic(qstore):
    ds, _ = qstore
    # reflected arithmetic (__rsub__): 100 - pt > 50  ->  pt < 50  ->  {10,20,30,40}
    assert _pts(ds[100 - col("pt") > 50]) == [10.0, 20.0, 30.0, 40.0]
    assert _pts(ds[col("pt") / 10 >= 3]) == [30.0, 40.0, 50.0]


def test_col_reflected_floordiv_mod_pow(qstore):
    ds, _ = qstore
    # __rfloordiv__: 100 // pt >= 5  ->  pt in {10, 20}
    assert _pts(ds[100 // col("pt") >= 5]) == [10.0, 20.0]
    # __rmod__: 100 % pt == 0  ->  pt in {10, 20, 50}
    assert _pts(ds[100 % col("pt") == 0]) == [10.0, 20.0, 50.0]
    # __rpow__: 2 ** flag > 1  ->  flag == 1  ->  pt in {20, 40}
    assert _pts(ds[2 ** col("flag") > 1]) == [20.0, 40.0]


def test_col_string_equality(qstore):
    ds, _ = qstore
    assert _pts(ds[col("region") == "SR"]) == [10.0, 30.0]


def test_col_with_column_projection(qstore):
    ds, _ = qstore
    cv = ds[col("pt") > 40, "region"]
    assert isinstance(cv, ColumnView)
    assert cv.array().tolist() == [b"CR"]  # pt 50 -> region CR


def test_query_accepts_col_expression(qstore):
    ds, _ = qstore
    assert _pts(ds.query(col("pt") > 25)) == [30.0, 40.0, 50.0]


def test_col_bool_guard():
    # and / or / not are not overloadable; the guard catches a stray bool() call.
    with pytest.raises(QueryError, match="no truth value"):
        bool(col("pt") > 5)
    with pytest.raises(QueryError, match="no truth value"):
        (col("pt") > 5) and (col("pt") < 50)


def test_col_unknown_column_raises_eagerly(qstore):
    ds, _ = qstore
    with pytest.raises(QueryError, match="unknown column"):
        ds[col("missing") > 5]


def test_col_non_boolean_raises_eagerly(qstore):
    ds, _ = qstore
    with pytest.raises(QueryError, match="boolean condition"):
        ds[col("pt") + 1]


def test_col_name_must_be_str():
    with pytest.raises(TypeError, match="must be a string"):
        col(123)  # type: ignore[arg-type]


# ---- laziness and evaluate() -----------------------------------------------


def test_query_carries_an_unresolved_predicate(qstore):
    ds, _ = qstore
    # The predicate rides the view as an expression, not yet a materialized mask.
    assert isinstance(ds.query("pt > 25")._row_part, _Expr)
    assert isinstance(ds[col("pt") > 25]._row_part, _Expr)


def test_evaluate_resolves_the_mask(qstore):
    ds, _ = qstore
    resolved = ds.query("pt > 25").evaluate()
    assert isinstance(resolved, TableView)
    # The row selection is now a concrete boolean mask, not an expression.
    assert isinstance(resolved._row_part, np.ndarray)
    assert resolved._row_part.dtype == bool
    assert _pts(resolved) == [30.0, 40.0, 50.0]


def test_query_lazy_false_returns_resolved_view(qstore):
    ds, _ = qstore
    view = ds.query("pt > 25", lazy=False)
    assert isinstance(view, TableView)
    assert isinstance(view._row_part, np.ndarray)
    assert _pts(view) == [30.0, 40.0, 50.0]


def test_columnview_evaluate(qstore):
    ds, _ = qstore
    resolved = ds[col("pt") > 25, "region"].evaluate()
    assert isinstance(resolved, ColumnView)
    assert resolved.array().tolist() == [b"SR", b"VR", b"CR"]  # pt 30,40,50 -> SR,VR,CR


def test_col_on_multifile_dataset(tmp_path):
    for i in range(2):
        colstore.store(
            {"pt": np.array([10.0, 60.0]) + i * 100},
            tmp_path / f"p{i}.cstore",
            show_progress=False,
        ).close()
    with colstore.open([str(tmp_path / "p0.cstore"), str(tmp_path / "p1.cstore")]) as ds:
        assert _pts(ds[col("pt") > 50]) == [60.0, 110.0, 160.0]
