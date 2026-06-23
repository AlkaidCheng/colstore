"""Tests for the deferred column-editing frame (ColStoreFrame, ds.edit()).

Covers the editing API end to end: opening a frame, assigning/replacing/
deleting/renaming columns, building transforms, eager validation, reference vs
copy semantics, cross-store columns, eager `compute`, and committing with
`write`. Building, editing, writing, and reading back all use the single-record
memmap path, so no compiled extension is required.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import colstore
from colstore import ColStoreFrame, col
from colstore.format import write_dataset


def _make_store(tmp_path, columns, name="src.cstore"):
    path = tmp_path / name
    write_dataset(columns, path, batch_size=None, show_progress=False)
    return colstore.open(path)


def _written(cf, tmp_path, **kwargs):
    """Commit a frame and return its columns as a dict, closing the reader."""
    reader = cf.write(tmp_path / "out.cstore", **kwargs)
    try:
        return reader.dict()
    finally:
        reader.close()


@pytest.fixture
def source_cols():
    rng = np.random.default_rng(0)
    n = 256
    return {
        "a": np.arange(n, dtype=np.int64),
        "b": rng.standard_normal(n).astype(np.float64),
        "c": rng.standard_normal(n).astype(np.float32),
    }


@pytest.fixture
def source(tmp_path, source_cols):
    store = _make_store(tmp_path, source_cols)
    yield store
    store.close()


# -- opening a frame --


def test_edit_seeds_source_columns(source, source_cols):
    cf = source.edit()
    assert isinstance(cf, ColStoreFrame)
    assert cf.columns == list(source_cols)
    assert cf.n_rows == source.n_rows
    assert len(cf) == len(source_cols)
    assert "a" in cf
    assert "missing" not in cf
    assert list(iter(cf)) == list(source_cols)


def test_getitem_missing_raises(source):
    with pytest.raises(KeyError, match="not in the frame"):
        _ = source.edit()["nope"]


def test_noop_edit_is_byte_identical(source, tmp_path):
    out_path = tmp_path / "noop.cstore"
    source.edit().write(out_path).close()
    src_digest = hashlib.sha256(source.path.read_bytes()).hexdigest()
    out_digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    assert src_digest == out_digest


def test_edit_does_not_modify_source(source, source_cols, tmp_path):
    before = source.dict()
    cf = source.edit()
    cf["a"] = cf["a"] * 0
    del cf["b"]
    cf.write(tmp_path / "o.cstore").close()
    after = source.dict()
    for name in source_cols:
        np.testing.assert_array_equal(after[name], before[name])


# -- assigning and transforming --


def test_assign_transform_column(source, source_cols, tmp_path):
    cf = source.edit()
    cf["scaled"] = cf["b"] * 2.0 + 1.0
    out = _written(cf, tmp_path)
    assert out["scaled"].dtype == np.float64
    np.testing.assert_allclose(out["scaled"], source_cols["b"] * 2.0 + 1.0)
    np.testing.assert_array_equal(out["a"], source_cols["a"])  # untouched columns survive


def test_replace_column(source, source_cols, tmp_path):
    cf = source.edit()
    cf["a"] = cf["a"] + 1
    out = _written(cf, tmp_path)
    np.testing.assert_array_equal(out["a"], source_cols["a"] + 1)
    assert out["a"].dtype == np.int64


def test_transform_chain(source, source_cols, tmp_path):
    cf = source.edit()
    cf["a"] = cf["a"] * 2
    cf["a"] = cf["a"] + 1
    out = _written(cf, tmp_path)
    np.testing.assert_array_equal(out["a"], source_cols["a"] * 2 + 1)


def test_expression_captures_column_not_name(source, source_cols, tmp_path):
    cf = source.edit()
    cf["a_copy"] = cf["a"]  # captures the leaf, not the name "a"
    cf["a"] = cf["a"] + 100  # rebinds "a" to a transform of the old "a"
    out = _written(cf, tmp_path)
    np.testing.assert_array_equal(out["a_copy"], source_cols["a"])  # unaffected
    np.testing.assert_array_equal(out["a"], source_cols["a"] + 100)


def test_assign_scalar_broadcasts(source, tmp_path):
    cf = source.edit()
    cf["flag"] = np.int8(1)
    out = _written(cf, tmp_path)
    np.testing.assert_array_equal(out["flag"], np.ones(source.n_rows, dtype=np.int8))


def test_with_columns_alias(source, source_cols, tmp_path):
    cf = source.edit()
    cf = cf.with_columns(d=cf["a"] + cf["a"])
    assert "d" in cf
    out = _written(cf, tmp_path)
    np.testing.assert_array_equal(out["d"], source_cols["a"] * 2)


def test_assign_via_method(source, source_cols, tmp_path):
    cf = source.edit()
    cf = cf.assign(p=cf["b"] * 3.0)
    out = _written(cf, tmp_path)
    np.testing.assert_allclose(out["p"], source_cols["b"] * 3.0)


# -- reference vs copy semantics --


def test_assigned_array_is_reference_by_default(source, tmp_path):
    extra = np.arange(source.n_rows, dtype=np.float64)
    cf = source.edit()
    cf["k"] = extra
    extra[:] = -1.0  # mutate after assignment, before write
    out = _written(cf, tmp_path)
    np.testing.assert_array_equal(out["k"], np.full(source.n_rows, -1.0))


def test_assigned_array_copy_snapshots(source, tmp_path):
    extra = np.arange(source.n_rows, dtype=np.float64)
    snapshot = extra.copy()
    cf = source.edit().assign(copy=True, k=extra)
    extra[:] = -1.0
    out = _written(cf, tmp_path)
    np.testing.assert_array_equal(out["k"], snapshot)


# -- deletion and renaming --


def test_delete_column(source, tmp_path):
    cf = source.edit()
    del cf["b"]
    out = _written(cf, tmp_path)
    assert set(out) == {"a", "c"}


def test_drop_multiple(source):
    cf = source.edit().drop("a", "b")
    assert cf.columns == ["c"]


def test_rename_simple(source, source_cols, tmp_path):
    cf = source.edit().rename({"a": "z"})
    assert cf.columns == ["z", "b", "c"]
    out = _written(cf, tmp_path)
    np.testing.assert_array_equal(out["z"], source_cols["a"])
    assert "a" not in out


def test_rename_swap(source, source_cols, tmp_path):
    cf = source.edit().rename({"a": "b", "b": "a"})
    out = _written(cf, tmp_path)
    np.testing.assert_array_equal(out["a"], source_cols["b"])  # name "a" now holds b's data
    np.testing.assert_array_equal(out["b"], source_cols["a"])


def test_rename_collision_raises(source):
    with pytest.raises(ValueError, match="duplicate"):
        source.edit().rename({"a": "c"})  # "c" already exists


def test_rename_missing_raises(source):
    with pytest.raises(KeyError, match="not in the frame"):
        source.edit().rename({"nope": "x"})


# -- eager validation --


def test_assign_length_mismatch_rejected(source):
    cf = source.edit()
    with pytest.raises(ValueError, match="does not match"):
        cf["bad"] = np.arange(source.n_rows + 1)


def test_assign_length_one_rejected(source):
    cf = source.edit()
    with pytest.raises(ValueError, match="does not match"):
        cf["bad"] = np.arange(1)


# -- cross-store columns --


def test_cross_store_column(source, tmp_path):
    other_cols = {"y": (np.arange(source.n_rows, dtype=np.int32) * 3)}
    other = _make_store(tmp_path, other_cols, name="other.cstore")
    try:
        cf = source.edit()
        cf["from_other"] = other.edit()["y"]  # a native leaf bound to `other`
        out = _written(cf, tmp_path)
        np.testing.assert_array_equal(out["from_other"], other_cols["y"])
    finally:
        other.close()


# -- eager compute --


def test_compute_native_and_transform(source, source_cols):
    cf = source.edit()
    np.testing.assert_array_equal(cf.array("a"), source_cols["a"])
    cf["ret"] = cf["b"] * 2.0
    np.testing.assert_allclose(cf.array("ret"), source_cols["b"] * 2.0)
    np.testing.assert_allclose(cf["ret"].compute(), source_cols["b"] * 2.0)


def test_compute_const_uses_frame_length(source):
    cf = source.edit()
    cf["flag"] = 1
    np.testing.assert_array_equal(cf.array("flag"), np.full(source.n_rows, 1))
    with pytest.raises(ValueError, match="all-constant"):
        cf["flag"].compute()  # length indeterminate without the frame


# -- committing --


def test_write_returns_open_reader(source, source_cols, tmp_path):
    reader = source.edit().write(tmp_path / "o.cstore")
    try:
        assert reader.n_rows == source.n_rows
        assert reader.columns == list(source_cols)
    finally:
        reader.close()


def test_write_respects_memory_budget(source, source_cols, tmp_path):
    cf = source.edit()
    cf["t"] = cf["b"] + cf["c"]
    out = _written(cf, tmp_path, memory_budget=256)
    np.testing.assert_allclose(out["t"], source_cols["b"] + source_cols["c"])


def test_write_empty_frame_raises(source, tmp_path):
    cf = source.edit().drop(*source.columns)
    with pytest.raises(ValueError, match="empty column mapping"):
        cf.write(tmp_path / "o.cstore")


# -- in-memory terminals: dict() / recarray() --


def test_frame_dict_matches_source_columns(source, source_cols):
    out = source.edit().dict()
    assert list(out) == ["a", "b", "c"]
    for name, arr in source_cols.items():
        assert np.array_equal(out[name], arr)


def test_frame_recarray_fields_and_values(source, source_cols):
    rec = source.edit().recarray()
    assert rec.dtype.names == ("a", "b", "c")
    assert len(rec) == source.n_rows
    for name, arr in source_cols.items():
        assert np.array_equal(rec[name], arr)


def test_frame_dict_reflects_derived_columns(source, source_cols):
    cf = source.edit()
    out = cf.assign(a2=cf["a"] * 2, s=cf["a"] + cf["b"]).dict()
    assert np.array_equal(out["a2"], source_cols["a"] * 2)
    assert np.array_equal(out["s"], source_cols["a"] + source_cols["b"])


def test_frame_in_memory_matches_write_roundtrip(source, source_cols, tmp_path):
    cf = source.edit()
    cf = cf.assign(s=cf["a"] + cf["b"]).drop("c").rename({"b": "bb"})
    in_memory = cf.dict()
    written = _written(cf, tmp_path)
    assert list(in_memory) == list(written)
    for name in in_memory:
        assert np.array_equal(in_memory[name], written[name])


def test_frame_terminals_allow_empty(source):
    cf = source.edit().drop(*source.columns)
    assert cf.dict() == {}
    assert len(cf.recarray()) == source.n_rows


def test_frame_recarray_fallback_matches_kernel(source, monkeypatch):
    from colstore import kernels

    cf = source.edit()
    cf = cf.assign(s=cf["a"] + cf["b"])
    expected = cf.recarray()
    monkeypatch.setattr(kernels, "cpp_available", lambda: False)  # force per-field assembly
    fallback = cf.recarray()
    assert fallback.dtype == expected.dtype
    for name in expected.dtype.names:
        assert np.array_equal(fallback[name], expected[name])


# -- casting: astype --


def test_frame_astype_casts_named_columns(source, source_cols):
    out = source.edit().astype({"a": "float32", "b": np.int64}).dict()
    assert out["a"].dtype == np.float32
    assert np.array_equal(out["a"], source_cols["a"].astype(np.float32))
    assert out["b"].dtype == np.int64
    assert np.array_equal(out["b"], source_cols["b"].astype(np.int64))  # float -> int truncates
    assert out["c"].dtype == source_cols["c"].dtype  # untouched column unchanged


def test_expr_astype_in_assign(source, source_cols):
    cf = source.edit()
    out = cf.assign(a32=cf["a"].astype("int32")).dict()
    assert out["a32"].dtype == np.int32


def test_astype_reflected_in_recarray_and_write(source, source_cols, tmp_path):
    rec = source.edit().astype({"a": "float32"}).recarray()
    assert rec.dtype["a"] == np.float32
    written = _written(source.edit().astype({"a": "float32"}), tmp_path)
    assert written["a"].dtype == np.float32
    assert np.array_equal(written["a"], source_cols["a"].astype(np.float32))


def test_astype_unknown_column_raises(source):
    with pytest.raises(KeyError, match="not in the frame"):
        source.edit().astype({"missing": "float32"})


def test_astype_invalid_dtype_raises(source):
    with pytest.raises(TypeError):
        source.edit().astype({"a": "definitely_not_a_dtype"})


def test_astype_validates_before_mutating(source, source_cols):
    cf = source.edit()
    with pytest.raises(TypeError):
        cf.astype({"a": "int8", "b": "definitely_not_a_dtype"})
    assert cf.dict()["a"].dtype == source_cols["a"].dtype  # the valid cast was not applied


# -- return-new default, inplace=True, and copy() --


def test_assign_returns_new_frame_by_default(source):
    cf = source.edit()
    new = cf.assign(x=cf["a"] + 1)
    assert new is not cf
    assert "x" in new.columns
    assert "x" not in cf.columns  # the original is untouched


def test_drop_rename_astype_return_new_by_default(source):
    cf = source.edit()
    assert cf.drop("c") is not cf and "c" in cf.columns
    assert cf.rename({"a": "x"}).columns[0] == "x" and "a" in cf.columns
    assert cf.astype({"a": "float32"}) is not cf
    assert cf.dict()["a"].dtype == source.dtypes["a"]  # original keeps its dtype


def test_inplace_true_edits_in_place(source):
    cf = source.edit()
    assert cf.assign(x=cf["a"] + 1, inplace=True) is cf and "x" in cf.columns
    assert cf.drop("c", inplace=True) is cf and "c" not in cf.columns
    assert cf.rename({"a": "y"}, inplace=True) is cf and "y" in cf.columns
    assert cf.astype({"y": "float32"}, inplace=True) is cf
    assert cf.dict()["y"].dtype == np.float32


def test_copy_is_independent(source):
    base = source.edit()
    base = base.assign(s=base["a"] + base["b"])
    clone = base.copy()
    clone.drop("s", inplace=True)
    assert "s" not in clone.columns
    assert "s" in base.columns  # the base is unaffected by the clone's in-place edit


def test_branch_off_a_shared_base(source, source_cols):
    base = source.edit()
    base = base.assign(d=base["a"] * 2)
    branch_a = base.astype({"a": "float32"})
    branch_b = base.drop("b")
    assert branch_a.dict()["a"].dtype == np.float32
    assert "b" not in branch_b.columns
    assert base.dict()["a"].dtype == source_cols["a"].dtype  # base unchanged by either branch
    assert "b" in base.columns


# -- select: column projection --


def test_select_projects_and_orders(source):
    cf = source.edit().select("c", "a")
    assert cf.columns == ["c", "a"]


def test_select_returns_new_by_default(source):
    base = source.edit()
    base.select("a")
    assert base.columns == ["a", "b", "c"]  # base unchanged


def test_select_inplace(source):
    cf = source.edit()
    assert cf.select("b", "a", inplace=True) is cf
    assert cf.columns == ["b", "a"]


def test_select_unknown_raises(source):
    with pytest.raises(KeyError):
        source.edit().select("a", "nope")


def test_select_duplicate_raises(source):
    with pytest.raises(ValueError, match="duplicate"):
        source.edit().select("a", "a")


# -- filter / where: row selection --


def test_filter_col_expression(source, source_cols):
    cf = source.edit().filter(col("a") >= 128)
    expected = source_cols["a"][source_cols["a"] >= 128]
    assert cf.dict()["a"].tolist() == expected.tolist()
    assert cf.n_rows == len(expected)  # where() is lazy -> n_rows resolves the selection


def test_filter_query_string(source, source_cols):
    cf = source.edit().filter("a % 4 == 1")
    mask = source_cols["a"] % 4 == 1
    assert cf.dict()["a"].tolist() == source_cols["a"][mask].tolist()


def test_filter_query_params(source, source_cols):
    cf = source.edit().filter("a >= @cut", params={"cut": 200})
    assert cf.dict()["a"].tolist() == source_cols["a"][source_cols["a"] >= 200].tolist()


def test_where_is_filter_alias(source):
    via_where = source.edit().where(col("a") < 10).dict()["a"].tolist()
    via_filter = source.edit().filter(col("a") < 10).dict()["a"].tolist()
    assert via_where == via_filter == list(range(10))


# -- frame[...] is column-name access only (no row/column indexing) --


def test_getitem_string_returns_column_expression(source, source_cols):
    cf = source.edit()
    assert cf["a"] is cf._columns["a"]  # the column's lazy expression, for building
    assert (
        cf.assign(s=cf["a"] + cf["b"]).dict()["s"].tolist()
        == (source_cols["a"] + source_cols["b"]).tolist()
    )


def test_getitem_non_name_keys_raise(source):
    cf = source.edit()
    for bad in [col("a") > 0, slice(0, 10), np.array([0, 2, 4]), 0, (slice(None), ["a", "b"])]:
        with pytest.raises(TypeError, match="does not slice or index rows or columns"):
            _ = cf[bad]


def test_filter_composes(source, source_cols):
    cf = source.edit().filter(col("a") % 2 == 0).filter(col("a") > 250)
    a = source_cols["a"]
    expected = a[(a % 2 == 0) & (a > 250)]
    assert cf.dict()["a"].tolist() == expected.tolist()


def test_filter_returns_new_by_default(source):
    base = source.edit()
    base.filter(col("a") < 5)
    assert base.n_rows == 256  # base unchanged


def test_filter_inplace(source):
    cf = source.edit()
    assert cf.filter(col("a") < 5, inplace=True) is cf
    assert cf.n_rows == 5


def test_filtered_terminals_agree(source, source_cols):
    cf = source.edit().filter(col("a") >= 250)
    a = source_cols["a"]
    keep = a >= 250
    assert cf.array("a").tolist() == a[keep].tolist()
    assert cf.dict()["a"].tolist() == a[keep].tolist()
    rec = cf.recarray()
    assert rec["a"].tolist() == a[keep].tolist()
    assert rec.shape[0] == int(keep.sum())


def test_filter_then_derive(source, source_cols):
    cf = source.edit().filter(col("a") >= 250)
    cf = cf.assign(d=cf["a"] + 1)
    a = source_cols["a"]
    keep = a >= 250
    assert cf.dict()["d"].tolist() == (a[keep] + 1).tolist()


def test_filter_empty_match(source):
    cf = source.edit().filter(col("a") > 10_000)
    assert cf.n_rows == 0
    assert cf.dict()["a"].tolist() == []
    assert cf.recarray().shape[0] == 0


def test_where_is_lazy_n_rows_resolves(source, source_cols):
    cf = source.edit().where(col("a") >= 200)
    expected = int((source_cols["a"] >= 200).sum())
    assert cf.n_rows == expected  # a pending predicate is resolved on access, not thrown
    assert cf.dict()["a"].tolist() == source_cols["a"][source_cols["a"] >= 200].tolist()
    # composing where() does not evaluate until materialized
    cf2 = cf.where(col("a") < 210)
    assert cf2.dict()["a"].tolist() == [200, 201, 202, 203, 204, 205, 206, 207, 208, 209]


# -- base-attach restriction: external data joins only at the base --


def test_raw_array_on_filtered_frame_rejected(source):
    cf = source.edit().where(col("a") >= 200)
    with pytest.raises(ValueError, match="row selection"):
        cf.assign(x=np.zeros(56))  # selected-length raw array onto a filtered frame
    with pytest.raises(ValueError, match="row selection"):
        cf.assign(x=np.zeros(source.n_rows))  # even source-length is rejected (ambiguous)


def test_base_attached_array_cofilters(source, source_cols):
    extra = np.arange(source.n_rows, dtype=np.int64) * 10
    cf = source.edit().assign(x=extra).where(col("a") >= 200)  # attach at base, then filter
    keep = source_cols["a"] >= 200
    assert cf.dict()["x"].tolist() == extra[keep].tolist()  # x co-filtered with the rest


def test_cross_frame_column_allowed_on_filtered_frame(source, source_cols, tmp_path):
    y = np.arange(source.n_rows, dtype=np.int64) * 2
    other = _make_store(tmp_path, {"y": y}, name="other2.cstore")
    try:
        # a column from another frame is a lazy Expr, not a raw array -> it co-filters
        cf = source.edit().where(col("a") >= 200).assign(z=other.edit()["y"])
        keep = source_cols["a"] >= 200
        assert cf.dict()["z"].tolist() == y[keep].tolist()
    finally:
        other.close()


def test_filter_unknown_column_raises(source):
    with pytest.raises(colstore.QueryError):
        source.edit().filter(col("nope") > 0)


def test_filter_non_boolean_raises(source):
    with pytest.raises(colstore.QueryError):
        source.edit().filter(col("a") + 1)


def test_filter_on_dropped_column_raises(source):
    # Filtering resolves against the frame's columns in order, so a column dropped
    # earlier is no longer a valid predicate target.
    with pytest.raises(colstore.QueryError):
        source.edit().drop("a").filter(col("a") >= 128)


def test_filter_on_derived_column(source, source_cols):
    cf = source.edit().assign(r=col("a") + 1).filter(col("r") > 128)
    a = source_cols["a"]
    keep = (a + 1) > 128
    assert cf.dict()["r"].tolist() == (a[keep] + 1).tolist()


def test_filter_after_rename_uses_new_name(source, source_cols):
    a = source_cols["a"]
    cf = source.edit().rename({"a": "x"}).filter(col("x") >= 200)
    assert cf.dict()["x"].tolist() == a[a >= 200].tolist()
    with pytest.raises(colstore.QueryError):
        source.edit().rename({"a": "x"}).filter(col("a") >= 200)


def test_filter_after_reassign_sees_new_definition(source, source_cols):
    cf = source.edit().assign(a=col("a") * 2).filter(col("a") > 200)
    doubled = source_cols["a"] * 2
    assert cf.dict()["a"].tolist() == doubled[doubled > 200].tolist()


def test_filtered_write_roundtrip(source, source_cols, tmp_path):
    cf = source.edit().filter(col("a") % 10 == 0)
    cf = cf.assign(d=cf["a"] * 2)
    out = _written(cf, tmp_path)
    a = source_cols["a"]
    keep = a % 10 == 0
    assert out["a"].tolist() == a[keep].tolist()
    assert out["d"].tolist() == (a[keep] * 2).tolist()


# -- col() as an assign value (one col() for both filter and assign) --


def test_assign_col_value_matches_native(source):
    cf = source.edit()
    via_col = cf.assign(s=col("a") + col("b")).dict()["s"]
    via_native = cf.assign(s=cf["a"] + cf["b"]).dict()["s"]
    assert np.array_equal(via_col, via_native)


def test_assign_col_comparison_value(source, source_cols):
    got = source.edit().assign(f=col("a") >= 128).dict()["f"]
    assert got.dtype == np.bool_
    assert got.tolist() == (source_cols["a"] >= 128).tolist()


def test_assign_col_invert_value(source, source_cols):
    got = source.edit().assign(n=~(col("a") >= 128)).dict()["n"]
    assert got.tolist() == (~(source_cols["a"] >= 128)).tolist()


def test_assign_col_alias(source, source_cols):
    got = source.edit().assign(z=col("a")).dict()["z"]
    assert got.tolist() == source_cols["a"].tolist()


def test_assign_col_isin_value(source, source_cols):
    got = source.edit().assign(m=col("a").isin([1, 2, 3])).dict()["m"]
    assert got.tolist() == np.isin(source_cols["a"], [1, 2, 3]).tolist()


def test_assign_col_references_derived_column(source, source_cols):
    cf = source.edit().assign(r=col("a") + col("b"))
    got = cf.assign(r2=col("r") * 2).dict()["r2"]
    assert np.allclose(got, (source_cols["a"] + source_cols["b"]) * 2)


def test_assign_col_mixed_with_frame_expr(source, source_cols):
    cf = source.edit()
    got = cf.assign(x=2 * col("a") + cf["b"]).dict()["x"]
    assert np.allclose(got, 2 * source_cols["a"] + source_cols["b"])


def test_elementwise_numpy_idioms_defer_and_stream(source, source_cols, tmp_path):
    b, c = source_cols["b"], source_cols["c"]
    cf = source.edit()
    cf["r"] = np.hypot(cf["b"], cf["c"])  # __setitem__ with an elementwise ufunc
    cf = cf.assign(phi=np.arctan2(cf["c"], cf["b"]))  # the assign idiom
    cf["clipped"] = np.clip(cf["b"], -0.5, 0.5)  # np.clip via __array_function__
    cf["pick"] = np.where(cf["b"] > 0, cf["b"], cf["c"])  # np.where
    got = cf.dict()
    np.testing.assert_allclose(got["r"], np.hypot(b, c))
    np.testing.assert_allclose(got["phi"], np.arctan2(c, b))
    np.testing.assert_allclose(got["clipped"], np.clip(b, -0.5, 0.5))
    np.testing.assert_array_equal(got["pick"], np.where(b > 0, b, c))
    out = _written(cf, tmp_path)  # and the same graph streams to a file
    np.testing.assert_allclose(out["r"], np.hypot(b, c))


def test_apply_derives_column_and_streams(source, source_cols, tmp_path):
    b, c = source_cols["b"], source_cols["c"]
    cf = source.edit()
    cf["r"] = cf.apply(lambda u, v: np.hypot(u, v), "b", "c")  # column names
    cf = cf.assign(s=cf.apply(lambda u, v: np.sqrt(u**2 + v**2), col("b"), col("c")))  # col() refs
    got = cf.dict()
    np.testing.assert_allclose(got["r"], np.hypot(b, c))
    np.testing.assert_allclose(got["s"], np.sqrt(b**2 + c**2))
    out = _written(cf, tmp_path)  # the function runs per streamed batch
    np.testing.assert_allclose(out["r"], np.hypot(b, c))


def test_apply_out_dtype_and_composes(source, source_cols):
    cf = source.edit()
    expr = cf.apply(lambda u: np.sqrt(u.astype(np.float64)), "a", out_dtype="float32")
    rooted = cf.assign(rooted=expr).dict()["rooted"]
    assert rooted.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        rooted, np.sqrt(source_cols["a"].astype(np.float64)).astype(np.float32), rtol=1e-6
    )
    scaled = cf.assign(scaled=expr * 2).dict()["scaled"]  # composes with operators
    np.testing.assert_allclose(scaled, rooted * 2, rtol=1e-6)


def test_apply_unknown_column_raises(source):
    with pytest.raises(KeyError, match="not a column of this frame"):
        source.edit().apply(lambda u: u, "nope")


def test_apply_no_columns_raises(source):
    with pytest.raises(TypeError, match="at least one column"):
        source.edit().apply(lambda: np.array([]))


def test_reductions_over_columns(source, source_cols):
    a, b = source_cols["a"], source_cols["b"]
    cf = source.edit()
    assert cf.count() == len(a)
    np.testing.assert_allclose(cf.sum("b"), b.sum())
    np.testing.assert_allclose(cf.mean("b"), b.mean())
    np.testing.assert_allclose(cf.min("b"), b.min())
    np.testing.assert_allclose(cf.max("b"), b.max())
    assert int(cf.sum("a")) == int(a.sum())  # integer column widens, no overflow


def test_reductions_respect_where_and_expressions(source, source_cols):
    a, b = source_cols["a"], source_cols["b"]
    mask = a >= 100
    cf = source.edit().where(col("a") >= 100)
    assert cf.count() == int(mask.sum())
    np.testing.assert_allclose(cf.sum("b"), b[mask].sum())
    np.testing.assert_allclose(cf.mean("b"), b[mask].mean())
    np.testing.assert_allclose(cf.sum(col("b") * 2), (b[mask] * 2).sum())  # derived expression


def test_reductions_empty_selection(source):
    cf = source.edit().where("a < 0")  # a = arange(n) >= 0, so nothing matches
    assert cf.count() == 0
    assert cf.sum("b") == 0
    assert np.isnan(cf.mean("b"))
    assert np.isnan(cf.min("b"))
    assert np.isnan(cf.max("b"))


def test_reductions_combine_across_batches(source, source_cols, monkeypatch):
    import colstore.frame as frame_mod

    monkeypatch.setattr(frame_mod, "_REDUCTION_CHUNK_BYTES", 16)  # ~2 float64/batch
    b = source_cols["b"]
    cf = source.edit()
    # A transform can't be a zero-copy view, so it folds in chunks; the per-batch
    # partials must combine across batches (a bare column would fold one view instead).
    expr = col("b") * 1
    np.testing.assert_allclose(cf.sum(expr), b.sum())
    np.testing.assert_allclose(cf.mean(expr), b.mean())
    np.testing.assert_allclose(cf.min(expr), b.min())
    np.testing.assert_allclose(cf.max(expr), b.max())


def test_reductions_float32_wide_accumulator(source, source_cols, monkeypatch):
    import colstore.frame as frame_mod

    c64 = source_cols["c"].astype(np.float64)  # the float32 column, promoted, for a reference
    cf = source.edit()
    full = float(cf.sum("c"))  # bare column -> one zero-copy view, single pass
    monkeypatch.setattr(frame_mod, "_REDUCTION_CHUNK_BYTES", 8)  # 2 float32 rows/batch
    expr = col("c") * 1  # a transform -> chunked fold across batches
    np.testing.assert_allclose(float(cf.sum(expr)), full, rtol=1e-10)  # chunk-independent
    np.testing.assert_allclose(full, c64.sum(), rtol=1e-10)  # matches a float64 single pass
    np.testing.assert_allclose(cf.mean(expr), c64.mean(), rtol=1e-10)
    assert np.asarray(cf.sum(expr)).dtype == np.dtype(np.float64)  # float32 sums to float64


def test_reduction_views_bare_column_without_chunking(source, source_cols, monkeypatch):
    import colstore.frame as frame_mod

    # A tiny chunk would force many batches if the column were materialized; a bare stored
    # column on a single-record store folds one read-only zero-copy view instead, no copy.
    monkeypatch.setattr(frame_mod, "_REDUCTION_CHUNK_BYTES", 8)
    cf = source.edit()
    expr = cf._resolve_value_column("b")
    batches = list(cf._stream_column(expr))
    assert len(batches) == 1  # one zero-copy view, not chunked
    np.testing.assert_allclose(np.asarray(batches[0]), source_cols["b"])
    np.testing.assert_allclose(cf.sum("b"), source_cols["b"].sum())


def test_reductions_non_numeric_column(tmp_path):
    store = _make_store(tmp_path, {"s": np.array(["a", "bb", "ccc"], "S3")}, name="strings.cstore")
    try:
        cf = store.edit()
        # NumPy has no add/minimum loop for strings; every reduction gives a clear error.
        for reduce in (cf.sum, cf.mean, cf.min, cf.max):
            with pytest.raises(TypeError, match="column"):
                reduce("s")
    finally:
        store.close()


def test_assign_col_unknown_raises(source):
    with pytest.raises(KeyError, match="not a column of this frame"):
        source.edit().assign(z=col("nope"))


def test_col_value_streams_to_write(source, source_cols, tmp_path):
    cf = source.edit().assign(s=col("a") * col("b"), m=col("a").isin([5, 10, 15]))
    out = _written(cf, tmp_path)
    assert np.allclose(out["s"], source_cols["a"] * source_cols["b"])
    assert out["m"].tolist() == np.isin(source_cols["a"], [5, 10, 15]).tolist()


# -- iter_batches: bounded-memory generator yielding materialized frames --


def test_iter_batches_yields_materialized_frames(source):
    cf = source.edit().assign(s=col("a") + col("b"))
    batches = list(cf.iter_batches(batch_size=100))
    assert all(isinstance(b, ColStoreFrame) for b in batches)
    assert [b.n_rows for b in batches] == [100, 100, 56]
    full = cf.recarray()
    assert batches[0].recarray().dtype == full.dtype
    assert np.array_equal(np.concatenate([b.recarray() for b in batches]), full)


def test_iter_batch_is_in_memory(source):
    first = next(iter(source.edit().iter_batches(batch_size=10)))
    assert first.n_rows == 10
    assert first.dict()["a"].tolist() == list(range(10))


def test_iter_batch_frame_is_composable(source):
    # each batch is a full frame: edit it, then convert to any format
    batch = next(iter(source.edit().iter_batches(batch_size=8)))
    assert batch.assign(x=col("a") + 1).dict()["x"].tolist() == list(range(1, 9))


def test_iter_batches_memory_string(source):
    cf = source.edit().select("a")  # one int64 column, 8 bytes/row
    batches = list(cf.iter_batches("1 KiB"))  # 1024 // 8 = 128 rows per batch
    assert [b.n_rows for b in batches] == [128, 128]
    assert np.concatenate([b.recarray() for b in batches]).tolist() == cf.recarray().tolist()


def test_iter_batches_default_equals_recarray(source):
    cf = source.edit()
    got = np.concatenate([b.recarray() for b in cf.iter_batches()])
    assert np.array_equal(got, cf.recarray())


def test_iter_batches_single_batch_large_budget(source):
    cf = source.edit()
    batches = list(cf.iter_batches("1 GiB"))
    assert len(batches) == 1 and batches[0].n_rows == cf.n_rows
    assert np.array_equal(batches[0].recarray(), cf.recarray())


def test_iter_batches_copy_false_returns_views(source, source_cols):
    # copy=False yields a read-only zero-copy view of a single-record native store: the
    # batch column shares memory with the store's open column memmap, no gather.
    cf = source.edit()
    backing = source._memmaps["b"]
    viewed = next(iter(cf.iter_batches(batch_size=100, copy=False))).dict()["b"]
    assert np.shares_memory(viewed, backing)
    owned = next(iter(cf.iter_batches(batch_size=100, copy=True))).dict()["b"]
    assert not np.shares_memory(owned, backing)  # copy=True owns its arrays


def test_iter_batches_copy_false_matches_copy_true(source, source_cols):
    # copy=False reuses per-column buffers on gathered batches, so a batch is valid only
    # until the next is drawn: copy each one out before advancing (the streaming contract).
    cf = source.edit()
    viewed = np.concatenate([b.dict()["b"].copy() for b in cf.iter_batches(100, copy=False)])
    np.testing.assert_array_equal(viewed, source_cols["b"])
    # a transform can't be a view -> still materialized correctly under copy=False
    derived = source.edit().assign(d=col("b") * 2)
    got = np.concatenate([b.dict()["d"].copy() for b in derived.iter_batches(100, copy=False)])
    np.testing.assert_allclose(got, source_cols["b"] * 2)
    # a filtered (fancy) selection gathers into a reused buffer -> still correct per batch
    mask = source_cols["a"] >= 100
    filt = source.edit().where(col("a") >= 100)
    got2 = np.concatenate([b.dict()["b"].copy() for b in filt.iter_batches(50, copy=False)])
    np.testing.assert_array_equal(got2, source_cols["b"][mask])


def test_iter_batches_copy_false_reuses_buffer_multirecord(tmp_path):
    from colstore import testing

    # A multi-record store interleaves columns on disk, so copy=False cannot view -- it
    # gathers into a per-column buffer reused across batches.
    full = testing.make_columns(2000, 2, names=("a", "b"), seed=0)
    testing.write_columns(tmp_path / "mr.cstore", full, records=50).close()
    store = colstore.open(tmp_path / "mr.cstore")
    try:
        cf = store.edit()
        it = iter(cf.iter_batches(200, copy=False))
        first = next(it).dict()["a"]
        second = next(it).dict()["a"]
        assert np.shares_memory(first, second)  # the same buffer, reused across batches
        got = np.concatenate([b.dict()["a"].copy() for b in cf.iter_batches(200, copy=False)])
        np.testing.assert_array_equal(got, full["a"])
    finally:
        store.close()


def test_iter_batches_copy_false_multifile(tmp_path):
    from colstore import testing

    # A boundary-spanning batch on a multi-file dataset gathers across the file seam. copy=False
    # must yield correct values there, and the dataset fills the caller's out= buffer (so the
    # per-column reuse works across the seam too, not just within a single file).
    a = testing.make_columns(10, 2, names=("x", "y"), seed=1)
    b = testing.make_columns(10, 2, names=("x", "y"), seed=2)
    testing.write_columns(tmp_path / "part_0.cstore", a, records=1).close()
    testing.write_columns(tmp_path / "part_1.cstore", b, records=1).close()
    ds = colstore.open(str(tmp_path / "part_*.cstore"))
    try:
        got = np.concatenate(
            [fr.dict()["x"].copy() for fr in ds.edit().iter_batches(7, copy=False)]
        )
        np.testing.assert_array_equal(got, np.concatenate([a["x"], b["x"]]))
        buf = np.empty(7, dtype=ds.dtypes["x"])  # a slice [7, 14) crossing the file seam
        filled = ds._gather_one("x", slice(7, 14), out=buf)
        assert np.shares_memory(filled, buf)  # the dataset filled the buffer; reuse works
        np.testing.assert_array_equal(filled, np.concatenate([a["x"], b["x"]])[7:14])
    finally:
        ds.close()


def test_iter_batches_filtered_gathers_survivors(source, source_cols):
    cf = source.edit().where(col("a") % 3 == 0)
    batches = list(cf.iter_batches(batch_size=20))
    assert sum(b.n_rows for b in batches) == int((source_cols["a"] % 3 == 0).sum())
    assert np.array_equal(np.concatenate([b.recarray() for b in batches]), cf.recarray())


def test_iter_batches_derived_column(source, source_cols):
    cf = source.edit().assign(d=col("a") * 2)
    got = np.concatenate([b.recarray() for b in cf.iter_batches(batch_size=64)])
    assert got["d"].tolist() == (source_cols["a"] * 2).tolist()


def test_iter_batches_empty_selection_yields_nothing(source):
    assert list(source.edit().where(col("a") > 10_000).iter_batches()) == []


def test_iter_batches_no_columns_yields_nothing(source):
    assert list(source.edit().drop("a", "b", "c").iter_batches()) == []


def test_iter_batches_invalid_size_raises(source):
    cf = source.edit()
    with pytest.raises(ValueError, match="must be positive"):
        list(cf.iter_batches(0))
    with pytest.raises(ValueError):  # unparseable memory string
        list(cf.iter_batches("not a size"))
    with pytest.raises(TypeError):  # wrong type
        list(cf.iter_batches(1.5))


# -- bounded filtered write: streams the selected rows, batch by batch --


def test_filtered_write_matches_in_memory(source, source_cols, tmp_path):
    cf = source.edit().where(col("a") >= 200)
    out = _written(cf, tmp_path, memory_budget=64)  # tiny budget -> many gather batches
    keep = source_cols["a"] >= 200
    assert out["a"].tolist() == source_cols["a"][keep].tolist()
    assert np.allclose(out["b"], source_cols["b"][keep])


def test_filtered_write_with_derived_column(source, source_cols, tmp_path):
    cf = source.edit().where(col("a") % 2 == 0).assign(d=col("a") * 10)
    out = _written(cf, tmp_path, memory_budget=128)
    keep = source_cols["a"] % 2 == 0
    assert out["d"].tolist() == (source_cols["a"][keep] * 10).tolist()


def test_filtered_write_empty_selection(source, tmp_path):
    out = _written(source.edit().where(col("a") > 10_000), tmp_path)
    assert out["a"].shape[0] == 0


def test_filtered_write_no_columns_raises(source, tmp_path):
    cf = source[np.arange(10)].edit().drop("a", "b", "c")
    with pytest.raises(ValueError, match="empty column mapping"):
        cf.write(tmp_path / "o.cstore")


# -- report(): labeled cuts + cutflow --


def test_where_label_named_and_positional(source):
    cf = source.edit().where(col("a") >= 100, "ge100").where(col("a") < 200, label="lt200")
    assert [c.label for c in cf.report()] == ["ge100", "lt200"]


def test_report_cutflow_counts_are_marginal(source, source_cols):
    a = source_cols["a"]
    cf = source.edit().where(col("a") >= 100, "ge100").where(col("a") % 2 == 0, "even")
    rep = cf.report()
    ge = a >= 100
    assert (rep["ge100"].entering, rep["ge100"].passing) == (256, int(ge.sum()))  # all in, ge pass
    assert rep["even"].entering == int(ge.sum())  # only ge100's survivors enter the next cut
    assert rep["even"].passing == int((ge & (a % 2 == 0)).sum())


def test_report_efficiency(source):
    info = source.edit().where(col("a") < 64, "q").report()["q"]
    assert info.efficiency == info.passing / info.entering == 64 / 256


def test_report_unlabeled_cut_gets_positional_name(source):
    rep = source.edit().where(col("a") >= 10).where(col("a") < 20, "win").report()
    assert [c.label for c in rep] == ["#0", "win"]


def test_report_empty_when_no_cuts(source):
    rep = source.edit().report()
    assert len(rep) == 0 and list(rep) == []
    assert "no cuts" in repr(rep)


def test_report_index_by_label_and_position(source):
    rep = source.edit().where(col("a") >= 100, "ge").report()
    assert rep[0] == rep["ge"]
    with pytest.raises(KeyError, match="no cut labeled"):
        _ = rep["missing"]


def test_filter_label_feeds_report(source):
    assert source.edit().filter(col("a") < 5, "small").report()["small"].passing == 5


def test_report_over_concrete_base(source, source_cols):
    # a concrete base selection (ds[idx].edit()) is the first cut's entering count
    idx = np.arange(50)
    rep = source[idx].edit().where(col("a") >= 20, "ge20").report()
    assert rep["ge20"].entering == 50
    assert rep["ge20"].passing == int((source_cols["a"][idx] >= 20).sum())


def test_report_final_passing_matches_selection(source):
    cf = source.edit().where(col("a") >= 100, "ge").where(col("a") < 200, "lt")
    assert cf.report()[-1].passing == cf.n_rows == len(cf.dict()["a"])


def test_report_repr_is_a_table(source):
    text = repr(source.edit().where(col("a") >= 100, "ge100").report())
    assert "ge100" in text and "entering" in text and "passing" in text


# -- report(): weighted cutflow --


def test_where_weight_sticky_and_sums(source, source_cols):
    a = source_cols["a"]
    wt = a.astype(np.float64) + 1.0
    cf = (
        source.edit()
        .assign(w=col("a") + 1.0)
        .where(col("a") >= 100, "ge", weight="w")
        .where(col("a") % 2 == 0, "even")  # no weight -> inherits "w"
    )
    rep = cf.report()
    ge = a >= 100
    assert np.isclose(rep["ge"].weighted_entering, wt.sum())  # all rows enter
    assert np.isclose(rep["ge"].weighted_passing, wt[ge].sum())
    assert np.isclose(rep["even"].weighted_entering, wt[ge].sum())  # sticky weight carries
    assert np.isclose(rep["even"].weighted_passing, wt[ge & (a % 2 == 0)].sum())


def test_report_weighted_efficiency(source):
    info = (
        source.edit()
        .assign(w=col("a") + 1.0)
        .where(col("a") < 128, "half", weight="w")
        .report()["half"]
    )
    assert info.weighted_efficiency == info.weighted_passing / info.weighted_entering


def test_where_weight_by_expression(source, source_cols):
    a = source_cols["a"]
    info = source.edit().where(col("a") >= 100, "ge", weight=col("a") * 2).report()["ge"]
    assert np.isclose(info.weighted_passing, (a[a >= 100] * 2).sum())


def test_report_show_modes(source):
    cf = source.edit().assign(w=col("a") + 1.0).where(col("a") >= 100, "ge", weight="w")
    raw, weighted, both = (repr(cf.report(m)) for m in ("raw", "weighted", "both"))
    assert "wt_entering" not in raw and "entering" in raw
    assert weighted.splitlines()[0].split() == ["cut", "wt_entering", "wt_passing", "wt_eff"]
    assert both.splitlines()[0].split() == [
        "cut",
        "entering",
        "passing",
        "eff",
        "wt_entering",
        "wt_passing",
        "wt_eff",
    ]


def test_report_show_invalid_raises(source):
    with pytest.raises(ValueError, match="show must be"):
        source.edit().where(col("a") >= 100).report("bogus")


def test_unweighted_cut_has_no_weighted_stats(source):
    info = source.edit().where(col("a") >= 100, "ge").report()["ge"]
    assert info.weighted_entering is None and info.weighted_efficiency is None


def test_where_weight_non_numeric_raises(source):
    with pytest.raises(TypeError, match="numeric"):
        source.edit().where(col("a") >= 100, weight=col("a") >= 5)  # boolean expr is not numeric


def test_report_repr_html(source):
    cf = source.edit().assign(w=col("a") + 1.0).where(col("a") >= 100, "ge100", weight="w")
    markup = cf.report()._repr_html_()
    assert "cstore-tbl" in markup and "ge100" in markup and "wt_entering" in markup
    assert "no cuts" in source.edit().report()._repr_html_()  # empty report still renders


def test_report_records(source):
    recs = (
        source.edit()
        .assign(w=col("a") + 1.0)
        .where(col("a") >= 100, "ge", weight="w")
        .report()
        .records()
    )
    assert isinstance(recs, list) and isinstance(recs[0], dict)
    assert set(recs[0]) == {
        "label",
        "entering",
        "passing",
        "efficiency",
        "weighted_entering",
        "weighted_passing",
        "weighted_efficiency",
    }
    assert recs[0]["label"] == "ge" and recs[0]["entering"] == 256
    for value in recs[0].values():  # JSON-friendly types only
        assert value is None or isinstance(value, (str, int, float))


def test_report_records_unweighted_has_none(source):
    rec = source.edit().where(col("a") >= 100, "ge").report().records()[0]
    assert rec["weighted_entering"] is None and rec["weighted_efficiency"] is None


# -- boolean-mask selection (kept as a mask; routed mask-native in test_calibration) --


def test_edit_mask_selection_terminals(source, source_cols):
    mask = source_cols["a"] % 3 == 0
    cf = source[mask].edit()
    assert cf.n_rows == int(mask.sum())  # popcount of the selected rows
    assert np.array_equal(cf.dict()["a"], source_cols["a"][mask])
    assert np.array_equal(cf.array("b"), source_cols["b"][mask])
    assert np.array_equal(cf.recarray()["c"], source_cols["c"][mask])


def test_edit_mask_selection_streams_through_indices(source, source_cols, tmp_path):
    mask = source_cols["a"] % 2 == 0
    # write() / iter_batches() / the reductions lower the mask to indices so a
    # per-batch slice selects source rows, not mask positions.
    assert np.array_equal(_written(source[mask].edit(), tmp_path)["a"], source_cols["a"][mask])
    batched = np.concatenate(
        [b.dict()["a"] for b in source[mask].edit().iter_batches(batch_size=16)]
    )
    assert np.array_equal(batched, source_cols["a"][mask])
    assert source[mask].edit().sum("a") == source_cols["a"][mask].sum()
    assert source[mask].edit().min("b") == source_cols["b"][mask].min()


def test_edit_mask_composes_with_where(source, source_cols):
    mask = source_cols["a"] % 2 == 0
    combined = mask & (source_cols["b"] > 0)
    cf = source[mask].edit().where(col("b") > 0)
    assert cf.n_rows == int(combined.sum())
    assert np.array_equal(cf.dict()["a"], source_cols["a"][combined])
    assert cf.report().records()[0]["entering"] == int(mask.sum())  # mask base = entering count


def test_edit_base_mask_lowered_on_single_record(source, source_cols):
    # A single-record store has no mask-native kernel (it is contiguous, gathered by
    # index), so a base mask -- even a dense one -- is lowered to indices at construction
    # rather than kept (which would only repeat flatnonzero per column for no gain).
    mask = source_cols["a"] % 2 == 0  # dense (0.5)
    cf = source[mask].edit()
    assert not source._is_multi_record
    assert cf._rows.dtype == np.int64
    assert cf.n_rows == int(mask.sum())
    assert np.array_equal(cf.dict()["a"], source_cols["a"][mask])


# -- index-memory guard (hard error when the survivor index would not fit RAM) --


def test_resolve_guards_index_against_available_memory(source, tmp_path, monkeypatch):
    # A terminal that materializes the survivor-index array hard-errors when the index would
    # exceed available RAM, instead of letting the process OOM.
    monkeypatch.setattr("colstore.frame._available_memory", lambda: 8)  # ~8 bytes free
    pred = col("a") >= 0  # selects every row -> count * 8 >> 8
    with pytest.raises(MemoryError, match="row index"):
        source[pred].edit().dict()
    with pytest.raises(MemoryError, match="row index"):
        source[pred].edit().write(tmp_path / "out.cstore")


def test_n_rows_and_count_throw_free_under_low_memory(source, source_cols, monkeypatch):
    # n_rows / count must never throw -- they count via the predicate mask, never the index.
    monkeypatch.setattr("colstore.frame._available_memory", lambda: 8)
    expected = int((source_cols["a"] >= 0).sum())
    cf = source[col("a") >= 0].edit()
    assert cf.n_rows == expected
    assert cf.count() == expected


def test_index_guard_skipped_when_memory_unknown(source, source_cols, monkeypatch):
    # Best-effort: when available memory can't be determined, the guard is skipped.
    monkeypatch.setattr("colstore.frame._available_memory", lambda: None)
    got = source[col("a") >= 0].edit().dict()["a"]
    assert np.array_equal(got, source_cols["a"][source_cols["a"] >= 0])
