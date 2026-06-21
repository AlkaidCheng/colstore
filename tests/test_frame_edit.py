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
from colstore import ColStoreFrame
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
    np.testing.assert_array_equal(cf.compute("a"), source_cols["a"])
    cf["ret"] = cf["b"] * 2.0
    np.testing.assert_allclose(cf.compute("ret"), source_cols["b"] * 2.0)
    np.testing.assert_allclose(cf["ret"].compute(), source_cols["b"] * 2.0)


def test_compute_const_uses_frame_length(source):
    cf = source.edit()
    cf["flag"] = 1
    np.testing.assert_array_equal(cf.compute("flag"), np.full(source.n_rows, 1))
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
