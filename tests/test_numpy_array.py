"""Tests for the NumPy array interface (``__array__``) on readers and views.

``np.asarray`` / ``np.array`` of a colstore object materialize through the same
seams as ``.array()`` / ``.recarray()``: a single-column selection becomes a 1-D
array, and several columns -- or a whole reader or dataset -- become a structured
record array. Covers dtype casting, the ``copy=False`` zero-copy/raise contract,
row subsets, multi-record and multi-file sources, and consumption by NumPy
functions.
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import testing


def _store(tmp_path, columns, *, records=1):
    """Open a store of ``columns`` (single- or multi-record)."""
    path = tmp_path / "a.cstore"
    if records == 1:
        return colstore.store(columns, path, show_progress=False)
    return testing.write_columns(path, columns, records=records)


def _dataset(tmp_path, blocks):
    """Open a multi-file dataset from a list of per-file column dicts."""
    paths = []
    for i, block in enumerate(blocks):
        path = tmp_path / f"f{i}.cstore"
        colstore.store(block, path, show_progress=False).close()
        paths.append(path)
    return colstore.open(paths)


# ---- single column -> 1-D array --------------------------------------------


def test_column_view_is_1d_array(small_store):
    arr = np.array(small_store["vol"])
    assert arr.ndim == 1
    assert arr.dtype == np.float64
    np.testing.assert_array_equal(arr, small_store.array("vol"))


def test_asarray_matches_array_method(small_store):
    for name in small_store.columns:
        np.testing.assert_array_equal(np.asarray(small_store[name]), small_store.array(name))


def test_column_array_is_owning_and_writeable(small_store):
    arr = np.array(small_store["id"])
    assert arr.flags["WRITEABLE"]
    arr[0] = 999  # owning copy: mutating it does not touch the store
    assert small_store.array("id")[0] == 0


# ---- whole store / multiple columns -> record array ------------------------


def test_reader_is_record_array(small_store):
    rec = np.array(small_store)
    assert rec.dtype.names == tuple(small_store.columns)
    np.testing.assert_array_equal(rec, small_store.recarray())


def test_table_view_is_record_array(small_store):
    rec = np.asarray(small_store[["vol", "id"]])
    assert rec.dtype.names == ("vol", "id")
    np.testing.assert_array_equal(rec["vol"], small_store.array("vol"))
    np.testing.assert_array_equal(rec["id"], small_store.array("id"))


def test_single_element_list_is_record_array_not_1d(small_store):
    # ds["x"] is 1-D; ds[["x"]] is a one-field record array -- the distinction is
    # preserved through __array__.
    assert np.array(small_store["vol"]).dtype.names is None
    assert np.asarray(small_store[["vol"]]).dtype.names == ("vol",)


# ---- dtype casting ----------------------------------------------------------


def test_dtype_cast_on_column(small_store):
    arr = np.array(small_store["qty"], dtype=np.float64)
    assert arr.dtype == np.float64
    np.testing.assert_array_equal(arr, small_store.array("qty").astype(np.float64))


def test_dtype_none_preserves_stored_dtype(small_store):
    assert np.asarray(small_store["flag"]).dtype == np.uint8
    assert np.asarray(small_store["price"]).dtype == np.float32


# ---- copy=False contract ----------------------------------------------------


def test_copy_false_single_record_is_readonly_view(small_store):
    view = np.array(small_store["vol"], copy=False)
    assert not view.flags["WRITEABLE"]
    np.testing.assert_array_equal(view, small_store.array("vol"))
    # aliases the same mapping as an explicit zero-copy read
    assert np.shares_memory(view, small_store.array("vol", copy=False))


def test_copy_false_record_array_raises(small_store):
    with pytest.raises(ValueError, match="without copying"):
        np.array(small_store, copy=False)
    with pytest.raises(ValueError, match="without copying"):
        np.array(small_store[["vol", "id"]], copy=False)


def test_copy_false_multirecord_column_raises(tmp_path):
    store = _store(tmp_path, {"a": np.arange(12, dtype=np.int64)}, records=3)
    with pytest.raises(ValueError):
        np.array(store["a"], copy=False)


def test_copy_false_with_casting_dtype_raises(small_store):
    # A dtype change forces a copy, which copy=False forbids (numpy's no-copy contract):
    # qty is int32, so casting to float64 cannot be a view.
    with pytest.raises(ValueError, match="requires a copy"):
        np.array(small_store["qty"], dtype=np.float64, copy=False)


def test_copy_false_with_matching_dtype_stays_a_view(small_store):
    # A no-op dtype (already the stored dtype) keeps the zero-copy read-only view.
    view = np.array(small_store["vol"], dtype=np.float64, copy=False)
    assert not view.flags["WRITEABLE"]
    assert np.shares_memory(view, small_store.array("vol", copy=False))


def test_default_copy_is_owning_for_record(small_store):
    # np.array defaults to copy=True; mutating the result must not alter the store.
    rec = np.array(small_store)
    rec["id"][0] = -1
    assert small_store.array("id")[0] == 0


# ---- row subsets ------------------------------------------------------------


def test_slice_rows(small_store):
    np.testing.assert_array_equal(
        np.array(small_store[10:20, "vol"]), small_store.array("vol")[10:20]
    )


def test_fancy_rows(small_store):
    idx = [5, 1, 1, 100, 0]
    np.testing.assert_array_equal(np.array(small_store[idx, "qty"]), small_store.array("qty")[idx])


def test_boolean_mask_rows(small_store):
    mask = small_store.array("flag").astype(bool)
    np.testing.assert_array_equal(
        np.array(small_store[mask, "vol"]), small_store.array("vol")[mask]
    )
    rec = np.asarray(small_store[mask])
    np.testing.assert_array_equal(rec["vol"], small_store.array("vol")[mask])


# ---- multi-record and multi-file sources ------------------------------------


def test_multirecord_column_and_record(tmp_path):
    cols = {"a": np.arange(15, dtype=np.int64), "b": np.linspace(0, 1, 15)}
    store = _store(tmp_path, cols, records=5)
    np.testing.assert_array_equal(np.array(store["a"]), cols["a"])
    rec = np.array(store)
    np.testing.assert_array_equal(rec["b"], cols["b"])


def test_dataset_record_array(tmp_path):
    blocks = [
        {"a": np.arange(3, dtype=np.int64), "b": np.arange(3, dtype=np.float64)},
        {"a": np.arange(3, 7, dtype=np.int64), "b": np.arange(3, 7, dtype=np.float64)},
    ]
    ds = _dataset(tmp_path, blocks)
    try:
        rec = np.array(ds)
        np.testing.assert_array_equal(rec["a"], np.arange(7, dtype=np.int64))
        np.testing.assert_array_equal(np.asarray(ds["b"]), np.arange(7, dtype=np.float64))
    finally:
        ds.close()


# ---- empty and closed -------------------------------------------------------


def test_empty_store(tmp_path):
    store = _store(tmp_path, {"a": np.empty(0, np.int64), "b": np.empty(0, np.float64)})
    assert np.array(store["a"]).shape == (0,)
    rec = np.array(store)
    assert rec.shape == (0,)
    assert rec.dtype.names == ("a", "b")


def test_empty_column_projection(small_store):
    # A no-column projection (drop-all, ds[rows, []]) is a valid field-less record
    # array, not an error; the record count follows the row selection.
    empty = np.asarray(small_store.drop(*small_store.columns))
    assert empty.dtype.names == ()
    assert empty.shape == (small_store.n_rows,)
    assert np.asarray(small_store[0:3, []]).shape == (3,)
    # the same shared path backs .recarray(), so it agrees
    assert small_store.drop(*small_store.columns).recarray().shape == (small_store.n_rows,)


def test_closed_store_raises(tmp_path):
    store = _store(tmp_path, {"a": np.arange(4, dtype=np.int64)})
    store.close()
    with pytest.raises(ValueError):
        np.array(store)


# ---- consumption by NumPy functions ----------------------------------------


def test_consumed_by_numpy_functions(small_store):
    assert np.mean(small_store["vol"]) == pytest.approx(small_store.array("vol").mean())
    np.testing.assert_array_equal(np.sort(small_store["id"]), np.sort(small_store.array("id")))
    combined = np.concatenate([small_store["id"], small_store["id"]])
    assert combined.shape == (2 * small_store.n_rows,)
