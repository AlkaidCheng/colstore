"""Tests for the module-level API: open, create, recreate, update, store.

These wrappers around ColStore/ColWriter are the recommended public surface.
The class-level constructors (`ColStore(path)`, `ColWriter(path, mode)`) keep
working too, but module-level functions are the documented entry points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import colstore
from colstore import ColStore, ColWriter

# ---- open ------------------------------------------------------------------


def test_open_returns_colstore(tmp_path):
    """``colstore.open`` returns a ColStore equivalent to direct construction."""
    path = tmp_path / "x.cstore"
    colstore.store({"a": np.arange(10, dtype=np.float32)}, path, show_progress=False).close()
    with colstore.open(path) as ds:
        assert isinstance(ds, ColStore)
        assert ds.n_rows == 10
        assert np.array_equal(ds[:, "a"].to_array(), np.arange(10, dtype=np.float32))


def test_open_missing_file_raises(tmp_path):
    """Opening a missing file raises FileNotFoundError (or a clear FormatError)."""
    with pytest.raises((FileNotFoundError, OSError)):
        colstore.open(tmp_path / "nope.cstore")


# ---- create / recreate -----------------------------------------------------


def test_create_returns_writer(tmp_path):
    """``colstore.create`` returns a ColWriter for a new file."""
    path = tmp_path / "c.cstore"
    with colstore.create(path) as w:
        assert isinstance(w, ColWriter)
        assert w.mode == "create"
        w.write({"a": np.arange(5, dtype=np.int32)})
    with colstore.open(path) as ds:
        assert ds.n_rows == 5


def test_create_fails_if_exists(tmp_path):
    """``create`` is non-destructive: refuses to overwrite an existing file."""
    path = tmp_path / "exists.cstore"
    path.write_bytes(b"")
    with pytest.raises(FileExistsError):
        colstore.create(path)


def test_recreate_truncates_existing(tmp_path):
    """``recreate`` happily replaces an existing file."""
    path = tmp_path / "r.cstore"
    colstore.store({"a": np.arange(100, dtype=np.float32)}, path, show_progress=False).close()
    with colstore.recreate(path) as w:
        w.write({"b": np.arange(3, dtype=np.int64)})
    with colstore.open(path) as ds:
        assert ds.columns == ["b"]
        assert ds.n_rows == 3


def test_recreate_works_when_file_does_not_exist(tmp_path):
    """``recreate`` doesn't require the file to exist."""
    path = tmp_path / "new.cstore"
    with colstore.recreate(path) as w:
        w.write({"x": np.array([1.0], dtype=np.float64)})
    assert path.exists()


# ---- update ----------------------------------------------------------------


def test_update_appends_records(tmp_path):
    """``update`` appends to an existing file; reader sees all records."""
    path = tmp_path / "u.cstore"
    with colstore.create(path) as w:
        w.write({"a": np.array([1, 2, 3], dtype=np.int32)})
    with colstore.update(path) as w:
        assert w.n_records == 1
        assert w.committed_rows == 3
        w.write({"a": np.array([4, 5], dtype=np.int32)})
        w.write({"a": np.array([6, 7, 8, 9], dtype=np.int32)})
    with colstore.open(path) as ds:
        assert ds.n_rows == 9
        assert np.array_equal(
            ds[:, "a"].to_array(), np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
        )


def test_update_fails_if_missing(tmp_path):
    """``update`` requires an existing file."""
    with pytest.raises(FileNotFoundError):
        colstore.update(tmp_path / "missing.cstore")


# ---- store: type dispatch --------------------------------------------------


def test_store_accepts_dict(tmp_path):
    """dict[str, ndarray] is the primary input form."""
    path = tmp_path / "d.cstore"
    columns = {"a": np.arange(10, dtype=np.float32), "b": np.arange(10, dtype=np.int64)}
    ds = colstore.store(columns, path, show_progress=False)
    try:
        assert ds.columns == ["a", "b"]
        assert ds.n_rows == 10
    finally:
        ds.close()


def test_store_accepts_structured_ndarray(tmp_path):
    """A structured ndarray dispatches to one column per field."""
    dtype = np.dtype([("x", "<f8"), ("y", "<i4")])
    records = np.array([(1.0, 10), (2.0, 20), (3.0, 30)], dtype=dtype)
    ds = colstore.store(records, tmp_path / "s.cstore", show_progress=False)
    try:
        assert ds.columns == ["x", "y"]
        assert np.array_equal(ds[:, "x"].to_array(), np.array([1.0, 2.0, 3.0]))
    finally:
        ds.close()


def test_store_accepts_dataframe(tmp_path):
    """A pandas DataFrame dispatches via duck-typed detection."""
    frame = pd.DataFrame({"a": np.arange(5, dtype=np.float32), "b": np.arange(5, dtype=np.int64)})
    ds = colstore.store(frame, tmp_path / "df.cstore", show_progress=False)
    try:
        assert ds.columns == ["a", "b"]
        assert ds.n_rows == 5
    finally:
        ds.close()


def test_store_rejects_plain_ndarray(tmp_path):
    """A plain (non-structured) ndarray is ambiguous; reject with a clear message."""
    with pytest.raises(TypeError, match="plain ndarray"):
        colstore.store(np.arange(10), tmp_path / "p.cstore", show_progress=False)


def test_store_rejects_list(tmp_path):
    """A list is not in scope; reject."""
    with pytest.raises(TypeError, match="does not know how to handle"):
        colstore.store([1, 2, 3], tmp_path / "l.cstore", show_progress=False)


def test_store_rejects_object_dtype_in_dataframe(tmp_path):
    """Object-backed pandas columns are caught with the column-aware error."""
    frame = pd.DataFrame({"a": ["x", "y", "z"]})  # dtype object
    with pytest.raises(TypeError, match="object array"):
        colstore.store(frame, tmp_path / "obj.cstore", show_progress=False)


# ---- store: mode -----------------------------------------------------------


def test_store_default_mode_is_create(tmp_path):
    """Default mode is 'create'; fails when the file already exists."""
    path = tmp_path / "m.cstore"
    colstore.store({"a": np.arange(3)}, path, show_progress=False).close()
    with pytest.raises(FileExistsError):
        colstore.store({"a": np.arange(3)}, path, show_progress=False)


def test_store_recreate_mode_overwrites(tmp_path):
    """mode='recreate' truncates."""
    path = tmp_path / "m.cstore"
    colstore.store({"a": np.arange(100)}, path, show_progress=False).close()
    ds = colstore.store({"b": np.arange(5)}, path, mode="recreate", show_progress=False)
    try:
        assert ds.columns == ["b"]
        assert ds.n_rows == 5
    finally:
        ds.close()


def test_store_invalid_mode_raises(tmp_path):
    """Only 'create' and 'recreate' are valid for store()."""
    with pytest.raises(ValueError, match="mode"):
        colstore.store({"a": np.arange(3)}, tmp_path / "x.cstore", mode="update")
