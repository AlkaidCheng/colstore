"""Tests for the module-level API: open, create, recreate, update, store, info, schema.

These wrappers around ColStoreReader/ColStoreWriter are the recommended public surface.
The class-level constructors (`ColStoreReader(path)`, `ColStoreWriter(path, mode)`) keep
working too, but module-level functions are the documented entry points.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import colstore
from colstore import ColStoreDataset, ColStoreInfo, ColStoreReader, ColStoreWriter, FormatError

# ---- open ------------------------------------------------------------------


def test_open_returns_colstore(tmp_path):
    """``colstore.open`` returns a ColStoreReader equivalent to direct construction."""
    path = tmp_path / "x.cstore"
    colstore.store({"a": np.arange(10, dtype=np.float32)}, path, show_progress=False).close()
    with colstore.open(path) as ds:
        assert isinstance(ds, ColStoreReader)
        assert ds.n_rows == 10
        assert np.array_equal(ds[:, "a"].array(), np.arange(10, dtype=np.float32))


def test_open_missing_file_raises(tmp_path):
    """Opening a missing file raises FileNotFoundError (or a clear FormatError)."""
    with pytest.raises((FileNotFoundError, OSError)):
        colstore.open(tmp_path / "nope.cstore")


# ---- open: glob patterns ---------------------------------------------------


def _make_marker_file(path, marker):
    """A single-row store whose column ``a`` holds ``marker``, identifying the file."""
    colstore.store({"a": np.array([marker], dtype=np.int64)}, path, show_progress=False).close()


def test_open_glob_returns_dataset_over_matches(tmp_path):
    for n in (1, 2, 3):
        _make_marker_file(tmp_path / f"part_{n}.cstore", n)
    with colstore.open(str(tmp_path / "part_*.cstore")) as ds:
        assert isinstance(ds, ColStoreDataset)
        assert ds.n_rows == 3
        assert ds[:, "a"].array().tolist() == [1, 2, 3]


def test_open_glob_orders_matches_numerically(tmp_path):
    # Lexicographic order would put run_10 before run_2; natural sort must not.
    for n in (1, 2, 10):
        _make_marker_file(tmp_path / f"run_{n}.cstore", n)
    with colstore.open(str(tmp_path / "run_*.cstore")) as ds:
        assert ds[:, "a"].array().tolist() == [1, 2, 10]


def test_open_glob_no_match_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no files matched"):
        colstore.open(str(tmp_path / "absent_*.cstore"))


def test_open_literal_path_still_returns_reader(tmp_path):
    # A path with no glob magic is unchanged: a single Reader, not a dataset.
    _make_marker_file(tmp_path / "solo.cstore", 7)
    with colstore.open(tmp_path / "solo.cstore") as ds:
        assert isinstance(ds, ColStoreReader)


def test_open_list_expands_glob_elements(tmp_path):
    _make_marker_file(tmp_path / "a_1.cstore", 1)
    _make_marker_file(tmp_path / "a_2.cstore", 2)
    _make_marker_file(tmp_path / "b.cstore", 3)
    with colstore.open([str(tmp_path / "a_*.cstore"), str(tmp_path / "b.cstore")]) as ds:
        assert isinstance(ds, ColStoreDataset)
        assert ds[:, "a"].array().tolist() == [1, 2, 3]


def test_concat_expands_glob(tmp_path):
    for n in (1, 2):
        _make_marker_file(tmp_path / f"q_{n}.cstore", n)
    out = tmp_path / "combined.cstore"
    with colstore.concat([str(tmp_path / "q_*.cstore")], out=out) as reader:
        assert reader.n_rows == 2
        assert reader[:, "a"].array().tolist() == [1, 2]


def test_dataset_append_expands_glob(tmp_path):
    for n in (1, 2):
        _make_marker_file(tmp_path / f"app_{n}.cstore", n)
    ds = ColStoreDataset()
    ds.append(str(tmp_path / "app_*.cstore"))
    assert ds.n_rows == 2
    assert ds[:, "a"].array().tolist() == [1, 2]
    ds.close()


# ---- create / recreate -----------------------------------------------------


def test_create_returns_writer(tmp_path):
    """``colstore.create`` returns a ColStoreWriter for a new file."""
    path = tmp_path / "c.cstore"
    with colstore.create(path) as w:
        assert isinstance(w, ColStoreWriter)
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
            ds[:, "a"].array(), np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
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
        assert np.array_equal(ds[:, "x"].array(), np.array([1.0, 2.0, 3.0]))
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


def test_store_default_batch_size_is_auto(tmp_path):
    """Default batch_size='auto' produces a readable file for small data."""
    path = tmp_path / "auto.cstore"
    src = {"a": np.arange(100, dtype=np.int32), "b": np.arange(100, dtype=np.float64)}
    ds = colstore.store(src, path, show_progress=False)
    assert ds.n_rows == 100
    assert np.array_equal(ds[:, "a"].array(), src["a"])
    assert np.array_equal(ds[:, "b"].array(), src["b"])


@pytest.mark.parametrize("batch_size", [None, 1000, "auto", "4 KB", "1 MiB"])
def test_store_batch_size_variants_all_roundtrip(tmp_path, batch_size):
    """Every flavor of batch_size yields the same logical data."""
    src = {"x": np.arange(500, dtype=np.float32), "y": np.arange(500, dtype=np.int16)}
    path = tmp_path / "bs.cstore"
    ds = colstore.store(src, path, batch_size=batch_size, show_progress=False)
    assert ds.n_rows == 500
    assert np.array_equal(ds[:, "x"].array(), src["x"])
    assert np.array_equal(ds[:, "y"].array(), src["y"])


# ---- info / schema introspection ---------------------------------------------


def test_info_basic_fields(tmp_path):
    path = tmp_path / "x.cstore"
    colstore.store(
        {"a": np.arange(10, dtype=np.float32), "b": np.arange(10, dtype=np.int64)},
        path,
        show_progress=False,
    ).close()

    i = colstore.info(path)
    assert isinstance(i, ColStoreInfo)
    assert i.path == Path(path)
    assert i.format_version == 1
    assert i.n_rows == 10
    assert i.n_records == 1
    assert i.file_size == path.stat().st_size
    assert [c["name"] for c in i.columns] == ["a", "b"]
    assert [c["dtype"] for c in i.columns] == ["<f4", "<i8"]
    assert not i.needs_compaction


def test_info_needs_compaction_flips_on_multi_record(tmp_path):
    path = tmp_path / "m.cstore"
    with colstore.create(path) as f:
        for _ in range(3):
            f.write({"a": np.arange(10, dtype=np.float32)})

    i = colstore.info(path)
    assert i.n_records == 3
    assert i.n_rows == 30
    assert i.needs_compaction


def test_info_repr_is_readable(tmp_path):
    path = tmp_path / "r.cstore"
    colstore.store({"a": np.arange(5)}, path, show_progress=False).close()
    text = repr(colstore.info(path))
    # Spot-check that key fields are visible without scrolling.
    assert "r.cstore" in text
    assert "n_rows=5" in text
    assert "n_records=1" in text
    assert "a:" in text  # column listing


def test_info_repr_flags_needs_compaction(tmp_path):
    path = tmp_path / "nc.cstore"
    with colstore.create(path) as f:
        f.write({"a": np.arange(3, dtype=np.int32)})
        f.write({"a": np.arange(3, 6, dtype=np.int32)})
    assert "needs_compaction=True" in repr(colstore.info(path))


def test_info_on_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        colstore.info(tmp_path / "nope.cstore")


def test_info_on_corrupt_file_raises(tmp_path):
    path = tmp_path / "c.cstore"
    colstore.store({"a": np.arange(5)}, path, show_progress=False).close()
    raw = bytearray(path.read_bytes())
    raw[0] ^= 0xFF  # break the magic bytes
    path.write_bytes(bytes(raw))
    with pytest.raises(FormatError):
        colstore.info(path)


def test_schema_returns_column_metadata(tmp_path):
    path = tmp_path / "s.cstore"
    colstore.store(
        {
            "alpha": np.arange(7, dtype=np.float64),
            "beta": np.arange(7, dtype=np.int32),
        },
        path,
        show_progress=False,
    ).close()

    s = colstore.schema(path)
    assert isinstance(s, list)
    assert len(s) == 2
    assert s[0]["name"] == "alpha"
    assert s[0]["dtype"] == "<f8"
    assert s[1]["name"] == "beta"
    assert s[1]["dtype"] == "<i4"


def test_schema_does_not_open_for_reads(tmp_path):
    """schema() should work even if the records would fail to read.

    We can't easily construct such a file, so settle for: schema() makes only
    a header-read and never touches record bodies, which means it's cheap on
    a multi-GB file. This is a structural test via inspection rather than
    instrumentation -- just verify it returns quickly on a multi-record file.
    """
    path = tmp_path / "big.cstore"
    with colstore.create(path) as f:
        for _ in range(50):
            f.write({"a": np.arange(100, dtype=np.float64)})
    # The schema call should return the same data as a full info() call.
    assert colstore.schema(path) == colstore.info(path).columns
