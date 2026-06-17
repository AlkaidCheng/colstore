"""Tests for :class:`colstore.dataset.ColStoreDataset`, the colstore data handle.

Multi-file reads are checked against a NumPy oracle built by concatenating the
per-file source arrays, so the dataset's decomposition is validated against the
single, obvious definition of "the files end to end". Fixtures use single-record
stores, whose reads run without the compiled gather extension.
"""

import numpy as np
import pytest

import colstore
from colstore import ColStoreDataset, ColStoreReader


def _build_files(tmp_path, sizes):
    """Write one single-record file per size; return paths and the oracle columns."""
    paths = []
    x_blocks = []
    y_blocks = []
    base = 0
    for index, n in enumerate(sizes):
        x = np.arange(base, base + n, dtype=np.int64)
        y = (x * 1.5).astype(np.float64)
        x_blocks.append(x)
        y_blocks.append(y)
        path = tmp_path / f"f{index}.cstore"
        colstore.store({"x": x, "y": y}, path, show_progress=False).close()
        paths.append(path)
        base += n
    return paths, np.concatenate(x_blocks), np.concatenate(y_blocks)


def _store_one(tmp_path, name, data):
    path = tmp_path / name
    colstore.store(data, path, show_progress=False).close()
    return path


# ---- open() dispatch ----------------------------------------------------


def test_open_scalar_returns_reader(tmp_path):
    paths, _, _ = _build_files(tmp_path, [5])
    obj = colstore.open(paths[0])
    assert isinstance(obj, ColStoreReader)
    assert not isinstance(obj, ColStoreDataset)
    obj.close()


def test_open_list_returns_dataset(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [4, 6])
    ds = colstore.open(paths)
    assert isinstance(ds, ColStoreDataset)
    assert ds.n_rows == len(ox)
    assert ds.shape == (len(ox), 2)
    assert ds.columns == ["x", "y"]
    assert {k: str(v) for k, v in ds.dtypes.items()} == {"x": "int64", "y": "float64"}
    assert len(ds.path) == 2
    assert ds.needs_compaction is True
    assert "ColStoreDataset" in repr(ds)
    ds.close()


def test_open_single_element_list_is_a_one_file_dataset(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [5])
    ds = colstore.open(paths)  # a list, even of one path, is a dataset now
    assert isinstance(ds, ColStoreDataset)
    assert ds.n_rows == 5
    assert ds.needs_compaction is False
    np.testing.assert_array_equal(ds["x"].array(), ox)
    ds.close()


def test_open_empty_list_is_empty_dataset(tmp_path):
    ds = colstore.open([])
    assert isinstance(ds, ColStoreDataset)
    assert ds.n_rows == 0
    assert ds.columns == []
    assert ds.path == ()
    assert ds.needs_compaction is False
    ds.close()


# ---- Constructor forms --------------------------------------------------


def test_constructor_empty():
    ds = ColStoreDataset()
    assert ds.n_rows == 0
    assert ds.columns == []
    assert ds.shape == (0, 0)
    ds.close()


def test_constructor_scalar_path_owns(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [5])
    ds = ColStoreDataset(paths[0])
    assert ds.n_rows == 5
    np.testing.assert_array_equal(ds["x"].array(), ox)
    child = ds._children[0]
    ds.close()
    assert child._closed is True  # opened from a path -> owned -> closed


def test_constructor_list_path_owns(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [5])
    ds = ColStoreDataset([paths[0]])
    np.testing.assert_array_equal(ds["x"].array(), ox)
    child = ds._children[0]
    ds.close()
    assert child._closed is True


def test_constructor_reader_borrows(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [5])
    reader = colstore.open(paths[0])
    ds = ColStoreDataset(reader)
    np.testing.assert_array_equal(ds["x"].array(), ox)
    ds.close()
    assert reader._closed is False  # handed in open -> borrowed -> left open
    reader.close()


def test_constructor_list_reader_borrows(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [5])
    reader = colstore.open(paths[0])
    ds = ColStoreDataset([reader])
    np.testing.assert_array_equal(ds["x"].array(), ox)
    ds.close()
    assert reader._closed is False
    reader.close()


def test_constructor_mixed_paths_and_readers(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [4, 6])
    borrowed = colstore.open(paths[1])
    ds = ColStoreDataset([paths[0], borrowed])  # path owned, reader borrowed
    np.testing.assert_array_equal(ds["x"].array(), ox)
    assert ds._owned == [True, False]
    opened_child = ds._children[0]
    ds.close()
    assert opened_child._closed is True
    assert borrowed._closed is False
    borrowed.close()


def test_constructor_flattens_dataset(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [2, 3, 4])
    inner = colstore.open(paths[:2])  # a 2-file dataset
    outer = ColStoreDataset([inner, paths[2]])
    assert len(outer._children) == 3  # flattened, not nested
    assert all(isinstance(child, ColStoreReader) for child in outer._children)
    np.testing.assert_array_equal(outer["x"].array(), ox)
    outer.close()
    inner.close()


def test_constructor_rejects_bad_type():
    with pytest.raises(TypeError, match="accepts a path"):
        ColStoreDataset(5)


def test_mapping_protocol(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [4, 6])
    with colstore.open(paths) as ds:
        assert len(ds) == len(ox)
        assert list(ds) == ["x", "y"]
        assert "x" in ds
        assert "missing" not in ds


# ---- append() and |= ----------------------------------------------------


def test_append_grows_and_chains(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [3, 4, 5])
    ds = ColStoreDataset()
    returned = ds.append(paths[0]).append(paths[1])
    assert returned is ds  # chainable
    assert ds.n_rows == 7
    ds.append([paths[2]])
    assert ds.n_rows == 12
    np.testing.assert_array_equal(ds["x"].array(), ox)
    ds.close()


def test_append_path_owns_reader_borrows(tmp_path):
    paths, _, _ = _build_files(tmp_path, [3, 4])
    borrowed = colstore.open(paths[1])
    ds = ColStoreDataset()
    ds.append(paths[0])  # owned
    ds.append(borrowed)  # borrowed
    owned_child = ds._children[0]
    ds.close()
    assert owned_child._closed is True
    assert borrowed._closed is False
    borrowed.close()


def test_append_establishes_then_validates_schema(tmp_path):
    good, _, _ = _build_files(tmp_path, [4])
    narrow = _store_one(
        tmp_path,
        "narrow.cstore",
        {"x": np.arange(3, dtype=np.int32), "y": np.arange(3, dtype=np.float64)},
    )
    ds = ColStoreDataset()
    ds.append(good[0])  # first child establishes the schema
    assert ds.columns == ["x", "y"]
    with pytest.raises(ValueError, match="Schema mismatch"):
        ds.append(narrow)
    ds.close()


def test_append_schema_mismatch_is_atomic(tmp_path):
    paths, _, _ = _build_files(tmp_path, [4])
    narrow = _store_one(
        tmp_path,
        "narrow.cstore",
        {"x": np.arange(3, dtype=np.int32), "y": np.arange(3, dtype=np.float64)},
    )
    ds = ColStoreDataset(paths[0])
    before = ds.n_rows
    with pytest.raises(ValueError, match="Schema mismatch"):
        ds.append(narrow)
    assert ds.n_rows == before  # the failed append changed nothing
    assert len(ds._children) == 1
    ds.close()


def test_append_open_error_leaves_unchanged(tmp_path):
    paths, _, _ = _build_files(tmp_path, [4])
    ds = ColStoreDataset(paths[0])
    with pytest.raises(FileNotFoundError):
        ds.append(tmp_path / "does_not_exist.cstore")
    assert ds.n_rows == 4
    assert len(ds._children) == 1
    ds.close()


def test_ior_appends_borrowed(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [4, 6])
    left = colstore.open(paths[0])
    right = colstore.open(paths[1])
    left_ds = ColStoreDataset(left)
    left_ds |= right
    np.testing.assert_array_equal(left_ds["x"].array(), ox)
    left_ds.close()
    assert left._closed is False
    assert right._closed is False
    left.close()
    right.close()


def test_ior_rejects_path(tmp_path):
    paths, _, _ = _build_files(tmp_path, [4])
    ds = ColStoreDataset(paths[0])
    with pytest.raises(TypeError, match="Unsupported operand for"):
        ds |= "some/path.cstore"
    ds.close()


# ---- Empty dataset behavior --------------------------------------------


def test_empty_dataset_whole_read_is_empty(tmp_path):
    ds = ColStoreDataset()
    assert ds[:].dict() == {}
    ds.close()


def test_empty_dataset_unknown_column_raises():
    ds = ColStoreDataset()
    with pytest.raises(KeyError, match="Unknown column"):
        ds["x"].array()
    ds.close()


# ---- Contiguous selectors against the oracle ---------------------------


def test_whole_store_matches_oracle(tmp_path):
    paths, ox, oy = _build_files(tmp_path, [4, 0, 6])
    with colstore.open(paths) as ds:
        np.testing.assert_array_equal(ds["x"].array(), ox)
        whole = ds[:].dict()
        np.testing.assert_array_equal(whole["x"], ox)
        np.testing.assert_array_equal(whole["y"], oy)


@pytest.mark.parametrize("g", [0, 3, 4, 5, 9])
def test_scalar_row_matches_oracle(tmp_path, g):
    paths, ox, oy = _build_files(tmp_path, [4, 0, 6])
    with colstore.open(paths) as ds:
        np.testing.assert_array_equal(ds[g, "x"].array(), ox[g : g + 1])
        np.testing.assert_array_equal(ds[g, "y"].array(), oy[g : g + 1])


@pytest.mark.parametrize(
    "sl",
    [
        slice(None),
        slice(1, 8),
        slice(0, 4),
        slice(4, 10),
        slice(3, 3),
        slice(2, 9, 2),
        slice(1, 10, 3),
        slice(None, None, 2),
        slice(None, None, 4),
        slice(5, 100),
    ],
)
def test_slice_matches_oracle(tmp_path, sl):
    paths, ox, oy = _build_files(tmp_path, [4, 0, 6])
    with colstore.open(paths) as ds:
        np.testing.assert_array_equal(ds[sl, "x"].array(), ox[sl])
        both = ds[sl, ["x", "y"]].dict()
        np.testing.assert_array_equal(both["x"], ox[sl])
        np.testing.assert_array_equal(both["y"], oy[sl])


def test_empty_slice_returns_empty_typed_array(tmp_path):
    paths, _, _ = _build_files(tmp_path, [4, 6])
    with colstore.open(paths) as ds:
        out = ds[5:5, "x"].array()
        assert out.shape == (0,)
        assert out.dtype == np.dtype("int64")


def test_zero_row_leading_and_trailing_files(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [0, 5, 0, 3, 0])
    with colstore.open(paths) as ds:
        assert ds.n_rows == len(ox)
        np.testing.assert_array_equal(ds["x"].array(), ox)
        np.testing.assert_array_equal(ds[1:7, "x"].array(), ox[1:7])


# ---- Single-file dataset matches the bare reader -----------------------


def test_single_file_dataset_matches_reader(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [8])
    ds = colstore.open(paths)
    # A one-file dataset short-circuits to its child, so it supports everything
    # the reader does, including fancy and boolean selection and negative steps.
    np.testing.assert_array_equal(ds[[1, 3, 5], "x"].array(), ox[[1, 3, 5]])
    mask = np.zeros(8, dtype=bool)
    mask[::2] = True
    np.testing.assert_array_equal(ds[mask, "x"].array(), ox[mask])
    np.testing.assert_array_equal(ds[::-1, "x"].array(), ox[::-1])
    np.testing.assert_array_equal(ds[2:6, "x"].array(copy=False), ox[2:6])
    ds.close()


# ---- Whole-store materializers -----------------------------------------


def test_recarray_and_frame(tmp_path):
    paths, ox, oy = _build_files(tmp_path, [4, 6])
    with colstore.open(paths) as ds:
        rec = ds.recarray()
        np.testing.assert_array_equal(rec["x"], ox)
        np.testing.assert_array_equal(rec["y"], oy)
        frame = ds.frame()
        assert list(frame.columns) == ["x", "y"]
        np.testing.assert_array_equal(frame["x"].to_numpy(), ox)
        np.testing.assert_array_equal(frame["y"].to_numpy(), oy)


# ---- Schema validation -------------------------------------------------


def test_schema_mismatch_column_names(tmp_path):
    good, _, _ = _build_files(tmp_path, [4])
    bad = _store_one(
        tmp_path,
        "bad.cstore",
        {"x": np.arange(3, dtype=np.int64), "z": np.arange(3, dtype=np.int64)},
    )
    with pytest.raises(ValueError, match="Schema mismatch"):
        colstore.open([good[0], bad])


def test_schema_mismatch_column_order(tmp_path):
    good, _, _ = _build_files(tmp_path, [4])
    reordered = _store_one(
        tmp_path,
        "reordered.cstore",
        {"y": np.arange(3, dtype=np.float64), "x": np.arange(3, dtype=np.int64)},
    )
    with pytest.raises(ValueError, match="Schema mismatch"):
        colstore.open([good[0], reordered])


def test_schema_mismatch_dtype(tmp_path):
    good, _, _ = _build_files(tmp_path, [4])
    narrow = _store_one(
        tmp_path,
        "narrow.cstore",
        {"x": np.arange(3, dtype=np.int32), "y": np.arange(3, dtype=np.float64)},
    )
    with pytest.raises(ValueError, match="Schema mismatch"):
        colstore.open([good[0], narrow])


# ---- The | operator ----------------------------------------------------


def test_or_operator_equals_open(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [4, 0, 6])
    readers = [colstore.open(p) for p in paths]
    combined = readers[0] | readers[1] | readers[2]
    assert isinstance(combined, ColStoreDataset)
    np.testing.assert_array_equal(combined["x"].array(), ox)
    combined.close()
    for reader in readers:
        reader.close()


def test_or_operator_flattens(tmp_path):
    paths, _, _ = _build_files(tmp_path, [2, 3, 4])
    readers = [colstore.open(p) for p in paths]
    combined = (readers[0] | readers[1]) | readers[2]
    assert len(combined._children) == 3
    assert all(isinstance(child, ColStoreReader) for child in combined._children)
    combined.close()
    for reader in readers:
        reader.close()


def test_or_operator_borrows_operands(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [4, 6])
    left = colstore.open(paths[0])
    right = colstore.open(paths[1])
    combined = left | right
    np.testing.assert_array_equal(combined["x"].array(), ox)
    combined.close()
    assert left._closed is False
    assert right._closed is False
    np.testing.assert_array_equal(left["x"].array(), ox[:4])
    left.close()
    right.close()


def test_or_operator_validates_schema(tmp_path):
    good, _, _ = _build_files(tmp_path, [4])
    bad = _store_one(
        tmp_path,
        "bad.cstore",
        {"x": np.arange(3, dtype=np.int32), "y": np.arange(3, dtype=np.float64)},
    )
    left = colstore.open(good[0])
    right = colstore.open(bad)
    with pytest.raises(ValueError, match="Schema mismatch"):
        _ = left | right
    left.close()
    right.close()


def test_or_operator_rejects_non_reader(tmp_path):
    paths, _, _ = _build_files(tmp_path, [4])
    reader = colstore.open(paths[0])
    with pytest.raises(TypeError, match="Unsupported operand"):
        _ = reader | 5
    with pytest.raises(TypeError, match="Unsupported operand"):
        _ = 5 | reader
    reader.close()


# ---- Ownership and lifecycle -------------------------------------------


def test_open_paths_owns_and_closes_children(tmp_path):
    paths, _, _ = _build_files(tmp_path, [4, 6])
    ds = colstore.open(paths)
    assert isinstance(ds, ColStoreDataset)
    children = list(ds._children)
    ds.close()
    assert all(child._closed for child in children)
    with pytest.raises(ValueError, match="closed"):
        ds["x"].array()


def test_constructor_rejects_empty_then_grows():
    ds = ColStoreDataset()
    assert ds.n_rows == 0
    ds.close()


# ---- Cross-file fancy / boolean / negative-step vs the oracle ----------


@pytest.mark.parametrize(
    "idx",
    [
        [1, 3, 5],  # ascending across the boundary
        [9, 0, 4, 4, 1],  # interleaved files, out of order, with a duplicate
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],  # full reversal
        [3, 4],  # straddles the boundary
        [],  # empty selection
    ],
)
def test_multifile_fancy_matches_oracle(tmp_path, idx):
    paths, ox, oy = _build_files(tmp_path, [4, 0, 6])
    index = np.array(idx, dtype=np.int64)
    with colstore.open(paths) as ds:
        np.testing.assert_array_equal(ds[index, "x"].array(), ox[index])
        both = ds[index, ["x", "y"]].dict()
        np.testing.assert_array_equal(both["x"], ox[index])
        np.testing.assert_array_equal(both["y"], oy[index])


def test_multifile_fancy_preserves_order_and_position(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [5, 5])
    # Indices deliberately jump back and forth across the boundary; the result
    # must follow the requested order, not file or sorted order.
    index = np.array([7, 0, 9, 4, 5, 1], dtype=np.int64)
    with colstore.open(paths) as ds:
        np.testing.assert_array_equal(ds[index, "x"].array(), ox[index])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_multifile_boolean_mask_matches_oracle(tmp_path, seed):
    paths, ox, oy = _build_files(tmp_path, [4, 0, 6])
    rng = np.random.default_rng(seed)
    mask = rng.random(len(ox)) < 0.5
    with colstore.open(paths) as ds:
        np.testing.assert_array_equal(ds[mask, "x"].array(), ox[mask])
        both = ds[mask, ["x", "y"]].dict()
        np.testing.assert_array_equal(both["x"], ox[mask])
        np.testing.assert_array_equal(both["y"], oy[mask])


def test_multifile_boolean_mask_all_and_none(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [4, 6])
    with colstore.open(paths) as ds:
        all_true = np.ones(ds.n_rows, dtype=bool)
        np.testing.assert_array_equal(ds[all_true, "x"].array(), ox)
        all_false = np.zeros(ds.n_rows, dtype=bool)
        out = ds[all_false, "x"].array()
        assert out.shape == (0,)
        assert out.dtype == np.dtype("int64")


@pytest.mark.parametrize("sl", [slice(None, None, -1), slice(8, 1, -2), slice(None, None, -3)])
def test_multifile_negative_step_matches_oracle(tmp_path, sl):
    paths, ox, oy = _build_files(tmp_path, [4, 0, 6])
    with colstore.open(paths) as ds:
        np.testing.assert_array_equal(ds[sl, "x"].array(), ox[sl])
        both = ds[sl, ["x", "y"]].dict()
        np.testing.assert_array_equal(both["x"], ox[sl])
        np.testing.assert_array_equal(both["y"], oy[sl])


# ---- Zero-copy seam rules ----------------------------------------------


def test_view_within_single_file(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [6, 4])
    with colstore.open(paths) as ds:
        np.testing.assert_array_equal(ds[1:4, "x"].array(copy=False), ox[1:4])
        np.testing.assert_array_equal(ds[2, "x"].array(copy=False), ox[2:3])


def test_view_whole_multifile_raises(tmp_path):
    paths, _, _ = _build_files(tmp_path, [4, 6])
    with colstore.open(paths) as ds, pytest.raises(ValueError, match="not contiguous in memory"):
        ds["x"].array(copy=False)


def test_view_cross_file_slice_raises(tmp_path):
    paths, _, _ = _build_files(tmp_path, [4, 6])
    with colstore.open(paths) as ds, pytest.raises(ValueError, match="spans multiple files"):
        ds[2:8, "x"].array(copy=False)


def test_view_fancy_raises_value_error(tmp_path):
    paths, _, _ = _build_files(tmp_path, [4, 6])
    with colstore.open(paths) as ds, pytest.raises(ValueError, match="fancy and boolean"):
        ds[[1, 3], "x"].array(copy=False)


def test_view_reversed_slice_raises(tmp_path):
    paths, _, _ = _build_files(tmp_path, [4, 6])
    with colstore.open(paths) as ds, pytest.raises(ValueError, match="reversed"):
        ds[::-1, "x"].array(copy=False)


# ---- Column errors -----------------------------------------------------


def test_unknown_column_raises_key_error(tmp_path):
    paths, _, _ = _build_files(tmp_path, [4, 6])
    with colstore.open(paths) as ds, pytest.raises(KeyError, match="Unknown column"):
        ds["nope"].array()
