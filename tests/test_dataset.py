"""Tests for :class:`colstore.dataset.ColStoreDataset`, the colstore data handle.

Multi-file reads are checked against a NumPy oracle built by concatenating the
per-file source arrays, so the dataset's decomposition is validated against the
single, obvious definition of "the files end to end". Fixtures use single-record
stores, whose reads run without the compiled gather extension.
"""

import numpy as np
import pytest

import colstore
from colstore import ColStoreDataset, ColStoreReader, col, config, kernels, testing


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
    assert len(ds.paths) == 2
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
    assert ds.paths == ()
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


def test_empty_dataset_recarray_and_frame():
    ds = ColStoreDataset()
    rec = ds.recarray()
    assert rec.shape == (0,)
    assert rec.dtype.names == ()
    frame = ds.frame()
    assert frame.shape == (0, 0)
    ds.close()


# ---- edit() over a dataset ---------------------------------------------


def test_edit_returns_frame_over_all_files(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [4, 6])
    with colstore.open(paths) as ds:
        frame = ds.edit()
        assert frame.n_rows == len(ox)
        np.testing.assert_array_equal(frame["x"].compute(), ox)


def test_edit_passthrough_concatenates_files(tmp_path):
    paths, ox, oy = _build_files(tmp_path, [4, 0, 6])
    out = tmp_path / "combined.cstore"
    with colstore.open(paths) as ds:
        reader = ds.edit().write(out)  # no edits -> the files written end to end
    with reader:
        np.testing.assert_array_equal(reader["x"].array(), ox)
        np.testing.assert_array_equal(reader["y"].array(), oy)
        assert reader.n_rows == len(ox)


def test_edit_transform_across_files(tmp_path):
    paths, ox, oy = _build_files(tmp_path, [4, 6])
    out = tmp_path / "derived.cstore"
    with colstore.open(paths) as ds:
        frame = ds.edit()
        reader = frame.assign(x2=frame["x"] * 2).write(out)
    with reader:
        np.testing.assert_array_equal(reader["x"].array(), ox)
        np.testing.assert_array_equal(reader["y"].array(), oy)
        np.testing.assert_array_equal(reader["x2"].array(), ox * 2)


def test_edit_single_file_dataset_matches_reader(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [8])
    out = tmp_path / "single.cstore"
    with colstore.open(paths) as ds:  # one-file dataset (short-circuits to child)
        reader = ds.edit().assign(x2=ds.edit()["x"] * 2).write(out)
    with reader:
        np.testing.assert_array_equal(reader["x2"].array(), ox * 2)


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


def test_multifile_fancy_sorted_grouping_matches_oracle(tmp_path):
    # The fancy gather groups indices by file with one sort and un-sorts the
    # gathered values back into requested order. Stress that grouping over many
    # files (including an empty one) with duplicates and a forced concurrent
    # budget, against the oracle.
    from colstore import config

    paths, ox, oy = _build_files(tmp_path, [30, 0, 50, 20, 40])
    rng = np.random.default_rng(0)
    index = rng.integers(0, len(ox), size=500, dtype=np.int64)
    index[::7] = index[0]  # inject duplicates
    previous = config.get_gather_thread_cap()
    config.set_gather_thread_cap(4)
    try:
        with colstore.open(paths) as ds:
            np.testing.assert_array_equal(ds[index, "x"].array(), ox[index])
            both = ds[index, ["x", "y"]].dict()
            np.testing.assert_array_equal(both["x"], ox[index])
            np.testing.assert_array_equal(both["y"], oy[index])
    finally:
        config.set_gather_thread_cap(previous)


def test_multifile_fancy_native_matches_fallback_and_oracle(tmp_path, monkeypatch):
    # A dataset mixing a single-record file and a multi-record one (built with
    # several writes) exercises both branches of the segment-table builder. The
    # native multi-file kernel and the portable sort-once fallback must each
    # equal the oracle. The native path is asserted live so a builder bug can't
    # hide behind a silent fallback.
    from colstore import kernels

    single = tmp_path / "single.cstore"
    x0 = np.arange(30, dtype=np.int64)
    colstore.store({"x": x0, "y": (x0 * 2.0)}, single, show_progress=False).close()

    multi = tmp_path / "multi.cstore"
    x_blocks, y_blocks = [x0], [x0 * 2.0]
    with colstore.create(multi) as writer:
        for i in range(3):
            xb = np.arange(30 + 12 * i, 30 + 12 * (i + 1), dtype=np.int64)
            writer.write({"x": xb, "y": xb * 2.0})
            x_blocks.append(xb)
            y_blocks.append(xb * 2.0)
    ox = np.concatenate(x_blocks)
    oy = np.concatenate(y_blocks)

    rng = np.random.default_rng(0)
    index = rng.integers(0, ox.size, size=400, dtype=np.int64)
    index[::5] = index[0]  # duplicates

    with colstore.open([single, multi]) as ds:
        assert any(child._is_multi_record for child in ds._children)
        assert ds._native_segment_table("x") is not None  # native path is taken
        native_x = ds[index, "x"].array()
        native_both = ds[index, ["x", "y"]].dict()

    monkeypatch.setattr(kernels, "cpp_available", lambda: False)
    with colstore.open([single, multi]) as ds:
        fallback_x = ds[index, "x"].array()
        fallback_both = ds[index, ["x", "y"]].dict()

    for result in (native_x, fallback_x):
        np.testing.assert_array_equal(result, ox[index])
    for both in (native_both, fallback_both):
        np.testing.assert_array_equal(both["x"], ox[index])
        np.testing.assert_array_equal(both["y"], oy[index])


def test_multifile_fancy_native_sorted_matches_fallback_and_oracle(tmp_path, monkeypatch):
    # Non-decreasing fancy indices route through the cursor-walk kernel (single
    # column and dict). It must match the portable fallback and the oracle.
    from colstore import kernels
    from colstore.reader import _indices_are_sorted

    single = tmp_path / "single.cstore"
    x0 = np.arange(30, dtype=np.int64)
    colstore.store({"x": x0, "y": (x0 * 2.0)}, single, show_progress=False).close()

    multi = tmp_path / "multi.cstore"
    x_blocks, y_blocks = [x0], [x0 * 2.0]
    with colstore.create(multi) as writer:
        for i in range(3):
            xb = np.arange(30 + 12 * i, 30 + 12 * (i + 1), dtype=np.int64)
            writer.write({"x": xb, "y": xb * 2.0})
            x_blocks.append(xb)
            y_blocks.append(xb * 2.0)
    ox = np.concatenate(x_blocks)
    oy = np.concatenate(y_blocks)

    index = np.sort(np.random.default_rng(0).integers(0, ox.size, size=400)).astype(np.int64)
    assert _indices_are_sorted(index)  # the sorted branch is taken

    with colstore.open([single, multi]) as ds:
        native_x = ds[index, "x"].array()
        native_both = ds[index, ["x", "y"]].dict()

    monkeypatch.setattr(kernels, "cpp_available", lambda: False)
    with colstore.open([single, multi]) as ds:
        fallback_x = ds[index, "x"].array()
        fallback_both = ds[index, ["x", "y"]].dict()

    for result in (native_x, fallback_x):
        np.testing.assert_array_equal(result, ox[index])
    for both in (native_both, fallback_both):
        np.testing.assert_array_equal(both["x"], ox[index])
        np.testing.assert_array_equal(both["y"], oy[index])


def test_segment_table_cache_invalidated_on_append(tmp_path):
    # The per-column native segment table is memoized across reads; growing the
    # dataset must clear that memo so later reads see the new file. The
    # correctness check holds whether or not the compiled kernel is present (a
    # stale table would fold the pre-append offsets and misread the new rows).
    from colstore import kernels

    paths, ox, oy = _build_files(tmp_path, [40, 25, 35])
    rng = np.random.default_rng(0)

    ds = ColStoreDataset(paths[:2])  # 65 rows
    try:
        first = rng.integers(0, 65, size=200, dtype=np.int64)
        np.testing.assert_array_equal(ds[first, "x"].array(), ox[first])
        if kernels.cpp_available():
            assert "x" in ds._segment_table_cache  # populated by the read

        ds.append(paths[2])  # 65 -> 100 rows
        assert ds._segment_table_cache == {}  # invalidated by the structure change

        second = rng.integers(0, 100, size=300, dtype=np.int64)
        second[:4] = [66, 80, 99, 70]  # force reads from the appended file
        both = ds[second, ["x", "y"]].dict()
        np.testing.assert_array_equal(both["x"], ox[second])
        np.testing.assert_array_equal(both["y"], oy[second])
    finally:
        ds.close()


def test_segment_table_memoized_across_reads(tmp_path):
    # A second read reuses the cached table object rather than rebuilding it.
    from colstore import kernels

    if not kernels.cpp_available():
        pytest.skip("native segment table requires the compiled gather extension")
    paths, _, _ = _build_files(tmp_path, [50, 50])
    rng = np.random.default_rng(1)
    with colstore.open(paths) as ds:
        ds[rng.integers(0, 100, size=100, dtype=np.int64), "x"].array()
        cached = ds._segment_table_cache["x"]
        ds[rng.integers(0, 100, size=100, dtype=np.int64), "x"].array()
        assert ds._native_segment_table("x") is cached  # same object, not rebuilt


# ---- Uniform-grid multi-file routing ----------------------------------------
# Equal-sized files (or equal records) make the global segment table a uniform
# grid, so an unsorted native gather divides (s = idx / rows_per_segment) instead
# of binary-searching. The route is a no-op fallback to the searching kernel when
# the grid does not hold, is taken only for unsorted reads (sorted keeps the
# cursor walk), and the grid test is memoized and reset on append.

_UNIFORM_SPY = [
    "gather_segment_uniform",
    "gather_segment_uniform_bins",
    "gather_segment",
    "gather_segment_bins",
    "gather_segment_sorted",
]


def test_uniform_grid_routes_to_division_binning(tmp_path, monkeypatch):
    from _helpers import kernel_spy

    if not kernels.cpp_available():
        pytest.skip("native gather requires the compiled extension")
    paths, ox, oy = _build_files(tmp_path, [5000] * 6)  # equal files: a uniform grid
    rng = np.random.default_rng(0)
    index = rng.integers(0, ox.size, size=8000, dtype=np.int64)
    with colstore.open(paths) as ds:
        calls = kernel_spy(monkeypatch, _UNIFORM_SPY)
        np.testing.assert_array_equal(ds[index, "x"].array(), ox[index])
        assert calls == ["gather_segment_uniform"]
        calls.clear()
        both = ds[index, ["x", "y"]].dict()
        assert calls == ["gather_segment_uniform_bins"]  # trailing column unspied (withbins)
        np.testing.assert_array_equal(both["x"], ox[index])
        np.testing.assert_array_equal(both["y"], oy[index])


def test_non_uniform_dataset_keeps_searching_kernel(tmp_path, monkeypatch):
    from _helpers import kernel_spy

    if not kernels.cpp_available():
        pytest.skip("native gather requires the compiled extension")
    paths, ox, oy = _build_files(tmp_path, [3000, 5000, 1000, 4000])  # unequal: not a grid
    rng = np.random.default_rng(1)
    index = rng.integers(0, ox.size, size=6000, dtype=np.int64)
    with colstore.open(paths) as ds:
        calls = kernel_spy(monkeypatch, _UNIFORM_SPY)
        np.testing.assert_array_equal(ds[index, "x"].array(), ox[index])
        assert calls == ["gather_segment"]
        calls.clear()
        both = ds[index, ["x", "y"]].dict()
        assert calls == ["gather_segment_bins"]
        np.testing.assert_array_equal(both["x"], ox[index])
        np.testing.assert_array_equal(both["y"], oy[index])


def test_uniform_grid_partial_global_tail_routes(tmp_path, monkeypatch):
    from _helpers import kernel_spy

    if not kernels.cpp_available():
        pytest.skip("native gather requires the compiled extension")
    paths, ox, _ = _build_files(tmp_path, [4000, 4000, 4000, 1500])  # last smaller: still a grid
    rng = np.random.default_rng(2)
    index = rng.integers(0, ox.size, size=6000, dtype=np.int64)
    with colstore.open(paths) as ds:
        calls = kernel_spy(monkeypatch, _UNIFORM_SPY)
        np.testing.assert_array_equal(ds[index, "x"].array(), ox[index])
        assert calls == ["gather_segment_uniform"]


def test_uniform_grid_sorted_keeps_cursor_walk(tmp_path, monkeypatch):
    from _helpers import kernel_spy

    if not kernels.cpp_available():
        pytest.skip("native gather requires the compiled extension")
    paths, ox, _ = _build_files(tmp_path, [2000] * 5)
    index = np.sort(np.random.default_rng(3).integers(0, ox.size, size=4000)).astype(np.int64)
    with colstore.open(paths) as ds:
        calls = kernel_spy(monkeypatch, _UNIFORM_SPY)
        np.testing.assert_array_equal(ds[index, "x"].array(), ox[index])
        assert calls == ["gather_segment_sorted"]  # sorted: cursor walk, not division


def test_uniform_grid_memo_reset_on_append(tmp_path, monkeypatch):
    from _helpers import kernel_spy

    if not kernels.cpp_available():
        pytest.skip("native gather requires the compiled extension")
    # A larger trailing file breaks the grid (a partial last segment would not);
    # appending it must reset the memo so the read switches to the searching kernel.
    paths, ox, _ = _build_files(tmp_path, [1000, 1000, 1500])
    rng = np.random.default_rng(4)
    ds = ColStoreDataset(paths[:2])  # two equal files: a grid
    try:
        idx1 = rng.integers(0, 2000, size=3000, dtype=np.int64)
        calls = kernel_spy(monkeypatch, _UNIFORM_SPY)
        np.testing.assert_array_equal(ds[idx1, "x"].array(), ox[idx1])
        assert calls == ["gather_segment_uniform"]  # grid holds

        ds.append(paths[2])  # 1500-row file: last segment larger than R, grid broken
        calls.clear()
        idx2 = rng.integers(0, ds.n_rows, size=3000, dtype=np.int64)
        np.testing.assert_array_equal(ds[idx2, "x"].array(), ox[idx2])
        assert calls == ["gather_segment"]  # now searches
    finally:
        ds.close()


def test_uniform_grid_route_matches_fallback_and_oracle(tmp_path, monkeypatch):
    # The division route and the portable sort-once fallback must agree, and both
    # equal the oracle, including duplicate indices.
    paths, ox, oy = _build_files(tmp_path, [4000] * 5)
    rng = np.random.default_rng(5)
    index = rng.integers(0, ox.size, size=7000, dtype=np.int64)
    index[::7] = index[0]  # duplicates
    with colstore.open(paths) as ds:
        native_x = ds[index, "x"].array()
        native_both = ds[index, ["x", "y"]].dict()
    monkeypatch.setattr(kernels, "cpp_available", lambda: False)
    with colstore.open(paths) as ds:
        fallback_x = ds[index, "x"].array()
        fallback_both = ds[index, ["x", "y"]].dict()
    for result in (native_x, fallback_x):
        np.testing.assert_array_equal(result, ox[index])
    for both in (native_both, fallback_both):
        np.testing.assert_array_equal(both["x"], ox[index])
        np.testing.assert_array_equal(both["y"], oy[index])


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


# ---- Parallel fill path ------------------------------------------------


def test_parallel_fill_matches_oracle_with_threads(tmp_path):
    # Force the file-level fill pool on (CI otherwise runs the budget=1 serial
    # path), so the concurrent disjoint writes are exercised against the oracle.
    from colstore import config

    paths, ox, oy = _build_files(tmp_path, [40, 0, 60, 35])
    previous = config.get_gather_thread_cap()
    config.set_gather_thread_cap(4)
    try:
        with colstore.open(paths) as ds:
            n = ds.n_rows
            np.testing.assert_array_equal(ds["x"].array(), ox)
            whole = ds[:].dict()
            np.testing.assert_array_equal(whole["x"], ox)
            np.testing.assert_array_equal(whole["y"], oy)
            for sl in (slice(1, n - 1), slice(3, n, 7), slice(None, None, -2)):
                np.testing.assert_array_equal(ds[sl, "x"].array(), ox[sl])
            idx = np.array([n - 1, 0, n // 2, 5, 5, 1], dtype=np.int64)
            both = ds[idx, ["x", "y"]].dict()
            np.testing.assert_array_equal(both["x"], ox[idx])
            np.testing.assert_array_equal(both["y"], oy[idx])
            mask = np.random.default_rng(0).random(n) < 0.4
            np.testing.assert_array_equal(ds[mask, "x"].array(), ox[mask])
    finally:
        config.set_gather_thread_cap(previous)


def test_contiguous_read_mixed_record_files_matches_oracle(tmp_path, monkeypatch):
    # A dataset mixing a single-record file (kernel-copyable) and a multi-record
    # file (not viewable -> gathered) exercises the gap path of the contiguous
    # fill: some regions copy through the parallel-copy kernel, the rest gather
    # alongside. The same read with the kernel forced off takes the host-language
    # path for every region. Both must equal the concatenated oracle.
    from colstore import kernels

    single = tmp_path / "s.cstore"
    x0 = np.arange(40, dtype=np.int64)
    colstore.store({"x": x0, "y": x0 * 1.5}, single, show_progress=False).close()

    multi = tmp_path / "m.cstore"
    x_blocks, y_blocks = [x0], [x0 * 1.5]
    with colstore.create(multi) as writer:
        for i in range(3):
            xb = np.arange(40 + 10 * i, 40 + 10 * (i + 1), dtype=np.int64)
            writer.write({"x": xb, "y": xb * 1.5})
            x_blocks.append(xb)
            y_blocks.append(xb * 1.5)
    ox, oy = np.concatenate(x_blocks), np.concatenate(y_blocks)

    def check(ds):
        n = ds.n_rows
        np.testing.assert_array_equal(ds["x"].array(), ox)
        whole = ds.dict()
        np.testing.assert_array_equal(whole["x"], ox)
        np.testing.assert_array_equal(whole["y"], oy)
        np.testing.assert_array_equal(ds[3 : n - 2, "x"].array(), ox[3 : n - 2])
        mask = np.zeros(n, dtype=bool)
        mask[::3] = True
        np.testing.assert_array_equal(ds[mask, "y"].array(), oy[mask])

    with colstore.open([single, multi]) as ds:
        assert any(child._is_multi_record for child in ds._children)  # gap path is live
        check(ds)
    monkeypatch.setattr(kernels, "cpp_available", lambda: False)
    with colstore.open([single, multi]) as ds:
        check(ds)


# ---- concat(): lazy dataset or eager written file ----------------------


def test_concat_out_none_returns_dataset(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [4, 6])
    with colstore.concat(paths) as ds:
        assert isinstance(ds, ColStoreDataset)
        np.testing.assert_array_equal(ds["x"].array(), ox)


def test_concat_writes_combined_file(tmp_path):
    paths, ox, oy = _build_files(tmp_path, [4, 0, 6])
    out = tmp_path / "all.cstore"
    reader = colstore.concat(paths, out=out)
    with reader:
        assert isinstance(reader, ColStoreReader)
        assert reader.n_rows == len(ox)
        assert reader.columns == ["x", "y"]
        np.testing.assert_array_equal(reader["x"].array(), ox)
        np.testing.assert_array_equal(reader["y"].array(), oy)


def test_concat_accepts_open_readers_and_leaves_them_open(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [4, 6])
    left = colstore.open(paths[0])
    right = colstore.open(paths[1])
    out = tmp_path / "combined.cstore"
    reader = colstore.concat([left, right], out=out)
    with reader:
        np.testing.assert_array_equal(reader["x"].array(), ox)
    assert left._closed is False  # borrowed sources are left open
    assert right._closed is False
    left.close()
    right.close()


def test_concat_single_source_copies(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [7])
    out = tmp_path / "copy.cstore"
    with colstore.concat(paths, out=out) as reader:
        np.testing.assert_array_equal(reader["x"].array(), ox)


def test_concat_streams_under_tight_memory_budget(tmp_path):
    paths, ox, oy = _build_files(tmp_path, [5, 5, 5])
    out = tmp_path / "streamed.cstore"
    with colstore.concat(paths, out=out, memory_budget=64) as reader:
        np.testing.assert_array_equal(reader["x"].array(), ox)
        np.testing.assert_array_equal(reader["y"].array(), oy)


def test_concat_out_equal_to_source_raises(tmp_path):
    paths, _, _ = _build_files(tmp_path, [4, 6])
    with pytest.raises(ValueError, match="also one of the sources"):
        colstore.concat(paths, out=paths[0])


def test_concat_zero_sources_to_file_raises(tmp_path):
    out = tmp_path / "empty.cstore"
    with pytest.raises(ValueError, match="at least one source"):
        colstore.concat([], out=out)


def test_concat_schema_mismatch_raises(tmp_path):
    good, _, _ = _build_files(tmp_path, [4])
    narrow = _store_one(
        tmp_path,
        "narrow.cstore",
        {"x": np.arange(3, dtype=np.int32), "y": np.arange(3, dtype=np.float64)},
    )
    out = tmp_path / "bad.cstore"
    with pytest.raises(ValueError, match="Schema mismatch"):
        colstore.concat([good[0], narrow], out=out)


def test_dataset_array_shortcut(tmp_path):
    paths, ox, _ = _build_files(tmp_path, [4, 6])
    ds = colstore.open(paths)
    try:
        np.testing.assert_array_equal(ds.array("x"), ox)  # gathers across files
        np.testing.assert_array_equal(ds.array("x"), ds.dict()["x"])
        np.testing.assert_array_equal(ds.array("x"), ds["x"].array())
    finally:
        ds.close()


# ---- multi-file + multirecord open via the C++ record-index kernel ----------


@pytest.mark.skipif(not kernels.cpp_available(), reason="multi-record reads need the gather kernel")
def test_multifile_multirecord_open_and_read(tmp_path):
    """A many-child, multi-record dataset opens and reads correctly.

    Each child's per-record index is built by ``format.read_record_index`` (the
    C++ kernel when the extension is built). A full read and a fancy read
    spanning files and records are checked against the concatenated source.
    """
    blocks: dict[str, list[np.ndarray]] = {}
    paths = []
    for i in range(4):
        cols = testing.make_columns(600, 3, dtype="float64", seed=i)
        path = tmp_path / f"part_{i}.cstore"
        testing.write_columns(path, cols, records=5).close()
        paths.append(path)
        for name, values in cols.items():
            blocks.setdefault(name, []).append(values)
    oracle = {name: np.concatenate(parts) for name, parts in blocks.items()}

    with colstore.open(paths) as ds:
        assert ds.n_rows == 4 * 600
        full = ds.dict()
        first = ds.columns[0]
        idx = np.array([0, 599, 600, 1801, 2399], dtype=np.int64)  # spans files and records
        picked = ds[idx, first].array()

    for name, expected in oracle.items():
        np.testing.assert_array_equal(full[name], expected)
    np.testing.assert_array_equal(picked, oracle[first][idx])


# ---- multi-file mask-native kernel ------------------------------------------
#
# A dense boolean mask on a multi-file dataset gathers through the native
# multi-file mask kernel (colstore_gather_segment_mask) over the cached segment
# table; a sparse mask, a non-native dtype, or no extension declines to the
# per-file path. Both must be byte-identical to numpy mask indexing.


def _build_multifile_mask(tmp_path, n_files, rows, cols, records):
    """Write ``n_files`` multi-record files; return paths and the concatenated oracle."""
    paths = []
    blocks: dict[str, list[np.ndarray]] = {}
    for i in range(n_files):
        columns = testing.make_columns(rows, cols, dtype="float64", seed=i)
        path = tmp_path / f"mk_{i}.cstore"
        testing.write_columns(path, columns, records=records).close()
        paths.append(path)
        for name, values in columns.items():
            blocks.setdefault(name, []).append(values)
    return paths, {name: np.concatenate(v) for name, v in blocks.items()}


@pytest.mark.skipif(not kernels.cpp_available(), reason="needs the compiled extension")
@pytest.mark.parametrize("records", [1, 20])
@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_multifile_mask_kernel_matches_per_file(tmp_path, records, fraction):
    """The native mask kernel and the per-file path are both byte-identical to numpy."""
    paths, oracle = _build_multifile_mask(tmp_path, 5, 4000, 4, records)
    names = list(oracle)
    rng = np.random.default_rng(records)
    with colstore.open(paths) as ds:
        mask = np.ones(ds.n_rows, bool) if fraction == 1.0 else (rng.random(ds.n_rows) < fraction)
        saved = config.get_multifile_mask_density_gate()
        try:
            config.set_multifile_mask_density_gate(0.0)  # force the kernel
            kern = ds[mask, names].dict()
            config.set_multifile_mask_density_gate(2.0)  # force the per-file path
            per_file = ds[mask, names].dict()
        finally:
            config.set_multifile_mask_density_gate(saved)
        for name in names:
            expected = oracle[name][mask]
            np.testing.assert_array_equal(kern[name], expected)
            np.testing.assert_array_equal(per_file[name], expected)
            assert kern[name].dtype == expected.dtype


@pytest.mark.skipif(not kernels.cpp_available(), reason="needs the compiled extension")
def test_multifile_mask_density_gate_declines_sparse(tmp_path):
    """_mask_native takes the kernel for a dense mask and declines a sparse one."""
    paths, oracle = _build_multifile_mask(tmp_path, 4, 4000, 3, 10)
    names = list(oracle)
    with colstore.open(paths) as ds:
        saved = config.get_multifile_mask_density_gate()
        try:
            config.set_multifile_mask_density_gate(0.25)
            dense = np.ones(ds.n_rows, bool)
            sparse = np.zeros(ds.n_rows, bool)
            sparse[::100] = True  # 1% selected, below the 0.25 gate
            assert ds._mask_native(names, dense) is not None
            assert ds._mask_native(names, sparse) is None
        finally:
            config.set_multifile_mask_density_gate(saved)


@pytest.mark.skipif(not kernels.cpp_available(), reason="needs the compiled extension")
def test_multifile_mask_frame_filter_matches_oracle(tmp_path):
    """frame.filter on a multi-file dataset keeps a dense mask and matches numpy."""
    paths, oracle = _build_multifile_mask(tmp_path, 5, 4000, 3, 20)
    names = list(oracle)
    with colstore.open(paths) as ds:
        keep = oracle[names[0]] > 0.0  # ~50% dense
        got = ds.edit().filter(col(names[0]) > 0.0).dict()
        for name in names:
            np.testing.assert_array_equal(got[name], oracle[name][keep])
