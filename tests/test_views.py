"""Tests for ColumnView and TableView, including the zero-copy read API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _helpers import opened

import colstore
from colstore import ColumnView, TableView
from colstore import reader as reader_mod


def test_string_indexing_returns_column_view(small_store):
    view = small_store["price"]
    assert isinstance(view, ColumnView)
    assert not isinstance(view, TableView)


def test_list_indexing_returns_table_view(small_store):
    view = small_store[["price", "qty"]]
    assert isinstance(view, TableView)
    assert not isinstance(view, ColumnView)


def test_row_only_indexing_returns_table_view(small_store):
    view = small_store[100:200]
    assert isinstance(view, TableView)


def test_row_with_single_column_returns_column_view(small_store):
    view = small_store[100:200, "price"]
    assert isinstance(view, ColumnView)


def test_row_with_multi_column_returns_table_view(small_store):
    view = small_store[100:200, ["price", "qty"]]
    assert isinstance(view, TableView)


def test_column_view_array_preserves_dtype(small_store, small_frame):
    result = small_store["price"].array()
    assert result.dtype == np.float32
    assert np.allclose(result, small_frame["price"].to_numpy())


def test_column_view_does_not_have_dict():
    """ColumnView does not implement dict / recarray / frame."""
    assert not hasattr(ColumnView, "dict")
    assert not hasattr(ColumnView, "recarray")
    assert not hasattr(ColumnView, "frame")


def test_table_view_array_and_indexing(small_store, small_frame):
    """TableView reads one column by name and projects columns by indexing."""
    tv = small_store[100:200, ["price", "qty"]]
    # array(name) -> a 1-D ndarray, the same as the reader / a column view
    np.testing.assert_array_equal(tv.array("price"), small_frame["price"].iloc[100:200].to_numpy())
    # table[name] -> a ColumnView; table[[names]] -> a narrowed TableView
    assert isinstance(tv["price"], ColumnView)
    np.testing.assert_array_equal(tv["price"].array(), tv.array("price"))
    assert isinstance(tv[["price"]], TableView)
    assert tv[["price"]].columns == ["price"]
    # no no-arg array(): heterogeneous columns can't pack into one homogeneous ndarray
    with pytest.raises(TypeError):
        tv.array()


def test_chained_indexing_consistent_with_direct(small_store):
    """ds[rows]['col'] matches ds[rows, 'col'] and stays a full eager column."""
    chained = small_store[100:200]["price"]  # TableView -> ColumnView
    direct = small_store[100:200, "price"]  # ColumnView directly
    assert isinstance(chained, ColumnView)
    np.testing.assert_array_equal(chained.array(), direct.array())
    # the chained column behaves like any column: eager operators and reductions
    np.testing.assert_array_equal(chained * 2, direct.array() * 2)
    assert chained.mean() == direct.mean()


def test_table_view_dict_matches_source(small_store, small_frame):
    result = small_store[100:200, ["price", "qty"]].dict()
    assert set(result) == {"price", "qty"}
    assert np.allclose(result["price"], small_frame["price"].iloc[100:200].to_numpy())
    assert np.array_equal(result["qty"], small_frame["qty"].iloc[100:200].to_numpy())


def test_table_view_recarray_preserves_each_dtype(small_store, small_frame):
    record_array = small_store[100:110, ["price", "qty", "flag"]].recarray()
    assert record_array.dtype.names == ("price", "qty", "flag")
    assert record_array["price"].dtype == np.float32
    assert record_array["qty"].dtype == np.int32
    assert record_array["flag"].dtype == np.uint8
    assert record_array.shape == (10,)


@pytest.mark.parametrize(
    "select",
    [
        slice(None),  # whole, contiguous views
        slice(100, 200),  # forward slice
        slice(None, None, 3),  # strided -> sources materialized
        np.array([7, 1, 200, 1, 50]),  # fancy -> sources materialized
    ],
)
def test_table_view_recarray_kernel_matches_fallback(small_store, select, monkeypatch):
    # A view's recarray() now interleaves through the kernel for any row
    # selection; it must equal the column-major host fallback field for field.
    from colstore import kernels

    columns = ["price", "qty", "flag", "id"]
    with_kernel = small_store[select, columns].recarray()
    monkeypatch.setattr(kernels, "cpp_available", lambda: False)
    fallback = small_store[select, columns].recarray()
    assert with_kernel.dtype == fallback.dtype
    for name in columns:
        np.testing.assert_array_equal(with_kernel[name], fallback[name])


def test_table_view_frame_returns_dataframe(small_store, small_frame):
    out_frame = small_store[100:110, ["price", "qty"]].frame()
    assert isinstance(out_frame, pd.DataFrame)
    assert list(out_frame.columns) == ["price", "qty"]
    assert len(out_frame) == 10


def test_column_view_repr_includes_column_name(small_store):
    repr_string = repr(small_store["price"])
    assert "ColumnView" in repr_string
    assert "price" in repr_string


def test_table_view_repr_includes_column_list(small_store):
    repr_string = repr(small_store[["price", "qty"]])
    assert "TableView" in repr_string
    assert "price" in repr_string
    assert "qty" in repr_string


def test_column_view_exposes_column_and_dtype(small_store):
    view = small_store["price"]
    assert view.column == "price"
    assert view.dtype == np.float32


def test_table_view_exposes_columns_and_dtypes(small_store):
    view = small_store[["price", "qty"]]
    assert view.columns == ["price", "qty"]
    assert view.n_columns == 2
    assert view.dtypes == {"price": np.float32, "qty": np.int32}


def test_lazy_view_does_not_read_until_materialized(small_store):
    """Building a view performs no I/O — verify by repeated index without read."""
    for _ in range(100):
        _ = small_store[100:200, ["price", "qty"]]
    # If reads happened on construction, this loop would be much slower
    # than a single read; assertion below sanity-checks values.
    materialized = small_store[100:200, ["price", "qty"]].dict()
    assert materialized["price"].shape == (100,)


# ---- Whole-store materialization shortcuts on ColStoreReader -----------


def test_reader_dict_returns_all_columns_in_order(small_store):
    """``ds.dict()`` returns one entry per column in on-disk order."""
    result = small_store.dict()
    assert list(result) == small_store.columns
    for name, arr in result.items():
        assert arr.shape == (small_store.n_rows,)
        assert arr.dtype == small_store.dtypes[name]


def test_reader_dict_matches_explicit_slice_view(small_store):
    """``ds.dict()`` is equivalent to ``ds[:].dict()``."""
    direct = small_store.dict()
    via_view = small_store[:].dict()
    assert list(direct) == list(via_view)
    for name in direct:
        np.testing.assert_array_equal(direct[name], via_view[name])


def test_reader_recarray_returns_structured_with_all_columns(small_store):
    """``ds.recarray()`` returns a structured ndarray with one field per column."""
    rec = small_store.recarray()
    assert rec.dtype.names == tuple(small_store.columns)
    assert rec.shape == (small_store.n_rows,)
    for name in small_store.columns:
        assert rec[name].dtype == small_store.dtypes[name]


def test_reader_recarray_matches_explicit_slice_view(small_store):
    """``ds.recarray()`` is equivalent to ``ds[:].recarray()``."""
    direct = small_store.recarray()
    via_view = small_store[:].recarray()
    np.testing.assert_array_equal(direct, via_view)


def test_reader_recarray_kernel_matches_fallback(small_store, monkeypatch):
    """``recarray()`` agrees whether the interleave kernel runs or the host path."""
    from colstore import kernels

    with_kernel = small_store.recarray()
    monkeypatch.setattr(kernels, "cpp_available", lambda: False)
    fallback = small_store.recarray()
    np.testing.assert_array_equal(with_kernel, fallback)


def test_recarray_over_multirecord_store_matches_oracle(tmp_path):
    """A multi-record store's columns are not viewable, so ``recarray()``
    gathers each before interleaving; the result must match the concatenation."""
    import colstore

    path = tmp_path / "multi.cstore"
    x_blocks, y_blocks = [], []
    with colstore.create(path) as writer:
        for i in range(3):
            x = np.arange(10 * i, 10 * (i + 1), dtype=np.int64)
            y = (x * 0.5).astype(np.float64)
            writer.write({"x": x, "y": y})
            x_blocks.append(x)
            y_blocks.append(y)
    with colstore.open(path) as reader:
        assert reader._is_multi_record  # the gather-source path is exercised
        rec = reader.recarray()
        np.testing.assert_array_equal(rec["x"], np.concatenate(x_blocks))
        np.testing.assert_array_equal(rec["y"], np.concatenate(y_blocks))


def test_reader_frame_returns_dataframe_with_all_columns(small_store):
    """``ds.frame()`` returns a DataFrame whose columns match the on-disk order."""
    df = small_store.frame()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == small_store.columns
    assert len(df) == small_store.n_rows


def test_reader_frame_matches_explicit_slice_view(small_store):
    """``ds.frame()`` is equivalent to ``ds[:].frame()``."""
    direct = small_store.frame()
    via_view = small_store[:].frame()
    pd.testing.assert_frame_equal(direct, via_view)


def test_reader_dict_after_close_raises(small_store):
    """The shortcut methods refuse to operate after close()."""
    import pytest

    small_store.close()
    with pytest.raises(ValueError, match="closed"):
        small_store.dict()
    with pytest.raises(ValueError, match="closed"):
        small_store.recarray()
    with pytest.raises(ValueError, match="closed"):
        small_store.frame()


# ---- frame() construction: no-consolidate path ------------------------------
#
# ``frame()`` skips pandas' default dtype-block consolidation copy so that
# whole-store materialization isn't dominated by a redundant 1 GB memcpy.
# These tests pin the observable properties of that path: equivalence with
# the consolidating constructor, fragmented block layout (one block per
# column), and correct behavior on the edge cases (empty / single column).


def test_reader_frame_is_one_block_per_column(tmp_path):
    """Optimized frame() produces a non-consolidated BlockManager.

    Many same-dtype columns are the case where consolidation is most
    expensive (50 float64 columns -> one big 2D block). The optimized
    path keeps each column in its own Block so there is no consolidation
    copy.
    """
    import colstore

    rng = np.random.default_rng(0)
    n_rows = 4096
    columns = {f"c{i:02d}": rng.standard_normal(n_rows) for i in range(8)}
    store_path = tmp_path / "homogeneous.cstore"
    store = colstore.store(columns, store_path, show_progress=False)
    try:
        df = store.frame()
        assert len(df._mgr.blocks) == len(columns)
    finally:
        store.close()


def test_reader_frame_values_match_baseline_constructor(tmp_path):
    """Optimized frame() is value-equivalent to ``pd.DataFrame(dict)``."""
    import colstore

    rng = np.random.default_rng(1)
    n_rows = 2048
    columns = {
        "f64_a": rng.standard_normal(n_rows),
        "f64_b": rng.standard_normal(n_rows),
        "i32": rng.integers(-1000, 1000, n_rows, dtype=np.int32),
        "u8": rng.integers(0, 255, n_rows, dtype=np.uint8),
    }
    store_path = tmp_path / "mixed.cstore"
    store = colstore.store(columns, store_path, show_progress=False)
    try:
        optimized = store.frame()
        baseline = pd.DataFrame(store.dict())
        pd.testing.assert_frame_equal(optimized, baseline)
    finally:
        store.close()


def test_reader_frame_with_single_column(tmp_path):
    """A 1-column store frames correctly (degenerate case for the helper)."""
    import colstore

    n_rows = 64
    store_path = tmp_path / "one_col.cstore"
    store = colstore.store(
        {"only": np.arange(n_rows, dtype=np.float64)}, store_path, show_progress=False
    )
    try:
        df = store.frame()
        assert list(df.columns) == ["only"]
        assert len(df) == n_rows
        np.testing.assert_array_equal(df["only"].to_numpy(), np.arange(n_rows, dtype=np.float64))
    finally:
        store.close()


def test_table_view_frame_is_one_block_per_column(tmp_path):
    """TableView.frame() also uses the no-consolidate path."""
    import colstore

    rng = np.random.default_rng(2)
    n_rows = 4096
    columns = {f"c{i:02d}": rng.standard_normal(n_rows) for i in range(6)}
    store_path = tmp_path / "view_homogeneous.cstore"
    store = colstore.store(columns, store_path, show_progress=False)
    try:
        df = store[:, list(columns)].frame()
        assert len(df._mgr.blocks) == len(columns)
        # Sliced TableView too.
        df_slice = store[100:1100, list(columns)].frame()
        assert len(df_slice._mgr.blocks) == len(columns)
        assert len(df_slice) == 1000
    finally:
        store.close()


def test_make_dataframe_no_consolidate_handles_empty():
    """The helper accepts an empty column dict and returns an empty frame.

    The frame() shortcut never produces an empty dict (every store has at
    least one column), but the helper is a public-ish surface and should
    degrade gracefully so that ``frame()`` on a future zero-column store
    or a test does not crash.
    """
    from colstore._pandas import _make_dataframe_no_consolidate

    df = _make_dataframe_no_consolidate({})
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 0
    assert len(df) == 0


def test_frame_falls_back_when_pandas_api_changes(tmp_path, monkeypatch):
    """frame() degrades gracefully when the private pandas API shifts.

    Simulates a future pandas where ``create_block_manager_from_column_arrays``
    has a different signature (TypeError at call time). frame() should
    still return a valid DataFrame and emit a UserWarning so the
    regression is visible to the user without breaking their code.

    ``ImportError`` and ``AttributeError`` paths are covered by the
    feature-detect ``try`` above the call. ``TypeError`` is the call-time
    failure mode -- signature drift after import succeeds -- and is what
    pins the fallback's defense against a pandas-internal API change.
    """
    import warnings

    from pandas.core.internals import managers as pd_managers

    import colstore

    def raising_stub(*args, **kwargs):
        raise TypeError("unexpected keyword 'consolidate' (simulated API change)")

    monkeypatch.setattr(pd_managers, "create_block_manager_from_column_arrays", raising_stub)

    columns = {
        "a": np.arange(10, dtype=np.float64),
        "b": np.arange(10, dtype=np.int32),
    }
    store_path = tmp_path / "fallback.cstore"
    store = colstore.store(columns, store_path, show_progress=False)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            df = store.frame()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 10
        np.testing.assert_array_equal(df["a"].to_numpy(), np.arange(10, dtype=np.float64))
        np.testing.assert_array_equal(df["b"].to_numpy(), np.arange(10, dtype=np.int32))
        assert any(
            "frame() optimized" in str(w.message) for w in captured
        ), f"expected a fallback warning; got {[str(w.message) for w in captured]}"
    finally:
        store.close()


# ---- Zero-copy read API (copy=False) ------------------------------------------
# Tests for the zero-copy read API (``copy=False``).
#
# Contract: ``array(copy=False)`` / ``dict(copy=False)`` return READ-ONLY
# ndarray views sharing memory with the store's open memmaps -- no bytes
# copied -- exactly when the store is single-record, the dtype is native byte
# order, and the selector is None / int / slice (any step). Every other case
# raises ``ValueError`` (never a silent copy): fancy and boolean selectors,
# multi-record stores (the error points at ``colstore.compact``), and
# non-native dtypes. Views pin the underlying mapping, so they remain valid
# after the reader is closed. ``copy=True`` (the default) is byte-for-byte
# the previous behavior: owning, writable arrays.


@pytest.fixture()
def single_record_store(tmp_path):
    rng = np.random.default_rng(51)
    data = {
        "f8": rng.standard_normal(10_000),
        "f4": rng.standard_normal(10_000).astype(np.float32),
        "i2": rng.integers(-(2**14), 2**14, 10_000).astype(np.int16),
    }
    path = tmp_path / "single.cstore"
    with colstore.create(path) as writer:
        writer.write(data)
    dataset = colstore.open(path)
    yield dataset, data
    dataset.close()


def test_full_column_view_shares_memory(single_record_store):
    dataset, data = single_record_store
    for name, values in data.items():
        view = dataset[name].array(copy=False)
        assert np.shares_memory(view, dataset._memmaps[name]), name
        assert not view.flags.writeable
        assert not view.flags.owndata
        assert type(view) is np.ndarray  # re-classed, not a memmap subclass
        assert view.dtype == values.dtype
        assert np.array_equal(view, values), name


@pytest.mark.parametrize(
    "selector",
    [slice(None), slice(100, 5_000), slice(7, 9_000, 13), slice(None, None, -1), slice(50, 10, -3)],
)
def test_slice_views_match_copies(single_record_store, selector):
    dataset, data = single_record_store
    view = dataset[selector, "f8"].array(copy=False)
    owning = dataset[selector, "f8"].array()
    assert np.shares_memory(view, dataset._memmaps["f8"])
    assert np.array_equal(view, owning)
    assert np.array_equal(view, data["f8"][selector])


def test_scalar_selector_view_shape_matches_copy(single_record_store):
    dataset, data = single_record_store
    view = dataset[42, "f8"].array(copy=False)
    assert view.shape == dataset[42, "f8"].array().shape == (1,)
    assert view[0] == data["f8"][42]
    assert np.shares_memory(view, dataset._memmaps["f8"])


def test_table_dict_views(single_record_store):
    dataset, data = single_record_store
    result = dataset[:, ["f8", "i2"]].dict(copy=False)
    assert list(result) == ["f8", "i2"]
    for name, view in result.items():
        assert np.shares_memory(view, dataset._memmaps[name]), name
        assert not view.flags.writeable
        assert np.array_equal(view, data[name]), name


def test_reader_dict_shortcut_views(single_record_store):
    dataset, data = single_record_store
    result = dataset.dict(copy=False)
    assert set(result) == set(data)
    for name, view in result.items():
        assert np.shares_memory(view, dataset._memmaps[name]), name


def test_views_are_immutable(single_record_store):
    dataset, _ = single_record_store
    view = dataset["f8"].array(copy=False)
    with pytest.raises(ValueError, match="read-only"):
        view[0] = 1.0


def test_copy_true_unchanged(single_record_store):
    dataset, data = single_record_store
    owning = dataset["f8"].array()
    assert owning.flags.writeable
    assert not np.shares_memory(owning, dataset._memmaps["f8"])
    owning[0] = 123.0  # mutating the copy must not touch the store
    assert dataset["f8"].array(copy=False)[0] == data["f8"][0]


def test_fancy_and_boolean_selectors_raise(single_record_store):
    dataset, _ = single_record_store
    with pytest.raises(ValueError, match="copy=True"):
        dataset[np.array([1, 5, 2]), "f8"].array(copy=False)
    with pytest.raises(ValueError, match="copy=True"):
        dataset[[1, 3, 5], "f8"].array(copy=False)  # Python-list fancy form
    mask = np.zeros(dataset.n_rows, dtype=bool)
    mask[::7] = True
    with pytest.raises(ValueError, match="copy=True"):
        dataset[mask, "f8"].array(copy=False)


def test_empty_index_selects_no_rows(small_store):
    """A bare empty index (float64 by NumPy's default dtype) selects no rows, not an error."""
    for sel in ([], np.array([], dtype=np.float64), np.array([], dtype=np.int64)):
        assert small_store[sel].recarray().shape[0] == 0
    assert len(small_store[[], "price"].array()) == 0  # empty index + single column
    with pytest.raises(IndexError, match="integer or boolean"):
        small_store[[1.5, 2.5]].recarray()  # a non-empty non-integer index is still rejected


def test_table_view_frame_copy_false_shares_memory(single_record_store):
    dataset, data = single_record_store
    df = dataset[:, ["f8", "i2"]].frame(copy=False)
    assert list(df.columns) == ["f8", "i2"]
    for name in ("f8", "i2"):
        assert np.shares_memory(df[name].values, dataset._memmaps[name]), name
        assert np.array_equal(df[name].values, data[name]), name


def test_reader_frame_copy_false_shares_memory(single_record_store):
    dataset, data = single_record_store
    df = dataset.frame(copy=False)  # whole-store reader shortcut
    assert set(df.columns) == set(data)
    for name in data:
        assert np.shares_memory(df[name].values, dataset._memmaps[name]), name


def test_frame_copy_false_is_immutable(single_record_store):
    dataset, _ = single_record_store
    df = dataset.frame(copy=False)
    # Writing through the read-only view must raise, not reach the store.
    with pytest.raises(ValueError, match="read-only"):
        df.iloc[0, 0] = 1.0


def test_frame_copy_true_owns_its_columns(single_record_store):
    dataset, data = single_record_store
    df = dataset.frame()  # default copy=True: owning columns
    assert not np.shares_memory(df["f8"].values, dataset._memmaps["f8"])
    assert np.array_equal(df["f8"].values, data["f8"])


def test_frame_keeps_one_block_per_column(tmp_path):
    # Two same-dtype columns consolidate into a single block under the default
    # pandas constructor; the per-column manager keeps them separate. A silent
    # fallback to a consolidating constructor would fail this assertion.
    data = {"a": np.arange(128, dtype=np.float64), "b": np.arange(128, dtype=np.float64) + 1}
    path = tmp_path / "two_f8.cstore"
    with colstore.create(path) as writer:
        writer.write(data)
    with colstore.open(path) as dataset:
        for frame in (dataset.frame(), dataset.frame(copy=False)):
            assert len(frame._mgr.blocks) == frame.shape[1] == 2


def test_frame_copy_false_rejected_for_fancy(single_record_store):
    dataset, _ = single_record_store
    with pytest.raises(ValueError, match="copy=True"):
        dataset[np.array([3, 1, 2]), ["f8", "i2"]].frame(copy=False)


def test_dataset_frame_copy_false_rejected_across_files(tmp_path):
    # A multi-file dataset's columns are not contiguous in memory, so the
    # whole-store zero-copy frame must raise rather than copy (inherited from
    # dict(copy=False)).
    paths = []
    for i in range(2):
        path = tmp_path / f"part{i}.cstore"
        with colstore.create(path) as writer:
            writer.write({"a": np.arange(100, dtype=np.float64) + i})
        paths.append(path)
    with colstore.open(paths) as dataset:
        assert dataset.n_rows == 200  # a combined multi-file dataset
        with pytest.raises(ValueError, match="copy=True"):
            dataset.frame(copy=False)


def test_multi_record_store_raises_with_compact_hint(tmp_path):
    rng = np.random.default_rng(52)
    full = rng.standard_normal(2_000)
    path = tmp_path / "multi.cstore"
    with colstore.create(path) as writer:
        writer.write({"a": full[:800]})
        writer.write({"a": full[800:]})
    with opened(path) as dataset:
        with pytest.raises(ValueError, match="compact"):
            dataset["a"].array(copy=False)
        with pytest.raises(ValueError, match="compact"):
            dataset.dict(copy=False)


def test_copy_false_rejected_for_native_mask_route(tmp_path, monkeypatch):
    # A dense boolean mask on a multi-record store is exactly what the
    # mask-native kernel optimizes for copy=True reads; copy=False must
    # still refuse (multi-record stores are never zero-copy), so the native
    # route cannot weaken the memory guarantee.
    from colstore import config as config_mod

    rng = np.random.default_rng(71)
    total = 6_000
    full = rng.standard_normal(total)
    path = tmp_path / "multi_mask.cstore"
    with colstore.create(path) as writer:
        for offset in range(0, total, 500):  # 12 records
            writer.write({"a": full[offset : offset + 500]})
    monkeypatch.setattr(config_mod, "_mask_density_gate", 0.1)  # native route eligible
    with opened(path) as dataset:
        mask = rng.random(total) < 0.6  # dense: above the gate
        assert np.array_equal(dataset[mask, "a"].array(), full[mask])  # copy=True ok
        with pytest.raises(ValueError, match="copy=True"):
            dataset[mask, "a"].array(copy=False)


def test_compacted_store_supports_zero_copy(tmp_path):
    rng = np.random.default_rng(53)
    full = rng.standard_normal(3_000)
    path = tmp_path / "tocompact.cstore"
    with colstore.create(path) as writer:
        for lo in range(0, 3_000, 500):
            writer.write({"a": full[lo : lo + 500]})
    compacted = tmp_path / "compacted.cstore"
    colstore.compact(path, out=compacted, show_progress=False)
    with opened(compacted) as dataset:
        view = dataset["a"].array(copy=False)
        assert np.shares_memory(view, dataset._memmaps["a"])
        assert np.array_equal(view, full)


def test_non_native_dtype_raises(single_record_store, monkeypatch):
    dataset, _ = single_record_store
    monkeypatch.setattr(reader_mod, "_dtype_is_native", lambda dtype: False)
    with pytest.raises(ValueError, match="native byte order"):
        dataset["f8"].array(copy=False)


def test_view_survives_reader_close(tmp_path):
    rng = np.random.default_rng(54)
    data = rng.standard_normal(5_000)
    path = tmp_path / "lifetime.cstore"
    with colstore.create(path) as writer:
        writer.write({"a": data})
    dataset = colstore.open(path)
    view = dataset["a"].array(copy=False)
    dataset.close()
    # The view pins the mapping via .base; reads remain valid after close.
    assert np.array_equal(view, data)
    assert float(view.sum()) == pytest.approx(float(data.sum()))


def test_closed_reader_rejects_new_views(single_record_store, tmp_path):
    rng = np.random.default_rng(55)
    path = tmp_path / "closed.cstore"
    with colstore.create(path) as writer:
        writer.write({"a": rng.standard_normal(100)})
    dataset = colstore.open(path)
    column_view = dataset["a"]  # lazy view created while open
    dataset.close()
    with pytest.raises(ValueError, match="closed"):
        column_view.array(copy=False)


# -- view.edit(): carry the view's row + column selection into an editing frame --


def test_table_view_edit_carries_query_rows(small_store):
    cf = small_store.query("id < 10").edit()
    assert cf.n_rows == 10
    assert cf.dict()["id"].tolist() == list(range(10))


def test_table_view_edit_carries_fancy_rows(small_store):
    cf = small_store[[2, 4, 6]].edit()
    assert cf.dict()["id"].tolist() == [2, 4, 6]


def test_table_view_edit_carries_mask(small_store, small_frame):
    ids = small_frame["id"].to_numpy()
    mask = ids % 100 == 0
    cf = small_store[mask].edit()
    assert cf.n_rows == int(mask.sum())  # popcount of the selected rows
    assert cf.dict()["id"].tolist() == ids[mask].tolist()


def test_table_view_edit_carries_col_predicate(small_store):
    cf = small_store[colstore.col("id") >= 1020].edit()
    assert cf.dict()["id"].tolist() == [1020, 1021, 1022, 1023]


def test_table_view_edit_subrange_slice(small_store):
    cf = small_store[100:105].edit()
    assert cf.n_rows == 5
    assert cf.dict()["id"].tolist() == [100, 101, 102, 103, 104]


def test_table_view_edit_full_slice_stays_unfiltered(small_store):
    cf = small_store[:].edit()
    assert cf.n_rows == small_store.n_rows
    assert cf._rows is None  # a full range keeps the unfiltered streaming-write path


def test_table_view_edit_projects_columns(small_store):
    cf = small_store[["price", "qty"]].edit()
    assert cf.columns == ["price", "qty"]


def test_table_view_select_then_edit(small_store):
    cf = small_store.query("id < 5").select("id", "qty").edit()
    assert cf.columns == ["id", "qty"]
    assert cf.dict()["id"].tolist() == [0, 1, 2, 3, 4]


def test_column_view_edit(small_store):
    cf = small_store[10:13, "qty"].edit()
    assert cf.columns == ["qty"]
    assert cf.n_rows == 3


def test_view_edit_then_transform_and_write(small_store, tmp_path):
    cf = small_store[colstore.col("id") >= 1020].edit()
    cf = cf.assign(twice=cf["id"] * 2)
    reader = cf.write(tmp_path / "ve.cstore")
    try:
        assert reader.dict()["twice"].tolist() == [2040, 2042, 2044, 2046]
    finally:
        reader.close()


def test_table_view_edit_predicate_is_lazy(small_store):
    cf = small_store[colstore.col("id") >= 1020].edit()
    assert cf._rows is None  # the predicate is not resolved to indices at the seam
    assert len(cf._predicates) == 1  # it is carried as a pending where()
    assert cf.n_rows == 4  # n_rows resolves the scan on access
    assert cf.dict()["id"].tolist() == [1020, 1021, 1022, 1023]


def test_table_view_edit_predicate_matches_frame_where(small_store):
    pred = colstore.col("id") % 3 == 0
    via_view = small_store[pred].edit().dict()["id"].tolist()
    via_frame = small_store.edit().where(pred).dict()["id"].tolist()
    assert via_view == via_frame


def test_table_view_edit_predicate_in_report(small_store):
    report = small_store[colstore.col("id") >= 1020].edit().report()
    assert len(report) == 1
    assert report[0].entering == small_store.n_rows
    assert report[0].passing == 4


def test_view_edit_predicate_composes_with_where(small_store):
    cf = small_store[colstore.col("id") >= 1020].edit().where(colstore.col("id") < 1023)
    assert cf.dict()["id"].tolist() == [1020, 1021, 1022]
    assert len(cf.report()) == 2


def test_view_edit_predicate_references_projected_away_column(small_store, small_frame):
    qty = small_frame["qty"].to_numpy()
    ids = small_frame["id"].to_numpy()
    cf = small_store[colstore.col("qty") >= 500, "id"].edit()  # filter qty, keep only id
    assert cf.columns == ["id"]
    assert cf.dict()["id"].tolist() == ids[qty >= 500].tolist()


def test_view_edit_evaluate_then_edit_is_eager(small_store):
    cf = small_store[colstore.col("id") >= 1020].evaluate().edit()
    assert cf._rows is not None  # resolved to a fixed row set (concrete, not a pending predicate)
    assert len(cf._predicates) == 0  # nothing left pending
    assert cf.n_rows == 4
    assert cf.dict()["id"].tolist() == [1020, 1021, 1022, 1023]


# -- reader/dataset array(name) shortcut --


def test_reader_array_shortcut_matches_column_view_and_dict(small_store):
    np.testing.assert_array_equal(small_store.array("price"), small_store["price"].array())
    np.testing.assert_array_equal(small_store.array("id"), small_store.dict()["id"])


def test_reader_array_shortcut_zero_copy(single_record_store):
    dataset, data = single_record_store
    view = dataset.array("f8", copy=False)
    assert np.shares_memory(view, dataset._memmaps["f8"])
    assert not view.flags.writeable
    np.testing.assert_array_equal(view, data["f8"])
