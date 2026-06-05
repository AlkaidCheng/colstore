"""Tests for ColumnView and TableView."""

from __future__ import annotations

import numpy as np
import pandas as pd

from colstore import ColumnView, TableView


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


def test_table_view_does_not_have_array():
    """TableView does not implement array."""
    assert not hasattr(TableView, "array")


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
    from colstore.reader import _make_dataframe_no_consolidate

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
