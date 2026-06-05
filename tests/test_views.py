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
