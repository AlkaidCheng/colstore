"""Tests for ColStore basics: properties, lifecycle, container protocol."""

from __future__ import annotations

import numpy as np
import pytest

from colstore import ColStore


def test_shape_matches_dataframe(small_store, small_frame):
    assert small_store.shape == (small_frame.shape[0], small_frame.shape[1])


def test_len_returns_n_rows(small_store):
    assert len(small_store) == small_store.n_rows


def test_contains_returns_true_for_existing_column(small_store):
    assert "price" in small_store
    assert "missing" not in small_store
    assert 42 not in small_store  # non-string keys also yield False


def test_iter_yields_columns_in_order(small_store):
    assert list(small_store) == small_store.columns


def test_dtypes_match_source(small_store):
    assert small_store.dtypes["price"] == np.float32
    assert small_store.dtypes["qty"] == np.int32
    assert small_store.dtypes["flag"] == np.uint8
    assert small_store.dtypes["id"] == np.int64


def test_repr_includes_path_and_shape(small_store):
    repr_text = repr(small_store)
    assert "ColStore" in repr_text
    assert "shape" in repr_text


def test_close_releases_memmaps(small_store):
    small_store.close()
    with pytest.raises(ValueError, match="closed"):
        small_store["price"].to_array()


def test_close_is_idempotent(small_store):
    small_store.close()
    small_store.close()  # second call must not raise


def test_context_manager_closes_on_exit(tmp_path, small_frame):
    path = tmp_path / "ctx.cstore"
    with ColStore.from_dataframe(small_frame, path, show_progress=False) as store:
        _ = store["price"].to_array()
    with pytest.raises(ValueError, match="closed"):
        store["price"].to_array()


def test_max_workers_property_uses_override_when_set(tmp_path, small_frame):
    path = tmp_path / "overr.cstore"
    with ColStore.from_dataframe(small_frame, path, show_progress=False, max_workers=3) as store:
        assert store.max_workers == 3


def test_backend_property_reflects_constructor_choice(tmp_path, small_frame):
    path = tmp_path / "back.cstore"
    with ColStore.from_dataframe(small_frame, path, show_progress=False, backend="numpy") as store:
        assert store.backend == "numpy"


def test_store_reopen_recovers_same_data(tmp_path, small_frame):
    """Closing and reopening the same file returns identical data."""
    path = tmp_path / "reopen.cstore"
    ColStore.from_dataframe(small_frame, path, show_progress=False).close()
    with ColStore(path) as store:
        assert store.shape == (len(small_frame), len(small_frame.columns))
        first = store["price"].to_array()
    with ColStore(path) as store_reopened:
        second = store_reopened["price"].to_array()
    assert np.array_equal(first, second)
