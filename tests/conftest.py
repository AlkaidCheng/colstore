"""Shared pytest fixtures for the colstore test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from colstore import ColStore


@pytest.fixture
def small_frame() -> pd.DataFrame:
    """Small DataFrame mixing several fixed-size dtypes."""
    rng = np.random.default_rng(0)
    n_rows = 1024
    return pd.DataFrame(
        {
            "price": rng.standard_normal(n_rows).astype(np.float32),
            "qty": rng.integers(0, 1000, size=n_rows, dtype=np.int32),
            "vol": rng.standard_normal(n_rows).astype(np.float64),
            "flag": rng.integers(0, 2, size=n_rows, dtype=np.uint8),
            "id": np.arange(n_rows, dtype=np.int64),
        }
    )


@pytest.fixture
def small_store(tmp_path, small_frame) -> ColStore:
    """A small opened ColStore built from `small_frame`."""
    store_path = tmp_path / "small.cstore"
    store = ColStore.from_dataframe(small_frame, store_path, show_progress=False)
    yield store
    store.close()


@pytest.fixture
def medium_frame() -> pd.DataFrame:
    """A medium-sized DataFrame that exercises the fancy-index path."""
    rng = np.random.default_rng(42)
    n_rows = 50_000
    return pd.DataFrame(
        {
            "a": rng.standard_normal(n_rows).astype(np.float32),
            "b": rng.standard_normal(n_rows).astype(np.float64),
            "c": rng.integers(-1_000, 1_000, size=n_rows, dtype=np.int32),
            "d": rng.integers(0, 2**16, size=n_rows, dtype=np.uint16),
        }
    )


@pytest.fixture
def medium_store(tmp_path, medium_frame) -> ColStore:
    store_path = tmp_path / "medium.cstore"
    store = ColStore.from_dataframe(medium_frame, store_path, show_progress=False)
    yield store
    store.close()
