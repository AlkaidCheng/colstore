"""Tests that batch_size only affects progress reporting, not output bytes."""

from __future__ import annotations

import numpy as np
import pytest

from colstore import ColStore
from colstore import format as fmt


@pytest.mark.parametrize("batch_size", [None, -1, 0])
def test_unbatched_write_matches_batched(tmp_path, batch_size):
    columns = {"x": np.arange(1000, dtype=np.float32)}
    batched = tmp_path / "batched.cstore"
    unbatched = tmp_path / "unbatched.cstore"
    fmt.write_dataset(columns, batched, batch_size=100, show_progress=False)
    fmt.write_dataset(columns, unbatched, batch_size=batch_size, show_progress=False)
    assert batched.read_bytes() == unbatched.read_bytes()


def test_factory_accepts_none_batch_size(tmp_path):
    store = ColStore.from_dict(
        {"x": np.arange(10, dtype=np.int32)},
        tmp_path / "f.cstore",
        batch_size=None,
        show_progress=False,
    )
    assert store.n_rows == 10
    store.close()
