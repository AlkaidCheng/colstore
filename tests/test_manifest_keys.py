"""Tests for reserved per-column manifest keys."""

from __future__ import annotations

import numpy as np

from colstore import format as fmt


def test_manifest_has_reserved_keys(tmp_path):
    path = tmp_path / "k.cstore"
    fmt.write_dataset(
        {"x": np.arange(4, dtype=np.float64)},
        path,
        batch_size=1000,
        show_progress=False,
    )
    manifest, _ = fmt.read_header(path)
    column = manifest["columns"][0]
    assert column["encoding"] == "raw"
    assert column["nullable"] is False
