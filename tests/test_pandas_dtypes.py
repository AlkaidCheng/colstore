"""Tests for the from_dataframe error on object-backed pandas dtypes."""

from __future__ import annotations

import pandas as pd
import pytest

from colstore import ColStore


def test_pandas_string_extension_dtype_message(tmp_path):
    frame = pd.DataFrame({"s": pd.array(["x", "y"], dtype="string")})
    with pytest.raises(TypeError, match="object array"):
        ColStore.from_dataframe(frame, tmp_path / "s.cstore", show_progress=False)
