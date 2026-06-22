"""Helpers for building pandas objects from colstore column data."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd


def _make_dataframe_no_consolidate(columns: dict[str, NDArray[Any]]) -> pd.DataFrame:
    """Build a DataFrame from a column dict with one block per column.

    Equivalent to ``pd.DataFrame(columns)`` through the public DataFrame API, but
    skips the eager dtype-block consolidation -- wasted work for the
    read-and-pass-along path. Falls back to ``pd.DataFrame(columns)``, emitting a
    ``UserWarning``, when the pandas internals it builds on are unavailable.
    """
    import pandas as pd

    if not columns:
        return pd.DataFrame(columns)

    try:
        from pandas import Index, RangeIndex
        from pandas.core.internals.managers import (
            create_block_manager_from_column_arrays,
        )

        arrays = list(columns.values())
        n_rows = arrays[0].shape[0]
        block_manager = create_block_manager_from_column_arrays(
            arrays,
            axes=[Index(list(columns)), RangeIndex(n_rows)],
            consolidate=False,
            refs=[None] * len(arrays),
        )
        return pd.DataFrame._from_mgr(block_manager, axes=block_manager.axes)
    except (ImportError, AttributeError, TypeError) as exc:
        # A pandas internal API drifted. ValueError is deliberately left to
        # propagate: it signals a data problem the fallback would raise too.
        warnings.warn(
            f"colstore.frame() optimized construction unavailable on this "
            f"pandas ({pd.__version__}); falling back to pd.DataFrame(dict). "
            f"The result is functionally identical but slower for whole-store "
            f"materialization. Cause: {type(exc).__name__}: {exc}",
            stacklevel=2,
        )
        return pd.DataFrame(columns)
