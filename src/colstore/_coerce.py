"""Coerce a supported input (dict, structured ndarray, pandas DataFrame) to columns.

Shared by :func:`colstore.store` and the shard writer so that neither reaches into
the other's facade to normalize an in-memory input to a uniform
``dict[name, ndarray]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def coerce_to_columns(data: Any) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    """Dispatch on the input type and return a uniform ``dict[name, ndarray]``."""
    if isinstance(data, dict):
        return {str(name): np.ascontiguousarray(array) for name, array in data.items()}
    if isinstance(data, np.ndarray):
        if data.dtype.names is None:
            raise TypeError(
                "store() received a plain ndarray; pass {name: array} as a dict "
                "(or a structured ndarray with named fields)."
            )
        return {name: np.ascontiguousarray(data[name]) for name in data.dtype.names}
    if _is_pandas_dataframe(data):
        return _dataframe_to_columns(data)
    raise TypeError(
        f"store() does not know how to handle {type(data).__name__}. "
        f"Expected dict[str, ndarray], structured ndarray, or pandas DataFrame."
    )


def _is_pandas_dataframe(data: Any) -> bool:
    """Return whether ``data`` looks like a pandas DataFrame (duck-typed)."""
    # Duck-typed (not isinstance) to avoid importing pandas at module load.
    return (
        hasattr(data, "columns")
        and hasattr(data, "to_numpy")
        and type(data).__name__ == "DataFrame"
    )


def _dataframe_to_columns(frame: Any) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    """Convert a pandas DataFrame to a column-name -> ndarray dict.

    Object-dtype columns are rejected up front with a clearer message than
    the writer's generic "unsupported dtype" error.
    """
    columns: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
    for column_name in frame.columns:
        series = frame[column_name]
        array = series.to_numpy()
        if array.dtype.kind == "O":
            raise TypeError(
                f"Column {column_name!r} (pandas dtype {series.dtype}) converts to "
                f"an object array and cannot be stored. Cast it to a fixed-size NumPy "
                f"dtype (e.g. float64, int64, or a fixed-width string like 'S16') first."
            )
        columns[str(column_name)] = array
    return columns
