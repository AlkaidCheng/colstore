"""Shared conversions between colstore columns and the richer file formats.

Parquet, Feather, JSON, and HDF5 can carry types a ``.cstore`` cannot: a
variable-length string column is **coerced to fixed-width** (``U``); a nested
(list / struct / object-of-objects) column is **rejected** with a clear error.
These helpers centralize that policy and the Arrow / pandas bridges so every
format applies it identically.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# NumPy dtype kinds a .cstore column stores directly.
_FIXED_KINDS = frozenset("fiubMmSU")


def _is_null(value: Any) -> bool:
    """Whether ``value`` is a null colstore cannot store (``None`` or a NaN float)."""
    return value is None or (isinstance(value, float) and value != value)


def _coerce_object_strings(name: str, array: Any) -> np.ndarray[Any, Any]:
    """Coerce an object array of *strings* to fixed-width ``U`` / ``S``.

    The whole array is scanned (not a sample), so the gate is exact: a null
    (``None`` / NaN), a nested value (list / dict / array), or any non-string
    object (a number, ``Decimal``, ``bool``) has no fixed-width string form and
    raises, rather than being silently stringified.
    """
    values = list(np.asarray(array, dtype=object).ravel())
    if any(_is_null(value) for value in values):
        raise TypeError(f"column {name!r} contains null values; colstore has no null support.")
    all_bytes = bool(values) and all(isinstance(value, bytes) for value in values)
    all_str = bool(values) and all(isinstance(value, str) for value in values)
    if values and not (all_bytes or all_str):
        raise TypeError(
            f"column {name!r} is not a fixed-width string column; colstore stores only "
            f"fixed-width columns (nested, mixed, or non-string object values cannot be stored)."
        )
    return np.asarray(array.astype("S" if all_bytes else "U"))


def storable_column(name: str, array: Any) -> np.ndarray[Any, Any]:
    """Return ``array`` as a fixed-width colstore column, coercing or rejecting."""
    arr = np.ascontiguousarray(array)
    if arr.dtype.kind in _FIXED_KINDS:
        return arr
    if arr.dtype.kind == "O":
        return _coerce_object_strings(name, arr)
    raise TypeError(f"column {name!r} has unsupported dtype {arr.dtype}; cannot store it.")


def arrow_to_columns(table: Any) -> dict[str, np.ndarray[Any, Any]]:
    """Convert a pyarrow Table to colstore columns (coerce strings, reject nulls/nested)."""
    import pyarrow as pa

    columns: dict[str, np.ndarray[Any, Any]] = {}
    for field in table.schema:
        name = field.name
        kind = field.type
        if pa.types.is_nested(kind):
            raise TypeError(
                f"column {name!r} has nested Arrow type {kind}; colstore stores only "
                f"fixed-width columns."
            )
        if pa.types.is_null(kind):
            raise TypeError(f"column {name!r} is all-null; colstore has no null support.")
        if pa.types.is_timestamp(kind) and kind.tz is not None:
            raise TypeError(
                f"column {name!r} is a timezone-aware timestamp; colstore has no "
                f"timezone-aware type (convert to UTC-naive first)."
            )
        chunked = table.column(name)
        if chunked.null_count:
            raise TypeError(f"column {name!r} contains null values; colstore has no null support.")
        if pa.types.is_string(kind) or pa.types.is_large_string(kind):
            columns[name] = np.array(chunked.to_pylist(), dtype="U")
        else:
            columns[name] = storable_column(name, chunked.to_numpy(zero_copy_only=False))
    return columns


def selection_to_arrow_table(selection: Any) -> Any:
    """Build a pyarrow Table from an interop ``Selection`` (one chunk per segment)."""
    import pyarrow as pa

    from .arrow import ArrowFormat

    obj = ArrowFormat().to_object(selection)
    if isinstance(obj, pa.Table):
        return obj
    return pa.table({selection.columns[0]: obj})


def frame_to_columns(frame: Any) -> dict[str, np.ndarray[Any, Any]]:
    """Convert a pandas DataFrame to colstore columns (coerce strings, reject nulls/nested)."""
    columns: dict[str, np.ndarray[Any, Any]] = {}
    for name in frame.columns:
        series = frame[name]
        if series.isna().any():
            raise TypeError(
                f"column {str(name)!r} contains null values; colstore has no null support."
            )
        columns[str(name)] = storable_column(str(name), series.to_numpy())
    return columns


def columns_to_frame(columns: dict[str, np.ndarray[Any, Any]]) -> Any:
    """Build a pandas DataFrame from a column mapping."""
    import pandas as pd

    return pd.DataFrame(dict(columns))
