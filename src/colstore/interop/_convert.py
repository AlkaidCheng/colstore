"""Shared conversions between colstore columns and the richer file formats.

Parquet, Feather, JSON, and HDF5 can carry types a ``.cstore`` cannot: a
variable-length string column is **coerced to fixed-width** (``U``); a nested
(list / struct / object-of-objects) column is **rejected** with a clear error.
These helpers centralize that policy and the Arrow / pandas bridges so every
format applies it identically.

A genuine null -- an out-of-band missing value (an object ``None``, a masked
nullable dtype, or an Arrow validity bit) -- mixed with real values is rejected,
but a NaN / NaT bit pattern in a native numeric column is a valid value and is
stored. A column whose values are *all* null carries no data, so it is stored as an
all-NaN ``float64`` column rather than rejected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np

if TYPE_CHECKING:
    from ..reader import ColStoreReader

# NumPy dtype kinds a .cstore column stores directly.
_FIXED_KINDS = frozenset("fiubMmSU")


def _is_null(value: Any) -> bool:
    """Whether an object-array element has no fixed-width string form (``None`` or NaN).

    Used only when coercing an *object* column to fixed-width strings, where a ``None``
    or a NaN float is a missing string with no fixed-width representation. A NaN in a
    native float column is a valid value and IS stored -- see :func:`frame_to_columns`.
    """
    return value is None or (isinstance(value, float) and value != value)


def _reject_nulls(name: object) -> NoReturn:
    """Raise the standard 'no null support' ``TypeError`` for column ``name``."""
    raise TypeError(f"column {name!r} contains null values; colstore has no null support.")


def _all_null_column(length: int) -> np.ndarray[Any, Any]:
    """A fully-null column as fixed-width ``float64`` NaN.

    A column whose every value is null carries no data and no type to preserve, so it
    stores as the canonical in-band "missing" form -- ``float64`` NaN, which round-trips
    losslessly -- instead of being rejected for having out-of-band nulls. A column that
    mixes nulls with real values has no fixed-width form and still raises.
    """
    return np.full(length, np.nan, dtype=np.float64)


# Target kinds whose zero value (``0`` / ``False`` / ``""``) is the natural stand-in for
# a NaN (missing) source value: a NaN has no representation there, and ``float -> int``
# of a NaN is undefined in NumPy, so the position is filled explicitly rather than cast.
# Float / complex targets keep the NaN, and datetime / timedelta targets get NaT, both
# through ``astype``.
_MISSING_FILL_KINDS = frozenset("biuSU")


def apply_dtype_overrides(
    columns: dict[str, np.ndarray[Any, Any]], dtypes: dict[str, Any] | None
) -> dict[str, np.ndarray[Any, Any]]:
    """Coerce named columns to a target dtype after the format-specific conversion.

    Each ``name -> dtype`` entry casts ``columns[name]`` with ``astype``. A NaN in a float
    source is treated as missing: cast to a bool / integer / string target it becomes that
    target's empty value (``False`` / ``0`` / ``""``) rather than an undefined ``astype``
    result, while a float target keeps the NaN and a datetime / timedelta target gets NaT.
    Real values follow NumPy ``astype`` rules, so a narrower target may truncate or overflow
    them without error. Raises ``KeyError`` for a name the file lacks, so a typo in an
    override is caught rather than silently ignored.
    """
    for name, spec in (dtypes or {}).items():
        if name not in columns:
            raise KeyError(
                f"dtype override names unknown column {name!r}; file has {sorted(columns)}."
            )
        target = np.dtype(spec)
        column = columns[name]
        if column.dtype == target:
            continue
        if column.dtype.kind == "f" and target.kind in _MISSING_FILL_KINDS:
            mask = np.isnan(column)
            out = np.zeros(len(column), dtype=target)
            if not mask.all():
                out[~mask] = column[~mask].astype(target)
            columns[name] = out
        else:
            columns[name] = np.ascontiguousarray(column.astype(target))
    return columns


def store_columns(
    columns: dict[str, Any], dest: Any, *, dtypes: dict[str, Any] | None = None, **kwargs: Any
) -> ColStoreReader:
    """Store a column dict to ``dest`` via the top-level ``store()``, defaulting progress off.

    ``dtypes`` coerces named columns to a target dtype first (see
    :func:`apply_dtype_overrides`).
    """
    from .. import api

    apply_dtype_overrides(columns, dtypes)
    kwargs.setdefault("show_progress", False)
    return api.store(columns, dest, **kwargs)


def _coerce_object_strings(name: str, array: Any) -> np.ndarray[Any, Any]:
    """Coerce an object array of *strings* to fixed-width ``U`` / ``S``.

    The whole array is scanned (not a sample), so the gate is exact: a null
    (``None`` / NaN), a nested value (list / dict / array), or any non-string
    object (a number, ``Decimal``, ``bool``) has no fixed-width string form and
    raises, rather than being silently stringified.
    """
    values = list(np.asarray(array, dtype=object).ravel())
    if any(_is_null(value) for value in values):
        _reject_nulls(name)
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
        chunked = table.column(name)
        n = len(chunked)
        if pa.types.is_null(kind) or (n > 0 and chunked.null_count == n):
            columns[name] = _all_null_column(n)
            continue
        if pa.types.is_timestamp(kind) and kind.tz is not None:
            raise TypeError(
                f"column {name!r} is a timezone-aware timestamp; colstore has no "
                f"timezone-aware type (convert to UTC-naive first)."
            )
        if chunked.null_count:
            _reject_nulls(name)
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
        array = series.to_numpy()
        # NaN / NaT in a native float / complex / datetime / timedelta column is a valid
        # bit pattern that round-trips losslessly; only an out-of-band null -- an object
        # None or a masked nullable/extension dtype, which converts to an object array --
        # has no fixed-width form.
        if array.dtype.kind not in "fcmM" and series.isna().any():
            if series.isna().all():
                columns[str(name)] = _all_null_column(len(series))
                continue
            _reject_nulls(str(name))
        columns[str(name)] = storable_column(str(name), array)
    return columns


def columns_to_frame(columns: dict[str, np.ndarray[Any, Any]]) -> Any:
    """Build a pandas DataFrame from a column mapping."""
    import pandas as pd

    return pd.DataFrame(dict(columns))
