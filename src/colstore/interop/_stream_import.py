"""Bounded-memory streaming import: read a foreign file in row-batches and stream
each batch into the ``.cstore`` writer, instead of materializing the whole file.

A foreign source can only stream when its schema is *stream-stable*: every column
maps to a fixed-width numeric / bool / temporal NumPy dtype with no nulls. The
per-batch converters infer a string column's width and the all-null fallback dtype
(:func:`._convert._all_null_column`) from each batch's own contents, and
:func:`._streaming.write_column_batches` locks the schema on the first batch and
validates every later one -- so a variable-width string or a column that is null in
one batch but valued in another would drift and be rejected mid-stream. When a file
is not stream-stable the importer falls back to a whole-file read and warns, so
``batch_size`` is best-effort, never a silent no-op or a mid-stream failure.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._types import Columns, StrPath
from ._convert import _reject_nulls, apply_dtype_overrides
from ._streaming import write_column_batches

if TYPE_CHECKING:
    from ..reader import ColStoreReader


@dataclass
class StreamPlan:
    """A streamable import: a lazy iterator of column batches and the total row count.

    Returned by :meth:`FileFormat.stream_import` for a file that can be read in
    row-batches; ``total_rows`` is used only for the progress bar (``None`` if unknown).
    Every batch must carry the same columns and dtypes (the writer locks the schema on
    the first), already converted to fixed-width NumPy columns.
    """

    batches: Iterator[Columns]
    total_rows: int | None = None


def warn_whole_file(source: Any, reason: str) -> None:
    """Warn that ``batch_size`` could not stream ``source`` and it was read whole."""
    warnings.warn(
        f"batch_size: {os.fspath(source)} was converted whole-file because it cannot "
        f"stream ({reason}).",
        RuntimeWarning,
        stacklevel=4,
    )


def write_stream(
    plan: StreamPlan,
    dest: StrPath,
    *,
    dtypes: dict[str, Any] | None,
    mode: str,
    compact: bool,
    show_progress: bool,
    desc: str | None,
) -> ColStoreReader:
    """Stream a :class:`StreamPlan` to ``dest``, applying ``dtypes`` per batch."""

    def overridden() -> Iterator[Columns]:
        for batch in plan.batches:
            apply_dtype_overrides(batch, dtypes)
            yield batch

    return write_column_batches(
        overridden(),
        dest,
        mode=mode,
        total_rows=plan.total_rows,
        compact=compact,
        show_progress=show_progress,
        desc=desc,
    )


def arrow_stream_dtypes(
    schema: Any, columns: list[str] | None
) -> tuple[dict[str, np.dtype[Any]], str]:
    """Pin each selected column's NumPy dtype from an Arrow ``schema``.

    Returns ``(dtypes, "")`` when every selected column is a fixed-width numeric / bool
    / naive-temporal type, or ``({}, reason)`` when one is not (a variable-width string,
    a nested / binary / null type, or a timezone-aware timestamp) so the caller falls
    back to a whole-file read. A pinned dtype keeps every batch's schema identical.
    """
    import pyarrow as pa

    names = [field.name for field in schema] if columns is None else columns
    fields = {field.name: field.type for field in schema}
    dtypes: dict[str, np.dtype[Any]] = {}
    for name in names:
        kind = fields.get(name)
        if kind is None:
            return {}, f"column {name!r} is not in the file"
        if pa.types.is_string(kind) or pa.types.is_large_string(kind):
            return {}, f"column {name!r} is a variable-width string"
        if pa.types.is_timestamp(kind) and kind.tz is not None:
            return {}, f"column {name!r} is a timezone-aware timestamp"
        try:
            dtype = np.dtype(kind.to_pandas_dtype())
        except (NotImplementedError, TypeError):
            return {}, f"column {name!r} has non-fixed-width type {kind}"
        if dtype.kind not in "iufbMm":
            return {}, f"column {name!r} has non-fixed-width type {kind}"
        dtypes[name] = dtype
    return dtypes, ""


def arrow_bytes_per_row(dtypes: dict[str, np.dtype[Any]]) -> int:
    """Bytes one row occupies, for turning a ``"256 MiB"`` budget into rows."""
    return max(1, sum(dtype.itemsize for dtype in dtypes.values()))


def columns_from_arrow_batch(batch: Any, dtypes: dict[str, np.dtype[Any]]) -> Columns:
    """Convert one Arrow ``RecordBatch`` to a column dict with the pinned dtypes.

    The dtypes are fixed up front (see :func:`arrow_stream_dtypes`), so the batch is
    cast to them directly -- no per-batch width or null inference that could drift. The
    cheap ``null_count`` guard rejects a null the up-front gate could not prove absent
    (wrong Parquet statistics, or a feather field declared non-nullable that holds nulls),
    matching the whole-file null policy instead of casting a null to a garbage value.
    """
    out: Columns = {}
    for name, dtype in dtypes.items():
        column = batch.column(name)
        if column.null_count:
            _reject_nulls(name)
        array = column.to_numpy(zero_copy_only=False)
        out[name] = np.ascontiguousarray(array.astype(dtype, copy=False))
    return out
