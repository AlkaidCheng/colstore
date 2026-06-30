"""Feather (Arrow IPC) file format (``colstore.interop.feather``), via pyarrow.

The on-disk container is the Arrow IPC / Feather v2 format.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

from .._sizes import resolve_batch_rows
from .._types import Columns
from ._convert import arrow_to_columns, selection_to_arrow_table
from ._stream_import import (
    StreamPlan,
    arrow_bytes_per_row,
    arrow_stream_dtypes,
    columns_from_arrow_batch,
)
from .arrow import columns_to_record_batch
from .base import FileFormat, Selection


class FeatherFormat(FileFormat):
    """Feather / Arrow IPC, via ``pyarrow.feather``."""

    name: ClassVar[str] = "feather"
    extensions: ClassVar[frozenset[str]] = frozenset({".feather"})

    def to_file(self, selection: Selection, dest: Any, **kwargs: Any) -> None:
        """Write the selection to a Feather file (kwargs forwarded to pyarrow)."""
        import pyarrow.feather

        feather: Any = pyarrow.feather
        feather.write_feather(selection_to_arrow_table(selection), os.fspath(dest), **kwargs)

    def stream_export(
        self, selection: Selection, dest: Any, *, batch_size: int | str | None, **kwargs: Any
    ) -> str | None:
        """Stream the selection to Feather, one Arrow record batch at a time.

        The incremental IPC writer carries no ``write_feather`` options, so a call that
        passes any (e.g. ``compression``) declines streaming and is written whole instead.
        """
        if kwargs:
            return f"streaming export cannot carry the write option(s) {sorted(kwargs)}"
        import pyarrow as pa

        writer = None
        try:
            for batch in selection.iter_batches(batch_size):
                record_batch = columns_to_record_batch(batch)
                if writer is None:
                    writer = pa.ipc.new_file(os.fspath(dest), record_batch.schema)
                writer.write_batch(record_batch)
        finally:
            if writer is not None:
                writer.close()
        return None

    def read_columns(
        self, source: Any, *, columns: list[str] | None = None, **kwargs: Any
    ) -> Columns:
        """Read the whole Feather file into a column mapping."""
        import pyarrow.feather

        feather: Any = pyarrow.feather
        return arrow_to_columns(feather.read_table(os.fspath(source), columns=columns))

    def stream_import(
        self,
        source: Any,
        *,
        columns: list[str] | None = None,
        batch_size: int | str | None,
        **kwargs: Any,
    ) -> StreamPlan | str:
        """Stream at the Arrow record-batch granularity for a stream-stable, non-nullable schema."""
        import pyarrow as pa

        reader = pa.ipc.open_file(os.fspath(source))
        stream_dtypes, reason = arrow_stream_dtypes(reader.schema, columns)
        if reason:
            return reason
        reason = _feather_null_reason(reader.schema, stream_dtypes)
        if reason:
            return reason
        rows = resolve_batch_rows(batch_size, bytes_per_row=arrow_bytes_per_row(stream_dtypes))
        return StreamPlan(_feather_batches(reader, columns, stream_dtypes, rows))


def _feather_null_reason(schema: Any, dtypes: dict[str, Any]) -> str:
    """Empty if the selected columns are declared non-nullable, else why streaming is unsafe.

    Arrow IPC has no cheap per-column null count, so a nullable field falls back to a
    whole-file read (which handles or rejects nulls), rather than risk a null reaching
    the fixed-width cast.
    """
    for name in dtypes:
        if schema.field(name).nullable:
            return f"column {name!r} may contain nulls"
    return ""


def _feather_batches(
    reader: Any, columns: list[str] | None, dtypes: dict[str, Any], rows: int | None
) -> Any:
    """Yield column dicts from the file's record batches, coarsened toward ``rows``."""
    import pyarrow as pa

    group: list[Any] = []
    group_rows = 0
    for index in range(reader.num_record_batches):
        batch = reader.get_batch(index)
        if columns is not None:
            batch = batch.select(columns)
        group.append(batch)
        group_rows += batch.num_rows
        if rows is not None and group_rows >= rows:
            yield columns_from_arrow_batch(pa.Table.from_batches(group), dtypes)
            group, group_rows = [], 0
    if group or reader.num_record_batches == 0:
        yield columns_from_arrow_batch(pa.Table.from_batches(group, schema=reader.schema), dtypes)
