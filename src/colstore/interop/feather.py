"""Feather (Arrow IPC) file format (``colstore.interop.feather``), via pyarrow.

The on-disk container is the Arrow IPC / Feather v2 format.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar

from .._sizes import resolve_batch_rows
from ._convert import arrow_to_columns, selection_to_arrow_table, store_columns
from ._stream_import import (
    arrow_bytes_per_row,
    arrow_stream_dtypes,
    columns_from_arrow_batch,
    stream_batches,
    warn_whole_file,
)
from .base import FileFormat, Selection

if TYPE_CHECKING:
    from ..reader import ColStoreReader


class FeatherFormat(FileFormat):
    """Feather / Arrow IPC, via ``pyarrow.feather``."""

    name: ClassVar[str] = "feather"
    extensions: ClassVar[frozenset[str]] = frozenset({".feather"})

    def to_file(self, selection: Selection, dest: Any, **kwargs: Any) -> None:
        """Write the selection to a Feather file (kwargs forwarded to pyarrow)."""
        import pyarrow.feather

        feather: Any = pyarrow.feather
        feather.write_feather(selection_to_arrow_table(selection), os.fspath(dest), **kwargs)

    def from_file(
        self,
        source: Any,
        dest: Any,
        *,
        columns: list[str] | None = None,
        batch_size: int | str | None = None,
        compact: bool = True,
        dtypes: dict[str, Any] | None = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> ColStoreReader:
        """Read a Feather file into a ``.cstore`` and open it.

        With ``batch_size`` set and a stream-stable, non-nullable schema, the file is
        streamed at its Arrow record-batch granularity (coarsened toward the requested
        size); otherwise it is read whole.
        """
        import pyarrow as pa
        import pyarrow.feather

        feather: Any = pyarrow.feather
        if batch_size is not None:
            reader = pa.ipc.open_file(os.fspath(source))
            stream_dtypes, reason = arrow_stream_dtypes(reader.schema, columns)
            if not reason:
                reason = _feather_null_reason(reader.schema, stream_dtypes)
            if not reason:
                rows = resolve_batch_rows(
                    batch_size, bytes_per_row=arrow_bytes_per_row(stream_dtypes)
                )
                return stream_batches(
                    _feather_batches(reader, columns, stream_dtypes, rows),
                    dest,
                    dtypes=dtypes,
                    total_rows=None,
                    compact=compact,
                    show_progress=show_progress,
                    desc=f"{os.fspath(dest)} <- feather",
                )
            warn_whole_file(source, reason)
        table = feather.read_table(os.fspath(source), columns=columns)
        return store_columns(
            arrow_to_columns(table), dest, dtypes=dtypes, show_progress=show_progress, **kwargs
        )


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
