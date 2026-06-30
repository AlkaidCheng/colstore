"""Apache Parquet file format (``colstore.interop.parquet``), via pyarrow.

Export builds a pyarrow Table from the selection (reusing the zero-copy Arrow
bridge) and writes it; import reads the Table and converts each column to a
fixed-width colstore column (strings widened, nested columns rejected).
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


class ParquetFormat(FileFormat):
    """Apache Parquet, via ``pyarrow.parquet``."""

    name: ClassVar[str] = "parquet"
    extensions: ClassVar[frozenset[str]] = frozenset({".parquet", ".pq"})

    def to_file(self, selection: Selection, dest: Any, **kwargs: Any) -> None:
        """Write the selection to a Parquet file (kwargs forwarded to pyarrow)."""
        import pyarrow.parquet

        pq: Any = pyarrow.parquet
        pq.write_table(selection_to_arrow_table(selection), os.fspath(dest), **kwargs)

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
        """Read a Parquet file into a ``.cstore`` and open it.

        With ``batch_size`` set and a stream-stable schema, the file is read in
        row-batches and streamed in bounded memory; otherwise it is read whole.
        """
        import pyarrow.parquet

        pq: Any = pyarrow.parquet
        if batch_size is not None:
            pf = pq.ParquetFile(os.fspath(source))
            stream_dtypes, reason = arrow_stream_dtypes(pf.schema_arrow, columns)
            if not reason:
                reason = _parquet_null_reason(pf.metadata, stream_dtypes)
            if not reason:
                rows = resolve_batch_rows(
                    batch_size, bytes_per_row=arrow_bytes_per_row(stream_dtypes)
                )
                batches = (
                    columns_from_arrow_batch(batch, stream_dtypes)
                    for batch in pf.iter_batches(
                        batch_size=rows or pf.metadata.num_rows, columns=columns
                    )
                )
                return stream_batches(
                    batches,
                    dest,
                    dtypes=dtypes,
                    total_rows=pf.metadata.num_rows,
                    compact=compact,
                    show_progress=show_progress,
                    desc=f"{os.fspath(dest)} <- parquet",
                )
            warn_whole_file(source, reason)
        table = pq.read_table(os.fspath(source), columns=columns)
        return store_columns(
            arrow_to_columns(table), dest, dtypes=dtypes, show_progress=show_progress, **kwargs
        )


def _parquet_null_reason(metadata: Any, dtypes: dict[str, Any]) -> str:
    """Empty if the selected columns provably have no nulls, else why streaming is unsafe.

    Nulls are read from the row-group statistics (no data decode). A column with any
    null -- or one whose statistics are absent -- falls back, since an all-null batch
    would store as float64 and clash with a typed batch.
    """
    index = {metadata.schema.column(i).name: i for i in range(metadata.num_columns)}
    for name in dtypes:
        position = index.get(name)
        if position is None:
            return f"column {name!r} is not a leaf column"
        for group in range(metadata.num_row_groups):
            statistics = metadata.row_group(group).column(position).statistics
            if statistics is None or not statistics.has_null_count:
                return f"column {name!r} has no null statistics"
            if statistics.null_count:
                return f"column {name!r} contains nulls"
    return ""
