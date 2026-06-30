"""Apache Parquet file format (``colstore.interop.parquet``), via pyarrow.

Export builds a pyarrow Table from the selection (reusing the zero-copy Arrow
bridge) and writes it; import reads the Table and converts each column to a
fixed-width colstore column (strings widened, nested columns rejected).
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
from .base import FileFormat, Selection


class ParquetFormat(FileFormat):
    """Apache Parquet, via ``pyarrow.parquet``."""

    name: ClassVar[str] = "parquet"
    extensions: ClassVar[frozenset[str]] = frozenset({".parquet", ".pq"})

    def to_file(self, selection: Selection, dest: Any, **kwargs: Any) -> None:
        """Write the selection to a Parquet file (kwargs forwarded to pyarrow)."""
        import pyarrow.parquet

        pq: Any = pyarrow.parquet
        pq.write_table(selection_to_arrow_table(selection), os.fspath(dest), **kwargs)

    def read_columns(
        self, source: Any, *, columns: list[str] | None = None, **kwargs: Any
    ) -> Columns:
        """Read the whole Parquet file into a column mapping."""
        import pyarrow.parquet

        pq: Any = pyarrow.parquet
        return arrow_to_columns(pq.read_table(os.fspath(source), columns=columns))

    def stream_import(
        self,
        source: Any,
        *,
        columns: list[str] | None = None,
        batch_size: int | str | None,
        **kwargs: Any,
    ) -> StreamPlan | str:
        """Stream the file in row-batches when its schema is stream-stable with no nulls."""
        import pyarrow.parquet

        pq: Any = pyarrow.parquet
        pf = pq.ParquetFile(os.fspath(source))
        stream_dtypes, reason = arrow_stream_dtypes(pf.schema_arrow, columns)
        if reason:
            return reason
        reason = _parquet_null_reason(pf.metadata, stream_dtypes)
        if reason:
            return reason
        rows = resolve_batch_rows(batch_size, bytes_per_row=arrow_bytes_per_row(stream_dtypes))
        batches = (
            columns_from_arrow_batch(batch, stream_dtypes)
            for batch in pf.iter_batches(batch_size=rows or pf.metadata.num_rows, columns=columns)
        )
        return StreamPlan(batches, pf.metadata.num_rows)


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
