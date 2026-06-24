"""Apache Parquet file format (``colstore.interop.parquet``), via pyarrow.

Export builds a pyarrow Table from the selection (reusing the zero-copy Arrow
bridge) and writes it; import reads the Table and converts each column to a
fixed-width colstore column (strings widened, nested columns rejected).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar

from ._convert import arrow_to_columns, selection_to_arrow_table
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
        self, source: Any, dest: Any, *, columns: list[str] | None = None, **kwargs: Any
    ) -> ColStoreReader:
        """Read a Parquet file into a ``.cstore`` and open it (extra kwargs -> store)."""
        import pyarrow.parquet

        from .. import api

        pq: Any = pyarrow.parquet
        table = pq.read_table(os.fspath(source), columns=columns)
        kwargs.setdefault("show_progress", False)
        return api.store(arrow_to_columns(table), dest, **kwargs)
