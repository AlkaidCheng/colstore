"""JSON file format (``colstore.interop.json``), via pandas.

``orient`` (default ``"records"``) is pandas' JSON layout and must match on read
and write. A row subset / column subset materializes through a DataFrame.

JSON is a lossy text format: values round-trip but the exact dtype width may not
(``float32`` reads back as ``float64``); ``datetime64`` / ``timedelta64`` degrade
to integer epoch values; ``+/-inf`` become NaN (the JSON ``null`` token); and a
zero-row store does not round-trip through the ``records`` layout. Use a binary
format (Parquet / Feather / HDF5 / NPZ) when those matter.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar

from ._convert import columns_to_frame, frame_to_columns, store_columns
from .base import FileFormat, Selection

if TYPE_CHECKING:
    from ..reader import ColStoreReader


class JsonFormat(FileFormat):
    """JSON, via pandas ``read_json`` / ``to_json``."""

    name: ClassVar[str] = "json"
    extensions: ClassVar[frozenset[str]] = frozenset({".json"})

    def to_file(
        self, selection: Selection, dest: Any, *, orient: str = "records", **kwargs: Any
    ) -> None:
        """Write the selection to a JSON file (kwargs forwarded to ``DataFrame.to_json``)."""
        data = selection.gather_all()
        columns_to_frame(data).to_json(os.fspath(dest), orient=orient, **kwargs)

    def from_file(
        self,
        source: Any,
        dest: Any,
        *,
        orient: str = "records",
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> ColStoreReader:
        """Read a JSON file into a ``.cstore`` and open it (extra kwargs -> store)."""
        import pandas as pd

        # dtype=False keeps the JSON-encoded type (a quoted "1" stays a string
        # instead of being inferred to int), so a string column round-trips.
        frame = pd.read_json(os.fspath(source), orient=orient, dtype=False)
        if columns is not None:
            frame = frame[columns]
        return store_columns(frame_to_columns(frame), dest, **kwargs)
