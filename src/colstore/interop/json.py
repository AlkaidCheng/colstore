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
from typing import Any, ClassVar

from .._types import Columns
from ._convert import columns_to_frame, frame_to_columns
from .base import FileFormat, Selection


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

    def read_columns(
        self,
        source: Any,
        *,
        columns: list[str] | None = None,
        orient: str = "records",
        **kwargs: Any,
    ) -> Columns:
        """Read the whole JSON file into a column mapping.

        A single-array JSON document has no row-batch reader, so :meth:`stream_import` is
        not overridden -- a ``batch_size`` request reads the file whole with a warning.
        """
        import pandas as pd

        # dtype=False keeps the JSON-encoded type (a quoted "1" stays a string
        # instead of being inferred to int), so a string column round-trips.
        frame = pd.read_json(os.fspath(source), orient=orient, dtype=False)
        if columns is not None:
            frame = frame[columns]
        return frame_to_columns(frame)
