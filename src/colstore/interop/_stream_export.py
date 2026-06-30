"""Bounded-memory streaming export: write a ``.cstore`` selection to a foreign file
one row-batch at a time, instead of materializing every column first.

The export counterpart of :mod:`._stream_import`. The cstore reader is the source
here: :meth:`Selection.iter_batches` pulls the selected rows in bounded-memory
row-batches, and a format's :meth:`FileFormat.stream_export` feeds each batch to an
incremental foreign writer (a Parquet row-group writer, an Arrow IPC record-batch
writer, a resizable HDF5 dataset, a ROOT Snapshot). A format whose writer cannot
append incrementally -- or whose write options a streamed write cannot carry -- returns
a reason string, so :meth:`FileFormat.write_file` warns and writes the whole selection,
keeping ``batch_size`` best-effort rather than a silent no-op or a hard failure.
"""

from __future__ import annotations

import os
import warnings
from typing import Any


def warn_whole_file_export(dest: Any, reason: str) -> None:
    """Warn that ``batch_size`` could not stream the write to ``dest`` and it was written whole."""
    warnings.warn(
        f"batch_size: {os.fspath(dest)} was written whole-file because it cannot "
        f"stream ({reason}).",
        RuntimeWarning,
        stacklevel=4,
    )
