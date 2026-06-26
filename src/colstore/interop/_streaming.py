"""Stream a sequence of column batches into a ``.cstore`` file.

Shared by the file-format importers (e.g. ROOT) to materialize a foreign source
into a ``.cstore`` in bounded memory -- one record per batch, validated as it is
written -- then optionally compact the result to a single record.
"""

from __future__ import annotations

from collections.abc import Iterable

from .._types import Columns, StrPath
from ..compaction import compact_file
from ..progress import progress_bar
from ..reader import ColStoreReader
from ..writer import ColStoreWriter


def write_column_batches(
    batches: Iterable[Columns],
    path: StrPath,
    *,
    mode: str = "create",
    total_rows: int | None = None,
    compact: bool = True,
    show_progress: bool = True,
    desc: str | None = None,
) -> ColStoreReader:
    """Stream column batches into a ``.cstore`` file, one record per batch.

    Each batch is one :meth:`colstore.ColStoreWriter.write` call, so dtypes are
    validated per batch and the schema is locked on the first non-empty one. The
    iterable must yield at least one batch (a zero-row but typed batch is fine for
    an empty source); otherwise no schema is captured and no valid file is written.

    Parameters
    ----------
    batches : iterable of dict[str, numpy.ndarray]
        Column-major batches in declaration order. Every batch must carry the same
        columns and dtypes.
    path : str or os.PathLike
        Destination ``.cstore`` file.
    mode : str, optional
        Writer mode: ``"create"`` (default), ``"recreate"``, or ``"update"``.
    total_rows : int or None, optional
        Total row count, used only for the progress bar's total.
    compact : bool, optional
        Collapse the multi-record result into a single record afterward. Skipped
        when only one record was written. Defaults to True.
    show_progress : bool, optional
        Whether to display a progress bar. Defaults to True.
    desc : str or None, optional
        Progress-bar description.

    Returns
    -------
    colstore.ColStoreReader
        An opened reader over the written (and optionally compacted) file.
    """
    n_records = 0
    with (
        progress_bar(
            total=total_rows,
            desc=desc,
            unit="row",
            unit_scale=True,
            enabled=show_progress,
        ) as bar,
        ColStoreWriter(path, mode=mode) as writer,
    ):
        for batch in batches:
            writer.write(batch)
            if batch:
                bar.update(len(next(iter(batch.values()))))
            n_records += 1
    if compact and n_records > 1:
        compact_file(path, None, show_progress=show_progress)
    return ColStoreReader(path)
