"""Base contract and shared helpers for colstore format parsers.

A parser bridges an external data format and a ``.cstore`` file in both
directions. Concrete parsers (one module per format in this package) subclass
:class:`Parser` and reuse the helpers here for the parts that do not depend on
the source format: resolving a polymorphic batch size to a row count and
streaming a sequence of column batches into a colstore file.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar, TypeAlias

import numpy as np

from ..compaction import compact_file
from ..progress import progress_bar
from ..reader import ColStoreReader
from ..writer import ColStoreWriter

StrPath: TypeAlias = str | os.PathLike[str]
ColumnBatch: TypeAlias = dict[str, np.ndarray[Any, np.dtype[Any]]]


class Parser(ABC):
    """Two-way bridge between an external data format and a ``.cstore`` file.

    Concrete subclasses live one per module in :mod:`colstore.parsers` and set
    :attr:`format_name`. The typed, format-specific entry points are the
    module-level functions each parser exposes (for example
    :func:`colstore.parsers.from_root`); the methods here give those a
    uniform object surface for programmatic dispatch.

    Attributes
    ----------
    format_name : str
        Short identifier for the external format, e.g. ``"root"``.
    """

    format_name: ClassVar[str]

    @abstractmethod
    def read(self, source: Any, path: StrPath, **kwargs: Any) -> ColStoreReader:
        """Read the external ``source`` into a ``.cstore`` file at ``path`` and open it."""

    @abstractmethod
    def write(self, source: ColStoreReader | StrPath, path: StrPath, **kwargs: Any) -> Any:
        """Write the ``.cstore`` ``source`` out to the external format at ``path``."""


def write_column_batches(
    batches: Iterable[ColumnBatch],
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
    validated per batch and the schema is locked on the first non-empty one.
    The iterable must yield at least one batch (a zero-row but typed batch is
    fine for an empty source); otherwise no schema is captured and no valid
    file is written.

    Parameters
    ----------
    batches : iterable of dict[str, numpy.ndarray]
        Column-major batches in declaration order. Every batch must carry the
        same columns and dtypes.
    path : str or os.PathLike
        Destination ``.cstore`` file.
    mode : str, optional
        Writer mode: ``"create"`` (default), ``"recreate"``, or ``"update"``.
    total_rows : int or None, optional
        Total row count, used only for the progress bar's total.
    compact : bool, optional
        Collapse the multi-record result into a single record afterward.
        Skipped when only one record was written. Defaults to True.
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
