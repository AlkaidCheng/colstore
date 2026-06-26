"""Feather (Arrow IPC) file format (``colstore.interop.feather``), via pyarrow.

The on-disk container is the Arrow IPC / Feather v2 format.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar

from ._convert import arrow_to_columns, selection_to_arrow_table, store_columns
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
        self, source: Any, dest: Any, *, columns: list[str] | None = None, **kwargs: Any
    ) -> ColStoreReader:
        """Read a Feather file into a ``.cstore`` and open it (extra kwargs -> store)."""
        import pyarrow.feather

        feather: Any = pyarrow.feather
        table = feather.read_table(os.fspath(source), columns=columns)
        return store_columns(arrow_to_columns(table), dest, **kwargs)
