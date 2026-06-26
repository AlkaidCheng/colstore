"""Shared structural type aliases used across the package.

A single home for the small aliases that several modules would otherwise spell
independently, so the spelling stays consistent and there is one place to change
it. ``StrPath`` and ``Columns`` are ordinary runtime alias objects; only the
class references in ``Source`` (``ColStoreReader`` / ``ColStoreDataset``) are
deferred under ``TYPE_CHECKING`` to keep this module a runtime leaf.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from .dataset import ColStoreDataset
    from .reader import ColStoreReader

#: A filesystem path accepted by the public read/write functions.
StrPath: TypeAlias = str | os.PathLike[str]

#: A column-name -> array mapping (a record's worth of columns, or a write batch).
Columns: TypeAlias = dict[str, np.ndarray[Any, np.dtype[Any]]]

#: A colstore source: a path to open, or an already-open reader/dataset. The
#: forward-reference string is self-contained (it does not name ``StrPath``) so
#: that ``typing.get_type_hints`` resolves it in any consumer's namespace.
Source: TypeAlias = "str | os.PathLike[str] | ColStoreReader | ColStoreDataset"
