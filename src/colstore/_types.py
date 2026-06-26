"""Shared structural type aliases used across the package.

A single home for the small aliases that several modules would otherwise spell
independently, so the spelling stays consistent and there is one place to change
it. These are type-checker constructs only; nothing here is imported at runtime
by the class-bearing aliases.
"""

from __future__ import annotations

import os
from typing import TypeAlias

#: A filesystem path accepted by the public read/write functions.
StrPath: TypeAlias = str | os.PathLike[str]
