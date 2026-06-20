"""Path helpers shared by the ``open`` / ``concat`` / dataset entry points.

Glob expansion for path inputs: a source string carrying a shell wildcard
(``*``, ``?``, ``[``; ``**`` recursive) is expanded to the files it matches,
ordered numerically so ``run_2`` precedes ``run_10``. File order is a dataset's
global row order, so a stable, intuitive ordering is correctness-adjacent, not
cosmetic. Globbing applies to path inputs only -- column selection is always
explicit and is never wildcarded, so a schema stays fixed.
"""

from __future__ import annotations

import glob
import os
import re

# Shell glob metacharacters. A source string containing one is treated as a
# pattern to expand; everything else is a literal path.
_GLOB_MAGIC = re.compile(r"[*?[]")


def has_glob_magic(text: str) -> bool:
    """Return whether ``text`` contains a shell glob metacharacter (``*?[``)."""
    return _GLOB_MAGIC.search(text) is not None


def _natural_sort_key(text: str) -> list[int | str]:
    """Sort key ordering embedded numbers numerically, so ``run_2`` < ``run_10``.

    Splitting on digit runs yields alternating non-digit / digit chunks; the
    digit chunks compare as ints and the rest as strings. The chunks line up by
    position across any two names (index 0 is always a string), so the mixed-type
    list comparison never pits an int against a string.
    """
    return [int(chunk) if chunk.isdigit() else chunk for chunk in re.split(r"(\d+)", text)]


def expand_glob(item: str | os.PathLike[str]) -> list[str]:
    """Expand a glob-pattern *string* to its matches; pass any other path through.

    Only a ``str`` carrying glob magic is treated as a pattern (``**`` is
    recursive); a literal name and any ``os.PathLike`` are returned as a
    one-element list, so a filename that happens to contain a bracket is never
    reinterpreted. Matches are natural-sorted (file order is the dataset's global
    row order), and a pattern matching nothing raises ``FileNotFoundError`` --
    a wildcard that resolves to no files is a mistake, not an empty dataset.
    """
    text = os.fspath(item)
    if not isinstance(item, str) or not has_glob_magic(text):
        return [text]
    matches = glob.glob(text, recursive=True)
    if not matches:
        raise FileNotFoundError(f"no files matched the pattern {text!r}.")
    return sorted(matches, key=_natural_sort_key)
