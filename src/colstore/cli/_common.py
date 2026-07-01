"""Shared helpers for the colstore CLI command modules.

Small, command-agnostic pieces (argparse ``type=`` callables and the like) that more
than one command reuses live here, rather than in any one command module or in the
package ``__init__`` (which imports the command modules and so cannot be imported back
from them).
"""

from __future__ import annotations

import argparse


def key_value(text: str) -> tuple[str, str]:
    """Parse a ``NAME=VALUE`` option argument into a ``(name, value)`` pair.

    An argparse ``type=`` for a repeatable option collected into a mapping (e.g.
    ``--dtype flag=bool`` or ``--rename raw=clean``). Raises
    :class:`argparse.ArgumentTypeError` for a missing ``=`` or an empty name, so argparse
    reports it as a usage error.
    """
    name, sep, value = text.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    return name, value
