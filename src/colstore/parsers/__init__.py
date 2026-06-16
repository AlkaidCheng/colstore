"""Pluggable parsers that convert external data formats to and from colstore.

Each format lives in its own module and exposes a :class:`Parser` subclass plus
module-level convenience functions. Importing this package pulls in no heavy
third-party dependency; format backends (e.g. ROOT) are imported lazily the
first time a conversion runs.

Examples
--------
>>> from colstore.parsers import root_to_colstore, colstore_to_root
>>> reader = root_to_colstore("events.root", "events.cstore")  # doctest: +SKIP
>>> rdf = colstore_to_root(reader, "roundtrip.root")  # doctest: +SKIP
"""

from .base import Parser
from .root import RootParser, colstore_to_root, root_to_colstore

__all__ = ["Parser", "RootParser", "colstore_to_root", "root_to_colstore"]
