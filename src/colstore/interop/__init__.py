"""Interoperability between colstore and external data formats.

Data is exchanged through :class:`Format` objects, registered under a short name
and discovered through this package. A :class:`DataFormat` bridges an in-memory
object (selected by name, e.g. Arrow); a :class:`FileFormat` bridges an on-disk
file (selected by extension, e.g. ROOT). A backend (``pyarrow``, PyROOT) is
imported lazily the first time a conversion runs, so importing this package -- or
colstore -- pulls in no heavy dependency.

Examples
--------
>>> import colstore
>>> colstore.interop.data_formats()       # frozenset({'arrow'})     # doctest: +SKIP
>>> table = ds.to("arrow")                # export, zero-copy        # doctest: +SKIP
>>> import polars as pl
>>> frame = pl.from_arrow(ds)             # via __arrow_c_stream__   # doctest: +SKIP
"""

from .._sizes import resolve_batch_rows
from . import cstore as _cstore  # noqa: F401  -- registers the native cstore file format
from ._stream_import import StreamPlan
from .base import (
    DataFormat,
    FileFormat,
    Format,
    InteropMixin,
    Selection,
    data_formats,
    file_format_for_extension,
    file_format_for_path,
    file_formats,
    from_object,
    get,
    register,
)

__all__ = [
    "DataFormat",
    "FileFormat",
    "Format",
    "InteropMixin",
    "Selection",
    "StreamPlan",
    "data_formats",
    "file_format_for_extension",
    "file_format_for_path",
    "file_formats",
    "from_object",
    "get",
    "register",
    "resolve_batch_rows",
]
