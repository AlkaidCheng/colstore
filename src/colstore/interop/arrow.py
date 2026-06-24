"""Apache Arrow data formatter for colstore.

A native colstore column is laid out exactly as an Arrow primitive array's values
buffer: a contiguous block of fixed-width little-endian elements with no validity
bitmap (columns are non-nullable) and no offset buffer (every element is the same
width). :class:`ArrowFormatter` wraps that buffer as a :class:`pyarrow.Array`
without copying -- the Arrow values buffer points straight at the memory-mapped
bytes, and the buffer keeps a reference to the source view so the file stays
mapped for as long as any Arrow consumer holds the data. A column split across
records or files becomes a :class:`pyarrow.ChunkedArray` with one zero-copy chunk
per segment; several columns become a :class:`pyarrow.Table`.

Numeric, fixed-width byte (``S`` -> ``fixed_size_binary``), and
``datetime64`` / ``timedelta64`` columns in second-to-nanosecond units wrap with
zero copy. Boolean columns (Arrow packs booleans to one bit per value), Unicode
columns (``U``, which Arrow has no fixed-width type for), non-native byte order,
and any row selection other than the whole column are converted through a copy. A
``datetime64`` / ``timedelta64`` column in a unit Arrow has no equivalent for
(coarser than a second or finer than a nanosecond) is rejected with a clear error.

The reader, dataset, and view classes also implement the Arrow PyCapsule
interface (``__arrow_c_array__`` / ``__arrow_c_stream__``), so an Arrow consumer
ingests colstore data directly, e.g. ``pyarrow.table(reader)`` or
``polars.from_arrow(reader)``.

``pyarrow`` is an optional dependency, imported only when a conversion runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import DataFormat

if TYPE_CHECKING:
    from .base import Selection


def _require_pyarrow() -> Any:
    """Import and return ``pyarrow``, or raise a clear install hint."""
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - exercised only without pyarrow
        raise ImportError(
            "Arrow export requires pyarrow; install it with "
            "'pip install colstore[arrow]' or 'pip install pyarrow'."
        ) from exc
    return pa


def _zero_copy_arrow_type(pa: Any, dtype: np.dtype[Any]) -> Any:
    """The Arrow type whose values buffer is byte-identical to ``dtype``, or ``None``.

    Returns ``None`` for dtypes Arrow stores differently (booleans are bit-packed,
    Unicode has no fixed-width Arrow type, and ``datetime64`` / ``timedelta64``
    units coarser than a second or finer than a nanosecond have no Arrow unit) --
    the caller then converts booleans/Unicode through a copy, and rejects an
    unrepresentable temporal unit with a clear error. ``dtype`` is assumed native
    byte order; non-native columns are converted by the caller before reaching here.
    """
    kind = dtype.kind
    itemsize = dtype.itemsize
    if kind == "f":
        return {2: pa.float16(), 4: pa.float32(), 8: pa.float64()}.get(itemsize)
    if kind == "i":
        return {1: pa.int8(), 2: pa.int16(), 4: pa.int32(), 8: pa.int64()}.get(itemsize)
    if kind == "u":
        return {1: pa.uint8(), 2: pa.uint16(), 4: pa.uint32(), 8: pa.uint64()}.get(itemsize)
    if kind == "S":
        # Fixed-width bytes map to fixed_size_binary; trailing NUL padding is
        # kept verbatim, so the bytes round-trip exactly.
        return pa.binary(itemsize)
    if kind in ("M", "m"):
        unit = np.datetime_data(dtype)[0]
        if unit not in ("s", "ms", "us", "ns"):
            return None
        return pa.timestamp(unit) if kind == "M" else pa.duration(unit)
    return None


def _array_from_values(pa: Any, values: np.ndarray[Any, np.dtype[Any]]) -> Any:
    """Wrap one contiguous native ndarray as a :class:`pyarrow.Array`.

    Zero copy when ``values``' dtype is byte-identical to an Arrow primitive type
    (see :func:`_zero_copy_arrow_type`): the Arrow values buffer aliases
    ``values``' memory and keeps ``values`` -- and any memmap behind it -- alive.
    Boolean and Unicode columns are converted by :func:`pyarrow.array`, which
    allocates the Arrow buffer. A ``datetime64`` / ``timedelta64`` column whose
    unit Arrow cannot represent is rejected with a clear error (``pyarrow.array``
    would otherwise raise opaquely, or silently narrow a ``datetime64[D]`` to a
    32-bit date).
    """
    arrow_type = _zero_copy_arrow_type(pa, values.dtype)
    if arrow_type is not None and values.flags["C_CONTIGUOUS"]:
        buffer = pa.foreign_buffer(values.ctypes.data, values.nbytes, base=values)
        # Two buffers for a non-nullable fixed-width array: no validity bitmap
        # (null_count 0), then the values buffer. No offset buffer -- the width is
        # constant.
        return pa.Array.from_buffers(arrow_type, values.shape[0], [None, buffer], null_count=0)
    if arrow_type is None and values.dtype.kind in ("M", "m"):
        np_kind = "datetime64" if values.dtype.kind == "M" else "timedelta64"
        arrow_kind = "timestamp" if values.dtype.kind == "M" else "duration"
        raise TypeError(
            f"Arrow has no {arrow_kind} unit for {values.dtype}; colstore exports "
            f"datetime64/timedelta64 only in second-to-nanosecond units. Cast the column "
            f"to a supported unit (e.g. astype('{np_kind}[us]')) before export."
        )
    return pa.array(values)


def _chunks_to_array(
    pa: Any, chunks: list[np.ndarray[Any, np.dtype[Any]]], dtype: np.dtype[Any]
) -> Any:
    """Assemble a column's per-segment views into an Array or ChunkedArray.

    A single contiguous segment becomes one :class:`pyarrow.Array`; several
    segments become a :class:`pyarrow.ChunkedArray`, one zero-copy chunk each. An
    empty column (no segments) yields an empty single array of ``dtype``.
    """
    arrays = [_array_from_values(pa, chunk) for chunk in chunks]
    if len(arrays) == 1:
        return arrays[0]
    if not arrays:
        return _array_from_values(pa, np.empty(0, dtype=dtype))
    return pa.chunked_array(arrays)


class ArrowFormat(DataFormat):
    """Export colstore data to Apache Arrow, zero-copy where the layout permits."""

    name = "arrow"

    def to_object(self, selection: Selection) -> Any:
        """Export the selection: one column to an Array/ChunkedArray, several to a Table."""
        if selection.single:
            return self._column(selection, selection.columns[0])
        pa = _require_pyarrow()
        return pa.table({name: self._column(selection, name) for name in selection.columns})

    def _column(self, selection: Selection, name: str) -> Any:
        """One column as an Arrow array: zero-copy for the whole column, else a gather."""
        pa = _require_pyarrow()
        if selection.is_whole_column():
            try:
                chunks = selection.column_chunks(name)
            except ValueError:
                # Non-native byte order: a view cannot byteswap. Fall through to
                # the gather, which converts to native order.
                chunks = None
            if chunks is not None:
                return _chunks_to_array(pa, chunks, selection.native_dtype(name))
        return _array_from_values(pa, selection.gather(name))

    # from_object is not overridden: Arrow import is a later addition, so
    # can_import is False and an attempt raises the base class's clear error.


def to_c_array(arrow_obj: Any, requested_schema: Any) -> Any:
    """Capsule for the Arrow C array interface (``__arrow_c_array__``).

    A :class:`pyarrow.ChunkedArray` is concatenated into one array first (the
    single-array interface cannot represent chunks), which copies; the stream
    interface keeps a chunked column zero-copy.
    """
    pa = _require_pyarrow()
    if isinstance(arrow_obj, pa.ChunkedArray):
        arrow_obj = pa.concat_arrays(arrow_obj.chunks)
    return arrow_obj.__arrow_c_array__(requested_schema)


def to_c_stream(arrow_obj: Any, requested_schema: Any) -> Any:
    """Capsule for the Arrow C stream interface (``__arrow_c_stream__``).

    A single :class:`pyarrow.Array` is wrapped in a one-chunk chunked array so the
    stream form is available without copying.
    """
    pa = _require_pyarrow()
    if isinstance(arrow_obj, pa.Array):
        arrow_obj = pa.chunked_array([arrow_obj])
    return arrow_obj.__arrow_c_stream__(requested_schema)
