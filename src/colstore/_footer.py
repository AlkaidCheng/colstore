"""Per-record statistics footer for colstore files.

A footer appended after the last record holds, for each record and each column,
the column's ``min``, ``max``, and a ``prunable`` flag (1 = min/max are valid and
usable for skipping; 0 = always read this record for this column). A read-side
comparison predicate can then skip whole records whose ``[min, max]`` cannot
satisfy it, without reading their bytes.

The footer is advisory. A reader that finds it absent, of an unknown version, or
failing its CRC falls back to a full read rather than raising. Its location is
recorded as ``stats_offset`` in the mutable counters block (see
:mod:`colstore.format`), so it is rewritten in place whenever the records change.

Layout (all little-endian, matching the column bytes on disk)::

    8B   magic b"CSTAT\\x00\\x01\\x00"
    4B   footer version (u32)
    4B   n_columns      (u32)
    4B   n_records      (u32)
    --- column directory, n_columns entries ---
      2B name length, name (utf-8), 2B dtype-str length, dtype str (utf-8)
    --- per-column stats blocks (column-major) ---
      for each column, using its directory dtype/itemsize:
        n_records * itemsize  : min values
        n_records * itemsize  : max values
        n_records * 1B        : prunable flags (0 / 1)
    4B   CRC32 over everything above
"""

from __future__ import annotations

import struct
import zlib
from typing import Any

import numpy as np

_FOOTER_MAGIC = b"CSTAT\x00\x01\x00"
_FOOTER_VERSION = 1
_HEADER_FMT = "<III"  # version, n_columns, n_records
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 12

# NumPy dtype kinds whose min/max support range pruning: floats, signed and
# unsigned integers, booleans, datetime64, and timedelta64. Fixed-width strings
# (``S`` / ``U``) and object/other kinds are excluded.
_PRUNABLE_KINDS = frozenset("fiubMm")

ColumnStat = tuple[Any, Any, bool]  # (min, max, prunable)
RecordStats = dict[str, dict[str, np.ndarray[Any, np.dtype[Any]]]]
# RecordStats[name] = {"min": (R,), "max": (R,), "prunable": (R,) bool}


def column_stat(dtype: np.dtype[Any], array: np.ndarray[Any, np.dtype[Any]]) -> ColumnStat:
    """Compute ``(min, max, prunable)`` for one record's column array.

    ``prunable`` is False when the kind is not range-comparable (strings), the
    record is empty, or a chunk holds a value that makes a min/max test unsound: a
    NaN/inf for floats, or a NaT for datetime64/timedelta64 (which propagate to
    min/max as the int64 sentinel). The min/max are the dtype's zero when not
    prunable.
    """
    zero = np.zeros((), dtype=dtype)
    if dtype.kind not in _PRUNABLE_KINDS or array.size == 0:
        return zero, zero, False
    if dtype.kind == "f" and not bool(np.isfinite(array).all()):
        return zero, zero, False
    # NaT does not compare like a real value (it views as int64-min, which is not
    # caught by isfinite); detect it directly so a NaT record is never prunable.
    if dtype.kind in "Mm" and bool(np.isnat(array).any()):
        return zero, zero, False
    return array.min(), array.max(), True


def serialize_stats(
    columns_meta: list[dict[str, Any]],
    per_record: list[dict[str, ColumnStat]],
) -> bytes:
    """Serialize per-record column stats to footer bytes.

    ``per_record[r][name]`` is the ``(min, max, prunable)`` for record ``r`` and
    column ``name``; columns are written in ``columns_meta`` order.
    """
    n_records = len(per_record)
    n_columns = len(columns_meta)
    parts: list[bytes] = [
        _FOOTER_MAGIC,
        struct.pack(_HEADER_FMT, _FOOTER_VERSION, n_columns, n_records),
    ]
    for meta in columns_meta:
        name_b = meta["name"].encode("utf-8")
        dtype_b = meta["dtype"].encode("utf-8")
        parts.append(struct.pack("<H", len(name_b)))
        parts.append(name_b)
        parts.append(struct.pack("<H", len(dtype_b)))
        parts.append(dtype_b)
    for meta in columns_meta:
        name = meta["name"]
        dtype = np.dtype(meta["dtype"])
        mins = np.empty(n_records, dtype=dtype)
        maxes = np.empty(n_records, dtype=dtype)
        flags = np.empty(n_records, dtype=np.uint8)
        for index, record in enumerate(per_record):
            minimum, maximum, prunable = record[name]
            mins[index] = minimum
            maxes[index] = maximum
            flags[index] = 1 if prunable else 0
        parts.append(mins.tobytes())
        parts.append(maxes.tobytes())
        parts.append(flags.tobytes())
    body = b"".join(parts)
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def parse_stats(raw: bytes) -> RecordStats | None:
    """Parse footer bytes into per-column ``(min, max, prunable)`` arrays.

    Returns ``None`` on any inconsistency (wrong magic, unknown version, CRC
    mismatch, truncation) -- the footer is advisory, so the caller falls back to a
    full read rather than raising.
    """
    try:
        if len(raw) < len(_FOOTER_MAGIC) + _HEADER_SIZE + 4:
            return None
        if raw[: len(_FOOTER_MAGIC)] != _FOOTER_MAGIC:
            return None
        body = raw[:-4]
        (stored_crc,) = struct.unpack_from("<I", raw, len(raw) - 4)
        if (zlib.crc32(body) & 0xFFFFFFFF) != stored_crc:
            return None
        version, n_columns, n_records = struct.unpack_from(_HEADER_FMT, raw, len(_FOOTER_MAGIC))
        if version != _FOOTER_VERSION:
            return None
        offset = len(_FOOTER_MAGIC) + _HEADER_SIZE
        columns: list[tuple[str, np.dtype[Any]]] = []
        for _ in range(n_columns):
            (name_len,) = struct.unpack_from("<H", raw, offset)
            offset += 2
            name = raw[offset : offset + name_len].decode("utf-8")
            offset += name_len
            (dtype_len,) = struct.unpack_from("<H", raw, offset)
            offset += 2
            dtype = np.dtype(raw[offset : offset + dtype_len].decode("utf-8"))
            offset += dtype_len
            columns.append((name, dtype))
        result: RecordStats = {}
        for name, dtype in columns:
            itemsize = dtype.itemsize
            mins = np.frombuffer(raw, dtype, n_records, offset).copy()
            offset += n_records * itemsize
            maxes = np.frombuffer(raw, dtype, n_records, offset).copy()
            offset += n_records * itemsize
            flags = np.frombuffer(raw, np.uint8, n_records, offset).astype(bool)
            offset += n_records
            result[name] = {"min": mins, "max": maxes, "prunable": flags}
        return result
    except (struct.error, ValueError, TypeError, UnicodeDecodeError, IndexError):
        return None
