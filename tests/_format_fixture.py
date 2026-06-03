"""Hand-built file fixtures for testing the multi-record reader.

To validate the reader independently of the writer, this module builds
files byte-by-byte from Python and feeds them through
:class:`ColStoreReader`. It is the single source of truth for "what a
valid multi-record file looks like" -- if the writer and this fixture
ever disagree, one of them is wrong, and this one is the spec.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

from colstore import format as fmt


def _to_le_bytes(arr: np.ndarray) -> bytes:
    """Serialize ``arr`` as little-endian column bytes, mirroring the writer."""
    if arr.dtype.kind in ("f", "i", "u", "M", "m") and arr.dtype.itemsize > 1:
        arr = arr.astype(arr.dtype.newbyteorder("<"), copy=False)
    return arr.tobytes(order="C")


def write_record_file(
    path: Path,
    schema: list[tuple[str, str]],
    records: list[dict[str, np.ndarray]],
) -> None:
    """Write a colstore file to ``path``.

    Parameters
    ----------
    path : Path
        Destination file. Truncates if it exists.
    schema : list of (name, dtype_str)
        Column definitions in the order they appear within each record body.
        Every record must contain exactly these columns with these dtypes.
    records : list of dict
        One dict per record. Each maps column name -> 1D ndarray. All arrays
        in one record must share the same length (rows in that record).

    Raises
    ------
    ValueError
        On schema mismatch between records or non-matching array lengths.
    """
    if not records:
        raise ValueError("Need at least one record to build a file.")
    expected_names = [name for name, _ in schema]

    # Validate every record matches the schema, and capture per-record n_rows.
    n_rows_per_record: list[int] = []
    for i, rec in enumerate(records):
        if set(rec.keys()) != set(expected_names):
            raise ValueError(
                f"Record {i} columns {sorted(rec.keys())} do not match schema "
                f"{sorted(expected_names)}."
            )
        lengths = {len(rec[name]) for name in expected_names}
        if len(lengths) != 1:
            raise ValueError(f"Record {i} has columns of different lengths: {lengths}.")
        for name, dt in schema:
            if np.dtype(rec[name].dtype) != np.dtype(dt):
                raise ValueError(
                    f"Record {i} column {name!r} dtype {rec[name].dtype} does not "
                    f"match schema dtype {dt}."
                )
        n_rows_per_record.append(lengths.pop())

    total_rows = sum(n_rows_per_record)
    n_records = len(records)
    columns_meta = [{"name": name, "dtype": dt, "encoding": "raw"} for name, dt in schema]

    # ---- Build the file header. ----
    manifest = {
        "format_version": 1,
        "columns": columns_meta,
        "manifest_crc32": fmt._manifest_checksum(columns_meta),
    }
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    header_size = (
        len(fmt._MAGIC) + fmt._COUNTERS_SIZE + fmt._MANIFEST_LEN_SIZE + len(manifest_bytes)
    )
    data_offset = fmt.align_up(header_size)

    with open(path, "wb") as out:
        out.write(fmt._MAGIC)
        out.write(fmt._pack_counters(n_records, total_rows))
        out.write(struct.pack(fmt._MANIFEST_LEN_FMT, len(manifest_bytes)))
        out.write(manifest_bytes)
        out.write(b"\x00" * (data_offset - header_size))

        # ---- Records: 32B header + column-major body + 8B padding. ----
        for record_index, rec in enumerate(records):
            n_rows = n_rows_per_record[record_index]
            # Header CRC is over the first 28 bytes of the header (everything
            # but the CRC slot itself). We build those 28 bytes first.
            header_prefix = struct.pack(
                "<4sqqq",
                fmt._RECORD_MAGIC,
                record_index,
                n_rows,
                0,  # reserved
            )
            assert len(header_prefix) == 28
            crc = zlib.crc32(header_prefix) & 0xFFFFFFFF
            out.write(header_prefix)
            out.write(struct.pack("<I", crc))

            # Column bodies, in schema order, contiguous, no padding between.
            body_bytes = 0
            for name, _ in schema:
                col_bytes = _to_le_bytes(rec[name])
                out.write(col_bytes)
                body_bytes += len(col_bytes)
            # Pad the body up to _RECORD_BODY_ALIGNMENT (8 bytes).
            pad = fmt.align_up(body_bytes, fmt._RECORD_BODY_ALIGNMENT) - body_bytes
            if pad:
                out.write(b"\x00" * pad)


def expected_column_values(records: list[dict[str, np.ndarray]], name: str) -> np.ndarray:
    """Logical concatenation of column ``name`` across all records.

    This is the ground truth: a ColStoreReader read with ``ds[:, name]`` or
    ``ds[indices, name]`` must agree with indexing into this array.
    """
    return np.concatenate([rec[name] for rec in records])
