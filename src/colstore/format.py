"""On-disk file format for colstore.

File layout::

    [magic 8B]
    [counters 32B: n_records(8) + committed_rows(8) + crc32(4) + reserved(12)]
    [manifest_len 8B (u64 little-endian)]
    [manifest_json: format_version + columns + manifest_crc32]
    [zero-padding to 64-byte alignment]
    [record_0 header 32B][record_0 body, padded to 8B]
    [record_1 header 32B][record_1 body, padded to 8B]
    ...

The 8-byte magic is followed by a 32-byte counters block at fixed offset 8
holding the mutable ``n_records`` and ``committed_rows`` (with its own
CRC32). The JSON manifest holds the immutable schema (``format_version``
and per-column ``{name, dtype, encoding, nullable}``). Splitting mutable
counters from the immutable manifest is what lets the writer commit a
session atomically -- it can rewrite the 32-byte counters block in place
without shifting any record byte offsets. The canonical file extension is
``.cstore``.

A file is a sequence of one or more records. Each record carries a 32-byte
header followed by a column-major body (columns laid out back-to-back in
schema declaration order, no inter-column padding) padded up to 8 bytes so
the next record's header is naturally aligned. Files written in one shot
(via :func:`colstore.store`) produce a single-record file;
:class:`ColStoreWriter` produces multi-record files.

Each record header is::

    4B  magic b"REC\\x01"
    8B  record_index (i64, sequential from 0)
    8B  n_rows       (i64)
    8B  reserved
    4B  header CRC32 (over the first 28 bytes)

Supported column dtypes are the fixed-size NumPy kinds: floating point,
signed/unsigned integers, booleans, ``datetime64``/``timedelta64``, and
fixed-width strings (``S`` bytes and ``U`` unicode). Object/variable-length
columns are rejected.

Byte order: column bytes are always written **little-endian** so files are
portable across hosts. On a big-endian host the data is byte-swapped on
write and again on read, so callers always see native-order arrays.

The 8-byte magic ``b"CSTORE\\x00\\x01"`` spells the format name (6 ASCII
bytes) followed by two reserved bytes; the bytes are constant for the life
of the format. Per-instance evolution is tracked via ``format_version``
inside the manifest, not by changing the magic.
"""

import json
import os
import struct
import sys
import zlib
from typing import IO, Any

import numpy as np

from .progress import progress_bar

FILE_EXTENSION = ".cstore"
_MAGIC = b"CSTORE\x00\x01"
_MANIFEST_LEN_FMT = "<Q"
_MANIFEST_LEN_SIZE = struct.calcsize(_MANIFEST_LEN_FMT)
_ALIGNMENT = 64
_FORMAT_VERSION = 1
_SUPPORTED_VERSIONS = frozenset({1})

# Counters block. Lives at a fixed offset (right after the magic bytes) so the
# writer can rewrite it in place on close() without shifting any record byte
# offsets. The block is 32 bytes:
#
#     8B  n_records       (i64 LE)
#     8B  committed_rows  (i64 LE)
#     4B  counters_crc32  (over the first 16 bytes)
#     12B reserved        (zero)
#
# Separating mutable counters from the immutable JSON manifest is what enables
# crash-safe streaming writes: each successful close() atomically commits the
# new counter values; the immutable manifest never moves.
_COUNTERS_OFFSET = len(_MAGIC)  # 8
_COUNTERS_FMT = "<qqI12s"
_COUNTERS_SIZE = struct.calcsize(_COUNTERS_FMT)  # 32

# Record header layout. A record is "[32B header][record body]" where the
# body is column-major (columns concatenated in schema order) and padded to a
# multiple of _RECORD_BODY_ALIGNMENT bytes. The alignment is 8 so that every
# column's first element is naturally aligned for the kernel's typed loads
# regardless of itemsize mix in the schema.
_RECORD_HEADER_SIZE = 32
_RECORD_HEADER_FMT = (
    "<4sqqqI"  # magic(4) + record_index(i64) + n_rows(i64) + reserved(i64) + crc(u32)
)
_RECORD_HEADER_PACK_SIZE = struct.calcsize(_RECORD_HEADER_FMT)  # 32
_RECORD_MAGIC = b"REC\x01"
_RECORD_BODY_ALIGNMENT = 8

# Default per-column metadata. Recorded on write so future readers/writers can
# branch on these keys without a format break; today only the defaults are
# written.
_DEFAULT_ENCODING = "raw"  # reserved for future "zstd", "dict", etc.
_DEFAULT_NULLABLE = False  # reserved for future null-bitmap support

# Path-like accepted by every public function in this module.
PathLike = str | os.PathLike[str]

ColumnLayout = dict[str, tuple[int, np.dtype[Any]]]


class FormatError(Exception):
    """Raised when a file does not match the expected colstore format."""


def align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    """Round `value` up to the next multiple of `alignment`."""
    return ((value + alignment - 1) // alignment) * alignment


def _manifest_checksum(columns_meta: list[dict[str, Any]]) -> int:
    """Compute a CRC32 over the immutable manifest fields.

    The manifest carries only schema (format_version, columns). Mutable
    counters (n_records, committed_rows) live in a separate fixed-position
    block so the writer can update them on close() without changing the
    manifest's bytes.
    """
    payload = json.dumps({"columns": columns_meta}, sort_keys=True).encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF


def _pack_counters(n_records: int, committed_rows: int) -> bytes:
    """Pack the 32-byte counters block including its CRC32."""
    body = struct.pack("<qq", n_records, committed_rows)
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return struct.pack(_COUNTERS_FMT, n_records, committed_rows, crc, b"\x00" * 12)


def _unpack_counters(raw: bytes) -> tuple[int, int]:
    """Parse and validate the 32-byte counters block; return (n_records, committed_rows)."""
    if len(raw) != _COUNTERS_SIZE:
        raise FormatError(
            f"Counters block truncated: expected {_COUNTERS_SIZE} bytes, got {len(raw)}."
        )
    n_records, committed_rows, stored_crc, _reserved = struct.unpack(_COUNTERS_FMT, raw)
    actual_crc = zlib.crc32(struct.pack("<qq", n_records, committed_rows)) & 0xFFFFFFFF
    if actual_crc != stored_crc:
        raise FormatError(
            f"Counters block CRC mismatch (stored {stored_crc}, computed {actual_crc}); "
            f"the file header is corrupt."
        )
    return n_records, committed_rows


def record_body_size(n_rows: int, itemsizes: list[int]) -> int:
    """Return the on-disk size of a record body given its row count and schema.

    The body is column-major (each column contiguous) with no inter-column
    padding; the whole body is padded up to ``_RECORD_BODY_ALIGNMENT`` (8B)
    so the next record's header -- and therefore its body -- is naturally
    aligned for the largest itemsize the kernel handles.
    """
    raw_bytes = n_rows * sum(itemsizes)
    return align_up(raw_bytes, _RECORD_BODY_ALIGNMENT)


def read_record_index(
    path: PathLike,
    data_offset: int,
    n_records: int,
    itemsizes: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk record headers to build the per-record index used by the reader.

    Returns three small int64 arrays:

    * ``record_starts_rows`` shape ``(R+1,)`` -- cumulative row counts. The
      reader's hot path uses this with ``np.searchsorted`` to bin indices to
      records.
    * ``record_starts_bytes`` shape ``(R,)`` -- file byte offset of each
      record's body (post-32B header). The base address for any byte-offset
      computation against that record.
    * ``n_rows_per_record`` shape ``(R,)`` -- needed at read time to compute
      per-column offsets within a record without materializing a 2D table:
      ``column_prefix_bytes[j] * n_rows_per_record[record_id]``.

    Total storage: 24 bytes per record. Cost to build: ``R`` * 32B reads,
    typically microseconds even for thousands of records.

    Raises
    ------
    FormatError
        On wrong record magic, mismatched record index, or CRC mismatch in
        any record header. All three indicate file corruption.
    """
    record_starts_rows = np.empty(n_records + 1, dtype=np.int64)
    record_starts_bytes = np.empty(n_records, dtype=np.int64)
    n_rows_per_record = np.empty(n_records, dtype=np.int64)
    record_starts_rows[0] = 0

    cumulative_rows = 0
    next_record_offset = data_offset
    with open(path, "rb") as input_file:
        for record_index in range(n_records):
            input_file.seek(next_record_offset)
            header_bytes = input_file.read(_RECORD_HEADER_PACK_SIZE)
            if len(header_bytes) != _RECORD_HEADER_PACK_SIZE:
                raise FormatError(
                    f"Truncated record header at offset {next_record_offset}: "
                    f"expected {_RECORD_HEADER_PACK_SIZE} bytes, got {len(header_bytes)}."
                )
            magic, stored_index, n_rows, _reserved, stored_crc = struct.unpack(
                _RECORD_HEADER_FMT, header_bytes
            )
            if magic != _RECORD_MAGIC:
                raise FormatError(
                    f"Bad record magic at offset {next_record_offset}: expected "
                    f"{_RECORD_MAGIC!r}, got {magic!r}."
                )
            if stored_index != record_index:
                raise FormatError(
                    f"Record index mismatch at offset {next_record_offset}: "
                    f"manifest expects record {record_index}, header says {stored_index}."
                )
            # CRC covers the first 28 bytes (everything but the CRC itself).
            actual_crc = zlib.crc32(header_bytes[:28]) & 0xFFFFFFFF
            if actual_crc != stored_crc:
                raise FormatError(
                    f"Record {record_index} header CRC mismatch "
                    f"(stored {stored_crc}, computed {actual_crc})."
                )

            body_offset = next_record_offset + _RECORD_HEADER_SIZE
            record_starts_bytes[record_index] = body_offset
            n_rows_per_record[record_index] = n_rows
            cumulative_rows += n_rows
            record_starts_rows[record_index + 1] = cumulative_rows
            next_record_offset = body_offset + record_body_size(n_rows, itemsizes)

        # After the walk, ``next_record_offset`` is where record N+1's header
        # would start, equivalently the end of the last record's padded body.
        # The file must extend at least this far; if it doesn't, the last
        # record's body is truncated. Inter-record truncation would already
        # have been caught above (header read or magic check on the next
        # record), but a truncation in the final record's body is only
        # visible here.
        file_size = os.fstat(input_file.fileno()).st_size
        if file_size < next_record_offset:
            raise FormatError(
                f"File is truncated: last record body ends at offset "
                f"{next_record_offset} but file is only {file_size} bytes."
            )

    return record_starts_rows, record_starts_bytes, n_rows_per_record


def write_header(
    file: IO[bytes],
    columns_meta: list[dict[str, Any]],
    n_records: int,
    committed_rows: int,
) -> int:
    """Write magic + counters + manifest + padding; return the data start offset.

    The on-disk header has four parts:

      * 8-byte magic (constant).
      * 32-byte counters block at fixed offset 8 -- ``(n_records,
        committed_rows, crc32)``. The writer rewrites this in place on
        :meth:`ColStoreWriter.close` without touching the manifest.
      * 8-byte manifest length prefix + JSON manifest (immutable schema).
      * Zero padding so the first record header lands at a 64-byte
        alignment boundary.

    Callers seeking only to update counters use :func:`write_counters`.
    """
    manifest = {
        "format_version": _FORMAT_VERSION,
        "columns": columns_meta,
        "manifest_crc32": _manifest_checksum(columns_meta),
    }
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    header_size = len(_MAGIC) + _COUNTERS_SIZE + _MANIFEST_LEN_SIZE + len(manifest_bytes)
    data_offset = align_up(header_size)

    file.write(_MAGIC)
    file.write(_pack_counters(n_records, committed_rows))
    file.write(struct.pack(_MANIFEST_LEN_FMT, len(manifest_bytes)))
    file.write(manifest_bytes)
    file.write(b"\x00" * (data_offset - header_size))
    return data_offset


def write_counters(file: IO[bytes], n_records: int, committed_rows: int) -> None:
    """Rewrite the 32-byte counters block at its fixed offset.

    Used by :meth:`ColStoreWriter.close` to commit the new record count and row
    total atomically. The 32-byte block is small enough that a single
    ``write()`` is generally atomic on common filesystems; even if it
    isn't, the embedded CRC catches a torn write on the next open.

    The caller must position the file at the right offset itself (the
    helper just packs and writes the 32 bytes), or use ``file.seek`` to
    move there before calling.
    """
    file.write(_pack_counters(n_records, committed_rows))


def write_record_header(file: IO[bytes], record_index: int, n_rows: int) -> None:
    """Write a 32-byte record header to ``file`` at its current position.

    The header CRC32 covers the first 28 bytes (everything but the CRC slot);
    a corrupt header is detected on read even if only the in-place fields
    were tampered with. Used by :func:`write_dataset` for the single-record
    write path and by :class:`ColStoreWriter` for the multi-record case.
    """
    header_prefix = struct.pack(
        "<4sqqq",
        _RECORD_MAGIC,
        record_index,
        n_rows,
        0,  # reserved
    )
    crc = zlib.crc32(header_prefix) & 0xFFFFFFFF
    file.write(header_prefix)
    file.write(struct.pack("<I", crc))


def read_header(path: PathLike) -> tuple[dict[str, Any], int]:
    """Read and validate the file header; return ``(header_dict, data_start_offset)``.

    The header_dict contains both the immutable manifest fields
    (``format_version``, ``columns``, ``manifest_crc32``) and the mutable
    counters (``n_records``, ``committed_rows``) read from the fixed
    32-byte counters block. Callers don't need to know they live in
    separate on-disk regions.

    Validates the file header only -- magic, counters CRC, format version,
    and manifest CRC. Per-record headers (and any truncation past the file
    header) are validated by :func:`read_record_index`, which the caller
    runs immediately after this to build the per-record index.

    Raises
    ------
    FormatError
        On wrong magic, counters CRC mismatch, unsupported
        ``format_version``, or manifest CRC mismatch.
    """
    with open(path, "rb") as input_file:
        magic = input_file.read(len(_MAGIC))
        if magic != _MAGIC:
            raise FormatError(f"Not a colstore file: expected magic {_MAGIC!r}, got {magic!r}")
        n_records, committed_rows = _unpack_counters(input_file.read(_COUNTERS_SIZE))
        manifest_size = struct.unpack(_MANIFEST_LEN_FMT, input_file.read(_MANIFEST_LEN_SIZE))[0]
        manifest_bytes = input_file.read(manifest_size)
    try:
        manifest = json.loads(manifest_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise FormatError(f"Manifest is not valid JSON; the header is corrupt ({e}).") from e

    version = manifest.get("format_version")
    if version not in _SUPPORTED_VERSIONS:
        raise FormatError(
            f"Unsupported format_version {version!r}; this build supports "
            f"{sorted(_SUPPORTED_VERSIONS)}."
        )

    expected_crc = manifest.get("manifest_crc32")
    if expected_crc is not None:
        actual_crc = _manifest_checksum(manifest["columns"])
        if actual_crc != expected_crc:
            raise FormatError(
                f"Manifest checksum mismatch (stored {expected_crc}, computed "
                f"{actual_crc}); the header is corrupt."
            )

    # Merge counters into the returned dict so callers see one unified view.
    manifest["n_records"] = n_records
    manifest["committed_rows"] = committed_rows

    header_size = len(_MAGIC) + _COUNTERS_SIZE + _MANIFEST_LEN_SIZE + manifest_size
    data_offset = align_up(header_size)
    return manifest, data_offset


def build_column_layout(manifest: dict[str, Any], body_offset: int, n_rows: int) -> ColumnLayout:
    """Compute per-column ``(byte_offset, dtype)`` within a single record.

    ``body_offset`` is the file offset of the record body's first byte --
    i.e. past the 32-byte record header. The reader's caller obtains this
    from :func:`read_record_index` so that the layout sits on top of an
    already-validated record.

    For a record with ``n_rows`` rows, each column's data is laid out
    back-to-back in schema declaration order with no inter-column padding.
    The reader uses the returned layout to build one ``np.memmap`` per
    column.

    Valid only for the single-record fast path. Multi-record files have no
    meaningful per-column layout (each column's bytes are scattered across
    records); the reader computes byte addresses on the fly instead.

    The dtype returned is the on-disk dtype, which is little-endian for
    multi-byte kinds. Memory-maps must use this dtype to interpret the
    bytes correctly; the store presents native-order arrays to callers via
    the gather path.
    """
    layout: ColumnLayout = {}
    current_offset = body_offset
    for column_info in manifest["columns"]:
        column_dtype = np.dtype(column_info["dtype"])
        layout[column_info["name"]] = (current_offset, column_dtype)
        current_offset += n_rows * column_dtype.itemsize
    return layout


_SUPPORTED_KINDS = frozenset({"f", "i", "u", "b", "M", "m", "S", "U"})


def _to_little_endian(array: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
    """Return `array` with little-endian byte order, copying only if needed.

    Multi-byte columns are stored little-endian on disk for portability. On a
    little-endian host this is a no-op; on a big-endian host it byte-swaps.
    Single-byte and string-of-bytes kinds have no byte order to normalize.
    """
    byteorder = array.dtype.byteorder
    if byteorder in ("|", "<"):
        return array
    if byteorder == ">" or (byteorder == "=" and sys.byteorder == "big"):
        return array.astype(array.dtype.newbyteorder("<"))
    return array


def normalize_columns(
    columns: dict[str, np.ndarray[Any, np.dtype[Any]]],
    *,
    expected_schema: list[dict[str, Any]] | None = None,
) -> tuple[list[str], int, dict[str, np.ndarray[Any, np.dtype[Any]]], list[dict[str, Any]]]:
    """Validate a column dict and return the pieces needed to write a record.

    Returns ``(column_names, n_rows, little_endian_columns, columns_meta)``.

    If ``expected_schema`` is given (non-None), the columns must match it
    exactly: same names in the same order, same dtypes. This is what the
    streaming writer uses for second-and-later ``write()`` calls. If
    ``expected_schema`` is None, the schema is inferred from the columns
    (the first-write case).

    Raises
    ------
    ValueError
        On an empty dict, ragged columns, or schema/dtype mismatch.
    TypeError
        On unsupported dtype kinds (object, void, etc.).
    """
    if not columns:
        raise ValueError("Cannot write an empty column mapping.")

    column_names = list(columns)
    n_rows = int(columns[column_names[0]].shape[0])
    for name, array in columns.items():
        if array.ndim != 1:
            raise ValueError(f"Column {name!r} must be 1D; got {array.ndim}D.")
        if array.dtype.kind == "O":
            raise TypeError(
                f"Column {name!r} has object dtype; only fixed-size dtypes are "
                f"supported (cast to a NumPy dtype, e.g. float64 or a fixed-width "
                f"string like 'S16'/'U16', first)."
            )
        if array.dtype.kind not in _SUPPORTED_KINDS:
            raise TypeError(
                f"Column {name!r} has unsupported dtype kind {array.dtype.kind!r} "
                f"({array.dtype}); supported kinds are {sorted(_SUPPORTED_KINDS)}."
            )
        if array.shape[0] != n_rows:
            raise ValueError(f"Column {name!r} has {array.shape[0]} rows; expected {n_rows}.")

    little_endian_columns = {name: _to_little_endian(columns[name]) for name in column_names}
    columns_meta = [
        {
            "name": name,
            "dtype": little_endian_columns[name].dtype.str,
            "encoding": _DEFAULT_ENCODING,
            "nullable": _DEFAULT_NULLABLE,
        }
        for name in column_names
    ]

    if expected_schema is not None:
        if len(expected_schema) != len(columns_meta):
            raise ValueError(
                f"Schema mismatch: file has {len(expected_schema)} columns, "
                f"got {len(columns_meta)}."
            )
        for expected, actual in zip(expected_schema, columns_meta, strict=True):
            if expected["name"] != actual["name"]:
                raise ValueError(
                    f"Column name mismatch: file expects {expected['name']!r}, "
                    f"got {actual['name']!r}."
                )
            if expected["dtype"] != actual["dtype"]:
                raise ValueError(
                    f"Column {expected['name']!r}: file dtype {expected['dtype']!r} "
                    f"does not match write dtype {actual['dtype']!r}."
                )

    return column_names, n_rows, little_endian_columns, columns_meta


def write_dataset(
    columns: dict[str, np.ndarray[Any, np.dtype[Any]]],
    path: PathLike,
    *,
    batch_size: int | None,
    show_progress: bool,
) -> None:
    """Serialize a dict of 1D NumPy columns to disk in colstore format.

    Writes a single-record file: file header + 32B record header + column-
    major body + 8B padding. For multi-record streaming writes, use
    :class:`colstore.ColStoreWriter`.

    Parameters
    ----------
    columns : dict[str, numpy.ndarray]
        Mapping of column name to a 1D fixed-size array. All columns must share
        the same length.
    path : str or os.PathLike
        Destination path.
    batch_size : int or None
        Number of rows written per ``tofile`` call, used only to drive the
        progress bar. ``None`` or any value ``<= 0`` writes each column in a
        single call (no batching). Has no effect on the bytes written.
    show_progress : bool
        Whether to display a tqdm progress bar.
    """
    column_names, n_rows, little_endian_columns, columns_meta = normalize_columns(columns)

    # ``None`` or any non-positive value means "write each column in one call".
    effective_batch = batch_size if (batch_size is not None and batch_size > 0) else 0
    if effective_batch > 0:
        total_units = (-(-n_rows // effective_batch)) * len(column_names)
        unit = "batch"
    else:
        total_units = len(column_names)
        unit = "col"

    with (
        open(path, "wb") as output_file,
        progress_bar(
            total_units, desc="Writing colstore", unit=unit, enabled=show_progress
        ) as progress,
    ):
        write_header(output_file, columns_meta, n_records=1, committed_rows=n_rows)
        # Single record at index 0 wrapping the entire dataset. Reads of this
        # file take the fast path (per-column memmaps, contiguous gather).
        write_record_header(output_file, record_index=0, n_rows=n_rows)
        body_bytes = 0
        for name in column_names:
            array = little_endian_columns[name]
            if effective_batch <= 0:
                array.tofile(output_file)
                body_bytes += array.nbytes
                progress.update(1)
                continue
            for batch_start in range(0, n_rows, effective_batch):
                batch_end = min(batch_start + effective_batch, n_rows)
                chunk = array[batch_start:batch_end]
                chunk.tofile(output_file)
                body_bytes += chunk.nbytes
                progress.update(1)
        # Pad the record body up to _RECORD_BODY_ALIGNMENT so any future
        # record (if this file were later opened for append) would start at
        # a naturally-aligned offset.
        pad = align_up(body_bytes, _RECORD_BODY_ALIGNMENT) - body_bytes
        if pad:
            output_file.write(b"\x00" * pad)
