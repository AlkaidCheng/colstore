"""On-disk file format for colstore.

File layout::

    [magic 8B: b"CSTORE\\x00\\x01" -- constant for the life of the format]
    [counters 32B: n_records(8) + committed_rows(8) + crc32(4) + reserved(12)]
    [manifest_len 8B (u64 little-endian)]
    [manifest_json: format_version + columns + manifest_crc32]
    [zero-padding to 64-byte alignment]
    [record_0 header 32B][record_0 body, padded to 8B]
    [record_1 header 32B][record_1 body, padded to 8B]
    ...

The mutable counters (``n_records``, ``committed_rows``, own CRC32) sit at
fixed offset 8, separate from the immutable JSON manifest
(``format_version`` and per-column ``{name, dtype, encoding, nullable}``).
This split lets the writer commit a session atomically: the 32-byte
counters block is rewritten in place without shifting any record byte
offsets. Format evolution is tracked via ``format_version`` in the
manifest, not by changing the magic. The canonical file extension is
``.cstore``.

A file is a sequence of one or more records, each a 32-byte header plus a
column-major body (columns back-to-back in schema declaration order, no
inter-column padding) padded to 8 bytes so the next header is naturally
aligned. One-shot writes (:func:`colstore.store`) produce a single-record
file; :class:`ColStoreWriter` produces multi-record files. Each record
header is::

    4B  magic b"REC\\x01"
    8B  record_index (i64, sequential from 0)
    8B  n_rows       (i64)
    8B  reserved
    4B  header CRC32 (over the first 28 bytes)

Supported column dtypes are the fixed-size NumPy kinds: floating point,
signed/unsigned integers, booleans, ``datetime64``/``timedelta64``, and
fixed-width strings (``S`` bytes and ``U`` unicode). Object/variable-length
columns are rejected. Column bytes are always written **little-endian** for
cross-host portability; big-endian hosts byte-swap on write and read, so
callers always see native-order arrays.
"""

import contextlib
import json
import os
import struct
import sys
import tempfile
import time
import zlib
from typing import IO, Any

import numpy as np

from . import _numa, config
from ._sizes import parse_byte_size
from .frame import Expr, evaluate, fusible_passthroughs, validate_length
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

    Returns three small int64 arrays (24 bytes per record total; built from
    ``R`` 32-byte reads):

    * ``record_starts_rows`` shape ``(R+1,)`` -- cumulative row counts,
      used to bin row indices to records.
    * ``record_starts_bytes`` shape ``(R,)`` -- file byte offset of each
      record's body (past the 32B header).
    * ``n_rows_per_record`` shape ``(R,)`` -- per-record row counts, used
      to compute per-column offsets within a record
      (``column_prefix_bytes[j] * n_rows_per_record[record_id]``).

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

    Seeks to the counters block first, so callers don't need to know
    where it lives on disk. Used by :meth:`ColStoreWriter.close` to
    commit the new record count and row total atomically. The 32-byte
    block is small enough that a single ``write()`` is generally atomic
    on common filesystems; even if it isn't, the embedded CRC catches a
    torn write on the next open.
    """
    file.seek(_COUNTERS_OFFSET)
    file.write(_pack_counters(n_records, committed_rows))


def record_header_bytes(record_index: int, n_rows: int) -> bytes:
    """Serialize one 32-byte record header.

    The header CRC32 covers the first 28 bytes (everything but the CRC slot);
    a corrupt header is detected on read even if only the in-place fields
    were tampered with. Exposed separately from :func:`write_record_header`
    so the streaming writer can place the header into a vectored write
    alongside the record body instead of issuing a separate ``write()``.
    """
    header_prefix = struct.pack(
        "<4sqqq",
        _RECORD_MAGIC,
        record_index,
        n_rows,
        0,  # reserved
    )
    crc = zlib.crc32(header_prefix) & 0xFFFFFFFF
    return header_prefix + struct.pack("<I", crc)


def write_record_header(file: IO[bytes], record_index: int, n_rows: int) -> None:
    """Write a 32-byte record header to ``file`` at its current position.

    Used by :func:`write_dataset` for the single-record write path and as
    the no-``writev`` fallback in :class:`ColStoreWriter`. See
    :func:`record_header_bytes` for the layout.
    """
    file.write(record_header_bytes(record_index, n_rows))


def read_header(path: PathLike) -> tuple[dict[str, Any], int]:
    """Read and validate the file header; return ``(header_dict, data_start_offset)``.

    The header_dict merges the immutable manifest fields
    (``format_version``, ``columns``, ``manifest_crc32``) with the mutable
    counters (``n_records``, ``committed_rows``); callers need not know
    they live in separate on-disk regions. Only the file header is
    validated (magic, counters CRC, format version, manifest CRC);
    per-record headers and truncation past the file header are validated
    by :func:`read_record_index`, which the caller runs next. Callers that
    already hold an open handle (the update-mode writer, the compactor
    under its lock) should use :func:`read_header_from_file` instead of
    opening a second one.

    Raises
    ------
    FormatError
        On wrong magic, counters CRC mismatch, unsupported
        ``format_version``, or manifest CRC mismatch.
    """
    with open(path, "rb") as input_file:
        return read_header_from_file(input_file)


def read_header_from_file(input_file: IO[bytes]) -> tuple[dict[str, Any], int]:
    """Variant of :func:`read_header` that reads from an already-open file.

    Reads starting from the current file position. The position is left
    just past the manifest JSON; the caller can seek wherever it needs
    next. Used by :class:`ColStoreWriter` in update mode so the header
    read goes through the writer's own fd (the one that holds the
    byte-0 lock on Windows).
    """
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

    ``body_offset`` is the file offset of the record body's first byte
    (past the 32-byte record header), obtained from
    :func:`read_record_index` so the layout sits on an already-validated
    record. Columns are laid out back-to-back in schema declaration order
    with no inter-column padding; the reader builds one ``np.memmap`` per
    column from the result.

    Valid only for the single-record fast path: multi-record files have no
    contiguous per-column layout, and the reader computes byte addresses
    on the fly instead. The returned dtype is the on-disk dtype
    (little-endian for multi-byte kinds); memmaps must use it to interpret
    the bytes, and the gather path presents native-order arrays to callers.
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


# ---- Batch size resolution -----------------------------------------------
#
# ``write_dataset``'s ``batch_size`` parameter is polymorphic:
#
#   * ``None`` -- single pass (one tofile call per column, no batching).
#   * ``int N`` -- "rows x cols per logical batch". Per inner step (which
#     writes one column at a time), ``rows_per_step = N // n_columns``.
#   * ``str "100 MB"`` / ``"1.5 GiB"`` etc. -- bytes per progress step.
#     Each column's rows-per-step derives from its own itemsize so the
#     byte granularity stays uniform across columns of different dtypes.
#   * ``str "auto"`` (default in the public ``store`` API) -- adaptive.
#     Probe with a 1 MiB initial batch, then size subsequent batches from
#     an EMA-smoothed bandwidth estimate, with a 2x growth-rate cap per
#     batch. The growth-rate cap is what bounds the impact of a single
#     wildly-wrong measurement (so a bad probe doubles the next batch,
#     not 1000x's it); the EMA smooths transient spikes once we're in
#     steady state. No upper bound on batch size -- the medium decides.
#     For datasets under _AUTO_MIN_TOTAL_FOR_BATCHING bytes, auto
#     degrades to single-pass because batching overhead dominates.
#
# Size strings follow IEC 80000-13 units (see colstore._sizes): decimal
# ``kB``/``MB``/``GB`` are powers of 1000 and binary ``KiB``/``MiB``/``GiB``
# are powers of 1024, so "1 MB" is 1,000,000 B and "1 MiB" is 1,048,576 B.

# Below this total size, auto goes single-pass -- batching overhead is
# bigger than the I/O savings.
_AUTO_MIN_TOTAL_FOR_BATCHING = 16 * 1024 * 1024  # 16 MiB

# Initial probe batch size in adaptive mode. Small enough that a wildly
# wrong measurement (e.g., from OS page-cache absorption) on the first
# batch is bounded -- the next batch is at most _AUTO_GROWTH_RATE * 1 MiB,
# not gigabytes. Large enough to amortize syscall overhead.
_AUTO_INITIAL_BYTES = 1 * 1024 * 1024  # 1 MiB

# Steady-state target time per batch in adaptive mode. 0.5s gives a smooth
# progress bar (~2 updates/sec) while keeping per-batch overhead negligible.
_AUTO_TARGET_SECONDS = 0.5

# Floor for adaptive batch size. Below this, syscall overhead dominates
# real I/O work. There is intentionally NO upper cap -- the medium decides
# steady-state batch size, and the growth-rate cap (below) protects
# against a single wildly-wrong bandwidth estimate.
_AUTO_MIN_BYTES_PER_BATCH = 1 * 1024 * 1024  # 1 MiB

# Maximum factor by which each successive batch can grow vs. the previous
# one. Together with the EMA smoothing below, this forms a TCP-slow-start-
# like ramp: a bad first measurement only doubles the next batch (not
# 1000x), and several measurements converge on the true bandwidth before
# we commit to large batches.
_AUTO_GROWTH_RATE = 2.0

# Exponential moving average weight on the most recent measurement when
# updating the bandwidth estimate. Higher = more responsive to changing
# conditions, lower = smoother. 0.5 weights the most recent batch equally
# with the smoothed history -- adapts quickly during ramp-up while still
# damping transient spikes once in steady state.
_AUTO_EMA_ALPHA = 0.5


def _resolve_rows_per_step(
    batch_size: int | str | None,
    *,
    n_rows: int,
    n_columns: int,
    total_bytes: int,
    column_itemsizes: list[int],
) -> list[int]:
    """Resolve a fixed-size ``batch_size`` to per-column rows-per-progress-step.

    Caller must NOT pass ``"auto"`` here -- the adaptive path uses a
    different code path (see ``write_dataset``). This function handles
    ``None``, ``int``, and concrete size strings only.

    Returns a list of length ``n_columns`` giving the chunk size (in rows)
    to use for each column's write loop. A value ``>= n_rows`` means
    "single-pass for this column" (one tofile call, one progress update).
    """
    if n_rows == 0 or batch_size is None:
        return [n_rows] * n_columns

    # bool is a subclass of int in Python; reject it explicitly to avoid
    # confusing behavior like batch_size=True silently meaning 1.
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, str)):
        raise TypeError(f"batch_size must be int, str, or None; got {type(batch_size).__name__}.")

    if isinstance(batch_size, int):
        if batch_size <= 0:
            return [n_rows] * n_columns
        # User's "rows x cols per logical batch" semantics: each inner step
        # writes one column, so rows_per_step = batch_size / n_columns.
        rows_per_step = max(1, batch_size // max(1, n_columns))
        return [rows_per_step] * n_columns

    # str (caller has already filtered out "auto")
    bytes_per_step = parse_byte_size(batch_size.strip())
    # Per-column rows-per-step: each column gets enough rows to fill roughly
    # ``bytes_per_step`` bytes, so a wide-dtype column does fewer rows per
    # step than a narrow-dtype one. Uniform byte granularity across columns.
    return [max(1, bytes_per_step // max(1, itemsize)) for itemsize in column_itemsizes]


# ---- Throughput formatting -----------------------------------------------
#
# Used by ``write_dataset`` to attach a human-readable rows/s line to the
# progress bar postfix. (Bytes/s is rendered natively by tqdm because the
# bar is byte-counted, with ``unit="B"`` and ``unit_scale=True``.)


def _format_rows_per_sec(rows_per_sec: float) -> str:
    """Auto-scaled rows/s with a space between number and unit.

    Output: ``'1.25 Mrows/s'``, ``'125.00 Krows/s'``, ``'850 rows/s'``.
    The unit name follows SI convention -- prefix (K/M/G) sits flush with
    the noun (no space inside the unit), space goes between number and
    unit.
    """
    if rows_per_sec >= 1_000_000_000:
        return f"{rows_per_sec / 1_000_000_000:.2f} Grows/s"
    if rows_per_sec >= 1_000_000:
        return f"{rows_per_sec / 1_000_000:.2f} Mrows/s"
    if rows_per_sec >= 1_000:
        return f"{rows_per_sec / 1_000:.2f} Krows/s"
    return f"{rows_per_sec:.0f} rows/s"


def write_dataset(
    columns: dict[str, np.ndarray[Any, np.dtype[Any]]],
    path: PathLike,
    *,
    batch_size: int | str | None,
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
    batch_size : int, str, or None
        Write chunking for the progress bar; no effect on the bytes
        written. Semantics are documented on :func:`colstore.store`.
    show_progress : bool
        Whether to display a tqdm progress bar. The bar's postfix shows
        cumulative throughput as ``rows=...Mrows/s, data=...MB/s``.
    """
    column_names, n_rows, little_endian_columns, columns_meta = normalize_columns(columns)
    column_itemsizes = [little_endian_columns[name].dtype.itemsize for name in column_names]
    total_bytes = sum(little_endian_columns[name].nbytes for name in column_names)
    n_columns = len(column_names)

    # Decide between adaptive ("auto" on a non-trivial dataset) and fixed
    # (everything else, including "auto" on a tiny dataset).
    is_auto_request = isinstance(batch_size, str) and batch_size.strip().lower() == "auto"
    use_adaptive = is_auto_request and total_bytes >= _AUTO_MIN_TOTAL_FOR_BATCHING

    fixed_rows_per_step: list[int] | None = None
    if not use_adaptive:
        # Either non-auto, or auto on a tiny dataset (degrades to single-pass).
        resolved_batch_size: int | str | None = None if is_auto_request else batch_size
        fixed_rows_per_step = _resolve_rows_per_step(
            resolved_batch_size,
            n_rows=n_rows,
            n_columns=n_columns,
            total_bytes=total_bytes,
            column_itemsizes=column_itemsizes,
        )

    # Progress bar is byte-counted: total = total_bytes, each batch updates
    # by chunk_bytes. tqdm's native unit_scale renders this as e.g.
    # "47.5MB/200MB [00:01<00:03, 50.2MB/s]" -- the bar fill, percentage,
    # and rate all derive directly from bytes written / total bytes, so
    # there is no estimation phase, no "?" total, and the bar fill always
    # matches the displayed percentage. The adaptive batch sizing (below)
    # is unchanged; it just affects HOW MANY tofile() calls happen, not
    # WHAT is shown to the user.

    with (
        open(path, "wb") as output_file,
        progress_bar(
            total_bytes,
            desc="Writing colstore",
            unit="B",
            unit_scale=True,
            enabled=show_progress,
        ) as progress,
        # Wrap body writes in MPOL_INTERLEAVE on multi-node Linux so the
        # kernel distributes page-cache pages across NUMA nodes at write
        # time. Identical semantics to ColStoreWriter.write -- the policy
        # gate lives in _numa.writer_policy_scope so this path and the
        # streaming path agree. The one-shot colstore.store() (which is
        # what most callers use) routes through here, so this is the
        # call site that actually fixes the warm-cache NUMA placement
        # the original benchmark identified.
        _numa.writer_policy_scope(),
    ):
        write_header(output_file, columns_meta, n_records=1, committed_rows=n_rows)
        # Single record at index 0 wrapping the entire dataset. Reads of this
        # file take the fast path (per-column memmaps, contiguous gather).
        write_record_header(output_file, record_index=0, n_rows=n_rows)
        body_bytes = 0
        cum_rows = 0
        cum_batches = 0
        start_time = time.monotonic()
        # Adaptive-mode state: only used when use_adaptive is True. The
        # EMA bandwidth starts None (initialized from the first measurement);
        # current_bytes_per_batch ramps via the growth-rate cap.
        current_bytes_per_batch = _AUTO_INITIAL_BYTES
        bandwidth_ema: float | None = None

        for col_idx, name in enumerate(column_names):
            array = little_endian_columns[name]
            itemsize = column_itemsizes[col_idx]

            if n_rows == 0:
                # Zero-row column: emit a single zero-byte write. Nothing
                # to add to the byte-counted progress bar.
                array.tofile(output_file)
                continue

            offset = 0
            while offset < n_rows:
                if use_adaptive:
                    chunk_rows = max(1, current_bytes_per_batch // max(1, itemsize))
                else:
                    assert fixed_rows_per_step is not None
                    rps = fixed_rows_per_step[col_idx]
                    chunk_rows = rps if 0 < rps < n_rows else n_rows - offset
                chunk_rows = min(chunk_rows, n_rows - offset)

                t_batch_start = time.monotonic()
                chunk = array[offset : offset + chunk_rows]
                chunk.tofile(output_file)
                t_batch_end = time.monotonic()

                chunk_bytes = chunk.nbytes
                body_bytes += chunk_bytes
                cum_rows += chunk_rows
                cum_batches += 1
                offset += chunk_rows

                # Update adaptive state for the next batch. Two-stage smoothing:
                #
                # 1. EMA on bandwidth: bandwidth_ema combines the current
                #    measurement (weight alpha) with the running estimate
                #    (weight 1-alpha). Damps single-batch noise.
                #
                # 2. Growth-rate cap on batch size: each batch can grow by
                #    at most _AUTO_GROWTH_RATE x the previous one. Even an
                #    extreme bandwidth estimate (e.g., a transient cache hit
                #    making the first probe look 1000x faster than reality)
                #    only doubles the next batch, giving us another
                #    measurement before we commit to a big batch.
                if use_adaptive:
                    chunk_elapsed = t_batch_end - t_batch_start
                    if chunk_elapsed > 0:
                        measured_bw = chunk_bytes / chunk_elapsed
                        if bandwidth_ema is None:
                            bandwidth_ema = measured_bw
                        else:
                            bandwidth_ema = (
                                _AUTO_EMA_ALPHA * measured_bw
                                + (1.0 - _AUTO_EMA_ALPHA) * bandwidth_ema
                            )
                        target_bytes = int(bandwidth_ema * _AUTO_TARGET_SECONDS)
                        growth_cap = int(current_bytes_per_batch * _AUTO_GROWTH_RATE)
                        current_bytes_per_batch = max(
                            _AUTO_MIN_BYTES_PER_BATCH,
                            min(target_bytes, growth_cap),
                        )

                # Postfix shows rows/s and the batch count. Bytes/s and the
                # bar fill come from tqdm itself via unit="B"/unit_scale=True.
                elapsed = t_batch_end - start_time
                if elapsed > 0:
                    progress.set_postfix(
                        batches=str(cum_batches),
                        rows=_format_rows_per_sec(cum_rows / elapsed),
                    )
                progress.update(chunk_bytes)

        # Pad the record body up to _RECORD_BODY_ALIGNMENT so any future
        # record (if this file were later opened for append) would start at
        # a naturally-aligned offset.
        pad = align_up(body_bytes, _RECORD_BODY_ALIGNMENT) - body_bytes
        if pad:
            output_file.write(b"\x00" * pad)


def _resolve_streaming_layout(
    specs: dict[str, Expr],
) -> tuple[list[dict[str, Any]], dict[str, np.dtype[Any]], int]:
    """Resolve the on-disk schema and per-row byte cost for a streaming write.

    Runs one zero-length pass over every column expression through a *shared*
    memo, exactly as a real batch does. That yields, for free and without
    reading data, each column's output dtype and -- because the shared memo
    holds one entry per distinct node materialized across all columns -- the
    total bytes a single batch row occupies in RAM (the sum of every distinct
    node's itemsize, since the per-batch memo keeps each computed array live
    until the batch ends). Returns ``(columns_meta, on_disk_dtypes,
    bytes_per_row)``.
    """
    probe_memo: dict[tuple[Any, ...], np.ndarray[Any, np.dtype[Any]]] = {}
    columns_meta: list[dict[str, Any]] = []
    on_disk_dtypes: dict[str, np.dtype[Any]] = {}
    for name, spec in specs.items():
        root = evaluate(spec, 0, 0, probe_memo)
        kind = root.dtype.kind
        if kind == "O":
            raise TypeError(
                f"Column {name!r} resolves to object dtype; only fixed-size dtypes "
                f"are supported."
            )
        if kind not in _SUPPORTED_KINDS:
            raise TypeError(
                f"Column {name!r} resolves to unsupported dtype kind {kind!r} "
                f"({root.dtype}); supported kinds are {sorted(_SUPPORTED_KINDS)}."
            )
        disk_dtype = _to_little_endian(root).dtype
        on_disk_dtypes[name] = disk_dtype
        columns_meta.append(
            {
                "name": name,
                "dtype": disk_dtype.str,
                "encoding": _DEFAULT_ENCODING,
                "nullable": _DEFAULT_NULLABLE,
            }
        )
    bytes_per_row = sum(int(array.itemsize) for array in probe_memo.values())
    return columns_meta, on_disk_dtypes, bytes_per_row


def _release_memmap(view: np.memmap[Any, np.dtype[Any]]) -> None:
    """Close a memmap's underlying mapping so the file can be renamed/reused.

    An open mapping blocks the rename on Windows and pins pages everywhere.
    """
    mmap_obj = getattr(view, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


def _fill_streaming(
    tmp_path: str,
    specs: dict[str, Expr],
    names: list[str],
    on_disk_dtypes: dict[str, np.dtype[Any]],
    body_offset: int,
    n_rows: int,
    batch_rows: int,
) -> None:
    """Fill a preallocated file body via per-column memmaps, one batch at a time.

    Each batch opens a fresh memo shared across all columns, so a subexpression
    used by several columns is computed once for the batch and released when the
    batch ends. Writes are scattered across the column-contiguous regions; the
    file is preallocated, so each column lands at its fixed offset. Byte order
    is normalized to little-endian on assignment into the on-disk-typed view (a
    no-op on a little-endian host).
    """
    views: dict[str, np.memmap[Any, np.dtype[Any]]] = {}
    offset = body_offset
    try:
        for name in names:
            dtype = on_disk_dtypes[name]
            views[name] = np.memmap(
                tmp_path, dtype=dtype, mode="r+", offset=offset, shape=(n_rows,)
            )
            offset += n_rows * dtype.itemsize
        fusible = fusible_passthroughs(specs)
        for start in range(0, n_rows, batch_rows):
            stop = min(start + batch_rows, n_rows)
            memo: dict[tuple[Any, ...], np.ndarray[Any, np.dtype[Any]]] = {}
            for name in names:
                target = views[name][start:stop]
                passthrough = fusible.get(name)
                if passthrough is not None:
                    # A plain native passthrough: fill the output region straight
                    # from the source, skipping the intermediate array (and the
                    # per-batch memo it would otherwise occupy).
                    passthrough._fill_into(target, start, stop)
                else:
                    target[:] = evaluate(specs[name], start, stop, memo)
        for name in names:
            views[name].flush()
    finally:
        # Release the mappings before the caller renames the file.
        for view in views.values():
            _release_memmap(view)
        views.clear()


# One merge-copy run: (source path, source byte offset, destination byte offset,
# byte count). A plan is these runs in destination-write order; copied in order
# they fill the body.
CopyRun = tuple[PathLike, int, int, int]

# Override for the merge-copy strategy, for benchmarking only (not public API).
# ``None`` autodetects (copy_file_range on Linux, else mmap); "mmap" or "cfr"
# forces that strategy regardless of platform.
_MERGE_COPY_OVERRIDE: str | None = None


def _merge_copy_plan(
    specs: dict[str, Expr],
    names: list[str],
    on_disk_dtypes: dict[str, np.dtype[Any]],
    body_offset: int,
    n_rows: int,
) -> list[CopyRun] | None:
    """Plan a raw byte-copy for a pure passthrough merge, or ``None``.

    A pure merge is one where every output column is a bare, unshared,
    native-dtype passthrough, so the destination body is the sources' column
    bytes concatenated. For such a write, returns a flat list of
    ``(src_path, src_offset, dst_offset, n_bytes)`` runs that, copied in order,
    fill the preallocated body byte-for-byte -- identical to the materializing
    streaming write, including the trailing body padding the preallocation
    leaves zero. Returns ``None`` for anything else, and the caller falls back
    to the materializing write.
    """
    passthroughs = fusible_passthroughs(specs)
    if len(passthroughs) != len(specs):
        return None
    plan: list[CopyRun] = []
    dst_column_offset = body_offset
    for name in names:
        column_bytes = n_rows * on_disk_dtypes[name].itemsize
        try:
            runs = passthroughs[name]._disk_runs()
        except ValueError:
            return None  # non-native source column: not raw-copyable
        if sum(nbytes for _, _, nbytes in runs) != column_bytes:
            return None  # run total disagrees with the column size; fall back
        write = dst_column_offset
        for src_path, src_offset, nbytes in runs:
            plan.append((src_path, src_offset, write, nbytes))
            write += nbytes
        dst_column_offset += column_bytes
    return plan


def _copy_plan_mmap(dst_path: str, plan: list[CopyRun]) -> None:
    """Fill the destination body by mmap memcpy, one run at a time.

    Maps the destination once and each distinct source once; the runs are
    disjoint and cover the body, so a single sequential pass fills it. The
    mappings are released before the caller renames the file (an open mapping
    blocks the rename on Windows and pins pages everywhere).
    """
    dst = np.memmap(dst_path, dtype=np.uint8, mode="r+")
    src_maps: dict[str, np.memmap[Any, np.dtype[Any]]] = {}
    try:
        for src_path, src_offset, dst_offset, nbytes in plan:
            key = os.fspath(src_path)
            src = src_maps.get(key)
            if src is None:
                src = np.memmap(key, dtype=np.uint8, mode="r")
                src_maps[key] = src
            dst[dst_offset : dst_offset + nbytes] = src[src_offset : src_offset + nbytes]
        dst.flush()
    finally:
        _release_memmap(dst)
        for src in src_maps.values():
            _release_memmap(src)


def _copy_plan_copy_file_range(dst_path: str, plan: list[CopyRun]) -> None:
    """Fill the destination body with ``copy_file_range`` (Linux only).

    Each run is copied range-to-range without a user-space bounce; on reflink
    filesystems and networked stores the kernel can share extents or copy
    server-side, so the bytes need not transit the client at all. Raises
    ``OSError`` if the platform or filesystem does not support it, which the
    caller catches to fall back to :func:`_copy_plan_mmap`.
    """
    if sys.platform != "linux":
        # copy_file_range is Linux-only; signal the caller to fall back to mmap.
        raise OSError("copy_file_range is unavailable on this platform.")
    src_fds: dict[str, int] = {}
    dst_fd = os.open(dst_path, os.O_WRONLY)
    try:
        for src_path, src_offset, dst_offset, nbytes in plan:
            key = os.fspath(src_path)
            src_fd = src_fds.get(key)
            if src_fd is None:
                src_fd = os.open(key, os.O_RDONLY)
                src_fds[key] = src_fd
            remaining, read_at, write_at = nbytes, src_offset, dst_offset
            while remaining:
                copied = os.copy_file_range(
                    src_fd, dst_fd, remaining, offset_src=read_at, offset_dst=write_at
                )
                if copied == 0:
                    raise OSError("copy_file_range made no progress (short source).")
                remaining -= copied
                read_at += copied
                write_at += copied
    finally:
        os.close(dst_fd)
        for fd in src_fds.values():
            os.close(fd)


def _execute_copy_plan(dst_path: str, plan: list[CopyRun]) -> None:
    """Run a merge-copy plan with the best strategy for this platform.

    Uses ``copy_file_range`` on Linux, where reflink and networked filesystems
    can complete the copy as a near-metadata-only operation, and falls back to
    an mmap memcpy elsewhere or when the kernel call is unsupported.
    ``_MERGE_COPY_OVERRIDE`` forces a strategy for benchmarking.
    """
    strategy = _MERGE_COPY_OVERRIDE or ("cfr" if sys.platform == "linux" else "mmap")
    if strategy == "cfr":
        try:
            _copy_plan_copy_file_range(dst_path, plan)
            return
        except OSError:
            # Unsupported platform/kernel/filesystem: fall back. The mmap pass
            # rewrites the whole body, so any partial progress is overwritten.
            pass
    _copy_plan_mmap(dst_path, plan)


def write_dataset_streaming(
    specs: dict[str, Expr],
    n_rows: int,
    path: PathLike,
    *,
    memory_budget: int | None = None,
) -> None:
    """Serialize lazily-produced columns to a single-record ``.cstore`` file.

    Unlike :func:`write_dataset`, which copies fully-materialized arrays, this
    sink evaluates each output column one row range at a time and writes the
    result straight into a memory-mapped, preallocated file, so the whole
    dataset is never resident at once. Peak RAM is bounded by ``memory_budget``
    (bytes; ``None`` uses :func:`colstore.config.get_default_memory_budget`):
    the batch row count is chosen so the live arrays for one row-range pass over
    all columns fit the budget. The memory-mapped output is file-backed and is
    not counted against it.

    ``specs`` maps each output column name to an expression (see
    :mod:`colstore.frame`); insertion order is the on-disk column order. Every
    column must produce exactly ``n_rows`` rows -- constant/scalar columns adopt
    ``n_rows``, sized columns must match it (checked before any byte is
    written). The file is built at a sibling temporary path and atomically
    renamed into place on success, so a failure part-way through -- including an
    error raised while evaluating a transform -- leaves any existing destination
    untouched.
    """
    if not specs:
        raise ValueError("Cannot write an empty column mapping.")
    if n_rows < 0:
        raise ValueError(f"n_rows must be >= 0, got {n_rows}.")

    names = list(specs)
    for name in names:
        validate_length(specs[name], n_rows)

    columns_meta, on_disk_dtypes, bytes_per_row = _resolve_streaming_layout(specs)

    budget = config.get_default_memory_budget() if memory_budget is None else int(memory_budget)
    if budget < 1:
        raise ValueError(f"memory_budget must be >= 1 byte, got {budget}.")
    # bytes_per_row is >= 1 (every column has at least one leaf of itemsize
    # >= 1), so the floor division below never divides by zero.
    batch_rows = max(1, min(n_rows, budget // bytes_per_row)) if n_rows else 0

    target = os.fspath(path)
    directory = os.path.dirname(target) or "."
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=f".{os.path.basename(target)}.", suffix=".tmp"
    )
    os.close(fd)
    try:
        with open(tmp_path, "wb") as output_file:
            write_header(output_file, columns_meta, n_records=1, committed_rows=n_rows)
            write_record_header(output_file, record_index=0, n_rows=n_rows)
            body_offset = output_file.tell()
            itemsizes = [on_disk_dtypes[name].itemsize for name in names]
            total_size = body_offset + record_body_size(n_rows, itemsizes)
            output_file.truncate(total_size)
            output_file.flush()
            os.fsync(output_file.fileno())

        if n_rows:
            # MPOL_INTERLEAVE on multi-node Linux while the body pages are
            # faulted in by the memmap writes -- same policy as write_dataset
            # and ColStoreWriter, via the shared gate in _numa.
            with _numa.writer_policy_scope():
                # A pure no-transform merge copies the sources' column bytes
                # straight into the body; anything else materializes per batch.
                plan = _merge_copy_plan(specs, names, on_disk_dtypes, body_offset, n_rows)
                if plan is not None:
                    _execute_copy_plan(tmp_path, plan)
                else:
                    _fill_streaming(
                        tmp_path, specs, names, on_disk_dtypes, body_offset, n_rows, batch_rows
                    )

        # Reopen read/write (not read-only) for the durability fsync: on Windows
        # os.fsync maps to _commit(), which fails with EBADF on a read-only
        # descriptor. The body was flushed through the memmaps and the header
        # through the writable handle above; this covers the whole file before
        # the atomic rename. All handles are closed before os.replace, which
        # Windows requires.
        with open(tmp_path, "r+b") as written:
            os.fsync(written.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
