"""On-disk file format for colstore.

File layout::

    [magic 8B][manifest_len 8B (u64 little-endian)][manifest_json]
    [zero-padding to 64-byte alignment]
    [column_0 raw bytes][column_1 raw bytes]...[column_n raw bytes]

The manifest is a small JSON object that records ``format_version``,
``n_rows``, and per-column ``{name, dtype}``. Column dtypes are preserved
exactly; columns are stored back-to-back with no per-row overhead. The
canonical file extension is ``.cstore``.

The 8-byte magic ``b"CSTORE\\x00\\x01"`` spells the format name (6 ASCII
bytes) followed by two reserved bytes; the bytes are constant for the life
of the format. Per-instance evolution is tracked via ``format_version``
inside the manifest, not by changing the magic.
"""

import json
import os
import struct
from typing import IO, Any

import numpy as np

FILE_EXTENSION = ".cstore"
_MAGIC = b"CSTORE\x00\x01"
_MANIFEST_LEN_FMT = "<Q"
_MANIFEST_LEN_SIZE = struct.calcsize(_MANIFEST_LEN_FMT)
_ALIGNMENT = 64
_FORMAT_VERSION = 1

# Path-like accepted by every public function in this module.
PathLike = str | os.PathLike[str]

ColumnLayout = dict[str, tuple[int, np.dtype[Any]]]


class FormatError(Exception):
    """Raised when a file does not match the expected colstore format."""


def align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    """Round `value` up to the next multiple of `alignment`."""
    return ((value + alignment - 1) // alignment) * alignment


def write_header(
    file: IO[bytes],
    columns_meta: list[dict[str, Any]],
    n_rows: int,
) -> int:
    """Write magic + manifest + padding; return the data start offset."""
    manifest = {
        "format_version": _FORMAT_VERSION,
        "n_rows": n_rows,
        "columns": columns_meta,
    }
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    header_size = len(_MAGIC) + _MANIFEST_LEN_SIZE + len(manifest_bytes)
    data_offset = align_up(header_size)

    file.write(_MAGIC)
    file.write(struct.pack(_MANIFEST_LEN_FMT, len(manifest_bytes)))
    file.write(manifest_bytes)
    file.write(b"\x00" * (data_offset - header_size))
    return data_offset


def read_header(path: PathLike) -> tuple[dict[str, Any], int]:
    """Read magic and manifest; return ``(manifest_dict, data_start_offset)``."""
    with open(path, "rb") as input_file:
        magic = input_file.read(len(_MAGIC))
        if magic != _MAGIC:
            raise FormatError(f"Not a colstore file: expected magic {_MAGIC!r}, got {magic!r}")
        manifest_size = struct.unpack(_MANIFEST_LEN_FMT, input_file.read(_MANIFEST_LEN_SIZE))[0]
        manifest = json.loads(input_file.read(manifest_size))
    header_size = len(_MAGIC) + _MANIFEST_LEN_SIZE + manifest_size
    return manifest, align_up(header_size)


def build_column_layout(manifest: dict[str, Any], data_offset: int) -> ColumnLayout:
    """Compute per-column ``(byte_offset, dtype)`` from manifest and start offset."""
    layout: ColumnLayout = {}
    n_rows = manifest["n_rows"]
    current_offset = data_offset
    for column_info in manifest["columns"]:
        column_dtype = np.dtype(column_info["dtype"])
        layout[column_info["name"]] = (current_offset, column_dtype)
        current_offset += n_rows * column_dtype.itemsize
    return layout


def write_dataset(
    columns: dict[str, np.ndarray[Any, np.dtype[Any]]],
    path: PathLike,
    *,
    batch_size: int,
    show_progress: bool,
) -> None:
    """Serialize a dict of 1D NumPy columns to disk in colstore format."""
    from tqdm.auto import tqdm

    if not columns:
        raise ValueError("Cannot write an empty column mapping.")

    column_names = list(columns)
    n_rows = columns[column_names[0]].shape[0]
    for name, array in columns.items():
        if array.ndim != 1:
            raise ValueError(f"Column {name!r} must be 1D; got {array.ndim}D.")
        if array.dtype.kind == "O":
            raise TypeError(
                f"Column {name!r} has object dtype; " f"only fixed-size dtypes are supported."
            )
        if array.shape[0] != n_rows:
            raise ValueError(f"Column {name!r} has {array.shape[0]} rows; expected {n_rows}.")

    columns_meta = [{"name": name, "dtype": columns[name].dtype.str} for name in column_names]
    total_batches = (-(-n_rows // batch_size)) * len(column_names)

    with (
        open(path, "wb") as output_file,
        tqdm(
            total=total_batches,
            desc="Writing colstore",
            unit="batch",
            disable=not show_progress,
        ) as progress,
    ):
        write_header(output_file, columns_meta, n_rows)
        for name in column_names:
            array = columns[name]
            for batch_start in range(0, n_rows, batch_size):
                batch_end = min(batch_start + batch_size, n_rows)
                array[batch_start:batch_end].tofile(output_file)
                progress.update(1)
