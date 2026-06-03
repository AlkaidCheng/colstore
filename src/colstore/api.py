"""Module-level convenience functions: open, create, recreate, update, store,
compact, info, schema.

These thin wrappers around :class:`ColStoreReader` and :class:`ColStoreWriter`
give the package a uproot-style entry-point surface where each function does
one obvious thing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import format as fmt
from .compaction import compact_file
from .reader import ColStoreReader
from .writer import ColStoreWriter


def open(path: str | os.PathLike[str], **kwargs: Any) -> ColStoreReader:
    """Open an existing ``.cstore`` file for read.

    Equivalent to ``ColStoreReader(path, **kwargs)``. The file must exist and be
    a valid colstore file; otherwise :class:`FileNotFoundError` or
    :class:`FormatError` propagates.
    """
    return ColStoreReader(path, **kwargs)


def create(path: str | os.PathLike[str]) -> ColStoreWriter:
    """Open a new file for streaming writes; fail if it already exists.

    Use this when you want to be sure you are not overwriting anything.
    """
    return ColStoreWriter(path, mode="create")


def recreate(path: str | os.PathLike[str]) -> ColStoreWriter:
    """Open a file for streaming writes, truncating any existing content.

    Use this when you intentionally want to overwrite. To fail on
    overwrite instead, use :func:`create`.
    """
    return ColStoreWriter(path, mode="recreate")


def update(path: str | os.PathLike[str]) -> ColStoreWriter:
    """Open an existing file for append.

    The schema is loaded from the existing manifest; every :meth:`write`
    must match it exactly. Orphan bytes from a crashed prior writer (if
    any) are truncated on open. Raises :class:`FileNotFoundError` if the
    file does not exist.
    """
    return ColStoreWriter(path, mode="update")


def store(
    data: Any,
    path: str | os.PathLike[str],
    *,
    mode: str = "create",
    show_progress: bool = True,
    **open_kwargs: Any,
) -> ColStoreReader:
    """One-shot: write a single-record file and return an opened reader.

    Accepted ``data`` types:

    * ``dict[str, numpy.ndarray]`` -- column-major mapping.
    * Structured ``numpy.ndarray`` (``dtype.names`` non-None) -- one
      column per field.
    * pandas ``DataFrame`` -- one column per series.

    ``mode`` is ``"create"`` (default; fail if file exists) or
    ``"recreate"`` (truncate if exists). For multi-record streaming
    writes, use :func:`create` / :func:`recreate` / :func:`update`
    directly.

    Returns the opened :class:`ColStoreReader` for immediate use.
    """
    if mode not in ("create", "recreate"):
        raise ValueError(f"Invalid mode {mode!r} for store(); expected 'create' or 'recreate'.")

    columns = _coerce_to_columns(data)

    # write_dataset is the single-record writer that includes a progress bar;
    # ColStoreWriter is for multi-record streams. For one-shot writes, write_dataset
    # is slightly cheaper (no counter rewrite at the end -- the counters are
    # right the first time) and surfaces a progress bar.
    if mode == "create" and os.path.exists(path):
        raise FileExistsError(f"{path} already exists; use mode='recreate' to overwrite.")
    fmt.write_dataset(columns, path, batch_size=100_000, show_progress=show_progress)
    return ColStoreReader(path, **open_kwargs)


def _coerce_to_columns(
    data: Any,
) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    """Dispatch on the input type and return a uniform ``dict[name, ndarray]``."""
    if isinstance(data, dict):
        return {str(name): np.ascontiguousarray(array) for name, array in data.items()}
    if isinstance(data, np.ndarray):
        if data.dtype.names is None:
            raise TypeError(
                "store() received a plain ndarray; pass {name: array} as a dict "
                "(or a structured ndarray with named fields)."
            )
        return {name: np.ascontiguousarray(data[name]) for name in data.dtype.names}
    if _is_pandas_dataframe(data):
        return _dataframe_to_columns(data)
    raise TypeError(
        f"store() does not know how to handle {type(data).__name__}. "
        f"Expected dict[str, ndarray], structured ndarray, or pandas DataFrame."
    )


def _is_pandas_dataframe(data: Any) -> bool:
    """Duck-typed pandas check so we don't import pandas at module load."""
    return (
        hasattr(data, "columns")
        and hasattr(data, "to_numpy")
        and type(data).__name__ == "DataFrame"
    )


def _dataframe_to_columns(frame: Any) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    """Convert a pandas DataFrame to a column-name -> ndarray dict.

    Object-dtype columns are rejected up front with a clearer message than
    the writer's generic "unsupported dtype" error.
    """
    columns: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
    for column_name in frame.columns:
        series = frame[column_name]
        array = series.to_numpy()
        if array.dtype.kind == "O":
            raise TypeError(
                f"Column {column_name!r} (pandas dtype {series.dtype}) converts to "
                f"an object array and cannot be stored. Cast it to a fixed-size NumPy "
                f"dtype (e.g. float64, int64, or a fixed-width string like 'S16') first."
            )
        columns[str(column_name)] = array
    return columns


# ---- Compaction --------------------------------------------------------


def compact(
    path: str | os.PathLike[str],
    *,
    out: str | os.PathLike[str] | None = None,
    show_progress: bool = True,
) -> Path:
    """Collapse a multi-record file into a single-record file.

    A streamed writer produces files with one record per :meth:`write`
    call. Reads of slice / sorted-fancy patterns scale near-flat with
    record count, but unsorted-fancy reads degrade as records
    accumulate. Compaction concatenates every record's column bytes
    into one contiguous block per column, after which all reads take
    the single-record fast path.

    The byte splice is done via :func:`os.sendfile` where available, so
    memory footprint is bounded by the kernel's I/O buffer (tens of KB)
    regardless of file size. Files much larger than RAM compact fine.

    Parameters
    ----------
    path : str or os.PathLike
        Source file. Must be a valid colstore file.
    out : str or os.PathLike, optional
        Destination. If ``None`` (the default), compaction is done
        in-place via a sibling temp file and an atomic
        :func:`os.replace`; the original is overwritten on success and
        untouched on failure. If given and different from ``path``, the
        compacted result is written there and the original is left as-is.
        If equal to ``path``, behaves as in-place.
    show_progress : bool, default True
        Whether to display a tqdm progress bar during the copy.

    Returns
    -------
    pathlib.Path
        The path the compacted file was written to (``path`` for
        in-place; ``out`` for out-of-place).

    Notes
    -----
    Already-compact files (``n_records <= 1``) are a no-op when ``out``
    is ``None`` or points to the same file. When ``out`` points
    elsewhere, the source is copied byte-for-byte (since the source is
    already in the optimal layout).

    Takes an advisory ``fcntl.flock`` on the source for the duration; a
    concurrent :func:`colstore.update` writer is blocked. Readers are
    unaffected (they don't take the lock, and on POSIX they continue
    reading the unlinked inode after the rename).
    """
    return compact_file(path, out, show_progress=show_progress)


# ---- Introspection: info / schema --------------------------------------


@dataclass(frozen=True)
class ColStoreInfo:
    """Summary of a colstore file's contents and on-disk shape.

    Returned by :func:`info`. Fields are populated from the file header
    without scanning any record bodies, so the call is fast even on
    large files.

    Attributes
    ----------
    path : pathlib.Path
        Filesystem path the info was read from.
    format_version : int
        On-disk format version. Currently always ``1``.
    n_rows : int
        Total committed row count across all records.
    n_records : int
        Number of records in the file. A one-shot write or a fully
        compacted file has ``n_records == 1``.
    columns : list[dict]
        Schema: one ``{"name": ..., "dtype": ..., "encoding": ...,
        "nullable": ...}`` per column, in declaration order.
    file_size : int
        Size of the file in bytes (``os.path.getsize``).
    """

    path: Path
    format_version: int
    n_rows: int
    n_records: int
    columns: list[dict[str, Any]] = field(repr=False)
    file_size: int

    @property
    def needs_compaction(self) -> bool:
        """``True`` if collapsing the file to one record would help.

        A multi-record file pays a per-pattern dispatch cost on reads
        (cheap for slice and sorted-fancy, expensive for unsorted-fancy
        as records accumulate). After :func:`compact`, all reads take
        the single-record fast path.
        """
        return self.n_records > 1

    def __repr__(self) -> str:
        col_summary = ", ".join(f"{c['name']}:{c['dtype']}" for c in self.columns)
        return (
            f"ColStoreInfo(path={self.path.name!r}, n_rows={self.n_rows:,}, "
            f"n_records={self.n_records}, columns=[{col_summary}], "
            f"file_size={self.file_size:,}B"
            f"{', needs_compaction=True' if self.needs_compaction else ''})"
        )


def info(path: str | os.PathLike[str]) -> ColStoreInfo:
    """Return a summary of the file at ``path`` without reading any record bodies.

    The summary is built from the validated file header (magic, counters
    block CRC, format version, manifest CRC). If any of those checks
    fail, the underlying :class:`FormatError` propagates -- ``info`` is
    therefore also useful as a cheap "is this a valid colstore file?"
    probe.

    See :class:`ColStoreInfo` for the returned fields.
    """
    manifest, _ = fmt.read_header(path)
    file_size = os.path.getsize(path)
    return ColStoreInfo(
        path=Path(path),
        format_version=int(manifest["format_version"]),
        n_rows=int(manifest["committed_rows"]),
        n_records=int(manifest["n_records"]),
        columns=list(manifest["columns"]),
        file_size=file_size,
    )


def schema(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Return the column schema of ``path`` without reading any record bodies.

    Each entry is a dict with at least ``"name"`` and ``"dtype"`` keys
    (and may carry ``"encoding"`` / ``"nullable"`` from the on-disk
    manifest). Use :func:`info` for the broader summary including row
    count and record count.
    """
    manifest, _ = fmt.read_header(path)
    return list(manifest["columns"])
