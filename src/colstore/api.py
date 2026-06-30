"""Module-level convenience functions: open, create, recreate, update, store,
compact, info, schema.

These thin wrappers around :class:`ColStoreReader` and :class:`ColStoreWriter`
give the package a uproot-style entry-point surface where each function does
one obvious thing.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, overload

from . import format as fmt
from ._coerce import coerce_to_columns
from ._paths import has_glob_magic
from ._types import Source
from .compaction import compact_file
from .dataset import ColStoreDataset
from .reader import ColStoreReader
from .writer import ColStoreWriter


@overload
def open(path: str | os.PathLike[str], **kwargs: Any) -> ColStoreReader: ...
@overload
def open(path: Sequence[str | os.PathLike[str]], **kwargs: Any) -> ColStoreDataset: ...
def open(
    path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]], **kwargs: Any
) -> ColStoreReader | ColStoreDataset:
    """Open an existing ``.cstore`` file, or several as one logical dataset.

    A single literal path (``str`` or ``os.PathLike``) returns a
    :class:`~colstore.reader.ColStoreReader`, equivalent to
    ``ColStoreReader(path, **kwargs)``. A list or tuple of paths returns a
    :class:`~colstore.dataset.ColStoreDataset` spanning the files in order (all
    must share one schema), including for a one-element list (a single-file
    dataset) and an empty list (an empty dataset, which can be grown later). The
    dataset owns the files it opened and closes them on :meth:`close`. Each file
    must exist and be valid; otherwise :class:`FileNotFoundError` or
    :class:`~colstore.FormatError` propagates.

    A ``str`` containing a shell glob (``*``, ``?``, ``[``; ``**`` is recursive)
    is expanded to its matches and returns a :class:`ColStoreDataset` -- e.g.
    ``open("run_*.cstore")`` opens every matching file as one logical table.
    Matches are ordered numerically (``run_2`` before ``run_10``), since file
    order is the dataset's row order, and a pattern matching no files raises
    :class:`FileNotFoundError`. Globbing applies to path arguments only -- column
    selection is always explicit -- and a list element may itself be a glob.

    A **directory** path returns a :class:`ColStoreDataset` over its ``.cstore``
    shards in numeric order -- the managed dataset that :func:`append` /
    :func:`appender` write to (an empty directory is an empty dataset).
    """
    if isinstance(path, (str, os.PathLike)) and os.path.isdir(path):
        # A directory is a managed shard dataset; the constructor expands it to
        # its ``.cstore`` shards (an empty directory is an empty dataset).
        return ColStoreDataset(path, **kwargs)
    if isinstance(path, str) and has_glob_magic(path):
        return ColStoreDataset(path, **kwargs)
    if isinstance(path, (str, os.PathLike)):
        return ColStoreReader(path, **kwargs)
    return ColStoreDataset(path, **kwargs)


def create(path: str | os.PathLike[str], *, statistics: bool = False) -> ColStoreWriter:
    """Open a new file for streaming writes; fail if it already exists.

    Use this when you want to be sure you are not overwriting anything.
    ``statistics=True`` records per-column statistics so later filters can skip
    data that cannot match; off by default, most useful for selective queries on
    sorted or clustered data.
    """
    return ColStoreWriter(path, mode="create", statistics=statistics)


def recreate(path: str | os.PathLike[str], *, statistics: bool = False) -> ColStoreWriter:
    """Open a file for streaming writes, truncating any existing content.

    Use this when you intentionally want to overwrite. To fail on
    overwrite instead, use :func:`create`. ``statistics=True`` records per-column
    statistics so later filters can skip data that cannot match (off by default).
    """
    return ColStoreWriter(path, mode="recreate", statistics=statistics)


def update(path: str | os.PathLike[str], *, statistics: bool = False) -> ColStoreWriter:
    """Open an existing file for append.

    The schema is loaded from the existing manifest; every :meth:`write`
    must match it exactly. Orphan bytes from a crashed prior writer (if
    any) are truncated on open. Raises :class:`FileNotFoundError` if the
    file does not exist. ``statistics=True`` keeps the per-column statistics
    current as records are appended; pass it on each update to keep them (off by
    default).
    """
    return ColStoreWriter(path, mode="update", statistics=statistics)


def store(
    data: Any,
    path: str | os.PathLike[str],
    *,
    mode: str = "create",
    batch_size: int | str | None = "auto",
    show_progress: bool = True,
    statistics: bool = False,
    **open_kwargs: Any,
) -> ColStoreReader:
    """One-shot: write a single-record file and return an opened reader.

    Accepted ``data`` types: ``dict[str, numpy.ndarray]`` (column-major
    mapping), structured ``numpy.ndarray`` (one column per field), or a
    pandas ``DataFrame`` (one column per series). ``mode`` is ``"create"``
    (default; fail if the file exists) or ``"recreate"`` (truncate if it
    exists). For multi-record streaming writes, use :func:`create` /
    :func:`recreate` / :func:`update` directly.

    Parameters
    ----------
    batch_size : int, str, or None, default ``"auto"``
        Write chunking for the progress bar; no effect on the bytes
        written.

        * ``"auto"`` -- adaptive: probe with a 1 MiB initial batch, then
          grow from EMA-smoothed measured bandwidth (2x growth cap per
          batch); fast NVMe ramps to GiB-class batches, slow disks settle
          at tens of MiB. Files under 16 MiB single-pass.
        * ``None`` -- single pass: one ``tofile`` call per column.
        * ``int N`` -- rows x columns per logical batch
          (``batch_size=100_000`` with 5 columns: 20 000-row writes).
        * ``str`` like ``"100 MB"``, ``"1.5 GiB"`` -- bytes per batch.
          Units follow IEC 80000-13: decimal ``kB``/``MB``/``GB`` are powers
          of 1000 and binary ``KiB``/``MiB``/``GiB`` are powers of 1024, so
          ``"1 MB"`` is 1,000,000 bytes and ``"1 MiB"`` is 1,048,576.

    show_progress : bool, default ``True``
        Whether to display a tqdm progress bar. The bar's postfix shows the
        batch count and ``rows=...Mrows/s``; the byte rate is rendered by the
        byte-counted bar itself.
    statistics : bool, default ``False``
        Record per-column statistics so later filters can skip data that cannot
        match. Most useful for selective queries on sorted or clustered data.

    Returns the opened :class:`ColStoreReader` for immediate use.
    """
    if mode not in ("create", "recreate"):
        raise ValueError(f"Invalid mode {mode!r} for store(); expected 'create' or 'recreate'.")

    columns = coerce_to_columns(data)

    # write_dataset is the single-record writer that includes a progress bar;
    # ColStoreWriter is for multi-record streams. For one-shot writes, write_dataset
    # is slightly cheaper (no counter rewrite at the end -- the counters are
    # right the first time) and surfaces a progress bar.
    if mode == "create" and os.path.exists(path):
        raise FileExistsError(f"{path} already exists; use mode='recreate' to overwrite.")
    fmt.write_dataset(
        columns, path, batch_size=batch_size, show_progress=show_progress, statistics=statistics
    )
    return ColStoreReader(path, **open_kwargs)


# ---- Foreign file formats: ingest / saveas -----------------------------


def ingest(
    source: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    *,
    format: str | None = None,
    dtypes: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ColStoreReader:
    """Import a foreign file into a new ``.cstore`` at ``dest`` and open it.

    The format is chosen from ``source``'s extension (e.g. ``.npz``); pass
    ``format=`` to override it (``format="npz"``). ``dest`` is required -- colstore
    mmaps only its own format, so the foreign file is materialized into a ``.cstore``
    first -- and the opened :class:`~colstore.reader.ColStoreReader` is returned.
    ``dest`` must not already exist (pass ``mode="recreate"`` to overwrite it). List
    the available file formats with :func:`colstore.interop.file_formats`; importing
    an in-memory object instead uses :func:`colstore.interop.from_object`.

    ``dtypes`` maps a column name to a target dtype (``{"flag": "bool"}``) and coerces
    that column on import -- handy to give a column the same dtype across files whose
    schemas differ (e.g. a flag that is ``bool`` in some files and all-null in others).
    A missing value (``NaN``) becomes the target's empty value when cast to a bool /
    integer / string dtype (``False`` / ``0`` / ``""``), keeps ``NaN`` for a float dtype,
    and becomes ``NaT`` for a datetime / timedelta dtype. Real values follow NumPy
    ``astype`` rules, so a too-narrow target may truncate or overflow them without error.
    Applies to the column-based formats (Parquet / Feather / JSON / HDF5 / NPZ).
    """
    from . import interop

    if dtypes is not None:
        kwargs["dtypes"] = dtypes
    return interop.file_format_for_path(source, format).from_file(source, dest, **kwargs)


def saveas(
    source: Any,
    dest: str | os.PathLike[str],
    *,
    format: str | None = None,
    **kwargs: Any,
) -> Any:
    """Write a reader, dataset, or view to a file -- the function form of ``source.saveas``.

    ``colstore.saveas(ds, "out.npz")`` is ``ds.saveas("out.npz")``: the format is
    chosen from ``dest``'s extension, overridable with ``format=``. The whole store
    is written, or just a selection (``colstore.saveas(ds[rows, cols], dest)``).
    """
    return source.saveas(dest, format=format, **kwargs)


def from_npz(
    source: str | os.PathLike[str], dest: str | os.PathLike[str], **kwargs: Any
) -> ColStoreReader:
    """Import a NumPy ``.npz`` file into a ``.cstore`` (``ingest(..., format="npz")``)."""
    return ingest(source, dest, format="npz", **kwargs)


def from_parquet(
    source: str | os.PathLike[str], dest: str | os.PathLike[str], **kwargs: Any
) -> ColStoreReader:
    """Import a Parquet file into a ``.cstore`` (``ingest(..., format="parquet")``)."""
    return ingest(source, dest, format="parquet", **kwargs)


def from_feather(
    source: str | os.PathLike[str], dest: str | os.PathLike[str], **kwargs: Any
) -> ColStoreReader:
    """Import a Feather file into a ``.cstore`` (``ingest(..., format="feather")``)."""
    return ingest(source, dest, format="feather", **kwargs)


def from_json(
    source: str | os.PathLike[str], dest: str | os.PathLike[str], **kwargs: Any
) -> ColStoreReader:
    """Import a JSON file into a ``.cstore`` (``ingest(..., format="json")``)."""
    return ingest(source, dest, format="json", **kwargs)


def from_hdf(
    source: str | os.PathLike[str], dest: str | os.PathLike[str], **kwargs: Any
) -> ColStoreReader:
    """Import an HDF5 file into a ``.cstore`` (``ingest(..., format="hdf5")``)."""
    return ingest(source, dest, format="hdf5", **kwargs)


# ---- Compaction --------------------------------------------------------


def compact(
    path: str | os.PathLike[str],
    *,
    out: str | os.PathLike[str] | None = None,
    show_progress: bool = True,
) -> Path:
    """Collapse a multi-record file into a single-record file.

    Streamed writers produce one record per :meth:`write` call, and
    unsorted-fancy reads degrade as records accumulate. Compaction
    concatenates every record's column bytes into one contiguous block
    per column, after which all reads take the single-record fast path.
    The byte splice runs in bounded memory regardless of file size
    (``os.sendfile`` on Linux, ``shutil.copyfileobj`` elsewhere; see
    :mod:`colstore.compaction`), so files much larger than RAM compact
    fine.

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
    is ``None`` or the same file, and a byte-for-byte copy when ``out``
    points elsewhere. An advisory lock is held on the source for the
    duration (see :mod:`colstore._lock`): a concurrent
    :func:`colstore.update` writer is blocked, while readers are
    unaffected (they don't take the lock, and on POSIX they continue
    reading the unlinked inode after the rename).
    """
    return compact_file(path, out, show_progress=show_progress)


# ---- Concatenation: many files as one (lazy or written) ----------------


@overload
def concat(
    sources: Sequence[Source],
    *,
    out: None = None,
    memory_budget: int | None = None,
    **reader_kwargs: Any,
) -> ColStoreDataset: ...
@overload
def concat(
    sources: Sequence[Source],
    *,
    out: str | os.PathLike[str],
    memory_budget: int | None = None,
    **reader_kwargs: Any,
) -> ColStoreReader: ...
def concat(
    sources: Sequence[Source],
    *,
    out: str | os.PathLike[str] | None = None,
    memory_budget: int | None = None,
    **reader_kwargs: Any,
) -> ColStoreReader | ColStoreDataset:
    """Combine several same-schema sources, lazily or into one written file.

    ``sources`` is a list or tuple mixing file paths and already-open readers or
    datasets; all must share one schema. Paths are opened (and owned by the
    result); readers and datasets are borrowed and left open. A path string may
    be a glob (e.g. ``"run_*.cstore"``), expanded to its matches in numeric order.

    Parameters
    ----------
    sources : sequence of path or reader or dataset
        The inputs, combined in the given order.
    out : str or os.PathLike, optional
        Destination. If ``None`` (the default), returns a lazy
        :class:`~colstore.dataset.ColStoreDataset` spanning the sources without
        copying -- equivalent to :func:`open` of the same list. If given, streams
        the combined data to a new ``.cstore`` at ``out`` in bounded memory and
        returns a :class:`~colstore.reader.ColStoreReader` opened on it; ``out``
        must be a new path, not one of the sources.
    memory_budget : int, optional
        Peak bytes for the streaming write (``out`` given only); ``None`` uses
        the configured default.
    **reader_kwargs
        Forwarded to :class:`~colstore.reader.ColStoreReader` when opening any
        source paths.

    Returns
    -------
    ColStoreDataset or ColStoreReader
        The lazy dataset when ``out`` is ``None``; otherwise a reader on the
        written file.
    """
    dataset = ColStoreDataset(sources, **reader_kwargs)
    if out is None:
        return dataset
    try:
        if not dataset.columns:
            raise ValueError(
                "concat() needs at least one source with columns to write an "
                "output file; got nothing to write."
            )
        out_resolved = Path(out).resolve()
        if any(Path(source).resolve() == out_resolved for source in dataset.paths):
            raise ValueError(
                f"concat() out={os.fspath(out)!r} is also one of the sources; "
                f"write to a new path."
            )
        return dataset.edit().write(out, memory_budget=memory_budget)
    finally:
        dataset.close()


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
