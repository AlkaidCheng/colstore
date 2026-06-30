"""Module-level convenience functions: open, create, recreate, update, store,
compact, info, schema.

These thin wrappers around :class:`ColStoreReader` and :class:`ColStoreWriter`
give the package a uproot-style entry-point surface where each function does
one obvious thing.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, overload

from . import format as fmt
from ._coerce import coerce_to_columns
from ._paths import expand_glob, has_glob_magic
from ._types import Source
from .compaction import compact_file
from .dataset import ColStoreDataset, OnMismatch
from .reader import ColStoreReader
from .writer import ColStoreWriter


@overload
def open(path: str | os.PathLike[str], **kwargs: Any) -> ColStoreReader: ...
@overload
def open(path: Sequence[str | os.PathLike[str]], **kwargs: Any) -> ColStoreDataset: ...
def open(
    path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    *,
    on_mismatch: OnMismatch = "strict",
    **kwargs: Any,
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

    For a multi-file dataset, ``on_mismatch`` chooses how files whose schemas differ
    are reconciled. ``"strict"`` (the default) requires every file to share one schema --
    the same column names and dtypes, though the column *order* may differ since reads are
    by name -- and raises :class:`ValueError` otherwise. ``"drop"`` instead opens the files
    anyway, exposing only the columns common to every file with one consistent dtype and
    warning about the rest -- useful for opening a set of files where a column's dtype
    drifted between writes. It is moot for a single file.
    """
    if isinstance(path, (str, os.PathLike)) and os.path.isdir(path):
        # A directory is a managed shard dataset; the constructor expands it to
        # its ``.cstore`` shards (an empty directory is an empty dataset).
        return ColStoreDataset(path, on_mismatch=on_mismatch, **kwargs)
    if isinstance(path, str) and has_glob_magic(path):
        return ColStoreDataset(path, on_mismatch=on_mismatch, **kwargs)
    if isinstance(path, (str, os.PathLike)):
        # A single literal file is trivially self-consistent, so ``on_mismatch`` is moot.
        return ColStoreReader(path, **kwargs)
    return ColStoreDataset(path, on_mismatch=on_mismatch, **kwargs)


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


# ---- Foreign file formats: convert / saveas ----------------------------


def _is_cstore_path(path: str | os.PathLike[str]) -> bool:
    """Whether ``path`` names colstore's own format, by its extension."""
    return os.fspath(path).lower().endswith(fmt.FILE_EXTENSION)


def _import_to_cstore(
    source: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    *,
    format: str | None = None,
    dtypes: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ColStoreReader:
    """Materialize a foreign file into a new ``.cstore`` and open it (the import path)."""
    from . import interop

    if dtypes is not None:
        kwargs["dtypes"] = dtypes
    return interop.file_format_for_path(source, format).from_file(source, dest, **kwargs)


def _prepare_output(dest: str | os.PathLike[str], overwrite: bool) -> Path:
    """Resolve ``dest`` to a ``Path`` and enforce the overwrite policy."""
    dest_path = Path(os.fspath(dest))
    if dest_path.exists():
        if not overwrite:
            raise FileExistsError(f"{dest_path} already exists; pass overwrite=True to replace it.")
        dest_path.unlink()
    return dest_path


def _convert_one(
    source: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    *,
    format: str | None = None,
    dtypes: dict[str, Any] | None = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> ColStoreReader | Path:
    """Convert one file; one endpoint must be a ``.cstore`` (see :func:`convert`)."""
    source_is_cstore = _is_cstore_path(source)
    dest_is_cstore = _is_cstore_path(dest)
    if not source_is_cstore and not dest_is_cstore:
        raise ValueError(
            f"convert needs one endpoint to be a .cstore file; got "
            f"{os.fspath(source)!r} -> {os.fspath(dest)!r}."
        )
    dest_path = _prepare_output(dest, overwrite)
    if not source_is_cstore:
        return _import_to_cstore(source, dest, format=format, dtypes=dtypes, **kwargs)
    if dtypes is not None:
        raise ValueError("dtypes= applies only when importing a foreign file into a .cstore.")
    reader = open(source)
    try:
        reader.saveas(dest, format=format, **kwargs)
    finally:
        reader.close()
    return open(dest) if dest_is_cstore else dest_path


def _resolve_convert_inputs(
    source: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
) -> tuple[list[str], bool]:
    """Resolve a source to concrete input paths, flagging whether it was a single path.

    A single literal path stays a one-element list (and is flagged scalar, so the result
    is returned bare); a glob string or a list/tuple is expanded (globs inside a list too).
    """
    if isinstance(source, (str, os.PathLike)):
        text = os.fspath(source)
        if isinstance(source, str) and has_glob_magic(text):
            return expand_glob(text), False
        return [text], True
    if isinstance(source, (list, tuple)):
        resolved: list[str] = []
        for item in source:
            text = os.fspath(item)
            if isinstance(item, str) and has_glob_magic(text):
                resolved.extend(expand_glob(text))
            else:
                resolved.append(text)
        return resolved, False
    raise TypeError(
        f"convert source must be a path, a glob, or a list of them; got {type(source).__name__}."
    )


def _target_extension(
    inputs: list[str], dest: str | None, is_template: bool, format: str | None
) -> str:
    """The output file extension for a one-to-one conversion."""
    if is_template and dest is not None:
        return Path(dest).suffix
    if format is not None:
        from . import interop
        from .interop import FileFormat

        resolved = interop.get(format)
        if not isinstance(resolved, FileFormat) or not resolved.extensions:
            raise ValueError(f"format {format!r} is not a file format with an extension.")
        return sorted(resolved.extensions)[0]
    if _is_cstore_path(inputs[0]):
        raise ValueError(
            "converting .cstore files needs a target format; pass format= or a dest "
            "with the target extension."
        )
    return fmt.FILE_EXTENSION


def _resolve_output_path(
    source: str,
    index: int,
    dest: str | None,
    is_template: bool,
    rename: Mapping[str, str] | Callable[[str], str] | None,
    output_dir: str | os.PathLike[str] | None,
    target_ext: str,
) -> Path:
    """The output path for one input under the naming rules (template / rename / auto)."""
    source_path = Path(source)
    if is_template and dest is not None:
        out = Path(
            dest.format(
                index=index,
                stem=source_path.stem,
                name=source_path.name,
                parent=str(source_path.parent),
            )
        )
    elif rename is not None:
        stem = (
            rename(source_path.stem)
            if callable(rename)
            else rename.get(source_path.stem, source_path.stem)
        )
        filename = stem if stem.lower().endswith(target_ext.lower()) else stem + target_ext
        out = source_path.with_name(filename)  # keep the source's directory
    else:
        out = source_path.with_suffix(target_ext)
    if output_dir is not None:
        out = Path(os.fspath(output_dir)) / out.name
    return out


def _convert_merge(
    inputs: list[str],
    dest: str,
    *,
    format: str | None,
    dtypes: dict[str, Any] | None,
    overwrite: bool,
    on_mismatch: OnMismatch,
    **kwargs: Any,
) -> ColStoreReader | Path:
    """Merge every input into the single file ``dest`` (a literal output path)."""
    if len(inputs) == 1:
        return _convert_one(
            inputs[0], dest, format=format, dtypes=dtypes, overwrite=overwrite, **kwargs
        )
    dest_is_cstore = _is_cstore_path(dest)
    if not dest_is_cstore and not any(_is_cstore_path(src) for src in inputs):
        raise ValueError(
            f"convert needs one endpoint to be a .cstore file; merging foreign files into "
            f"{dest!r} has none."
        )
    dest_path = _prepare_output(dest, overwrite)
    scratch = tempfile.mkdtemp(prefix="colstore_convert_")
    readers: list[ColStoreReader] = []
    try:
        for index, source in enumerate(inputs):
            if _is_cstore_path(source):
                readers.append(ColStoreReader(source))
            else:
                part = os.path.join(scratch, f"part_{index:05d}.cstore")
                _import_to_cstore(source, part, format=format, dtypes=dtypes, **kwargs)
                readers.append(ColStoreReader(part))
        ColStoreDataset(readers, on_mismatch=on_mismatch).saveas(
            dest, format=None if dest_is_cstore else format
        )
    finally:
        for reader in readers:
            reader.close()
        shutil.rmtree(scratch, ignore_errors=True)
    return ColStoreReader(dest) if dest_is_cstore else dest_path


def convert(
    source: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    dest: str | os.PathLike[str] | None = None,
    *,
    format: str | None = None,
    dtypes: dict[str, Any] | None = None,
    rename: Mapping[str, str] | Callable[[str], str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
    on_mismatch: OnMismatch = "strict",
    **kwargs: Any,
) -> ColStoreReader | Path | list[ColStoreReader | Path]:
    """Convert files between colstore's format and another, one endpoint being ``.cstore``.

    ``source`` is a single path, a glob, or a list of them; direction is inferred from the
    extensions (one endpoint must be ``.cstore``, or :class:`ValueError` is raised):

    - **foreign -> .cstore** (import) returns the opened
      :class:`~colstore.reader.ColStoreReader`; ``.cstore -> foreign`` (export) returns the
      output :class:`~pathlib.Path`; ``.cstore -> .cstore`` copies (or, across many inputs,
      merges) and returns a reader.

    ``dest`` selects how the outputs are written:

    - **omitted** -- each input is converted one-to-one, auto-named by swapping its
      extension (``convert("data.h5")`` -> ``data.cstore``; ``convert("*.h5")`` -> one
      ``.cstore`` per file).
    - **a literal path** -- every input is **merged** into that one file
      (``convert("*.h5", "all.cstore")``).
    - **a template** with ``{index}`` / ``{stem}`` / ``{name}`` / ``{parent}`` -- one-to-one
      with custom names (``convert("*.h5", "run_{index}.cstore")``).

    ``rename`` overrides the output name per input (one-to-one): a callable
    ``stem -> new_stem`` or a mapping ``{stem: new_stem}`` (a stem absent from the mapping
    keeps its name). ``output_dir`` writes the outputs into that directory. ``overwrite``
    replaces existing outputs (off by default, so an existing output raises
    :class:`FileExistsError`). ``on_mismatch`` reconciles schemas when merging (see
    :func:`open`). ``format`` overrides the foreign endpoint's format, and ``dtypes``
    (import only) coerces columns as they are read (see :func:`open` for the rules).

    Returns a single result for a single ``source`` path or a merge, and a list (one per
    input) for a glob or list converted one-to-one.
    """
    inputs, scalar = _resolve_convert_inputs(source)
    if not inputs:
        raise FileNotFoundError(f"convert: no files matched {source!r}.")
    dest_text = None if dest is None else os.fspath(dest)
    is_template = dest_text is not None and "{" in dest_text
    if dest_text is not None and not is_template:
        return _convert_merge(
            inputs,
            dest_text,
            format=format,
            dtypes=dtypes,
            overwrite=overwrite,
            on_mismatch=on_mismatch,
            **kwargs,
        )
    target_ext = _target_extension(inputs, dest_text, is_template, format)
    results: list[ColStoreReader | Path] = []
    for index, source_path in enumerate(inputs):
        out = _resolve_output_path(
            source_path, index, dest_text, is_template, rename, output_dir, target_ext
        )
        results.append(
            _convert_one(
                source_path, out, format=format, dtypes=dtypes, overwrite=overwrite, **kwargs
            )
        )
    return results[0] if scalar else results


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
    """Import a NumPy ``.npz`` file into a ``.cstore`` (``convert(..., format="npz")``)."""
    return _import_to_cstore(source, dest, format="npz", **kwargs)


def from_parquet(
    source: str | os.PathLike[str], dest: str | os.PathLike[str], **kwargs: Any
) -> ColStoreReader:
    """Import a Parquet file into a ``.cstore`` (``convert(..., format="parquet")``)."""
    return _import_to_cstore(source, dest, format="parquet", **kwargs)


def from_feather(
    source: str | os.PathLike[str], dest: str | os.PathLike[str], **kwargs: Any
) -> ColStoreReader:
    """Import a Feather file into a ``.cstore`` (``convert(..., format="feather")``)."""
    return _import_to_cstore(source, dest, format="feather", **kwargs)


def from_json(
    source: str | os.PathLike[str], dest: str | os.PathLike[str], **kwargs: Any
) -> ColStoreReader:
    """Import a JSON file into a ``.cstore`` (``convert(..., format="json")``)."""
    return _import_to_cstore(source, dest, format="json", **kwargs)


def from_hdf(
    source: str | os.PathLike[str], dest: str | os.PathLike[str], **kwargs: Any
) -> ColStoreReader:
    """Import an HDF5 file into a ``.cstore`` (``convert(..., format="hdf5")``)."""
    return _import_to_cstore(source, dest, format="hdf5", **kwargs)


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
