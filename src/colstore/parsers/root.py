"""Two-way bridge between ROOT files and colstore, via RDataFrame.

ROOT (PyROOT) is imported lazily inside the functions that need it, so
importing this module -- or :mod:`colstore.parsers` -- never pulls ROOT into a
process that will not use it. Annotations that name ROOT types are evaluated
only by a type checker (``from __future__ import annotations`` keeps them as
strings at runtime); ROOT ships no type stubs, and the project's mypy config
already ignores missing imports.

Only fixed-size scalar branches can be stored: a ``.cstore`` column is a
contiguous block of fixed-width values, so jagged branches (``RVec``,
``vector``), array branches, and ``char*`` are skipped. Everything stored
round-trips through ``numpy``.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import tempfile
import warnings
from collections.abc import Iterator
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from .. import api
from ..progress import progress_bar
from ..reader import ColStoreReader
from .base import ColumnBatch, Parser, StrPath, resolve_batch_rows, write_column_batches

if TYPE_CHECKING:
    import ROOT

#: A ROOT source accepted by :func:`from_root`.
RootSource: TypeAlias = "ROOT.RDataFrame | str | os.PathLike[str] | dict[str, str | list[str]]"

# Fixed-size scalar ROOT/C++ types that AsNumpy materializes as fixed-width
# numeric arrays. Container, array, and pointer types are excluded below
# regardless of this set, because colstore stores only fixed-size columns.
_STORABLE_ROOT_TYPES: frozenset[str] = frozenset(
    {
        "bool",
        "Bool_t",
        "Byte_t",
        "char",
        "Char_t",
        "double",
        "Double32_t",
        "Double_t",
        "float",
        "Float16_t",
        "Float_t",
        "int",
        "Int_t",
        "long",
        "long long",
        "Long_t",
        "Long64_t",
        "short",
        "Short_t",
        "Size_t",
        "UChar_t",
        "UInt_t",
        "ULong64_t",
        "ULong_t",
        "UShort_t",
        "unsigned",
        "unsigned char",
        "unsigned int",
        "unsigned long",
        "unsigned long long",
        "unsigned short",
    }
)

# Substrings that mark a non-fixed-size branch (jagged/array/pointer), which
# AsNumpy returns as object arrays that colstore cannot store.
_NON_SCALAR_MARKERS: tuple[str, ...] = ("<", ">", "[", "]", "*", "vector", "RVec")

# Any run of characters that is illegal in a ROOT branch/leaf name. ROOT's leaf
# list grammar gives "/", "[", "]", ":", and whitespace special meaning, and a
# branch name with them makes Snapshot abort (and can crash the process), so a
# colstore column name is reduced to word characters before it is written.
_INVALID_BRANCH_CHARS = re.compile(r"[^0-9A-Za-z_]+")

_DEFAULT_BATCH_SIZE = "512 MiB"
_DEFAULT_TREE_NAME = "events"

# colstore stores already-materialized fixed-width columns, so the common case
# is to write them straight through without paying compression cost; ROOT's own
# Snapshot default is level 5. Override to uncompressed here.
_DEFAULT_COMPRESSION_LEVEL = 0

# Multithreading is on by default to speed up the Snapshot event loop.
_DEFAULT_MULTITHREADING = True

# Output-format and compression-algorithm string aliases mapped to the ROOT
# enum members they name. Non-string values are passed through unchanged, so a
# caller may also hand in a ROOT enum value directly.
_OUTPUT_FORMAT_ALIASES = {"default": "kDefault", "ttree": "kTTree", "rntuple": "kRNTuple"}
_COMPRESSION_ALGORITHM_ALIASES = {
    "zlib": "kZLIB",
    "lzma": "kLZMA",
    "lz4": "kLZ4",
    "zstd": "kZSTD",
}

# Prefix for the per-chunk scratch directory used by the bounded-memory export.
_TMP_DIR_PREFIX = ".colstore_to_root_"


def _import_root() -> ModuleType:
    """Import and return the PyROOT module, with a clear error if it is missing."""
    try:
        import ROOT
    except ImportError as exc:  # pragma: no cover - exercised only without ROOT
        raise ImportError(
            "The ROOT parser requires PyROOT, which is not installed. Install ROOT "
            "(e.g. 'conda install -c conda-forge root') to use colstore.parsers.root."
        ) from exc
    return cast(ModuleType, ROOT)


def get_rdf_column_type(rdf: ROOT.RDF.RNode, column_name: str) -> str:
    """Return the declared column type, or ``""`` if it cannot be determined.

    Handles both a local ``RDataFrame`` (which has ``GetColumnType``) and a
    distributed one (which exposes it on ``_headnode._localdf``).
    """
    get_column_type = getattr(rdf, "GetColumnType", None)
    if get_column_type is None:
        get_column_type = rdf._headnode._localdf.GetColumnType
    try:
        return str(get_column_type(column_name))
    except Exception:
        # PyROOT surfaces assorted C++-bound exceptions for unknown or
        # untyped columns; treat any of them as "type unknown" so the caller
        # skips the column rather than crashing the whole conversion.
        return ""


def _is_storable_root_type(column_type: str) -> bool:
    """Return whether a ROOT column type maps to a fixed-size colstore column."""
    stripped = column_type.strip()
    if not stripped or any(marker in stripped for marker in _NON_SCALAR_MARKERS):
        return False
    return stripped in _STORABLE_ROOT_TYPES


def _select_storable_columns(
    rdf: ROOT.RDF.RNode,
    requested: list[str] | None,
) -> list[str]:
    """Choose the columns to store, skipping (or rejecting) non-storable ones.

    When ``requested`` is ``None`` every column is considered and non-storable
    ones are skipped with a warning. When columns are named explicitly, a
    non-storable one is an error rather than a silent skip.
    """
    names = (
        [str(name) for name in rdf.GetColumnNames()]
        if requested is None
        else [str(name) for name in requested]
    )
    if not names:
        raise ValueError("No columns to read from the source.")

    kept: list[str] = []
    skipped: list[str] = []
    for name in names:
        if _is_storable_root_type(get_rdf_column_type(rdf, name)):
            kept.append(name)
        else:
            skipped.append(name)

    if requested is not None and skipped:
        detail = ", ".join(skipped)
        raise ValueError(
            f"Requested column(s) cannot be stored as fixed-size colstore columns: {detail}."
        )
    if not kept:
        raise ValueError("None of the source columns are storable as fixed-size colstore columns.")
    if skipped:
        warnings.warn(
            f"Skipping non-fixed-size column(s): {', '.join(skipped)}.",
            RuntimeWarning,
            stacklevel=3,
        )
    return kept


def _bytes_per_row(rdf: ROOT.RDF.RNode, columns: list[str]) -> int:
    """Estimate bytes per row from a one-row sample of the selected columns."""
    sample = rdf.Range(0, 1).AsNumpy(columns=columns)
    total = 0
    for name, array in sample.items():
        if array.dtype == object:
            raise TypeError(
                f"Column {name!r} sampled to object dtype; colstore stores only fixed-size columns."
            )
        total += int(array.nbytes)
    if total <= 0:
        raise ValueError("Could not estimate bytes per row from a one-row sample.")
    return total


def _resolve_tree_name(root: ModuleType, path: StrPath, treename: str | None) -> str:
    """Return ``treename`` if given, else the sole TTree in the file.

    Raises
    ------
    ValueError
        If the file cannot be opened, holds no tree, or holds more than one
        and no ``treename`` was supplied.
    """
    if treename is not None:
        return treename
    file_path = os.fspath(path)
    root_file = root.TFile.Open(file_path)
    if not root_file or root_file.IsZombie():
        if root_file:
            root_file.Close()
        raise ValueError(f"Could not open {file_path!r} as a ROOT file.")
    try:
        tree_names = {
            str(key.GetName())
            for key in root_file.GetListOfKeys()
            if _inherits_from_tree(root, key)
        }
    finally:
        root_file.Close()

    if not tree_names:
        raise ValueError(f"No TTree found in {file_path!r}; pass treename=... to select one.")
    if len(tree_names) > 1:
        listed = ", ".join(sorted(tree_names))
        raise ValueError(
            f"{file_path!r} contains multiple trees ({listed}); pass treename=... to choose one."
        )
    return next(iter(tree_names))


def _inherits_from_tree(root: ModuleType, key: Any) -> bool:
    """Return whether a TKey points at a TTree (or subclass such as TNtuple)."""
    cls = root.TClass.GetClass(key.GetClassName())
    return bool(cls) and bool(cls.InheritsFrom("TTree"))


# Leading URL scheme ("root://", "https://", ...) whose "://" colon is not a
# file/object separator.
_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")
# Leading Windows drive letter ("C:\" or "C:/") whose colon is part of the path.
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


def _split_path_and_tree(spec: str) -> tuple[str, str | None]:
    """Split ``"file.root:tree"`` into ``("file.root", "tree")``.

    Follows the uproot convention: the object path is whatever follows the last
    colon, except that a colon belonging to a URL scheme (``root://``,
    ``https://``) or a Windows drive letter (``C:\\``) is not a separator. A
    file whose name genuinely contains a colon should be passed as a
    :class:`pathlib.Path` (never split) or via the ``{treename: files}`` form.
    Returns ``(spec, None)`` when no object path is present.
    """
    prefix = ""
    rest = spec
    scheme = _URL_SCHEME.match(rest)
    if scheme:
        prefix, rest = scheme.group(0), rest[scheme.end() :]
    if _WINDOWS_DRIVE.match(rest):
        prefix, rest = prefix + rest[:2], rest[2:]
    if ":" in rest:
        path_part, tree_part = rest.rsplit(":", 1)
        return prefix + path_part, tree_part or None
    return prefix + rest, None


def _as_rdataframe(source: RootSource, treename: str | None) -> ROOT.RDF.RNode:
    """Build (or pass through) an RDataFrame from a path, a mapping, or an RDF."""
    if not isinstance(source, (str, os.PathLike, dict)):
        return source  # already an RDataFrame/RNode; use duck-typed, no ROOT import
    root = _import_root()
    if isinstance(source, dict):
        if len(source) != 1:
            raise ValueError(
                f"A {{treename: files}} mapping must name exactly one tree; got {len(source)}."
            )
        ((tree, files),) = source.items()
        return root.RDataFrame(tree, files)

    # A str may carry an embedded ":tree"; an os.PathLike is strictly a path.
    if isinstance(source, str):
        file_path, embedded_tree = _split_path_and_tree(source)
    else:
        file_path, embedded_tree = os.fspath(source), None

    if embedded_tree is not None:
        if treename is not None and treename != embedded_tree:
            raise ValueError(
                f"Conflicting tree names: treename={treename!r} but the path names "
                f"{embedded_tree!r}; pass only one."
            )
        return root.RDataFrame(embedded_tree, file_path)
    resolved_tree = _resolve_tree_name(root, file_path, treename)
    return root.RDataFrame(resolved_tree, file_path)


def _ingest_batches(
    rdf: ROOT.RDF.RNode,
    columns: list[str],
    rows_per_batch: int | None,
    total_rows: int,
) -> Iterator[ColumnBatch]:
    """Yield AsNumpy batches, one per output record.

    A zero-row source still yields one typed (empty) batch so the schema is
    captured. ``RDataFrame.Range`` is incompatible with implicit MT; disable
    ``ROOT.EnableImplicitMT()`` before a chunked ingest.
    """
    if total_rows == 0 or rows_per_batch is None:
        yield rdf.AsNumpy(columns=columns)
        return
    for start in range(0, total_rows, rows_per_batch):
        end = min(start + rows_per_batch, total_rows)
        yield rdf.Range(start, end).AsNumpy(columns=columns)


def from_root(
    source: RootSource,
    path: StrPath,
    *,
    treename: str | None = None,
    columns: list[str] | None = None,
    batch_size: int | str | None = _DEFAULT_BATCH_SIZE,
    compact: bool = True,
    mode: str = "create",
    show_progress: bool = True,
) -> ColStoreReader:
    """Convert a ROOT source into a ``.cstore`` file and open it for reading.

    Parameters
    ----------
    source : ROOT.RDataFrame, str, os.PathLike, or dict
        An existing ``RDataFrame``; a path to a ``.root`` file (its single tree
        is used, or ``treename`` selects one); or a ``{treename: files}``
        mapping with exactly one entry (``files`` may be a path or a list). A
        ``str`` path may embed the tree as ``"file.root:treename"`` (the uproot
        convention; URL-scheme and Windows-drive colons are not separators). To
        read a file whose name contains a colon, pass it as a
        :class:`pathlib.Path` (never split) or use the mapping form.
    path : str or os.PathLike
        Destination ``.cstore`` file.
    treename : str or None, optional
        Tree to read when ``source`` is a bare ``.root`` path. ``None``
        auto-detects the file's sole tree and errors if there are several. It is
        an error to pass a ``treename`` that disagrees with an embedded
        ``"file.root:treename"``.
    columns : list[str] or None, optional
        Columns to store. ``None`` stores every fixed-size scalar column and
        skips the rest with a warning; naming a non-storable column is an error.
    batch_size : int, str, or None, optional
        Memory budget per batch. ``None`` reads everything in one pass; an
        ``int`` is rows per batch; a ``str`` (default ``"512 MiB"``) is a byte
        budget converted to rows from a one-row sample. Each batch becomes one
        record.
    compact : bool, optional
        Collapse the records into one afterward for the single-record fast
        read path. Defaults to True.
    mode : str, optional
        ``"create"`` (default), ``"recreate"``, or ``"update"``.
    show_progress : bool, optional
        Whether to display a progress bar. Defaults to True.

    Returns
    -------
    colstore.ColStoreReader
        An opened reader over the written file.
    """
    rdf = _as_rdataframe(source, treename)
    selected = _select_storable_columns(rdf, columns)
    total_rows = int(rdf.Count().GetValue())

    if total_rows == 0:
        rows_per_batch: int | None = None
    elif isinstance(batch_size, str):
        rows_per_batch = resolve_batch_rows(batch_size, bytes_per_row=_bytes_per_row(rdf, selected))
    else:
        rows_per_batch = resolve_batch_rows(batch_size)

    return write_column_batches(
        _ingest_batches(rdf, selected, rows_per_batch, total_rows),
        path,
        mode=mode,
        total_rows=total_rows,
        compact=compact,
        show_progress=show_progress,
        desc=f"{os.fspath(path)} <- ROOT",
    )


def _sanitize_branch_name(name: str) -> str:
    """Reduce a column name to a valid ROOT branch name (word characters only)."""
    sanitized = _INVALID_BRANCH_CHARS.sub("_", name).strip("_")
    if not sanitized:
        sanitized = "branch"
    if sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


def _sanitized_name_map(names: list[str]) -> dict[str, str]:
    """Map each column name to a unique valid ROOT branch name, preserving order.

    Names that collide after sanitizing (``"a b"`` and ``"a-b"`` both reduce to
    ``"a_b"``) get a numeric suffix so every branch name stays distinct.
    """
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for name in names:
        base = _sanitize_branch_name(name)
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        mapping[name] = candidate
    return mapping


def _relabel(chunk: ColumnBatch, name_map: dict[str, str]) -> ColumnBatch:
    """Return ``chunk`` with its keys renamed through ``name_map``."""
    return {name_map[name]: column for name, column in chunk.items()}


def _resolve_columns(reader: ColStoreReader, requested: list[str] | None) -> list[str]:
    """Return the columns to write, validated against the store's schema."""
    if requested is None:
        return reader.columns
    if not requested:
        raise ValueError("columns must name at least one column.")
    available = set(reader.columns)
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"Column(s) not found in the colstore file: {', '.join(missing)}.")
    return list(requested)


def to_root(
    source: ColStoreReader | StrPath,
    path: StrPath,
    *,
    treename: str = _DEFAULT_TREE_NAME,
    columns: list[str] | None = None,
    batch_size: int | str | None = _DEFAULT_BATCH_SIZE,
    compression_level: int = _DEFAULT_COMPRESSION_LEVEL,
    compression_algorithm: str | int | None = None,
    output_format: str | int | None = None,
    multithreading: bool | int = _DEFAULT_MULTITHREADING,
    show_progress: bool = True,
) -> ROOT.RDataFrame:
    """Write a ``.cstore`` file to a ROOT file and return an RDataFrame over it.

    RDataFrame writes one complete tree per Snapshot and has no supported way to
    append entries to an existing tree, so a store that does not fit in memory
    cannot be written with a single streaming Snapshot. Two paths handle this:

    * When the whole selection fits in one ``batch_size`` (or ``batch_size`` is
      ``None``), the columns are written in a single Snapshot. A colstore reader
      is memory-mapped, so the column arrays are handed to ROOT as zero-copy
      views and Snapshot streams them to disk; for a single-record store peak
      memory stays bounded by the page cache.
    * Otherwise each row-chunk is written to a temporary ROOT file and the chunk
      files are then merged into ``path`` by a single Snapshot over an
      RDataFrame built from all of them, so peak memory stays near one chunk
      regardless of file size. The temporary files are removed afterward, even
      on failure.

    The Snapshot that writes ``path`` (the single direct write, or the merge of
    the chunk files) is configured from ``compression_level``,
    ``compression_algorithm``, and ``output_format``; the transient chunk files
    are always written uncompressed since they are read back and deleted
    immediately. The whole write runs inside the requested implicit-MT state,
    which is restored afterward.

    Row order is preserved when implicit multithreading is off. ROOT may reorder
    rows of a Snapshot when MT is enabled, which is the default here for speed;
    pass ``multithreading=False`` when the row order must be preserved.

    Column names that are not valid ROOT branch names (containing spaces,
    brackets, or other symbols) are reduced to word characters before writing,
    with a warning naming each change; colliding results are disambiguated with
    a numeric suffix.

    Parameters
    ----------
    source : colstore.ColStoreReader, str, or os.PathLike
        An opened reader, or a path to a ``.cstore`` file.
    path : str or os.PathLike
        Destination ``.root`` file; recreated if it already exists.
    treename : str, optional
        Name of the tree to write. Defaults to ``"events"``.
    columns : list[str] or None, optional
        Columns to write, in this order. ``None`` (default) writes every column.
        Naming a column absent from the store is an error.
    batch_size : int, str, or None, optional
        Per-chunk memory budget for stores larger than one chunk: an ``int`` is
        rows per chunk; a ``str`` (default ``"512 MiB"``) is a byte budget;
        ``None`` forces a single Snapshot (no temporary files).
    compression_level : int, optional
        ``RSnapshotOptions.fCompressionLevel`` for the output. Defaults to ``0``
        (uncompressed); ROOT's own Snapshot default is ``5``.
    compression_algorithm : str, int, or None, optional
        ``RSnapshotOptions.fCompressionAlgorithm``. A string names a ROOT
        algorithm (``"zlib"``, ``"lzma"``, ``"lz4"``, ``"zstd"``); any other
        value is passed through, so a ROOT enum member may be given directly.
        ``None`` (default) leaves ROOT's choice in place.
    output_format : str, int, or None, optional
        ``RSnapshotOptions.fOutputFormat``. A string names a ROOT output format
        (``"default"``, ``"ttree"``, ``"rntuple"``); any other value is passed
        through. ``None`` (default) leaves ROOT's choice in place.
    multithreading : bool or int, optional
        Implicit-MT setting for the write, restored to its prior state on
        return. ``True`` (default) enables MT with all cores, an ``int`` enables
        MT with that many threads (``0`` means all cores), and ``False`` disables
        it.
    show_progress : bool, optional
        Whether to display a progress bar. Defaults to True.

    Returns
    -------
    ROOT.RDataFrame
        A data frame over the freshly written ROOT file.
    """
    root = _import_root()
    reader = source if isinstance(source, ColStoreReader) else api.open(source)
    selected = _resolve_columns(reader, columns)
    out_path = os.fspath(path)
    total_rows = reader.n_rows

    name_map = _sanitized_name_map(selected)
    renamed = {old: new for old, new in name_map.items() if old != new}
    if renamed:
        detail = ", ".join(f"{old!r} -> {new!r}" for old, new in renamed.items())
        warnings.warn(
            f"Sanitized colstore column name(s) to valid ROOT branch names: {detail}.",
            RuntimeWarning,
            stacklevel=2,
        )

    row_nbytes = sum(reader.dtypes[name].itemsize for name in selected)
    total_bytes = total_rows * row_nbytes
    rows_per_chunk = _resolve_chunk_rows(batch_size, row_nbytes)
    options = _build_options(
        root,
        fMode="RECREATE",
        compression_level=compression_level,
        compression_algorithm=compression_algorithm,
        output_format=output_format,
    )

    with (
        _implicit_mt(root, multithreading),
        progress_bar(
            total=total_bytes,
            desc=f"{out_path} <- colstore",
            unit="B",
            unit_scale=True,
            enabled=show_progress,
        ) as bar,
    ):
        if rows_per_chunk is None or total_rows <= rows_per_chunk:
            data = _relabel(reader[:, selected].dict(copy=False), name_map)
            _snapshot(root, data, treename, out_path, options)
            bar.update(total_bytes)
        else:
            _write_chunked(
                root,
                reader,
                selected,
                name_map,
                treename,
                out_path,
                rows_per_chunk,
                row_nbytes,
                options,
                bar,
            )

    return root.RDataFrame(treename, out_path)


def _compression_algorithm_value(root: ModuleType, value: str | int) -> Any:
    """Resolve a compression-algorithm alias to a ROOT enum member.

    A string is looked up in the alias table and mapped to the matching
    ``RCompressionSetting.EAlgorithm`` member; any other value is returned
    unchanged so a ROOT enum may be passed directly.
    """
    if not isinstance(value, str):
        return value
    try:
        member = _COMPRESSION_ALGORITHM_ALIASES[value.strip().lower()]
    except KeyError:
        valid = ", ".join(sorted(_COMPRESSION_ALGORITHM_ALIASES))
        raise ValueError(
            f"Unknown compression_algorithm {value!r}; expected one of {valid}, "
            "or a ROOT RCompressionSetting.EAlgorithm value."
        ) from None
    return getattr(root.RCompressionSetting.EAlgorithm, member)


def _output_format_value(root: ModuleType, value: str | int) -> Any:
    """Resolve an output-format alias to a ROOT ``ESnapshotOutputFormat`` member.

    A string is mapped through the alias table; any other value is returned
    unchanged so a ROOT enum may be passed directly.
    """
    if not isinstance(value, str):
        return value
    try:
        member = _OUTPUT_FORMAT_ALIASES[value.strip().lower()]
    except KeyError:
        valid = ", ".join(sorted(_OUTPUT_FORMAT_ALIASES))
        raise ValueError(
            f"Unknown output_format {value!r}; expected one of {valid}, "
            "or a ROOT RDF.ESnapshotOutputFormat value."
        ) from None
    return getattr(root.RDF.ESnapshotOutputFormat, member)


def _build_options(
    root: ModuleType,
    *,
    fMode: str,
    compression_level: int,
    compression_algorithm: str | int | None,
    output_format: str | int | None,
) -> Any:
    """Build an ``RSnapshotOptions`` from the export's options."""
    options = root.RDF.RSnapshotOptions()
    options.fMode = fMode
    options.fCompressionLevel = compression_level
    if compression_algorithm is not None:
        options.fCompressionAlgorithm = _compression_algorithm_value(root, compression_algorithm)
    if output_format is not None:
        options.fOutputFormat = _output_format_value(root, output_format)
    return options


def _apply_implicit_mt(root: ModuleType, setting: bool | int) -> None:
    """Apply an implicit-MT ``setting`` (bool toggles, int sets the thread count)."""
    if setting is True:
        root.EnableImplicitMT()
    elif setting is False:
        root.DisableImplicitMT()
    elif isinstance(setting, int):
        root.EnableImplicitMT(setting)
    else:
        raise TypeError(f"multithreading must be a bool or int, got {type(setting).__name__}.")


@contextlib.contextmanager
def _implicit_mt(root: ModuleType, setting: bool | int) -> Iterator[None]:
    """Apply ``setting`` for the duration of the block, then restore ROOT's state.

    The prior implicit-MT state (and its thread-pool size, when enabled) is
    captured up front and put back on exit, so the export does not leak a global
    MT change into the caller's process.
    """
    was_enabled = root.IsImplicitMTEnabled()
    previous_threads = root.GetThreadPoolSize() if was_enabled else 0
    _apply_implicit_mt(root, setting)
    try:
        yield
    finally:
        root.DisableImplicitMT()
        if was_enabled:
            root.EnableImplicitMT(previous_threads)


def _resolve_chunk_rows(batch_size: int | str | None, row_nbytes: int) -> int | None:
    """Resolve ``batch_size`` to rows per chunk (``None`` means a single Snapshot)."""
    if isinstance(batch_size, str):
        return resolve_batch_rows(batch_size, bytes_per_row=row_nbytes or 1)
    return resolve_batch_rows(batch_size)


def _snapshot(
    root: ModuleType, data: ColumnBatch, treename: str, out_path: str, options: Any
) -> None:
    """Write ``data`` as a single tree with the given Snapshot ``options``."""
    root.RDF.FromNumpy(data).Snapshot(treename, out_path, list(data), options)


def _write_chunked(
    root: ModuleType,
    reader: ColStoreReader,
    selected: list[str],
    name_map: dict[str, str],
    treename: str,
    out_path: str,
    rows_per_chunk: int,
    row_nbytes: int,
    options: Any,
    bar: Any,
) -> None:
    """Snapshot each row-chunk to a temporary file, then merge them into out_path.

    Each chunk is one valid single-tree Snapshot; the chunk files are read back
    as one RDataframe (a TChain) and snapshotted into the final tree with the
    caller's ``options``. The chunk files are transient and so are written
    uncompressed in the default format. The scratch directory is created beside
    the output (same filesystem) and removed afterward whether or not the merge
    succeeds.
    """
    total_rows = reader.n_rows
    branch_names = [name_map[name] for name in selected]
    chunk_options = _build_options(
        root,
        fMode="RECREATE",
        compression_level=0,
        compression_algorithm=None,
        output_format=None,
    )
    scratch_dir = tempfile.mkdtemp(prefix=_TMP_DIR_PREFIX, dir=os.path.dirname(out_path) or ".")
    try:
        chunk_paths: list[str] = []
        for index, start in enumerate(range(0, total_rows, rows_per_chunk)):
            end = min(start + rows_per_chunk, total_rows)
            chunk = _relabel(reader[start:end, selected].dict(copy=False), name_map)
            chunk_path = os.path.join(scratch_dir, f"chunk_{index:06d}.root")
            _snapshot(root, chunk, treename, chunk_path, chunk_options)
            chunk_paths.append(chunk_path)
            bar.update((end - start) * row_nbytes)

        root.RDataFrame(treename, chunk_paths).Snapshot(treename, out_path, branch_names, options)
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


class RootParser(Parser):
    """Two-way parser between ROOT files and colstore.

    Thin object wrapper over :func:`from_root` and :func:`to_root`, which carry
    the full typed signatures: :meth:`read` parses a ROOT file (``from_root``)
    and :meth:`write` emits one (``to_root``).
    """

    format_name = "root"

    def read(self, source: Any, path: StrPath, **kwargs: Any) -> ColStoreReader:
        return from_root(source, path, **kwargs)

    def write(
        self, source: ColStoreReader | StrPath, path: StrPath, **kwargs: Any
    ) -> ROOT.RDataFrame:
        return to_root(source, path, **kwargs)
