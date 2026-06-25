"""The ROOT file format for colstore, with two interchangeable backends.

A ROOT file is read and written through one of two backends, selected by
``backend``: ROOT's own RDataFrame (PyROOT) or the pure-Python uproot library.
``"auto"`` (the default) uses PyROOT when it is importable and otherwise falls
back to uproot. Each backend is imported lazily, inside the backend that uses it,
so importing this module -- and therefore ``import colstore`` -- pulls in
neither ROOT nor uproot. Annotations that name ROOT types stay strings at
runtime (``from __future__ import annotations``); ROOT ships no type stubs and
the mypy config ignores missing imports.

Only fixed-size scalar branches can be stored: a ``.cstore`` column is a
contiguous block of fixed-width values, so jagged branches (``RVec``,
``vector``), array branches, and ``char*`` are skipped (``keep_valid_only``) or
rejected. Everything stored round-trips through ``numpy``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import re
import shutil
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias, cast

import numpy as np

from .. import api
from .._base import _ReaderBase
from .._sizes import resolve_batch_rows
from ..progress import progress_bar
from ..reader import ColStoreReader
from ._streaming import ColumnBatch, StrPath, write_column_batches
from .base import FileFormat, Selection

if TYPE_CHECKING:
    import ROOT

#: The backend selector accepted by :func:`from_root` / :func:`to_root`.
RootBackendName: TypeAlias = str

#: A ROOT source accepted by :func:`from_root`.
RootSource: TypeAlias = "ROOT.RDataFrame | str | os.PathLike[str] | dict[str, str | list[str]]"

# NumPy dtype kinds that map to a fixed-size colstore column. Both backends judge
# storability from the MATERIALIZED dtype (a one-row sample), not a C++ type-name
# table, so they agree exactly: a jagged / array branch samples to an object or
# multi-dimensional array and is rejected; a scalar branch samples to a 1-D
# numeric/bool array regardless of how RDataFrame or uproot spells its C++ type.
_SCALAR_KINDS = frozenset("fiub")

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
            "The ROOT backend requires PyROOT, which is not installed. Install ROOT "
            "(e.g. 'conda install -c conda-forge root'), or use backend='uproot'."
        ) from exc
    return cast(ModuleType, ROOT)


def require_existing(requested: list[str], available: list[str]) -> None:
    """Raise ``ValueError`` for any requested column absent from the source tree.

    Existence is checked before storability so a typo'd or missing name is a hard
    error (like the export side), not a silent skip under ``keep_valid_only=True``.
    """
    known = set(available)
    missing = [name for name in requested if name not in known]
    if missing:
        raise ValueError(f"Column(s) not found in the ROOT tree: {', '.join(missing)}.")


def filter_storable(
    names: list[str],
    is_storable: Any,
    keep_valid_only: bool,
) -> list[str]:
    """Partition ``names`` into the columns to keep, applying ``keep_valid_only``.

    ``is_storable(name) -> bool`` reports whether a column maps to a fixed-size
    colstore column. ``keep_valid_only`` governs every column uniformly --
    auto-discovered or explicitly requested: ``True`` keeps the storable ones and
    skips the rest with a warning naming them; ``False`` raises if any column is
    not storable. Shared by both backends so the policy is identical.
    """
    if not names:
        raise ValueError("No columns to read from the source.")
    kept = [name for name in names if is_storable(name)]
    skipped = [name for name in names if name not in kept]
    if skipped and not keep_valid_only:
        raise ValueError(
            f"Column(s) cannot be stored as fixed-size colstore columns: {', '.join(skipped)} "
            "(pass keep_valid_only=True to skip them instead)."
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


def _select_storable_columns(
    rdf: ROOT.RDF.RNode,
    requested: list[str] | None,
    keep_valid_only: bool,
) -> list[str]:
    """Choose the RDataFrame columns to store, applying ``keep_valid_only``.

    Storability is read from the materialized one-row sample (1-D, numeric/bool),
    not the declared C++ type name -- so RDataFrame's ``std::int32_t`` spelling for
    a uproot-written or RNTuple integer branch is recognized like any other
    integer, matching the uproot backend exactly.
    """
    available = [str(name) for name in rdf.GetColumnNames()]
    names = available if requested is None else [str(name) for name in requested]
    if requested is not None:
        require_existing(names, available)
    sample = rdf.Range(0, 1).AsNumpy(columns=names) if names else {}

    def is_storable(name: str) -> bool:
        array = sample[name]
        return getattr(array, "ndim", 0) == 1 and array.dtype.kind in _SCALAR_KINDS

    return filter_storable(names, is_storable, keep_valid_only)


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


class RootBackend(ABC):
    """A backend that reads and writes ROOT files for the ROOT file format.

    Two backends implement this: :class:`RootCppBackend` (PyROOT / RDataFrame) and
    :class:`~colstore.interop._uproot.UprootBackend` (the pure-Python uproot
    library). Both stream in bounded memory -- one batch per record in each
    direction. :func:`resolve_backend` picks one from the ``backend`` selector.
    """

    name: ClassVar[str]

    @staticmethod
    @abstractmethod
    def available() -> bool:
        """Whether this backend's backend is importable."""

    @abstractmethod
    def read_batches(
        self,
        source: Any,
        *,
        treename: str | None,
        columns: list[str] | None,
        keep_valid_only: bool,
        batch_size: int | str | None,
    ) -> tuple[Iterator[ColumnBatch], int]:
        """Read ``source``; return ``(batches, total_rows)``, one batch per record."""

    @abstractmethod
    def write(
        self,
        reader: _ReaderBase,
        *,
        columns: list[str],
        dest: StrPath,
        treename: str,
        batch_size: int | str | None,
        show_progress: bool,
        **options: Any,
    ) -> None:
        """Stream ``reader``'s ``columns`` to a ROOT file at ``dest``."""


class RootCppBackend(RootBackend):
    """ROOT's own RDataFrame, via PyROOT: reads with ``AsNumpy``, writes with ``Snapshot``."""

    name: ClassVar[str] = "ROOT"

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("ROOT") is not None

    def read_batches(
        self,
        source: Any,
        *,
        treename: str | None,
        columns: list[str] | None,
        keep_valid_only: bool,
        batch_size: int | str | None,
    ) -> tuple[Iterator[ColumnBatch], int]:
        rdf = _as_rdataframe(source, treename)
        selected = _select_storable_columns(rdf, columns, keep_valid_only)
        total = int(rdf.Count().GetValue())
        if total == 0:
            rows_per_batch: int | None = None
        elif isinstance(batch_size, str):
            rows_per_batch = resolve_batch_rows(
                batch_size, bytes_per_row=_bytes_per_row(rdf, selected)
            )
        else:
            rows_per_batch = resolve_batch_rows(batch_size)
        return _ingest_batches(rdf, selected, rows_per_batch, total), total

    def write(
        self,
        reader: _ReaderBase,
        *,
        columns: list[str],
        dest: StrPath,
        treename: str,
        batch_size: int | str | None,
        show_progress: bool,
        **options: Any,
    ) -> None:
        root = _import_root()
        dtypes = reader.dtypes
        # ROOT's Snapshot cannot build a leaflist for an 8-bit fundamental integer
        # (Char_t): it silently drops the column under MT and segfaults otherwise.
        # Reject up front rather than crash; the uproot backend writes int8 fine.
        tiny_ints = [n for n in columns if dtypes[n].kind in "iu" and dtypes[n].itemsize == 1]
        if tiny_ints:
            raise TypeError(
                f"the ROOT backend cannot write 8-bit integer column(s) {tiny_ints} "
                f"(ROOT's Snapshot mishandles them); use backend='uproot', or cast to int16."
            )
        out_path = os.fspath(dest)
        total_rows = reader.n_rows
        name_map = _sanitized_name_map(columns)
        _warn_branch_renames(name_map)
        row_nbytes = sum(dtypes[name].itemsize for name in columns)
        total_bytes = total_rows * row_nbytes
        rows_per_chunk = _resolve_chunk_rows(batch_size, row_nbytes)
        snapshot_options = _build_options(
            root,
            fMode="RECREATE",
            compression_level=options.get("compression_level", _DEFAULT_COMPRESSION_LEVEL),
            compression_algorithm=options.get("compression_algorithm"),
            output_format=options.get("output_format"),
        )
        with (
            _implicit_mt(root, options.get("multithreading", _DEFAULT_MULTITHREADING)),
            progress_bar(
                total=total_bytes,
                desc=f"{out_path} <- colstore",
                unit="B",
                unit_scale=True,
                enabled=show_progress,
            ) as bar,
        ):
            if rows_per_chunk is None or total_rows <= rows_per_chunk:
                data = _relabel(_batch_dict(reader[:, columns]), name_map)
                _snapshot(root, data, treename, out_path, snapshot_options)
                bar.update(total_bytes)
            else:
                _write_chunked(
                    root,
                    reader,
                    columns,
                    name_map,
                    treename,
                    out_path,
                    rows_per_chunk,
                    row_nbytes,
                    snapshot_options,
                    bar,
                )


def resolve_backend(backend: RootBackendName, source: Any = None) -> RootBackend:
    """Pick a :class:`RootBackend` from the ``backend`` selector.

    ``"auto"`` (default) uses PyROOT when it is importable, otherwise uproot,
    raising if neither is installed. ``"ROOT"`` and ``"uproot"`` request a
    specific backend and raise if it is missing. The match is case-insensitive.
    """
    name = backend.lower()
    if name == "auto":
        if _is_rdataframe_source(source):
            # Only the ROOT backend can read an in-memory RDataFrame.
            if RootCppBackend.available():
                return RootCppBackend()
            raise ImportError(
                "reading an RDataFrame source needs PyROOT; pass a path or {tree: files} "
                "to read with the uproot backend instead."
            )
        if RootCppBackend.available():
            return RootCppBackend()
        if _uproot_available():
            return _uproot_backend()
        raise ImportError(
            "Reading or writing ROOT files needs PyROOT or uproot; neither is installed "
            "(install one, e.g. 'pip install uproot' or 'conda install -c conda-forge root')."
        )
    if name == "root":
        if not RootCppBackend.available():
            raise ImportError("backend='ROOT' needs PyROOT; install ROOT or use backend='uproot'.")
        return RootCppBackend()
    if name == "uproot":
        if not _uproot_available():
            raise ImportError("backend='uproot' needs uproot; install it or use backend='ROOT'.")
        return _uproot_backend()
    raise ValueError(f"unknown backend {backend!r}; expected 'auto', 'ROOT', or 'uproot'.")


def _is_rdataframe_source(source: Any) -> bool:
    """Whether ``source`` is an in-memory RDataFrame (ROOT-backend-only) rather than a path.

    A path / mapping / list, a colstore reader or dataset (the export side), or
    ``None`` are all readable without an RDataFrame; anything else is taken to be an
    RDataFrame/RNode, which only the PyROOT backend can read.
    """
    return source is not None and not isinstance(
        source, (str, os.PathLike, dict, list, tuple, _ReaderBase)
    )


def _uproot_available() -> bool:
    return importlib.util.find_spec("uproot") is not None


def _uproot_backend() -> RootBackend:
    from ._uproot import UprootBackend  # lazy: do not import uproot until it is selected

    return UprootBackend()


def from_root(
    source: RootSource,
    path: StrPath,
    *,
    backend: RootBackendName = "auto",
    treename: str | None = None,
    columns: list[str] | None = None,
    keep_valid_only: bool = True,
    batch_size: int | str | None = _DEFAULT_BATCH_SIZE,
    compact: bool = True,
    mode: str = "create",
    show_progress: bool = True,
) -> ColStoreReader:
    """Convert a ROOT source into a ``.cstore`` file and open it for reading.

    Parameters
    ----------
    source : ROOT.RDataFrame, str, os.PathLike, or dict
        An existing ``RDataFrame`` (read only by the ``"ROOT"`` backend); a path
        to a ``.root`` file (its single tree is used, or ``treename`` selects
        one); or a ``{treename: files}`` mapping with exactly one entry (``files``
        may be a path or a list). A ``str`` path may embed the tree as
        ``"file.root:treename"`` (the uproot convention; URL-scheme and
        Windows-drive colons are not separators). To read a file whose name
        contains a colon, pass it as a :class:`pathlib.Path` (never split) or use
        the mapping form.
    path : str or os.PathLike
        Destination ``.cstore`` file.
    backend : str, optional
        Backend to read with: ``"auto"`` (default) uses PyROOT if importable,
        else uproot; ``"ROOT"`` forces PyROOT/RDataFrame; ``"uproot"`` forces
        uproot. Raises if the requested backend is missing.
    treename : str or None, optional
        Tree to read when ``source`` is a bare ``.root`` path. ``None``
        auto-detects the file's sole tree and errors if there are several. It is
        an error to pass a ``treename`` that disagrees with an embedded
        ``"file.root:treename"``.
    columns : list[str] or None, optional
        Columns to store, in this order. ``None`` (default) considers every
        column in the tree.
    keep_valid_only : bool, optional
        ``True`` (default) keeps the fixed-size scalar columns and skips any
        non-storable (jagged / array / pointer) column with a warning, whether it
        was auto-discovered or named in ``columns``. ``False`` raises if any
        column in scope is not storable.
    batch_size : int, str, or None, optional
        Memory budget per batch. ``None`` reads everything in one pass; an
        ``int`` is rows per batch; a ``str`` (default ``"512 MiB"``) is a byte
        budget converted to rows. Each batch becomes one record.
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
    return RootFormat().from_file(
        source,
        path,
        backend=backend,
        treename=treename,
        columns=columns,
        keep_valid_only=keep_valid_only,
        batch_size=batch_size,
        compact=compact,
        mode=mode,
        show_progress=show_progress,
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


def _batch_dict(view: Any) -> ColumnBatch:
    """Materialize a view's columns, zero-copy where the layout allows it.

    A single-record store hands ROOT zero-copy memmap views; a dataset slice that
    spans files cannot be viewed contiguously, so it falls back to a copy.
    """
    try:
        return cast(ColumnBatch, view.dict(copy=False))
    except ValueError:
        return cast(ColumnBatch, view.dict())


class _InMemoryView:
    """A row/column slice of in-memory columns, exposing the ``dict()`` accessor
    :func:`_batch_dict` calls. The slice is a NumPy view, so a batch handed to a
    ROOT backend copies nothing.
    """

    def __init__(self, columns: ColumnBatch) -> None:
        self._columns = columns

    def dict(self, copy: bool = True) -> ColumnBatch:
        if copy:
            return {name: array.copy() for name, array in self._columns.items()}
        return self._columns


class _InMemorySource:
    """A reader-like wrapper over already-gathered column arrays.

    A row/column subset export has no contiguous store to stream, so its columns
    are gathered into memory. This exposes the small slice of the store-reader
    protocol the ROOT backends consume -- ``dtypes``, ``n_rows``, and
    ``[rows, columns]`` chunk indexing -- so that in-memory subset streams to ROOT
    directly, rather than being written to a scratch ``.cstore`` and reopened only
    to present the same protocol.
    """

    def __init__(self, data: ColumnBatch) -> None:
        self._data = data
        self.dtypes: dict[str, np.dtype[Any]] = {name: array.dtype for name, array in data.items()}
        self.n_rows: int = len(next(iter(data.values()))) if data else 0

    def __getitem__(self, key: tuple[Any, list[str]]) -> _InMemoryView:
        rows, columns = key
        return _InMemoryView({name: self._data[name][rows] for name in columns})


def _warn_branch_renames(name_map: dict[str, str]) -> None:
    """Warn (once) about column names that were sanitized to valid branch names."""
    renamed = {old: new for old, new in name_map.items() if old != new}
    if renamed:
        detail = ", ".join(f"{old!r} -> {new!r}" for old, new in renamed.items())
        warnings.warn(
            f"Sanitized colstore column name(s) to valid ROOT branch names: {detail}.",
            RuntimeWarning,
            stacklevel=2,
        )


def _resolve_columns(reader: _ReaderBase, requested: list[str] | None) -> list[str]:
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
    source: _ReaderBase | StrPath,
    path: StrPath,
    *,
    backend: RootBackendName = "auto",
    treename: str = _DEFAULT_TREE_NAME,
    columns: list[str] | None = None,
    batch_size: int | str | None = _DEFAULT_BATCH_SIZE,
    compression_level: int = _DEFAULT_COMPRESSION_LEVEL,
    compression_algorithm: str | int | None = None,
    output_format: str | int | None = None,
    multithreading: bool | int = _DEFAULT_MULTITHREADING,
    show_progress: bool = True,
) -> Path:
    """Write a colstore reader, dataset, or selection out to a ROOT file.

    The columns are streamed to ``path`` in bounded memory, one ``batch_size``
    batch at a time, so a store larger than memory writes fine. Column names that
    are not valid ROOT branch names (containing spaces, brackets, or other
    symbols) are reduced to word characters first, with a warning naming each
    change; colliding results are disambiguated with a numeric suffix.

    Parameters
    ----------
    source : colstore reader or dataset, str, or os.PathLike
        An opened reader or dataset, or a path to a ``.cstore`` file.
    path : str or os.PathLike
        Destination ``.root`` file; recreated if it already exists.
    backend : str, optional
        Backend to write with: ``"auto"`` (default) uses PyROOT if importable,
        else uproot; ``"ROOT"`` forces PyROOT/RDataFrame; ``"uproot"`` forces
        uproot. Raises if the requested backend is missing.
    treename : str, optional
        Name of the tree to write. Defaults to ``"events"``.
    columns : list[str] or None, optional
        Columns to write, in this order. ``None`` (default) writes every column.
        Naming a column absent from the store is an error.
    batch_size : int, str, or None, optional
        Per-batch memory budget: an ``int`` is rows per batch; a ``str`` (default
        ``"512 MiB"``) is a byte budget; ``None`` writes in a single pass.
    compression_level, compression_algorithm, output_format, multithreading
        ROOT-backend options (ignored by the uproot backend). ``compression_level``
        defaults to ``0`` (uncompressed; ROOT's own default is 5). The string
        ``compression_algorithm`` (``"zlib"``/``"lzma"``/``"lz4"``/``"zstd"``) and
        ``output_format`` (``"default"``/``"ttree"``/``"rntuple"``) name ROOT
        enums; any other value is passed through. ``multithreading`` toggles
        implicit MT for the Snapshot (an ``int`` sets the thread count), restored
        afterward; MT may reorder rows, so pass ``False`` to preserve order.
    show_progress : bool, optional
        Whether to display a progress bar. Defaults to True.

    Returns
    -------
    pathlib.Path
        The path the ROOT file was written to.
    """
    reader = source if isinstance(source, _ReaderBase) else api.open(source)
    selected = _resolve_columns(reader, columns)
    target = reader if columns is None else reader[:, selected]
    return RootFormat().to_file(
        target._interop_selection(),
        path,
        backend=backend,
        treename=treename,
        batch_size=batch_size,
        show_progress=show_progress,
        compression_level=compression_level,
        compression_algorithm=compression_algorithm,
        output_format=output_format,
        multithreading=multithreading,
    )


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
    reader: _ReaderBase,
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
            chunk = _relabel(_batch_dict(reader[start:end, selected]), name_map)
            chunk_path = os.path.join(scratch_dir, f"chunk_{index:06d}.root")
            _snapshot(root, chunk, treename, chunk_path, chunk_options)
            chunk_paths.append(chunk_path)
            bar.update((end - start) * row_nbytes)

        root.RDataFrame(treename, chunk_paths).Snapshot(treename, out_path, branch_names, options)
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


class RootFormat(FileFormat):
    """The ROOT file format (``colstore.interop.root``), via PyROOT or uproot.

    :meth:`to_file` / :meth:`from_file` carry the conversion; :func:`to_root` /
    :func:`from_root` are the typed module-level shorthands that delegate here, and
    ``ds.saveas("x.root")`` / ``colstore.ingest("x.root", dest)`` reach them through
    the registry.
    """

    name: ClassVar[str] = "root"
    extensions: ClassVar[frozenset[str]] = frozenset({".root"})

    def to_file(
        self,
        selection: Selection,
        dest: Any,
        *,
        backend: RootBackendName = "auto",
        treename: str = _DEFAULT_TREE_NAME,
        batch_size: int | str | None = _DEFAULT_BATCH_SIZE,
        show_progress: bool = True,
        **options: Any,
    ) -> Path:
        """Write a colstore ``selection`` to a ROOT file at ``dest``; see :func:`to_root`.

        A whole-store selection streams straight from the store; a row subset is
        gathered into memory and streamed from there. ``**options`` carries the
        ROOT-backend write options (``compression_level``,
        ``compression_algorithm``, ``output_format``, ``multithreading``).
        """
        impl = resolve_backend(backend, selection.store)
        columns = list(selection.columns)
        string_columns = [name for name in columns if selection.native_dtype(name).kind in "US"]
        if string_columns:
            raise TypeError(
                f"the ROOT backend cannot write string column(s) {string_columns}; drop or cast "
                f"them (ROOT branches have no fixed-width string type)."
            )
        if selection.is_whole_column():
            impl.write(
                selection.store,
                columns=columns,
                dest=dest,
                treename=treename,
                batch_size=batch_size,
                show_progress=show_progress,
                **options,
            )
        else:
            data = {name: selection.gather(name) for name in columns}
            impl.write(
                cast(_ReaderBase, _InMemorySource(data)),
                columns=columns,
                dest=dest,
                treename=treename,
                batch_size=batch_size,
                show_progress=show_progress,
                **options,
            )
        return Path(os.fspath(dest))

    def from_file(
        self,
        source: Any,
        dest: Any,
        *,
        backend: RootBackendName = "auto",
        treename: str | None = None,
        columns: list[str] | None = None,
        keep_valid_only: bool = True,
        batch_size: int | str | None = _DEFAULT_BATCH_SIZE,
        compact: bool = True,
        mode: str = "create",
        show_progress: bool = True,
    ) -> ColStoreReader:
        """Read a ROOT ``source`` into a ``.cstore`` and open it. See :func:`from_root`."""
        impl = resolve_backend(backend, source)
        batches, total_rows = impl.read_batches(
            source,
            treename=treename,
            columns=columns,
            keep_valid_only=keep_valid_only,
            batch_size=batch_size,
        )
        return write_column_batches(
            batches,
            dest,
            mode=mode,
            total_rows=total_rows,
            compact=compact,
            show_progress=show_progress,
            desc=f"{os.fspath(dest)} <- ROOT ({impl.name})",
        )
