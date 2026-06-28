"""Framework for two-way interoperability between colstore and external formats.

colstore exchanges data with external formats, each represented by a
:class:`Format` registered under a short name. There are two kinds, kept distinct
because colstore mmaps only its own format:

* a :class:`DataFormat` bridges an in-memory object (e.g. an Arrow table): export
  returns the object, import writes a ``.cstore`` from it; selected by name;
* a :class:`FileFormat` bridges an on-disk file (e.g. a ROOT file): export writes
  the file, import reads it into a ``.cstore``; selected by the file
  :attr:`~FileFormat.extensions` it claims.

Importing any external source is a materialization -- colstore mmaps only its own
format, so a ``.cstore`` is written and then opened. Formats are discovered by
name through the registry here (:func:`get`, :func:`data_formats`,
:func:`file_formats`, :func:`register`). The reader, dataset, and view classes
mix in :class:`InteropMixin` for the data-format export surface (``to(name)`` and
the ``arrow()`` shorthand).

A backend (e.g. ``pyarrow``) is imported only when a conversion runs -- ``import
colstore`` loads neither the format modules nor their optional dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from .._base import _ReaderBase
    from ..reader import ColStoreReader


def is_whole_column(row_indexer: Any, n_rows: int) -> bool:
    """Whether a resolved row selector covers an entire column in order.

    ``None`` and a unit-step slice spanning ``[0, n_rows)`` (the resolved form of
    ``ds[:, name]``) both mean the whole column -- the case a format can serve with
    zero copy; any other selector requires a gather.
    """
    if row_indexer is None:
        return True
    if isinstance(row_indexer, slice):
        start, stop, step = row_indexer.indices(n_rows)
        return step == 1 and start == 0 and stop == n_rows
    return False


@dataclass(frozen=True)
class Selection:
    """A resolved column-and-row selection handed to a format's exporter.

    Built by :class:`InteropMixin` from each object's :meth:`_interop_target`, so
    a reader, dataset, ``ColumnView``, or ``TableView`` all present one uniform
    handle. The accessors forward to the store's existing read seams -- a format
    materializes the selection however it needs without re-implementing indexing.
    """

    store: _ReaderBase
    columns: list[str]
    row_indexer: Any
    single: bool

    def gather(self, name: str) -> NDArray[Any]:
        """One selected column as an owning 1-D array."""
        return self.store._gather_one(name, self.row_indexer)

    def gather_all(self) -> dict[str, NDArray[Any]]:
        """Every selected column as a name -> owning array dict."""
        return {name: self.gather(name) for name in self.columns}

    def column_chunks(self, name: str) -> list[NDArray[Any]]:
        """Zero-copy native views of one whole column, one per on-disk segment."""
        return self.store._column_chunks(name)

    def native_dtype(self, name: str) -> np.dtype[Any]:
        """One column's dtype in native byte order."""
        return self.store._native_dtype(name)

    def is_whole_column(self) -> bool:
        """Whether the row selection covers every row in order (zero-copy eligible)."""
        return is_whole_column(self.row_indexer, self.store.n_rows)


class Format:
    """Base class for a two-way bridge between colstore and one external format.

    A concrete format subclasses :class:`DataFormat` or :class:`FileFormat` (for its
    I/O model) and sets a :attr:`name`; defining it is enough to make it available
    under that name.

    Attributes
    ----------
    name : str
        Canonical short name for the format, e.g. ``"arrow"``.
    kind : str
        ``"data"`` (in-memory object) or ``"file"`` (on-disk path).
    aliases : frozenset[str]
        Additional names the format answers to, for formats known by several names.
    """

    name: ClassVar[str]
    kind: ClassVar[str]
    aliases: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, *, override: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # The kind bases (DataFormat/FileFormat) set ``kind`` but no ``name``; only a
        # concrete format -- one declaring its own ``name`` -- registers itself.
        # ``class MyFmt(DataFormat, override=True)`` replaces an existing name.
        if "name" not in cls.__dict__:
            return
        if not getattr(cls, "kind", None):
            raise TypeError(
                f"{cls.__name__} must subclass DataFormat or FileFormat (it sets no 'kind')."
            )
        # Auto-registration: the class declares its own name, so skip the entry-point
        # claim scan (an importlib.metadata.entry_points() walk over every installed
        # distribution) -- it would only confirm the name the class already owns.
        register(cls(), override=override, _check_claims=False)


class DataFormat(Format):
    """A format whose external representation is an in-memory object.

    Override :meth:`to_object` to support export (colstore -> object) and
    :meth:`from_object` to support import (object -> ``.cstore``). The unsupported
    direction raises; :attr:`can_export` / :attr:`can_import` report which are
    available.
    """

    kind: ClassVar[str] = "data"

    def to_object(self, selection: Selection) -> Any:
        """Export a colstore ``selection`` to this format's object (override to support)."""
        raise NotImplementedError(f"the {self.name!r} format does not support export.")

    def from_object(self, obj: Any, dest: Any, *args: Any, **kwargs: Any) -> ColStoreReader:
        """Import ``obj`` into a ``.cstore`` at ``dest`` and open it (override to support)."""
        raise NotImplementedError(f"the {self.name!r} format does not support import.")

    @property
    def can_export(self) -> bool:
        """Whether this format implements export (colstore -> object)."""
        return type(self).to_object is not DataFormat.to_object

    @property
    def can_import(self) -> bool:
        """Whether this format implements import (object -> colstore)."""
        return type(self).from_object is not DataFormat.from_object


class FileFormat(Format):
    """A format whose external representation is an on-disk file.

    The :attr:`extensions` it claims drive ``colstore.ingest`` / ``saveas``
    dispatch. Override :meth:`to_file` to support export (colstore -> file) and
    :meth:`from_file` to support import (file -> ``.cstore``).
    """

    kind: ClassVar[str] = "file"
    #: File extensions (lowercase, with the dot) this format handles.
    extensions: ClassVar[frozenset[str]] = frozenset()

    def to_file(self, selection: Selection, dest: Any, *args: Any, **kwargs: Any) -> Any:
        """Write a colstore ``selection`` to a file at ``dest`` (override to support)."""
        raise NotImplementedError(f"the {self.name!r} format does not support export.")

    def from_file(self, source: Any, dest: Any, *args: Any, **kwargs: Any) -> ColStoreReader:
        """Read ``source`` into a ``.cstore`` at ``dest`` and open it (override to support)."""
        raise NotImplementedError(f"the {self.name!r} format does not support import.")

    @property
    def can_export(self) -> bool:
        """Whether this format implements export (colstore -> file)."""
        return type(self).to_file is not FileFormat.to_file

    @property
    def can_import(self) -> bool:
        """Whether this format implements import (file -> colstore)."""
        return type(self).from_file is not FileFormat.from_file


# Formats are discovered through packaging entry points -- one group per kind --
# so the registry is not a hard-coded table: a built-in or third-party format is
# registered by declaring an entry point (colstore's own are in pyproject). The
# group encodes the kind, so the accessors list formats by kind without importing
# the format module or its backend; the class loads only on get(). The cache also
# holds runtime registrations made through register().
_GROUPS: dict[str, str] = {"data": "colstore.data_formats", "file": "colstore.file_formats"}
_REGISTRY: dict[str, Format] = {}


def _entry_point_class(name: str) -> str | None:
    """The class name an entry point declares for ``name``, or ``None`` if none does."""
    for group in _GROUPS.values():
        for ep in entry_points(group=group):
            if ep.name == name:
                return ep.value.rsplit(":", 1)[-1]
    return None


def register(fmt: Format, *, override: bool = False, _check_claims: bool = True) -> None:
    """Register ``fmt`` under its :attr:`~Format.name` and any :attr:`~Format.aliases`.

    Usually automatic -- defining a :class:`Format` subclass registers it -- but
    callable directly for a dynamically built format. A name or alias already taken
    by a different format raises ``ValueError`` unless ``override=True``.

    ``_check_claims`` guards a name against an entry point that declares a *different*
    class. That scans the installed entry points, so auto-registration -- where the
    class owns the name it declares -- passes ``False`` to skip the walk.
    """
    names = (fmt.name, *sorted(fmt.aliases))
    if not override:
        for nm in names:
            existing = _REGISTRY.get(nm)
            if existing is not None and type(existing).__qualname__ != type(fmt).__qualname__:
                raise ValueError(
                    f"format name {nm!r} is already registered to {type(existing).__name__!r}; "
                    f"choose another name/alias or pass override=True."
                )
            if _check_claims:
                claimed = _entry_point_class(nm)
                if claimed is not None and claimed != type(fmt).__name__:
                    raise ValueError(
                        f"format name {nm!r} is claimed by the {claimed!r} entry point; "
                        f"choose another name/alias or pass override=True."
                    )
        collision = _extension_collision(fmt)
        if collision is not None:
            ext, owner = collision
            raise ValueError(
                f"file extension {ext!r} is already claimed by the {owner!r} format; "
                f"choose another extension or pass override=True."
            )
    for nm in names:
        _REGISTRY[nm] = fmt


def _extension_collision(fmt: Format) -> tuple[str, str] | None:
    """An (extension, owner-name) a *different* file format already claims, or ``None``.

    File extensions match case-insensitively, so the check compares lowercased.
    """
    if not isinstance(fmt, FileFormat):
        return None
    claimed = {
        ext.lower(): other.name
        for other in _REGISTRY.values()
        if isinstance(other, FileFormat) and type(other).__qualname__ != type(fmt).__qualname__
        for ext in other.extensions
    }
    for ext in fmt.extensions:
        if ext.lower() in claimed:
            return ext, claimed[ext.lower()]
    return None


def _entry_point_names(kind: str) -> frozenset[str]:
    return frozenset(ep.name for ep in entry_points(group=_GROUPS[kind]))


def _canonical_loaded(kind: str) -> frozenset[str]:
    # Canonical names only: an alias has ``name != fmt.name``, so it is excluded.
    return frozenset(
        name for name, fmt in _REGISTRY.items() if fmt.kind == kind and fmt.name == name
    )


def data_formats() -> frozenset[str]:
    """The names of every available data format (in-memory object).

    Aliases resolve through :func:`get` but are not listed here.
    """
    return _entry_point_names("data") | _canonical_loaded("data")


def file_formats() -> frozenset[str]:
    """The names of every available file format (on-disk file)."""
    return _entry_point_names("file") | _canonical_loaded("file")


def get(name: str) -> Format:
    """The format registered under ``name`` or one of its aliases, loading it on first use.

    Raises ``KeyError`` for an unknown name, listing the available formats.
    """
    if name in _REGISTRY:
        return _REGISTRY[name]
    for group in _GROUPS.values():
        for ep in entry_points(group=group):
            if ep.name == name:
                ep.load()  # importing the module auto-registers the format
                if name in _REGISTRY:
                    return _REGISTRY[name]
    available = sorted(data_formats() | file_formats())
    raise KeyError(f"unknown format {name!r}; available formats: {available}.")


def from_object(name: str, obj: Any, dest: Any, *args: Any, **kwargs: Any) -> ColStoreReader:
    """Import an in-memory ``obj`` from data format ``name`` into ``dest`` and open it.

    Dispatches to the named :class:`DataFormat`'s :meth:`~DataFormat.from_object`.
    Raises ``TypeError`` if ``name`` is a file format (read a file with
    :func:`colstore.ingest`) and ``NotImplementedError`` if the format does not
    support import.
    """
    fmt = get(name)
    if not isinstance(fmt, DataFormat):
        raise TypeError(f"{name!r} is a {fmt.kind} format; import a file with colstore.ingest().")
    return fmt.from_object(obj, dest, *args, **kwargs)


def _load_file_formats() -> None:
    """Import every declared file format so its extensions become known.

    Loading a format's class (via its entry point) registers it through
    ``__init_subclass__`` but does not import its backend -- the backend loads
    only when a conversion runs -- so this is cheap enough to call per dispatch.
    """
    for ep in entry_points(group=_GROUPS["file"]):
        if ep.name not in _REGISTRY:
            ep.load()


def file_format_for_extension(extension: str) -> FileFormat:
    """The registered file format claiming ``extension`` (e.g. ``".npz"``).

    The lookup is case-insensitive on the extension (including its leading dot).
    Raises ``KeyError`` -- listing the known extensions -- if none claims it.
    """
    ext = extension.lower()
    _load_file_formats()
    for fmt in _REGISTRY.values():
        if isinstance(fmt, FileFormat) and ext in {e.lower() for e in fmt.extensions}:
            return fmt
    known = sorted(
        e.lower() for f in _REGISTRY.values() if isinstance(f, FileFormat) for e in f.extensions
    )
    raise KeyError(f"no file format handles extension {extension!r}; known extensions: {known}.")


def file_format_for_path(path: Any, format: str | None = None) -> FileFormat:
    """Resolve the file format for a path: ``format`` by name, else by extension."""
    fmt = get(format) if format is not None else file_format_for_extension(_extension(path))
    if not isinstance(fmt, FileFormat):
        raise TypeError(
            f"{fmt.name!r} is a {fmt.kind} format, not a file format; "
            f"exchange an in-memory object with .to({fmt.name!r}) / "
            f"colstore.interop.from_object({fmt.name!r}, ...)."
        )
    return fmt


def _extension(path: str | os.PathLike[str]) -> str:
    """The file extension of ``path`` (with its dot), or ``""``.

    ``os.fsdecode`` normalizes a ``bytes`` path to ``str`` so it dispatches like
    any other path rather than failing a ``bytes``-vs-``str`` extension compare.
    """
    return os.path.splitext(os.fsdecode(path))[1]


class InteropMixin:
    """The data-format export surface shared by readers, datasets, and views.

    Mixed into the reader and view base classes so ``to()``, ``arrow()``, and the
    Arrow C stream interface are defined once. Each concrete class supplies only
    :meth:`_interop_target`, the small seam describing its column/row selection.
    """

    def _interop_target(self) -> tuple[_ReaderBase, list[str], Any, bool]:
        """``(store, column_names, resolved_row_indexer, single_column)`` for export."""
        raise NotImplementedError

    def _interop_selection(self) -> Selection:
        return Selection(*self._interop_target())

    def to(self, name: str) -> Any:
        """Export this selection to in-memory data format ``name`` (e.g. ``ds.to("arrow")``).

        Dispatches to the named :class:`DataFormat`. List the available data formats
        with :func:`colstore.interop.data_formats`. Raises ``TypeError`` for a file
        format, which is written with ``saveas`` instead.
        """
        fmt = get(name)
        if not isinstance(fmt, DataFormat):
            raise TypeError(f"{name!r} is a {fmt.kind} format; write a file with .saveas(path).")
        return fmt.to_object(self._interop_selection())

    def saveas(self, dest: Any, *, format: str | None = None, **kwargs: Any) -> Any:
        """Write this selection to a file, choosing the format by ``dest``'s extension.

        ``ds.saveas("out.npz")`` writes the whole store; ``ds[rows, cols].saveas(...)``
        writes just the selection. Pass ``format=`` to override the extension (e.g.
        ``format="npz"``). An existing file at ``dest`` is overwritten. List the
        available file formats with :func:`colstore.interop.file_formats`. Raises
        ``TypeError`` for an in-memory data format (exported with :meth:`to` instead)
        and ``ValueError`` for a selection with no columns.
        """
        selection = self._interop_selection()
        if not selection.columns:
            raise ValueError("cannot write a file from a selection with no columns.")
        return file_format_for_path(dest, format).to_file(selection, dest, **kwargs)

    def to_npz(self, dest: Any, **kwargs: Any) -> Any:
        """Write this selection to a NumPy ``.npz`` file (``saveas(dest, format="npz")``)."""
        return self.saveas(dest, format="npz", **kwargs)

    def to_parquet(self, dest: Any, **kwargs: Any) -> Any:
        """Write this selection to a Parquet file (``saveas(dest, format="parquet")``)."""
        return self.saveas(dest, format="parquet", **kwargs)

    def to_feather(self, dest: Any, **kwargs: Any) -> Any:
        """Write this selection to a Feather file (``saveas(dest, format="feather")``)."""
        return self.saveas(dest, format="feather", **kwargs)

    def to_json(self, dest: Any, **kwargs: Any) -> Any:
        """Write this selection to a JSON file (``saveas(dest, format="json")``)."""
        return self.saveas(dest, format="json", **kwargs)

    def to_hdf(self, dest: Any, **kwargs: Any) -> Any:
        """Write this selection to an HDF5 file (``saveas(dest, format="hdf5")``)."""
        return self.saveas(dest, format="hdf5", **kwargs)

    def arrow(self) -> Any:
        """Export this selection to Apache Arrow -- shorthand for ``to("arrow")``.

        A single column yields a ``pyarrow.Array`` (or a ``ChunkedArray`` over many
        records/files); several columns yield a ``pyarrow.Table``. The whole column
        on a native store is zero-copy. Requires ``pyarrow``.
        """
        return self.to("arrow")

    def __array__(
        self, dtype: np.dtype[Any] | None = None, copy: bool | None = None
    ) -> NDArray[Any]:
        """NumPy array interface: ``np.asarray(ds)`` / ``np.array(ds["x"])`` materialize here.

        A single-column selection yields a 1-D array of that column; several
        columns -- or a whole reader or dataset -- yield a structured record array
        with one field per column (``result[name]`` is the column). ``dtype`` casts
        the result. The default returns an owning array; ``copy=False`` returns a
        read-only zero-copy view of a single native column and raises rather than
        copy when no view is possible (a record array, a column needing a gather, or
        a dtype cast).
        """
        store, columns, row_indexer, single = self._interop_target()
        if single:
            result = (
                store._view_one(columns[0], row_indexer)
                if copy is False
                else store._gather_one(columns[0], row_indexer)
            )
        elif copy is False:
            raise ValueError(
                "a colstore record array repacks its columns and cannot be created "
                "without copying; use np.asarray(...) or copy=True."
            )
        else:
            result = store._build_recarray(row_indexer, columns)
        if dtype is None or result.dtype == dtype:
            return result
        # A dtype change forces a copy, which copy=False forbids (numpy's no-copy
        # contract raises here rather than silently allocating).
        if copy is False:
            raise ValueError(
                "casting to a different dtype requires a copy; use np.asarray(...) or copy=True."
            )
        return result.astype(dtype)

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        """Arrow C stream interface: any Arrow consumer can ingest this selection.

        For example ``pyarrow.table(ds)`` or ``polars.from_arrow(ds)``; zero-copy
        where :meth:`arrow` is.
        """
        from .arrow import to_c_stream

        return to_c_stream(self.arrow(), requested_schema)
