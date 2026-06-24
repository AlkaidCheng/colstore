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
        register(cls(), override=override)


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


def register(fmt: Format, *, override: bool = False) -> None:
    """Register ``fmt`` under its :attr:`~Format.name` and any :attr:`~Format.aliases`.

    Usually automatic -- defining a :class:`Format` subclass registers it -- but
    callable directly for a dynamically built format. A name or alias already taken
    by a different format raises ``ValueError`` unless ``override=True``.
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
            claimed = _entry_point_class(nm)
            if claimed is not None and claimed != type(fmt).__name__:
                raise ValueError(
                    f"format name {nm!r} is claimed by the {claimed!r} entry point; "
                    f"choose another name/alias or pass override=True."
                )
    for nm in names:
        _REGISTRY[nm] = fmt


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
    raise KeyError(
        f"unknown format {name!r}; available formats: {sorted(data_formats() | file_formats())}."
    )


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

    def arrow(self) -> Any:
        """Export this selection to Apache Arrow -- shorthand for ``to("arrow")``.

        A single column yields a ``pyarrow.Array`` (or a ``ChunkedArray`` over many
        records/files); several columns yield a ``pyarrow.Table``. The whole column
        on a native store is zero-copy. Requires ``pyarrow``.
        """
        return self.to("arrow")

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        """Arrow C stream interface: any Arrow consumer can ingest this selection.

        For example ``pyarrow.table(ds)`` or ``polars.from_arrow(ds)``; zero-copy
        where :meth:`arrow` is.
        """
        from .arrow import to_c_stream

        return to_c_stream(self.arrow(), requested_schema)
