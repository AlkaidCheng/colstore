"""Append-shard datasets: a directory of ``.cstore`` shards you add to in place.

A managed dataset is a directory whose ``.cstore`` files are its shards, read in
numeric filename order. :func:`append` writes one new shard; :func:`appender`
streams batches and rolls a new shard at a size budget. The directory listing is
the membership -- there is no separate index file to keep in sync. Each shard is
an ordinary single-file ``.cstore`` and is immutable once written; a shard is
committed atomically (written to a temporary name, then renamed into place), so a
crash leaves an ignored temporary file rather than a half-written shard. One
writer at a time holds an advisory lock on the directory; a second is rejected.

``colstore.open(directory)`` reads the dataset across its shards (see
:class:`~colstore.dataset.ColStoreDataset`); the shards can equally be listed by
hand (``colstore.open([...])``) or opened individually.
"""

from __future__ import annotations

import contextlib
import glob
import os
import re
import tempfile
import warnings
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from . import _lock
from . import format as fmt
from ._paths import _natural_sort_key
from ._sizes import resolve_batch_rows

#: Default shard filename template; ``{index}`` is the next free shard number.
DEFAULT_SHARD_NAME = "shard_{index:05d}.cstore"
_LOCK_NAME = ".colstore.lock"
_INDEX_FIELD = re.compile(r"\{index(?::[^}]*)?\}")

PathLike = str | os.PathLike[str]
Columns = dict[str, np.ndarray[Any, np.dtype[Any]]]


def _shard_pattern(template: str) -> re.Pattern[str] | None:
    """A regex matching ``template`` with its ``{index}`` field as a digit capture.

    Returns ``None`` when the template has no ``{index}`` field (a literal name).
    """
    field = _INDEX_FIELD.search(template)
    if field is None:
        return None
    before = re.escape(template[: field.start()])
    after = re.escape(template[field.end() :])
    return re.compile(f"^{before}(\\d+){after}$")


def _shard_name(template: str, index: int) -> str:
    """Resolve a shard filename for ``index``; a literal template ignores it."""
    return template if _shard_pattern(template) is None else template.format(index=index)


def _next_index(directory: Path, template: str) -> int:
    """The next free ``{index}`` for ``template`` in ``directory`` (gap-tolerant).

    Scans existing filenames rather than counting, so a deleted shard does not
    cause a name collision. Must be called while holding the directory lock.
    """
    pattern = _shard_pattern(template)
    if pattern is None:
        return 0
    indices = [
        int(match.group(1))
        for entry in os.scandir(directory)
        if (match := pattern.match(entry.name))
    ]
    return max(indices) + 1 if indices else 0


def _list_shards(directory: PathLike) -> list[str]:
    """The directory's shard files (``*.cstore``), in numeric filename order.

    The ``.colstore.lock`` sentinel and any ``.<name>.tmp`` orphan from a crashed
    append are excluded -- they do not end in ``.cstore`` / are dotfiles.
    """
    matches = glob.glob(os.path.join(os.fspath(directory), "*.cstore"))
    return sorted(matches, key=lambda path: _natural_sort_key(os.path.basename(path)))


def _coerce_append_data(data: Any) -> Columns:
    """Columns from append data: anything :func:`colstore.store` takes, or a reader."""
    from . import api
    from ._base import _ReaderBase

    if isinstance(data, _ReaderBase):
        return {name: data[name].array() for name in data.columns}
    return api._coerce_to_columns(data)


def _existing_schema(directory: Path) -> list[dict[str, Any]] | None:
    """The schema of the directory's first shard, or ``None`` if it has none."""
    shards = _list_shards(directory)
    if not shards:
        return None
    manifest, _ = fmt.read_header(shards[0])
    return list(manifest["columns"])


def _acquire_directory_lock(directory: Path) -> int:
    """Take the single-writer lock for ``directory``; raise if another holds it."""
    os.makedirs(directory, exist_ok=True)
    fd = os.open(directory / _LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        _lock.lock_exclusive_nonblocking(fd)
    except BlockingIOError as exc:
        os.close(fd)
        raise OSError(f"Another writer holds the lock on {directory}; close it first.") from exc
    return fd


def _release_directory_lock(fd: int) -> None:
    with contextlib.suppress(OSError):
        _lock.unlock(fd)
    with contextlib.suppress(OSError):
        os.close(fd)


def _write_shard_atomic(
    directory: Path,
    columns: Columns,
    name: str,
    *,
    statistics: bool,
    batch_size: int | str | None,
    show_progress: bool,
) -> Path:
    """Write ``columns`` to ``directory/name`` atomically; return the shard path.

    The shard is written to a temporary dotfile, fsynced, then renamed into place,
    so a reader never sees a partial shard and a crash leaves an ignored temp file.
    """
    final = directory / name
    if final.exists():
        raise FileExistsError(f"shard {final} already exists; shards are immutable.")
    fd, tmp = tempfile.mkstemp(dir=os.fspath(directory), prefix=f".{name}.", suffix=".tmp")
    os.close(fd)
    try:
        fmt.write_dataset(
            columns, tmp, batch_size=batch_size, show_progress=show_progress, statistics=statistics
        )
        sync_fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(sync_fd)
        finally:
            os.close(sync_fd)
        os.replace(tmp, final)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return final


def append(
    directory: PathLike,
    data: Any,
    *,
    name: str = DEFAULT_SHARD_NAME,
    statistics: bool = False,
    batch_size: int | str | None = "auto",
    show_progress: bool = False,
) -> Path:
    """Append ``data`` to the dataset at ``directory`` as one new shard.

    The directory is created if needed and its ``.cstore`` shards form the dataset
    (read with ``colstore.open(directory)``). ``data`` is anything
    :func:`colstore.store` accepts -- a ``{name: array}`` dict, a structured array,
    or a DataFrame -- or an open reader / dataset. It must match the existing
    shards' schema. The shard is named from ``name`` (a template whose ``{index}``
    is the next free shard number, or a literal filename) and committed atomically.
    Returns the new shard's path. ``statistics=True`` records per-record statistics
    in the shard (see :func:`colstore.store`).

    Raises :class:`OSError` if another writer holds the directory lock, and
    :class:`ValueError` if ``data`` does not match the existing schema.
    """
    directory = Path(directory)
    columns = _coerce_append_data(data)
    fd = _acquire_directory_lock(directory)
    try:
        fmt.normalize_columns(columns, expected_schema=_existing_schema(directory))
        index = _next_index(directory, name)
        return _write_shard_atomic(
            directory,
            columns,
            _shard_name(name, index),
            statistics=statistics,
            batch_size=batch_size,
            show_progress=show_progress,
        )
    finally:
        _release_directory_lock(fd)


class Appender:
    """Streaming append: roll new shards into a dataset directory from a batch stream.

    Construct via :func:`colstore.appender`. Each :meth:`write` accumulates a batch;
    a new shard is committed whenever the buffer reaches ``shard_size`` (rows, a byte
    string like ``"512 MiB"``, or ``None`` to roll only on an explicit
    :meth:`flush`). :meth:`close` flushes any remainder and releases the directory
    lock. Use as a context manager. Like :class:`~colstore.ColStoreWriter`, a
    forgotten close is committed from ``__del__`` with a :class:`ResourceWarning`.
    """

    def __init__(
        self,
        directory: PathLike,
        *,
        name: str = DEFAULT_SHARD_NAME,
        shard_size: int | str | None = None,
        statistics: bool = False,
    ) -> None:
        # Stays True until construction fully succeeds, so __del__ is a no-op if
        # __init__ raises (e.g. lock contention) on a partially-built object.
        self._closed = True
        self._directory = Path(directory)
        self._name = name
        self._is_literal = _shard_pattern(name) is None
        if self._is_literal and shard_size is not None:
            raise ValueError(
                f"a literal shard name {name!r} writes exactly one shard and cannot roll at a "
                "shard_size; use a name template containing '{index}', or shard_size=None."
            )
        self._shard_size = shard_size
        self._statistics = statistics
        self._buffer: list[Columns] = []
        self._buffered_rows = 0
        self._n_shards = 0
        # A row budget is known up front; a byte budget needs the per-row size, so
        # it is resolved from the schema on the first write.
        self._rows_per_shard: int | None = (
            max(1, shard_size)
            if isinstance(shard_size, int) and not isinstance(shard_size, bool)
            else None
        )
        self._budget_pending = isinstance(shard_size, str)
        self._lock_fd = _acquire_directory_lock(self._directory)
        # Reading the existing schema can raise on a damaged first shard; release
        # the lock we just took rather than leaking it for the process lifetime
        # (__del__ stays a no-op while _closed is True, so it cannot release it).
        try:
            self._schema: list[dict[str, Any]] | None = _existing_schema(self._directory)
        except BaseException:
            _release_directory_lock(self._lock_fd)
            raise
        self._closed = False

    @property
    def directory(self) -> Path:
        """The dataset directory being appended to."""
        return self._directory

    @property
    def n_shards(self) -> int:
        """Number of shards committed by this appender so far."""
        return self._n_shards

    @property
    def pending_rows(self) -> int:
        """Rows buffered but not yet committed to a shard."""
        return self._buffered_rows

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has run."""
        return self._closed

    def write(self, data: Any) -> None:
        """Buffer one batch; commit a shard if ``shard_size`` is reached."""
        if self._closed:
            raise ValueError("Appender is closed.")
        columns = _coerce_append_data(data)
        _, n_rows, _le, columns_meta = fmt.normalize_columns(columns, expected_schema=self._schema)
        if self._schema is None:
            self._schema = columns_meta
        if self._budget_pending:
            bytes_per_row = sum(np.dtype(meta["dtype"]).itemsize for meta in columns_meta)
            self._rows_per_shard = resolve_batch_rows(
                self._shard_size, bytes_per_row=bytes_per_row or 1
            )
            self._budget_pending = False
        self._buffer.append(columns)
        self._buffered_rows += n_rows
        if self._rows_per_shard is not None and self._buffered_rows >= self._rows_per_shard:
            self.flush()

    def flush(self) -> Path | None:
        """Commit the buffered batches as one shard now; ``None`` if nothing buffered."""
        if self._closed:
            raise ValueError("Appender is closed.")
        if not self._buffer:
            return None
        if self._is_literal and self._n_shards > 0:
            raise ValueError(
                f"shard name {self._name!r} is a literal; it cannot roll more than one "
                "shard. Use a template containing '{index}'."
            )
        columns = self._buffer[0] if len(self._buffer) == 1 else self._concat_buffer()
        index = _next_index(self._directory, self._name)
        path = _write_shard_atomic(
            self._directory,
            columns,
            _shard_name(self._name, index),
            statistics=self._statistics,
            batch_size="auto",
            show_progress=False,
        )
        self._buffer = []
        self._buffered_rows = 0
        self._n_shards += 1
        return path

    def close(self) -> None:
        """Flush any remainder, release the directory lock; idempotent."""
        if self._closed:
            return
        try:
            self.flush()
        finally:
            _release_directory_lock(self._lock_fd)
            self._closed = True

    def _concat_buffer(self) -> Columns:
        names = list(self._buffer[0])
        return {name: np.concatenate([entry[name] for entry in self._buffer]) for name in names}

    def __enter__(self) -> Appender:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        if not self._closed:
            warnings.warn(
                f"Appender for {self._directory} was not closed explicitly; committing "
                f"from __del__. Prefer 'with' or an explicit .close().",
                ResourceWarning,
                stacklevel=2,
            )
            with contextlib.suppress(Exception):
                self.close()

    def __repr__(self) -> str:
        return (
            f"Appender(directory={self._directory.name!r}, n_shards={self._n_shards}, "
            f"pending_rows={self._buffered_rows}{', closed' if self._closed else ''})"
        )


def appender(
    directory: PathLike,
    *,
    name: str = DEFAULT_SHARD_NAME,
    shard_size: int | str | None = None,
    statistics: bool = False,
) -> Appender:
    """Open a streaming :class:`Appender` for the dataset at ``directory``.

    Holds the single-writer lock for the session. ``shard_size`` rolls a new shard
    when the buffer reaches that many rows, that byte budget (a string like
    ``"512 MiB"``), or -- when ``None`` -- only on an explicit :meth:`Appender.flush`
    or on close. ``name`` and ``statistics`` are as for :func:`append`.
    """
    return Appender(directory, name=name, shard_size=shard_size, statistics=statistics)
