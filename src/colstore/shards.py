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
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from . import _lock
from . import format as fmt
from ._coerce import coerce_to_columns
from ._paths import natural_sort_key
from ._sizes import resolve_batch_rows
from ._types import Columns, StrPath
from .writer import warn_unclosed_and_close

if TYPE_CHECKING:
    from .dataset import ColStoreDataset
    from .reader import ColStoreReader

#: An already-open colstore source to append from (borrowed) or to open from a path.
ShardSource: TypeAlias = "ColStoreReader | ColStoreDataset"

#: Default shard filename template; ``{index}`` is the next free shard number.
DEFAULT_SHARD_NAME = f"shard_{{index:05d}}{fmt.FILE_EXTENSION}"
_LOCK_NAME = ".colstore.lock"
_INDEX_FIELD = re.compile(r"\{index(?::[^}]*)?\}")


#: Target bytes per copy stream: a single-file shard copy uses
#: ``min(size // this, _PARALLEL_COPY_MAX_STREAMS)`` streams, so a file smaller
#: than twice this size copies in one stream.
_PARALLEL_COPY_MIN_CHUNK = 128 * 1024 * 1024
#: Upper bound on concurrent copy streams. The useful count is what the storage can
#: serve in parallel (a handful saturates a parallel filesystem; more only contends),
#: not the core count -- so this is a small fixed ceiling, not hardware-derived.
_PARALLEL_COPY_MAX_STREAMS = 4
#: Per-read size for the ``os.pread`` / ``os.pwrite`` copy fallback.
_COPY_BUFSIZE = 8 * 1024 * 1024
#: ``os.copy_file_range`` when the runtime exposes it (Linux), else ``None``.
_COPY_FILE_RANGE = getattr(os, "copy_file_range", None)
#: Whether a positional copy primitive exists. ``os.pread`` / ``os.pwrite`` are
#: Unix-only, so Windows (with no ``copy_file_range`` either) copies single-stream.
_CAN_PARALLEL_COPY = _COPY_FILE_RANGE is not None or hasattr(os, "pwrite")


@lru_cache
def _shard_pattern(template: str) -> re.Pattern[str] | None:
    """A regex matching ``template`` with its ``{index}`` field as a digit capture.

    Returns ``None`` when the template has no ``{index}`` field (a literal name).
    Memoized: a template is compiled once and reused across every shard roll and
    every appender rather than recompiled per call.
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


def _validate_shard_name(name: str) -> None:
    """Require a shard name or template to carry the ``.cstore`` extension.

    The dataset is read by globbing ``*.cstore`` (:func:`list_shards`), so a shard
    written under any other extension would not be seen by the reader.
    """
    if not name.endswith(fmt.FILE_EXTENSION):
        raise ValueError(
            f"shard name {name!r} must end in {fmt.FILE_EXTENSION!r}; the dataset is "
            f"read by listing {fmt.FILE_EXTENSION} files."
        )


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


def list_shards(directory: StrPath) -> list[str]:
    """The directory's shard files (``*.cstore``), in numeric filename order.

    The ``.colstore.lock`` sentinel and any ``.<name>.tmp`` orphan from a crashed
    append are excluded -- they do not end in ``.cstore`` / are dotfiles.
    """
    matches = glob.glob(os.path.join(os.fspath(directory), f"*{fmt.FILE_EXTENSION}"))
    return sorted(matches, key=lambda path: natural_sort_key(os.path.basename(path)))


def _open_source(data: Any) -> tuple[ShardSource, bool] | None:
    """The colstore source for ``data`` and whether this call owns it (must close).

    An open reader or dataset is borrowed (``False``); a path is opened and owned
    (``True``). Returns ``None`` when ``data`` is in-memory (a dict, structured
    array, or DataFrame) and must be materialized instead.
    """
    from . import api
    from .dataset import ColStoreDataset
    from .reader import ColStoreReader

    if isinstance(data, (ColStoreReader, ColStoreDataset)):
        return data, False
    if isinstance(data, (str, os.PathLike)):
        return api.open(data), True
    return None


def _materialize_source(source: ShardSource) -> Columns:
    """Read every column of ``source`` into memory as a column dict."""
    return {name: source[name].array() for name in source.columns}


def _coerce_append_data(data: Any) -> Columns:
    """Materialize append data to a column dict: a :func:`colstore.store` input,
    an open reader/dataset, or a path to read."""
    opened = _open_source(data)
    if opened is not None:
        source, owned = opened
        try:
            return _materialize_source(source)
        finally:
            if owned:
                source.close()
    return coerce_to_columns(data)


def _validate_source_schema(
    source: ShardSource, expected_schema: list[dict[str, Any]] | None
) -> None:
    """Require ``source`` to match the dataset's existing shard schema, or raise.

    Compared from headers only -- no column is read -- so a mismatch is rejected
    before the streaming/copy write begins. A first shard (``None``) defines it.
    """
    if expected_schema is None:
        return
    expected = [(col["name"], col["dtype"]) for col in expected_schema]
    actual = [(name, dtype.str) for name, dtype in source._column_dtypes.items()]
    if actual != expected:
        raise ValueError(
            f"shard schema mismatch: the source has columns {actual}, but the dataset "
            f"schema is {expected} (same names, order, and dtypes required)."
        )


def _source_files(source: ShardSource) -> list[Path]:
    """The physical ``.cstore`` files backing ``source`` (one for a single reader)."""
    return list(source.paths)


def _reject_self_append(source: ShardSource, directory: Path) -> None:
    """Refuse to append a dataset to itself; every existing row would be duplicated.

    A source whose files are shards of the locked directory aliases the dataset's
    own membership (mirrors :func:`concat`'s out-aliases-source guard).
    """
    target = directory.resolve()
    for path in _source_files(source):
        if Path(path).resolve().parent == target:
            raise ValueError(
                f"cannot append the dataset at {directory} to itself; the source "
                f"includes its own shard {os.path.basename(path)}."
            )


def _write_shard_from_source(
    source: ShardSource, final: Path, *, memory_budget: int | None
) -> None:
    """Write ``source`` to the shard ``final`` without materializing it into memory.

    A source backed by a single file is copied byte-for-byte -- the shard already
    has the optimal layout, so no column is read or rewritten. A multi-file source
    is streamed into one shard one row range at a time (bounded by ``memory_budget``;
    a no-transform source takes the merge-copy fast path). Both commit atomically.
    """
    files = _source_files(source)
    if len(files) == 1:
        _copy_shard_atomic(files[0], final)
    else:
        source.edit().write(final, memory_budget=memory_budget).close()


def _existing_schema(directory: Path) -> list[dict[str, Any]] | None:
    """The schema of the directory's first shard, or ``None`` if it has none."""
    shards = list_shards(directory)
    if not shards:
        return None
    manifest, _ = fmt.read_header(shards[0])
    return list(manifest["columns"])


def _acquire_directory_lock(directory: Path) -> int:
    """Take the single-writer lock for ``directory``; raise if another holds it."""
    os.makedirs(directory, exist_ok=True)
    fd = os.open(directory / _LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        _lock.lock_or_raise(fd, directory)
    except BaseException:
        # Any lock error -- the standardized lock-held OSError, an unexpected
        # error, or an interrupt -- must not leak the open fd; it has no
        # finalizer, so close it before the exception propagates.
        os.close(fd)
        raise
    return fd


def _release_directory_lock(fd: int) -> None:
    with contextlib.suppress(OSError):
        _lock.unlock(fd)
    with contextlib.suppress(OSError):
        os.close(fd)


def _assert_shard_absent(final: Path) -> None:
    """Refuse to overwrite an existing shard; shards are immutable once written."""
    if final.exists():
        raise FileExistsError(f"shard {final} already exists; shards are immutable.")


def _write_shard_atomic(
    directory: Path,
    columns: Columns,
    name: str,
    *,
    statistics: bool,
    batch_size: int | str | None,
    show_progress: bool,
) -> Path:
    """Write ``columns`` to ``directory/name`` atomically; return the shard path."""
    final = directory / name
    _assert_shard_absent(final)
    with fmt.atomic_publish(final) as tmp:
        fmt.write_dataset(
            columns, tmp, batch_size=batch_size, show_progress=show_progress, statistics=statistics
        )
    return final


def _copy_bytes(src_fd: int, dst_fd: int, offset: int, count: int) -> None:
    """Copy ``count`` bytes at ``offset`` from ``src_fd`` to ``dst_fd``.

    Uses ``os.copy_file_range`` (kernel-to-kernel) where the runtime exposes it,
    else ``os.pread`` / ``os.pwrite``; both release the GIL, so concurrent calls on
    disjoint ranges parallelize the I/O.
    """
    pos, remaining = offset, count
    while remaining:
        if _COPY_FILE_RANGE is not None:
            sent = _COPY_FILE_RANGE(src_fd, dst_fd, remaining, pos, pos)
        else:
            chunk = os.pread(src_fd, min(remaining, _COPY_BUFSIZE), pos)
            sent = os.pwrite(dst_fd, chunk, pos) if chunk else 0
        if sent == 0:
            break
        pos += sent
        remaining -= sent


def _copy_file(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst``, parallelizing a large file across a few I/O streams.

    The file is split into ``min(size // _PARALLEL_COPY_MIN_CHUNK,
    _PARALLEL_COPY_MAX_STREAMS)`` disjoint byte ranges copied concurrently, so a
    file smaller than twice :data:`_PARALLEL_COPY_MIN_CHUNK` copies in one stream.
    The stream count is bounded by what the
    storage serves in parallel, not the core count, so the speedup comes from I/O
    width on a parallel filesystem without depending on the host's hardware. A
    platform with no positional copy primitive (Windows) copies in one stream.
    """
    size = src.stat().st_size
    streams = min(max(1, size // _PARALLEL_COPY_MIN_CHUNK), _PARALLEL_COPY_MAX_STREAMS)
    if streams == 1 or not _CAN_PARALLEL_COPY:
        shutil.copyfile(src, dst)
        return
    os.truncate(dst, size)
    bounds = [size * i // streams for i in range(streams + 1)]

    def _copy_range(span: tuple[int, int]) -> None:
        lo, hi = span
        src_fd = os.open(src, os.O_RDONLY)
        dst_fd = os.open(dst, os.O_WRONLY)
        try:
            _copy_bytes(src_fd, dst_fd, lo, hi - lo)
        finally:
            os.close(dst_fd)
            os.close(src_fd)

    with ThreadPoolExecutor(max_workers=streams) as pool:
        list(pool.map(_copy_range, pairwise(bounds)))


def _copy_shard_atomic(src_file: Path, final: Path) -> None:
    """Copy ``src_file`` to the shard ``final`` byte-for-byte, atomically.

    A large source is split across a few concurrent I/O streams (see
    :func:`_copy_file`); the bytes never pass through Python.
    """
    with fmt.atomic_publish(final) as tmp:
        _copy_file(src_file, Path(tmp))


def _normalize_and_write_shard(
    directory: Path,
    columns: Columns,
    shard_name: str,
    *,
    expected: list[dict[str, Any]] | None,
    statistics: bool,
    batch_size: int | str | None,
    show_progress: bool,
) -> Path:
    """Normalize ``columns`` against ``expected`` and write them as a shard."""
    fmt.normalize_columns(columns, expected_schema=expected)
    return _write_shard_atomic(
        directory,
        columns,
        shard_name,
        statistics=statistics,
        batch_size=batch_size,
        show_progress=show_progress,
    )


def append(
    directory: StrPath,
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
    or a DataFrame -- or an open reader / dataset, or a path to a ``.cstore`` file
    (or directory of them). It must match the existing shards' schema. The shard is
    named from ``name`` (a template whose ``{index}`` is the next free shard number,
    or a literal filename; it must end in ``.cstore``) and committed atomically.
    Returns the new shard's path. ``statistics=True`` records per-record statistics
    in the shard (see :func:`colstore.store`).

    A colstore source (path, reader, or dataset) is written without materializing
    it into memory: its columns are streamed into the shard in bounded memory, so a
    file far larger than RAM can be appended. ``statistics=True`` uses the
    materializing path (the streaming writer records no footer).

    Raises :class:`OSError` if another writer holds the directory lock, and
    :class:`ValueError` if ``name`` lacks the ``.cstore`` extension, ``data`` does
    not match the existing schema, or ``data`` is the dataset's own directory (a
    self-append would duplicate every existing row).
    """
    _validate_shard_name(name)
    directory = Path(directory)
    fd = _acquire_directory_lock(directory)
    try:
        expected = _existing_schema(directory)
        shard_name = _shard_name(name, _next_index(directory, name))
        opened = _open_source(data)
        if opened is None:
            # In-memory data (dict / structured array / DataFrame): materialize.
            columns = _coerce_append_data(data)
            return _normalize_and_write_shard(
                directory,
                columns,
                shard_name,
                expected=expected,
                statistics=statistics,
                batch_size=batch_size,
                show_progress=show_progress,
            )
        source, owned = opened
        try:
            _reject_self_append(source, directory)
            if statistics:
                # The streaming/copy paths record no footer, so materialize here.
                columns = _materialize_source(source)
                return _normalize_and_write_shard(
                    directory,
                    columns,
                    shard_name,
                    expected=expected,
                    statistics=True,
                    batch_size=batch_size,
                    show_progress=show_progress,
                )
            final = directory / shard_name
            _assert_shard_absent(final)
            _validate_source_schema(source, expected)
            _write_shard_from_source(source, final, memory_budget=None)
            return final
        finally:
            if owned:
                source.close()
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
        directory: StrPath,
        *,
        name: str = DEFAULT_SHARD_NAME,
        shard_size: int | str | None = None,
        statistics: bool = False,
    ) -> None:
        # Stays True until construction fully succeeds, so __del__ is a no-op if
        # __init__ raises (e.g. lock contention) on a partially-built object.
        self._closed = True
        _validate_shard_name(name)
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
            bytes_per_row = sum(fmt.itemsizes_of(columns_meta))
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
            warn_unclosed_and_close(self, "Appender", self._directory)

    def __repr__(self) -> str:
        return (
            f"Appender(directory={self._directory.name!r}, n_shards={self._n_shards}, "
            f"pending_rows={self._buffered_rows}{', closed' if self._closed else ''})"
        )


def appender(
    directory: StrPath,
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
