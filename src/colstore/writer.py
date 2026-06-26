"""Streaming writer for record-based colstore files.

A :class:`ColStoreWriter` appends one record per :meth:`write` call.
Record bytes are appended immediately, but the counters block
(``n_records`` and ``committed_rows``) -- which is what makes records
visible to readers -- is rewritten in place only on :meth:`close`,
atomically committing the session. If the process dies before close, the
appended-but-uncommitted bytes are orphaned past the committed end and
truncated on the next update-mode open; readers always see exactly the
records committed by the last successful close.

Open modes
----------

* ``"create"``  -- new file, fail if it already exists.
* ``"recreate"`` -- new file, truncate if it exists.
* ``"update"``  -- open an existing file for append. The schema is loaded
  from the existing manifest; every :meth:`write` must match it exactly.
  Orphan bytes from a crashed prior writer are truncated on open.

Module-level entry points :func:`colstore.create`, :func:`colstore.recreate`
and :func:`colstore.update` are the recommended way to construct a writer.

Lifecycle
---------

The writer holds an advisory file lock for the session (see
:mod:`colstore._lock`); concurrent writers on the same path are rejected
at :func:`__init__`, and the lock is released on :meth:`close`. Closing
twice is a no-op. Forgetting to close emits a :class:`ResourceWarning`
and runs a best-effort commit from ``__del__``; still call :meth:`close`
(or use ``with``) explicitly, because GC timing is not guaranteed.
"""

from __future__ import annotations

import contextlib
import os
import warnings
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from . import _footer, _lock, _numa
from . import format as fmt

# ---- vectored record emission ---------------------------------------------

# A whole record (32-byte header + column bodies + alignment padding) is
# written with a single writev() per record where the platform provides it,
# instead of one buffered write per piece -- each numpy tofile() also forces
# a flush of the buffered layer, so the sequential path costs one syscall
# plus a flush per column. For many-small-record streams that machinery, not
# the data movement, dominates the write cost. Output bytes are identical on
# both paths.
_HAS_WRITEV = hasattr(os, "writev")

# Upper bound on iovec segments per writev() call; longer records are split
# into successive calls. sysconf is the authoritative source where exposed;
# 1024 is the universal POSIX floor.
_IOV_MAX: int = 1024
with contextlib.suppress(AttributeError, ValueError, OSError):
    _sysconf_iov = os.sysconf("SC_IOV_MAX")
    if _sysconf_iov > 0:
        _IOV_MAX = int(_sysconf_iov)


def _writev_full(fd: int, buffers: list[Any]) -> None:
    """writev() the buffers to ``fd`` completely, in order.

    Handles the two POSIX permissions writev reserves for itself: writing
    fewer bytes than requested (resume mid-buffer; views are cast to byte
    granularity so a write may split anywhere) and rejecting more than
    IOV_MAX segments (chunk). Zero-length buffers are legal and skipped by
    the bookkeeping naturally.
    """
    views = [memoryview(b).cast("B") for b in buffers]
    index = 0
    while index < len(views):
        chunk = views[index : index + _IOV_MAX]
        written = os.writev(fd, chunk)
        if written == 0 and any(view.nbytes for view in chunk):
            # POSIX permits a zero return; for a regular file it should never
            # happen, but looping on it would spin forever, so surface it as
            # the I/O failure it is. (A zero return for an all-empty chunk is
            # not an error -- there was nothing to write.)
            raise OSError(f"writev made no progress with {len(views) - index} buffer(s) remaining")
        while index < len(views) and written >= views[index].nbytes:
            written -= views[index].nbytes
            index += 1
        if index < len(views) and written:
            views[index] = views[index][written:]


def warn_unclosed_and_close(resource: Any, label: str, identifier: object) -> None:
    """Warn that a write resource was garbage-collected unclosed, then close best-effort.

    The shared ``__del__`` safety net: emit a :class:`ResourceWarning` naming
    ``label`` (e.g. ``"ColStoreWriter"``) and ``identifier`` (its path or directory),
    then run a suppressed ``close()``. Callers should still close explicitly.
    """
    warnings.warn(
        f"{label} for {identifier} was not closed explicitly; committing from "
        f"__del__. Prefer 'with' or an explicit .close().",
        ResourceWarning,
        stacklevel=3,
    )
    with contextlib.suppress(Exception):
        resource.close()


class ColStoreWriter:
    """Append-only writer for a colstore file. See module docstring.

    Use :func:`colstore.create`, :func:`colstore.recreate`, or
    :func:`colstore.update` rather than constructing directly. ``statistics=True``
    records per-column statistics so later filters can skip data that cannot
    match; off by default.
    """

    _VALID_MODES = frozenset({"create", "recreate", "update"})

    def __init__(
        self, path: str | os.PathLike[str], mode: str, *, statistics: bool = False
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}; expected one of {sorted(self._VALID_MODES)}.")
        self._path = Path(path)
        self._mode = mode
        self._closed = False
        self._schema: list[dict[str, Any]] | None = None
        self._n_records = 0
        self._committed_rows = 0
        # Whether to write the statistics footer on close (off by default).
        self._statistics = statistics
        # Per-record statistics, one dict {name: (min, max, prunable)} per record,
        # in record order, populated only when ``statistics`` is on.
        self._record_stats_acc: list[dict[str, _footer.ColumnStat]] = []

        # Determine open mode and existence checks.
        if mode == "create":
            if self._path.exists():
                raise FileExistsError(
                    f"{self._path} already exists; use mode='recreate' to overwrite "
                    f"or mode='update' to append."
                )
            self._file = open(self._path, "w+b")  # noqa: SIM115
            self._has_header = False
        elif mode == "recreate":
            # Truncate if exists; otherwise create.
            self._file = open(self._path, "w+b")  # noqa: SIM115
            self._has_header = False
        else:  # update
            if not self._path.exists():
                raise FileNotFoundError(
                    f"{self._path} does not exist; use mode='create' or 'recreate' "
                    f"to make a new file."
                )
            self._file = open(self._path, "r+b")  # noqa: SIM115
            self._has_header = True

        # Take the advisory lock BEFORE any destructive operations. In update
        # mode the load step below truncates orphan bytes past the last
        # committed record, so if we tried to load first and then lock, a
        # losing-the-race writer would corrupt a winning writer's in-progress
        # data. Lock first, load second: a contended open leaves the file
        # byte-for-byte unchanged. The cross-platform shim in
        # :mod:`colstore._lock` dispatches to fcntl.flock on POSIX and
        # msvcrt.locking on Windows; both raise BlockingIOError on contention.
        try:
            _lock.lock_or_raise(self._file.fileno(), self._path)
        except OSError:
            self._file.close()
            raise

        if mode == "update":
            self._load_existing_state_for_update()

    # ---- internals ----------------------------------------------------

    def _load_existing_state_for_update(self) -> None:
        """Read schema + counters from disk; seek to end of last committed record."""
        self._file.seek(0)
        # Read the header through the already-open r+b handle (at offset 0) instead
        # of re-opening the path: one fewer open syscall, the same bytes.
        manifest, data_offset = fmt.read_header_from_file(self._file)
        self._schema = manifest["columns"]
        self._n_records = int(manifest["n_records"])
        self._committed_rows = int(manifest["committed_rows"])

        # Walk records to find where the last one ends; that's where we append.
        # Even though read_record_index also validates each record, we need
        # the byte position past the last record for the seek.
        if self._n_records > 0:
            itemsizes = fmt.itemsizes_of(self._schema)
            _, record_starts_bytes, n_rows_per_record = fmt.read_record_index(
                self._path, data_offset, self._n_records, itemsizes
            )
            last_body_start = int(record_starts_bytes[-1])
            last_n_rows = int(n_rows_per_record[-1])
            end_of_committed = last_body_start + fmt.record_body_size(last_n_rows, itemsizes)
        else:
            end_of_committed = data_offset

        # Recover the existing records' statistics (the old footer sits past
        # end_of_committed and is truncated away next), so the footer rewritten on
        # close keeps them.
        if self._statistics:
            self._record_stats_acc = self._load_existing_stats(int(manifest["stats_offset"]))

        # Truncate any orphan bytes past the last committed record (left by a
        # crashed writer). After this, the file ends exactly at end_of_committed
        # and future writes append at that offset.
        self._file.truncate(end_of_committed)
        self._file.seek(end_of_committed)

    def _write_header_from_first_write(self, columns_meta: list[dict[str, Any]]) -> None:
        """First write() in a create/recreate session: write the file header.

        Counter values start at zero; they're updated either by close() or
        in-place as records get appended (we update them lazily on close).
        """
        self._schema = columns_meta
        self._file.seek(0)
        fmt.write_header(self._file, columns_meta, n_records=0, committed_rows=0)
        self._has_header = True

    def _commit_counters(self, stats_offset: int = 0) -> None:
        """Seek to the counters block, rewrite it, and fsync.

        The 64-byte counters block is small enough that a single write is
        typically atomic at the syscall level; the embedded CRC catches a
        torn write on next open if it isn't. ``stats_offset`` is the location
        of the statistics footer written just before this commit.
        """
        fmt.write_counters(self._file, self._n_records, self._committed_rows, stats_offset)
        self._file.flush()
        os.fsync(self._file.fileno())

    def _write_stats_footer(self) -> int:
        """Append the statistics footer at end-of-file; return its offset (0 if none).

        Written after the last record and before the counters commit, so a crash in
        between leaves ``stats_offset`` unchanged and the orphan footer is truncated
        on the next update-mode open. The statistics are advisory: a crash during an
        append that has overwritten the prior footer leaves the pre-existing records
        non-prunable on recovery (their data is intact; they are read in full until
        a later rewrite regenerates the footer).
        """
        if self._n_records == 0 or not self._record_stats_acc:
            return 0
        self._file.seek(0, os.SEEK_END)
        offset = self._file.tell()
        self._file.write(_footer.serialize_stats(self._schema or [], self._record_stats_acc))
        return offset

    def _load_existing_stats(self, stats_offset: int) -> list[dict[str, _footer.ColumnStat]]:
        """Recover the existing records' per-record stats for an update-mode append.

        Reads the prior footer (past the committed end, before it is truncated
        away) so a rewritten footer keeps the old records' min/max. A file with no
        usable footer yields non-prunable placeholders -- those records are never
        skipped at read time.
        """
        schema = self._schema or []
        parsed = None
        if stats_offset:
            self._file.seek(stats_offset)
            parsed = _footer.parse_stats(self._file.read())
        usable = parsed is not None and all(
            name in parsed and len(parsed[name]["prunable"]) >= self._n_records
            for name in (col["name"] for col in schema)
        )
        if usable and parsed is not None:
            return [
                {
                    col["name"]: (
                        parsed[col["name"]]["min"][index],
                        parsed[col["name"]]["max"][index],
                        bool(parsed[col["name"]]["prunable"][index]),
                    )
                    for col in schema
                }
                for index in range(self._n_records)
            ]
        placeholder = {
            col["name"]: _footer.column_stat(
                np.dtype(col["dtype"]), np.zeros(0, dtype=np.dtype(col["dtype"]))
            )
            for col in schema
        }
        return [dict(placeholder) for _ in range(self._n_records)]

    # ---- public API ---------------------------------------------------

    @property
    def path(self) -> Path:
        """Filesystem path the writer is appending to."""
        return self._path

    @property
    def mode(self) -> str:
        """The mode this writer was opened in (``"create"``, ``"recreate"``, ``"update"``)."""
        return self._mode

    @property
    def n_records(self) -> int:
        """Number of records written (including any pre-existing in update mode)."""
        return self._n_records

    @property
    def committed_rows(self) -> int:
        """Total row count across all records written so far."""
        return self._committed_rows

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has run."""
        return self._closed

    def __len__(self) -> int:
        return self._committed_rows

    def write(
        self,
        columns: dict[str, np.ndarray[Any, np.dtype[Any]]],
    ) -> None:
        """Append one record. Schema is locked on the first non-empty call.

        Parameters
        ----------
        columns : dict[str, numpy.ndarray]
            Column-major data. Names must match (and dtypes match) the
            schema captured on the first :meth:`write` (or loaded from the
            existing file in update mode). All columns must share the same
            length; that length is the new record's row count.

        Empty dicts are a no-op (no record written, schema not locked yet
        in create/recreate mode). To write a zero-row record, pass
        ``{name: empty_array_of_dtype, ...}`` explicitly.

        Raises
        ------
        ValueError
            On empty schema (first call after empty dicts), ragged
            columns, or schema mismatch on a non-first call.
        TypeError
            On unsupported column dtypes.
        """
        if self._closed:
            raise ValueError("ColStoreWriter is closed.")
        if not columns:
            return  # no-op; don't lock schema, don't write record header

        _, n_rows, le_columns, columns_meta = fmt.normalize_columns(
            columns, expected_schema=self._schema
        )

        # If this is the first non-empty write in create/recreate mode, lay
        # down the file header now that we know the schema.
        if not self._has_header:
            self._write_header_from_first_write(columns_meta)

        # Wrap the body writes in MPOL_INTERLEAVE on multi-node Linux so
        # the kernel distributes page-cache pages across NUMA nodes as
        # they're allocated by write(). On the default
        # config.set_numa_policy("auto") this spreads the file's pages
        # across nodes at write time, so every subsequent reader (this
        # process or any other) sees the distributed layout without any
        # reader-side migration -- which mbind on a MAP_SHARED read
        # mapping cannot do. Spreading balances controller load for
        # readers whose threads span nodes; for node-confined reads,
        # write under "local" so the pages stay together (see
        # config.set_numa_policy). No-op on single-node hosts, non-Linux,
        # "local" policy, or when the syscall fails.
        with _numa.writer_policy_scope():
            # Append a record at the current file position (end-of-file
            # in update mode after the constructor seek, or right past
            # the header padding in create/recreate after the header
            # write).
            self._emit_record(self._n_records, n_rows, le_columns, columns_meta)

        self._n_records += 1
        self._committed_rows += n_rows
        # Capture this record's per-column min/max for the statistics footer.
        if self._statistics:
            self._record_stats_acc.append(
                {
                    meta["name"]: _footer.column_stat(
                        np.dtype(meta["dtype"]), le_columns[meta["name"]]
                    )
                    for meta in columns_meta
                }
            )

    def _emit_record(
        self,
        record_index: int,
        n_rows: int,
        le_columns: dict[str, np.ndarray[Any, np.dtype[Any]]],
        columns_meta: list[dict[str, Any]],
    ) -> None:
        """Write one record: 32-byte header + column bodies + padding.

        Vectored on POSIX: the record is assembled as an iovec (header
        bytes, one entry per column buffer, padding) and emitted with a
        single ``writev`` per record, bypassing the buffered layer -- which
        is flushed first so the raw fd sits at the logical append position.
        All other file operations seek absolutely, so the buffered file's
        stale position after the raw write is never observed. Falls back to
        the per-piece buffered path where ``os.writev`` is unavailable;
        both paths produce identical bytes.
        """
        header = fmt.record_header_bytes(record_index, n_rows)
        arrays = [le_columns[m["name"]] for m in (self._schema or columns_meta)]
        body_bytes = sum(array.nbytes for array in arrays)
        pad = fmt.record_body_padding(body_bytes)
        if not _HAS_WRITEV:
            self._emit_record_sequential(header, arrays, pad)
            return
        buffers: list[Any] = [header]
        # normalize_columns lets contiguous arrays through untouched but can
        # return strided views; writev needs real buffers, so copy only the
        # strided case (the same data movement tofile() would do internally).
        buffers.extend(np.ascontiguousarray(array) for array in arrays)
        if pad:
            buffers.append(b"\x00" * pad)
        self._file.flush()
        _writev_full(self._file.fileno(), buffers)

    def _emit_record_sequential(
        self,
        header: bytes,
        arrays: list[np.ndarray[Any, np.dtype[Any]]],
        pad: int,
    ) -> None:
        """Buffered per-piece fallback for platforms without ``os.writev``."""
        self._file.write(header)
        for array in arrays:
            array.tofile(self._file)
        if pad:
            self._file.write(b"\x00" * pad)

    def close(self) -> None:
        """Commit counters, fsync, release the lock, close the file.

        Idempotent: calling close() on an already-closed writer is a no-op.

        In create/recreate mode, if nothing was ever written, the (zero-
        byte) file is removed: there's no manifest to commit, and a
        zero-byte ``.cstore`` would fail every reader. In update mode,
        nothing-written means no counter change is needed.
        """
        if self._closed:
            return
        try:
            if self._has_header:
                stats_offset = self._write_stats_footer()
                self._commit_counters(stats_offset)
            else:
                # create/recreate, never wrote: drop the empty file.
                pass
        finally:
            _lock.unlock(self._file.fileno())
            self._file.close()
            if not self._has_header and self._mode in ("create", "recreate"):
                # Remove the zero-byte file we created.
                with contextlib.suppress(FileNotFoundError):
                    self._path.unlink()
            self._closed = True

    def __enter__(self) -> ColStoreWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort commit on GC. Users should call close() explicitly;
        # this is the safety net for forgotten close() calls.
        if not self._closed:
            warn_unclosed_and_close(self, "ColStoreWriter", self._path)

    def __repr__(self) -> str:
        return (
            f"ColStoreWriter(path={self._path.name!r}, mode={self._mode!r}, "
            f"n_records={self._n_records}, committed_rows={self._committed_rows}"
            f"{', closed' if self._closed else ''})"
        )
