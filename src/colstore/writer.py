"""Streaming writer for record-based colstore files.

A :class:`ColStoreWriter` appends one record per :meth:`write` call. The
counters block (``n_records`` and ``committed_rows``) is rewritten in
place on :meth:`close`, atomically committing the session's writes;
readers opening the file mid-write see only what was committed by the
last successful close.

Open modes
----------

* ``"create"``  -- new file, fail if it already exists.
* ``"recreate"`` -- new file, truncate if it exists.
* ``"update"``  -- open an existing file for append. The schema is loaded
  from the existing manifest; the first :meth:`write` (and every one
  after) must match it exactly. Orphan bytes from a crashed prior writer
  are truncated on open.

Module-level entry points :func:`colstore.create`, :func:`colstore.recreate`
and :func:`colstore.update` are the recommended way to construct a writer.

Crash safety
------------

The on-disk counters block is what makes a record visible to readers.
The writer appends raw record bytes one at a time, but does not update
the counters until :meth:`close`. If the process dies before close, the
appended-but-uncommitted bytes are orphaned past the file's "committed
end"; reopening for update truncates them. Readers always see exactly
the records that were committed on the last close, regardless of any
in-progress write that died.

Lifecycle
---------

The writer holds an advisory ``fcntl.flock`` on the file for the
duration of the session. Concurrent writers on the same path are
rejected at :func:`__init__`. The lock is released on :meth:`close`.

Closing twice is a no-op. Forgetting to close emits a
:class:`ResourceWarning` and runs a best-effort commit from
``__del__``; users should still call :meth:`close` (or use ``with``)
explicitly because GC timing is not guaranteed.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import warnings
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from . import format as fmt


class ColStoreWriter:
    """Append-only writer for a colstore file. See module docstring.

    Use :func:`colstore.create`, :func:`colstore.recreate`, or
    :func:`colstore.update` rather than constructing directly.
    """

    _VALID_MODES = frozenset({"create", "recreate", "update"})

    def __init__(self, path: str | os.PathLike[str], mode: str) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}; expected one of {sorted(self._VALID_MODES)}.")
        self._path = Path(path)
        self._mode = mode
        self._closed = False
        self._schema: list[dict[str, Any]] | None = None
        self._n_records = 0
        self._committed_rows = 0
        self._data_offset = 0  # set when header is written / read

        # Determine open mode and existence checks.
        if mode == "create":
            if self._path.exists():
                raise FileExistsError(
                    f"{self._path} already exists; use mode='recreate' to overwrite "
                    f"or mode='update' to append."
                )
            self._file = open(self._path, "w+b")  # noqa: SIM115
            self._has_header = False
            self._wrote_anything = False
        elif mode == "recreate":
            # Truncate if exists; otherwise create.
            self._file = open(self._path, "w+b")  # noqa: SIM115
            self._has_header = False
            self._wrote_anything = False
        else:  # update
            if not self._path.exists():
                raise FileNotFoundError(
                    f"{self._path} does not exist; use mode='create' or 'recreate' "
                    f"to make a new file."
                )
            self._file = open(self._path, "r+b")  # noqa: SIM115
            self._load_existing_state_for_update()
            self._has_header = True
            self._wrote_anything = True  # something is already there

        # Take an advisory lock on the file. Catches concurrent writers on the
        # same path. flock is no-op on platforms that don't support it; on
        # Linux/macOS this is a per-file-descriptor advisory lock that
        # readers also don't see (they don't take it).
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            self._file.close()
            raise OSError(f"Another writer holds the lock on {self._path}; close it first.") from e

    # ---- internals ----------------------------------------------------

    def _load_existing_state_for_update(self) -> None:
        """Read schema + counters from disk; seek to end of last committed record."""
        manifest, data_offset = fmt.read_header(self._path)
        self._schema = manifest["columns"]
        self._n_records = int(manifest["n_records"])
        self._committed_rows = int(manifest["committed_rows"])
        self._data_offset = data_offset

        # Walk records to find where the last one ends; that's where we append.
        # Even though read_record_index also validates each record, we need
        # the byte position past the last record for the seek.
        if self._n_records > 0:
            itemsizes = [np.dtype(col["dtype"]).itemsize for col in self._schema]
            _, record_starts_bytes, n_rows_per_record = fmt.read_record_index(
                self._path, data_offset, self._n_records, itemsizes
            )
            last_body_start = int(record_starts_bytes[-1])
            last_n_rows = int(n_rows_per_record[-1])
            end_of_committed = last_body_start + fmt.record_body_size(last_n_rows, itemsizes)
        else:
            end_of_committed = data_offset

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
        self._data_offset = fmt.write_header(
            self._file, columns_meta, n_records=0, committed_rows=0
        )
        self._has_header = True

    def _commit_counters(self) -> None:
        """Seek to the counters block, rewrite it, and fsync.

        The 32-byte counters block is small enough that a single write is
        typically atomic at the syscall level; the embedded CRC catches a
        torn write on next open if it isn't.
        """
        fmt.write_counters(self._file, self._n_records, self._committed_rows)
        self._file.flush()
        os.fsync(self._file.fileno())

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

        # Append a record at the current file position (which is end-of-file
        # in update mode after the constructor seek, or right past the
        # header padding in create/recreate after the header write).
        record_index = self._n_records
        fmt.write_record_header(self._file, record_index, n_rows)
        body_bytes = 0
        for col_meta in self._schema or columns_meta:
            array = le_columns[col_meta["name"]]
            array.tofile(self._file)
            body_bytes += array.nbytes
        pad = fmt.align_up(body_bytes, fmt._RECORD_BODY_ALIGNMENT) - body_bytes
        if pad:
            self._file.write(b"\x00" * pad)

        self._n_records += 1
        self._committed_rows += n_rows
        self._wrote_anything = True

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
                self._commit_counters()
            else:
                # create/recreate, never wrote: drop the empty file.
                pass
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
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
            warnings.warn(
                f"ColStoreWriter for {self._path} was not closed explicitly; "
                f"committing from __del__. Prefer 'with' or an explicit "
                f".close() call.",
                ResourceWarning,
                stacklevel=2,
            )
            with contextlib.suppress(Exception):
                self.close()

    def __repr__(self) -> str:
        return (
            f"ColStoreWriter(path={self._path.name!r}, mode={self._mode!r}, "
            f"n_records={self._n_records}, committed_rows={self._committed_rows}"
            f"{', closed' if self._closed else ''})"
        )
