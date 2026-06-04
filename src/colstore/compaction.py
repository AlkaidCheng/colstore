"""File compaction: collapse a multi-record file into a single-record file.

The public entry point is :func:`colstore.compact`; this module holds the
implementation. The work splits into two halves:

* A *header re-emit* that writes the output's magic, counters block, manifest,
  padding, and record header. These are tens to hundreds of bytes total --
  cost is negligible.
* A *byte splice* of each column's payload from every input record into one
  contiguous run in the output. This is bandwidth-bound and dominates the
  runtime. We use :func:`os.sendfile` on Linux (kernel-space copy, no
  Python-side surfacing) and :func:`shutil.copyfileobj` on other platforms.
  macOS has ``os.sendfile`` but it requires a socket destination and so is
  unusable for file-to-file copies; Windows has no ``os.sendfile``. The
  platform gate is therefore ``sys.platform == "linux"``.

Memory footprint is bounded by the kernel's sendfile buffer on Linux and
by ``shutil.COPY_BUFSIZE`` (64 KB) elsewhere, independent of file size.
The output file can be many times larger than RAM.

Atomicity (in-place mode): the output is written to a sibling temp file and
moved into place with :func:`os.replace`. On any error the temp file is
removed and the source is untouched. Concurrent writers are blocked for the
duration via an advisory lock on the source (``fcntl.flock`` on POSIX,
``msvcrt.locking`` on Windows; see :mod:`colstore._lock`).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
from pathlib import Path
from typing import IO, Any

import numpy as np

from . import _lock
from . import format as fmt
from .progress import progress_bar

# Linux's os.sendfile accepts regular-file destinations. macOS's sendfile
# requires a socket destination and raises ENOTSOCK for our use case.
# Windows has no os.sendfile at all. We dispatch by platform.
_USE_SENDFILE = sys.platform == "linux" and hasattr(os, "sendfile")


def compact_file(
    src_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str] | None,
    *,
    show_progress: bool,
) -> Path:
    """Implement :func:`colstore.compact`. See that function for semantics."""
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist.")

    out = Path(out_path) if out_path is not None else None
    same_target = out is None or out.resolve() == src.resolve()

    # Take the lock BEFORE reading the header. The ordering doesn't matter
    # for correctness (the lock is on a sentinel offset that nothing else
    # touches; see :mod:`colstore._lock`), but a contended open this way
    # yields a clean "writer holds the lock" error before we attempt any
    # format work -- the error precedence is friendlier to the caller.
    #
    # Lock lifecycle: protects the byte copy only. We release and close
    # lock_fd BEFORE os.replace, because:
    #
    #   1. Windows refuses to replace a file that anyone has open
    #      (WinError 5: Access is denied). Our lock_fd is exactly that.
    #
    #   2. Once os.replace runs, the lock is meaningless anyway. POSIX
    #      rename unlinks the old inode; our lock_fd still references it,
    #      but a new writer opening `src` lands on the NEW inode and is
    #      unaffected by the old lock. The protection window structurally
    #      ends at the rename; the explicit unlock just matches.
    lock_fd = os.open(src, os.O_RDONLY)
    lock_acquired = False
    try:
        try:
            _lock.lock_exclusive_nonblocking(lock_fd)
            lock_acquired = True
        except BlockingIOError as e:
            raise OSError(f"Cannot compact {src}: a writer holds the lock. Close it first.") from e

        # Read the header through the locking fd. We could use a fresh
        # open() -- the lock is at a sentinel offset that doesn't block
        # reads of any data byte -- but reusing lock_fd saves a syscall
        # and matches the writer's pattern. closefd=False keeps lock_fd
        # alive past the wrapper's close so we can manage its lifetime
        # explicitly below.
        with os.fdopen(lock_fd, "rb", closefd=False) as lock_file:
            lock_file.seek(0)
            manifest, src_data_offset = fmt.read_header_from_file(lock_file)

        src_n_records = int(manifest["n_records"])
        src_committed_rows = int(manifest["committed_rows"])
        columns = list(manifest["columns"])

        # Fast path: source is already compact.
        #   - In-place: no work to do. Return the source path.
        #   - Out-of-place: user asked for a copy at `out`; do a kernel-level
        #     file copy (much cheaper than re-running the format work and
        #     produces a byte-identical result).
        if src_n_records <= 1:
            if same_target:
                return src
            assert out is not None
            _copy_whole_file(src, out)
            return out

        # ---- Non-trivial compaction. -----------------------------------------
        # Walk every record header to validate them and to learn each record's
        # body offset and row count. The reader's _gather_one path does the
        # same thing; we reuse the same helper.
        itemsizes = [np.dtype(col["dtype"]).itemsize for col in columns]
        _, record_body_offsets, n_rows_per_record = fmt.read_record_index(
            src, src_data_offset, src_n_records, itemsizes
        )

        target = (
            out
            if (out is not None and not same_target)
            else src.with_name(src.name + ".compacting")
        )

        try:
            _write_compacted(
                src_path=src,
                target_path=target,
                columns=columns,
                committed_rows=src_committed_rows,
                record_body_offsets=record_body_offsets,
                n_rows_per_record=n_rows_per_record,
                itemsizes=itemsizes,
                show_progress=show_progress,
            )
        except BaseException:
            # On any failure (including KeyboardInterrupt), clean up the
            # temp/output file so we don't leave partial state behind.
            # The source is untouched at this point.
            with contextlib.suppress(FileNotFoundError):
                target.unlink()
            raise

        # Release lock + close lock_fd BEFORE the rename -- see Windows
        # os.replace note in the lock-acquisition comment above.
        _lock.unlock(lock_fd)
        lock_acquired = False
        os.close(lock_fd)
        lock_fd = -1

        # All bytes written and fsynced. Atomic rename into place.
        if same_target:
            os.replace(target, src)
            return src
        return target

    finally:
        if lock_acquired:
            _lock.unlock(lock_fd)
        if lock_fd != -1:
            os.close(lock_fd)


def _write_compacted(
    *,
    src_path: Path,
    target_path: Path,
    columns: list[dict[str, Any]],
    committed_rows: int,
    record_body_offsets: np.ndarray[Any, np.dtype[Any]],
    n_rows_per_record: np.ndarray[Any, np.dtype[Any]],
    itemsizes: list[int],
    show_progress: bool,
) -> None:
    """Write a single-record compacted file at ``target_path``.

    The output layout matches what ``format.write_dataset`` produces for a
    one-shot write: file header (with the final n_records=1, committed_rows
    counters baked in -- no later rewrite needed), record header, and a
    column-major body assembled from the source records.
    """
    # Per-column prefix sums into the input file's record bodies -- byte
    # offset of column c's data within record r is
    # ``record_body_offsets[r] + col_prefix_in_record[c] * n_rows_per_record[r]``.
    # (Mirrors the reader's gather math.)
    col_prefixes = np.cumsum([0, *itemsizes[:-1]])

    with open(src_path, "rb") as src_fp, open(target_path, "wb") as dst_fp:
        # ---- File header with FINAL counters. -----------------------------
        # The writer normally writes n_records=0 here and rewrites on close;
        # we know the answer up front, so we bake it in.
        fmt.write_header(dst_fp, columns, n_records=1, committed_rows=committed_rows)
        # ---- Single record header. ----------------------------------------
        fmt.write_record_header(dst_fp, record_index=0, n_rows=committed_rows)

        # ---- Column bodies. -----------------------------------------------
        # For each output column, splice the matching byte range from every
        # input record. The output column is contiguous; the source bytes
        # come from R separate ranges in the input file.
        body_bytes = 0
        n_records = len(n_rows_per_record)
        with progress_bar(
            len(columns) * n_records,
            desc="Compacting",
            unit="copy",
            enabled=show_progress,
        ) as progress:
            for col_index, _col in enumerate(columns):
                itemsize = itemsizes[col_index]
                col_prefix = int(col_prefixes[col_index])
                for record_index in range(n_records):
                    n_rows = int(n_rows_per_record[record_index])
                    if n_rows == 0:
                        progress.update(1)
                        continue
                    src_offset = int(record_body_offsets[record_index]) + col_prefix * n_rows
                    nbytes = n_rows * itemsize
                    _copy_range(src_fp, dst_fp, src_offset, nbytes)
                    body_bytes += nbytes
                    progress.update(1)

        # ---- Record body padding. -----------------------------------------
        # record_body_size returns the body's on-disk size *including* the
        # 8-byte alignment pad; subtracting what we've actually written
        # gives the pad length without reaching into format internals.
        pad = fmt.record_body_size(committed_rows, itemsizes) - body_bytes
        if pad:
            dst_fp.write(b"\x00" * pad)

        # Durability before atomic rename: without fsync the OS may buffer
        # everything in cache and a crash before flush leaves the rename
        # pointing at a torn file.
        dst_fp.flush()
        os.fsync(dst_fp.fileno())


def _copy_range(src_fp: IO[bytes], dst_fp: IO[bytes], offset: int, count: int) -> None:
    """Copy ``count`` bytes from ``offset`` in ``src_fp`` to ``dst_fp``'s current position.

    On Linux, this is one or more ``os.sendfile`` calls -- a kernel-space
    copy that never surfaces the bytes through Python. On macOS and
    Windows, it's a :func:`shutil.copyfileobj` call wrapped in a
    :class:`_BoundedReader` so it can't read past the requested range.

    On both paths the memory footprint is independent of ``count``:
    bounded by the kernel's sendfile buffer on Linux, by
    ``shutil.COPY_BUFSIZE`` (64 KB) elsewhere.
    """
    if count == 0:
        return

    if _USE_SENDFILE:
        # sendfile may copy less than requested in one call; loop until done.
        # Flush dst's Python buffer first so we don't interleave Python-
        # buffered writes with kernel-direct writes to the same fd.
        dst_fp.flush()
        dst_fd = dst_fp.fileno()
        src_fd = src_fp.fileno()
        remaining = count
        cur_offset = offset
        while remaining > 0:
            sent = os.sendfile(dst_fd, src_fd, cur_offset, remaining)
            if sent == 0:  # pragma: no cover -- EOF before count satisfied
                raise OSError(
                    f"sendfile returned 0 with {remaining} bytes left; "
                    f"source file is shorter than expected."
                )
            cur_offset += sent
            remaining -= sent
        return

    # macOS / Windows: read+write via Python buffer.
    src_fp.seek(offset)
    shutil.copyfileobj(_BoundedReader(src_fp, count), dst_fp)


def _copy_whole_file(src: Path, dst: Path) -> None:
    """Same-filesystem-friendly whole-file copy.

    Used only for the ``out_path != src`` + ``n_records == 1`` short-circuit,
    where the bytes are already optimally arranged and we just want a copy
    at the new path. ``shutil.copyfile`` already uses ``os.sendfile`` /
    ``copy_file_range`` internally where available -- including on macOS for
    same-filesystem clones via ``fclonefileat`` -- so it's the right tool
    for whole-file copies even though we can't use it for range copies.
    """
    shutil.copyfile(src, dst)


class _BoundedReader:
    """Restrict a file-like to at most ``n`` bytes from current position.

    Used to bound :func:`shutil.copyfileobj` so it doesn't read past the
    end of the requested range. Only used on the non-Linux fallback path
    (Linux uses ``os.sendfile`` directly).
    """

    __slots__ = ("_remaining", "_src")

    def __init__(self, src: IO[bytes], n: int) -> None:
        self._src = src
        self._remaining = n

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        request = self._remaining if size < 0 else min(size, self._remaining)
        chunk = self._src.read(request)
        self._remaining -= len(chunk)
        return chunk
