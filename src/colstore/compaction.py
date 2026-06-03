"""File compaction: collapse a multi-record file into a single-record file.

The public entry point is :func:`colstore.compact`; this module holds the
implementation. The work splits into two halves:

* A *header re-emit* that writes the output's magic, counters block, manifest,
  padding, and record header. These are tens to hundreds of bytes total --
  cost is negligible.
* A *byte splice* of each column's payload from every input record into one
  contiguous run in the output. This is bandwidth-bound and dominates the
  runtime. We hand the splice to the kernel via :func:`os.sendfile` where
  available, falling back to :func:`shutil.copyfileobj` on platforms (Windows)
  that lack ``sendfile`` for file-to-file copies.

Memory footprint is bounded by the I/O buffer the kernel chooses (tens of
KB), independent of file size. The output file can be many times larger than
RAM.

Atomicity (in-place mode): the output is written to a sibling temp file and
moved into place with :func:`os.replace`. On any error the temp file is
removed and the source is untouched. Concurrent writers are blocked for the
duration via an advisory ``fcntl.flock`` on the source.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
from pathlib import Path
from typing import IO, Any

import numpy as np

from . import format as fmt
from .progress import progress_bar

# Linux/macOS expose os.sendfile for file-to-file copies. Windows does not;
# the fallback path uses shutil.copyfileobj which is portable but slower.
_HAS_SENDFILE = hasattr(os, "sendfile") and os.name == "posix"


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

    # Read and validate header. Cheap, also catches corrupt files early.
    manifest, src_data_offset = fmt.read_header(src)
    src_n_records = int(manifest["n_records"])
    src_committed_rows = int(manifest["committed_rows"])
    columns = list(manifest["columns"])

    out = Path(out_path) if out_path is not None else None
    same_target = out is None or out.resolve() == src.resolve()

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
    # body offset and row count. The reader's _gather_one path does the same
    # thing; we reuse the same helper.
    itemsizes = [np.dtype(col["dtype"]).itemsize for col in columns]
    _, record_body_offsets, n_rows_per_record = fmt.read_record_index(
        src, src_data_offset, src_n_records, itemsizes
    )

    target = (
        out if (out is not None and not same_target) else src.with_name(src.name + ".compacting")
    )

    # Take an advisory lock on the source to block concurrent writers from
    # appending while we copy. We open the lock fd separately from the
    # sendfile read fd so we can keep it through the rename without
    # conflicting with the writer's own flock conventions.
    lock_fd = os.open(src, os.O_RDONLY)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise OSError(f"Cannot compact {src}: a writer holds the lock. Close it first.") from e

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

        # All bytes written and fsynced. Atomic rename into place.
        # os.replace is atomic on POSIX for same-filesystem moves and works
        # on Windows for non-open destinations.
        if same_target:
            os.replace(target, src)
            return src
        return target

    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
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
        src_fd = src_fp.fileno()
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
                    _copy_range(src_fd, dst_fp, src_offset, nbytes)
                    body_bytes += nbytes
                    progress.update(1)

        # ---- Record body padding. -----------------------------------------
        pad = fmt.align_up(body_bytes, fmt._RECORD_BODY_ALIGNMENT) - body_bytes
        if pad:
            dst_fp.write(b"\x00" * pad)

        # Durability before atomic rename: without fsync the OS may buffer
        # everything in cache and a crash before flush leaves the rename
        # pointing at a torn file.
        dst_fp.flush()
        os.fsync(dst_fp.fileno())


def _copy_range(src_fd: int, dst_fp: IO[bytes], offset: int, count: int) -> None:
    """Copy ``count`` bytes from ``offset`` in ``src_fd`` to ``dst_fp``'s current position.

    On Linux/macOS this is a single ``sendfile`` call (or a small loop if the
    kernel returns partial counts) that copies in kernel space without
    surfacing the bytes through Python. On Windows we fall back to
    ``shutil.copyfileobj`` with a per-range bounded reader.
    """
    if count == 0:
        return
    if _HAS_SENDFILE:
        # sendfile may copy less than requested in one call; loop until done.
        # The first arg is the destination fd; the in_offset argument
        # advances as we go.
        remaining = count
        cur_offset = offset
        dst_fd = dst_fp.fileno()
        # Ensure the dst_fp's buffered position is flushed to the underlying
        # fd before we write to that fd directly.
        dst_fp.flush()
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

    # Fallback: read+write via Python buffer. Slower but portable.
    # pragma: no cover -- only runs on Windows.
    os.lseek(src_fd, offset, os.SEEK_SET)  # pragma: no cover
    with os.fdopen(os.dup(src_fd), "rb", closefd=True) as src_fp:  # pragma: no cover
        shutil.copyfileobj(_BoundedReader(src_fp, count), dst_fp)  # pragma: no cover


def _copy_whole_file(src: Path, dst: Path) -> None:
    """Same-filesystem-friendly whole-file copy.

    Used only for the ``out_path != src`` + ``n_records == 1`` short-circuit,
    where the bytes are already optimally arranged and we just want a copy
    at the new path. ``shutil.copyfile`` already uses ``os.sendfile`` /
    ``copy_file_range`` internally where available.
    """
    shutil.copyfile(src, dst)


class _BoundedReader:  # pragma: no cover -- Windows fallback only
    """Restrict a file-like to at most ``n`` bytes from current position.

    Used to bound ``shutil.copyfileobj`` so it doesn't read past the end of
    the requested range.
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
