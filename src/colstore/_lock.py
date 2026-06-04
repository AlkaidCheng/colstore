"""Cross-platform advisory file locking.

The colstore writer and compactor use a per-file lock to detect concurrent
modification (a second writer on the same path, or a compact running while
a writer is appending). POSIX gives us :func:`fcntl.flock` for the
whole-file advisory case; Windows has no direct equivalent and uses
:func:`msvcrt.locking` for byte-range locks instead. This module hides
the difference.

The Windows path locks one byte at offset 0 of the file. msvcrt's lock
is *mandatory* (the kernel enforces it for all I/O, not just cooperating
processes), but byte 0 is part of the immutable 8-byte magic constant
``b"CSTORE\\x00\\x01"`` -- no non-cooperating code path needs to touch
it once the file is open, so mandatory-vs-advisory doesn't matter in
practice. The lock is automatically released when the file descriptor
closes, so a forgotten :func:`unlock` is recovered the moment the
process exits.

Both functions take a raw file descriptor (the result of
``os.open()`` or ``file_obj.fileno()``) rather than a file object, so
the caller is free to choose how it tracks the file. The Windows
implementation save/restores the fd's offset around the lock call so
the caller's logical file position is undisturbed.
"""

from __future__ import annotations

import contextlib
import sys

if sys.platform == "win32":
    import msvcrt
    import os

    def lock_exclusive_nonblocking(fd: int) -> None:
        """Acquire an exclusive non-blocking lock; raise BlockingIOError if held."""
        saved_position = os.lseek(fd, 0, os.SEEK_CUR)
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as e:
            # Windows raises plain OSError on contention; normalize to
            # BlockingIOError so callers can catch one type cross-platform
            # (which is what fcntl.flock raises on POSIX with LOCK_NB).
            raise BlockingIOError(str(e)) from e
        finally:
            os.lseek(fd, saved_position, os.SEEK_SET)

    def unlock(fd: int) -> None:
        """Release a previously-acquired lock. Idempotent: errors are suppressed."""
        saved_position = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            with contextlib.suppress(OSError):
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        finally:
            with contextlib.suppress(OSError):
                os.lseek(fd, saved_position, os.SEEK_SET)

else:
    import fcntl

    def lock_exclusive_nonblocking(fd: int) -> None:
        """Acquire an exclusive non-blocking lock; raise BlockingIOError if held."""
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def unlock(fd: int) -> None:
        """Release a previously-acquired lock. Idempotent: errors are suppressed."""
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
