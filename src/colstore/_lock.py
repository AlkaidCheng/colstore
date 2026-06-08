"""Cross-platform advisory file locking.

The colstore writer and compactor take a per-file lock to detect
concurrent modification. POSIX provides whole-file advisory locking via
:func:`fcntl.flock`; Windows has no equivalent and uses
:func:`msvcrt.locking` byte-range locks instead. This module hides the
difference.

The Windows path locks a single byte at a sentinel offset far beyond any
practical file size (``1 << 62``). msvcrt's byte-range lock is
*mandatory*: it blocks reads and writes of the locked bytes from any
other handle, even within the same process. That gives contention
detection (a second lock attempt raises ``OSError``, normalized to
``BlockingIOError``) but means the locked byte must be one that no read
path ever touches -- locking byte 0 would make :func:`colstore.info` and
:func:`colstore.open` fail with ``PermissionError`` while a writer is
active. Locking far past EOF avoids this: the lock exists only in the
kernel's lock table, and the file is not extended (msvcrt.locking past
EOF does not allocate).

Both functions take a raw file descriptor rather than a file object, so
the caller chooses how it tracks the file. The Windows implementation
saves and restores the fd's offset around the lock call.
"""

from __future__ import annotations

import contextlib
import sys

if sys.platform == "win32":
    import msvcrt
    import os

    # Sentinel offset for the Windows byte-range lock. 4 EiB -- well beyond
    # any practical colstore file (or any filesystem's maximum file size).
    # msvcrt.locking at this offset creates a logical lock record without
    # extending the file, so it doesn't collide with any reader path.
    _LOCK_OFFSET = 1 << 62

    def lock_exclusive_nonblocking(fd: int) -> None:
        """Acquire an exclusive non-blocking lock; raise BlockingIOError if held."""
        saved_position = os.lseek(fd, 0, os.SEEK_CUR)
        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
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
            os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
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
