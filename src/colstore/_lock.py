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
import errno
import sys
import warnings

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

    # Some network / parallel filesystems (for example certain NFS mounts) do
    # not implement flock and report it as one of these errnos. ENOTSUPP (524)
    # is a kernel-internal code with no errno-module name that leaks to
    # userspace from exactly these mounts; EOPNOTSUPP and ENOLCK are the
    # POSIX-named variants. On such a mount advisory locking is unavailable to
    # every process, so there is no lock to contend for: proceeding without one
    # is the only option and loses no guarantee that was achievable there.
    _ENOTSUPP = 524
    _LOCK_UNSUPPORTED = frozenset({errno.ENOLCK, errno.EOPNOTSUPP, _ENOTSUPP})

    def lock_exclusive_nonblocking(fd: int) -> None:
        """Acquire an exclusive non-blocking lock; raise BlockingIOError if held.

        On a filesystem that does not implement flock, advisory locking is
        unavailable to all processes, so this proceeds without a lock (emitting
        a one-time warning) rather than failing the write -- there is no lock to
        contend for. Genuine contention (another holder) still raises
        BlockingIOError, which the caller turns into an actionable error.
        """
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise  # a real holder; not the filesystem refusing to lock
        except OSError as e:
            if e.errno not in _LOCK_UNSUPPORTED:
                raise
            warnings.warn(
                "advisory file locking is not supported on this filesystem; "
                "proceeding without a writer lock (concurrent writers to the "
                "same file will not be detected)",
                RuntimeWarning,
                stacklevel=2,
            )

    def unlock(fd: int) -> None:
        """Release a previously-acquired lock. Idempotent: errors are suppressed."""
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
