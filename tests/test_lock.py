"""Advisory-lock behavior on filesystems that do not implement flock.

The POSIX writer lock proceeds without a lock (with a warning) when the
filesystem reports flock unsupported, but still surfaces genuine contention and
unexpected errors. These tests mock ``fcntl.flock`` to drive each branch; the
degradation path is POSIX-only, so the module is skipped on Windows.
"""

from __future__ import annotations

import errno
import sys

import numpy as np
import pytest

import colstore
from colstore import _lock, testing

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX flock-degradation path; Windows uses msvcrt"
)

if sys.platform != "win32":
    import fcntl


@pytest.mark.parametrize("code", [524, errno.EOPNOTSUPP, errno.ENOLCK])
def test_lock_proceeds_when_filesystem_lacks_flock(code, monkeypatch):
    def refuse(fd, operation):
        raise OSError(code, "flock not supported")

    monkeypatch.setattr(fcntl, "flock", refuse)
    with open("/dev/null") as handle, pytest.warns(RuntimeWarning, match="not supported"):
        _lock.lock_exclusive_nonblocking(handle.fileno())  # must not raise


def test_lock_raises_on_real_contention(monkeypatch):
    def held(fd, operation):
        raise BlockingIOError(errno.EWOULDBLOCK, "held by another writer")

    monkeypatch.setattr(fcntl, "flock", held)
    with open("/dev/null") as handle, pytest.raises(BlockingIOError):
        _lock.lock_exclusive_nonblocking(handle.fileno())


def test_lock_reraises_unexpected_oserror(monkeypatch):
    def boom(fd, operation):
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(fcntl, "flock", boom)
    with open("/dev/null") as handle, pytest.raises(OSError) as excinfo:
        _lock.lock_exclusive_nonblocking(handle.fileno())
    assert excinfo.value.errno == errno.EACCES


def test_writer_succeeds_when_filesystem_lacks_flock(tmp_path, monkeypatch):
    def refuse(fd, operation):
        raise OSError(524, "flock not supported")

    monkeypatch.setattr(fcntl, "flock", refuse)
    path = tmp_path / "nolock.cstore"
    with pytest.warns(RuntimeWarning, match="not supported"):
        testing.make_store(path, rows=100, cols=2, dtype="int64").close()
    with colstore.open(str(path)) as reader:
        columns = reader.dict()
    assert all(len(values) == 100 for values in columns.values())
    assert np.asarray(columns["c0"]).shape == (100,)
