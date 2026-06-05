"""Linux NUMA memory-policy helpers for colstore memmaps.

Sets ``MPOL_INTERLEAVE`` on the page-aligned regions covering file-backed
memmaps so that page-cache pages get distributed across NUMA nodes as
they fault in, instead of concentrating on whichever node first touched
them (the kernel's default first-touch policy).

On multi-socket / multi-NPS hardware the win is significant. Measured
on a dual-socket EPYC 7763 (8 NUMA nodes, 256 cores):

  Scenario                          Default   Interleaved   Speedup
  ds.dict()  1 GB / 50 cols         45.5 ms      25.4 ms     1.79x
  ds.frame() 1 GB / 50 cols         42.7 ms      27.6 ms     1.55x

The wall-time speedup understates the underlying improvement: process
CPU time on the same gather drops from 688 ms to 292 ms, i.e. each
worker thread spends ~2.4x less time stalled on remote-memory loads.

This module:

  * No-ops cleanly on non-Linux platforms (macOS, Windows).
  * No-ops on single-node Linux hosts (most desktops/laptops, single-
    socket parts with NPS=1, anything not in a server topology).
  * Honors cgroup ``cpuset.mems`` restrictions: the interleave mask
    covers only nodes the process is actually allowed to allocate on.
  * Falls back silently on older kernels without ``mbind(2)`` or on
    seccomp-restricted environments that block syscall 237. A warning
    is emitted exactly once so the operator can investigate, but the
    store keeps working.
  * Has zero new system-library dependencies (no ``libnuma``). The
    ``mbind`` ABI is stable Linux history back to ~2.6.7 (2004).

Implemented as a direct syscall via :mod:`ctypes`. The kernel-side
syscall number varies by architecture; this module supports the
mainline server archs (x86_64, aarch64, ppc64le, s390x) and falls
back to no-op on anything else.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import platform
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import numpy as np


# ---- Syscall constants -----------------------------------------------------


class _NumaSyscalls(NamedTuple):
    """Per-arch syscall numbers for the three NUMA syscalls we use.

    Looked up from ``arch/<arch>/.../syscall*.tbl`` in the kernel
    source. Stable across kernel versions for each architecture.
    """

    mbind: int
    set_mempolicy: int
    get_mempolicy: int


# x86_64:  mbind, set_mempolicy, get_mempolicy in that order.
# aarch64: uses asm-generic/unistd.h; mbind=235, get=236, set=237.
# ppc64le: get and set are swapped relative to x86_64 (get=260, set=261).
# s390x:   similarly out of order (get=269, set=270).
_NUMA_SYSCALLS: dict[str, _NumaSyscalls] = {
    "x86_64": _NumaSyscalls(mbind=237, set_mempolicy=238, get_mempolicy=239),
    "aarch64": _NumaSyscalls(mbind=235, set_mempolicy=237, get_mempolicy=236),
    "ppc64le": _NumaSyscalls(mbind=259, set_mempolicy=261, get_mempolicy=260),
    "s390x": _NumaSyscalls(mbind=268, set_mempolicy=270, get_mempolicy=269),
}


# Policy modes from <linux/mempolicy.h>.
_MPOL_DEFAULT = 0
_MPOL_INTERLEAVE = 3


# ---- Host capability detection (one-time, at import) -----------------------


def _parse_cpu_or_node_list(spec: str) -> list[int]:
    """Parse "0-7" / "0,2-4" style lists from /sys and /proc into ids.

    Empty strings are valid (a process restricted out of all nodes
    would have one), and yield an empty list.
    """
    ids: list[int] = []
    spec = spec.strip()
    if not spec:
        return ids
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-", maxsplit=1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return ids


def _detect_allowed_nodes() -> list[int]:
    """Return NUMA node ids this process may allocate on.

    Consults ``/proc/self/status`` for ``Mems_allowed_list`` so that
    cgroup ``cpuset.mems`` restrictions are honored -- without this,
    a containerized process could build a mask that includes nodes
    it's been denied, and ``mbind`` would return ``EPERM``. Falls
    back to ``/sys/devices/system/node/online`` if the proc file is
    unreadable.
    """
    status_path = Path("/proc/self/status")
    if status_path.exists():
        try:
            for line in status_path.read_text().splitlines():
                if line.startswith("Mems_allowed_list:"):
                    return _parse_cpu_or_node_list(line.split(":", 1)[1])
        except OSError:
            pass
    online_path = Path("/sys/devices/system/node/online")
    if online_path.exists():
        try:
            return _parse_cpu_or_node_list(online_path.read_text())
        except OSError:
            pass
    return []


_PLATFORM_IS_LINUX = platform.system() == "Linux"
_SYSCALLS: _NumaSyscalls | None = _NUMA_SYSCALLS.get(platform.machine())
_ALLOWED_NODES: list[int] = _detect_allowed_nodes() if _PLATFORM_IS_LINUX else []

# Treat the host as "NUMA-capable for our purposes" only when there's
# more than one allowed node. On single-node systems interleave is
# either a no-op or a slight pessimization (the kernel has to do
# extra bookkeeping for a policy that picks the only available node
# every time), so we skip the syscall entirely.
_AVAILABLE: bool = _PLATFORM_IS_LINUX and _SYSCALLS is not None and len(_ALLOWED_NODES) > 1


# Page size for mbind alignment. POSIX guarantees ``SC_PAGESIZE`` on Linux;
# 4 KiB is the universal fallback if the query somehow fails.
def _detect_page_size() -> int:
    try:
        if hasattr(os, "sysconf"):
            return int(os.sysconf("SC_PAGESIZE"))
    except (ValueError, OSError):
        pass
    return 4096


_PAGE_SIZE: int = _detect_page_size()

# Bits per ``unsigned long`` on this build. 64 on every supported arch
# we ship for, but spelled out so the bitmap-size math reads correctly
# anywhere ctypes is configured differently.
_BITS_PER_LONG: int = ctypes.sizeof(ctypes.c_ulong) * 8


def _maxnode_for_bitmap(n_words: int) -> int:
    """Compute the ``maxnode`` argument for ``mbind`` / ``set_mempolicy``.

    The syscall's userspace convention -- documented in ``mbind(2)`` and
    matched by ``libnuma`` -- is "the size of the nodes bitmap in bits,
    plus one". For an ``n_words``-word bitmap that is
    ``n_words * BITS_PER_LONG + 1``.

    The bug this replaces was ``max_node_id + 1``, which looks
    superficially right ("highest valid index plus one") but doesn't
    match what the kernel does. Tracing through ``mm/mempolicy.c``::

        --maxnode;                              # kernel decrements
        endmask = (maxnode % BITS_PER_LONG == 0)
                    ? ~0UL
                    : (1UL << (maxnode % BITS_PER_LONG)) - 1;
        nodes_addr[nlongs - 1] &= endmask;      # mask the last word

    With ``maxnode = max_id + 1`` and ``max_id = 7`` (an 8-node host),
    the kernel computes ``endmask = (1 << 7) - 1 = 0x7f`` and ANDs that
    into the last word -- silently dropping bit 7 (i.e. node 7). The
    user-visible symptom is ``/proc/self/numa_maps`` reporting
    ``interleave:0-6`` for what was intended as ``interleave:0-7``.

    Returning ``n_words * BITS_PER_LONG + 1`` gives the kernel a
    ``maxnode`` that's exactly a multiple of ``BITS_PER_LONG`` after
    its ``--maxnode``, which makes ``endmask = ~0UL`` and preserves
    every bit of the user's bitmap. Equivalent to what libnuma uses.
    """
    return n_words * _BITS_PER_LONG + 1


# ---- ctypes setup, only on capable hosts -----------------------------------


_libc: ctypes.CDLL | None = None
_INTERLEAVE_MASK: ctypes.Array[ctypes.c_ulong] | None = None
_MAXNODE: int = 0

if _AVAILABLE:
    try:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _libc.syscall.restype = ctypes.c_long

        # Bitmask covering allowed nodes, sized to one ``unsigned long``
        # per 64 nodes. Most hardware has < 16 nodes so one word suffices,
        # but cap it correctly so high-node-id machines (large 4-socket
        # AMD parts, multi-rack POWER systems) still work.
        _max_allowed = max(_ALLOWED_NODES)
        _n_words = (_max_allowed // _BITS_PER_LONG) + 1
        _NodemaskType = ctypes.c_ulong * _n_words
        _INTERLEAVE_MASK = _NodemaskType()
        for _node in _ALLOWED_NODES:
            _INTERLEAVE_MASK[_node // _BITS_PER_LONG] |= 1 << (_node % _BITS_PER_LONG)
        _MAXNODE = _maxnode_for_bitmap(_n_words)
    except OSError:
        # libc.so.6 not found -- bizarre on Linux but possible in
        # minimal containers. Demote to no-op rather than fail open.
        _AVAILABLE = False
        _libc = None


# ---- Public surface --------------------------------------------------------


def is_available() -> bool:
    """True if NUMA interleaving can be applied on this host.

    False on non-Linux platforms, on single-node hosts, on
    unsupported CPU architectures, and on Linux where ``libc.so.6``
    isn't loadable (minimal containers).
    """
    return _AVAILABLE


def allowed_nodes() -> list[int]:
    """List of NUMA node ids this process may allocate on.

    Honors cgroup ``cpuset.mems`` restrictions. Returns an empty list
    on non-Linux hosts and on Linux where the relevant ``/sys`` and
    ``/proc`` files are unreadable.
    """
    return list(_ALLOWED_NODES)


def _page_align_range(addr: int, length: int) -> tuple[int, int]:
    """Round ``[addr, addr + length)`` outward to whole pages.

    ``np.memmap`` with a non-page-aligned ``offset`` mmap's a page-
    aligned floor of that offset and exposes the user's view inside
    it; the front-pad and back-pad belong to the same VMA. Rounding
    outward is therefore safe (we never cross into a neighbor's
    mapping), and avoids the kernel's silent truncation of unaligned
    mbind ranges that some kernel versions exhibit.
    """
    aligned_start = addr - (addr % _PAGE_SIZE)
    aligned_end = (addr + length + _PAGE_SIZE - 1) // _PAGE_SIZE * _PAGE_SIZE
    return aligned_start, aligned_end - aligned_start


_WARNED_FAILURES: set[str] = set()


def _warn_once(syscall_name: str, errno: int) -> None:
    """Emit a once-per-process warning for a failed NUMA syscall.

    Different syscalls warn independently so the operator gets one
    message for each kind of failure; repeated failures of the same
    syscall stay silent. The warning identifies which syscall failed
    so the diagnosis (cgroup restriction vs. seccomp filter vs. old
    kernel) is immediate.
    """
    if syscall_name in _WARNED_FAILURES:
        return
    _WARNED_FAILURES.add(syscall_name)
    warnings.warn(
        f"{syscall_name} failed (errno={errno}); NUMA placement was not "
        f"applied. The store will still work correctly but may run slower "
        f"on multi-node hosts. Further failures of this syscall will not "
        f"be reported.",
        stacklevel=3,
    )


def apply_interleave_to_memmap(memmap: np.ndarray) -> bool:
    """Apply ``MPOL_INTERLEAVE`` to the pages backing ``memmap``.

    Returns ``True`` if the policy was applied, ``False`` otherwise
    (non-applicable host, zero-length region, syscall failure).

    Page-cache pages backing the underlying file region will be
    allocated round-robin across the allowed NUMA nodes **on first
    fault** -- pages already in the page cache are not migrated by
    this call. For files we write ourselves, see
    :func:`interleave_thread_policy`, which controls placement at
    write time and is the lever that actually changes warm-cache
    behavior; reader-side ``mbind`` is the cold-cache complement.

    On the first syscall failure in this process, emits a
    ``UserWarning`` so the operator can investigate (cgroup
    restriction, seccomp filter, old kernel). Subsequent failures
    are silent to avoid log spam.
    """
    if not _AVAILABLE or _libc is None or _INTERLEAVE_MASK is None or _SYSCALLS is None:
        return False
    length = int(memmap.nbytes)
    if length == 0:
        return False
    aligned_addr, aligned_length = _page_align_range(int(memmap.ctypes.data), length)
    rc = _libc.syscall(
        ctypes.c_long(_SYSCALLS.mbind),
        ctypes.c_void_p(aligned_addr),
        ctypes.c_ulong(aligned_length),
        ctypes.c_int(_MPOL_INTERLEAVE),
        ctypes.cast(_INTERLEAVE_MASK, ctypes.c_void_p),
        ctypes.c_ulong(_MAXNODE),
        ctypes.c_uint(0),
    )
    if rc != 0:
        _warn_once("mbind(MPOL_INTERLEAVE)", ctypes.get_errno())
        return False
    return True


# ---- Thread-local mempolicy (writer-side) ----------------------------------


@contextlib.contextmanager
def interleave_thread_policy() -> Iterator[bool]:
    """Set ``MPOL_INTERLEAVE`` on the calling thread for the scope.

    Yields ``True`` if the policy was actually applied, ``False`` if
    the helper degraded to no-op (non-applicable host, syscall
    failure). The boolean lets callers branch on whether the
    optimization is active without re-querying the module.

    ``set_mempolicy(2)`` affects the calling thread's default
    mempolicy, which is what the kernel uses for page-cache
    allocations during ``write(2)`` syscalls on this thread. Wrapping
    :meth:`ColStoreWriter.write` in this scope causes the page-cache
    pages backing the file to be allocated round-robin across the
    allowed NUMA nodes at write time. Every subsequent reader of the
    file -- this process or any other, now or weeks later -- sees the
    distributed placement without needing any reader-side work.

    This is the lever that actually delivers the NUMA win.
    Reader-side ``apply_interleave_to_memmap`` only affects pages
    not yet in the page cache (cold reads); for warm reads of files
    we wrote ourselves under default policy, page placement is
    already locked and reader-side ``mbind`` can't move it.

    The previous policy is captured via ``get_mempolicy(2)`` and
    restored on exit, so processes running under ``numactl
    --interleave=all`` (which sets a non-default thread policy on
    startup) get exactly what they had before the scope.

    No-op on non-applicable hosts.
    """
    if not _AVAILABLE or _libc is None or _INTERLEAVE_MASK is None or _SYSCALLS is None:
        yield False
        return

    # Capture the previous policy so the exit can restore it exactly.
    # On a thread with default policy this returns mode=MPOL_DEFAULT
    # and an empty mask; the restore then sets MPOL_DEFAULT again,
    # which is a no-op but harmless.
    prev_mode = ctypes.c_int(0)
    prev_mask = (ctypes.c_ulong * _n_words)()
    rc = _libc.syscall(
        ctypes.c_long(_SYSCALLS.get_mempolicy),
        ctypes.byref(prev_mode),
        ctypes.cast(prev_mask, ctypes.c_void_p),
        ctypes.c_ulong(_MAXNODE),
        ctypes.c_void_p(0),  # addr=NULL: query the thread default, not a VMA
        ctypes.c_ulong(0),  # flags=0
    )
    if rc != 0:
        _warn_once("get_mempolicy", ctypes.get_errno())
        yield False
        return

    # Set MPOL_INTERLEAVE.
    rc = _libc.syscall(
        ctypes.c_long(_SYSCALLS.set_mempolicy),
        ctypes.c_int(_MPOL_INTERLEAVE),
        ctypes.cast(_INTERLEAVE_MASK, ctypes.c_void_p),
        ctypes.c_ulong(_MAXNODE),
    )
    if rc != 0:
        _warn_once("set_mempolicy(MPOL_INTERLEAVE)", ctypes.get_errno())
        yield False
        return

    try:
        yield True
    finally:
        # Restore. For MPOL_DEFAULT the kernel rejects a non-empty
        # mask, but get_mempolicy returns an empty mask when the
        # current policy is MPOL_DEFAULT, so the captured pair is
        # always self-consistent.
        _libc.syscall(
            ctypes.c_long(_SYSCALLS.set_mempolicy),
            ctypes.c_int(prev_mode.value),
            ctypes.cast(prev_mask, ctypes.c_void_p),
            ctypes.c_ulong(_MAXNODE),
        )


def writer_policy_scope() -> contextlib.AbstractContextManager[bool]:
    """Return the NUMA context manager wrapping writer body writes.

    Single source of truth for "should this writer enter
    ``MPOL_INTERLEAVE`` for its scope?". Both
    :class:`colstore.writer.ColStoreWriter` (streaming writes) and
    :func:`colstore.format.write_dataset` (one-shot
    :func:`colstore.store` path) call this so the two write paths
    have identical NUMA semantics.

    Resolves :func:`colstore.config.get_numa_policy`:

    * ``"local"`` -> :class:`contextlib.nullcontext` (no-op)
    * non-applicable host -> :class:`contextlib.nullcontext`
    * ``"auto"`` / ``"interleave"`` on multi-node Linux ->
      :func:`interleave_thread_policy`

    Importing :mod:`colstore.config` is deferred to call time to keep
    this module free of intra-package import cycles at load.
    """
    from . import config  # local import: _numa is imported by reader/writer

    if config.get_numa_policy() == "local" or not is_available():
        return contextlib.nullcontext(False)
    return interleave_thread_policy()
