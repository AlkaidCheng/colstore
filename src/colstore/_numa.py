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

import ctypes
import os
import platform
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


# ---- Syscall constants -----------------------------------------------------

# mbind(2) syscall numbers, by uname machine. From the Linux kernel's
# arch/<arch>/include/uapi/asm/unistd*.h tables; stable across kernel
# versions for each arch.
_MBIND_SYSCALL_NRS: dict[str, int] = {
    "x86_64": 237,
    "aarch64": 235,
    "ppc64le": 259,
    "s390x": 268,
}

# Policy modes from <linux/mempolicy.h>.
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
_MBIND_SYSCALL_NR = _MBIND_SYSCALL_NRS.get(platform.machine(), -1)
_ALLOWED_NODES: list[int] = _detect_allowed_nodes() if _PLATFORM_IS_LINUX else []

# Treat the host as "NUMA-capable for our purposes" only when there's
# more than one allowed node. On single-node systems interleave is
# either a no-op or a slight pessimization (the kernel has to do
# extra bookkeeping for a policy that picks the only available node
# every time), so we skip the syscall entirely.
_AVAILABLE: bool = _PLATFORM_IS_LINUX and _MBIND_SYSCALL_NR > 0 and len(_ALLOWED_NODES) > 1


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
        _n_words = (_max_allowed // 64) + 1
        _NodemaskType = ctypes.c_ulong * _n_words
        _INTERLEAVE_MASK = _NodemaskType()
        for _node in _ALLOWED_NODES:
            _INTERLEAVE_MASK[_node // 64] |= 1 << (_node % 64)
        # ``maxnode`` in mbind is the maximum-permitted node id PLUS ONE.
        _MAXNODE = _max_allowed + 1
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


_WARNED_FAILURE = False


def apply_interleave_to_memmap(memmap: np.ndarray) -> bool:
    """Apply ``MPOL_INTERLEAVE`` to the pages backing ``memmap``.

    Returns ``True`` if the policy was applied, ``False`` otherwise
    (non-applicable host, zero-length region, syscall failure).

    Page-cache pages backing the underlying file region will be
    allocated round-robin across the allowed NUMA nodes on first
    fault, instead of all on whichever node serviced the I/O.

    Must be called BEFORE the memmap is accessed in any way that
    faults pages in. Pages already faulted in with a different
    policy are not migrated by this call -- ``MPOL_MF_MOVE`` would
    do that but is expensive and seldom needed in the open ->
    apply -> gather flow we target.

    On the first syscall failure in this process, emits a
    ``UserWarning`` so the operator can investigate (cgroup
    restriction, seccomp filter, old kernel). Subsequent failures
    are silent to avoid log spam.
    """
    global _WARNED_FAILURE
    if not _AVAILABLE or _libc is None or _INTERLEAVE_MASK is None:
        return False
    length = int(memmap.nbytes)
    if length == 0:
        return False
    aligned_addr, aligned_length = _page_align_range(int(memmap.ctypes.data), length)
    rc = _libc.syscall(
        ctypes.c_long(_MBIND_SYSCALL_NR),
        ctypes.c_void_p(aligned_addr),
        ctypes.c_ulong(aligned_length),
        ctypes.c_int(_MPOL_INTERLEAVE),
        ctypes.cast(_INTERLEAVE_MASK, ctypes.c_void_p),
        ctypes.c_ulong(_MAXNODE),
        ctypes.c_uint(0),
    )
    if rc != 0:
        errno = ctypes.get_errno()
        if not _WARNED_FAILURE:
            warnings.warn(
                f"mbind(MPOL_INTERLEAVE) failed (errno={errno}); NUMA "
                f"interleaving was not applied. The store will still "
                f"work correctly but may run slower on multi-node hosts. "
                f"Further failures will not be reported.",
                stacklevel=2,
            )
            _WARNED_FAILURE = True
        return False
    return True
