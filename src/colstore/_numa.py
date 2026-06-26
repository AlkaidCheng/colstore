"""Linux NUMA memory-policy helpers for colstore memmaps.

Sets ``MPOL_INTERLEAVE`` on the page-aligned regions covering file-backed
memmaps so that page-cache pages distribute across NUMA nodes as they
fault in, instead of concentrating on whichever node first touched them
(the kernel's default first-touch policy). On multi-socket /
multi-NUMA-node hardware, page placement can change gather throughput
substantially, but which policy wins depends on the access pattern and on
whether pages are faulted cold or already resident -- so callers should
measure on their own hardware rather than assume a fixed speedup.

Scope: this module places *memory*, and (via the runtime thread-binding
helpers at the end) can pin the gather's OpenMP pool to cores. The largest
multi-node gather speedup comes from binding the gather's threads, which was
measured to dominate data placement; :func:`bind_gather_threads` applies that
at runtime (``OMP_PROC_BIND``/``OMP_PLACES`` are read too early in import to
set from a library). It is an opt-in primitive -- nothing here calls it
automatically -- so the equivalent can still be done externally at process
start (e.g. ``numactl --cpunodebind`` or ``OMP_PROC_BIND`` / ``OMP_PLACES``).
See :func:`colstore.config.set_numa_policy` for the recipe.

This module:

  * No-ops cleanly on non-Linux platforms and on single-node Linux hosts.
  * Honors cgroup ``cpuset.mems`` restrictions: the interleave mask
    covers only nodes the process may allocate on.
  * Falls back silently on kernels without ``mbind(2)`` or under seccomp
    filters that block it, warning exactly once.
  * Has zero new system-library dependencies (no ``libnuma``): the
    syscalls are issued directly via :mod:`ctypes`. Syscall numbers vary
    by architecture; the mainline server archs (x86_64, aarch64, ppc64le,
    s390x) are supported and anything else no-ops.
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

# /sys topology roots, used by the thread-binding helpers below.
_SYS_NODE = Path("/sys/devices/system/node")
_SYS_CPU = Path("/sys/devices/system/cpu")

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

    The userspace convention -- documented in ``mbind(2)`` and matched by
    ``libnuma`` -- is "the size of the nodes bitmap in bits, plus one":
    ``n_words * BITS_PER_LONG + 1``. The intuitive ``max_node_id + 1`` is
    WRONG; the kernel (``mm/mempolicy.c``) masks the bitmap's last word::

        --maxnode;                              # kernel decrements
        endmask = (maxnode % BITS_PER_LONG == 0)
                    ? ~0UL
                    : (1UL << (maxnode % BITS_PER_LONG)) - 1;
        nodes_addr[nlongs - 1] &= endmask;      # mask the last word

    so ``maxnode = 8`` on an 8-node host yields ``endmask = 0x7f`` and
    silently drops node 7 (symptom: ``/proc/self/numa_maps`` reports
    ``interleave:0-6``). The whole-word value returned here makes
    ``endmask = ~0UL`` after the kernel's decrement, preserving every bit.
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
        # but cap it correctly so high-node-count systems (many sockets
        # or many NUMA domains) still work.
        _max_allowed = max(_ALLOWED_NODES)
        _n_words = (_max_allowed // _BITS_PER_LONG) + 1
        NodemaskType = ctypes.c_ulong * _n_words
        _INTERLEAVE_MASK = NodemaskType()
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

    Yields ``True`` if the policy was applied, ``False`` if the helper
    degraded to no-op (non-applicable host, syscall failure), so callers
    can branch without re-querying the module.

    ``set_mempolicy(2)`` governs page-cache allocations made during
    ``write(2)`` on this thread, so wrapping writer body writes in this
    scope distributes the file's page-cache pages round-robin across the
    allowed nodes at write time -- and every subsequent reader, in any
    process, sees the distributed placement. This is the lever that
    delivers the NUMA win: reader-side ``apply_interleave_to_memmap``
    affects only pages not yet in the page cache, and cannot move warm
    pages already placed under the default policy.

    The previous policy is captured via ``get_mempolicy(2)`` and restored
    on exit, so processes under ``numactl --interleave=all`` keep their
    configured policy after the scope.
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

    Single source of truth for whether a writer enters
    ``MPOL_INTERLEAVE``; both :class:`colstore.writer.ColStoreWriter` and
    :func:`colstore.format.write_dataset` call this so the two write
    paths have identical NUMA semantics. Resolves
    :func:`colstore.config.get_numa_policy`: ``"local"`` or a
    non-applicable host yields :class:`contextlib.nullcontext`;
    ``"auto"`` / ``"interleave"`` on multi-node Linux yields
    :func:`interleave_thread_policy`. The :mod:`colstore.config` import
    is deferred to call time to avoid an import cycle.
    """
    from . import config  # local import: _numa is imported by reader/writer

    if config.get_numa_policy() == "local" or not is_available():
        return contextlib.nullcontext(False)
    return interleave_thread_policy()


# ---- Runtime thread binding (reader-side) ----------------------------------
#
# OMP_PROC_BIND / OMP_PLACES are read once, at OpenMP initialization, which
# numpy/BLAS typically trigger during import -- too early for a library to set
# them and have the runtime notice. These helpers instead pin libgomp's worker
# pool at runtime, from inside a parallel region (see
# ``_gather.bind_threads_to_cpus``), reproducing ``OMP_PROC_BIND=spread`` with
# ``OMP_PLACES=cores``. Binding threads to cores (independent of NUMA data
# placement) was measured ~1.3x faster for the conversion gathers at scale;
# confirm on the target host. Pure no-op off Linux or without the extension.


def _read_list_file(path: Path) -> list[int]:
    """Parse a ``/sys`` cpulist/nodelist file; ``[]`` if unreadable."""
    try:
        return _parse_cpu_or_node_list(path.read_text())
    except OSError:
        return []


def _cpu_nodes() -> list[int]:
    """Sorted NUMA node ids that have at least one online CPU."""
    if not _SYS_NODE.is_dir():
        return []
    nodes: list[int] = []
    for entry in _SYS_NODE.glob("node[0-9]*"):
        try:
            num = int(entry.name[len("node") :])
        except ValueError:
            continue
        if _read_list_file(entry / "cpulist"):
            nodes.append(num)
    return sorted(nodes)


def _node_core_cpus(node: int) -> list[int]:
    """One logical CPU per physical core on ``node``, in ascending order.

    Hyperthread siblings are collapsed to their lowest-numbered CPU so the
    spread targets distinct physical cores, matching ``OMP_PLACES=cores``
    (siblings share memory ports and add no bandwidth).
    """
    seen: set[int] = set()
    cores: list[int] = []
    for cpu in _read_list_file(_SYS_NODE / f"node{node}" / "cpulist"):
        siblings = _read_list_file(_SYS_CPU / f"cpu{cpu}" / "topology" / "thread_siblings_list")
        representative = min(siblings) if siblings else cpu
        if representative not in seen:
            seen.add(representative)
            cores.append(representative)
    return cores


def spread_cpu_order(n: int) -> list[int]:
    """Up to ``n`` CPU ids interleaved one-per-node across physical cores.

    Replicates ``OMP_PROC_BIND=spread`` + ``OMP_PLACES=cores``: round-robin one
    physical core from each NUMA node in turn (node0 core0, node1 core0, ...,
    then node0 core1, ...), so consecutive workers land on different memory
    controllers. Returns ``[]`` where the topology is unreadable (non-Linux,
    minimal containers); callers treat that as "binding not applicable".
    """
    if n <= 0:
        return []
    per_node = [_node_core_cpus(node) for node in _cpu_nodes()]
    order: list[int] = []
    depth = max((len(cores) for cores in per_node), default=0)
    for level in range(depth):
        for cores in per_node:
            if level < len(cores):
                order.append(cores[level])
                if len(order) >= n:
                    return order
    return order


_LAST_BOUND: tuple[int, int] | None = None


def _native_bind_to_cpus(order: list[int]) -> int:
    """Call the native pin primitive for ``order``; ``-1`` if unavailable.

    Isolated so the binding orchestration has a single, directly patchable seam
    over the compiled extension (tests replace this rather than reaching through
    ``from . import _gather``). Returns the count pinned, or ``-1`` when the
    extension is missing or the platform is unsupported.
    """
    try:
        import numpy as np

        from . import _gather  # type: ignore[attr-defined]
    except ImportError:
        return -1
    return int(_gather.bind_threads_to_cpus(np.asarray(order, dtype=np.intc)))


def bind_gather_threads(cap: int | None = None, *, force: bool = False) -> int | None:
    """Pin the OpenMP gather pool ``spread`` across cores at runtime.

    Sizes the binding to ``cap`` workers (default: the configured gather thread
    cap) and pins worker ``t`` to ``spread_cpu_order(cap)[t]``. Returns the
    number of workers pinned, or ``None`` when binding is skipped or impossible:

      * off Linux, or where the ``/sys`` topology is unreadable;
      * when ``OMP_PROC_BIND`` is set in the environment -- an explicit
        launch-time policy wins, and re-pinning would fight it;
      * when the compiled extension is unavailable or the platform is
        unsupported (the native primitive returns ``-1``).

    Idempotent: re-pinning for an unchanged ``cap`` returns the prior count
    without touching affinities unless ``force=True``. Note this pins libgomp's
    *shared* pool, so it also governs other OpenMP work in the process; that is
    the same global scope as setting ``OMP_PROC_BIND`` at launch.
    """
    global _LAST_BOUND
    if not _PLATFORM_IS_LINUX or os.environ.get("OMP_PROC_BIND"):
        return None
    if cap is None:
        from . import config  # local import: _numa is imported by reader/config

        cap = config.get_gather_thread_cap()
    cap = int(cap)
    if cap <= 0:
        return None
    if not force and _LAST_BOUND is not None and _LAST_BOUND[0] == cap:
        return _LAST_BOUND[1]
    order = spread_cpu_order(cap)
    if not order:
        return None
    bound = _native_bind_to_cpus(order)
    if bound < 0:
        return None
    _LAST_BOUND = (cap, bound)
    return bound


def thread_binding_report() -> dict[str, object]:
    """Snapshot the process's per-thread CPU affinities, for verification.

    Returns a dict with the number of OS threads, how many *distinct* CPU
    masks they hold (``1`` means every thread shares one mask -- an unbound
    pool), whether ``OMP_PROC_BIND`` is set, and a small sample of masks.
    Pure inspection of ``/proc/self/task``; empty dict off Linux.
    """
    task = Path("/proc/self/task")
    if not task.is_dir():
        return {}
    masks: list[str] = []
    for status in task.glob("*/status"):
        try:
            text = status.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("Cpus_allowed_list:"):
                masks.append(line.split(":", 1)[1].strip())
                break
    distinct = sorted(set(masks))
    return {
        "n_threads": len(masks),
        "distinct_masks": len(distinct),
        "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
        "sample": distinct[:4],
    }


# ---- Startup binding policy + per-gather gate ------------------------------
#
# The parts of the binding decision that depend only on the host -- is spread
# binding applicable, and what is the aggregate-L3 threshold -- are resolved
# once and cached. The per-gather part (does *this* gather's working set exceed
# the threshold) is cheap and evaluated per call. Spreading threads needs more
# than one memory domain to spread across, so a single-NUMA-node host (and the
# single-core / virtualized hosts that report one node) is never applicable --
# binding there was measured neutral-to-harmful, so the gate skips it.


class BindingPolicy(NamedTuple):
    """Host-level inputs to the spread-binding gate, resolved once per process.

    ``applicable`` is the hardware precondition (Linux, multiple NUMA nodes);
    the per-call gate additionally consults the config knob and working set,
    and the actual pin (:func:`bind_gather_threads`) re-checks ``OMP_PROC_BIND``
    and extension availability.
    """

    applicable: bool
    aggregate_llc_bytes: int
    numa_nodes: int


_BINDING_POLICY: BindingPolicy | None = None


def binding_policy() -> BindingPolicy:
    """Resolve (and cache) the host-level binding policy.

    Computed on first use from the NUMA topology and cache hierarchy, then
    memoized for the process. ``applicable`` is ``True`` only on Linux with at
    least two NUMA nodes that have online CPUs -- the precondition for spread
    binding to place threads on distinct memory controllers.
    """
    global _BINDING_POLICY
    if _BINDING_POLICY is None:
        from . import autotune  # local import: autotune imports config which imports _numa

        nodes = len(_cpu_nodes()) if _PLATFORM_IS_LINUX else 0
        _BINDING_POLICY = BindingPolicy(
            applicable=_PLATFORM_IS_LINUX and nodes >= 2,
            aggregate_llc_bytes=autotune.aggregate_llc_bytes(),
            numa_nodes=nodes,
        )
    return _BINDING_POLICY


def _reset_binding_policy() -> None:
    """Clear the cached policy so the next call re-resolves it (tests only)."""
    global _BINDING_POLICY
    _BINDING_POLICY = None


def maybe_bind_for_gather(working_set_bytes: int, cap: int | None = None) -> int | None:
    """Pin the gather pool for a DRAM-bound gather, or return ``None``.

    Binds (via :func:`bind_gather_threads`) only when all hold: the optimization
    is enabled (:func:`colstore.config.get_gather_binding`); the host policy is
    applicable (Linux, multiple NUMA nodes); and ``working_set_bytes`` exceeds
    ``margin * aggregate_llc_bytes`` (the resident-vs-DRAM boundary). Returns
    the number of workers pinned, or ``None`` when the gate declines or binding
    is unavailable. Idempotent and cheap to call on every gather.
    """
    from . import config  # local import: _numa is imported by reader/config

    if not config.get_gather_binding():
        return None
    policy = binding_policy()
    if not policy.applicable:
        return None
    threshold = config.get_gather_bind_llc_margin() * policy.aggregate_llc_bytes
    if working_set_bytes <= threshold:
        return None
    return bind_gather_threads(cap)
