"""Global configuration for the colstore package.

Settings can be inspected and changed at runtime. They are used as defaults
when constructing :class:`~colstore.ColStoreReader` instances; per-instance overrides
take precedence over these globals.
"""

import os
from typing import Literal

try:
    import psutil

    _physical_cores = psutil.cpu_count(logical=False)
    _DEFAULT_MAX_WORKERS = _physical_cores or os.cpu_count() or 1
except ImportError:
    _physical_cores = None
    _DEFAULT_MAX_WORKERS = os.cpu_count() or 1

MadviseOption = Literal["normal", "sequential", "random", "willneed", "dontneed"]
GatherBackend = Literal["cpp", "numpy", "numba"]
NumaPolicy = Literal["auto", "interleave", "local"]

# Hard ceiling on gather threads. The kernel is memory-bandwidth-bound, so
# throughput saturates at a small thread count well below the core count on
# essentially all single-socket hardware; beyond ~8 threads extra parallelism
# buys nothing and the fork/join + contention cost dominates (the difference
# between 8 and 244 threads was measured at ~40x slower).
_GATHER_THREAD_CEILING = 8


def _default_gather_thread_cap() -> int:
    """Derive a near-optimal default gather thread cap from the hardware.

    Half the physical cores is a robust proxy for the memory-bandwidth
    saturation point, clamped to ``[1, _GATHER_THREAD_CEILING]``. Uses physical
    (not logical) cores because hyperthreads share memory ports and do not add
    bandwidth. A cached autotuned value, if present, overrides this; see
    :mod:`colstore.autotune`.
    """
    physical = _physical_cores if _physical_cores else (os.cpu_count() or 1)
    return max(1, min(_GATHER_THREAD_CEILING, physical // 2))


# Typed module-level state. Held as separate variables rather than a single
# `dict[str, object]` so mypy can preserve the precise types through the
# get_* / set_* accessors without casts.
_max_workers: int = _DEFAULT_MAX_WORKERS
_default_madvise: MadviseOption | None = "sequential"
_default_backend: GatherBackend = "cpp"
_gather_thread_cap: int = _default_gather_thread_cap()
_numa_policy: NumaPolicy = "auto"


def get_max_workers() -> int:
    """Return the package-wide default thread count for multi-column reads."""
    return _max_workers


def set_max_workers(n: int) -> None:
    """Set the package-wide thread count for multi-column reads.

    Recommend setting to the number of *physical* CPU cores; hyperthreaded
    logical cores rarely help memory-bound workloads.
    """
    global _max_workers
    if n < 1:
        raise ValueError(f"max_workers must be >= 1, got {n}.")
    _max_workers = int(n)


def get_gather_thread_cap() -> int:
    """Return the maximum OpenMP threads a single gather kernel call may use.

    This caps within-column parallelism in the C++/Numba backends. The default
    is derived from the physical core count (see
    :func:`_default_gather_thread_cap`) or, if calibration has been run, the
    cached autotuned value.
    """
    return _gather_thread_cap


def set_gather_thread_cap(n: int) -> None:
    """Set the per-call gather thread cap (``>= 1``).

    Lower is better for small/scattered reads; the memory-bandwidth ceiling
    means values much above the physical memory channels rarely help.
    """
    global _gather_thread_cap
    if n < 1:
        raise ValueError(f"gather_thread_cap must be >= 1, got {n}.")
    _gather_thread_cap = int(n)


def get_default_madvise() -> MadviseOption | None:
    """Return the default ``madvise`` hint applied to new ``ColStoreReader`` opens."""
    return _default_madvise


def set_default_madvise(advice: MadviseOption | None) -> None:
    """Set the package-wide default ``madvise`` hint for new ``ColStoreReader`` opens."""
    global _default_madvise
    _default_madvise = advice


def get_default_backend() -> GatherBackend:
    """Return the default gather backend for new ``ColStoreReader`` opens."""
    return _default_backend


def set_default_backend(backend: GatherBackend) -> None:
    """Set the default gather backend (``"cpp"``, ``"numpy"``, or ``"numba"``)."""
    global _default_backend
    if backend not in ("cpp", "numpy", "numba"):
        raise ValueError(f"backend must be 'cpp', 'numpy', or 'numba'; got {backend!r}.")
    _default_backend = backend


def get_numa_policy() -> NumaPolicy:
    """Return the package-wide NUMA memory policy applied to new opens.

    See :func:`set_numa_policy` for the available policies.
    """
    return _numa_policy


def set_numa_policy(policy: NumaPolicy) -> None:
    """Set the NUMA memory policy applied to file-backed memmaps at open time.

    Policies:

    * ``"auto"`` (default) -- apply ``MPOL_INTERLEAVE`` on multi-node Linux
      hosts so that page-cache pages distribute across NUMA nodes as they
      fault in, instead of concentrating on whichever node serviced the
      I/O. No-op on single-node Linux, on macOS, on Windows, and on any
      host where the syscall is blocked or unsupported. Significant win
      on multi-socket / multi-NPS server hardware (measured ~1.8x on
      ``ds.dict()`` on a dual EPYC 7763, 8 NUMA nodes).
    * ``"interleave"`` -- force interleave even where ``"auto"`` would
      skip. Mainly useful for testing; in practice ``"auto"`` already
      enables interleave whenever it would help.
    * ``"local"`` -- no-op. Pages fall under the kernel's default
      first-touch policy. Set this if you have a low-concurrency
      workload (e.g. ``max_workers=1``) where forced interleaving
      causes more remote-memory hops than it saves.
    """
    if policy not in ("auto", "interleave", "local"):
        raise ValueError(f"numa policy must be 'auto', 'interleave', or 'local'; got {policy!r}.")
    global _numa_policy
    _numa_policy = policy
