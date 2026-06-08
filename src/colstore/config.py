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

# Default software-prefetch look-ahead for the gather kernels, in elements.
# Mirrors ``DEFAULT_PREFETCH_DISTANCE`` in ``include/colstore/gather.hpp``,
# which is the single authoritative value -- a test pins the two against each
# other via ``_gather.default_prefetch_distance()``. The right distance is
# hardware-dependent (it must cover the memory latency the prefetch is hiding,
# divided by the loop-iteration cost); use
# ``benchmark/sweep_prefetch_distance.py`` to measure on a target host.
_DEFAULT_PREFETCH_DISTANCE = 8

# Sanity ceiling for the prefetch distance: beyond a few thousand elements the
# prefetched lines are evicted again before use on any realistic cache, so a
# larger value is almost certainly a typo (e.g. bytes instead of elements).
_PREFETCH_DISTANCE_CEILING = 1 << 14


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
_prefetch_distance: int | Literal["auto"] = "auto"

# Lazily-loaded per-regime distance table for "auto" mode, populated either
# from the autotune cache (first resolution) or directly by
# ``autotune.calibrate_prefetch``. ``None`` + loaded=True means "no
# calibration available; use the compiled default".
_auto_prefetch_table: dict[str, int] | None = None
_auto_prefetch_table_loaded: bool = False


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


def get_prefetch_distance() -> int | Literal["auto"]:
    """Return the prefetch-distance setting for the gather kernels.

    Either an explicit look-ahead in elements (``0`` means prefetching is
    disabled) or ``"auto"`` (the default), in which case each gather resolves
    its distance per call via :func:`resolve_prefetch_distance`.
    """
    return _prefetch_distance


def set_prefetch_distance(distance: int | Literal["auto"]) -> None:
    """Set the software-prefetch look-ahead for the gather kernels.

    ``"auto"`` (the default) resolves the distance per call from two cheap
    signals -- source size vs last-level-cache size, and index sortedness --
    using the per-regime table measured by
    :func:`colstore.autotune.calibrate_prefetch`. Without a calibration cache
    ``"auto"`` falls back to the compiled default, so it is always safe.

    An explicit ``distance > 0`` prefetches that many iterations ahead for
    every gather; ``0`` disables prefetching, which can win when the gathered
    source is cache-resident and the prefetch instructions are pure overhead.
    Larger distances help when the source lives in DRAM: the prefetch must be
    issued early enough to cover the miss latency. Sweep with
    ``benchmark/sweep_prefetch_distance.py`` on the target host.
    """
    global _prefetch_distance
    if distance == "auto":
        _prefetch_distance = "auto"
        return
    if not isinstance(distance, int) or distance < 0 or distance > _PREFETCH_DISTANCE_CEILING:
        raise ValueError(
            f"prefetch_distance must be 'auto' or an int in "
            f"[0, {_PREFETCH_DISTANCE_CEILING}], got {distance!r}."
        )
    _prefetch_distance = int(distance)


def _set_auto_prefetch_table(table: dict[str, int] | None) -> None:
    """Install (or clear) the per-regime distance table used by ``"auto"``.

    Called by :func:`colstore.autotune.calibrate_prefetch` so a fresh
    calibration takes effect in-process without re-reading the cache file.
    Passing ``None`` resets to the not-yet-loaded state (used by tests).
    """
    global _auto_prefetch_table, _auto_prefetch_table_loaded
    _auto_prefetch_table = table
    _auto_prefetch_table_loaded = table is not None


def resolve_prefetch_distance(source_nbytes: int, indices_sorted: bool) -> int:
    """Return the effective prefetch distance for one gather call.

    With an explicit setting this is a passthrough, so call sites can use it
    unconditionally. With ``"auto"`` the access regime is classified from the
    caller's two signals -- the gathered source's size against the
    last-level-cache size (resident vs DRAM-bound) and whether the index
    array is monotonically non-decreasing -- and looked up in the calibrated
    table. No table (calibration never run, or a different machine) falls
    back to the compiled default distance.
    """
    setting = _prefetch_distance
    if setting != "auto":
        return setting

    global _auto_prefetch_table, _auto_prefetch_table_loaded
    if not _auto_prefetch_table_loaded:
        from . import autotune  # deferred: autotune imports config at module level

        _auto_prefetch_table = autotune.load_cached_prefetch()
        _auto_prefetch_table_loaded = True
    if _auto_prefetch_table is None:
        return _DEFAULT_PREFETCH_DISTANCE

    from . import autotune

    size_name = "resident" if source_nbytes <= autotune.llc_bytes() else "dram"
    order_name = "sorted" if indices_sorted else "unsorted"
    return _auto_prefetch_table[f"{size_name}_{order_name}"]


_MASK_DENSITY_GATE_DEFAULT = 0.0
_mask_density_gate: float | Literal["auto"] = "auto"
_auto_mask_density: float | None = None
_auto_mask_density_loaded = False


def get_mask_density_gate() -> float | Literal["auto"]:
    """Return the boolean-mask-native density-gate setting.

    Either an explicit selected-fraction threshold or ``"auto"`` (the
    default), in which case mask reads resolve the gate per call via
    :func:`resolve_mask_density_gate`.
    """
    return _mask_density_gate


def set_mask_density_gate(gate: float | Literal["auto"]) -> None:
    """Set the density gate for the boolean-mask-native read route.

    Multi-record boolean-mask reads with native dtypes take the mask-native
    kernel when the mask's selected fraction (``count_nonzero / n_rows``)
    is at or above the gate, and lower to ``np.flatnonzero`` + the fancy
    paths below it; the crossover is hardware-dependent.

    ``"auto"`` (the default) uses the per-host gate measured by
    :func:`colstore.autotune.calibrate_mask_density` (the ``mask-density``
    target of ``colstore calibrate``), falling back to the compiled
    default of 0.0 (route on at every density) when no calibration cache
    exists -- on multi-threaded hosts the lowered route loses at every
    measured density, so calibration exists to *raise* or disable the gate
    on hosts where sparse masks lose (e.g. single-core environments). An
    explicit float >= 0 overrides both; values above 1.0 disable the route
    entirely, which is also the benchmark baseline toggle.
    """
    global _mask_density_gate
    if gate == "auto":
        _mask_density_gate = "auto"
        return
    if isinstance(gate, bool) or not isinstance(gate, (int, float)) or gate < 0.0:
        raise ValueError(f"mask_density_gate must be 'auto' or a float >= 0, got {gate!r}.")
    _mask_density_gate = float(gate)


def _set_auto_mask_density(gate: float | None) -> None:
    """Install (or clear) the calibrated gate used by ``"auto"``.

    Called by :func:`colstore.autotune.calibrate_mask_density` so a fresh
    calibration takes effect in-process without re-reading the cache file.
    Passing ``None`` resets to the not-yet-loaded state (used by cache
    clearing and tests).
    """
    global _auto_mask_density, _auto_mask_density_loaded
    _auto_mask_density = gate
    _auto_mask_density_loaded = gate is not None


def resolve_mask_density_gate() -> float:
    """Return the effective mask-density gate for one read.

    With an explicit setting this is a passthrough. With ``"auto"`` the
    calibrated per-host gate is used when its cache exists and its hardware
    fingerprint matches this machine (loaded lazily once per process);
    otherwise the compiled default applies.
    """
    setting = _mask_density_gate
    if setting != "auto":
        return setting

    global _auto_mask_density, _auto_mask_density_loaded
    if not _auto_mask_density_loaded:
        from . import autotune  # deferred: autotune imports config at module level

        _auto_mask_density = autotune.load_cached_mask_density()
        _auto_mask_density_loaded = True
    if _auto_mask_density is None:
        return _MASK_DENSITY_GATE_DEFAULT
    return _auto_mask_density


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

    * ``"auto"`` (default) -- ``MPOL_INTERLEAVE`` on multi-node Linux so
      page-cache pages distribute across nodes as they fault in; no-op
      everywhere else (single-node Linux, macOS, Windows, blocked
      syscall). Significant win on multi-socket / multi-NPS hardware
      (see :mod:`colstore._numa` for measurements).
    * ``"interleave"`` -- force interleave even where ``"auto"`` would
      skip; mainly for testing.
    * ``"local"`` -- no-op; the kernel's default first-touch policy. Use
      for low-concurrency workloads (e.g. ``max_workers=1``) where forced
      interleaving costs more remote-memory hops than it saves.
    """
    if policy not in ("auto", "interleave", "local"):
        raise ValueError(f"numa policy must be 'auto', 'interleave', or 'local'; got {policy!r}.")
    global _numa_policy
    _numa_policy = policy
