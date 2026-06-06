"""One-time, cached calibration of the gather thread cap.

The static default from :func:`colstore.config._default_gather_thread_cap`
(physical cores // 2, clamped to a small ceiling) lands within ~10-20% of
optimal on most machines. :func:`calibrate` closes the remaining gap by
measuring the gather kernel at a range of thread counts on a synthetic
scatter, picking the smallest count whose throughput is within a tolerance of
the best, and caching the result keyed by a hardware fingerprint.

The cache lives at ``$XDG_CACHE_HOME/colstore/threads.json`` (falling back to
``~/.cache``). It is consulted once, lazily, the first time a default cap is
needed; calibration itself is only ever run when the user explicitly calls
:func:`calibrate` (or :func:`ensure_calibrated`), so import stays fast and no
benchmark runs behind the user's back.

The same pattern covers the gather prefetch distance: :func:`calibrate_prefetch`
sweeps the distance over four access regimes ({cache-resident, DRAM-bound} x
{sorted, unsorted}) and caches a per-regime table in
``$XDG_CACHE_HOME/colstore/prefetch.json``. With
``config.set_prefetch_distance("auto")`` (the default), each gather resolves
its distance from that table using two cheap call-time signals -- source size
vs last-level-cache size, and index sortedness. Uncalibrated ``"auto"`` falls
back to the compiled default, so behavior is unchanged until calibration is
explicitly run.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import config

# Bump when the calibration procedure changes in a way that invalidates old
# cached results (e.g. different workload shape or scoring rule).
_CALIBRATION_VERSION = 1

# Candidate thread counts to sweep. Capped to the OpenMP max at runtime.
_CANDIDATE_THREADS = (1, 2, 4, 8, 16)

# Synthetic workload sizing. Large enough to be firmly in the parallel regime
# (above the kernel's serial threshold) and to exercise memory bandwidth.
_CALIB_SOURCE_ROWS = 50_000_000
_CALIB_N_INDICES = 4_000_000
_CALIB_REPEATS = 5

# A thread count counts as "good enough" if it reaches this fraction of the
# best observed throughput. We then pick the *smallest* such count, since
# fewer threads means less contention and better behaviour under an outer
# multi-column thread pool.
_KNEE_TOLERANCE = 0.95


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(Path.home(), ".cache")
    return Path(base) / "colstore"


def _cache_path() -> Path:
    return _cache_dir() / "threads.json"


def _hardware_fingerprint() -> str:
    """Stable-ish key for the current machine's relevant characteristics.

    Combines logical/physical CPU counts, the processor string, and the
    platform. Deliberately coarse: it should change when the hardware changes
    but stay stable across runs on the same box.
    """
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or 0
    except ImportError:
        physical = 0
    logical = os.cpu_count() or 0
    processor = platform.processor() or platform.machine()
    return f"v{_CALIBRATION_VERSION}|{processor}|phys={physical}|log={logical}|{platform.system()}"


def load_cached_cap() -> int | None:
    """Return the cached thread cap for this machine, or ``None`` if absent.

    Never raises: a missing, unreadable, or stale cache simply returns
    ``None`` so callers fall back to the static default.
    """
    path = _cache_path()
    try:
        with open(path, encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)
    except (OSError, ValueError):
        return None
    if payload.get("fingerprint") != _hardware_fingerprint():
        return None
    cap = payload.get("thread_cap")
    if isinstance(cap, int) and cap >= 1:
        return cap
    return None


def _write_cache(thread_cap: int, measurements: dict[int, float]) -> None:
    path = _cache_path()
    payload = {
        "fingerprint": _hardware_fingerprint(),
        "thread_cap": thread_cap,
        "throughput_gbps": {str(k): v for k, v in measurements.items()},
        "calibration_version": _CALIBRATION_VERSION,
        "timestamp": time.time(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)  # atomic on POSIX
    except OSError:
        # Calibration still applies in-process even if the cache can't be
        # written (read-only home, etc.); just skip persistence.
        pass


def _measure_cap(
    source: np.ndarray,
    indices: np.ndarray,
    output: np.ndarray,
    cap: int,
    repeats: int,
) -> float:
    """Return best-of-N throughput in GB/s of output for a given thread cap."""
    from . import _gather  # type: ignore[attr-defined]

    best_wall = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        _gather.gather(source, indices, output, cap)
        best_wall = min(best_wall, time.perf_counter() - start)
    output_bytes = output.nbytes
    return output_bytes / max(best_wall, 1e-12) / 1e9


def calibrate(*, persist: bool = True, verbose: bool = False) -> int:
    """Measure and select a near-optimal gather thread cap for this machine.

    Sweeps a range of thread counts on a synthetic scatter, picks the smallest
    count within :data:`_KNEE_TOLERANCE` of the best throughput, applies it via
    :func:`colstore.config.set_gather_thread_cap`, and (by default) caches it.

    Requires the compiled C++ extension; raises :class:`RuntimeError` if it is
    unavailable. Returns the chosen cap.
    """
    from .kernels import cpp_available, max_threads

    if not cpp_available():
        raise RuntimeError(
            "Calibration requires the compiled C++ gather extension, which is "
            "not available in this build."
        )

    omp_max = max_threads()
    candidates = sorted({c for c in _CANDIDATE_THREADS if c <= omp_max} | {1})

    rng = np.random.default_rng(0)
    source = rng.standard_normal(_CALIB_SOURCE_ROWS).astype(np.float32)
    n_indices = min(_CALIB_N_INDICES, _CALIB_SOURCE_ROWS)
    indices = rng.permutation(_CALIB_SOURCE_ROWS)[:n_indices].astype(np.int64)
    output = np.empty(n_indices, dtype=np.float32)

    # Warm the page cache and the kernel once before timing.
    from . import _gather  # type: ignore[attr-defined]

    _gather.gather(source, indices, output, 1)

    measurements: dict[int, float] = {}
    for cap in candidates:
        gbps = _measure_cap(source, indices, output, cap, _CALIB_REPEATS)
        measurements[cap] = gbps
        if verbose:
            print(f"  cap={cap:>3}: {gbps:6.2f} GB/s")

    best_gbps = max(measurements.values())
    threshold = best_gbps * _KNEE_TOLERANCE
    chosen = min(cap for cap, gbps in measurements.items() if gbps >= threshold)

    config.set_gather_thread_cap(chosen)
    if persist:
        _write_cache(chosen, measurements)
    if verbose:
        print(f"Selected gather thread cap: {chosen} (best {best_gbps:.2f} GB/s)")
    return chosen


def ensure_calibrated(*, verbose: bool = False) -> int:
    """Apply a cached cap if present, otherwise calibrate once and cache it.

    Convenience entry point: cheap on every call after the first successful
    calibration, since it just reads the cached value.
    """
    cached = load_cached_cap()
    if cached is not None:
        config.set_gather_thread_cap(cached)
        return cached
    return calibrate(persist=True, verbose=verbose)


def apply_cached_cap_if_present() -> bool:
    """Apply a cached cap if one exists for this machine; return whether it did.

    Called at import time so a previously calibrated machine benefits without
    any explicit call, while an uncalibrated one keeps the static default.
    """
    cached = load_cached_cap()
    if cached is None:
        return False
    config.set_gather_thread_cap(cached)
    return True


# ---- Prefetch-distance calibration --------------------------------------

# Candidate distances to sweep, per regime. 0 means prefetching disabled.
# The list extends to 512 so the knee is bracketed rather than clipped: a
# first calibration round on the target host showed regimes still improving
# at 128, and a chosen value sitting on the sweep boundary is a sign the true
# optimum was never measured.
_PREFETCH_CANDIDATES = (0, 2, 4, 8, 16, 32, 64, 128, 256, 512)

# The four access regimes "auto" distinguishes at call time. resident/dram is
# decided by comparing the gather's source size against the LLC size;
# sorted/unsorted by the (cheap) monotonicity of the index array.
_PREFETCH_REGIMES = (
    "resident_unsorted",
    "resident_sorted",
    "dram_unsorted",
    "dram_sorted",
)

# Workload sizing. The resident source targets half the LLC; the DRAM source
# targets several LLCs, capped so calibration never allocates more than 1 GiB.
_PREFETCH_CALIB_RESIDENT_FRACTION = 0.5
_PREFETCH_CALIB_DRAM_MULTIPLE = 4
_PREFETCH_CALIB_DRAM_CAP_BYTES = 1 << 30
_PREFETCH_CALIB_N_INDICES = 1_000_000

# Timing rounds per regime. Each round measures every candidate distance once
# (interleaved), so slow drift in background load -- the dominant noise source
# on shared nodes -- biases all candidates equally instead of penalizing
# whichever distance happened to be measured during a spike. The per-distance
# statistic is the median over rounds, which stays honest when no quiet
# window ever occurs. The first rounds are warmup and discarded.
_PREFETCH_CALIB_ROUNDS = 7
_PREFETCH_CALIB_WARMUP_ROUNDS = 2

# Same knee rule as the thread cap, but over distances: prefer the *smallest*
# distance within tolerance of the best throughput. Fewer outstanding
# prefetches means less pressure on the shared memory subsystem, and it lets
# 0 (disabled) win whenever prefetching buys nothing.
_PREFETCH_KNEE_TOLERANCE = 0.95

_LLC_FALLBACK_BYTES = 32 * 1024 * 1024


def _prefetch_cache_path() -> Path:
    return _cache_dir() / "prefetch.json"


def llc_bytes() -> int:
    """Return the last-level-cache size in bytes (per socket), best effort.

    Reads the Linux sysfs cache hierarchy of cpu0 and takes the largest
    reported level. Falls back to a 32 MiB guess on non-Linux platforms or
    restricted sysfs. The value only steers the resident-vs-DRAM regime
    boundary, so a coarse number is fine.
    """
    best = 0
    cache_root = Path("/sys/devices/system/cpu/cpu0/cache")
    try:
        for index_dir in cache_root.glob("index*"):
            raw = (index_dir / "size").read_text().strip()
            if raw.endswith("K"):
                size = int(raw[:-1]) * 1024
            elif raw.endswith("M"):
                size = int(raw[:-1]) * 1024 * 1024
            else:
                size = int(raw)
            best = max(best, size)
    except (OSError, ValueError):
        pass
    return best or _LLC_FALLBACK_BYTES


def load_cached_prefetch() -> dict[str, int] | None:
    """Return the cached per-regime prefetch table, or ``None`` if absent.

    Never raises; a missing, unreadable, stale, or malformed cache returns
    ``None`` so ``"auto"`` falls back to the compiled default distance.
    """
    try:
        with open(_prefetch_cache_path(), encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)
    except (OSError, ValueError):
        return None
    if payload.get("fingerprint") != _hardware_fingerprint():
        return None
    table = payload.get("prefetch_distances")
    if not isinstance(table, dict):
        return None
    out: dict[str, int] = {}
    for regime in _PREFETCH_REGIMES:
        value = table.get(regime)
        if not isinstance(value, int) or value < 0:
            return None
        out[regime] = value
    return out


def _write_prefetch_cache(table: dict[str, int], timings_ms: dict[str, dict[int, float]]) -> None:
    payload = {
        "fingerprint": _hardware_fingerprint(),
        "prefetch_distances": table,
        "timings_ms": {r: {str(d): t for d, t in ts.items()} for r, ts in timings_ms.items()},
        "llc_bytes": llc_bytes(),
        "thread_cap_used": config.get_gather_thread_cap(),
        "calibration_version": _CALIBRATION_VERSION,
        "timestamp": time.time(),
    }
    path = _prefetch_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass


def _pick_knee(times_ms: dict[int, float]) -> int:
    """Smallest distance within the knee tolerance of the best time."""
    best = min(times_ms.values())
    return min(d for d, t in times_ms.items() if best / t >= _PREFETCH_KNEE_TOLERANCE)


def _sweep_regime_interleaved(
    source: np.ndarray,
    indices: np.ndarray,
    output: np.ndarray,
    cap: int,
    rounds: int,
) -> dict[int, list[float]]:
    """Time every candidate distance once per round, ``rounds`` kept rounds.

    Interleaving is the point: consecutive per-distance repeats would let a
    background-load spike penalize one distance specifically; round-robin
    spreads drift evenly across all candidates.
    """
    from . import _gather  # type: ignore[attr-defined]

    kept: dict[int, list[float]] = {d: [] for d in _PREFETCH_CANDIDATES}
    for round_index in range(_PREFETCH_CALIB_WARMUP_ROUNDS + rounds):
        for distance in _PREFETCH_CANDIDATES:
            start = time.perf_counter()
            _gather.gather(source, indices, output, cap, distance)
            elapsed = time.perf_counter() - start
            if round_index >= _PREFETCH_CALIB_WARMUP_ROUNDS:
                kept[distance].append(elapsed * 1e3)
    return kept


def calibrate_prefetch(
    *, persist: bool = True, verbose: bool = False, rounds: int = _PREFETCH_CALIB_ROUNDS
) -> dict[str, int]:
    """Measure per-regime prefetch distances for this machine.

    For each of the four regimes, sweeps :data:`_PREFETCH_CANDIDATES` on a
    synthetic gather at the *configured* thread cap (run :func:`calibrate`
    first so the cap reflects this host). Every distance is timed once per
    round, interleaved; the per-distance statistic is the median over
    ``rounds`` rounds, and the pick is the smallest distance within
    :data:`_PREFETCH_KNEE_TOLERANCE` of the best. Applies the table to
    ``"auto"`` resolution in-process, optionally persists it, and returns it.

    As a stability diagnostic, the pick is recomputed from the first and
    second halves of the rounds separately; a disagreement is reported via
    :mod:`warnings` (and printed when ``verbose``), which usually means the
    machine is busy -- prefer a dedicated compute node, or raise ``rounds``.

    Requires the compiled C++ extension; raises :class:`RuntimeError` if it
    is unavailable.
    """
    from .kernels import cpp_available

    if not cpp_available():
        raise RuntimeError(
            "Prefetch calibration requires the compiled C++ gather extension, "
            "which is not available in this build."
        )

    llc = llc_bytes()
    resident_rows = max(1 << 20, int(llc * _PREFETCH_CALIB_RESIDENT_FRACTION) // 8)
    dram_bytes = min(_PREFETCH_CALIB_DRAM_CAP_BYTES, llc * _PREFETCH_CALIB_DRAM_MULTIPLE)
    dram_rows = max(resident_rows * 2, dram_bytes // 8)
    cap = config.get_gather_thread_cap()
    rng = np.random.default_rng(0)

    table: dict[str, int] = {}
    timings: dict[str, dict[int, float]] = {}
    for size_name, rows in (("resident", resident_rows), ("dram", dram_rows)):
        source = rng.standard_normal(rows)  # float64
        n_idx = min(_PREFETCH_CALIB_N_INDICES, rows)
        unsorted_idx = rng.integers(0, rows, size=n_idx).astype(np.int64)
        for order_name, indices in (
            ("unsorted", unsorted_idx),
            ("sorted", np.sort(unsorted_idx)),
        ):
            regime = f"{size_name}_{order_name}"
            output = np.empty(n_idx, dtype=np.float64)
            samples = _sweep_regime_interleaved(source, indices, output, cap, rounds)
            regime_times = {d: float(np.median(ts)) for d, ts in samples.items()}
            chosen = _pick_knee(regime_times)
            # Stability diagnostic: picks from the two halves of the rounds.
            half = rounds // 2
            pick_a = _pick_knee({d: float(np.median(ts[:half])) for d, ts in samples.items()})
            pick_b = _pick_knee({d: float(np.median(ts[half:])) for d, ts in samples.items()})
            if verbose:
                for distance in _PREFETCH_CANDIDATES:
                    print(f"  {regime:<18} d={distance:>3}: {regime_times[distance]:8.3f} ms")
                print(f"  {regime:<18} -> d={chosen} (halves: d={pick_a} / d={pick_b})")
            if pick_a != pick_b:
                time_a = regime_times[pick_a]
                time_b = regime_times[pick_b]
                spread = abs(time_a - time_b) / min(time_a, time_b)
                if spread > 1.0 - _PREFETCH_KNEE_TOLERANCE:
                    import warnings

                    warnings.warn(
                        f"prefetch calibration for regime '{regime}' is unstable "
                        f"(half-picks d={pick_a} vs d={pick_b}); the machine may be "
                        f"busy -- prefer a dedicated compute node or raise rounds=.",
                        stacklevel=2,
                    )
            table[regime] = chosen
            timings[regime] = regime_times
        del source

    config._set_auto_prefetch_table(table)
    if persist:
        _write_prefetch_cache(table, timings)
    return table
