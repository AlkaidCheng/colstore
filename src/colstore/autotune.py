"""One-time, cached calibration of the gather thread cap.

The static default from :func:`colstore.config._default_gather_thread_cap`
(physical cores // 2, clamped to a small ceiling) lands within ~10-20% of
optimal on most machines. :func:`calibrate` closes the gap by measuring
the gather kernel at a range of thread counts on a synthetic scatter,
picking the smallest count within a tolerance of the best, and caching
the result keyed by a hardware fingerprint at
``$XDG_CACHE_HOME/colstore/threads.json`` (falling back to ``~/.cache``).
The cache is consulted once, lazily; calibration itself runs only on an
explicit :func:`calibrate` / :func:`ensure_calibrated` call, so import
stays fast and no benchmark runs behind the user's back.

The same pattern covers the prefetch distance: :func:`calibrate_prefetch`
sweeps it over four access regimes ({cache-resident, DRAM-bound} x
{sorted, unsorted}) into ``prefetch.json``. With
``config.set_prefetch_distance("auto")`` (the default), each gather
resolves its distance from that table using two cheap call-time signals
(source size vs last-level-cache size, index sortedness); uncalibrated
``"auto"`` falls back to the compiled default, so behavior is unchanged
until calibration is explicitly run.
"""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config

# Bump when the calibration procedure changes in a way that invalidates old
# cached results (e.g. different workload shape or scoring rule).
_CALIBRATION_VERSION = 1

# Candidate thread counts to sweep. Capped to the OpenMP max at runtime.
_CANDIDATE_THREADS = (1, 2, 4, 8, 16)

# Synthetic workload sizing. The gather is memory-latency-bound and its thread
# scaling saturates at a size-dependent knee: a sweep on the EPYC 7763 (NPS4)
# put the knee at ~4 threads for 4M indices but ~16 for >= 64M, so calibrating
# at 4M sampled below the regime the cap actually governs and under-picked the
# knee. Sample in the saturated regime so _pick_knee reflects production-scale
# gathers (bounded by _CALIB_SOURCE_ROWS; raises one-time calibration cost).
_CALIB_SOURCE_ROWS = 50_000_000
_CALIB_N_INDICES = 32_000_000

# Timing rounds shared by every calibration target. Each round measures every
# candidate once (interleaved), so slow drift in background load -- the
# dominant noise source on shared nodes -- biases all candidates equally
# instead of penalizing whichever candidate happened to be measured during a
# spike. The per-candidate statistic is the median over rounds, which stays
# honest when no quiet window ever occurs. The first rounds are warmup and
# discarded. Individual targets may override the count via their ``rounds``
# parameter.
_CALIB_ROUNDS = 10
_CALIB_WARMUP_ROUNDS = 2

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


def _interleaved_samples_ms(
    candidates: Sequence[int],
    time_once: Callable[[int], float],
    rounds: int,
) -> dict[int, list[float]]:
    """Time every candidate once per round; return kept samples in ms.

    ``time_once(candidate)`` runs one measurement and returns seconds. The
    first :data:`_CALIB_WARMUP_ROUNDS` rounds are discarded. Shared by all
    calibration targets so they inherit the same drift-resistant methodology;
    see the note on :data:`_CALIB_ROUNDS`.
    """
    kept: dict[int, list[float]] = {c: [] for c in candidates}
    for round_index in range(_CALIB_WARMUP_ROUNDS + rounds):
        for candidate in candidates:
            elapsed = time_once(candidate)
            if round_index >= _CALIB_WARMUP_ROUNDS:
                kept[candidate].append(elapsed * 1e3)
    return kept


def _median_ms(samples: dict[int, list[float]]) -> dict[int, float]:
    return {c: float(np.median(ts)) for c, ts in samples.items()}


def _pick_knee(times_ms: dict[int, float]) -> int:
    """Smallest candidate within the knee tolerance of the best time.

    Used for thread caps (smallest cap within 95% of best throughput --
    equivalent in the time domain) and for prefetch distances (smallest
    distance, letting 0 = disabled win ties). Fewer threads and fewer
    outstanding prefetches both mean less pressure on the shared memory
    subsystem at equal speed.
    """
    best = min(times_ms.values())
    return min(c for c, t in times_ms.items() if best / t >= _KNEE_TOLERANCE)


def _half_picks(samples: dict[int, list[float]]) -> tuple[int, int] | None:
    """Knee picks recomputed from the two halves of the rounds, or ``None``.

    A disagreement between the halves is the stability diagnostic shared by
    all calibration targets: it usually means the machine is busy. ``None``
    when there are too few rounds to split.
    """
    n_rounds = len(next(iter(samples.values())))
    half = n_rounds // 2
    if half == 0:
        return None
    pick_a = _pick_knee({c: float(np.median(ts[:half])) for c, ts in samples.items()})
    pick_b = _pick_knee({c: float(np.median(ts[half:])) for c, ts in samples.items()})
    return pick_a, pick_b


def _warn_if_unstable(
    label: str, samples: dict[int, list[float]], times_ms: dict[int, float]
) -> None:
    halves = _half_picks(samples)
    if halves is None or halves[0] == halves[1]:
        return
    pick_a, pick_b = halves
    spread = abs(times_ms[pick_a] - times_ms[pick_b]) / min(times_ms[pick_a], times_ms[pick_b])
    if spread > 1.0 - _KNEE_TOLERANCE:
        import warnings

        warnings.warn(
            f"calibration for '{label}' is unstable (half-picks {pick_a} vs "
            f"{pick_b}); the machine may be busy -- prefer a dedicated compute "
            f"node or raise rounds=.",
            stacklevel=3,
        )


def calibrate(*, persist: bool = True, verbose: bool = False, rounds: int = _CALIB_ROUNDS) -> int:
    """Measure and select a near-optimal gather thread cap for this machine.

    Sweeps a range of thread counts on a synthetic scatter -- every candidate
    timed once per round, interleaved, with the median over ``rounds`` rounds
    as the statistic (see :data:`_CALIB_ROUNDS`) -- picks the smallest count
    within :data:`_KNEE_TOLERANCE` of the best throughput, applies it via
    :func:`colstore.config.set_gather_thread_cap`, and (by default) caches it.
    A half-vs-half pick disagreement emits a stability warning.

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

    def _time_once(cap: int) -> float:
        start = time.perf_counter()
        _gather.gather(source, indices, output, cap)
        return time.perf_counter() - start

    samples = _interleaved_samples_ms(candidates, _time_once, rounds)
    times_ms = _median_ms(samples)
    measurements = {cap: output.nbytes / (t / 1e3) / 1e9 for cap, t in times_ms.items()}
    chosen = _pick_knee(times_ms)
    if verbose:
        for cap in candidates:
            print(f"  cap={cap:>3}: {measurements[cap]:6.2f} GB/s")
        halves = _half_picks(samples)
        suffix = f" (halves: {halves[0]} / {halves[1]})" if halves else ""
        print(
            f"Selected gather thread cap: {chosen} "
            f"(best {max(measurements.values()):.2f} GB/s){suffix}"
        )
    _warn_if_unstable("threads", samples, times_ms)

    config.set_gather_thread_cap(chosen)
    if persist:
        _write_cache(chosen, measurements)
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


def calibrate_prefetch(
    *, persist: bool = True, verbose: bool = False, rounds: int = _CALIB_ROUNDS
) -> dict[str, int]:
    """Measure per-regime prefetch distances for this machine.

    For each of the four regimes, sweeps :data:`_PREFETCH_CANDIDATES` on a
    synthetic gather at the *configured* thread cap (run :func:`calibrate`
    first so the cap reflects this host), using the shared interleaved-median
    methodology and stability diagnostic (see :data:`_CALIB_ROUNDS`). The
    pick is the smallest distance within :data:`_KNEE_TOLERANCE` of the best,
    letting 0 (disabled) win whenever prefetching buys nothing. Applies the
    table to ``"auto"`` resolution in-process, optionally persists it, and
    returns it.

    Requires the compiled C++ extension; raises :class:`RuntimeError` if it
    is unavailable.
    """
    from .kernels import cpp_available

    if not cpp_available():
        raise RuntimeError(
            "Prefetch calibration requires the compiled C++ gather extension, "
            "which is not available in this build."
        )
    from . import _gather  # type: ignore[attr-defined]

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

            def _time_once(
                distance: int,
                src: np.ndarray = source,
                idx: np.ndarray = indices,
                out: np.ndarray = output,
            ) -> float:
                start = time.perf_counter()
                _gather.gather(src, idx, out, cap, distance)
                return time.perf_counter() - start

            samples = _interleaved_samples_ms(_PREFETCH_CANDIDATES, _time_once, rounds)
            regime_times = _median_ms(samples)
            chosen = _pick_knee(regime_times)
            if verbose:
                for distance in _PREFETCH_CANDIDATES:
                    print(f"  {regime:<18} d={distance:>3}: {regime_times[distance]:8.3f} ms")
                halves = _half_picks(samples)
                suffix = f" (halves: d={halves[0]} / d={halves[1]})" if halves else ""
                print(f"  {regime:<18} -> d={chosen}{suffix}")
            _warn_if_unstable(f"prefetch:{regime}", samples, regime_times)
            table[regime] = chosen
            timings[regime] = regime_times
        del source

    config._set_auto_prefetch_table(table)
    if persist:
        _write_prefetch_cache(table, timings)
    return table


# ---- Mask-density gate calibration ----------------------------------------
# The boolean-mask-native route's density gate (see
# config.set_mask_density_gate) sits at a hardware-dependent crossover:
# below it, per-column int64 index traffic (8 bytes per SELECTED element)
# undercuts re-reading the 1-byte-per-ROW mask. The compiled default (0.15)
# is conservative; this calibration measures the crossover where jobs run.
_MASK_DENSITY_GRID = (0.02, 0.05, 0.08, 0.12, 0.15, 0.2, 0.3)
_MASK_DENSITY_WIN_RATIO = 1.05
_MASK_CALIB_N_RECORDS = 2_000
_MASK_CALIB_ROWS_PER_RECORD = 2_500
_MASK_CALIB_N_COLUMNS = 2  # the binding (multi-column) shape; single-column
#                            reads then run slightly conservative


def _mask_density_cache_path() -> Path:
    return _cache_dir() / "mask_density.json"


def _pick_mask_density_gate(ratios: dict[float, float]) -> float:
    """Crossover rule for the mask-density gate.

    ``ratios[d]`` is lowered-time / mask-route-time at grid density ``d``
    (>1 means the mask route wins). The gate is placed at the midpoint
    between the smallest grid density that wins by
    :data:`_MASK_DENSITY_WIN_RATIO` *at itself and every denser grid point*
    and the previous grid point (0 for the first) -- the monotonicity
    requirement keeps one noisy near-parity cell from dragging the gate
    into a regime the data does not support. If no density qualifies, the
    gate is 1.0: only all-true masks (a pure run-coalesced copy) take the
    route, effectively disabling it on hosts where it never pays.
    """
    grid = sorted(ratios)
    for position, density in enumerate(grid):
        if all(ratios[d] >= _MASK_DENSITY_WIN_RATIO for d in grid[position:]):
            lower = grid[position - 1] if position > 0 else 0.0
            return (lower + density) / 2.0
    return 1.0


def load_cached_mask_density() -> float | None:
    """Return the cached mask-density gate, or ``None`` if absent.

    Never raises; a missing, unreadable, foreign-fingerprint, or malformed
    cache returns ``None`` so resolution falls back to the compiled
    default gate.
    """
    try:
        with open(_mask_density_cache_path(), encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)
    except (OSError, ValueError):
        return None
    if payload.get("fingerprint") != _hardware_fingerprint():
        return None
    gate = payload.get("gate")
    if not isinstance(gate, (int, float)) or isinstance(gate, bool) or not 0.0 <= gate <= 1.0:
        return None
    return float(gate)


def _write_mask_density_cache(
    gate: float,
    ratios: dict[float, float],
    timings_ms: dict[float, dict[str, float]],
) -> None:
    payload = {
        "fingerprint": _hardware_fingerprint(),
        "gate": gate,
        "ratios": {str(d): r for d, r in ratios.items()},
        "timings_ms": {str(d): t for d, t in timings_ms.items()},
        "thread_cap_used": config.get_gather_thread_cap(),
        "calibration_version": _CALIBRATION_VERSION,
        "timestamp": time.time(),
    }
    path = _mask_density_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass


def calibrate_mask_density(
    *, persist: bool = True, verbose: bool = False, rounds: int = _CALIB_ROUNDS
) -> float:
    """Measure this machine's mask-density gate; apply, persist, return it.

    Builds a synthetic multi-record store (two f8 columns -- the binding
    multi-column shape; single-column reads then run slightly
    conservative), and for each grid density times the mask-native route
    against the lowered (flatnonzero) route through the public read API at
    the *configured* thread cap, toggling via the gate setting itself.
    Cells are interleaved round-robin per the shared methodology (see
    :data:`_CALIB_ROUNDS`), with medians as the statistic; the gate is
    placed by :func:`_pick_mask_density_gate` and a half-vs-half pick
    disagreement warns, mirroring the other targets' stability diagnostic.

    Requires the compiled C++ extension; raises :class:`RuntimeError` if it
    is unavailable.
    """
    import tempfile

    from .kernels import cpp_available

    if not cpp_available():
        raise RuntimeError(
            "Mask-density calibration requires the compiled C++ gather "
            "extension, which is not available in this build."
        )
    from .api import create as _create
    from .api import open as _open

    rng = np.random.default_rng(0)
    rows = _MASK_CALIB_ROWS_PER_RECORD
    total = _MASK_CALIB_N_RECORDS * rows
    columns = [f"c{i}" for i in range(_MASK_CALIB_N_COLUMNS)]
    densities = list(_MASK_DENSITY_GRID)
    masks = {d: rng.random(total) < d for d in densities}
    # Cell encoding for the shared integer-keyed helpers: cell 2*i is the
    # mask route at densities[i], cell 2*i + 1 the lowered route.
    cells = list(range(2 * len(densities)))

    saved_gate = config.get_mask_density_gate()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mask_density_calib.cstore"
        data = {name: rng.standard_normal(total) for name in columns}
        with _create(path) as writer:
            for r in range(_MASK_CALIB_N_RECORDS):
                writer.write({k: v[r * rows : (r + 1) * rows] for k, v in data.items()})
        del data
        dataset = _open(path)
        try:

            def _time_once(cell: int) -> float:
                density = densities[cell // 2]
                config.set_mask_density_gate(0.0 if cell % 2 == 0 else 2.0)
                start = time.perf_counter()
                dataset[masks[density], columns].dict()
                return time.perf_counter() - start

            samples = _interleaved_samples_ms(cells, _time_once, rounds)
        finally:
            config.set_mask_density_gate(saved_gate)
            dataset.close()

    times = _median_ms(samples)

    def _ratios_from(times_ms: dict[int, float]) -> dict[float, float]:
        return {d: times_ms[2 * i + 1] / times_ms[2 * i] for i, d in enumerate(densities)}

    ratios = _ratios_from(times)
    gate = _pick_mask_density_gate(ratios)

    n_rounds = len(next(iter(samples.values())))
    half = n_rounds // 2
    halves: tuple[float, float] | None = None
    if half:
        gate_a = _pick_mask_density_gate(
            _ratios_from({c: float(np.median(ts[:half])) for c, ts in samples.items()})
        )
        gate_b = _pick_mask_density_gate(
            _ratios_from({c: float(np.median(ts[half:])) for c, ts in samples.items()})
        )
        halves = (gate_a, gate_b)
        if gate_a != gate_b:
            import warnings

            warnings.warn(
                f"calibration for 'mask-density' is unstable (half-picks "
                f"{gate_a} vs {gate_b}); the machine may be busy -- prefer a "
                f"dedicated compute node or raise rounds=.",
                stacklevel=2,
            )
    if verbose:
        for i, density in enumerate(densities):
            print(
                f"  density {density:<5} mask {times[2 * i]:8.2f} ms   "
                f"lowered {times[2 * i + 1]:8.2f} ms   ratio {ratios[density]:5.2f}"
            )
        suffix = f" (halves: {halves[0]} / {halves[1]})" if halves else ""
        state = "route disabled on this host" if gate == 1.0 else "gate"
        print(f"  -> {state} = {gate}{suffix}")

    config._set_auto_mask_density(gate)
    if persist:
        _write_mask_density_cache(
            gate,
            ratios,
            {
                d: {"mask": times[2 * i], "lowered": times[2 * i + 1]}
                for i, d in enumerate(densities)
            },
        )
    return gate


# ---- Calibration cache management ----------------------------------------
def _remove_cache_file(path: Path) -> bool:
    """Delete one cache file; return whether it existed.

    A missing file is the success case for "clear" (idempotent), so it
    returns ``False`` rather than raising. Real failures (e.g. permissions)
    propagate -- silently failing to clear would leave the user believing
    stale calibration is gone when it is not.
    """
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def clear_cached_cap(*, reset_in_process: bool = True) -> bool:
    """Remove this machine's cached thread-cap calibration.

    Deletes ``threads.json`` if present. With ``reset_in_process`` (the
    default) the live cap is also reset to the static hardware default, so
    the running process behaves as if calibration had never happened; pass
    ``False`` to clear only the persisted cache and keep the current cap
    (e.g. when it was set manually after calibrating). Returns whether a
    cache file was removed. Idempotent.
    """
    removed = _remove_cache_file(_cache_path())
    if reset_in_process:
        config.set_gather_thread_cap(config._default_gather_thread_cap())
    return removed


def clear_cached_prefetch(*, reset_in_process: bool = True) -> bool:
    """Remove this machine's cached prefetch-distance calibration.

    Deletes ``prefetch.json`` if present. With ``reset_in_process`` (the
    default) the in-process ``"auto"`` table is also dropped, so subsequent
    gathers fall back to the compiled default distance immediately rather
    than at the next interpreter start. Returns whether a cache file was
    removed. Idempotent.
    """
    removed = _remove_cache_file(_prefetch_cache_path())
    if reset_in_process:
        config._set_auto_prefetch_table(None)
    return removed


def clear_cached_mask_density(*, reset_in_process: bool = True) -> bool:
    """Remove this machine's cached mask-density-gate calibration.

    Deletes ``mask_density.json`` if present. With ``reset_in_process`` (the
    default) the in-process calibrated gate is also dropped, so subsequent
    mask reads fall back to the compiled default gate immediately. Returns
    whether a cache file was removed. Idempotent.
    """
    removed = _remove_cache_file(_mask_density_cache_path())
    if reset_in_process:
        config._set_auto_mask_density(None)
    return removed


def clear_calibration(*, reset_in_process: bool = True) -> dict[str, bool]:
    """Remove all cached calibration for this machine.

    Returns ``{"threads": removed, "prefetch": removed, "mask-density":
    removed}`` indicating which cache files existed. See the per-target
    ``clear_cached_*`` functions for the ``reset_in_process`` semantics.
    """
    return {
        "threads": clear_cached_cap(reset_in_process=reset_in_process),
        "prefetch": clear_cached_prefetch(reset_in_process=reset_in_process),
        "mask-density": clear_cached_mask_density(reset_in_process=reset_in_process),
    }
