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
