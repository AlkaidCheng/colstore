"""Shared scaffolding for the colstore benchmark scripts.

Centralizes the parts every benchmark repeated by hand: the timing
primitive, the correctness-gate helper (always run before any timing),
the standard argparse options, synthetic column/store builders, the
machine fingerprint, and the structured-JSON writer used by the
comprehensive benchmark (``run_benchmarks.py``). Keeping these in one
place means timing methodology and store construction stay identical
across the suite, and the JSON summary stays consistent with
``perf_suite.py``'s fingerprint shape.

Every benchmark needs ``colstore`` importable from a source checkout;
importing this module puts ``src`` on ``sys.path`` as a side effect, so
scripts can simply ``import _common`` first. Run, e.g.::

    PYTHONPATH=src python benchmark/run_benchmarks.py --json summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Make ``colstore`` importable from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import colstore
from colstore.config import get_gather_thread_cap
from colstore.kernels import cpp_available, max_threads
from colstore.profiling import (
    ProfileResult,
    peak_thread_watcher,
    profile,
    profile_interleaved,
)

__all__ = [
    "ProfileResult",
    "Result",
    "TimeStats",
    "add_common_args",
    "apply_runtime_config",
    "check_equal",
    "colstore",
    "compare",
    "cpp_available",
    "drop_pagecache",
    "machine_fingerprint",
    "max_threads",
    "peak_thread_watcher",
    "profile",
    "profile_interleaved",
    "scaled_rows",
    "time_stats",
    "write_summary",
]


# ---- Timing -----------------------------------------------------------------


@dataclass
class TimeStats:
    """median / min / p95 milliseconds over a timed cell."""

    median_ms: float
    min_ms: float
    p95_ms: float
    repeat: int


def time_stats(fn: Callable[[], Any], *, repeat: int, warmup: int = 1) -> TimeStats:
    """median/min/p95 milliseconds over ``repeat`` runs (``warmup`` discarded).

    Median is reported alongside min because on a noisy multi-tenant box it
    is more representative; p95 surfaces tail variance.
    """
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    p95 = statistics.quantiles(samples, n=20)[18] if len(samples) >= 20 else max(samples)
    return TimeStats(statistics.median(samples), min(samples), p95, repeat)


# ---- Correctness gate -------------------------------------------------------


def check_equal(got: np.ndarray, expected: np.ndarray, label: str, *, dtype: bool = True) -> None:
    """Assert value (and, by default, dtype) equality before timing.

    Every benchmark runs its correctness gate before the timed region so a
    miscompiled or mis-routed kernel is never reported as "fast".
    """
    if dtype and got.dtype != expected.dtype:
        raise AssertionError(f"{label}: dtype {got.dtype} != {expected.dtype}")
    if not np.array_equal(got, expected):
        raise AssertionError(f"{label}: value mismatch")


# ---- Argument parsing -------------------------------------------------------


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    repeat: int = 10,
    warmup: int = 1,
    rows: int | None = None,
    cols: int | None = None,
    record_counts: list[int] | None = None,
    indices: int | None = None,
    dtype: str | None = None,
    tmpdir: bool = False,
    threads: bool = False,
    scale: bool = False,
    json: bool = False,
    skip_correctness: bool = True,
) -> argparse.ArgumentParser:
    """Attach the canonical benchmark options, in one place, by the same names.

    The harmonized vocabulary for the whole suite. Timing controls
    (``--repeat``, ``--warmup``, ``--skip-bench`` and, by default,
    ``--skip-correctness``) are always added. Every other group is opt-in: a
    sizing keyword both *enables* its flag and *sets its default*, so a script
    advertises only the knobs it honors while the name, type, and meaning stay
    identical across benchmarks. A flag that means the same thing in two
    scripts is always spelled the same way.

    Pass ``threads=True`` for ``--thread`` / ``--worker`` (the gather-thread
    and column-worker *caps*; apply them with :func:`apply_runtime_config`),
    ``scale=True`` for ``--scale`` (apply with :func:`scaled_rows`),
    ``json=True`` for ``--json``, and ``tmpdir=True`` for ``--tmpdir`` (``None``
    default -> caller uses a TemporaryDirectory).
    """
    parser.add_argument("--repeat", type=int, default=repeat, help="timed runs per cell")
    parser.add_argument("--warmup", type=int, default=warmup, help="warmup runs before timing")
    parser.add_argument("--skip-bench", action="store_true", help="run only the correctness gate")
    if skip_correctness:
        parser.add_argument(
            "--skip-correctness", action="store_true", help="skip the correctness gate"
        )
    if rows is not None:
        parser.add_argument("--rows", type=int, default=rows, help="rows in the synthetic store")
    if cols is not None:
        parser.add_argument("--cols", type=int, default=cols, help="number of columns")
    if record_counts is not None:
        parser.add_argument(
            "--record-counts",
            type=int,
            nargs="+",
            default=list(record_counts),
            help="record counts to sweep",
        )
    if indices is not None:
        parser.add_argument("--indices", type=int, default=indices, help="number of gather indices")
    if dtype is not None:
        parser.add_argument("--dtype", default=dtype, help="column dtype")
    if tmpdir:
        parser.add_argument(
            "--tmpdir",
            type=Path,
            default=None,
            metavar="DIR",
            help="store directory (default: a TemporaryDirectory)",
        )
    if threads:
        parser.add_argument(
            "--thread",
            type=int,
            default=None,
            help="cap on gather threads (config.gather_thread_cap)",
        )
        parser.add_argument(
            "--worker",
            type=int,
            default=None,
            help="cap on column-pool workers (config.max_workers)",
        )
    if scale:
        parser.add_argument("--scale", type=float, default=1.0, help="multiply all row counts")
    if json:
        parser.add_argument(
            "--json", type=Path, default=None, metavar="PATH", help="write the JSON summary to PATH"
        )
    return parser


def apply_runtime_config(args: argparse.Namespace) -> None:
    """Apply ``--thread`` / ``--worker`` caps to ``colstore.config`` if set.

    A no-op for either cap left at its ``None`` default, so a script can call
    this unconditionally whether or not it opted into ``threads=True``.
    """
    from colstore import config

    cap = getattr(args, "thread", None)
    if cap is not None:
        config.set_gather_thread_cap(cap)
    workers = getattr(args, "worker", None)
    if workers is not None:
        config.set_max_workers(workers)


def scaled_rows(n: int, args: argparse.Namespace) -> int:
    """``n`` multiplied by ``--scale`` (1.0 when the flag is absent), as an int."""
    return int(n * getattr(args, "scale", 1.0))


# ---- Result records ---------------------------------------------------------


@dataclass
class Result:
    """One timed measurement, shaped for plotting.

    A flat list of these is the JSON payload: plots group by ``scenario``,
    put a parameter from ``params`` on the x-axis, ``median_ms`` (or
    ``speedup``) on the y-axis, and use ``variant`` as the series.
    """

    scenario: str
    variant: str
    params: dict[str, Any]
    median_ms: float
    min_ms: float
    p95_ms: float
    repeat: int
    rows: int | None = None
    throughput_rows_per_s: float | None = None
    speedup_vs: str | None = None
    speedup: float | None = None

    @classmethod
    def from_stats(
        cls,
        scenario: str,
        variant: str,
        params: dict[str, Any],
        stats: TimeStats,
        *,
        rows: int | None = None,
    ) -> Result:
        throughput = None
        if rows is not None and stats.median_ms > 0:
            throughput = rows / (stats.median_ms / 1000.0)
        return cls(
            scenario=scenario,
            variant=variant,
            params=dict(params),
            median_ms=stats.median_ms,
            min_ms=stats.min_ms,
            p95_ms=stats.p95_ms,
            repeat=stats.repeat,
            rows=rows,
            throughput_rows_per_s=throughput,
        )


def set_speedup(fast: Result, slow: Result) -> None:
    """Record ``fast``'s speedup over the ``slow`` baseline (median/median)."""
    if fast.median_ms > 0:
        fast.speedup_vs = slow.variant
        fast.speedup = slow.median_ms / fast.median_ms


# ---- Machine fingerprint and JSON output ------------------------------------


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def machine_fingerprint() -> dict[str, Any]:
    """OS / CPU / version snapshot, matching ``perf_suite.py``'s shape."""
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
    except ImportError:
        physical = None
        logical = os.cpu_count()
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count_physical": physical,
        "cpu_count_logical": logical,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "colstore_version": getattr(colstore, "__version__", "unknown"),
        "openmp_max_threads": max_threads(),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "gather_thread_cap": get_gather_thread_cap(),
        "cpp_available": cpp_available(),
        "git_sha": _git_sha(),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def write_summary(path: Path, results: list[Result], *, meta: dict[str, Any] | None = None) -> None:
    """Write ``{fingerprint, meta?, results[]}`` as indented JSON to ``path``."""
    payload: dict[str, Any] = {
        "fingerprint": machine_fingerprint(),
        "results": [asdict(r) for r in results],
    }
    if meta:
        payload["meta"] = meta
    path.write_text(json.dumps(payload, indent=2))


# ---- Rich harness: wall + CPU + peak threads + page faults ------------------
# Used by the benchmarks that need to *prove* parallelism (cpu/wall ratio),
# observe peak thread counts, or distinguish cold (major-fault) from warm
# reads -- check_numa, check_parallel_copy, check_frame_construction. Linux
# specifics (resource.getrusage, posix_fadvise) are imported lazily so this
# module still imports on platforms without them.


def drop_pagecache(paths: list[Path]) -> None:
    """Evict file pages via ``posix_fadvise(DONTNEED)`` (no root needed).

    Best-effort: the kernel ignores the hint for pages with live references
    (e.g. an open ``np.memmap``), so callers wanting a true cold read must
    not hold a reader across the eviction.
    """
    import ctypes

    posix_fadv_dontneed = 4
    libc = ctypes.CDLL("libc.so.6")
    for path in paths:
        targets = [p for p in path.iterdir() if p.is_file()] if path.is_dir() else [path]
        for target in targets:
            fd = os.open(str(target), os.O_RDONLY)
            try:
                libc.posix_fadvise(fd, 0, 0, posix_fadv_dontneed)
            finally:
                os.close(fd)


def compare(
    specs: list[tuple[str, Callable[[], Any]]],
    *,
    repeat: int = 5,
    warmup: int = 2,
    baseline: int = 0,
    setups: list[Callable[[], Any] | None] | None = None,
    throughput_rows: int | None = None,
    show: bool = True,
) -> list[ProfileResult]:
    """Interleaved A/B comparison with the suite's standard presentation.

    The harmonized path for variant comparisons: profiles the ``(label, fn)``
    specs interleaved (via :func:`colstore.profiling.profile_interleaved`, so
    page-cache and scheduler state stay comparable), then prints one
    standard line per variant -- wall, cpu, cpu/wall ratio, peak threads,
    page-fault deltas -- followed by a speedup column versus the variant at
    index ``baseline``, and an optional rows/s throughput column when
    ``throughput_rows`` is given. Returns the :class:`ProfileResult` per spec
    in input order, so a caller can print a domain-specific footer (e.g. MB/s)
    from the same measurements.

    ``setups`` supplies a per-variant setup run outside the timed region
    (``None`` for variants that need none) -- use it for destructive workloads
    that must rebuild state (e.g. a fresh store) before each timed write.
    """
    labels = [label for label, _ in specs]
    fns = [fn for _, fn in specs]
    results: list[ProfileResult] = profile_interleaved(
        labels, fns, repeat=repeat, warmup=warmup, setups=setups
    )
    if show:
        base_wall = results[baseline].wall_ms
        for i, result in enumerate(results):
            if i == baseline:
                speedup = "  (baseline)"
            else:
                ratio = base_wall / result.wall_ms if result.wall_ms > 0 else float("inf")
                speedup = f"  speedup={ratio:5.2f}x"
            tput = ""
            if throughput_rows is not None:
                tput = f"  {result.throughput(throughput_rows) / 1e6:7.1f}M rows/s"
            print(result.report() + speedup + tput)
    return results
