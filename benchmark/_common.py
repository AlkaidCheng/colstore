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

__all__ = [
    "Result",
    "TimeStats",
    "best_time",
    "check_equal",
    "colstore",
    "cpp_available",
    "machine_fingerprint",
    "make_parser",
    "make_store",
    "max_threads",
    "random_column",
    "standard_columns",
    "time_stats",
    "uniform_rows",
    "write_multirecord",
    "write_summary",
]


# ---- Timing -----------------------------------------------------------------


def best_time(fn: Callable[[], Any], *, repeat: int, warmup: int = 3) -> float:
    """Best (minimum) wall-clock seconds over ``repeat`` runs.

    Minimum is the right summary for a like-for-like kernel comparison: it
    is the run least perturbed by the OS. ``warmup`` runs are discarded so
    page-cache fill and first-touch allocation are not timed.
    """
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


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


def make_parser(description: str) -> argparse.ArgumentParser:
    """Standard benchmark options: ``--repeat``, ``--skip-*``, ``--json``, ``--scale``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repeat", type=int, default=10, help="timed runs per cell (default 10)")
    parser.add_argument("--scale", type=float, default=1.0, help="multiply all row counts")
    parser.add_argument(
        "--skip-correctness", action="store_true", help="skip the per-cell correctness gate"
    )
    parser.add_argument("--skip-bench", action="store_true", help="run only the correctness gates")
    parser.add_argument(
        "--json", type=Path, default=None, metavar="PATH", help="write the JSON summary to PATH"
    )
    return parser


# ---- Synthetic data and stores ----------------------------------------------


def random_column(rng: np.random.Generator, n: int, dtype: Any) -> np.ndarray:
    """A length-``n`` column of ``dtype`` with representative random values."""
    dt = np.dtype(dtype)
    if dt.kind == "f":
        return rng.standard_normal(n).astype(dt)
    info = np.iinfo(dt)
    return rng.integers(info.min // 2, info.max // 2, size=n, dtype=np.int64).astype(dt)


def standard_columns(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """An f8/f4/i4/i2 column set, the default multi-column workload."""
    return {
        "f8": random_column(rng, n, np.float64),
        "f4": random_column(rng, n, np.float32),
        "i4": random_column(rng, n, np.int32),
        "i2": random_column(rng, n, np.int16),
    }


def uniform_rows(total: int, n_records: int) -> list[int]:
    """Split ``total`` rows into ``n_records`` near-equal records."""
    per = total // n_records
    rows = [per] * (n_records - 1)
    rows.append(total - per * (n_records - 1))
    return rows


def write_multirecord(
    path: Path, columns: dict[str, np.ndarray], rows_per_record: list[int]
) -> None:
    """Stream ``columns`` into ``path`` as one record per ``rows_per_record`` entry."""
    offset = 0
    with colstore.create(path) as writer:
        for n in rows_per_record:
            writer.write({k: v[offset : offset + n] for k, v in columns.items()})
            offset += n


def make_store(
    path: Path, columns: dict[str, np.ndarray], rows_per_record: list[int]
) -> colstore.ColStoreReader:
    """Build a multi-record store and return an open reader for it."""
    write_multirecord(path, columns, rows_per_record)
    return colstore.open(path)


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
