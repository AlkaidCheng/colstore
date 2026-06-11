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
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
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
    "RichResult",
    "Run",
    "TimeStats",
    "add_common_args",
    "apply_runtime_config",
    "bench",
    "bench_interleaved",
    "best_time",
    "check_equal",
    "colstore",
    "cpp_available",
    "drop_pagecache",
    "machine_fingerprint",
    "make_parser",
    "make_store",
    "max_threads",
    "random_column",
    "run_script",
    "scaled_rows",
    "standard_columns",
    "thread_watcher",
    "time_call",
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


# ---- Rich harness: wall + CPU + peak threads + page faults ------------------
# Used by the benchmarks that need to *prove* parallelism (cpu/wall ratio),
# observe peak thread counts, or distinguish cold (major-fault) from warm
# reads -- check_numa, check_parallel_copy, check_frame_construction. Linux
# specifics (resource.getrusage, posix_fadvise) are imported lazily so this
# module still imports on platforms without them.


@dataclass
class Run:
    """One timed call with process metrics."""

    wall_ms: float
    cpu_ms: float
    peak_threads: int
    major_pf: int
    minor_pf: int


@dataclass
class RichResult:
    """A labelled set of :class:`Run` samples, summarized by :meth:`report`."""

    label: str
    runs: list[Run] = field(default_factory=list)

    def report(self) -> str:
        if not self.runs:
            return f"  {self.label:<62}  (no runs)"
        # Best-of-N for wall/cpu (lowest noise floor); medians for the rest.
        best = min(self.runs, key=lambda r: r.wall_ms)
        med_threads = statistics.median(r.peak_threads for r in self.runs)
        med_major = statistics.median(r.major_pf for r in self.runs)
        med_minor = statistics.median(r.minor_pf for r in self.runs)
        ratio = best.cpu_ms / best.wall_ms if best.wall_ms > 0 else float("nan")
        return (
            f"  {self.label:<62} "
            f"wall={best.wall_ms:8.2f}ms  cpu={best.cpu_ms:8.1f}ms  "
            f"ratio={ratio:5.2f}x  threads={int(med_threads):2d}  "
            f"pf={int(med_major)}/{int(med_minor)}"
        )


@contextmanager
def thread_watcher(interval_s: float = 0.001) -> Any:
    """Track the peak ``threading.active_count()`` for the scope.

    Yields a callable returning the peak observed so far.
    """
    peak = [threading.active_count()]
    stop = [False]

    def poll() -> None:
        while not stop[0]:
            n = threading.active_count()
            if n > peak[0]:
                peak[0] = n
            time.sleep(interval_s)

    watcher = threading.Thread(target=poll, daemon=True)
    watcher.start()
    try:
        yield lambda: peak[0]
    finally:
        stop[0] = True
        watcher.join(timeout=0.5)


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


def time_call(fn: Callable[[], Any], *, drop_cache_paths: list[Path] | None = None) -> Run:
    """One call of ``fn``, capturing wall/cpu time, peak threads, and faults."""
    import resource

    if drop_cache_paths:
        drop_pagecache(drop_cache_paths)
    ru_before = resource.getrusage(resource.RUSAGE_SELF)
    cpu_before = time.process_time()
    wall_before = time.perf_counter()
    with thread_watcher() as peak_fn:
        fn()
        peak = peak_fn()
    wall_ms = (time.perf_counter() - wall_before) * 1000.0
    cpu_ms = (time.process_time() - cpu_before) * 1000.0
    ru_after = resource.getrusage(resource.RUSAGE_SELF)
    return Run(
        wall_ms=wall_ms,
        cpu_ms=cpu_ms,
        peak_threads=peak,
        major_pf=ru_after.ru_majflt - ru_before.ru_majflt,
        minor_pf=ru_after.ru_minflt - ru_before.ru_minflt,
    )


def bench(
    fn: Callable[[], Any],
    *,
    label: str = "",
    n_iter: int = 5,
    n_warmup: int = 2,
    drop_cache_paths: list[Path] | None = None,
) -> RichResult:
    """Time ``fn`` over ``n_iter`` runs after ``n_warmup`` throwaways."""
    result = RichResult(label=label)
    for _ in range(n_warmup):
        fn()
    for _ in range(n_iter):
        result.runs.append(time_call(fn, drop_cache_paths=drop_cache_paths))
    return result


def bench_interleaved(
    labels: list[str], fns: list[Callable[[], Any]], *, n_iter: int = 5, n_warmup: int = 2
) -> list[RichResult]:
    """A/B/A/B-style timing of several functions; returns one result per fn.

    Interleaving keeps page-cache and scheduler state comparable across the
    variants; running A...A then B...B confounds the comparison.
    """
    results = [RichResult(label=label) for label in labels]
    for fn in fns:
        for _ in range(n_warmup):
            fn()
    for _ in range(n_iter):
        for fn, result in zip(fns, results, strict=True):
            result.runs.append(time_call(fn))
    return results


# ---- Script driver and store context managers -------------------------------


def run_script(
    *,
    correctness: Callable[[], Any] | None = None,
    bench: Callable[[int], Any] | None = None,
    default_repeat: int = 5,
    skip_correctness_flag: bool = False,
    description: str = "",
) -> None:
    """Standard ``main()`` for a single-purpose check script.

    Owns the ``--repeat`` / ``--skip-bench`` (and optional
    ``--skip-correctness``) parser, runs the correctness gate, then the
    benchmark unless skipped. Replaces the near-identical hand-written
    ``main()`` in each ``check_*.py``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repeat", type=int, default=default_repeat)
    if skip_correctness_flag:
        parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()
    if correctness is not None and not getattr(args, "skip_correctness", False):
        correctness()
    if bench is not None and not args.skip_bench:
        bench(args.repeat)
