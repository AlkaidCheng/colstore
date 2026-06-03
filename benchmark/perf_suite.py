"""Performance regression suite for colstore.

A curated, deterministic, machine-readable benchmark intended for two uses:

1. **Routine check** -- run after a change to confirm nothing got slower in a
   visible way. The summary table shows median time, min, and p95 for each
   workload; eyeballing it catches gross regressions.

2. **Baseline comparison** -- capture a baseline with ``--output base.json``,
   make a change, run again, and use ``--compare base.json`` to see the per-
   workload delta. Cells outside ``--noise-band`` (default 10%) are flagged.

The matrix is intentionally small (~36 cells, <60s on a typical box) so it is
cheap to run frequently. The trade-off is that it covers representative
points, not the full configuration space; for one-off investigations, the
other benchmark scripts (``profile_gather.py``, ``find_kernel_threshold.py``,
``check_kernel_threading.py``) drill into specific questions in depth.

What this suite does:

* Warms up each cell before timing (discards the first run).
* Runs each cell N times (default 7), reporting median + min + p95. Median is
  the primary number because it's more representative than min on a noisy
  multi-tenant machine.
* Captures a machine fingerprint (OS, CPU count, NumPy version, OpenMP max,
  current ``gather_thread_cap``) and the colstore git SHA if available, so
  comparisons stay honest across commits and across hardware.
* Emits JSON to ``--output`` and a human-readable table to stdout.

What it does NOT do:

* Cold-cache measurements. Use ``profile_gather.py --cold`` for those; mixing
  cache state into a regression suite adds variance that drowns out signal.
* Hardware counter capture. That is what ``run_perf.sh`` is for.
* Fail the build on regression. ``--compare`` *reports* deltas, exits zero.
  Noise on memory-bound workloads is real; trust the operator, not the CI.
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
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

import colstore
from colstore import ColStoreReader, cpp_available, max_threads
from colstore.config import get_gather_thread_cap

# ---------------------------------------------------------------------------
# Workload matrix
# ---------------------------------------------------------------------------
#
# Each cell is a (workload, backend) pair. Workloads are picked to exercise
# distinct code paths and size regimes, not to enumerate every combination.

# Sizes chosen to span the three regimes the dispatcher hits:
#   - Small (1k indices): per-call dispatch / allocation cost dominates.
#   - Medium (100k): kernel body dominates, single-threaded.
#   - Large (10M): kernel body dominates, parallel cap engaged.
SIZES = {"small": 1_000, "medium": 100_000, "large": 10_000_000}

# Index patterns. Sorted exercises near-sequential access (hardware prefetcher
# dominates); unsorted exercises scattered access (software prefetch wins).
PATTERNS = ("sorted", "unsorted")

# Backends exercised in the default matrix. Numba is opt-in via --include-numba
# because its JIT warmup adds variance that hurts regression detection.
DEFAULT_BACKENDS = ("cpp", "numpy")

# Default store shape. Chosen to fit in memory on any developer machine (~80MB
# for float32) while being big enough that 10M gathers exercise real work.
STORE_ROWS = 20_000_000
STORE_COLS = 8


@dataclass
class CellResult:
    """One workload-x-backend timing result."""

    workload: str
    backend: str
    size_label: str
    n_indices: int
    pattern: str | None
    median_ms: float
    min_ms: float
    p95_ms: float
    repeats: int


@dataclass
class SuiteResult:
    """Full suite output: machine info plus all cells."""

    fingerprint: dict[str, Any]
    cells: list[CellResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _time_fn(fn: Any, repeats: int) -> tuple[float, float, float]:
    """Return (median_ms, min_ms, p95_ms) over ``repeats`` runs.

    One warmup run is discarded. Each timed run is single-shot; we do not
    use ``timeit`` because for the ~ms-to-100ms work the wall clock is plenty
    precise and we want to keep each cell to one Python call so any leaked
    state shows up rather than hiding.
    """
    fn()  # warmup; discarded
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    median = statistics.median(samples)
    # statistics.quantiles with n=20 gives 5%, 10%, ..., 95% -- the 18th
    # element (index 18) is the 95th percentile cut point.
    p95 = statistics.quantiles(samples, n=20)[18] if len(samples) >= 20 else max(samples)
    return median, min(samples), p95


# ---------------------------------------------------------------------------
# Workload definitions
# ---------------------------------------------------------------------------
#
# Each workload returns a thunk (zero-arg callable) suitable for ``_time_fn``.
# Workloads are responsible for any setup (allocating output buffers, sorting
# indices); they should NOT include any first-touch cost in the timed region.


def _make_store(path: str, dtype: np.dtype) -> ColStoreReader:
    """Create (or reuse) the synthetic store. Matches profile_gather's validator."""
    requested = np.dtype(dtype)
    if os.path.exists(path):
        try:
            from colstore import format as format_module

            manifest, _ = format_module.read_header(path)
            existing_cols = manifest["columns"]
            shape_matches = (
                manifest["n_rows"] == STORE_ROWS
                and len(existing_cols) == STORE_COLS
                and all(np.dtype(c["dtype"]).str == requested.str for c in existing_cols)
            )
        except Exception:
            shape_matches = False
        if not shape_matches:
            os.remove(path)
    if not os.path.exists(path):
        rng = np.random.default_rng(0)
        columns = {
            f"f{i}": rng.standard_normal(STORE_ROWS).astype(requested) for i in range(STORE_COLS)
        }
        colstore.store(columns, path, show_progress=False).close()
    return ColStoreReader(path, backend="cpp")  # backend overridden per-workload below


def _indices(n: int, pattern: str, max_row: int) -> np.ndarray:
    """Deterministic indices of length ``n`` with the requested access pattern."""
    rng = np.random.default_rng(seed=hash((n, pattern, max_row)) & 0xFFFFFFFF)
    if pattern == "sorted":
        return np.sort(rng.choice(max_row, size=n, replace=False)).astype(np.int64)
    return rng.permutation(max_row)[:n].astype(np.int64)


def _build_workloads(
    store_path: str, dtype: np.dtype, backends: list[str]
) -> list[tuple[str, str, str, int, str | None, Any]]:
    """Return a list of (workload, backend, size_label, n, pattern, thunk)."""

    out: list[tuple[str, str, str, int, str | None, Any]] = []

    # --- gather: fancy-index one column. The main hot path. ---
    for backend in backends:
        store = ColStoreReader(store_path, backend=backend)
        for size_label, n in SIZES.items():
            for pattern in PATTERNS:
                idx = _indices(n, pattern, STORE_ROWS)

                # Capture store + idx by default-arg trick so each closure
                # binds the correct values.
                def thunk(s=store, i=idx):
                    return s[i, "f0"].array()

                out.append(("gather", backend, size_label, n, pattern, thunk))

    # --- dict: multi-column gather, exercises _gather_many's budget split. ---
    for backend in backends:
        store = ColStoreReader(store_path, backend=backend)
        all_cols = store.columns
        for size_label, n in SIZES.items():
            # One pattern (sorted) is enough for dict; varying both axes
            # quadruples runtime for marginal regression-detection value.
            idx = _indices(n, "sorted", STORE_ROWS)

            def thunk(s=store, i=idx, cols=all_cols):
                return s[i, cols].dict()

            out.append(("dict", backend, size_label, n, "sorted", thunk))

    # --- slice read: contiguous range, exercises memmap + copy path. ---
    for backend in backends:
        store = ColStoreReader(store_path, backend=backend)
        for size_label, n in SIZES.items():
            sl = slice(0, n)

            def thunk(s=store, _sl=sl):
                return s[_sl, "f0"].array()

            out.append(("slice", backend, size_label, n, None, thunk))

    return out


# ---------------------------------------------------------------------------
# Machine fingerprint
# ---------------------------------------------------------------------------


def _git_sha() -> str | None:
    """Return the current git SHA if we are in a checkout, else None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _fingerprint() -> dict[str, Any]:
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
        "gather_thread_cap": get_gather_thread_cap(),
        "cpp_available": cpp_available(),
        "git_sha": _git_sha(),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


# ---------------------------------------------------------------------------
# Suite driver
# ---------------------------------------------------------------------------


def run_suite(store_path: str, dtype: np.dtype, repeats: int, backends: list[str]) -> SuiteResult:
    """Run the full matrix and return a structured result."""
    _make_store(store_path, dtype).close()
    cells: list[CellResult] = []
    workloads = _build_workloads(store_path, dtype, backends)
    print(
        f"Running {len(workloads)} cells, {repeats} repeats each (~"
        f"{len(workloads) * (repeats + 1) // 60 + 1} min upper bound)...",
        file=sys.stderr,
    )
    for i, (workload, backend, size_label, n, pattern, thunk) in enumerate(workloads, 1):
        median, lo, p95 = _time_fn(thunk, repeats)
        cells.append(
            CellResult(
                workload=workload,
                backend=backend,
                size_label=size_label,
                n_indices=n,
                pattern=pattern,
                median_ms=median,
                min_ms=lo,
                p95_ms=p95,
                repeats=repeats,
            )
        )
        pat = pattern or "-"
        print(
            f"  [{i:>2}/{len(workloads)}] {workload:<8} {backend:<6} "
            f"{size_label:<6} pat={pat:<8} n={n:>10,}  "
            f"median={median:>9.3f} ms  min={lo:>9.3f}  p95={p95:>9.3f}",
            file=sys.stderr,
        )
    return SuiteResult(fingerprint=_fingerprint(), cells=cells)


# ---------------------------------------------------------------------------
# Output / comparison
# ---------------------------------------------------------------------------


def _print_table(result: SuiteResult) -> None:
    fp = result.fingerprint
    print()
    print(
        f"Machine: {fp['hostname']} | {fp['processor']} | "
        f"phys={fp['cpu_count_physical']} log={fp['cpu_count_logical']} | "
        f"OMP_max={fp['openmp_max_threads']} cap={fp['gather_thread_cap']}"
    )
    print(
        f"Versions: python {fp['python_version']} | numpy {fp['numpy_version']} | "
        f"colstore {fp['colstore_version']} | git {fp['git_sha'] or 'unknown'}"
    )
    print()
    header = (
        f"{'workload':<10} {'backend':<7} {'size':<7} {'pattern':<9} "
        f"{'n_indices':>10}  {'median ms':>10}  {'min ms':>10}  {'p95 ms':>10}"
    )
    print(header)
    print("-" * len(header))
    for c in result.cells:
        pattern = c.pattern or "-"
        print(
            f"{c.workload:<10} {c.backend:<7} {c.size_label:<7} {pattern:<9} "
            f"{c.n_indices:>10,}  {c.median_ms:>10.3f}  {c.min_ms:>10.3f}  {c.p95_ms:>10.3f}"
        )


def _compare(current: SuiteResult, baseline_path: str, noise_band: float) -> int:
    """Compare current run against a saved baseline; return regression count."""
    with open(baseline_path) as f:
        baseline_raw = json.load(f)
    baseline = {
        (c["workload"], c["backend"], c["size_label"], c["pattern"]): c
        for c in baseline_raw["cells"]
    }
    print()
    print(f"Comparison vs {baseline_path}")
    print(
        f"Baseline machine: {baseline_raw['fingerprint']['hostname']} @ "
        f"{baseline_raw['fingerprint']['captured_at']} (git "
        f"{baseline_raw['fingerprint'].get('git_sha', 'unknown')})"
    )
    if baseline_raw["fingerprint"].get("hostname") != current.fingerprint.get("hostname"):
        print("WARNING: baseline is from a different host; comparison may be noisy.")
    if baseline_raw["fingerprint"].get("cpu_count_physical") != current.fingerprint.get(
        "cpu_count_physical"
    ):
        print("WARNING: baseline has a different CPU count; comparison may be noisy.")
    print()
    header = (
        f"{'workload':<10} {'backend':<7} {'size':<7} {'pattern':<9} "
        f"{'base ms':>10}  {'curr ms':>10}  {'delta':>9}  flag"
    )
    print(header)
    print("-" * len(header))
    regressions = 0
    for c in current.cells:
        key = (c.workload, c.backend, c.size_label, c.pattern)
        base = baseline.get(key)
        if base is None:
            print(
                f"{c.workload:<10} {c.backend:<7} {c.size_label:<7} "
                f"{c.pattern or '-':<9} {'(no base)':>10}  {c.median_ms:>10.3f}  "
                f"{'?':>9}  NEW CELL"
            )
            continue
        ratio = c.median_ms / base["median_ms"] if base["median_ms"] > 0 else float("inf")
        delta_pct = (ratio - 1.0) * 100.0
        flag = ""
        if ratio > 1.0 + noise_band:
            flag = "REGRESSION"
            regressions += 1
        elif ratio < 1.0 - noise_band:
            flag = "IMPROVEMENT"
        print(
            f"{c.workload:<10} {c.backend:<7} {c.size_label:<7} "
            f"{c.pattern or '-':<9} {base['median_ms']:>10.3f}  "
            f"{c.median_ms:>10.3f}  {delta_pct:>+8.1f}%  {flag}"
        )
    print()
    print(f"Summary: {regressions} cell(s) regressed beyond +{noise_band * 100:.0f}% noise band.")
    return regressions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store-path",
        default="/tmp/perf_suite.cstore",
        help="Path to (re)use for the synthetic store.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=7,
        help="Timed iterations per cell after one warmup (default 7).",
    )
    parser.add_argument("--dtype", default="float32", help="Source dtype (default float32).")
    parser.add_argument("--output", help="Write structured JSON results to this path.")
    parser.add_argument(
        "--compare",
        metavar="BASELINE_JSON",
        help="Compare this run against a previously-captured baseline.",
    )
    parser.add_argument(
        "--noise-band",
        type=float,
        default=0.10,
        help="Tolerance for the comparison; relative deltas beyond this count "
        "as regressions or improvements (default 0.10 = 10%%).",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(DEFAULT_BACKENDS),
        choices=["cpp", "numpy"],
        help="Backends to include in the matrix (default: cpp, numpy).",
    )
    args = parser.parse_args()

    if not cpp_available() and "cpp" in args.backends:
        print(
            "WARNING: C++ extension not built; cpp cells will fall back to numpy.",
            file=sys.stderr,
        )

    result = run_suite(args.store_path, np.dtype(args.dtype), args.repeats, args.backends)
    _print_table(result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "fingerprint": result.fingerprint,
                    "cells": [asdict(c) for c in result.cells],
                },
                f,
                indent=2,
            )
        print(f"\nWrote JSON results to {args.output}")

    if args.compare:
        _compare(result, args.compare, args.noise_band)


if __name__ == "__main__":
    main()
