"""Robust benchmark for ColStoreReader frame() construction.

Compares the no-consolidate construction path against pandas' default
consolidating constructor head-to-head. Both paths consume the same
``store.dict()`` output; the only thing being timed is the
DataFrame construction step.

For each scenario we report:

  wall    : best-of-N wall-clock time (ms)
  cpu     : best-of-N process CPU time (user + sys) (ms)
  ratio   : cpu / wall (utilization; >1.0 proves real parallelism)
  threads : peak active thread count observed during the run
  faults  : (major, minor) page-fault delta

The construction step itself is single-threaded, so the interesting
quantity is wall time. The thread/CPU/fault columns are kept for
parity with check_parallel_copy.py and to confirm we are not
accidentally introducing background work.

Runs are interleaved A/B/A/B across rounds rather than A...A then
B...B, because separate runs in separate batches see different
page-cache and scheduler state and that confounds the comparison.

Run with PYTHONPATH=src and an extension built into src/colstore/.
"""

from __future__ import annotations

import ctypes
import os
import resource
import statistics
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import colstore
from colstore.reader import _make_dataframe_no_consolidate


@dataclass
class Run:
    wall_ms: float
    cpu_ms: float
    peak_threads: int
    major_pf: int
    minor_pf: int


@dataclass
class Result:
    label: str
    runs: list[Run] = field(default_factory=list)

    def report(self) -> str:
        if not self.runs:
            return f"  {self.label:<60}  (no runs)"
        best = min(self.runs, key=lambda r: r.wall_ms)
        med_threads = statistics.median(r.peak_threads for r in self.runs)
        med_major = statistics.median(r.major_pf for r in self.runs)
        med_minor = statistics.median(r.minor_pf for r in self.runs)
        ratio = best.cpu_ms / best.wall_ms if best.wall_ms > 0 else float("nan")
        return (
            f"  {self.label:<60} "
            f"wall={best.wall_ms:8.2f}ms  cpu={best.cpu_ms:7.1f}ms  "
            f"ratio={ratio:5.2f}x  threads={int(med_threads):2d}  "
            f"pf={int(med_major)}/{int(med_minor)}"
        )


def drop_pagecache_softly(paths: list[Path]) -> None:
    """Evict file pages via posix_fadvise(DONTNEED) (no root needed)."""
    POSIX_FADV_DONTNEED = 4
    libc = ctypes.CDLL("libc.so.6")
    for path in paths:
        if path.is_dir():
            for sub in path.iterdir():
                if sub.is_file():
                    fd = os.open(str(sub), os.O_RDONLY)
                    try:
                        libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
        else:
            fd = os.open(str(path), os.O_RDONLY)
            try:
                libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


@contextmanager
def thread_watcher(interval_s: float = 0.001):
    """Context manager that tracks the peak active thread count."""
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


def time_call(fn) -> Run:
    """Time one call of `fn` and capture process metrics."""
    ru_before = resource.getrusage(resource.RUSAGE_SELF)
    cpu_before = time.process_time()
    wall_before = time.perf_counter()
    with thread_watcher() as peak_fn:
        fn()
        peak = peak_fn()
    wall_ms = (time.perf_counter() - wall_before) * 1000
    cpu_ms = (time.process_time() - cpu_before) * 1000
    ru_after = resource.getrusage(resource.RUSAGE_SELF)
    return Run(
        wall_ms=wall_ms,
        cpu_ms=cpu_ms,
        peak_threads=peak,
        major_pf=ru_after.ru_majflt - ru_before.ru_majflt,
        minor_pf=ru_after.ru_minflt - ru_before.ru_minflt,
    )


def bench_interleaved(
    labels: list[str], fns: list, n_iter: int = 5, n_warmup: int = 2
) -> list[Result]:
    """Run several functions A/B/A/B-style and return per-fn Result lists.

    Interleaving keeps page-cache and scheduler state comparable across
    the variants; running A...A then B...B confounds the comparison.
    """
    results = [Result(label=label) for label in labels]
    for fn in fns:
        for _ in range(n_warmup):
            fn()
    for _ in range(n_iter):
        for fn, result in zip(fns, results, strict=True):
            result.runs.append(time_call(fn))
    return results


def make_store(td: str, name: str, n_rows: int, n_cols: int, dtypes) -> Path:
    """Materialize a store with `n_cols` columns cycling through `dtypes`."""
    path = Path(td) / name
    rng = np.random.default_rng(0)
    columns = {}
    for i in range(n_cols):
        dtype = dtypes[i % len(dtypes)]
        if np.issubdtype(dtype, np.floating):
            arr = rng.standard_normal(n_rows).astype(dtype)
        else:
            arr = rng.integers(0, 10_000, size=n_rows, dtype=dtype)
        columns[f"c{i:03d}"] = arr
    colstore.store(columns, str(path), show_progress=False)
    return path


def banner(s):
    print(f"\n=== {s} ===")


def construction_pair(
    columns_dict: dict[str, np.ndarray],
) -> tuple[callable, callable]:
    """Return (baseline_constructor, no_consolidate_constructor) closures.

    Both close over the same already-materialized dict so the timing
    isolates the DataFrame construction step from the gather.
    """
    baseline = lambda: pd.DataFrame(columns_dict)  # noqa: E731
    optimized = lambda: _make_dataframe_no_consolidate(columns_dict)  # noqa: E731
    return baseline, optimized


def main():
    print("Environment:")
    print(f"  os.cpu_count() = {os.cpu_count()}")
    print(f"  pandas         = {pd.__version__}")
    print(f"  numpy          = {np.__version__}")

    with tempfile.TemporaryDirectory() as td:
        # ---- Scenario A: many same-dtype columns ----------------------------
        # 50 float64 columns x 2.5M rows = 1 GB. The maximum-impact case: a
        # consolidating constructor groups all 50 columns into one 2D float64
        # block, which is a 1 GB extra allocation + memcpy on top of the dict
        # that already owns 1 GB of data. The no-consolidate path keeps each
        # column in its own Block.
        many_homog = make_store(td, "homog.cstore", 2_500_000, 50, [np.float64])
        ds = colstore.open(str(many_homog))
        try:
            dict_data = ds.dict()  # do the gather once, time only construction
            total_mb = sum(arr.nbytes for arr in dict_data.values()) / 1e6
            banner(f"SAME-DTYPE: 50 cols x 2.5M rows float64 ({total_mb:.0f} MB)")
            baseline, optimized = construction_pair(dict_data)
            results = bench_interleaved(
                [
                    "pd.DataFrame(dict)                       (baseline)",
                    "_make_dataframe_no_consolidate(dict)     (optimized)",
                ],
                [baseline, optimized],
            )
            for r in results:
                print(r.report())

            # Full end-to-end timing including the gather, since that is what
            # users actually see.
            banner(f"END-TO-END: 50 cols x 2.5M rows float64 ({total_mb:.0f} MB)")
            drop_pagecache_softly([many_homog])
            results = bench_interleaved(
                [
                    "ds.dict()                                (gather only)",
                    "ds.frame()                               (gather + new frame)",
                ],
                [lambda: ds.dict(), lambda: ds.frame()],
                n_warmup=1,
            )
            for r in results:
                print(r.report())
        finally:
            ds.close()

        # ---- Scenario B: mixed dtypes ---------------------------------------
        # 50 cols cycling through 4 dtypes -> 4 consolidated blocks of
        # ~12 cols each. Smaller per-block copies, more blocks, but the total
        # consolidation copy is still on the order of the full data size.
        many_mixed = make_store(
            td,
            "mixed.cstore",
            2_500_000,
            50,
            [np.float64, np.float32, np.int32, np.int64],
        )
        ds = colstore.open(str(many_mixed))
        try:
            dict_data = ds.dict()
            total_mb = sum(arr.nbytes for arr in dict_data.values()) / 1e6
            banner(f"MIXED-DTYPE: 50 cols x 2.5M rows (4 dtypes, {total_mb:.0f} MB)")
            baseline, optimized = construction_pair(dict_data)
            results = bench_interleaved(
                [
                    "pd.DataFrame(dict)                       (baseline)",
                    "_make_dataframe_no_consolidate(dict)     (optimized)",
                ],
                [baseline, optimized],
            )
            for r in results:
                print(r.report())
        finally:
            ds.close()

        # ---- Scenario C: wide store -----------------------------------------
        # 200 cols x 100K rows mixed dtypes. The user's hot workload was
        # 198 cols; this is a smaller version of the same shape.
        wide = make_store(
            td,
            "wide.cstore",
            100_000,
            200,
            [np.float64, np.float32, np.int32, np.int64],
        )
        ds = colstore.open(str(wide))
        try:
            dict_data = ds.dict()
            total_mb = sum(arr.nbytes for arr in dict_data.values()) / 1e6
            banner(f"WIDE: 200 cols x 100K rows ({total_mb:.0f} MB)")
            baseline, optimized = construction_pair(dict_data)
            results = bench_interleaved(
                [
                    "pd.DataFrame(dict)                       (baseline)",
                    "_make_dataframe_no_consolidate(dict)     (optimized)",
                ],
                [baseline, optimized],
            )
            for r in results:
                print(r.report())
        finally:
            ds.close()

        # ---- Scenario D: tiny per-call overhead -----------------------------
        # 1K rows x 50 cols. Must not regress here: the helper's setup cost
        # (Index construction, list copy, pandas private imports) needs to
        # stay small. The previous PR caught a +6 us per-call regression on
        # a tiny dict materialization that this scenario is designed to
        # surface.
        tiny = make_store(td, "tiny.cstore", 1_000, 50, [np.float64])
        ds = colstore.open(str(tiny))
        try:
            dict_data = ds.dict()
            total_kb = sum(arr.nbytes for arr in dict_data.values()) / 1e3
            banner(f"TINY: 50 cols x 1K rows float64 ({total_kb:.0f} KB)")
            baseline, optimized = construction_pair(dict_data)
            results = bench_interleaved(
                [
                    "pd.DataFrame(dict)                       (baseline)",
                    "_make_dataframe_no_consolidate(dict)     (optimized)",
                ],
                [baseline, optimized],
                n_iter=20,
                n_warmup=5,
            )
            for r in results:
                print(r.report())
        finally:
            ds.close()

        # ---- Scenario E: TableView.frame() row-sliced -----------------------
        # The same optimization applies through the view path. ``ds[a:b].frame()``
        # is a common idiom for materializing a row subset; verify it benefits.
        ds = colstore.open(str(many_homog))
        try:
            slice_dict = ds[500_000:2_000_000].dict()
            slice_mb = sum(arr.nbytes for arr in slice_dict.values()) / 1e6
            banner(
                f"SLICED VIEW: ds[500K:2M].frame()  " f"(50 cols x 1.5M rows, {slice_mb:.0f} MB)"
            )
            baseline, optimized = construction_pair(slice_dict)
            results = bench_interleaved(
                [
                    "pd.DataFrame(dict)                       (baseline)",
                    "_make_dataframe_no_consolidate(dict)     (optimized)",
                ],
                [baseline, optimized],
            )
            for r in results:
                print(r.report())
        finally:
            ds.close()


if __name__ == "__main__":
    main()
