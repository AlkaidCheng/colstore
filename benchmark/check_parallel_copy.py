"""Robust benchmark for ColStoreReader contiguous-copy parallelism.

For each scenario we report:

  wall    : best-of-N wall-clock time (ms)
  cpu     : best-of-N process CPU time (user + sys) (ms)
  ratio   : cpu / wall (utilization; values > 1.0 prove real parallelism)
  threads : peak active thread count observed during the run
  faults  : (major, minor) page-fault delta -- major = disk read, minor = page table
  drop_cache : if True, drops the page cache before the timed iterations (cold)

Cold runs (drop_cache=True) require sudo or CAP_SYS_ADMIN; if unavailable we
fall back to a vmtouch-style madvise(DONTNEED) self-eviction.

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

import colstore
from colstore import config


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
        # Use best-of-N for wall and cpu (lowest noise floor); medians for the rest.
        best = min(self.runs, key=lambda r: r.wall_ms)
        med_threads = statistics.median(r.peak_threads for r in self.runs)
        med_major = statistics.median(r.major_pf for r in self.runs)
        med_minor = statistics.median(r.minor_pf for r in self.runs)
        ratio = best.cpu_ms / best.wall_ms if best.wall_ms > 0 else float("nan")
        return (
            f"  {self.label:<60} "
            f"wall={best.wall_ms:7.1f}ms  cpu={best.cpu_ms:7.1f}ms  "
            f"ratio={ratio:5.2f}x  threads={int(med_threads):2d}  "
            f"pf={int(med_major)}/{int(med_minor)}"
        )


def drop_pagecache_softly(paths: list[Path]) -> None:
    """Evict file pages from the kernel page cache via posix_fadvise(DONTNEED).

    This is best-effort; it requires the file to be opened. Works without
    root privileges.
    """
    POSIX_FADV_DONTNEED = 4
    libc = ctypes.CDLL("libc.so.6")
    for path in paths:
        # Walk into the cstore dir if path is a directory.
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
    """Context manager that tracks the peak active thread count.

    Polls threading.active_count() at `interval_s` resolution. Returns a
    callable that yields the peak observed so far.
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


def time_call(fn, *, drop_cache_paths: list[Path] | None = None) -> Run:
    """Time one call of `fn`. Optionally evict page cache beforehand."""
    if drop_cache_paths:
        drop_pagecache_softly(drop_cache_paths)
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


def bench(label, fn, n_iter=5, *, n_warmup=2, drop_cache_paths=None) -> Result:
    result = Result(label=label)
    for _ in range(n_warmup):
        fn()
    for _ in range(n_iter):
        result.runs.append(time_call(fn, drop_cache_paths=drop_cache_paths))
    return result


def make_store(td: str, name: str, n_rows: int, n_cols: int, dtype) -> Path:
    """Materialize a store with `n_cols` columns of `n_rows` rows."""
    path = Path(td) / name
    arr = np.arange(n_rows, dtype=dtype)
    cols = {f"c{i}": arr for i in range(n_cols)}
    colstore.store(cols, str(path), show_progress=False)
    return path


def banner(s):
    print(f"\n=== {s} ===")


def main():
    print("Environment:")
    print(f"  os.cpu_count()              = {os.cpu_count()}")
    print(f"  config.get_max_workers()    = {config.get_max_workers()}")
    print(f"  config.get_gather_thread_cap() = {config.get_gather_thread_cap()}")

    with tempfile.TemporaryDirectory() as td:
        # --- Scenario A: 1 big column (~80 MB). The sweet spot for parallel
        #     copy: no column-pool concurrency, just a single big memcpy.
        single = make_store(td, "single.cstore", 10_000_000, 1, np.float64)
        col_bytes = 10_000_000 * 8
        banner(f"SINGLE COLUMN, {col_bytes/1e6:.0f} MB (warm cache)")
        ds = colstore.open(str(single))
        results = [
            bench("ds.dict()                              (new shortcut)", lambda: ds.dict()),
            bench("ds[:].dict()                           (via view)", lambda: ds[:].dict()),
            bench(
                "ds[:, 'c0'].array()                   (single column view)",
                lambda: ds[:, "c0"].array(),
            ),
        ]
        for r in results:
            print(r.report())
        ds.close()

        banner(f"SINGLE COLUMN, {col_bytes/1e6:.0f} MB (cold cache)")
        ds = colstore.open(str(single))
        results = [
            bench(
                "ds.dict()                              (cold)",
                lambda: ds.dict(),
                n_warmup=0,
                n_iter=3,
                drop_cache_paths=[single],
            ),
        ]
        for r in results:
            print(r.report())
        ds.close()

        # --- Scenario B: many columns (~50 cols x ~20 MB each = 1 GB).
        #     This is the regime where the column pool dominates and per-column
        #     parallel-copy is most likely to harm rather than help.
        many = make_store(td, "many.cstore", 2_500_000, 50, np.float64)
        total = 2_500_000 * 50 * 8
        per_col = 2_500_000 * 8
        banner(f"MANY COLUMNS, 50 x {per_col/1e6:.0f} MB = {total/1e9:.1f} GB (warm)")
        ds = colstore.open(str(many))
        results = [
            bench("ds.dict()                              (new shortcut)", lambda: ds.dict()),
            bench("ds[:].dict()                           (via view)", lambda: ds[:].dict()),
            bench("ds.frame()                             (via shortcut)", lambda: ds.frame()),
        ]
        for r in results:
            print(r.report())
        ds.close()

        # --- Scenario C: a really wide store (200 small cols).
        #     Mimics shape of the user's 198-col workload at smaller scale.
        wide = make_store(td, "wide.cstore", 200_000, 200, np.float64)
        total = 200_000 * 200 * 8
        per_col = 200_000 * 8
        banner(f"WIDE STORE, 200 x {per_col/1e6:.1f} MB = {total/1e6:.0f} MB (warm)")
        ds = colstore.open(str(wide))
        results = [
            bench("ds.dict()                              (new shortcut)", lambda: ds.dict()),
            bench("ds[:].dict()                           (via view)", lambda: ds[:].dict()),
        ]
        for r in results:
            print(r.report())
        ds.close()

        # --- Scenario D: above-threshold slice (parallel-copy active path).
        big_slice = make_store(td, "slice.cstore", 5_000_000, 1, np.float64)
        col_bytes = 5_000_000 * 8
        banner(f"BIG STEP-1 SLICE, {col_bytes/1e6:.0f} MB (warm)")
        ds = colstore.open(str(big_slice))
        results = [
            bench(
                "ds[100_000:4_900_000, 'c0'].array()   (37 MiB step-1)",
                lambda: ds[100_000:4_900_000, "c0"].array(),
            ),
            bench(
                "ds[::2, 'c0'].array()                  (step-2 strided)",
                lambda: ds[::2, "c0"].array(),
            ),
        ]
        for r in results:
            print(r.report())
        ds.close()

        # --- Scenario E: FORCE the parallel-copy path even on a 1-CPU sandbox
        #     by raising gather_thread_cap. ratio > 1.0 would prove the
        #     ThreadPoolExecutor inside _parallel_contiguous_copy is actually
        #     running chunks concurrently; ratio == 1.0 means the GIL or
        #     scheduling is serializing them; ratio < 1.0 means we're paying
        #     pool overhead for no benefit.
        for forced_cap in (1, 2, 4, 8):
            config.set_gather_thread_cap(forced_cap)
            config.set_max_workers(1)  # keep _gather_many on the serial branch
            banner(
                f"FORCED gather_thread_cap={forced_cap}, max_workers=1 "
                f"(exercise inner ThreadPool)"
            )
            ds = colstore.open(str(single))
            results = [
                bench(f"ds.dict()  80MB 1-col  cap={forced_cap}", lambda ds=ds: ds.dict()),
            ]
            for r in results:
                print(r.report())
            ds.close()

        # --- Scenario F: nested pools. max_workers > 1 with a multi-col
        #     store: column pool active, AND per-column parallel-copy active.
        #     This is the configuration most likely to oversubscribe.
        for forced_workers, forced_cap in [(1, 1), (2, 2), (4, 4), (8, 8), (4, 16)]:
            config.set_max_workers(forced_workers)
            config.set_gather_thread_cap(forced_cap)
            banner(
                f"FORCED max_workers={forced_workers}, "
                f"gather_thread_cap={forced_cap} (nested-pool stress)"
            )
            ds = colstore.open(str(many))
            results = [
                bench(
                    f"ds.dict()  1GB 50-col  workers={forced_workers} cap={forced_cap}",
                    lambda ds=ds: ds.dict(),
                ),
            ]
            for r in results:
                print(r.report())
            ds.close()


if __name__ == "__main__":
    main()
