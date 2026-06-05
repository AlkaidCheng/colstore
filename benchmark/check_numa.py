"""Robust benchmark for the NUMA interleave optimization.

Two policies head-to-head on the same process:

  ``local``      -- no-op; pages fall under the kernel's default
                    first-touch policy. Mimics colstore behavior
                    before the NUMA work.
  ``interleave`` -- apply ``MPOL_INTERLEAVE`` at open time. Page-
                    cache pages distribute across NUMA nodes as they
                    fault in.

For each scenario we report:

  wall    : best-of-N wall-clock time (ms)
  cpu     : best-of-N process CPU time (user + sys) (ms)
  ratio   : cpu / wall (utilization; > 1.0 proves real parallelism)
  threads : peak active thread count observed during the run
  faults  : (major, minor) page-fault delta -- major = disk read,
            minor = page-table walk. Watch the minor count: under
            interleave it should drop because pages stop bouncing
            through one node's TLB.

Runs are A/B INTERLEAVED across rounds rather than A...A then B...B.
Separate runs in separate batches see different page-cache state
and that confounds the comparison; interleaving keeps both policies
operating on the same warm pages.

The cold variant explicitly drops the page cache via
``posix_fadvise(DONTNEED)`` before each timed call, so that the
INTERLEAVE policy is exercised on first-fault pages -- which is the
case it actually affects. Warm-cache runs measure steady-state.

Run with ``PYTHONPATH=src`` after building the extension into
``src/colstore/``.
"""

from __future__ import annotations

import ctypes
import os
import resource
import statistics
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import colstore
from colstore import _numa, config


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
            f"wall={best.wall_ms:8.2f}ms  cpu={best.cpu_ms:8.1f}ms  "
            f"ratio={ratio:5.2f}x  threads={int(med_threads):2d}  "
            f"pf={int(med_major)}/{int(med_minor)}"
        )


def drop_pagecache_softly(paths: list[Path]) -> None:
    """Evict file pages via posix_fadvise(DONTNEED). No root needed."""
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


def make_store(td: str, name: str, n_rows: int, n_cols: int, dtype) -> Path:
    """Materialize a store with `n_cols` columns of `n_rows` rows."""
    path = Path(td) / name
    arr = np.arange(n_rows, dtype=dtype)
    cols = {f"c{i}": arr for i in range(n_cols)}
    colstore.store(cols, str(path), show_progress=False)
    return path


def bench_interleaved_policies(
    label_local: str,
    label_interleave: str,
    builder,
    *,
    drop_cache_paths: list[Path] | None = None,
    n_iter: int = 5,
    n_warmup: int = 2,
) -> tuple[Result, Result]:
    """Run two policies head-to-head, interleaved.

    ``builder`` is a zero-arg callable that returns a callable to time.
    A fresh store is opened *inside* the builder for each policy so
    the policy applied at open time takes effect on first-fault.
    """
    res_local = Result(label=label_local)
    res_inter = Result(label=label_interleave)
    # warmup
    for _ in range(n_warmup):
        with policy_scope("local"):
            builder()()
        with policy_scope("interleave"):
            builder()()
    for _ in range(n_iter):
        with policy_scope("local"):
            fn = builder()
            res_local.runs.append(time_call(fn, drop_cache_paths=drop_cache_paths))
        with policy_scope("interleave"):
            fn = builder()
            res_inter.runs.append(time_call(fn, drop_cache_paths=drop_cache_paths))
    return res_local, res_inter


@contextmanager
def policy_scope(policy):
    previous = config.get_numa_policy()
    config.set_numa_policy(policy)
    try:
        yield
    finally:
        config.set_numa_policy(previous)


def banner(s):
    print(f"\n=== {s} ===")


def main():
    print("Environment:")
    print(f"  os.cpu_count()              = {os.cpu_count()}")
    print(f"  config.get_max_workers()    = {config.get_max_workers()}")
    print(f"  config.get_gather_thread_cap() = {config.get_gather_thread_cap()}")
    print(f"  _numa.is_available()        = {_numa.is_available()}")
    print(f"  _numa.allowed_nodes()       = {_numa.allowed_nodes()}")
    if not _numa.is_available():
        print()
        print("  NOTE: NUMA interleave is a no-op on this host. The benchmark")
        print("  will still run and exercise the code paths, but A/B numbers")
        print("  will be within noise of each other. To see the real win,")
        print("  run on a multi-socket / multi-NPS Linux server.")

    with tempfile.TemporaryDirectory() as td:
        # ---- Scenario A: the hot case ---------------------------------------
        # 50 cols x 2.5M rows float64 = 1 GB. Maximum NUMA pressure: many
        # workers, big working set, all-to-one funnel if pages concentrate.
        many = make_store(td, "many.cstore", 2_500_000, 50, np.float64)
        per_col = 2_500_000 * 8
        total = per_col * 50

        def builder_many_dict():
            ds = colstore.open(str(many))
            return lambda: (ds.dict(), ds.close())

        def builder_many_frame():
            ds = colstore.open(str(many))
            return lambda: (ds.frame(), ds.close())

        banner(f"MANY COLS warm 50 x {per_col / 1e6:.0f} MB = {total / 1e9:.1f} GB  ds.dict()")
        res_l, res_i = bench_interleaved_policies(
            "policy=local      ds.dict()",
            "policy=interleave ds.dict()",
            builder_many_dict,
        )
        print(res_l.report())
        print(res_i.report())

        banner(f"MANY COLS warm 50 x {per_col / 1e6:.0f} MB = {total / 1e9:.1f} GB  ds.frame()")
        res_l, res_i = bench_interleaved_policies(
            "policy=local      ds.frame()",
            "policy=interleave ds.frame()",
            builder_many_frame,
        )
        print(res_l.report())
        print(res_i.report())

        banner(f"MANY COLS cold 50 x {per_col / 1e6:.0f} MB = {total / 1e9:.1f} GB  ds.dict()")
        res_l, res_i = bench_interleaved_policies(
            "policy=local      ds.dict() cold",
            "policy=interleave ds.dict() cold",
            builder_many_dict,
            drop_cache_paths=[many],
            n_warmup=0,
            n_iter=3,
        )
        print(res_l.report())
        print(res_i.report())

        # ---- Scenario B: single column, large ------------------------------
        # 1 column x 10M rows float64 = 80 MB. Single-column reads stay
        # within one node's L3 on EPYC parts (~64 MiB per CCD pair) plus
        # some DRAM, so NUMA effects are modest. Expect ~1.0-1.2x.
        single = make_store(td, "single.cstore", 10_000_000, 1, np.float64)
        bytes_total = 10_000_000 * 8

        def builder_single():
            ds = colstore.open(str(single))
            return lambda: (ds.dict(), ds.close())

        banner(f"SINGLE COL warm-cache: 1 x {bytes_total / 1e6:.0f} MB  ds.dict()")
        res_l, res_i = bench_interleaved_policies(
            "policy=local      ds.dict()",
            "policy=interleave ds.dict()",
            builder_single,
        )
        print(res_l.report())
        print(res_i.report())

        # ---- Scenario C: wide / many small columns -------------------------
        # 200 cols x 100K rows float64 = 160 MB. Wide stores fit in
        # aggregate L3 on EPYC parts; expect roughly noise.
        wide = make_store(td, "wide.cstore", 100_000, 200, np.float64)

        def builder_wide():
            ds = colstore.open(str(wide))
            return lambda: (ds.dict(), ds.close())

        banner("WIDE STORE warm-cache: 200 x 0.8 MB = 160 MB  ds.dict()")
        res_l, res_i = bench_interleaved_policies(
            "policy=local      ds.dict()",
            "policy=interleave ds.dict()",
            builder_wide,
        )
        print(res_l.report())
        print(res_i.report())

        # ---- Scenario D: low-concurrency regression check ------------------
        # The case where interleave is suspected to be pessimal: one
        # consumer thread reading a large store. With pages spread, that
        # one thread does mostly-remote loads. Expect interleave slightly
        # slower or within noise; this is what justifies the "local"
        # opt-out documented in set_numa_policy.
        banner("LOW-CONCURRENCY: workers=1, 1 GB / 50-col  ds.dict()")
        previous_workers = config.get_max_workers()
        previous_cap = config.get_gather_thread_cap()
        try:
            config.set_max_workers(1)
            config.set_gather_thread_cap(1)
            res_l, res_i = bench_interleaved_policies(
                "policy=local      workers=1",
                "policy=interleave workers=1",
                builder_many_dict,
            )
            print(res_l.report())
            print(res_i.report())
        finally:
            config.set_max_workers(previous_workers)
            config.set_gather_thread_cap(previous_cap)


if __name__ == "__main__":
    if sys.platform != "linux":
        print("This benchmark requires Linux (NUMA syscalls). Exiting.")
        sys.exit(0)
    main()
