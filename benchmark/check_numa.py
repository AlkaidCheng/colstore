"""Robust benchmark for the NUMA optimization (writer + reader sides).

This benchmark is the third revision in the NUMA series. Earlier versions
opened a fresh ColStoreReader inside the timed region, which dragged
~30 ms of per-iteration setup cost into every wall-time measurement and
masked the actual gather delta between policies. This version opens once
per (file, policy) combination, runs warmups, and times only the gather.

Three things this measures:

  1. Writer-side policy A/B: write the SAME data twice -- once under
     ``"local"`` policy (writer thread's mempolicy = MPOL_DEFAULT, kernel
     places page-cache pages by first touch), once under ``"interleave"``
     (writer thread enters ``MPOL_INTERLEAVE`` via set_mempolicy, kernel
     distributes pages across nodes). Then time reads of each file.

     This is the actual win on warm reads -- the original numactl
     experiment changed writer-side placement, not reader-side. With
     the writer-side ``set_mempolicy`` from commit 4, calling
     ``colstore.store`` under ``config.set_numa_policy("auto")``
     (the default) produces files whose page-cache pages are spread
     from the moment they are written.

  2. Reader-side policy A/B (informational): apply ``mbind`` to the
     reader memmap or not, on the SAME file. This will be ~noise on
     warm cache (because the writer-side optimization already spread
     the pages, or because mbind cannot move warm pages) but may
     show a small effect on cold cache.

  3. End-to-end ``ds.dict()`` / ``ds.frame()`` numbers with the
     default config so the user can see steady-state behavior after
     the writer-side change lands.

For each scenario we report:

  wall    : best-of-N wall-clock time (ms)
  cpu     : best-of-N process CPU time (ms)
  ratio   : cpu / wall (utilization)
  threads : peak active thread count
  faults  : (major, minor) page-fault delta

Run with ``PYTHONPATH=src`` after building the extension.
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
            return f"  {self.label:<62}  (no runs)"
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


def bench(fn, *, n_iter=5, n_warmup=2, drop_cache_paths=None) -> Result:
    """Time `fn` over n_iter rounds after n_warmup throwaways."""
    result = Result(label="")
    for _ in range(n_warmup):
        fn()
    for _ in range(n_iter):
        result.runs.append(time_call(fn, drop_cache_paths=drop_cache_paths))
    return result


def labeled(label: str, result: Result) -> Result:
    result.label = label
    return result


@contextmanager
def policy_scope(policy):
    previous = config.get_numa_policy()
    config.set_numa_policy(policy)
    try:
        yield
    finally:
        config.set_numa_policy(previous)


def write_store_under_policy(path: Path, columns: dict, policy: str) -> None:
    """Write a fresh store at `path` with the given NUMA policy active.

    The point of this benchmark is to compare files written under
    different policies; this helper enforces that each variant gets
    a clean write under exactly the policy it claims.
    """
    if path.exists():
        path.unlink()
    with policy_scope(policy):
        colstore.store(columns, str(path), show_progress=False).close()


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
        print("  NOTE: NUMA optimization is a no-op on this host. The benchmark")
        print("  will still run and exercise the code paths, but A/B numbers")
        print("  will be within noise of each other. To see the actual win,")
        print("  run on a multi-socket / multi-NPS Linux server.")

    with tempfile.TemporaryDirectory() as td:
        n_rows, n_cols = 2_500_000, 50
        total_bytes = n_rows * n_cols * 8

        # Two copies of the same data, written under different policies.
        local_path = Path(td) / "store_written_under_local.cstore"
        interleave_path = Path(td) / "store_written_under_interleave.cstore"
        rng = np.random.default_rng(0)
        cols = {f"c{i:02d}": rng.standard_normal(n_rows) for i in range(n_cols)}
        print()
        print("Writing two stores (one under each writer policy)...")
        write_store_under_policy(local_path, cols, "local")
        write_store_under_policy(interleave_path, cols, "interleave")
        print(f"  {local_path.name}: written under policy=local")
        print(f"  {interleave_path.name}: written under policy=interleave")

        # ---- Writer-side A/B, warm cache: SAME reads on two files written
        # under different writer policies. This is the headline measurement
        # -- writer-side placement is what actually controls warm-cache
        # gather throughput, because reader-side mbind cannot move pages
        # that are already in the page cache (MAP_SHARED read mapping).
        banner(f"WRITER-SIDE A/B (warm)  50 x 20 MB = {total_bytes / 1e9:.1f} GB  ds.dict()")
        local_reader = colstore.open(str(local_path))
        interleave_reader = colstore.open(str(interleave_path))
        try:
            with policy_scope("local"):
                # Reader policy fixed at "local" to isolate the writer-side
                # effect. Whatever delta we see is attributable to where the
                # writer placed the pages.
                r_local = labeled(
                    "writer=local      reader=local  ds.dict()",
                    bench(lambda: local_reader.dict()),
                )
                r_inter = labeled(
                    "writer=interleave reader=local  ds.dict()",
                    bench(lambda: interleave_reader.dict()),
                )
            print(r_local.report())
            print(r_inter.report())

            banner(f"WRITER-SIDE A/B (warm)  50 x 20 MB = {total_bytes / 1e9:.1f} GB  ds.frame()")
            with policy_scope("local"):
                r_local = labeled(
                    "writer=local      reader=local  ds.frame()",
                    bench(lambda: local_reader.frame()),
                )
                r_inter = labeled(
                    "writer=interleave reader=local  ds.frame()",
                    bench(lambda: interleave_reader.frame()),
                )
            print(r_local.report())
            print(r_inter.report())
        finally:
            local_reader.close()
            interleave_reader.close()

        # ---- Writer-side A/B, COLD cache: drop pages, then time first read.
        # On cold reads the page-cache allocation happens DURING the gather,
        # so the reader-side policy on the VMA matters. Writer-side policy
        # is recorded on the file's inode and may or may not influence
        # fresh allocation; this scenario reveals which.
        banner(f"WRITER-SIDE A/B (cold)  50 x 20 MB = {total_bytes / 1e9:.1f} GB  ds.dict()")
        with policy_scope("local"):
            local_reader = colstore.open(str(local_path))
            interleave_reader = colstore.open(str(interleave_path))
            try:
                r_local = labeled(
                    "writer=local      reader=local  ds.dict() cold",
                    bench(
                        lambda: local_reader.dict(),
                        drop_cache_paths=[local_path],
                        n_warmup=0,
                        n_iter=3,
                    ),
                )
                r_inter = labeled(
                    "writer=interleave reader=local  ds.dict() cold",
                    bench(
                        lambda: interleave_reader.dict(),
                        drop_cache_paths=[interleave_path],
                        n_warmup=0,
                        n_iter=3,
                    ),
                )
            finally:
                local_reader.close()
                interleave_reader.close()
        print(r_local.report())
        print(r_inter.report())

        # ---- Reader-side A/B on a file written under "local". The writer
        # placed all pages on one node; reader-side mbind sets a VMA policy
        # but cannot redistribute warm pages. Expect ~noise. This pin-tests
        # the diagnosis that warm-cache reader-side mbind is a no-op.
        banner("READER-SIDE A/B on writer=local file (warm)  ds.dict()")
        with policy_scope("local"):
            r_off = labeled(
                "reader=local      writer=local  ds.dict()",
                bench(lambda: colstore.open(str(local_path)).dict()),
            )
        with policy_scope("interleave"):
            r_on = labeled(
                "reader=interleave writer=local  ds.dict()",
                bench(lambda: colstore.open(str(local_path)).dict()),
            )
        print(r_off.report())
        print(r_on.report())
        print("  (expected: ~noise -- reader mbind cannot move warm pages)")

        # ---- End-to-end: what the user actually sees with config defaults.
        # With config.set_numa_policy("auto") (the default), the writer
        # interleaves at write time and the reader's mbind is a small
        # cold-cache complement. This is the "after the PR lands, what do
        # the workloads in question look like" number.
        banner(f"END-TO-END default policy  50 x 20 MB = {total_bytes / 1e9:.1f} GB")
        end_to_end_path = Path(td) / "end_to_end.cstore"
        write_store_under_policy(end_to_end_path, cols, "auto")
        with policy_scope("auto"):
            ds = colstore.open(str(end_to_end_path))
            try:
                r_dict = labeled(
                    "ds.dict()  writer=auto reader=auto",
                    bench(lambda: ds.dict()),
                )
                r_frame = labeled(
                    "ds.frame() writer=auto reader=auto",
                    bench(lambda: ds.frame()),
                )
            finally:
                ds.close()
        print(r_dict.report())
        print(r_frame.report())

        # ---- Low-concurrency regression check. With only one consumer
        # thread, "interleave" forces 7/8 remote loads on an 8-node host;
        # writer-side and reader-side both can hurt slightly here. The
        # "local" opt-out documented in set_numa_policy exists for this case.
        banner("LOW-CONCURRENCY regression check (workers=1, 1 GB / 50-col)")
        prev_workers = config.get_max_workers()
        prev_cap = config.get_gather_thread_cap()
        try:
            config.set_max_workers(1)
            config.set_gather_thread_cap(1)
            ds_local = colstore.open(str(local_path))
            ds_inter = colstore.open(str(interleave_path))
            try:
                with policy_scope("local"):
                    r_local = labeled(
                        "workers=1  writer=local      reader=local",
                        bench(lambda: ds_local.dict()),
                    )
                    r_inter = labeled(
                        "workers=1  writer=interleave reader=local",
                        bench(lambda: ds_inter.dict()),
                    )
                print(r_local.report())
                print(r_inter.report())
            finally:
                ds_local.close()
                ds_inter.close()
        finally:
            config.set_max_workers(prev_workers)
            config.set_gather_thread_cap(prev_cap)


if __name__ == "__main__":
    if sys.platform != "linux":
        print("This benchmark requires Linux (NUMA syscalls). Exiting.")
        sys.exit(0)
    main()
