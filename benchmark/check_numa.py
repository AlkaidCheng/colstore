"""Robust benchmark for the NUMA optimization (writer + reader sides).

This benchmark is the fourth revision in the NUMA series. The earlier
issue was that opening a fresh ``ColStoreReader`` inside the timed
region dragged ~30 ms of setup cost into every wall-time measurement;
the version before this fixed that but had a second bug: ``posix_
fadvise(DONTNEED)`` was being called while readers were still open
across the eviction. The ``np.memmap`` references kept the pages
pinned in the cache and the "cold" measurements were actually warm
(seen in the field: writer=local "cold" 56 ms < writer=local "warm"
81 ms -- physically impossible if eviction worked).

This revision fixes the second bug via ``time_cold``: no reader is
open across the eviction, ``gc.collect()`` is called to drop straggler
references, and a fresh reader is allocated inside each measurement.
``_warn_if_not_cold`` watches the major-fault counter to surface
incomplete eviction.

The benchmark also makes one structural change: cold-cache A/B is
now reader-side, not writer-side. Cold reads reallocate page-cache
pages on the FAULTING thread per its mempolicy, not per the original
writer's mempolicy. Writer-side placement is forgotten on eviction.
The right A/B for cold reads is reader-side mbind, which controls
how re-faulted pages get distributed; the original PR's reader-side
mbind earns its place here. Writer-side cold A/B stays as a pin-test
of the "evicted pages forget writer placement" claim.

Three things this measures:

  1. Writer-side policy A/B, WARM cache (headline). Write the SAME
     data twice -- once under ``"local"`` policy, once under
     ``"interleave"`` -- then read each with the same reader policy.
     Warm-cache reads pick up whatever pages the writer placed; this
     is where the writer-side ``set_mempolicy`` from commit 4 shows
     up in the numbers.

  2. Reader-side policy A/B, COLD cache. Same file (writer=local),
     read with each reader policy after proper eviction. The VMA
     mempolicy governs where re-faulted pages get allocated.

  3. Pin-tests:

     * Writer-side cold A/B should be ~noise (evicted -> reallocated
       by faulting thread, regardless of where the writer placed).
     * Reader-side warm A/B should be ~noise (mbind cannot move
       already-resident pages on a MAP_SHARED read mapping).

  4. End-to-end ``ds.dict()`` / ``ds.frame()`` numbers under the
     default ``"auto"`` policy so the user sees what the PR
     delivers without any explicit config change.

  5. Low-concurrency regression check (workers=1) so the ``"local"``
     opt-out has empirical justification.

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
import gc
import os
import resource
import statistics
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def time_cold(path: Path, gather_fn, *, drop_cache_paths: list[Path] | None = None) -> Run:
    """One cold-cache measurement.

    Cold-cache benchmarking has a subtle pitfall: ``posix_fadvise(
    DONTNEED)`` is an advisory hint that the kernel ignores for any
    page that still has live references. If a ``ColStoreReader`` is
    open across the eviction, its ``np.memmap`` keeps the pages
    pinned and the "cold" reads are actually warm. The original
    cold scenario in check_numa.py made exactly this mistake and
    reported writer=local at 56 ms warm vs 81 ms cold -- cold faster
    than warm, which is physically impossible if eviction worked.

    The fix is this helper: no reader is open during eviction. Each
    cold measurement allocates a fresh reader inside the function
    after the eviction, times only the gather (open is outside the
    timed region by design; the few hundred microseconds it costs
    are independent of NUMA policy), and closes the reader before
    returning. ``gc.collect()`` after the close releases any stray
    memmap references before the NEXT measurement evicts.

    The major-fault count in the returned ``Run`` is the empirical
    check: real cold reads of a 1 GB store should record thousands
    of major faults. Zero major faults means the eviction was
    declined by the kernel; ``_warn_if_not_cold`` surfaces that.
    """
    if drop_cache_paths is None:
        drop_cache_paths = [path]
    gc.collect()  # release any straggling memmap refs before evicting
    drop_pagecache_softly(drop_cache_paths)

    reader = colstore.open(str(path))
    try:
        return time_call(lambda: gather_fn(reader))
    finally:
        reader.close()
        gc.collect()  # release the memmap before the next iteration's evict


def bench_cold_pair(
    pair_a: tuple[str, Path, Callable[[Any], Any]],
    pair_b: tuple[str, Path, Callable[[Any], Any]],
    *,
    n_iter: int = 3,
) -> tuple[Result, Result]:
    """A/B cold benchmark: alternate A and B across n_iter iterations.

    Each iteration measures A then B, with the target file's pages
    evicted immediately before each measurement. Per-measurement
    eviction (vs. evicting once at iteration start) means A's now-warm
    cache state after its measurement doesn't influence B's cold
    measurement -- the kernel sees clean per-file eviction signals.
    """
    label_a, path_a, fn_a = pair_a
    label_b, path_b, fn_b = pair_b
    result_a = Result(label=label_a)
    result_b = Result(label=label_b)
    for _ in range(n_iter):
        result_a.runs.append(time_cold(path_a, fn_a))
        result_b.runs.append(time_cold(path_b, fn_b))
    return result_a, result_b


def _warn_if_not_cold(result: Result) -> None:
    """Print a warning if the major-fault count suggests the cache wasn't cold.

    Real cold reads of a 1 GB file with readahead should record
    hundreds to thousands of major faults. Zero majors across every
    run of a "cold" scenario means ``posix_fadvise(DONTNEED)`` was
    ignored -- either because the kernel declined the hint (rare),
    or because some process / reference held the file open during
    eviction (the bug fixed by ``time_cold``).
    """
    if not result.runs:
        return
    if all(run.major_pf == 0 for run in result.runs):
        print(
            f"  warning: '{result.label.strip()}' recorded 0 major page faults "
            "across all runs; the page cache may not have been evicted, "
            "so these numbers reflect warm reads, not true cold reads."
        )


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

        # ---- Writer-side A/B, COLD cache (PIN-TEST: ~noise expected).
        # On cold reads, writer-side placement is irrelevant: evicted pages
        # are reallocated by the FAULTING thread according to its own
        # mempolicy, not by whoever wrote the file originally. The page-
        # cache placement decision happens fresh at re-fault time.
        # This A/B should therefore land in noise. If it doesn't, our
        # mental model of MAP_SHARED page placement is wrong and we need
        # to revisit the diagnosis. The "evicted -> reallocated by
        # faulting thread" claim is exactly what makes writer-side a
        # warm-cache-only phenomenon.
        banner(f"WRITER-SIDE A/B (cold)  50 x 20 MB = {total_bytes / 1e9:.1f} GB  ds.dict()")
        with policy_scope("local"):
            r_local, r_inter = bench_cold_pair(
                (
                    "writer=local      reader=local  ds.dict() cold",
                    local_path,
                    lambda ds: ds.dict(),
                ),
                (
                    "writer=interleave reader=local  ds.dict() cold",
                    interleave_path,
                    lambda ds: ds.dict(),
                ),
            )
        print(r_local.report())
        print(r_inter.report())
        print("  (pin-test: expected ~noise -- evicted pages forget the writer's placement)")
        _warn_if_not_cold(r_local)
        _warn_if_not_cold(r_inter)

        # ---- Reader-side A/B, COLD cache (where reader-side mbind earns
        # its place). On cold reads the page-cache allocation happens
        # DURING the gather's read faults, and the VMA's mempolicy
        # governs that allocation. This is the scenario where the
        # original PR's reader-side mbind actually does work: cold reads
        # of files written under "local" (or by external tools that
        # didn't apply writer-side interleave) benefit from setting
        # MPOL_INTERLEAVE on the read mapping so the re-faulted pages
        # spread across nodes. Same file used for both runs; only the
        # reader policy changes.
        banner(f"READER-SIDE A/B (cold)  50 x 20 MB = {total_bytes / 1e9:.1f} GB  ds.dict()")
        with policy_scope("local"):
            r_off, _ = bench_cold_pair(
                (
                    "reader=local      writer=local  ds.dict() cold",
                    local_path,
                    lambda ds: ds.dict(),
                ),
                # bench_cold_pair always measures two; pair the second with
                # a throwaway file so the per-iteration cadence is preserved.
                # The second result is discarded.
                ("(throwaway)", local_path, lambda ds: ds.dict()),
                n_iter=3,
            )
        with policy_scope("interleave"):
            r_on, _ = bench_cold_pair(
                (
                    "reader=interleave writer=local  ds.dict() cold",
                    local_path,
                    lambda ds: ds.dict(),
                ),
                ("(throwaway)", local_path, lambda ds: ds.dict()),
                n_iter=3,
            )
        print(r_off.report())
        print(r_on.report())
        print("  (cold: reader-side mbind controls re-fault allocation)")
        _warn_if_not_cold(r_off)
        _warn_if_not_cold(r_on)

        # ---- Reader-side A/B, WARM cache (PIN-TEST: ~noise expected).
        # Kept from before the cold-methodology fix because it pins a
        # different claim: mbind cannot move pages that are already in
        # the page cache (MAP_SHARED read mapping). The writer placed
        # them on whichever node serviced the I/O; reader-side mbind
        # recorded a VMA policy but the kernel has nothing to do with
        # it for already-resident pages. Expect ~noise here too.
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
        print("  (pin-test: expected ~noise -- reader mbind cannot move warm pages)")

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
