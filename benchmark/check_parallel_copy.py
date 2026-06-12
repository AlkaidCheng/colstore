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

import argparse
import os
import tempfile
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import config, testing

_REPEAT, _WARMUP = 5, 2


def bench(label, fn, *, repeat=None, warmup=None, drop_cache_paths=None):
    """Profile one variant via the public profiler; cold runs evict per iter."""
    setup = (lambda: _c.drop_pagecache(drop_cache_paths)) if drop_cache_paths else None
    return _c.profile(
        fn,
        repeat=_REPEAT if repeat is None else repeat,
        warmup=_WARMUP if warmup is None else warmup,
        setup=setup,
        label=label,
    )


def make_store(td: str, name: str, n_rows: int, n_cols: int, dtype) -> Path:
    """Materialize a store with `n_cols` columns of `n_rows` rows."""
    path = Path(td) / name
    testing.make_store(path, rows=n_rows, cols=n_cols, dtype=np.dtype(dtype).str, seed=0).close()
    return path


def banner(s):
    print(f"\n=== {s} ===")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(parser, repeat=5, warmup=2, scale=True)
    args = parser.parse_args()
    global _REPEAT, _WARMUP
    _REPEAT, _WARMUP = args.repeat, args.warmup

    print("Environment:")
    print(f"  os.cpu_count()              = {os.cpu_count()}")
    print(f"  config.get_max_workers()    = {config.get_max_workers()}")
    print(f"  config.get_gather_thread_cap() = {config.get_gather_thread_cap()}")

    with tempfile.TemporaryDirectory() as td:
        # --- Scenario A: 1 big column (~80 MB at scale=1). The sweet spot for
        #     parallel copy: no column-pool concurrency, just a single big memcpy.
        single_rows = _c.scaled_rows(10_000_000, args)
        single = make_store(td, "single.cstore", single_rows, 1, np.float64)
        col_bytes = single_rows * 8
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
                warmup=0,
                repeat=3,
                drop_cache_paths=[single],
            ),
        ]
        for r in results:
            print(r.report())
        ds.close()

        # --- Scenario B: many columns (~50 cols x ~20 MB each = 1 GB).
        #     This is the regime where the column pool dominates and per-column
        #     parallel-copy is most likely to harm rather than help.
        many_rows = _c.scaled_rows(2_500_000, args)
        many = make_store(td, "many.cstore", many_rows, 50, np.float64)
        total = many_rows * 50 * 8
        per_col = many_rows * 8
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
        wide_rows = _c.scaled_rows(200_000, args)
        wide = make_store(td, "wide.cstore", wide_rows, 200, np.float64)
        total = wide_rows * 200 * 8
        per_col = wide_rows * 8
        banner(f"WIDE STORE, 200 x {per_col/1e6:.1f} MB = {total/1e6:.0f} MB (warm)")
        ds = colstore.open(str(wide))
        results = [
            bench("ds.dict()                              (new shortcut)", lambda: ds.dict()),
            bench("ds[:].dict()                           (via view)", lambda: ds[:].dict()),
        ]
        for r in results:
            print(r.report())
        ds.close()

        # --- Scenario D: slice reads, contiguous and strided. Sized so the
        #     step-2 strided view (half the rows) clears the parallel-copy
        #     threshold -- strided reads now take the same row-range split as
        #     contiguous ones. The ::16 line stays below the threshold to show
        #     the serial fallback is intact for small strided reads.
        slice_rows = _c.scaled_rows(16_000_000, args)
        big_slice = make_store(td, "slice.cstore", slice_rows, 1, np.float64)
        col_bytes = slice_rows * 8
        lo, hi = slice_rows // 160, slice_rows - slice_rows // 160
        banner(f"SLICE READS, {col_bytes/1e6:.0f} MB column (warm)")
        ds = colstore.open(str(big_slice))
        results = [
            bench(
                f"ds[{lo}:{hi}, 'c0'].array()  (step-1, {(hi - lo) * 8 / 2**20:.0f} MiB)",
                lambda: ds[lo:hi, "c0"].array(),
            ),
            bench(
                f"ds[::2, 'c0'].array()        (step-2, {slice_rows // 2 * 8 / 2**20:.0f} MiB)",
                lambda: ds[::2, "c0"].array(),
            ),
            bench(
                f"ds[::16, 'c0'].array()       (step-16, {slice_rows // 16 * 8 / 2**20:.0f} MiB)",
                lambda: ds[::16, "c0"].array(),
            ),
        ]
        for r in results:
            print(r.report())
        ds.close()

        # --- Scenario E: FORCE the parallel-copy path even on a 1-CPU sandbox
        #     by raising gather_thread_cap. ratio > 1.0 would prove the
        #     ThreadPoolExecutor inside _parallel_copy is actually
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
