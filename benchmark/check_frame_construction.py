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

import os
import tempfile
from pathlib import Path

import _common as _c
import numpy as np
import pandas as pd

import colstore
from colstore.reader import _make_dataframe_no_consolidate

Result = _c.RichResult
bench_interleaved = _c.bench_interleaved
drop_pagecache_softly = _c.drop_pagecache


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
