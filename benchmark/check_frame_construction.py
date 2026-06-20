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

import argparse
import os
import tempfile
from pathlib import Path

import _common as _c
import numpy as np
import pandas as pd

import colstore
from colstore import testing
from colstore.reader import _make_dataframe_no_consolidate

drop_pagecache_softly = _c.drop_pagecache


def make_store(td: str, name: str, n_rows: int, n_cols: int, dtypes) -> Path:
    """Materialize a store with `n_cols` columns cycling through `dtypes`."""
    path = Path(td) / name
    spec = tuple(np.dtype(d).str for d in dtypes)
    testing.make_store(path, rows=n_rows, cols=n_cols, dtype=spec, seed=0).close()
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


def _compare_construction(dict_data: dict[str, np.ndarray], repeat: int, warmup: int) -> None:
    """Time the consolidating vs no-consolidate DataFrame constructor, A/B."""
    baseline, optimized = construction_pair(dict_data)
    _c.compare(
        [
            ("pd.DataFrame(dict)    (baseline)", baseline),
            ("no_consolidate(dict)  (optimized)", optimized),
        ],
        repeat=repeat,
        warmup=warmup,
        baseline=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(parser, repeat=5, warmup=2, scale=True)
    args = parser.parse_args()

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
        many_homog = make_store(
            td, "homog.cstore", _c.scaled_rows(2_500_000, args), 50, [np.float64]
        )
        ds = colstore.open(str(many_homog))
        try:
            dict_data = ds.dict()  # do the gather once, time only construction
            total_mb = sum(arr.nbytes for arr in dict_data.values()) / 1e6
            banner(f"SAME-DTYPE: 50 cols float64 ({total_mb:.0f} MB)")
            _compare_construction(dict_data, args.repeat, args.warmup)

            # Full end-to-end timing including the gather, since that is what
            # users actually see.
            banner(f"END-TO-END: 50 cols float64 ({total_mb:.0f} MB)")
            drop_pagecache_softly([many_homog])
            _c.compare(
                [
                    ("ds.dict()   (gather only)", lambda: ds.dict()),
                    ("ds.frame()  (gather + new frame)", lambda: ds.frame()),
                ],
                repeat=args.repeat,
                warmup=1,
                baseline=0,
            )
        finally:
            ds.close()

        # ---- Scenario B: mixed dtypes ---------------------------------------
        # 50 cols cycling through 4 dtypes -> 4 consolidated blocks of
        # ~12 cols each. Smaller per-block copies, more blocks, but the total
        # consolidation copy is still on the order of the full data size.
        many_mixed = make_store(
            td,
            "mixed.cstore",
            _c.scaled_rows(2_500_000, args),
            50,
            [np.float64, np.float32, np.int32, np.int64],
        )
        ds = colstore.open(str(many_mixed))
        try:
            dict_data = ds.dict()
            total_mb = sum(arr.nbytes for arr in dict_data.values()) / 1e6
            banner(f"MIXED-DTYPE: 50 cols (4 dtypes, {total_mb:.0f} MB)")
            _compare_construction(dict_data, args.repeat, args.warmup)
        finally:
            ds.close()

        # ---- Scenario C: wide store -----------------------------------------
        # 200 cols x 100K rows mixed dtypes. The user's hot workload was
        # 198 cols; this is a smaller version of the same shape.
        wide = make_store(
            td,
            "wide.cstore",
            _c.scaled_rows(100_000, args),
            200,
            [np.float64, np.float32, np.int32, np.int64],
        )
        ds = colstore.open(str(wide))
        try:
            dict_data = ds.dict()
            total_mb = sum(arr.nbytes for arr in dict_data.values()) / 1e6
            banner(f"WIDE: 200 cols ({total_mb:.0f} MB)")
            _compare_construction(dict_data, args.repeat, args.warmup)
        finally:
            ds.close()

        # ---- Scenario D: tiny per-call overhead -----------------------------
        # 1K rows x 50 cols. Must not regress here: the helper's setup cost
        # (Index construction, list copy, pandas private imports) needs to
        # stay small. The previous PR caught a +6 us per-call regression on
        # a tiny dict materialization that this scenario is designed to
        # surface, so it oversamples.
        tiny = make_store(td, "tiny.cstore", _c.scaled_rows(1_000, args), 50, [np.float64])
        ds = colstore.open(str(tiny))
        try:
            dict_data = ds.dict()
            total_kb = sum(arr.nbytes for arr in dict_data.values()) / 1e3
            banner(f"TINY: 50 cols float64 ({total_kb:.0f} KB)")
            _compare_construction(dict_data, max(20, args.repeat), max(5, args.warmup))
        finally:
            ds.close()

        # ---- Scenario E: TableView.frame() row-sliced -----------------------
        # The same optimization applies through the view path. ``ds[a:b].frame()``
        # is a common idiom for materializing a row subset; verify it benefits.
        ds = colstore.open(str(many_homog))
        try:
            lo, hi = _c.scaled_rows(500_000, args), _c.scaled_rows(2_000_000, args)
            slice_dict = ds[lo:hi].dict()
            slice_mb = sum(arr.nbytes for arr in slice_dict.values()) / 1e6
            banner(f"SLICED VIEW: ds[{lo}:{hi}].frame() ({slice_mb:.0f} MB)")
            _compare_construction(slice_dict, args.repeat, args.warmup)
        finally:
            ds.close()

        # ---- Scenario F: zero-copy frame(copy=False) ------------------------
        # frame(copy=False) feeds the dict(copy=False) views into the per-column
        # BlockManager, so the frame aliases the mapping with no gather copy
        # (and halves peak resident memory). The realistic workload is
        # read-and-reduce: copy=True gathers a full second buffer then reduces
        # it; copy=False reduces straight from the page cache. df.sum() reduces
        # per block, so it does not re-consolidate the zero-copy columns.
        ds = colstore.open(str(many_homog))
        try:
            total_mb = ds.n_rows * len(ds.columns) * 8 / 1e6
            banner(f"ZERO-COPY FRAME read-and-reduce: 50 cols float64 ({total_mb:.0f} MB)")
            drop_pagecache_softly([many_homog])
            _c.compare(
                [
                    (
                        "frame(copy=True).sum()   (gather + reduce)",
                        lambda: ds.frame(copy=True).sum(),
                    ),
                    (
                        "frame(copy=False).sum()  (zero-copy + reduce)",
                        lambda: ds.frame(copy=False).sum(),
                    ),
                ],
                repeat=args.repeat,
                warmup=1,
                baseline=0,
            )
        finally:
            ds.close()


if __name__ == "__main__":
    main()
