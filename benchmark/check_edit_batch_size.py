"""Characterize commit throughput across the memory budget (batch size).

The streaming commit turns a memory budget into a batch row count
(``budget // bytes_per_row``): smaller budgets mean more, smaller row ranges.
Too small and per-range overhead (ufunc dispatch, memo setup, the Python loop)
dominates; past a knee the per-range cost amortizes and throughput flattens, and
on real hardware it can fall again once a range's working set spills the cache.

This sweep writes one fixed dataset at a range of budgets and reports rows/s and
MB/s for each, to show the shape of that curve. It is diagnostic -- it does not
change the configured default; ``check_edit_budget.py`` is the script that
recommends a default. Run on the deployment hardware:

    PYTHONPATH=src python benchmark/check_edit_batch_size.py
    PYTHONPATH=src python benchmark/check_edit_batch_size.py --skip-bench
"""

from __future__ import annotations

import argparse
import functools
import tempfile
from pathlib import Path

import _common as _c
import _edit_workload as _w

import colstore
from colstore.format import write_dataset_streaming

_MIB = 1024 * 1024
_DEFAULT_BUDGETS = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # MiB


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(parser, repeat=5, warmup=2, rows=4_000_000, cols=12, json=True)
    parser.add_argument(
        "--budgets-mib",
        type=int,
        nargs="+",
        default=_DEFAULT_BUDGETS,
        help="memory budgets to sweep, in MiB",
    )
    args = parser.parse_args()

    n, k = args.rows, args.cols
    specs = _w.shared_graph(n, k)
    names = list(specs)
    out_bytes = n * k * 8  # f8 columns
    print(f"{k} columns x {n:,} rows (f8), {out_bytes / 1e6:.0f} MB output, shared subexpression")

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "sweep.cstore")

        if not getattr(args, "skip_correctness", False):
            write_dataset_streaming(specs, n, path, memory_budget=_MIB)
            reader = colstore.open(path)
            try:
                got = reader.dict()
            finally:
                reader.close()
            for name in names:
                from colstore.frame import evaluate

                _c.check_equal(got[name], evaluate(specs[name], 0, n, {}), f"batch[{name}]")
            print("  correctness: a small-budget write reproduces the columns\n")

        if args.skip_bench:
            return

        budgets = [mib * _MIB for mib in args.budgets_mib]
        specs_for = [
            (
                f"budget={mib:>4} MiB",
                functools.partial(write_dataset_streaming, specs, n, path, memory_budget=b),
            )
            for mib, b in zip(args.budgets_mib, budgets, strict=True)
        ]
        results = _c.compare(
            specs_for,
            repeat=args.repeat,
            warmup=args.warmup,
            baseline=len(specs_for) - 1,  # largest budget (closest to single pass)
            throughput_rows=n,
        )
        print("\nMB/s by budget:")
        for mib, r in zip(args.budgets_mib, results, strict=True):
            mbps = out_bytes / 1e6 / (r.wall_ms / 1000.0) if r.wall_ms > 0 else float("inf")
            batch_rows = max(1, min(n, (mib * _MIB) // (k * 8)))
            print(f"  {mib:>4} MiB -> {mbps:8.0f} MB/s  (batch_rows={batch_rows:,})")

        if getattr(args, "json", None) is not None:
            records = [
                _c.Result.from_stats(
                    "edit_batch_size",
                    f"{mib}MiB",
                    {"rows": n, "cols": k, "budget_mib": mib},
                    _c.TimeStats(r.wall_ms, r.wall_ms, r.wall_ms, r.repeat),
                    rows=n,
                )
                for mib, r in zip(args.budgets_mib, results, strict=True)
            ]
            _c.write_summary(args.json, records, meta={"benchmark": "edit_batch_size"})
            print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
