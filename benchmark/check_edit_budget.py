"""Sweep candidate default memory budgets and recommend one.

The configured default budget governs how large each streamed row range is. The
right value is a compromise: large enough that per-range overhead is amortized
(and, on real hardware, that a range's working set still fits cache), but no
larger, since a bigger budget only raises peak RAM once throughput has flattened.
This script sweeps a set of candidate defaults across two shapes that bracket the
regimes -- a compute-bound graph with a shared subexpression, and a memcpy-bound
passthrough of plain columns -- and for each shape reports the budget at which
throughput plateaus, then recommends the smallest budget within a tolerance of
each shape's peak (smaller being preferred, since it bounds memory tighter).

It prints a recommendation; it does not change the configured default. Apply the
result by editing the default in the library if the measurement supports it. Run
on the deployment hardware, where cache sizing is representative:

    PYTHONPATH=src python benchmark/check_edit_budget.py
    PYTHONPATH=src python benchmark/check_edit_budget.py --tolerance 0.05
"""

from __future__ import annotations

import argparse
import functools
import tempfile
from pathlib import Path

import _common as _c
import _edit_workload as _w

import colstore
from colstore.config import get_default_memory_budget
from colstore.format import write_dataset_streaming
from colstore.frame import evaluate

_MIB = 1024 * 1024
_DEFAULT_BUDGETS = [8, 16, 32, 64, 128, 256]  # MiB; candidate defaults


def _sweep_shape(label, specs, n, names, budgets_mib, args, path):
    print(f"\n=== {label}: {len(names)} columns x {n:,} rows ===")
    if not getattr(args, "skip_correctness", False):
        write_dataset_streaming(specs, n, path, memory_budget=_MIB)
        reader = colstore.open(path)
        try:
            got = reader.dict()
        finally:
            reader.close()
        for name in names:
            _c.check_equal(got[name], evaluate(specs[name], 0, n, {}), f"{label}[{name}]")
        print("  correctness: write reproduces the columns")
    if args.skip_bench:
        return None
    results = _c.compare(
        [
            (
                f"{mib:>4} MiB",
                functools.partial(
                    write_dataset_streaming, specs, n, path, memory_budget=mib * _MIB
                ),
            )
            for mib in budgets_mib
        ],
        repeat=args.repeat,
        warmup=args.warmup,
        baseline=len(budgets_mib) - 1,
        throughput_rows=n,
    )
    return [(mib, r.throughput(n)) for mib, r in zip(budgets_mib, results, strict=True)]


def _recommend(curve, tolerance):
    """Smallest budget whose throughput is within ``tolerance`` of the peak."""
    peak = max(tput for _, tput in curve)
    cutoff = peak * (1.0 - tolerance)
    return min(mib for mib, tput in curve if tput >= cutoff)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(parser, repeat=5, warmup=2, rows=4_000_000, cols=12)
    parser.add_argument(
        "--budgets-mib", type=int, nargs="+", default=_DEFAULT_BUDGETS, help="candidate defaults"
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.05, help="fraction below peak still 'at plateau'"
    )
    args = parser.parse_args()

    n, k = args.rows, args.cols
    print(f"current configured default: {get_default_memory_budget() / _MIB:.0f} MiB")

    shapes = {
        "compute-bound (shared subexpression)": _w.shared_graph(n, k),
        "memcpy-bound (passthrough)": _w.passthrough_graph(n, k),
    }
    with tempfile.TemporaryDirectory() as tmp:
        picks = []
        for label, specs in shapes.items():
            path = str(Path(tmp) / "budget.cstore")
            curve = _sweep_shape(label, specs, n, list(specs), args.budgets_mib, args, path)
            if curve is not None:
                pick = _recommend(curve, args.tolerance)
                picks.append(pick)
                print(f"  plateau within {args.tolerance:.0%} of peak from {pick} MiB up")

        if picks:
            recommended = max(picks)
            print(
                f"\nrecommended default: {recommended} MiB "
                f"(smallest serving every shape within {args.tolerance:.0%} of its peak)"
            )


if __name__ == "__main__":
    main()
