"""Benchmark the shared-memo CSE the streaming commit relies on.

When several output columns reuse one subexpression, the commit evaluates them
through a single per-row-range memo, so that subexpression is computed once per
range rather than once per column. This benchmark isolates that effect at the
evaluation level (no file IO): it builds ``--cols`` columns that all reuse one
expensive subexpression, then compares evaluating them through one shared memo
against giving each column its own memo (which recomputes the shared work).

The speedup over the per-column variant approaches the column count when the
shared subexpression dominates, and shrinks toward 1.0 as the cheap per-column
tail grows. It says nothing about IO; the layout and budget benchmarks cover the
write path. Run on the deployment hardware:

    PYTHONPATH=src python benchmark/check_edit_cse.py
    PYTHONPATH=src python benchmark/check_edit_cse.py --skip-bench   # gate only
"""

from __future__ import annotations

import argparse

import _common as _c
import _edit_workload as _w

from colstore.frame import evaluate


def _evaluate_shared(specs, names, n):
    """Evaluate every column through one shared memo (the commit's behavior)."""
    memo = {}
    return [evaluate(specs[name], 0, n, memo) for name in names]


def _evaluate_per_column(specs, names, n):
    """Evaluate every column with its own memo (recomputes shared work)."""
    return [evaluate(specs[name], 0, n, {}) for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(parser, repeat=10, warmup=2, rows=2_000_000, cols=12, json=True)
    args = parser.parse_args()

    n, k = args.rows, args.cols
    specs = _w.shared_graph(n, k)
    names = list(specs)
    print(f"shared subexpression reused by {k} columns of {n:,} rows (f8)")

    if not getattr(args, "skip_correctness", False):
        shared = _evaluate_shared(specs, names, n)
        per_column = _evaluate_per_column(specs, names, n)
        for name, got, expected in zip(names, shared, per_column, strict=True):
            _c.check_equal(got, expected, f"cse[{name}]")
        print("  correctness: shared-memo and per-column results agree\n")

    if args.skip_bench:
        return

    results = _c.compare(
        [
            ("shared-memo (CSE)", lambda: _evaluate_shared(specs, names, n)),
            ("per-column memo  ", lambda: _evaluate_per_column(specs, names, n)),
        ],
        repeat=args.repeat,
        warmup=args.warmup,
        baseline=1,
        throughput_rows=n * k,
    )

    if getattr(args, "json", None) is not None:
        records = [
            _c.Result.from_stats(
                "edit_cse",
                label.strip(),
                {"rows": n, "cols": k},
                _c.TimeStats(r.wall_ms, r.wall_ms, r.wall_ms, r.repeat),
                rows=n * k,
            )
            for label, r in zip(("shared", "per_column"), results, strict=True)
        ]
        _c.set_speedup(records[0], records[1])
        _c.write_summary(args.json, records, meta={"benchmark": "edit_cse", "note": "n*k cells"})
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
