"""Measure record skipping driven by the statistics footer.

A ``col(name) <op> scalar`` filter on a multi-record file written with
``statistics=True`` consults each record's ``[min, max]`` and reads only the
records that can contain a matching row; the rest are masked out without touching
their bytes. This benchmark A/Bs that skip against a full read of an identical
file written without statistics, on the honest cases: a clustered / sorted column
(the skip should prune most records) and a random column (nothing prunes, so the
skip must not regress). Each is timed warm and, where the page cache can be
evicted, cold -- the skip's largest win, since pruned records' pages never fault
in. The skip's result must equal the full read; the correctness gate checks that
before any timing.

    PYTHONPATH=src python benchmark/check_record_skip.py --tmpdir /tmp
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import _common as _c
import numpy as np

from colstore import col

_VARIANTS = ("skip", "full")  # skip = statistics file; full = identical no-stats file


def _build(path: Path, n_rows: int, rec_rows: int, *, statistics: bool, seed: int) -> int:
    """Write a multi-record file: a clustered/sorted key, a random column, a payload."""
    rng = np.random.default_rng(seed)
    n_records = max(1, n_rows // rec_rows)
    with _c.colstore.create(str(path), statistics=statistics) as writer:
        for i in range(n_records):
            lo = i * rec_rows
            writer.write(
                {
                    "key": np.arange(lo, lo + rec_rows, dtype=np.int64),  # globally sorted
                    "rnd": rng.integers(0, n_rows, rec_rows, dtype=np.int64),  # spans full range
                    "payload": (np.arange(lo, lo + rec_rows) * 1.5).astype(np.float64),
                }
            )
    return n_records * rec_rows


def _scenarios(n_rows: int) -> dict[str, Any]:
    """One predicate per honest case: a clustered column (prunes) and a random one."""
    return {
        "clustered key>p99": col("key") > int(n_rows * 0.99),
        "clustered key>p50": col("key") > n_rows // 2,
        "random    rnd>p99": col("rnd") > int(n_rows * 0.99),
    }


def _skip_ab(
    paths: dict[str, Path], predicate: Any, *, cold: bool, repeat: int, warmup: int
) -> list[_c.ProfileResult]:
    """Interleaved skip-vs-full A/B for one predicate, in ``_VARIANTS`` order.

    Each variant reads its own file (the statistics file uses the skip, the
    plain file reads in full). A cold run evicts both files and reopens before
    each timed read so pruned records' pages fault in only for the survivors.
    """
    holders: dict[str, dict[str, Any]] = {variant: {} for variant in _VARIANTS}

    def make_setup(variant: str) -> Any:
        holder = holders[variant]

        def setup() -> None:
            # Close every reader before evicting -- an open memmap pins the pages
            # and declines the eviction, leaving a warm read.
            for other in holders.values():
                if "reader" in other:
                    other["reader"].close()
                    del other["reader"]
            gc.collect()
            _c.drop_pagecache([paths[variant]])
            holder["reader"] = _c.colstore.open(str(paths[variant]))

        return setup

    if not cold:
        for variant in _VARIANTS:
            holders[variant]["reader"] = _c.colstore.open(str(paths[variant]))

    specs = [
        (variant, (lambda h=holders[variant]: h["reader"][predicate, "key"].array()))
        for variant in _VARIANTS
    ]
    setups = [make_setup(variant) for variant in _VARIANTS] if cold else None
    results = _c.compare(specs, repeat=repeat, warmup=warmup, baseline=1, setups=setups)
    for holder in holders.values():
        if "reader" in holder:
            holder["reader"].close()
    gc.collect()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(
        parser, repeat=7, warmup=2, rows=10_000_000, tmpdir=True, threads=True, json=True
    )
    parser.add_argument("--rec-rows", type=int, default=50_000, help="rows per record")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    _c.apply_runtime_config(args)

    work = Path(args.tmpdir) if args.tmpdir is not None else Path(".")
    work.mkdir(parents=True, exist_ok=True)
    paths = {"skip": work / "record_skip_stats.cstore", "full": work / "record_skip_plain.cstore"}
    for path in paths.values():
        path.unlink(missing_ok=True)
    n_rows = _build(paths["skip"], args.rows, args.rec_rows, statistics=True, seed=args.seed)
    _build(paths["full"], args.rows, args.rec_rows, statistics=False, seed=args.seed)
    print(f"# rows={n_rows:,}  records={n_rows // args.rec_rows}  rec_rows={args.rec_rows}")

    scenarios = _scenarios(n_rows)

    # Correctness gate first: the skip must reproduce the full read exactly.
    skip_reader, full_reader = (_c.colstore.open(str(paths[v])) for v in _VARIANTS)
    for label, predicate in scenarios.items():
        got = skip_reader[predicate, "key"].array()
        expected = full_reader[predicate, "key"].array()
        _c.check_equal(got, expected, f"skip vs full: {label}")
    skip_reader.close()
    full_reader.close()
    print("# correctness gate passed: skip == full for every scenario")
    if args.skip_bench:
        return 0

    modes = ["warm"] + (["cold"] if sys.platform == "linux" else [])
    results: list[_c.Result] = []
    for mode in modes:
        for label, predicate in scenarios.items():
            print(f"\n=== {mode}: {label} ===")
            ab = _skip_ab(
                paths, predicate, cold=(mode == "cold"), repeat=args.repeat, warmup=args.warmup
            )
            for variant, profile in zip(_VARIANTS, ab, strict=True):
                results.append(
                    _c.Result(
                        scenario="record_skip",
                        variant=variant,
                        params={"mode": mode, "predicate": label},
                        median_ms=profile.wall_ms,
                        min_ms=profile.wall_ms,
                        p95_ms=profile.wall_ms,
                        repeat=args.repeat,
                    )
                )

    if args.json is not None:
        _c.write_summary(args.json, results, meta={"benchmark": "check_record_skip"})
        print(f"\n# wrote {args.json}")
    if not args.tmpdir:
        for path in paths.values():
            path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
