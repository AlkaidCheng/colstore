"""Measure append-shard cost vs the concat-rewrite baseline, and read cost by shard count.

Growing a dataset by appending each shard writes only the new rows (O(total) over
the whole build); rebuilding it with ``store(..., mode="recreate")`` after each
addition rewrites everything every time (O(shards x total) -- quadratic), which is
the baseline append replaces. This A/Bs the two build strategies to the same
dataset, with a correctness gate that both produce identical rows first. It then
reports read cost (open + a selective filter) as the shard count grows -- the cost
a later shard-compaction step would address.

    PYTHONPATH=src python benchmark/check_shard_append.py --tmpdir $SCRATCH/
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import _common as _c
import numpy as np

from colstore import col


def _batch(index: int, rows: int) -> dict[str, np.ndarray]:
    lo = index * rows
    return {
        "key": np.arange(lo, lo + rows, dtype=np.int64),
        "val": (np.arange(lo, lo + rows) * 1.5).astype(np.float64),
    }


def _build_append(directory: Path, shards: int, rows: int) -> None:
    for index in range(shards):
        _c.colstore.append(directory, _batch(index, rows))


def _build_rewrite(out_file: Path, shards: int, rows: int) -> None:
    """The O(shards x total) baseline: rewrite all batches so far on each addition."""
    accumulated: dict[str, np.ndarray] = {}
    for index in range(shards):
        batch = _batch(index, rows)
        accumulated = (
            batch
            if not accumulated
            else {name: np.concatenate([accumulated[name], batch[name]]) for name in batch}
        )
        _c.colstore.store(accumulated, out_file, mode="recreate", show_progress=False).close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(parser, repeat=3, warmup=1, tmpdir=True, json=True)
    parser.add_argument("--shards", type=int, default=50, help="shards in the built dataset")
    parser.add_argument("--shard-rows", type=int, default=100_000, help="rows per shard")
    args = parser.parse_args()

    work = Path(args.tmpdir) if args.tmpdir is not None else Path(".")
    work.mkdir(parents=True, exist_ok=True)
    append_dir = work / "shard_append_ds"
    rewrite_file = work / "shard_append_rewrite.cstore"
    shards, rows = args.shards, args.shard_rows

    # Correctness gate: both strategies produce the same rows, in order.
    shutil.rmtree(append_dir, ignore_errors=True)
    rewrite_file.unlink(missing_ok=True)
    _build_append(append_dir, shards, rows)
    _build_rewrite(rewrite_file, shards, rows)
    a = _c.colstore.open(append_dir)
    b = _c.colstore.open(rewrite_file)
    _c.check_equal(a.array("key"), b.array("key"), "append vs rewrite: key")
    _c.check_equal(a.array("val"), b.array("val"), "append vs rewrite: val")
    a.close()
    b.close()
    print(f"# correctness gate passed: append == rewrite ({shards} shards x {rows} rows)")
    if args.skip_bench:
        return 0

    def setup_append() -> None:
        shutil.rmtree(append_dir, ignore_errors=True)

    def setup_rewrite() -> None:
        rewrite_file.unlink(missing_ok=True)

    print(f"\n=== build a {shards}-shard dataset ===")
    specs = [
        ("append", lambda: _build_append(append_dir, shards, rows)),
        ("concat-rewrite", lambda: _build_rewrite(rewrite_file, shards, rows)),
    ]
    results = _c.compare(
        specs,
        repeat=args.repeat,
        warmup=args.warmup,
        baseline=1,
        setups=[setup_append, setup_rewrite],
    )

    print("\n=== read cost vs shard count (same total rows, open + selective filter) ===")
    read_total = shards * rows  # held constant so only the shard count varies
    summary: list[_c.Result] = []
    for n_shards in (10, 100, 1000):
        per_shard = max(1, read_total // n_shards)
        read_dir = work / f"shard_read_{n_shards}"
        shutil.rmtree(read_dir, ignore_errors=True)
        with _c.colstore.appender(read_dir, statistics=True) as ap:
            for index in range(n_shards):
                ap.write(_batch(index, per_shard))
        threshold = int(n_shards * per_shard * 0.99)

        def read(directory: Path = read_dir, cut: int = threshold) -> None:
            ds = _c.colstore.open(directory)
            try:
                ds[col("key") > cut, "key"].array()
            finally:
                ds.close()

        stats = _c.time_stats(read, repeat=args.repeat, warmup=args.warmup)
        print(f"  {n_shards:>5} shards  open+filter median={stats.median_ms:8.2f} ms")
        summary.append(
            _c.Result(
                scenario="shard_read_cost",
                variant="open+filter",
                params={"shards": n_shards},
                median_ms=stats.median_ms,
                min_ms=stats.min_ms,
                p95_ms=stats.p95_ms,
                repeat=args.repeat,
            )
        )

    if args.json is not None:
        rows_built = shards * rows
        build = [
            _c.Result(
                scenario="shard_build",
                variant=label,
                params={"shards": shards},
                median_ms=res.wall_ms,
                min_ms=res.wall_ms,
                p95_ms=res.wall_ms,
                repeat=args.repeat,
                rows=rows_built,
            )
            for label, res in zip(("append", "concat-rewrite"), results, strict=True)
        ]
        _c.write_summary(args.json, build + summary, meta={"benchmark": "check_shard_append"})
        print(f"\n# wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
