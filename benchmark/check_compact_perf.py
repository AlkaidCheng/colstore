"""Compaction perf characterization.

Two questions:

1. Compaction itself: how long does it take per byte? Should be
   bandwidth-limited since the inner loop is os.sendfile. We measure
   throughput across a few file sizes.

2. Read perf delta: how much does compaction help the patterns it was
   designed to fix? We focus on unsorted-fancy reads at large R, where
   the multi-record path degrades most. The before/after reads are timed
   interleaved (A/B/A/B) so the comparison isn't confounded by first-touch
   placement or page-cache warmth differing between two sequential blocks.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import _common as _c
import numpy as np

import colstore


def _read_fn(ds: Any, idx: np.ndarray) -> Callable[[], Any]:
    """A zero-arg closure timing one unsorted-fancy read of column ``x``."""
    return lambda: ds[idx, "x"].array()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--n-indices", type=int, default=200_000)
    parser.add_argument("--record-counts", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--repeats", type=int, default=5, help="interleaved A/B rounds (n_iter)")
    parser.add_argument("--warmup", type=int, default=2, help="warmup passes per variant")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--tmpdir", default="/tmp/colstore_compact_bench")
    args = parser.parse_args()

    dtype = np.dtype(args.dtype)
    tmpdir = Path(args.tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    data = rng.standard_normal(args.rows).astype(dtype)
    bytes_per_row = dtype.itemsize

    print(
        f"\nSetup: {args.rows:,} rows of {dtype}, "
        f"file size ~{args.rows * bytes_per_row / 1e6:.1f} MB"
    )
    print(f"Indices: {args.n_indices:,} (unsorted-fancy is the worst-case pattern)")

    unsorted_idx = rng.permutation(args.rows)[: args.n_indices].astype(np.int64)

    for n_rec in args.record_counts:
        print(f"\n---- R = {n_rec} records ----")
        path = tmpdir / f"compact_r{n_rec}.cstore"
        if path.exists():
            path.unlink()
        chunk = args.rows // n_rec
        with colstore.create(path) as f:
            for i in range(n_rec):
                s = i * chunk
                e = (i + 1) * chunk if i < n_rec - 1 else args.rows
                f.write({"x": data[s:e]})

        info_before = colstore.info(path)
        print(f"  size on disk: {info_before.file_size / 1e6:.1f} MB")

        # Compact out-of-place so both files stay live and the before/after
        # reads can be timed interleaved against each other (see below).
        compacted = tmpdir / f"compact_r{n_rec}_done.cstore"
        if compacted.exists():
            compacted.unlink()
        t_compact = time.perf_counter()
        colstore.compact(path, out=compacted, show_progress=False)
        t_compact = time.perf_counter() - t_compact
        mb = info_before.file_size / 1e6
        print(f"  compact: {t_compact * 1000:7.2f} ms ({mb / t_compact:.1f} MB/s throughput)")

        # Read perf: BEFORE vs AFTER, timed *interleaved* (A/B/A/B) rather than
        # as two sequential blocks. Timing them in sequence -- all BEFORE
        # samples, then all AFTER samples, on two separately written files --
        # lets first-touch NUMA placement and page-cache warmth differ between
        # the two measurements, which at large sizes (where both paths are
        # bandwidth-bound and effectively at parity) manufactured misleading
        # sub-1x ratios. bench_interleaved warms both readers first, then
        # alternates them each round, keeping cache and scheduler state
        # comparable; it also reports cpu/wall ratio, thread count, and
        # page-fault deltas so a near-1x ratio can be read as genuine parity
        # rather than noise.
        with colstore.open(path) as ds_before, colstore.open(compacted) as ds_after:
            before, after = _c.bench_interleaved(
                [
                    "unsorted-fancy read BEFORE (fragmented)",
                    "unsorted-fancy read AFTER  (compacted) ",
                ],
                [
                    _read_fn(ds_before, unsorted_idx),
                    _read_fn(ds_after, unsorted_idx),
                ],
                n_iter=args.repeats,
                n_warmup=args.warmup,
            )
        print(before.report())
        print(after.report())
        best_before = min(r.wall_ms for r in before.runs)
        best_after = min(r.wall_ms for r in after.runs)
        speedup = best_before / best_after if best_after > 0 else float("inf")
        print(f"  -> compaction speedup on this pattern: {speedup:.2f}x (best wall / best wall)")


if __name__ == "__main__":
    main()
