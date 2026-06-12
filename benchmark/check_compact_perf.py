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
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import _common as _c
import numpy as np

import colstore
from colstore import testing


def _read_fn(ds: Any, idx: np.ndarray) -> Callable[[], Any]:
    """A zero-arg closure timing one unsorted-fancy read of column ``x``."""
    return lambda: ds[idx, "x"].array()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(
        parser,
        repeat=5,
        warmup=2,
        rows=2_000_000,
        indices=200_000,
        record_counts=[10, 100, 1000],
        dtype="float32",
        tmpdir=True,
        skip_correctness=False,
    )
    args = parser.parse_args()

    dtype = np.dtype(args.dtype)
    tmpdir = args.tmpdir or Path(tempfile.mkdtemp(prefix="colstore_compact_bench"))
    tmpdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    data = testing.make_columns(args.rows, 1, dtype=args.dtype, seed=0)["c0"]
    bytes_per_row = dtype.itemsize

    print(
        f"\nSetup: {args.rows:,} rows of {dtype}, "
        f"file size ~{args.rows * bytes_per_row / 1e6:.1f} MB"
    )
    print(f"Indices: {args.indices:,} (unsorted-fancy is the worst-case pattern)")

    unsorted_idx = rng.permutation(args.rows)[: args.indices].astype(np.int64)

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

        # Read perf: BEFORE vs AFTER, timed *interleaved* (A/B/A/B) via
        # compare() rather than as two sequential blocks. Timing them in
        # sequence -- all BEFORE samples, then all AFTER samples, on two
        # separately written files -- lets first-touch NUMA placement and
        # page-cache warmth differ between the two measurements, which at large
        # sizes (where both paths are bandwidth-bound and at parity)
        # manufactured misleading sub-1x ratios. compare() warms both readers,
        # alternates them each round, and reports cpu/wall ratio, threads, and
        # page-fault deltas so a near-1x ratio reads as genuine parity.
        with colstore.open(path) as ds_before, colstore.open(compacted) as ds_after:
            before, after = _c.compare(
                [
                    ("unsorted-fancy read BEFORE (fragmented)", _read_fn(ds_before, unsorted_idx)),
                    ("unsorted-fancy read AFTER  (compacted) ", _read_fn(ds_after, unsorted_idx)),
                ],
                repeat=args.repeat,
                warmup=args.warmup,
                baseline=0,
            )
        speedup = before.wall_ms / after.wall_ms if after.wall_ms > 0 else float("inf")
        print(f"  -> compaction speedup on this pattern: {speedup:.2f}x (best wall / best wall)")


if __name__ == "__main__":
    main()
