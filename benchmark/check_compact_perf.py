"""Compaction perf characterization.

Two questions:

1. Compaction itself: how long does it take per byte? Should be
   bandwidth-limited since the inner loop is os.sendfile. We measure
   throughput across a few file sizes.

2. Read perf delta: how much does compaction help the patterns it was
   designed to fix? We focus on unsorted-fancy reads at large R, where
   the multi-record path degrades most.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import colstore


def _best(fn, repeats: int) -> float:
    fn()
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--n-indices", type=int, default=200_000)
    parser.add_argument("--record-counts", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--repeats", type=int, default=3)
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

        # Read perf BEFORE compaction.
        with colstore.open(path) as ds:
            t_before = _best(lambda d=ds: d[unsorted_idx, "x"].array(), args.repeats)
        print(f"  unsorted-fancy read BEFORE: {t_before * 1000:7.2f} ms")

        # Compact (out-of-place so we can re-time the "before" case from the
        # untouched original on subsequent re-runs of the script).
        compacted = tmpdir / f"compact_r{n_rec}_done.cstore"
        if compacted.exists():
            compacted.unlink()
        t_compact = time.perf_counter()
        colstore.compact(path, out=compacted, show_progress=False)
        t_compact = time.perf_counter() - t_compact
        mb = info_before.file_size / 1e6
        print(f"  compact: {t_compact * 1000:7.2f} ms ({mb / t_compact:.1f} MB/s throughput)")

        # Read perf AFTER compaction.
        with colstore.open(compacted) as ds:
            t_after = _best(lambda d=ds: d[unsorted_idx, "x"].array(), args.repeats)
        speedup = t_before / t_after if t_after > 0 else float("inf")
        print(f"  unsorted-fancy read AFTER:  {t_after * 1000:7.2f} ms  ({speedup:.1f}x speedup)")


if __name__ == "__main__":
    main()
