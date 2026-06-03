"""Compare streaming ColStoreWriter vs one-shot ``colstore.store`` perf.

Streaming a fixed total row count as many small records should be slower
per-row than one big write (more record headers, more counter updates),
but the overhead per record should be bounded and small. This benchmark
characterizes the overhead so future PRs can catch regressions.
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
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--record-counts", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--tmpdir", default="/tmp/colstore_writer_bench")
    args = parser.parse_args()

    dtype = np.dtype(args.dtype)
    tmpdir = Path(args.tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    data = rng.standard_normal(args.rows).astype(dtype)

    print(f"Setup: {args.rows:,} rows of {dtype}")
    print(f"{'records':<10} {'wall ms':>10}  {'us/record':>12}  {'rows/record':>14}")
    print("-" * 55)

    for n_rec in args.record_counts:
        chunk = args.rows // n_rec
        path = tmpdir / f"w_r{n_rec}.cstore"

        def write_all(p=path, c=chunk, nr=n_rec):
            if p.exists():
                p.unlink()
            with colstore.create(p) as w:
                for i in range(nr):
                    s = i * c
                    e = (i + 1) * c if i < nr - 1 else args.rows
                    w.write({"x": data[s:e]})

        t = _best(write_all, args.repeats)
        us_per_record = t * 1e6 / n_rec
        print(f"R={n_rec:<8} {t * 1000:>10.3f}  {us_per_record:>11.2f}   {chunk:>14,}")


if __name__ == "__main__":
    main()
