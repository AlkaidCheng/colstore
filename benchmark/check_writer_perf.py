"""Compare streaming ColStoreWriter vs one-shot ``colstore.store`` perf.

Streaming a fixed total row count as many small records should be slower
per-row than one big write (more record headers, more counter updates),
but the overhead per record should be bounded and small. This benchmark
characterizes the overhead so future PRs can catch regressions.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import testing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(
        parser,
        repeat=3,
        rows=1_000_000,
        record_counts=[1, 10, 100, 1000],
        dtype="float32",
        tmpdir=True,
        skip_correctness=False,
    )
    args = parser.parse_args()

    dtype = np.dtype(args.dtype)
    tmpdir = args.tmpdir or Path(tempfile.mkdtemp(prefix="colstore_writer_bench"))
    tmpdir.mkdir(parents=True, exist_ok=True)

    data = testing.make_columns(args.rows, 1, dtype=args.dtype, seed=0)["c0"]

    print(f"Setup: {args.rows:,} rows of {dtype}")
    if args.skip_bench:
        return

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

        result = _c.profile(
            write_all, repeat=args.repeat, warmup=args.warmup, label=f"R={n_rec:<6}"
        )
        print(
            f"{result.report()}  {result.wall_ms * 1000 / n_rec:.1f}us/record  {chunk:,} rows/rec"
        )


if __name__ == "__main__":
    main()
