"""A/B ``iter_batches(prefetch=True)`` against the synchronous iteration.

``iter_batches`` is pull-driven: the gather of batch N+1 starts only after the
consumer finishes batch N. ``prefetch=True`` gathers N+1 on a single background
thread while the consumer holds N, overlapping the read (the gather releases the
GIL) with the consumer's work. The win is realized only when the consumer is
slower than the gather and its work overlaps the gather (a compute / IO / GPU
consumer, not a bandwidth-bound one that contends with the gather for memory).

This drives a tunable, compute-bound consumer (``--consumer-reps`` passes of
``sqrt(|col|).sum()`` per batch, ALU-bound so it overlaps the bandwidth-bound
gather) over a multi-record store, ``copy=False`` (the streaming fast path), and
compares total wall with prefetch off vs on. The correctness gate asserts the two
accumulate the identical result before any timing.

Run on the deployment node with ``--tmpdir`` (or ``TMPDIR``) on the parallel
filesystem; local numbers are indicative only. Sweep ``--consumer-reps`` to trace
where the consumer cost crosses the gather cost (below it, prefetch is a no-op;
above it, the gather hides under the consumer).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import config, testing

_BATCH = "8 MiB"
_CORRECTNESS_ROWS = 200_000


def _consume(batch: colstore.ColStoreFrame, reps: int) -> float:
    """A deterministic, ALU-bound consumer: ``reps`` passes of sqrt(|col|).sum().

    Compute-bound (transcendental) rather than a plain streaming reduction, so its
    work overlaps the bandwidth-bound gather instead of contending for memory.
    """
    columns = batch.dict()
    total = 0.0
    for _ in range(reps):
        for value in columns.values():
            total += float(np.sqrt(np.abs(value)).sum())
    return total


def _run_pass(frame: colstore.ColStoreFrame, reps: int, *, prefetch: bool) -> float:
    total = 0.0
    for batch in frame.iter_batches(_BATCH, copy=False, prefetch=prefetch):
        total += _consume(batch, reps)
    return total


def check_correctness(directory: Path, args: argparse.Namespace) -> None:
    rows = min(_CORRECTNESS_ROWS, _c.scaled_rows(args.rows, args))
    path = directory / "correctness.cstore"
    path.unlink(missing_ok=True)
    testing.write_columns(
        path, testing.make_columns(rows, args.cols, dtype=args.dtype, seed=0), records=min(50, rows)
    ).close()
    store = colstore.open(path)
    try:
        off = _run_pass(store.edit(), args.consumer_reps, prefetch=False)
        on = _run_pass(store.edit(), args.consumer_reps, prefetch=True)
        if off != on:
            raise AssertionError(f"prefetch changed the consumed result: {off!r} != {on!r}")
    finally:
        store.close()
    print("  CORRECTNESS OK (prefetch on/off accumulate the identical result)\n")


def run_bench(directory: Path, args: argparse.Namespace) -> None:
    rows = _c.scaled_rows(args.rows, args)
    path = directory / f"src_r{rows}_c{args.cols}.cstore"
    path.unlink(missing_ok=True)
    testing.write_columns(
        path,
        testing.make_columns(rows, args.cols, dtype=args.dtype, seed=0),
        records=min(args.records, rows),
    ).close()
    store = colstore.open(path)
    try:
        print("Environment:")
        print(f"  rows={rows:,}  cols={args.cols}  records={args.records}  dtype={args.dtype}")
        print(
            f"  batch={_BATCH}  consumer_reps={args.consumer_reps}"
            f"  gather thread cap={config.get_gather_thread_cap()}"
        )
        print(f"  repeat={args.repeat}  warmup={args.warmup}  store dir={directory}\n")
        frame = store.edit()
        _c.compare(
            [
                ("prefetch off", lambda: _run_pass(frame, args.consumer_reps, prefetch=False)),
                ("prefetch on ", lambda: _run_pass(frame, args.consumer_reps, prefetch=True)),
            ],
            repeat=args.repeat,
            warmup=args.warmup,
            baseline=0,
            throughput_rows=rows,
        )
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(
        parser,
        repeat=5,
        warmup=2,
        rows=20_000_000,
        cols=8,
        dtype="float64",
        threads=True,
        scale=True,
        tmpdir=True,
    )
    parser.add_argument("--records", type=int, default=1000, help="records in the source store")
    parser.add_argument(
        "--consumer-reps", type=int, default=4, help="per-batch consumer passes (its cost)"
    )
    args = parser.parse_args()
    _c.apply_runtime_config(args)
    if args.tmpdir is not None:
        directory = Path(args.tmpdir)
        directory.mkdir(parents=True, exist_ok=True)
        if not args.skip_correctness:
            check_correctness(directory, args)
        if not args.skip_bench:
            run_bench(directory, args)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            if not args.skip_correctness:
                check_correctness(directory, args)
            if not args.skip_bench:
                run_bench(directory, args)


if __name__ == "__main__":
    main()
