"""Measure how much the streaming write memory budget costs throughput.

A filtered (or transforming) ``write()`` evaluates and gathers the selected rows in
row-range batches sized by ``memory_budget``; a smaller budget means more, smaller batches
(more per-batch gather + write-loop overhead). This sweeps the budget on a filtered write --
from one batch (effectively unlimited) down -- to quantify how much bounding peak RAM costs,
so the default budget (currently 128 MiB) can be set from data rather than assumption.

An unfiltered, no-transform ``write()`` takes the raw merge-copy fast path (no batching), so
the budget does not apply there; ``iter_batches`` is per-batch by design (its memory is the
current batch) and is intentionally not swept here.

Run on the deployment node, both thread regimes:

    python benchmark/check_memory_budget.py
    OMP_NUM_THREADS=8 python benchmark/check_memory_budget.py
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import col, testing

_BYTES_PER_ROW = 4 * 8  # a, b, c, d are float64


def _build_store(directory: Path, n_records: int, rows: int):
    total = n_records * rows
    full = testing.make_columns(total, 4, names=("a", "b", "c", "d"), seed=0)
    path = directory / f"r{n_records}.cstore"
    testing.write_columns(path, full, records=n_records).close()
    return path, total


def _budgets(output_bytes: int) -> list[tuple[str, int]]:
    return [
        ("1 batch", output_bytes * 2),
        ("128 MiB", 128 << 20),
        ("32 MiB", 32 << 20),
        ("8 MiB", 8 << 20),
    ]


def check_correctness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        path, _ = _build_store(tmpd, 100, 1000)
        store = colstore.open(path)
        cf = store.edit().where(col("a") > 0)
        ref: dict | None = None
        for budget in (1 << 30, 8 << 10):  # one batch vs a tiny, many-batch budget
            reader = cf.write(tmpd / "out.cstore", memory_budget=budget)
            try:
                got = reader.dict()
            finally:
                reader.close()
            (tmpd / "out.cstore").unlink()
            if ref is None:
                ref = got
            else:
                for name in ref:
                    assert np.array_equal(got[name], ref[name]), name
        store.close()
    print("  CORRECTNESS OK (write output identical across budgets)\n")


def run_bench(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        rows = args.rows // args.records
        path, total = _build_store(tmpd, args.records, rows)
        store = colstore.open(path)
        cf = store.edit().where(col("a") > 0)  # filtered write -> batched gather; budget applies
        selected = cf.n_rows
        output_bytes = selected * _BYTES_PER_ROW
        out = tmpd / "out.cstore"

        def write_at(budget: int) -> None:
            reader = cf.write(out, memory_budget=budget)
            reader.close()

        print(
            f"records={args.records} rows={total:,} selected={selected:,} "
            f"output~{output_bytes / 2**20:.0f} MiB"
        )
        budgets = _budgets(output_bytes)
        _c.compare(
            [(f"write {label:>8}", (lambda b=b: write_at(b))) for label, b in budgets],
            setups=[(lambda: out.unlink(missing_ok=True)) for _ in budgets],
            repeat=args.repeat,
            warmup=args.warmup,
            baseline=0,
            throughput_rows=selected,
        )
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(parser, rows=10_000_000, threads=True)
    parser.add_argument("--records", type=int, default=1000, help="record count in the source")
    args = parser.parse_args()
    _c.apply_runtime_config(args)
    if not args.skip_correctness:
        check_correctness()
    if not args.skip_bench:
        run_bench(args)


if __name__ == "__main__":
    main()
