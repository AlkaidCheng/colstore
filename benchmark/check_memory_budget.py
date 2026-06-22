"""Validate the default streaming memory budget across practical workloads.

The write / reduction / iter_batches paths evaluate and gather in row-range batches sized
by ``memory_budget``; the sweet spot trades cache locality (a smaller batch stays in
last-level cache) against per-batch overhead (too-small batches pay fixed setup per batch).
A single workload is not enough to trust a default, so this sweeps the budget across a set
of representative scenarios and reports, per scenario, where the default (32 MiB) lands
versus the fastest budget and versus one batch (effectively unlimited):

* selectivity -- tight (5%) vs mid (50%) filtered writes (the gather pattern differs);
* schema width -- narrow (1 col) vs standard (4) vs wide (12, mixed dtype) writes;
* transform depth -- a couple of derived columns, and a deeper expression whose per-batch
  intermediates are NOT counted in the budget (the case most likely to want a smaller one);
* the other budget-consuming paths -- a full-column reduction and an iter_batches consume;
* a passthrough write as a control (the no-transform merge-copy ignores the budget).

Filter thresholds are quantile-picked so selectivity is exact regardless of the data
distribution. Run on the deployment node, both thread regimes; ``--records 1`` for the
single-record (contiguous) layout:

    python benchmark/check_memory_budget.py
    OMP_NUM_THREADS=8 python benchmark/check_memory_budget.py --records 1
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import col, config, testing
from colstore.frame import result_dtype

_NCOLS = 12
_NAMES = tuple(f"c{i}" for i in range(_NCOLS))
_DTYPES = ("f8",) * 8 + ("f4",) * 2 + ("i2",) * 2  # mixed widths, like a real schema
_STD = ["c0", "c1", "c2", "c3"]
_BUDGETS = [
    ("1 batch", None),
    ("128 MiB", 128 << 20),
    ("64 MiB", 64 << 20),
    ("32 MiB", 32 << 20),
    ("16 MiB", 16 << 20),
    ("8 MiB", 8 << 20),
]
_DEFAULT = "32 MiB"


def _build_store(directory: Path, n_records: int, rows: int):
    total = n_records * rows
    full = testing.make_columns(total, _NCOLS, dtype=_DTYPES, names=_NAMES, seed=0)
    path = directory / f"store_r{n_records}.cstore"
    testing.write_columns(path, full, records=n_records).close()
    return path, total, full


def _frame_bytes_per_row(frame) -> int:
    return sum(result_dtype(frame[name]).itemsize for name in frame.columns)


def _scenarios(thr50: float, thr05: float):
    """[(name, make_frame, kind)] -- make_frame(store) -> frame; kind in write/reduce/iter."""

    def passthrough(s):
        return s.edit()

    def filt(cols, thr):
        return lambda s: s.edit().select(*cols).where(col("c0") > thr)

    def transform(s):
        cf = s.edit().select(*_STD).where(col("c0") > thr50)
        return cf.assign(p=cf["c1"] + cf["c2"], q=cf["c2"] * cf["c3"])

    def deep(s):
        cf = s.edit().select(*_STD).where(col("c0") > thr50)
        return cf.assign(r=np.sqrt(cf["c1"] ** 2 + cf["c2"] ** 2 + cf["c3"] ** 2))

    return [
        ("passthrough (control)", passthrough, "write"),
        ("filter 50% std (4 col)", filt(_STD, thr50), "write"),
        ("filter 5% std (tight)", filt(_STD, thr05), "write"),
        ("filter 50% wide (12 col)", filt(list(_NAMES), thr50), "write"),
        ("filter 50% narrow (1 col)", filt(["c0"], thr50), "write"),
        ("transform (+2 derived)", transform, "write"),
        ("deep transform (sqrt sum)", deep, "write"),
        ("reduction sum", passthrough, "reduce"),
        ("iter_batches consume", filt(list(_NAMES), thr50), "iter"),
    ]


def _op(frame, kind: str, out: Path, budget: int):
    if kind == "write":
        return lambda: frame.write(out, memory_budget=budget).close()
    if kind == "reduce":
        return lambda: (config.set_default_memory_budget(budget), frame.sum("c0"))[1]
    return lambda: (
        config.set_default_memory_budget(budget),
        [b.recarray() for b in frame.iter_batches()],
    )[1]


def _sweep(name: str, frame, kind: str, total: int, out: Path, args: argparse.Namespace):
    if kind == "reduce":
        rows = total
        stream_bytes = total * result_dtype(frame["c0"]).itemsize
    else:
        rows = frame.n_rows
        stream_bytes = rows * _frame_bytes_per_row(frame)
    budgets = [(lbl, stream_bytes * 2 if b is None else b) for lbl, b in _BUDGETS]
    setups = [lambda: out.unlink(missing_ok=True)] * len(budgets) if kind == "write" else None
    print(f"\n[{name}]  rows={rows:,}  stream~{stream_bytes / 2**20:.0f} MiB")
    results = _c.compare(
        [(f"{lbl:>8}", _op(frame, kind, out, b)) for lbl, b in budgets],
        setups=setups,
        repeat=args.repeat,
        warmup=args.warmup,
        baseline=0,
        throughput_rows=rows,
    )
    config.set_default_memory_budget(32 << 20)  # restore after reduce/iter mutate it
    walls = [r.wall_ms for r in results]
    labels = [lbl for lbl, _ in budgets]
    best = min(range(len(walls)), key=lambda i: walls[i])
    i32 = labels.index(_DEFAULT)
    return name, labels[best], walls[best] / walls[i32], walls[0] / walls[i32]


def _print_summary(summary) -> None:
    print("\n" + "=" * 74)
    print(f"SUMMARY (default = {_DEFAULT}; 32MiB/best <=1.0, 32MiB/1batch >1.0 means faster)")
    print(f"{'scenario':<28}{'best':>10}{'32MiB/best':>13}{'32MiB/1batch':>14}")
    print("-" * 74)
    worst = 1.0
    for name, best, r_best, r_1b in summary:
        print(f"{name:<28}{best:>10}{r_best:>12.2f}x{r_1b:>13.2f}x")
        worst = min(worst, r_best)
    print("-" * 74)
    verdict = "GOOD" if worst >= 0.95 else ("OK" if worst >= 0.90 else "REVISIT")
    print(f"worst case: 32 MiB is {(1 - worst) * 100:.0f}% off the fastest budget -> {verdict}")


def check_correctness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        path, _, full = _build_store(tmpd, 50, 1000)
        thr = float(np.quantile(full["c0"], 0.5))
        store = colstore.open(path)
        out = tmpd / "o.cstore"
        cf = store.edit().select(*_STD).where(col("c0") > thr)
        cf = cf.assign(p=cf["c1"] + cf["c2"])
        ref = None
        for budget in (1 << 30, 8 << 10):  # one batch vs a tiny, many-batch budget
            reader = cf.write(out, memory_budget=budget)
            try:
                got = reader.dict()
            finally:
                reader.close()
            out.unlink()
            if ref is None:
                ref = got
            else:
                for name in ref:
                    assert np.array_equal(got[name], ref[name]), f"write {name}"
        config.set_default_memory_budget(1 << 30)
        big = store.edit().sum("c0")
        big_iter = np.concatenate(
            [b.dict()["c0"] for b in store.edit().where(col("c0") > thr).iter_batches()]
        )
        config.set_default_memory_budget(8 << 10)
        small = store.edit().sum("c0")
        small_iter = np.concatenate(
            [b.dict()["c0"] for b in store.edit().where(col("c0") > thr).iter_batches()]
        )
        config.set_default_memory_budget(32 << 20)
        assert np.isclose(big, small), "reduction not budget-invariant"
        assert np.array_equal(big_iter, small_iter), "iter_batches not budget-invariant"
        store.close()
    print("  CORRECTNESS OK (write / reduction / iter_batches budget-invariant)\n")


def run_bench(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        rows_per = args.rows // args.records
        path, total, full = _build_store(tmpd, args.records, rows_per)
        thr50 = float(np.quantile(full["c0"], 0.5))
        thr05 = float(np.quantile(full["c0"], 0.95))
        store = colstore.open(path)
        out = tmpd / "out.cstore"
        print(f"store: records={args.records} rows={total:,} cols={_NCOLS} (mixed dtype)")
        summary = [
            _sweep(name, make_frame(store), kind, total, out, args)
            for name, make_frame, kind in _scenarios(thr50, thr05)
        ]
        store.close()
        _print_summary(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(parser, rows=30_000_000, threads=True)
    parser.add_argument("--records", type=int, default=1000, help="record count in the source")
    args = parser.parse_args()
    _c.apply_runtime_config(args)
    if not args.skip_correctness:
        check_correctness()
    if not args.skip_bench:
        run_bench(args)


if __name__ == "__main__":
    main()
