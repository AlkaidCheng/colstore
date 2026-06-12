"""Sweep the gather prefetch distance across access regimes.

The right prefetch distance is hardware-dependent: it must be large enough
that the prefetch covers the memory latency it is hiding (distance ~ latency /
per-iteration cost), but prefetching is pure overhead when the source is
already cache-resident. This script sweeps the distance for the three gather
kernels over the regimes where the answer differs, so the default can be
chosen from measurements on the *target* host rather than guessed:

  * source fits in L2/L3 vs source far exceeds it (DRAM-bound)
  * unsorted (random) vs sorted (hardware prefetcher already effective)
  * single-record element gather vs fused multi-record gather

Run on the deployment hardware:

    python benchmark/sweep_prefetch_distance.py
    python benchmark/sweep_prefetch_distance.py --thread 8   # at a real cap

Single-threaded by default so the per-element behavior is isolated; with
--thread the sweep runs at that thread cap, which shifts the knee (each
thread sees a fraction of the loop, but the same memory latency).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import colstore
from colstore import _gather as g  # type: ignore[attr-defined]
from colstore.reader import ColStoreReader

DISTANCES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def _time(fn, *, repeat: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _row(label: str, times_ms: list[float]) -> None:
    best_i = int(np.argmin(times_ms))
    cells = "".join(f"{t:8.2f}{'*' if i == best_i else ' '}" for i, t in enumerate(times_ms))
    base = times_ms[DISTANCES.index(8)]  # current default
    print(f"  {label:<34}{cells}  best=d{DISTANCES[best_i]} ({base / times_ms[best_i]:.2f}x vs d8)")


def sweep_single_record(total_rows: int, k: int, threads: int, repeat: int, label: str) -> None:
    rng = np.random.default_rng(7)
    source = rng.standard_normal(total_rows)  # float64
    unsorted_idx = rng.integers(0, total_rows, size=k).astype(np.int64)
    sorted_idx = np.sort(unsorted_idx)
    out = np.empty(k, dtype=np.float64)

    for name, idx in (("unsorted", unsorted_idx), ("sorted", sorted_idx)):
        times = [
            _time(lambda d=d, i=idx: g.gather(source, i, out, threads, d), repeat=repeat) * 1e3
            for d in DISTANCES
        ]
        _row(f"{label} {name}", times)


def sweep_multirecord(
    total_rows: int, n_records: int, k: int, threads: int, repeat: int, label: str
) -> None:
    rng = np.random.default_rng(7)
    full = rng.standard_normal(total_rows)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "s.cstore"
        per = total_rows // n_records
        with colstore.create(path) as w:
            off = 0
            for r in range(n_records):
                n = per if r < n_records - 1 else total_rows - off
                w.write({"a": full[off : off + n]})
                off += n
        ds = ColStoreReader(path)
        rsr, rsb, nrr = ds._record_starts_rows, ds._record_starts_bytes, ds._n_rows_per_record
        idx = rng.integers(0, total_rows, size=k).astype(np.int64)
        out = np.empty(k, dtype=np.float64)
        times = [
            _time(
                lambda dd=dd: g.gather_multirecord(
                    ds._file_mmap, idx, out, rsr, rsb, nrr, 0, threads, dd
                ),
                repeat=repeat,
            )
            * 1e3
            for dd in DISTANCES
        ]
        _row(f"{label} multirecord unsorted", times)
        ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread", type=int, default=1, help="kernel thread cap")
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument(
        "--big-rows",
        type=int,
        default=64_000_000,
        help="rows for the DRAM-bound regime (default 64M f8 = 512 MB)",
    )
    args = parser.parse_args()

    header = "".join(f"     d{d:<3}" for d in DISTANCES)
    print(f"Prefetch distance sweep (thread={args.thread}, best-of-{args.repeat}, ms)")
    print(f"  {'regime':<34}{header}")

    # Cache-resident-ish: 2M f8 = 16 MB (fits many L3s). Prefetch is expected
    # to matter little here; d0 may even win.
    sweep_single_record(2_000_000, 1_000_000, args.thread, args.repeat, "16MB src,")
    sweep_multirecord(2_000_000, 1000, 1_000_000, args.thread, args.repeat, "16MB src,")

    # DRAM-bound: well past any L3. This is the regime the distance exists
    # for; the knee here is the number that should drive the default.
    sweep_single_record(
        args.big_rows,
        2_000_000,
        args.thread,
        args.repeat,
        f"{args.big_rows * 8 // 1_000_000}MB src,",
    )
    sweep_multirecord(
        args.big_rows,
        1000,
        2_000_000,
        args.thread,
        args.repeat,
        f"{args.big_rows * 8 // 1_000_000}MB src,",
    )

    print(
        "\n  '*' marks the fastest distance per regime; the 'vs d8' column is the\n"
        "  gain over the current default. To adopt a value process-wide:\n"
        "      colstore.config.set_prefetch_distance(<d>)\n"
        "  A compiled-default change should come from the DRAM-bound rows on the\n"
        "  deployment host, not from the cache-resident ones. Alternatively, run\n"
        "      colstore.autotune.calibrate_prefetch(verbose=True)\n"
        "  once on this host: it sweeps these regimes itself, caches a per-regime\n"
        "  table, and the default 'auto' setting then resolves each gather from it."
    )


if __name__ == "__main__":
    main()
