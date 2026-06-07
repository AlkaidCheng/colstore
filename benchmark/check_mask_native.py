"""Verify the boolean-mask-native read path: correctness and timing.

Boolean masks previously lowered to int64 indices immediately
(np.flatnonzero), paying the O(N) conversion, an 8-bytes-per-selected-row
index allocation, the sortedness pass, and 8 bytes/element of index
traffic inside the sorted kernel -- per column. The mask kernel reads the
1-byte-per-row mask directly with a monotone record walk, processing the
mask a word (8 rows) at a time: all-zero words skip, runs of all-ones
words become one memcpy (run coalescing costs nothing here because the
mask byte is the datum being scanned anyway), and mixed words compact
branchlessly, so the per-element branch that mispredicts at mid densities
never executes.

The route is gated on mask density (selected fraction): below the gate,
per-column index traffic undercuts re-reading the mask, and the lowered
flatnonzero path runs unchanged. Single-record stores always lower
(preserving the backend parameter's contract on fancy reads).

Run on the deployment hardware (quiet compute node), both thread regimes:

    python benchmark/check_mask_native.py
    OMP_NUM_THREADS=8 python benchmark/check_mask_native.py

The density sweep brackets the gate so host data can confirm or move it;
the gate is per-host calibrated (`colstore calibrate mask-density`), with a
compiled default of 0.0 (route on at every density) when uncalibrated.
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
from colstore import config

LAYOUTS = ((1_000, 20_000), (10_000, 2_000), (100_000, 200))
DENSITIES = (0.9, 0.5, 0.2, 0.1, 0.05, 0.01)
RECORD_CUT_FRACTION = 0.3


class _force_lowered:
    def __enter__(self):
        self._original = config.get_mask_density_gate()
        config.set_mask_density_gate(2.0)  # no selected fraction reaches it
        return self

    def __exit__(self, *exc):
        config.set_mask_density_gate(self._original)
        return False


def _build_store(directory: Path, n_records: int, rows: int):
    rng = np.random.default_rng(0)
    total = n_records * rows
    full = {"a": rng.standard_normal(total), "b": rng.standard_normal(total)}
    path = directory / f"r{n_records}.cstore"
    with colstore.create(path) as writer:
        for r in range(n_records):
            writer.write({k: v[r * rows : (r + 1) * rows] for k, v in full.items()})
    return path, full, total


def check_correctness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(1)
        path, full, total = _build_store(Path(tmp), 500, 300)
        dataset = colstore.open(path)
        masks = [rng.random(total) < p for p in (0.9, 0.4, 0.05)]
        masks += [np.ones(total, dtype=bool), np.zeros(total, dtype=bool)]
        for mask in masks:
            one = dataset[mask, "a"].array()
            many = dataset[mask, ["a", "b"]].dict()
            with _force_lowered():
                one_lowered = dataset[mask, "a"].array()
                many_lowered = dataset[mask, ["a", "b"]].dict()
            assert np.array_equal(one, full["a"][mask])
            assert np.array_equal(one, one_lowered)
            for name in ("a", "b"):
                assert np.array_equal(many[name], full[name][mask]), name
                assert np.array_equal(many[name], many_lowered[name]), name
        dataset.close()
    print("  ALL CORRECTNESS CHECKS PASSED (mask route == lowered route == ground truth)\n")


def _best(f, repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - start)
    return best


def run_bench(repeat: int) -> None:
    for n_records, rows in LAYOUTS:
        with tempfile.TemporaryDirectory() as tmp:
            path, _, total = _build_store(Path(tmp), n_records, rows)
            rng = np.random.default_rng(2)
            dataset = colstore.open(path)
            print(f"layout R={n_records} rows/rec={rows}")
            print(
                f"  {'selector':<24}{'K':>10}"
                f"{'1col low':>10}{'1col mask':>10}{'x':>6}"
                f"{'2col low':>10}{'2col mask':>10}{'x':>6}"
            )
            cases = [(f"density {p}", rng.random(total) < p) for p in DENSITIES]
            cases.append(
                (
                    f"record cut {RECORD_CUT_FRACTION}",
                    np.repeat(rng.random(n_records) < RECORD_CUT_FRACTION, rows),
                )
            )
            for label, mask in cases:
                dataset[mask, "a"].array()  # fault pages before either side
                t1_mask = _best(lambda m=mask, ds=dataset: ds[m, "a"].array(), repeat)
                t2_mask = _best(lambda m=mask, ds=dataset: ds[m, ["a", "b"]].dict(), repeat)
                with _force_lowered():
                    t1_low = _best(lambda m=mask, ds=dataset: ds[m, "a"].array(), repeat)
                    t2_low = _best(lambda m=mask, ds=dataset: ds[m, ["a", "b"]].dict(), repeat)
                print(
                    f"  {label:<24}{int(mask.sum()):>10}"
                    f"{t1_low * 1e3:>8.1f}ms{t1_mask * 1e3:>8.1f}ms{t1_low / t1_mask:>6.2f}"
                    f"{t2_low * 1e3:>8.1f}ms{t2_mask * 1e3:>8.1f}ms{t2_low / t2_mask:>6.2f}"
                )
            dataset.close()
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()
    check_correctness()
    if not args.skip_bench:
        run_bench(args.repeat)


if __name__ == "__main__":
    main()
