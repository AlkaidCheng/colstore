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
import tempfile
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import config, testing

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
    total = n_records * rows
    full = testing.make_columns(total, 2, names=("a", "b"), seed=0)
    path = directory / f"r{n_records}.cstore"
    testing.write_columns(path, full, records=n_records).close()
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


def _lowered(dataset, mask, cols):
    """One mask read forced through the lowered flatnonzero path."""
    with _force_lowered():
        view = dataset[mask, cols]
        return view.dict() if isinstance(cols, list) else view.array()


def run_bench(args: argparse.Namespace) -> None:
    for n_records in args.record_counts:
        rows = args.rows // n_records
        with tempfile.TemporaryDirectory() as tmp:
            path, _, total = _build_store(Path(tmp), n_records, rows)
            rng = np.random.default_rng(2)
            dataset = colstore.open(path)
            cases = [(f"density {p}", rng.random(total) < p) for p in args.densities]
            cases.append(
                (
                    f"record cut {RECORD_CUT_FRACTION}",
                    np.repeat(rng.random(n_records) < RECORD_CUT_FRACTION, rows),
                )
            )
            for label, mask in cases:
                dataset[mask, "a"].array()  # fault pages first
                print(
                    f"R={n_records:<7} rows/rec={rows:<7} {label:<16} selected={int(mask.sum()):,}"
                )
                _c.compare(
                    [
                        ("lowered 1col", lambda d=dataset, m=mask: _lowered(d, m, "a")),
                        ("mask    1col", lambda d=dataset, m=mask: d[m, "a"].array()),
                    ],
                    repeat=args.repeat,
                    warmup=args.warmup,
                    baseline=0,
                )
                _c.compare(
                    [
                        ("lowered 2col", lambda d=dataset, m=mask: _lowered(d, m, ["a", "b"])),
                        ("mask    2col", lambda d=dataset, m=mask: d[m, ["a", "b"]].dict()),
                    ],
                    repeat=args.repeat,
                    warmup=args.warmup,
                    baseline=0,
                )
                print()
            dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(
        parser,
        rows=20_000_000,
        record_counts=[1_000, 10_000, 100_000],
        threads=True,
    )
    parser.add_argument(
        "--densities",
        type=float,
        nargs="+",
        default=[0.9, 0.5, 0.2, 0.1, 0.05, 0.01],
        help="mask selected-fraction sweep",
    )
    args = parser.parse_args()
    _c.apply_runtime_config(args)
    if not args.skip_correctness:
        check_correctness()
    if not args.skip_bench:
        run_bench(args)


if __name__ == "__main__":
    main()
