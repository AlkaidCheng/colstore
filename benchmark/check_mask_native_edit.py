"""Verify the mask-native gather on the *edit* path: correctness and timing.

``ds[mask].edit()`` keeps a boolean mask as the frame's row selection only when it
will gather mask-natively -- a multi-record store at or above the density gate --
and otherwise (a single-record store, or a sparse selection) lowers it to int64
indices at construction, the sorted-fancy path the pre-change seam always took.
The mask-native kernel exists only for multi-record (interleaved) layouts; a
single-record store is contiguous and gathered directly by index.

This A/B times ``ds[...].edit()`` two ways across a density sweep -- ``lower``
(convert the mask to indices with ``np.flatnonzero``, then gather: the pre-change
behavior) vs ``mask`` (keep it where it gathers mask-natively). Each pays its own
conversion inside the timed cell, so the comparison is apples-to-apples: on a
multi-record store a dense mask wins (mask-native); a sparse selection, and any
single-record store, read as parity (both lower to the same fancy gather). Both
produce identical output (checked first).

The mask route is gated on density (per-host calibrated via
``colstore calibrate mask-density``; compiled default 0.0 routes at every
density), so the sweep brackets the gate to let host data confirm or move it.

Run on the deployment hardware (quiet compute node), both thread regimes:

    python benchmark/check_mask_native_edit.py
    OMP_NUM_THREADS=8 python benchmark/check_mask_native_edit.py
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import testing

RECORD_CUT_FRACTION = 0.3


def _build_store(directory: Path, n_records: int, rows: int):
    total = n_records * rows
    full = testing.make_columns(total, 2, names=("a", "b"), seed=0)
    path = directory / f"r{n_records}.cstore"
    testing.write_columns(path, full, records=n_records).close()
    return path, full, total


def _lower(store, mask, cols):
    """The pre-change edit path: lower the mask to indices first, then gather."""
    view = store[np.flatnonzero(mask), cols].edit()
    return view.dict() if isinstance(cols, list) else view.array(cols)


def _kept(store, mask, cols):
    """The kept-mask edit path; the frame lowers it internally where it would not gather
    mask-natively (a single-record store, or a selection below the density gate)."""
    view = store[mask, cols].edit()
    return view.dict() if isinstance(cols, list) else view.array(cols)


def check_correctness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(1)
        path, full, total = _build_store(Path(tmp), 500, 300)
        store = colstore.open(path)
        masks = [rng.random(total) < p for p in (0.9, 0.4, 0.05)]
        masks += [np.ones(total, dtype=bool), np.zeros(total, dtype=bool)]
        for mask in masks:
            idx = np.flatnonzero(mask)
            mask_dict = store[mask].edit().dict()
            index_dict = store[idx].edit().dict()
            one = store[mask, "a"].edit().array("a")
            for name in ("a", "b"):
                truth = full[name][mask]
                assert np.array_equal(mask_dict[name], truth), name
                assert np.array_equal(index_dict[name], truth), name
            assert np.array_equal(one, full["a"][mask])
        store.close()
    print("  ALL CORRECTNESS CHECKS PASSED (mask == index == ground truth)\n")


def run_bench(args: argparse.Namespace) -> None:
    for n_records in args.record_counts:
        rows = args.rows // n_records
        with tempfile.TemporaryDirectory() as tmp:
            path, _, total = _build_store(Path(tmp), n_records, rows)
            rng = np.random.default_rng(2)
            store = colstore.open(path)
            cases = [(f"density {p}", rng.random(total) < p) for p in args.densities]
            cases.append(
                (
                    f"record cut {RECORD_CUT_FRACTION}",
                    np.repeat(rng.random(n_records) < RECORD_CUT_FRACTION, rows),
                )
            )
            for label, mask in cases:
                store[mask].edit().dict()  # fault pages first
                print(
                    f"R={n_records:<7} rows/rec={rows:<7} {label:<16} selected={int(mask.sum()):,}"
                )
                _c.compare(
                    [
                        ("lower 1col", lambda s=store, m=mask: _lower(s, m, "a")),
                        ("mask  1col", lambda s=store, m=mask: _kept(s, m, "a")),
                    ],
                    repeat=args.repeat,
                    warmup=args.warmup,
                    baseline=0,
                )
                _c.compare(
                    [
                        ("lower 2col", lambda s=store, m=mask: _lower(s, m, ["a", "b"])),
                        ("mask  2col", lambda s=store, m=mask: _kept(s, m, ["a", "b"])),
                    ],
                    repeat=args.repeat,
                    warmup=args.warmup,
                    baseline=0,
                )
                print()
            store.close()


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
