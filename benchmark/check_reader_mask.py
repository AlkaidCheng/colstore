"""Time a single-file multi-record boolean-mask read (the segment mask kernel).

A boolean mask on a multi-record ``ColStoreReader`` gathers through the unified
segment mask kernel (``colstore_gather_segment_mask``) over the reader's per-column
segment table. This benchmark A/Bs that kernel route against the index-lowering
fallback (``np.flatnonzero`` + the fancy gather) by toggling the mask-density gate,
and sweeps the record count and mask density.

It also serves as a cross-branch check that the unification did not regress the
single-file path: run it on this branch (the segment kernel) and on ``main``
(the previous ``gather_multirecord_mask``); ``ds[mask].dict()`` is the same public
read on both, so the kernel column is directly comparable.

The correctness gate asserts both routes match the NumPy oracle before timing.
Run on the deployment node with ``--tmpdir`` on the parallel filesystem; local
numbers are indicative.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import config, kernels, testing

_FRACTION_SEED = 20260623


def _build(path: Path, rows: int, cols: int, records: int, dtype: str) -> dict[str, np.ndarray]:
    path.unlink(missing_ok=True)
    columns = testing.make_columns(rows, cols, dtype=dtype, seed=0)
    testing.write_columns(path, columns, records=min(records, rows)).close()
    return columns


def _read(reader: colstore.ColStoreReader, mask: np.ndarray, names: list[str], gate: float):
    config.set_mask_density_gate(gate)  # 0 -> kernel, high -> flatnonzero fallback
    return reader[mask, names].dict()


def check_correctness(directory: Path, args: argparse.Namespace) -> None:
    path = directory / "src.cstore"
    rows = min(args.rows, 500_000)
    records = args.record_counts[len(args.record_counts) // 2]
    columns = _build(path, rows, args.cols, records, args.dtype)
    names = list(columns)
    saved = config.get_mask_density_gate()
    try:
        with colstore.open(path) as reader:
            for fraction in (0.01, 0.5, 1.0):
                mask = (
                    np.random.default_rng(_FRACTION_SEED).random(reader.n_rows) < fraction
                    if fraction < 1.0
                    else np.ones(reader.n_rows, bool)
                )
                for gate in (0.0, 2.0):  # kernel, then the flatnonzero fallback
                    got = _read(reader, mask, names, gate)
                    for name in names:
                        _c.check_equal(
                            got[name], columns[name][mask], f"g={gate} f={fraction}:{name}"
                        )
    finally:
        config.set_mask_density_gate(saved)
    path.unlink(missing_ok=True)
    print("  CORRECTNESS OK (kernel == flatnonzero fallback == numpy oracle)\n")


def run_bench(directory: Path, args: argparse.Namespace) -> None:
    print("Environment:")
    print(f"  rows={args.rows:,}  cols={args.cols}  dtype={args.dtype}")
    print(f"  records swept={args.record_counts}  fraction={args.fraction}\n")
    saved = config.get_mask_density_gate()
    try:
        for records in args.record_counts:
            path = directory / f"src_r{records}.cstore"
            columns = _build(path, args.rows, args.cols, records, args.dtype)
            names = list(columns)
            with colstore.open(path) as reader:
                mask = np.random.default_rng(_FRACTION_SEED).random(reader.n_rows) < args.fraction
                sel = int(mask.sum())
                print(f"=== {records} records, {reader.n_rows:,} rows, {sel:,} selected ===")
                variants: list[tuple[str, Callable[[], object]]] = [
                    ("flatnonzero", lambda r=reader, m=mask, c=names: _read(r, m, c, 2.0)),
                    ("kernel     ", lambda r=reader, m=mask, c=names: _read(r, m, c, 0.0)),
                ]
                _c.compare(variants, repeat=args.repeat, warmup=args.warmup, baseline=0)
                print()
            path.unlink(missing_ok=True)
    finally:
        config.set_mask_density_gate(saved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(
        parser,
        repeat=7,
        warmup=2,
        rows=10_000_000,
        cols=8,
        dtype="float64",
        record_counts=[100, 2000, 50000],
        threads=True,
        tmpdir=True,
    )
    parser.add_argument("--fraction", type=float, default=0.5, help="mask selection fraction")
    args = parser.parse_args()
    _c.apply_runtime_config(args)

    if not kernels.cpp_available():
        raise SystemExit("the compiled extension is required for this benchmark")

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
