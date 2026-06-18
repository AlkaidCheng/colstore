"""Benchmark: multi-file dataset read overhead vs. one equivalent file.

A ``ColStoreDataset`` reads N files as one logical table by decomposing each
selection against the files' cumulative row offsets, dispatching per file, and
stitching the results. This measures the cost of that decomposition against the
single equivalent file for three things:

  whole read    : ``dataset[col].array()`` / ``dataset.dict()`` vs. the single file
  fancy gather  : a random cross-file fancy index (the group-by-file + scatter-back
                  path) vs. the single file
  eager write   : the throughput of ``concat(parts, out=...)`` (rows/s)

The single-file baseline is built with ``colstore.concat(parts, out=...)``, so the
A/B compares the lazy multi-file path against the eager single-file result of the
very same data; the correctness gate asserts the two agree before any timing.

Runs are interleaved A/B across rounds (via ``_common.compare``), so page-cache
and scheduler state stay comparable. No numbers are baked in -- run it on the
target node.

    PYTHONPATH=src python benchmark/check_dataset_concat.py
    PYTHONPATH=src python benchmark/check_dataset_concat.py --files 16 --rows 2000000
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import testing


def banner(s: str) -> None:
    print(f"\n=== {s} ===")


def build_parts(td: str, n_files: int, rows_per_file: int, cols: int, dtype: str) -> list[Path]:
    """Write ``n_files`` single-record stores (distinct seeds); return their paths."""
    paths = []
    for i in range(n_files):
        path = Path(td) / f"part_{i:03d}.cstore"
        testing.make_store(path, rows=rows_per_file, cols=cols, dtype=dtype, seed=i).close()
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=8, help="number of files in the dataset")
    _c.add_common_args(
        parser,
        repeat=5,
        warmup=2,
        rows=1_000_000,
        cols=4,
        indices=1_000_000,
        dtype="float64",
        threads=True,
        scale=True,
    )
    args = parser.parse_args()
    _c.apply_runtime_config(args)

    rows_per_file = _c.scaled_rows(args.rows, args)
    total_rows = rows_per_file * args.files

    print("Environment:")
    print(f"  cpp_available = {_c.cpp_available()}")
    print(
        f"  files={args.files}  rows/file={rows_per_file:,}  "
        f"total={total_rows:,}  cols={args.cols}  dtype={args.dtype}"
    )

    with tempfile.TemporaryDirectory() as td:
        parts = build_parts(td, args.files, rows_per_file, args.cols, args.dtype)
        single = colstore.concat(parts, out=Path(td) / "combined.cstore")  # eager single file
        dataset = colstore.open(parts)  # lazy multi-file
        try:
            col = dataset.columns[0]

            # ---- Whole-store read: one column ----------------------------------
            _c.check_equal(dataset[col].array(), single[col].array(), "whole/one-column")
            banner(f"WHOLE READ (one column): {args.files} files vs 1 file ({total_rows:,} rows)")
            _c.compare(
                [
                    (
                        f"dataset[{col!r}].array()  ({args.files} files)",
                        lambda: dataset[col].array(),
                    ),
                    (f"single[{col!r}].array()   (1 file)", lambda: single[col].array()),
                ],
                repeat=args.repeat,
                warmup=args.warmup,
                baseline=1,
                throughput_rows=total_rows,
            )

            # ---- Whole-store read: all columns ---------------------------------
            _c.check_equal(dataset.dict()[col], single.dict()[col], "whole/dict")
            banner(f"WHOLE READ (dict, {args.cols} cols): {args.files} files vs 1 file")
            _c.compare(
                [
                    (f"dataset.dict()  ({args.files} files)", lambda: dataset.dict()),
                    ("single.dict()   (1 file)", lambda: single.dict()),
                ],
                repeat=args.repeat,
                warmup=args.warmup,
                baseline=1,
                throughput_rows=total_rows,
            )

            # ---- Cross-file fancy gather (scatter-back) ------------------------
            rng = np.random.default_rng(0)
            n_idx = _c.scaled_rows(args.indices, args)
            indices = rng.integers(0, total_rows, size=n_idx, dtype=np.int64)
            _c.check_equal(dataset[indices, col].array(), single[indices, col].array(), "fancy")
            banner(
                f"FANCY GATHER (scatter-back): {n_idx:,} random rows, {args.files} files vs 1 file"
            )
            _c.compare(
                [
                    (
                        f"dataset[idx, {col!r}]  ({args.files} files)",
                        lambda: dataset[indices, col].array(),
                    ),
                    (f"single[idx, {col!r}]   (1 file)", lambda: single[indices, col].array()),
                ],
                repeat=args.repeat,
                warmup=args.warmup,
                baseline=1,
                throughput_rows=n_idx,
            )
        finally:
            dataset.close()
            single.close()

        # ---- Eager concat write throughput ------------------------------------
        # A fresh destination each timed run so nothing is overwritten in place.
        banner(f"EAGER WRITE: concat({args.files} files) -> 1 file ({total_rows:,} rows)")
        counter = {"i": 0}

        def write_once() -> None:
            out = Path(td) / f"out_{counter['i']}.cstore"
            counter["i"] += 1
            colstore.concat(parts, out=out).close()

        stats = _c.time_stats(write_once, repeat=max(3, args.repeat // 2), warmup=1)
        throughput = total_rows / (stats.median_ms / 1000.0) if stats.median_ms > 0 else 0.0
        print(
            f"  median={stats.median_ms:9.1f} ms   min={stats.min_ms:9.1f} ms   "
            f"({throughput / 1e6:6.1f}M rows/s)"
        )


if __name__ == "__main__":
    main()
