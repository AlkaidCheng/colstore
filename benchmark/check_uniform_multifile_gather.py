"""Verify the uniform-grid multi-file gather: correctness and timing.

When a multi-file dataset's segment table is a *uniform grid* -- every segment the
same row count except possibly the global-last -- an unsorted fancy gather finds
each row's segment by a division ``s = idx / rows_per_segment`` (one magic-reciprocal
multiply) instead of a per-index binary search over the segment table. The
multi-file search is deeper than a single file's, so the division saves more as the
file/segment count grows; the route is a no-op fallback to the searching kernel
(``gather_segment``) when the grid does not hold, and sorted reads keep the cursor
walk regardless.

The baseline forces the searching kernel through the documented seam
(``_uniform_segment_grid`` -> None), so both sides run the same dataset and indices.

Run on the deployment hardware:

    python benchmark/check_uniform_multifile_gather.py
    python benchmark/check_uniform_multifile_gather.py --skip-bench   # correctness only

Expected shape (local indicative A/B): ~1.5-2.8x, growing with the segment count
(the search depth the division replaces).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import testing
from colstore.dataset import ColStoreDataset


def _build_grid(directory: Path, n_files: int, rows_per_file: int, n_cols: int):
    """``n_files`` equal-sized single-record files: a uniform global segment grid."""
    total = n_files * rows_per_file
    full = testing.make_columns(total, n_cols, seed=0)
    names = list(full)
    paths = []
    for file_index in range(n_files):
        lo = file_index * rows_per_file
        part = {name: full[name][lo : lo + rows_per_file] for name in names}
        path = directory / f"f{file_index}.cstore"
        colstore.store(part, path, show_progress=False).close()
        paths.append(path)
    return paths, names, full


def _force_search():
    """Disable the uniform-grid detection so the read takes the searching kernel."""
    original = ColStoreDataset._uniform_segment_grid
    ColStoreDataset._uniform_segment_grid = lambda self, starts: None  # type: ignore[assignment]
    return original


def _restore(original) -> None:
    ColStoreDataset._uniform_segment_grid = original  # type: ignore[assignment]


def _read_search(dataset, indices, names):
    """One read forced onto the searching kernel."""
    original = _force_search()
    try:
        return dataset[indices, names].dict()
    finally:
        _restore(original)


def check_correctness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths, names, full = _build_grid(Path(tmp), 16, 5_000, 3)
        dataset = colstore.open(paths)
        total = dataset.n_rows
        indices = np.random.default_rng(5).integers(0, total, size=10_000).astype(np.int64)
        routed = dataset[indices, names].dict()
        fallback = _read_search(dataset, indices, names)
        for name in names:
            assert np.array_equal(routed[name], full[name][indices]), name
            assert np.array_equal(routed[name], fallback[name]), name
        # Sorted selector keeps the cursor walk; must still agree.
        sorted_indices = np.sort(indices)
        sorted_read = dataset[sorted_indices, names].dict()
        for name in names:
            assert np.array_equal(sorted_read[name], full[name][sorted_indices]), name
        dataset.close()
    print("  ALL CORRECTNESS CHECKS PASSED (uniform == searching == ground truth)\n")


def run_bench(args: argparse.Namespace) -> None:
    for n_files in args.file_counts:
        rows_per_file = args.rows // n_files
        total = rows_per_file * n_files
        with tempfile.TemporaryDirectory() as tmp:
            paths, names, _ = _build_grid(Path(tmp), n_files, rows_per_file, args.cols)
            dataset = colstore.open(paths)
            indices = (
                np.random.default_rng(1).integers(0, total, size=args.indices).astype(np.int64)
            )
            dataset[indices[:1000], names].dict()  # warm mmaps + segment-table memo
            print(
                f"files={n_files:<6} rows/file={rows_per_file:<9} C={args.cols} K={args.indices:,}"
            )
            _c.compare(
                [
                    ("search (1 col)", lambda d=dataset, i=indices: _read_search(d, i, ["c0"])),
                    ("uniform (1 col)", lambda d=dataset, i=indices: d[i, "c0"].array()),
                ],
                repeat=args.repeat,
                warmup=args.warmup,
                baseline=0,
            )
            _c.compare(
                [
                    ("search (dict)", lambda d=dataset, i=indices, n=names: _read_search(d, i, n)),
                    ("uniform (dict)", lambda d=dataset, i=indices, n=names: d[i, n].dict()),
                ],
                repeat=args.repeat,
                warmup=args.warmup,
                baseline=0,
            )
            print()
            dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(parser, rows=20_000_000, cols=4, indices=2_000_000, threads=True)
    parser.add_argument(
        "--file-counts",
        type=int,
        nargs="+",
        default=[16, 64, 256, 1024],
        help="file/segment counts to sweep (the search depth the division replaces)",
    )
    args = parser.parse_args()
    _c.apply_runtime_config(args)
    if not args.skip_correctness:
        check_correctness()
    if not args.skip_bench:
        run_bench(args)


if __name__ == "__main__":
    main()
