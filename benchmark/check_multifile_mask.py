"""A/B the multi-file boolean-mask read: per-file lowering vs one global lowering.

A boolean row mask on a multi-file ``ColStoreDataset`` used to be split into
per-file sub-masks and gathered one ``(file, column)`` portion at a time, each
portion re-lowering its sub-mask to indices (``np.flatnonzero``) inside the child
reader -- so the boolean->index conversion ran once per (file x column) and the
read fanned out across ``files x columns`` threadpool jobs. ``_classify_rows`` now
lowers the mask once at the dataset level (``np.flatnonzero``) and routes it
through the native multi-file *sorted* fancy gather, the same path a fancy index
takes. ``flatnonzero`` is ascending == file order, so the output is byte-identical.

This benchmark times the public read (``new``, the routed path) against a faithful
reconstruction of the previous per-file path (``old``, built from the still-present
contiguous-fill primitives) and sweeps the file count, where the win grows: the
removed cost is per-file/per-column interpreter + threadpool overhead, and the
underlying selected-row gather is unchanged.

The correctness gate asserts both paths are byte-identical to the NumPy oracle
before any timing. Run on the deployment node with ``--tmpdir`` on the parallel
filesystem; local numbers are indicative.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import testing
from colstore.dataset import ColStoreDataset

_FRACTION_SEED = 20260623


def _build(directory: Path, n_files: int, rows_each: int, cols: int, records: int, dtype: str):
    """Write ``n_files`` multi-record files; return paths and the concatenated oracle."""
    paths = []
    blocks: dict[str, list[np.ndarray]] = {}
    for i in range(n_files):
        columns = testing.make_columns(rows_each, cols, dtype=dtype, seed=i)
        path = directory / f"part_{i:04d}.cstore"
        path.unlink(missing_ok=True)
        testing.write_columns(path, columns, records=min(records, rows_each)).close()
        paths.append(path)
        for name, values in columns.items():
            blocks.setdefault(name, []).append(values)
    return paths, {name: np.concatenate(parts) for name, parts in blocks.items()}


def _old_per_file(ds: ColStoreDataset, mask: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """Reconstruct the pre-reroute per-file mask path for the A/B baseline.

    Splits the mask into per-file sub-masks (popcount lengths) and fills each
    ``(file, column)`` portion through the dataset's contiguous-region fill jobs --
    exactly what ``_gather_many_contiguous`` did for a mask before the reroute.
    """
    offsets = np.asarray(ds._offsets)
    parts = []
    for file_index in range(len(ds._children)):
        lo, hi = int(offsets[file_index]), int(offsets[file_index + 1])
        sub = mask[lo:hi]
        if sub.any():
            parts.append((file_index, sub))
    lengths = [int(sub.sum()) for _, sub in parts]
    total = sum(lengths)
    out = {name: np.empty(total, dtype=ds._native_dtype(name)) for name in names}
    ds._fill_contiguous_columns(out, names, ds._contiguous_regions(parts, lengths))
    return out


def _new_routed(ds: ColStoreDataset, mask: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """The shipped path: the mask is lowered once and routed through the fancy gather."""
    return ds._gather_many(names, mask)


def _mask(n_rows: int, fraction: float) -> np.ndarray:
    return np.random.default_rng(_FRACTION_SEED).random(n_rows) < fraction


def check_correctness(directory: Path, args: argparse.Namespace) -> None:
    n_files = min(args.files)
    paths, oracle = _build(
        directory, n_files, args.rows_per_file, args.cols, args.records, args.dtype
    )
    names = list(oracle)
    with colstore.open(paths) as ds:
        for fraction in (0.0, 0.01, 0.5, 1.0):
            mask = (
                _mask(ds.n_rows, fraction)
                if 0.0 < fraction < 1.0
                else (np.ones(ds.n_rows, bool) if fraction == 1.0 else np.zeros(ds.n_rows, bool))
            )
            expected = {name: oracle[name][mask] for name in names}
            for label, fn in (("old", _old_per_file), ("new", _new_routed)):
                got = fn(ds, mask, names)
                for name in names:
                    _c.check_equal(got[name], expected[name], f"{label} f={fraction}:{name}")
    for path in paths:
        path.unlink(missing_ok=True)
    print("  CORRECTNESS OK (old == new == numpy oracle for every density)\n")


def run_bench(directory: Path, args: argparse.Namespace) -> None:
    print("Environment:")
    print(
        f"  rows/file={args.rows_per_file:,}  cols={args.cols}  records={args.records}  "
        f"dtype={args.dtype}  mask fraction={args.fraction}"
    )
    print(f"  files swept={args.files}  repeat={args.repeat}  warmup={args.warmup}\n")
    for n_files in args.files:
        paths, oracle = _build(
            directory, n_files, args.rows_per_file, args.cols, args.records, args.dtype
        )
        names = list(oracle)
        with colstore.open(paths) as ds:
            mask = _mask(ds.n_rows, args.fraction)
            n_out = int(mask.sum())
            print(f"=== {n_files} files  ({ds.n_rows:,} rows, {n_out:,} selected) ===")
            variants: list[tuple[str, Callable[[], object]]] = [
                ("per-file(old)", lambda d=ds, m=mask, c=names: _old_per_file(d, m, c)),
                ("fancy(new)   ", lambda d=ds, m=mask, c=names: _new_routed(d, m, c)),
            ]
            _c.compare(variants, repeat=args.repeat, warmup=args.warmup, baseline=0)
            print()
        for path in paths:
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(
        parser, repeat=7, warmup=2, cols=8, dtype="float64", threads=True, tmpdir=True
    )
    parser.add_argument("--rows-per-file", type=int, default=500_000, help="rows per child file")
    parser.add_argument("--records", type=int, default=200, help="records per child file")
    parser.add_argument("--fraction", type=float, default=0.5, help="mask selection fraction")
    parser.add_argument(
        "--files", type=int, nargs="+", default=[4, 16, 64, 256], help="file counts to sweep"
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
