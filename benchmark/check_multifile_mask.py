"""A/B the multi-file boolean-mask read: per-file fan-out vs the native mask kernel.

Three routes for ``ds[mask]`` on a multi-file ``ColStoreDataset``, all producing
byte-identical output, swept over the file count:

* ``per-file``  -- split the mask per file, gather each ``(file, column)`` portion
  on a threadpool via the child's own mask/flatnonzero path (current default for
  sparse masks; the path PR #196 lost to at scale).
* ``fancy``     -- ``np.flatnonzero(mask)`` once, then the native multi-file sorted
  fancy gather. This is the rejected PR #196 reroute, kept here as the reference
  it regressed against (materializes an int64 index array + a per-index segment
  search).
* ``kernel``    -- the native segment mask kernel (``colstore_gather_segment_mask``)
  over the cached segment table: a 1-byte/row mask, a monotonic segment cursor, and
  popcount/prefix-sum output offsets, in one parallel pass -- no index array, no
  per-row search.

The win is expected to grow with file count (more per-file orchestration folded
into one C call) and to be bandwidth-bound at large selected counts; the kernel
should reverse the ``fancy`` route's at-scale regression. The correctness gate
asserts all three match the NumPy oracle before timing. Run on the deployment
node with ``--tmpdir`` on the parallel filesystem; local numbers are indicative.
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
from colstore.dataset import ColStoreDataset

_FRACTION_SEED = 20260623


def _build(directory: Path, n_files: int, rows_each: int, cols: int, records: int, dtype: str):
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


def _per_file(ds: ColStoreDataset, mask: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """The per-file fan-out: split the mask per file, gather each portion."""
    return ds._gather_many_contiguous(names, ds._mask_parts(mask), True)


def _fancy(ds: ColStoreDataset, mask: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """The rejected PR #196 reroute: lower the whole mask to indices, then fancy gather."""
    return ds._fancy_many(names, np.flatnonzero(mask).astype(np.int64))


def _kernel(ds: ColStoreDataset, mask: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """The native multi-file mask kernel (forced; the density gate is bypassed here)."""
    result = ds._mask_native(names, mask)
    if result is None:
        raise RuntimeError("kernel declined (non-native columns or no extension)")
    return result


def _mask(n_rows: int, fraction: float) -> np.ndarray:
    return np.random.default_rng(_FRACTION_SEED).random(n_rows) < fraction


_ROUTES: tuple[tuple[str, Callable[[ColStoreDataset, np.ndarray, list[str]], dict]], ...] = (
    ("per-file", _per_file),
    ("fancy   ", _fancy),
    ("kernel  ", _kernel),
)


def check_correctness(directory: Path, args: argparse.Namespace) -> None:
    paths, oracle = _build(
        directory, min(args.files), args.rows_per_file, args.cols, args.records, args.dtype
    )
    names = list(oracle)
    with colstore.open(paths) as ds:
        for fraction in (0.01, 0.5, 1.0):
            mask = _mask(ds.n_rows, fraction) if fraction < 1.0 else np.ones(ds.n_rows, bool)
            expected = {name: oracle[name][mask] for name in names}
            for label, route in _ROUTES:
                got = route(ds, mask, names)
                for name in names:
                    _c.check_equal(
                        got[name], expected[name], f"{label.strip()} f={fraction}:{name}"
                    )
    for path in paths:
        path.unlink(missing_ok=True)
    print("  CORRECTNESS OK (per-file == fancy == kernel == numpy oracle)\n")


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
            print(f"=== {n_files} files  ({ds.n_rows:,} rows, {int(mask.sum()):,} selected) ===")
            variants = [
                (label, lambda d=ds, m=mask, c=names, r=route: r(d, m, c))
                for label, route in _ROUTES
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
    parser.add_argument("--records", type=int, default=1, help="records per child file")
    parser.add_argument("--fraction", type=float, default=0.5, help="mask selection fraction")
    parser.add_argument(
        "--files", type=int, nargs="+", default=[4, 16, 64, 256], help="file counts to sweep"
    )
    args = parser.parse_args()
    _c.apply_runtime_config(args)

    if not kernels.cpp_available():
        raise SystemExit("the compiled extension is required for this benchmark")
    # Force the kernel route at every density so the A/B measures it directly; the
    # density gate is a routing policy, not a property of the kernel under test.
    config.set_multifile_mask_density_gate(0.0)

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
