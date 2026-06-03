"""Compare read perf across single-record and multi-record files.

A colstore file is a sequence of records. After a single ``from_dict`` /
``from_dataframe`` / ``from_records`` call the file has exactly one record;
after many small writes via :class:`ColStoreWriter` (PR 3) it has many. The
reader takes different code paths for these two cases:

1. Single record: per-column memmaps + the element-indexed ``gather``
   kernel. Contiguous, prefetcher-friendly.

2. Multiple records: a whole-file mmap + an in-memory per-record index
   (cumulative rows + record body offsets, ~24 bytes per record). Each
   read does ``np.searchsorted`` to bin indices into records, computes
   byte addresses on the fly, then calls ``gather_bytes``.

This script confirms two claims:

* Single-record reads are as fast as a pure contiguous gather would be.
* Multi-record reads pay a bounded penalty that scales sub-linearly with
  record count. Past a threshold the user should call ``compact()``
  (PR 4) to collapse back to a single record.

To compare, this benchmark builds several files containing the *same*
logical data:

  * 1 record (the post-compaction common case)
  * N records, for several N (split equally)

then times the same fancy-index reads on each.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow running this script directly without installing the package:
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from _format_fixture import write_record_file

from colstore import ColStoreReader


def _make_file(path: Path, n_rows: int, n_records: int, dtype: np.dtype) -> None:
    """Build a colstore file with ``n_records`` records totalling ``n_rows`` rows.

    Records are roughly equal-sized; any remainder lands in the last record
    so the total exactly matches.
    """
    rng = np.random.default_rng(0)
    all_data = rng.standard_normal(n_rows).astype(dtype)
    base_size = n_rows // n_records
    remainder = n_rows % n_records
    records = []
    cursor = 0
    for i in range(n_records):
        size = base_size + (1 if i == n_records - 1 else 0) * remainder
        records.append({"x": all_data[cursor : cursor + size]})
        cursor += size
    write_record_file(path, [("x", dtype.str)], records)


def _best(fn, repeats: int) -> float:
    fn()  # warmup discarded
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--n-indices", type=int, default=1_000_000)
    parser.add_argument("--record-counts", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--tmpdir", default="/tmp/colstore_v2_bench")
    args = parser.parse_args()

    dtype = np.dtype(args.dtype)
    tmpdir = Path(args.tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    print(f"Setup: {args.rows:,} rows of {dtype}, {args.n_indices:,} fancy-index reads")
    print(f"Repeats: {args.repeats} (reporting best)")
    print()

    # R=1 (post-compaction baseline) plus the requested record counts.
    record_counts = [1, *list(args.record_counts)]
    paths: dict[int, Path] = {}
    for n_rec in record_counts:
        p = tmpdir / f"data_r{n_rec}.cstore"
        if p.exists():
            p.unlink()
        _make_file(p, args.rows, n_rec, dtype)
        paths[n_rec] = p
        print(f"  wrote R={n_rec:<5}: {p} ({p.stat().st_size / 1e6:.1f} MB)")
    print()

    rng = np.random.default_rng(0)
    sorted_indices = np.sort(rng.choice(args.rows, size=args.n_indices, replace=False)).astype(
        np.int64
    )
    unsorted_indices = rng.permutation(args.rows)[: args.n_indices].astype(np.int64)
    # Slice covering ~10% of the rows (some records partially, some fully)
    slice_obj = slice(args.rows // 10, 2 * args.rows // 10)

    patterns: list[tuple[str, object]] = [
        ("full table [:]", slice(None)),
        ("slice (10% span)", slice_obj),
        ("fancy sorted", sorted_indices),
        ("fancy unsorted", unsorted_indices),
    ]
    for pattern, selector in patterns:
        print(f"---- pattern: {pattern} ----")
        header = f"{'records':<10} {'wall ms':>10}  {'vs R=1':>8}"
        print(header)
        print("-" * len(header))

        # Baseline: R=1 (post-compaction). All other counts compared against it.
        ds_baseline = ColStoreReader(paths[1])
        assert ds_baseline._is_multi_record is False, "R=1 should take the fast path"
        t_baseline = _best(lambda d=ds_baseline, s=selector: d[s, "x"].to_array(), args.repeats)
        ds_baseline.close()
        print(f"{'R=1':<10} {t_baseline * 1000:>10.3f}  {'1.00x':>8}")

        for n_rec in args.record_counts:
            ds = ColStoreReader(paths[n_rec])
            assert ds._is_multi_record is True
            t = _best(lambda d=ds, s=selector: d[s, "x"].to_array(), args.repeats)
            ds.close()
            label = f"R={n_rec}"
            print(f"{label:<10} {t * 1000:>10.3f}  {t / t_baseline:>7.2f}x")
        print()


if __name__ == "__main__":
    main()
