"""Verify the bin-reuse multi-column gather: correctness and timing.

For a multi-column **unsorted fancy** read of a multi-record store, the
per-index record binning (a branchless binary search, measured 87-93% of the
fused gather kernel's cost on the target hardware) is identical for every
column, so the reader computes it once (``gather_multirecord_bins``) and
reuses it for the remaining columns (``gather_multirecord_withbins``). This
script checks the route end-to-end through the public reader API and times
it against the per-column path it replaces.

Run on the deployment hardware:

    python benchmark/check_multicolumn_gather.py
    python benchmark/check_multicolumn_gather.py --skip-bench   # correctness only

Expected shape (from the standalone premise check on the same hardware):
~1.9-2.5x at realistic thread caps, growing with the column count and the
record count. Sorted reads and single-column reads are unaffected by design.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore.reader import ColStoreReader

K = 1_000_000
LAYOUTS = ((1000, 20_000), (10_000, 2_000))
COLUMN_COUNTS = (4, 8)


def _build_store(directory: Path, n_records: int, rows: int, n_cols: int):
    rng = np.random.default_rng(0)
    names = [f"c{i}" for i in range(n_cols)]
    full = {name: rng.standard_normal(n_records * rows) for name in names}
    path = directory / f"r{n_records}_c{n_cols}.cstore"
    with colstore.create(path) as writer:
        for r in range(n_records):
            writer.write({name: full[name][r * rows : (r + 1) * rows] for name in names})
    return path, names, full


def _disable_route(monkey_target=ColStoreReader):
    original = monkey_target._gather_many_bin_reuse
    monkey_target._gather_many_bin_reuse = lambda self, names, indexer: None  # type: ignore[assignment]
    return original


def _restore_route(original, monkey_target=ColStoreReader):
    monkey_target._gather_many_bin_reuse = original  # type: ignore[assignment]


def check_correctness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path, names, full = _build_store(Path(tmp), 50, 800, 4)
        dataset = colstore.open(path)
        indices = np.random.default_rng(5).integers(0, 50 * 800, size=10_000).astype(np.int64)
        routed = dataset[indices, names].dict()
        original = _disable_route()
        try:
            fallback = dataset[indices, names].dict()
        finally:
            _restore_route(original)
        for name in names:
            assert np.array_equal(routed[name], full[name][indices]), name
            assert np.array_equal(routed[name], fallback[name]), name
        # Sorted selector must agree too (route declines it by design).
        sorted_indices = np.sort(indices)
        sorted_read = dataset[sorted_indices, names].dict()
        for name in names:
            assert np.array_equal(sorted_read[name], full[name][sorted_indices]), name
        dataset.close()
    print("  ALL CORRECTNESS CHECKS PASSED (routed == per-column == ground truth)\n")


def _time_read(dataset, indices, names, repeat: int) -> float:
    return _c.best_time(lambda: dataset[indices, names].dict(), repeat=repeat, warmup=0)


def run_bench(repeat: int) -> None:
    print(f"{'layout':<34}{'per-column':>12}{'bin-reuse':>12}{'speedup':>9}")
    for n_records, rows in LAYOUTS:
        for n_cols in COLUMN_COUNTS:
            with tempfile.TemporaryDirectory() as tmp:
                path, names, _ = _build_store(Path(tmp), n_records, rows, n_cols)
                dataset = colstore.open(path)
                indices = (
                    np.random.default_rng(1).integers(0, n_records * rows, size=K).astype(np.int64)
                )
                dataset[indices[:1000], names].dict()  # warm mmap + route
                t_routed = _time_read(dataset, indices, names, repeat)
                original = _disable_route()
                try:
                    t_fallback = _time_read(dataset, indices, names, repeat)
                finally:
                    _restore_route(original)
                dataset.close()
            print(
                f"  R={n_records:<6} rows/rec={rows:<8} C={n_cols}:"
                f"{t_fallback * 1e3:10.1f}ms{t_routed * 1e3:10.1f}ms"
                f"{t_fallback / t_routed:8.2f}x"
            )


def main() -> None:
    _c.run_script(correctness=check_correctness, bench=run_bench, description=__doc__)


if __name__ == "__main__":
    main()
