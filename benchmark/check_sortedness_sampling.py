"""Verify sampled-rejection sortedness detection: correctness and timing.

Routing between the sorted and unsorted fancy paths previously ran the full
O(K) non-decreasing check on every fancy selector. For random unsorted
selectors -- the common case for event-selection reads -- that pass (plus
its K-1-byte comparison temporary) is pure overhead, and it is serial, so
its share of the read grows with the gather kernels' thread count. The
sampled pass probes 16 evenly spaced adjacent pairs first: any descent
proves unsortedness immediately (a random unsorted selector fails with
probability 1 - 2**-16); an all-ascending sample falls through to the full
pass, so sorted selectors keep their exact proof. Below the size threshold
the sampler is skipped (its fixed overhead exceeds the full pass there).

This script checks helper/full-check agreement on a battery including
adversarial single-descent arrays, then times the check itself and the
end-to-end unsorted fancy read against the full-check baseline, toggled by
monkeypatching the seam (``reader._indices_are_sorted``).

Run on the deployment hardware (quiet compute node):

    python benchmark/check_sortedness_sampling.py
    python benchmark/check_sortedness_sampling.py --skip-bench

Expected shape of the result: the check-level speedup is large (the 8 ms
full pass at K=10^7 becomes ~15 us); the end-to-end win is the check's
share of the read, which is small single-threaded and grows with
OMP_NUM_THREADS because the kernels parallelize and the check does not.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import reader as reader_mod
from colstore import testing


def _full_check(indices: np.ndarray) -> bool:
    """The full-scan sortedness gate -- the O(K) reference."""
    return bool(np.all(indices[1:] >= indices[:-1]))


def _baseline_gate(indices):
    n = indices.shape[0]
    if n <= 1:
        return True
    return _full_check(indices)


class _force_baseline:
    def __enter__(self):
        self._original = reader_mod._indices_are_sorted
        reader_mod._indices_are_sorted = _baseline_gate
        return self

    def __exit__(self, *exc):
        reader_mod._indices_are_sorted = self._original
        return False


def check_correctness() -> None:
    rng = np.random.default_rng(0)
    threshold = reader_mod._SORTEDNESS_SAMPLE_MIN_SIZE
    battery = []
    for n in (0, 1, 2, 1_000, threshold - 1, threshold, threshold * 4):
        rand = rng.integers(0, 10**9, n).astype(np.int64)
        battery += [rand, np.sort(rand), np.full(n, 3, dtype=np.int64)]
    # Adversarial: sorted except one descent between probe positions.
    n = threshold * 2
    adversarial = np.arange(n, dtype=np.int64)
    probed = set(((reader_mod._SORTEDNESS_SAMPLE_FRACTIONS * (n - 2)).astype(np.int64)).tolist())
    pos = next(p for p in range(n // 3, n) if p not in probed and (p + 1) not in probed)
    adversarial[pos], adversarial[pos + 1] = adversarial[pos + 1], adversarial[pos]
    battery.append(adversarial)
    for indices in battery:
        assert reader_mod._indices_are_sorted(indices) == _full_check(indices), indices.shape
    print("  ALL CORRECTNESS CHECKS PASSED (sampled gate == full check, incl. adversarial)\n")


def _read_baseline(dataset, indices, cols):
    """One read forced through the full-check sortedness gate."""
    with _force_baseline():
        view = dataset[indices, cols]
        return view.dict() if isinstance(cols, list) else view.array()


def run_bench(args: argparse.Namespace) -> None:
    # Part 1: the gate in isolation (no store) -- full pass vs sampled.
    rand = np.random.default_rng(1).integers(0, 10**9, args.indices).astype(np.int64)
    print(f"check only (random)  K={args.indices:,}")
    _c.compare(
        [
            ("full pass", lambda a=rand: _full_check(a)),
            ("sampled", lambda a=rand: reader_mod._indices_are_sorted(a)),
        ],
        repeat=args.repeat * 4,  # the check is microseconds; oversample it
        warmup=args.warmup,
        baseline=0,
    )
    print()

    # Part 2: end-to-end unsorted reads -- the gate's share of the whole read.
    for n_records in args.record_counts:
        rows = args.rows // n_records
        total = rows * n_records
        with tempfile.TemporaryDirectory() as tmp:
            full = testing.make_columns(total, 1, names=("value",), seed=2)["value"]
            path = Path(tmp) / "s.cstore"
            with colstore.create(path) as writer:
                for r in range(n_records):
                    chunk = full[r * rows : (r + 1) * rows]
                    writer.write({"value": chunk, "aux": chunk})
            dataset = colstore.open(path)
            indices = np.random.default_rng(3).integers(0, total, args.indices).astype(np.int64)
            dataset[indices[: max(1, args.indices // 100)], "value"].array()  # warm pages
            print(f"end-to-end  R={n_records:<7} rows/rec={rows:<7} K={args.indices:,}")
            _c.compare(
                [
                    ("full-gate 1col", lambda d=dataset, i=indices: _read_baseline(d, i, "value")),
                    ("sampled   1col", lambda d=dataset, i=indices: d[i, "value"].array()),
                ],
                repeat=args.repeat,
                warmup=args.warmup,
                baseline=0,
            )
            _c.compare(
                [
                    (
                        "full-gate 2col",
                        lambda d=dataset, i=indices: _read_baseline(d, i, ["value", "aux"]),
                    ),
                    ("sampled   2col", lambda d=dataset, i=indices: d[i, ["value", "aux"]].dict()),
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
        record_counts=[10_000],
        indices=10_000_000,
        threads=True,
    )
    args = parser.parse_args()
    _c.apply_runtime_config(args)
    if not args.skip_correctness:
        check_correctness()
    if not args.skip_bench:
        run_bench(args)


if __name__ == "__main__":
    main()
