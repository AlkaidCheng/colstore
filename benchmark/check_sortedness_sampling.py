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

K_SIZES = (200_000, 1_000_000, 10_000_000)


def _full_check(indices: np.ndarray) -> bool:
    """The pre-change gate, verbatim."""
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


def _best(f, repeat: int) -> float:
    return _c.best_time(f, repeat=repeat, warmup=0)


def run_bench(repeat: int) -> None:
    rng = np.random.default_rng(1)
    print(f"{'check only':<24}{'full pass':>12}{'sampled':>10}{'speedup':>9}")
    for k in K_SIZES:
        rand = rng.integers(0, 10**9, k).astype(np.int64)
        t_full = _best(lambda arr=rand: _full_check(arr), repeat * 4)
        t_sampled = _best(lambda arr=rand: reader_mod._indices_are_sorted(arr), repeat * 4)
        print(
            f"  K={k:<12} (random){t_full * 1e6:10.1f}us{t_sampled * 1e6:8.1f}us"
            f"{t_full / t_sampled:8.1f}x"
        )
    print()
    print(f"{'end-to-end unsorted read':<34}{'full pass':>11}{'sampled':>10}{'speedup':>9}")
    with tempfile.TemporaryDirectory() as tmp:
        n_records, rows = 10_000, 2_000
        total = n_records * rows
        full = np.random.default_rng(2).standard_normal(total)
        path = Path(tmp) / "s.cstore"
        with colstore.create(path) as writer:
            for r in range(n_records):
                chunk = full[r * rows : (r + 1) * rows]
                writer.write({"value": chunk, "aux": chunk})
        dataset = colstore.open(path)
        for k in (2_000_000, 10_000_000):
            indices = np.random.default_rng(3).integers(0, total, k).astype(np.int64)
            dataset[indices[: k // 100], "value"].array()  # warm pages
            t_new_1col = _best(lambda idx=indices: dataset[idx, "value"].array(), repeat)
            t_new_2col = _best(lambda idx=indices: dataset[idx, ["value", "aux"]].dict(), repeat)
            with _force_baseline():
                t_old_1col = _best(lambda idx=indices: dataset[idx, "value"].array(), repeat)
                t_old_2col = _best(
                    lambda idx=indices: dataset[idx, ["value", "aux"]].dict(), repeat
                )
            print(
                f"  K={k:<10} single column {t_old_1col * 1e3:8.1f}ms{t_new_1col * 1e3:8.1f}ms"
                f"{t_old_1col / t_new_1col:8.2f}x"
            )
            print(
                f"  K={k:<10} 2-col bin-reuse{t_old_2col * 1e3:7.1f}ms{t_new_2col * 1e3:8.1f}ms"
                f"{t_old_2col / t_new_2col:8.2f}x"
            )
        dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()
    check_correctness()
    if not args.skip_bench:
        run_bench(args.repeat)


if __name__ == "__main__":
    main()
