"""Verify the zero-copy read API: correctness and timing.

``array(copy=False)`` / ``dict(copy=False)`` on single-record (compacted)
stores return read-only views of the open memmaps instead of copying into
owning arrays. The materialization call becomes O(1) regardless of column
size -- the copy, the allocation, and (for cold reads) the eager page
traffic all disappear from the call; pages fault lazily as the consumer
touches them. The flag itself is the toggle, so no routing seam is needed.

Two timings are reported per column size:

* call latency -- ``array()`` vs ``array(copy=False)``: isolates the
  removed copy + allocation;
* read-and-reduce -- materialize then ``np.sum``: a consumer workload
  where the view path reads each byte once (from the page cache) and the
  copy path reads, writes, and re-reads it.

Run on the deployment hardware (quiet compute node):

    python benchmark/check_zero_copy.py
    python benchmark/check_zero_copy.py --skip-bench
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import _common as _c

import colstore
from colstore import testing

SIZES = (1_000_000, 10_000_000, 100_000_000)  # f8 rows: 8 MB / 80 MB / 800 MB


def check_correctness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data = testing.make_columns(100_000, 2, names=("a", "b"), seed=0)
        path = Path(tmp) / "z.cstore"
        with colstore.create(path) as writer:
            writer.write(data)
        dataset = colstore.open(path)
        for selector in (slice(None), slice(10, 90_000, 7), slice(None, None, -1)):
            view = dataset[selector, "a"].array(copy=False)
            owning = dataset[selector, "a"].array()
            assert np.array_equal(view, owning)
            assert not view.flags.writeable
        table = dataset.dict(copy=False)
        for name, values in data.items():
            assert np.array_equal(table[name], values), name
        dataset.close()
        assert np.array_equal(table["a"], data["a"])  # views survive close
    print("  ALL CORRECTNESS CHECKS PASSED (views == owning arrays; read-only; survive close)\n")


def _best(f, repeat: int, warmup: int) -> float:
    for _ in range(warmup):
        f()
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - start)
    return best


def run_bench(repeat: int, warmup: int) -> None:
    print(
        f"{'rows':>12}{'call copy=True':>16}{'call copy=False':>17}"
        f"{'sum copy=True':>15}{'sum copy=False':>16}"
    )
    for n in SIZES:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"n{n}.cstore"
            with colstore.create(path) as writer:
                writer.write({"a": testing.make_columns(n, 1, names=("a",), seed=1)["a"]})
            dataset = colstore.open(path)
            dataset["a"].array(copy=False).sum()  # warm page cache
            t_call_copy = _best(lambda ds=dataset: ds["a"].array(), repeat, warmup)
            t_call_view = _best(lambda ds=dataset: ds["a"].array(copy=False), repeat, warmup)
            t_sum_copy = _best(lambda ds=dataset: ds["a"].array().sum(), repeat, warmup)
            t_sum_view = _best(lambda ds=dataset: ds["a"].array(copy=False).sum(), repeat, warmup)
            dataset.close()
        print(
            f"{n:>12}{t_call_copy * 1e3:>14.2f}ms{t_call_view * 1e6:>14.1f}us"
            f"{t_sum_copy * 1e3:>13.2f}ms{t_sum_view * 1e3:>14.2f}ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(parser, repeat=5, skip_correctness=False)
    args = parser.parse_args()
    check_correctness()
    if not args.skip_bench:
        run_bench(args.repeat, args.warmup)


if __name__ == "__main__":
    main()
