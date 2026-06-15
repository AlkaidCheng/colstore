"""Measure whether the best cold-read NUMA placement depends on the access pattern.

NUMA page placement is fixed at first fault. For a cold read -- the first touch
of a store's pages from disk or scratch, before they are resident -- the
reader-side policy set at open governs where the faulted pages land:
``interleave`` spreads them across nodes, ``local`` leaves the kernel's
first-touch placement near the faulting thread.

A warm placement x binding x cap sweep found interleave best for the large
random-scatter conversions. This benchmark asks the open question for *cold*
reads across access patterns -- contiguous range, sorted fancy, random scatter:
does the winning policy change with the pattern? A sequential single-consumer
scan might prefer local (pages next to the one reader, no cross-node traffic),
while a many-threaded scatter prefers interleave (memory controllers balanced).
If the winner is the same for every pattern, the single ``auto`` default
(interleave on multi-node) stands and no per-pattern placement is warranted; if
it flips, a per-pattern cold-read placement is worth implementing.

Cold reads need an empty page cache, so each timed gather is preceded by a
setup that closes every reader, evicts the file with no memmap pinning it, and
reopens under the policy -- the timed region then faults pages fresh. The two
policies are interleaved across rounds by the shared compare harness. Run on a
multi-node host; on a single-node host placement is a no-op and the two columns
land within noise (the run still reports that, so it is safe to run anywhere).

    PYTHONPATH=src python benchmark/check_cold_read_placement.py --tmpdir /tmp
"""

from __future__ import annotations

import argparse
import gc
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import _common as _c
import numpy as np

from colstore import _numa, config, testing

_POLICIES = ("local", "interleave")


@contextmanager
def _policy_scope(policy: str) -> Iterator[None]:
    previous = config.get_numa_policy()
    config.set_numa_policy(policy)
    try:
        yield
    finally:
        config.set_numa_policy(previous)


def _selectors(n_rows: int, n_indices: int, seed: int) -> dict[str, Any]:
    """One selector per access pattern, all covering the same row count."""
    rng = np.random.default_rng(seed)
    k = min(n_indices, n_rows)
    scatter = rng.choice(n_rows, size=k, replace=False).astype(np.int64)
    return {
        "contiguous": slice(0, k),  # sequential range
        "sorted-fancy": np.sort(scatter),  # ascending, strided
        "random-scatter": scatter,  # unsorted -- the conversion bottleneck
    }


def _cold_placement_ab(
    path: Path, selector: Any, *, repeat: int, warmup: int
) -> list[_c.ProfileResult]:
    """Cold local-vs-interleave A/B for one selector; results in ``_POLICIES`` order."""
    holders: list[dict[str, Any]] = [{} for _ in _POLICIES]

    def make_setup(i: int) -> Any:
        policy = _POLICIES[i]
        holder = holders[i]

        def setup() -> None:
            # Close EVERY reader before evicting -- both policies read the same
            # file, so the other variant's open memmap would otherwise pin the
            # pages and decline the eviction (leaving a warm, not cold, read).
            for other in holders:
                if "reader" in other:
                    other["reader"].close()
                    del other["reader"]
            gc.collect()
            _c.drop_pagecache([path])
            with _policy_scope(policy):
                holder["reader"] = _c.colstore.open(str(path))

        return setup

    def make_fn(i: int) -> Any:
        holder = holders[i]
        return lambda: holder["reader"][selector].dict()

    specs = [(_POLICIES[i], make_fn(i)) for i in range(len(_POLICIES))]
    results = _c.compare(
        specs,
        repeat=repeat,
        warmup=warmup,
        baseline=0,
        setups=[make_setup(i) for i in range(len(_POLICIES))],
    )
    for holder in holders:
        if "reader" in holder:
            holder["reader"].close()
    gc.collect()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(
        parser,
        repeat=5,
        warmup=1,
        rows=8_000_000,
        cols=8,
        indices=4_000_000,
        dtype="float32",
        tmpdir=True,
        threads=True,
        json=True,
    )
    parser.add_argument("--records", type=int, default=16, help="record count for the store")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    _c.apply_runtime_config(args)

    print(f"# numa available: {_numa.is_available()}  nodes: {_numa.allowed_nodes()}")
    if not _numa.is_available():
        print("# NOTE: single-node / non-NUMA host -- placement is a no-op; columns are noise.")

    n_rows, n_cols = args.rows, args.cols
    work = Path(args.tmpdir) if args.tmpdir is not None else Path(".")
    work.mkdir(parents=True, exist_ok=True)
    store = work / "cold_placement.cstore"
    if store.exists():
        store.unlink()
    testing.make_store(
        store, rows=n_rows, cols=n_cols, records=args.records, dtype=args.dtype, seed=args.seed
    ).close()
    selectors = _selectors(n_rows, args.indices, args.seed)

    winners: dict[str, str] = {}
    results: list[_c.Result] = []
    for pattern, selector in selectors.items():
        print(f"\n=== {pattern} (cold) ===")
        res = _cold_placement_ab(store, selector, repeat=args.repeat, warmup=args.warmup)
        by_policy = dict(zip(_POLICIES, res, strict=True))
        winner = min(_POLICIES, key=lambda p: by_policy[p].wall_ms)
        winners[pattern] = winner
        for policy in _POLICIES:
            r = by_policy[policy]
            if r.major_pf == 0:
                print(f"  warning: '{policy}' logged 0 major faults -- may be warm, not cold")
            results.append(
                _c.Result(
                    scenario="cold_read_placement",
                    variant=policy,
                    params={"pattern": pattern},
                    median_ms=r.wall_ms,
                    min_ms=r.wall_ms,
                    p95_ms=r.wall_ms,
                    repeat=args.repeat,
                )
            )

    print("\n# verdict")
    for pattern, winner in winners.items():
        print(f"#   {pattern:>16}: {winner}")
    if len(set(winners.values())) > 1:
        print("#   -> winning policy DIFFERS by pattern: per-pattern cold placement may help")
    else:
        only = next(iter(set(winners.values())))
        print(f"#   -> '{only}' wins for every pattern: the single auto default stands")

    if args.json is not None:
        _c.write_summary(args.json, results, meta={"benchmark": "check_cold_read_placement"})
        print(f"\n# wrote {args.json}")
    if not args.tmpdir:
        store.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
