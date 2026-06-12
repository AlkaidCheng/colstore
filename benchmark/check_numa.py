"""Robust benchmark for the NUMA optimization (writer + reader sides).

Cold-cache benchmarking has a subtle pitfall: ``posix_fadvise(DONTNEED)`` is
advisory and the kernel ignores it for any page that still has a live
reference. If a ``ColStoreReader`` is open across the eviction, its
``np.memmap`` pins the pages and the "cold" reads are actually warm (the
field symptom was writer=local "cold" 56 ms < "warm" 81 ms -- impossible if
eviction worked). The cold A/B here therefore runs through ``_cold_compare``:
each variant's setup closes the prior reader, ``gc.collect()``s, evicts the
file's pages with **no reader open**, then reopens under the policy, so the
timed gather faults pages fresh. The reader open is outside the timed region
(its cost is policy-independent). ``_warn_if_not_cold`` watches the major-fault
counter to surface incomplete eviction.

Cold A/B is reader-side, not writer-side: evicted pages are re-faulted by the
faulting thread per its mempolicy, so writer-side placement is forgotten on
eviction and only reader-side mbind controls cold re-fault distribution.

The scenarios:

  1. Writer-side A/B, WARM (headline): the same data written under "local" vs
     "interleave", read with reader policy fixed local. Warm reads pick up
     whatever pages the writer placed.
  2. Reader-side A/B, COLD: one file (writer=local), read under each reader
     policy after proper eviction; the VMA mempolicy governs re-fault placement.
  3. Pin-tests: writer-side COLD (~noise -- evicted pages forget the writer);
     reader-side WARM (~noise -- mbind cannot move resident pages).
  4. End-to-end ds.dict()/ds.frame() under the default "auto" policy.
  5. Low-concurrency regression check (workers=1), justifying the "local" opt-out.

Run with ``PYTHONPATH=src`` after building the extension. ``--scale`` shrinks
every store for a quick smoke run.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import _numa, config

drop_pagecache_softly = _c.drop_pagecache


@contextmanager
def policy_scope(policy):
    previous = config.get_numa_policy()
    config.set_numa_policy(policy)
    try:
        yield
    finally:
        config.set_numa_policy(previous)


def write_store_under_policy(path: Path, columns: dict, policy: str) -> None:
    """Write a fresh store at ``path`` under the given NUMA policy."""
    if path.exists():
        path.unlink()
    with policy_scope(policy):
        colstore.store(columns, str(path), show_progress=False).close()


def _warn_if_not_cold(result) -> None:
    """Warn when a 'cold' result's best run logged no major faults.

    Real cold reads of a large store record many major faults; zero means the
    eviction was declined (kernel hint ignored, or a reference held the file
    open during eviction) and the numbers reflect warm reads.
    """
    if result.major_pf == 0:
        print(
            f"  warning: '{result.label.strip()}' best run logged 0 major faults; "
            "the cache may not have been evicted (warm, not cold)."
        )


def _cold_compare(specs, *, repeat: int, warmup: int):
    """Cold-cache A/B over ``(label, path, reader_policy, gather_fn)`` specs.

    Each variant's setup closes the prior reader, evicts the file with no
    reader open, then reopens under the policy -- so the timed gather faults
    pages fresh. Open is excluded from timing; it is policy-independent.
    """
    holders: list[dict] = [{} for _ in specs]

    def make_setup(i: int):
        _, path, policy, _ = specs[i]
        holder = holders[i]

        def setup() -> None:
            if "reader" in holder:
                holder["reader"].close()
                del holder["reader"]
            gc.collect()  # release the memmap before evicting
            drop_pagecache_softly([path])
            with policy_scope(policy):
                holder["reader"] = colstore.open(str(path))

        return setup

    def make_fn(i: int):
        gather_fn = specs[i][3]
        holder = holders[i]
        return lambda: gather_fn(holder["reader"])

    pairs = [(specs[i][0], make_fn(i)) for i in range(len(specs))]
    results = _c.compare(
        pairs,
        repeat=repeat,
        warmup=warmup,
        baseline=0,
        setups=[make_setup(i) for i in range(len(specs))],
    )
    for holder in holders:
        if "reader" in holder:
            holder["reader"].close()
    gc.collect()
    for result in results:
        _warn_if_not_cold(result)
    return results


def banner(s: str) -> None:
    print(f"\n=== {s} ===")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(
        parser, repeat=5, warmup=2, rows=2_500_000, cols=50, scale=True, threads=True
    )
    args = parser.parse_args()
    _c.apply_runtime_config(args)

    print("Environment:")
    print(f"  os.cpu_count()                 = {os.cpu_count()}")
    print(f"  config.get_max_workers()       = {config.get_max_workers()}")
    print(f"  config.get_gather_thread_cap() = {config.get_gather_thread_cap()}")
    print(f"  _numa.is_available()           = {_numa.is_available()}")
    print(f"  _numa.allowed_nodes()          = {_numa.allowed_nodes()}")
    if not _numa.is_available():
        print(
            "\n  NOTE: NUMA is a no-op on this host; A/B numbers will be within noise.\n"
            "  Run on a multi-socket Linux server to see the actual win."
        )

    n_rows = _c.scaled_rows(args.rows, args)
    n_cols = args.cols
    total_gb = n_rows * n_cols * 8 / 1e9
    with tempfile.TemporaryDirectory() as td:
        local_path = Path(td) / "store_local.cstore"
        interleave_path = Path(td) / "store_interleave.cstore"
        rng = np.random.default_rng(0)
        cols = {f"c{i:02d}": rng.standard_normal(n_rows) for i in range(n_cols)}
        print(f"\nWriting two stores ({n_cols} x {n_rows:,} = {total_gb:.2f} GB each)...")
        write_store_under_policy(local_path, cols, "local")
        write_store_under_policy(interleave_path, cols, "interleave")

        local_reader = colstore.open(str(local_path))
        interleave_reader = colstore.open(str(interleave_path))
        try:
            # 1. Writer-side A/B, warm (headline). Reader policy fixed local so
            #    the delta is attributable to where the writer placed pages.
            for op_label, op in (
                ("ds.dict()", lambda r: r.dict()),
                ("ds.frame()", lambda r: r.frame()),
            ):
                banner(f"WRITER-SIDE A/B (warm)  {total_gb:.2f} GB  {op_label}")
                with policy_scope("local"):
                    _c.compare(
                        [
                            (
                                f"writer=local      reader=local  {op_label}",
                                lambda o=op: o(local_reader),
                            ),
                            (
                                f"writer=interleave reader=local  {op_label}",
                                lambda o=op: o(interleave_reader),
                            ),
                        ],
                        repeat=args.repeat,
                        warmup=args.warmup,
                        baseline=0,
                    )

            # 3a. Writer-side A/B, COLD (pin-test: ~noise expected).
            banner(f"WRITER-SIDE A/B (cold)  {total_gb:.2f} GB  ds.dict()")
            _cold_compare(
                [
                    (
                        "writer=local      reader=local  ds.dict() cold",
                        local_path,
                        "local",
                        lambda r: r.dict(),
                    ),
                    (
                        "writer=interleave reader=local  ds.dict() cold",
                        interleave_path,
                        "local",
                        lambda r: r.dict(),
                    ),
                ],
                repeat=args.repeat,
                warmup=args.warmup,
            )
            print("  (pin-test: expected ~noise -- evicted pages forget the writer's placement)")

            # 2. Reader-side A/B, COLD (where reader-side mbind earns its place).
            banner(f"READER-SIDE A/B (cold)  {total_gb:.2f} GB  ds.dict()")
            _cold_compare(
                [
                    (
                        "reader=local      writer=local  ds.dict() cold",
                        local_path,
                        "local",
                        lambda r: r.dict(),
                    ),
                    (
                        "reader=interleave writer=local  ds.dict() cold",
                        local_path,
                        "interleave",
                        lambda r: r.dict(),
                    ),
                ],
                repeat=args.repeat,
                warmup=args.warmup,
            )
            print("  (cold: reader-side mbind controls re-fault allocation)")

            # 3b. Reader-side A/B, WARM (pin-test: ~noise expected). Open each
            #     reader under its policy once, then time warm reads.
            banner(f"READER-SIDE A/B (warm)  {total_gb:.2f} GB  ds.dict()")
            with policy_scope("local"):
                warm_local = colstore.open(str(local_path))
            with policy_scope("interleave"):
                warm_inter = colstore.open(str(local_path))
            try:
                _c.compare(
                    [
                        ("reader=local      writer=local  ds.dict()", lambda: warm_local.dict()),
                        ("reader=interleave writer=local  ds.dict()", lambda: warm_inter.dict()),
                    ],
                    repeat=args.repeat,
                    warmup=args.warmup,
                    baseline=0,
                )
            finally:
                warm_local.close()
                warm_inter.close()
            print("  (pin-test: expected ~noise -- reader mbind cannot move warm pages)")
        finally:
            local_reader.close()
            interleave_reader.close()

        # 4. End-to-end under the default "auto" policy.
        banner(f"END-TO-END default policy  {total_gb:.2f} GB")
        e2e_path = Path(td) / "end_to_end.cstore"
        write_store_under_policy(e2e_path, cols, "auto")
        with policy_scope("auto"):
            ds = colstore.open(str(e2e_path))
            try:
                _c.compare(
                    [
                        ("ds.dict()  writer=auto reader=auto", lambda: ds.dict()),
                        ("ds.frame() writer=auto reader=auto", lambda: ds.frame()),
                    ],
                    repeat=args.repeat,
                    warmup=args.warmup,
                    baseline=0,
                )
            finally:
                ds.close()

        # 5. Low-concurrency regression check (workers=1): "interleave" forces
        #    mostly-remote loads with one consumer; the "local" opt-out exists
        #    for exactly this case.
        banner(f"LOW-CONCURRENCY regression check (workers=1)  {total_gb:.2f} GB")
        prev_workers, prev_cap = config.get_max_workers(), config.get_gather_thread_cap()
        try:
            config.set_max_workers(1)
            config.set_gather_thread_cap(1)
            ds_local = colstore.open(str(local_path))
            ds_inter = colstore.open(str(interleave_path))
            try:
                with policy_scope("local"):
                    _c.compare(
                        [
                            ("workers=1  writer=local      reader=local", lambda: ds_local.dict()),
                            ("workers=1  writer=interleave reader=local", lambda: ds_inter.dict()),
                        ],
                        repeat=args.repeat,
                        warmup=args.warmup,
                        baseline=0,
                    )
            finally:
                ds_local.close()
                ds_inter.close()
        finally:
            config.set_max_workers(prev_workers)
            config.set_gather_thread_cap(prev_cap)


if __name__ == "__main__":
    if sys.platform != "linux":
        print("This benchmark requires Linux (NUMA syscalls). Exiting.")
        sys.exit(0)
    main()
