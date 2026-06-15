"""Validate the reader-side spread thread-binding gate (``config.gather_binding``).

Two checks, mirroring how the gate is meant to behave:

1. Decision boundary (always; deterministic, no timing). Resolves the host
   binding policy and prints, for working sets bracketing the gate threshold
   (``margin x aggregate L3``), whether the gate would bind. This confirms on
   the target host that binding engages exactly in the DRAM-bound regime and is
   skipped for cache-resident gathers and on single-node hosts -- and shows the
   per-domain vs aggregate L3 contrast that motivates gating on the aggregate.

2. A/B timing (unless ``--skip-bench``). Compares the unbound pool, the gate's
   automatic decision, and a forced spread bind, at a sub-threshold and a
   supra-threshold working set, to confirm the gate delivers the spread win
   where it engages and adds no overhead where it does not.

   Runtime binding pins libgomp's worker pool for the lifetime of the process
   (the affinity persists and the pool is shared), so once any gather has
   bound it a single process can no longer measure the unbound baseline. Each
   timed cell therefore runs in a fresh subprocess, started unbound; the parent
   interleaves the variants across rounds -- off, auto, forced, off, auto,
   forced -- so slow-moving drift (frequency, scheduler, cache warmth) is shared
   evenly across them, the same confound-cancelling the in-process harness gets
   from interleaving. This is the only benchmark that needs subprocess
   isolation, and only because the pool state it measures is not reversible
   in-process.

Run from a source checkout::

    PYTHONPATH=src python benchmark/check_gather_binding.py --tmpdir /tmp
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import _common
import numpy as np
from _common import Result, add_common_args, apply_runtime_config, machine_fingerprint, time_stats

from colstore import _numa, autotune, config

_MODES = ("off", "auto", "forced")

# Working sets to bracket the gate threshold, as multiples of margin x aggregate
# L3. The span straddles 1.0 so the table shows the boundary either side.
_DECISION_MULTIPLES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

_MIB = 1024 * 1024


def _per_row_bytes(cols: int, dtype: str) -> int:
    return cols * np.dtype(dtype).itemsize


def _indices_for_working_set(target_bytes: float, per_row: int) -> int:
    return max(1, int(target_bytes // per_row))


def _open_or_make(path: Path, rows: int, cols: int, dtype: str) -> _common.colstore.ColStoreReader:
    """Open the store at ``path``, building it (rows == indices) if absent."""
    from colstore.testing import make_store

    if path.exists():
        return _common.colstore.open(path)
    return make_store(path, rows=rows, cols=cols, dtype=dtype)


# ---- Worker: one isolated measurement in a fresh (unbound) process ----------


def run_worker(args: argparse.Namespace) -> int:
    apply_runtime_config(args)
    store = Path(args.store_path)
    ds = _open_or_make(store, args.indices, args.cols, args.dtype)
    try:
        cap = config.get_gather_thread_cap()
        if args.mode == "off":
            config.set_gather_binding(False)
        elif args.mode == "auto":
            config.set_gather_binding(True)
        elif args.mode == "forced":
            # Force the spread bind regardless of the gate: disable the gate so
            # it does not also fire, then pin the pool directly.
            config.set_gather_binding(False)
            _numa.bind_gather_threads(cap, force=True)

        rng = np.random.default_rng(args.seed)
        idx = rng.integers(0, ds.n_rows, size=args.indices, dtype=np.int64)
        stats = time_stats(lambda: ds[idx].dict(), repeat=args.repeat, warmup=args.warmup)
        report = _numa.thread_binding_report()
    finally:
        ds.close()

    print(
        "RESULT "
        + json.dumps(
            {
                "mode": args.mode,
                "median_ms": stats.median_ms,
                "min_ms": stats.min_ms,
                "distinct_masks": int(report.get("distinct_masks", 0)),
                "n_threads": int(report.get("n_threads", 0)),
            }
        )
    )
    return 0


def _measure(mode: str, indices: int, store: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Run one worker subprocess and return its parsed RESULT dict."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--mode",
        mode,
        "--indices",
        str(indices),
        "--cols",
        str(args.cols),
        "--dtype",
        args.dtype,
        "--store-path",
        str(store),
        "--repeat",
        str(args.repeat),
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
    ]
    if args.thread is not None:
        cmd += ["--thread", str(args.thread)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line is None:
        sys.stderr.write(proc.stderr[-400:] + "\n")
        return {
            "mode": mode,
            "median_ms": float("nan"),
            "min_ms": float("nan"),
            "distinct_masks": 0,
        }
    parsed: dict[str, Any] = json.loads(line[len("RESULT ") :])
    return parsed


# ---- Parent: policy summary, decision table, interleaved A/B ----------------


def _print_policy(per_row: int) -> tuple[_numa.BindingPolicy, float]:
    policy = _numa.binding_policy()
    margin = config.get_gather_bind_llc_margin()
    threshold = margin * policy.aggregate_llc_bytes
    print("# binding policy (resolved once for this host)")
    print(f"#   applicable          : {policy.applicable}")
    print(f"#   numa_nodes          : {policy.numa_nodes}")
    print(f"#   aggregate L3        : {policy.aggregate_llc_bytes / _MIB:.0f} MiB")
    print(f"#   per-domain llc_bytes: {autotune.llc_bytes() / _MIB:.0f} MiB")
    print(f"#   margin              : {margin}")
    print(f"#   bind threshold (ws) : {threshold / _MIB:.0f} MiB")
    print(f"#   gather_binding knob : {config.get_gather_binding()}")
    print(f"#   per-row bytes        : {per_row}")
    return policy, threshold


def _decision_table(policy: _numa.BindingPolicy, threshold: float, per_row: int) -> None:
    print("\n# decision boundary (would the gate bind?)")
    print(f"   {'indices':>12}  {'working set':>12}  {'ws/thresh':>9}  bind")
    enabled = config.get_gather_binding()
    for mult in _DECISION_MULTIPLES:
        n = _indices_for_working_set(mult * threshold, per_row)
        ws = n * per_row
        would_bind = enabled and policy.applicable and ws > threshold
        ratio = ws / threshold if threshold else float("inf")
        print(
            f"   {n:>12,}  {ws / _MIB:>9.0f} MiB  {ratio:>9.2f}  " f"{'Y' if would_bind else 'N'}"
        )


def _ab_timing(
    args: argparse.Namespace, threshold: float, per_row: int, work: Path
) -> list[Result]:
    cells = {
        "sub-threshold": _indices_for_working_set(0.5 * threshold, per_row),
        "supra-threshold": _indices_for_working_set(2.0 * threshold, per_row),
    }
    results: list[Result] = []
    for label, n in cells.items():
        store = work / f"bind_{label}.cstore"
        _open_or_make(store, n, args.cols, args.dtype).close()  # build once, reuse
        rounds: dict[str, list[float]] = {m: [] for m in _MODES}
        bound: dict[str, bool] = {}
        for _ in range(args.rounds):
            for mode in _MODES:  # interleave the variants across rounds
                res = _measure(mode, n, store, args)
                rounds[mode].append(res["median_ms"])
                bound[mode] = res.get("distinct_masks", 0) > 1
        ws = n * per_row
        print(f"\n# {label}: {n:,} indices, working set {ws / _MIB:.0f} MiB")
        base = statistics.median([v for v in rounds["off"] if v == v]) or float("nan")
        for mode in _MODES:
            med = statistics.median([v for v in rounds[mode] if v == v])
            speedup = base / med if med and med == med else float("nan")
            tag = "bound" if bound.get(mode) else "unbound"
            note = "  (baseline)" if mode == "off" else f"  speedup={speedup:5.2f}x"
            print(f"   {mode:>7}: {med:8.2f} ms  [{tag:>7}]{note}")
            r = Result(
                scenario="gather_binding",
                variant=mode,
                params={"cell": label, "indices": n, "working_set_bytes": ws},
                median_ms=med,
                min_ms=med,
                p95_ms=med,
                repeat=args.rounds * args.repeat,
                rows=n,
            )
            if mode != "off":
                r.speedup_vs = "off"
                r.speedup = speedup
            results.append(r)
        if not args.keep:
            store.unlink(missing_ok=True)
    return results


def run_parent(args: argparse.Namespace) -> int:
    fp = machine_fingerprint()
    print(
        f"# host: {fp['processor']}  cores={fp['cpu_count_physical']}  "
        f"cap={fp['gather_thread_cap']}"
    )
    per_row = _per_row_bytes(args.cols, args.dtype)
    policy, threshold = _print_policy(per_row)
    _decision_table(policy, threshold, per_row)

    if args.skip_bench:
        return 0

    if args.tmpdir is not None:
        work = Path(args.tmpdir)
        work.mkdir(parents=True, exist_ok=True)
        results = _ab_timing(args, threshold, per_row, work)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            results = _ab_timing(args, threshold, per_row, Path(tmp))

    if args.json is not None:
        _common.write_summary(
            args.json,
            results,
            meta={
                "benchmark": "check_gather_binding",
                "margin": config.get_gather_bind_llc_margin(),
            },
        )
        print(f"\n# wrote {args.json}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        repeat=3,
        warmup=1,
        cols=8,
        dtype="float32",
        tmpdir=True,
        threads=True,
        json=True,
        skip_correctness=False,
    )
    parser.add_argument("--rounds", type=int, default=5, help="interleaved A/B passes per cell")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the gather indices")
    parser.add_argument("--keep", action="store_true", help="do not delete the synthetic stores")
    # Worker-only switches (hidden from the orchestrator help).
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=_MODES, help=argparse.SUPPRESS)
    parser.add_argument("--indices", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--store-path", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    return run_worker(args) if args.child else run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
