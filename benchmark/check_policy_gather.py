"""A/B microbenchmark for the policy-based gather prototype.

Times the two single-record kernels under test -- gather_indexed (via
``_gather.gather``) and gather_bytes (via ``_gather.gather_bytes``) -- in
isolation, over many rounds, and reports mean / stdev / median / min / p95
so a regression is visible above the run-to-run noise.

The prototype is a compile-time switch (COLSTORE_USE_POLICY_GATHER), so a
single build measures one implementation; the script reports which one via
``_gather.build_flags()``. To compare, build and run twice:

    # legacy (hand-written loops)
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.dev0 pip install -e . \\
        --no-build-isolation --force-reinstall
    PYTHONPATH=src python benchmark/check_policy_gather.py --json legacy.json

    # policy (gather_core + IndexedPolicy/BytesPolicy)
    COLSTORE_USE_POLICY_GATHER=1 SETUPTOOLS_SCM_PRETEND_VERSION=0.0.dev0 \\
        pip install -e . --no-build-isolation --force-reinstall
    PYTHONPATH=src python benchmark/check_policy_gather.py --json policy.json

then diff the two JSON files (or eyeball the two tables). "No regression"
means the policy mean is within noise of the legacy mean -- expected, since
the policy inlines to the same inner loop. Run on the target hardware; a
single-socket dev box will not reflect the production NUMA/bandwidth regime.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import asdict, dataclass

import _common as _c
import numpy as np

from colstore import _gather


@dataclass
class Stats:
    op: str
    rounds: int
    mean_ms: float
    stdev_ms: float
    median_ms: float
    min_ms: float
    p95_ms: float
    throughput_rows_per_s: float


def _samples(fn, *, rounds: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    out: list[float] = []
    for _ in range(rounds):
        start = time.perf_counter()
        fn()
        out.append((time.perf_counter() - start) * 1000.0)
    return out


def _summarize(op: str, samples: list[float], rows: int) -> Stats:
    samples_sorted = sorted(samples)
    p95 = (
        statistics.quantiles(samples_sorted, n=20)[18]
        if len(samples_sorted) >= 20
        else max(samples_sorted)
    )
    mean = statistics.fmean(samples)
    return Stats(
        op=op,
        rounds=len(samples),
        mean_ms=mean,
        stdev_ms=statistics.pstdev(samples),
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        p95_ms=p95,
        throughput_rows_per_s=rows / (mean / 1000.0) if mean > 0 else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=100, help="timed rounds (default 100)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--scale", type=float, default=1.0, help="multiply source/index sizes")
    parser.add_argument("--threads", type=int, default=0, help="thread cap (0 = OpenMP max)")
    parser.add_argument("--prefetch", type=int, default=8, help="prefetch distance (0 disables)")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    n_src = max(1024, int(16_000_000 * args.scale))
    k = max(256, int(4_000_000 * args.scale))
    rng = np.random.default_rng(20240608)
    source = rng.standard_normal(n_src)  # float64, itemsize 8
    indices = rng.integers(0, n_src, size=k).astype(np.int64)
    byte_offsets = indices * np.int64(source.itemsize)
    expected = source[indices]
    out = np.empty(k, dtype=np.float64)

    flags = _gather.build_flags()
    enabled = "COLSTORE_USE_POLICY_GATHER" in flags
    impl = "policy" if enabled else "legacy"

    # Correctness gate before any timing.
    _gather.gather(source, indices, out, args.threads, args.prefetch)
    if not np.array_equal(out, expected):
        raise AssertionError("gather_indexed: value mismatch")
    out[:] = 0
    _gather.gather_bytes(source, byte_offsets, out, args.threads, args.prefetch)
    if not np.array_equal(out, expected):
        raise AssertionError("gather_bytes: value mismatch")

    def run_indexed() -> None:
        _gather.gather(source, indices, out, args.threads, args.prefetch)

    def run_bytes() -> None:
        _gather.gather_bytes(source, byte_offsets, out, args.threads, args.prefetch)

    results = [
        _summarize(
            "gather_indexed", _samples(run_indexed, rounds=args.repeat, warmup=args.warmup), k
        ),
        _summarize("gather_bytes", _samples(run_bytes, rounds=args.repeat, warmup=args.warmup), k),
    ]

    print(
        f"\nimplementation: {impl}  (COLSTORE_USE_POLICY_GATHER "
        f"{'defined' if enabled else 'not defined'})"
    )
    print(
        f"source={n_src:,} f8 ({n_src * 8 / 1e6:.0f} MB)  k={k:,}  "
        f"threads={args.threads or 'max'}  prefetch={args.prefetch}  rounds={args.repeat}"
    )
    print(f"\n{'op':<16}{'mean':>9}{'stdev':>9}{'median':>9}{'min':>9}{'p95':>9}{'Mrows/s':>11}")
    print("-" * 72)
    for r in results:
        print(
            f"{r.op:<16}{r.mean_ms:>8.3f}{r.stdev_ms:>9.3f}{r.median_ms:>9.3f}"
            f"{r.min_ms:>9.3f}{r.p95_ms:>9.3f}{r.throughput_rows_per_s / 1e6:>11.1f}"
        )

    if args.json:
        import json

        payload = {
            "fingerprint": _c.machine_fingerprint(),
            "meta": {
                "implementation": impl,
                "build_flags": sorted(flags),
                "n_src": n_src,
                "k": k,
                "threads": args.threads,
                "prefetch": args.prefetch,
                "rounds": args.repeat,
            },
            "results": [asdict(r) for r in results],
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
