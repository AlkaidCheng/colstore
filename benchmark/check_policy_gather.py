"""In-process A/B for the policy-based gather prototype.

Compares the legacy hand-written gather kernels against the policy-based
gather_core on identical data, *in one process*, by calling
``_gather.gather_variant`` / ``_gather.gather_bytes_variant`` with
``use_policy=False`` then ``True`` on the same arrays. Because both run in
the same process against the same pages, thread team, affinity, and turbo
state, the per-launch NUMA-placement variance that dominates a two-launch
comparison cancels -- the measurement is *paired*, so only the per-round
difference matters.

This needs only a SINGLE build (any toggle state); the diagnostic
``_variant`` entries are always compiled. ``build_flags()`` reports which
implementation the production ``gather`` symbol uses, for reference.

    PYTHONPATH=src python benchmark/check_policy_gather.py --repeat 200 \\
        --threads 16 --prefetch 8 --json policy_ab.json

Run on the target hardware, on an idle node, ideally under
``numactl --interleave=all`` with ``OMP_PROC_BIND=close OMP_PLACES=cores``
and an explicit ``--threads`` (not max). A near-zero paired delta whose 95%
CI straddles 0 is the expected "no regression" result -- the policy inlines
to the same inner loop, so you are looking for the *absence* of a signal.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import asdict, dataclass

import _common as _c
import numpy as np

from colstore import _gather


@dataclass
class Paired:
    op: str
    rounds: int
    legacy_mean_ms: float
    legacy_median_ms: float
    legacy_min_ms: float
    policy_mean_ms: float
    policy_median_ms: float
    policy_min_ms: float
    delta_mean_ms: float  # policy - legacy, paired
    delta_stdev_ms: float
    delta_ci95_ms: float  # half-width of the 95% CI on the mean delta
    delta_pct: float
    verdict: str


def _paired_samples(fn_legacy, fn_policy, *, rounds: int, warmup: int):
    for _ in range(warmup):
        fn_legacy()
        fn_policy()
    legacy: list[float] = []
    policy: list[float] = []
    for i in range(rounds):
        # Alternate which runs first to cancel any first/second-call bias.
        if i % 2 == 0:
            order = (("legacy", fn_legacy, legacy), ("policy", fn_policy, policy))
        else:
            order = (("policy", fn_policy, policy), ("legacy", fn_legacy, legacy))
        for _, fn, bucket in order:
            start = time.perf_counter()
            fn()
            bucket.append((time.perf_counter() - start) * 1000.0)
    return legacy, policy


def _summarize(op: str, legacy: list[float], policy: list[float]) -> Paired:
    diffs = [p - lg for p, lg in zip(policy, legacy, strict=True)]
    n = len(diffs)
    delta_mean = statistics.fmean(diffs)
    delta_sd = statistics.pstdev(diffs)
    ci95 = 1.96 * delta_sd / math.sqrt(n) if n else 0.0
    legacy_mean = statistics.fmean(legacy)
    pct = (delta_mean / legacy_mean * 100.0) if legacy_mean else 0.0
    if abs(delta_mean) <= ci95:
        verdict = "no significant difference"
    elif delta_mean > 0:
        verdict = "policy SLOWER"
    else:
        verdict = "policy FASTER"
    return Paired(
        op=op,
        rounds=n,
        legacy_mean_ms=legacy_mean,
        legacy_median_ms=statistics.median(legacy),
        legacy_min_ms=min(legacy),
        policy_mean_ms=statistics.fmean(policy),
        policy_median_ms=statistics.median(policy),
        policy_min_ms=min(policy),
        delta_mean_ms=delta_mean,
        delta_stdev_ms=delta_sd,
        delta_ci95_ms=ci95,
        delta_pct=pct,
        verdict=verdict,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=200, help="paired rounds (default 200)")
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
    tc, pf = args.threads, args.prefetch

    # Correctness gate for both implementations of both kernels, before timing.
    for use_policy in (False, True):
        out[:] = 0
        _gather.gather_variant(source, indices, out, use_policy, tc, pf)
        if not np.array_equal(out, expected):
            raise AssertionError(f"gather_variant(use_policy={use_policy}): mismatch")
        out[:] = 0
        _gather.gather_bytes_variant(source, byte_offsets, out, use_policy, tc, pf)
        if not np.array_equal(out, expected):
            raise AssertionError(f"gather_bytes_variant(use_policy={use_policy}): mismatch")

    results = [
        _summarize(
            "gather_indexed",
            *_paired_samples(
                lambda: _gather.gather_variant(source, indices, out, False, tc, pf),
                lambda: _gather.gather_variant(source, indices, out, True, tc, pf),
                rounds=args.repeat,
                warmup=args.warmup,
            ),
        ),
        _summarize(
            "gather_bytes",
            *_paired_samples(
                lambda: _gather.gather_bytes_variant(source, byte_offsets, out, False, tc, pf),
                lambda: _gather.gather_bytes_variant(source, byte_offsets, out, True, tc, pf),
                rounds=args.repeat,
                warmup=args.warmup,
            ),
        ),
    ]

    flags = sorted(_gather.build_flags())
    default = "policy" if "COLSTORE_USE_POLICY_GATHER" in flags else "legacy"
    print(f"\nproduction default (gather symbol): {default}   build_flags={flags or '[]'}")
    print(
        f"source={n_src:,} f8 ({n_src * 8 / 1e6:.0f} MB)  k={k:,}  "
        f"threads={tc or 'max'}  prefetch={pf}  paired_rounds={args.repeat}"
    )
    print(
        f"\n{'op':<16}{'legacy_med':>11}{'policy_med':>11}"
        f"{'Δmean':>10}{'95%CI':>9}{'Δ%':>8}  verdict"
    )
    print("-" * 84)
    for r in results:
        print(
            f"{r.op:<16}{r.legacy_median_ms:>11.3f}{r.policy_median_ms:>11.3f}"
            f"{r.delta_mean_ms:>+10.3f}{r.delta_ci95_ms:>9.3f}{r.delta_pct:>+7.1f}%  {r.verdict}"
        )

    if args.json:
        import json

        payload = {
            "fingerprint": _c.machine_fingerprint(),
            "meta": {
                "build_flags": flags,
                "production_default": default,
                "n_src": n_src,
                "k": k,
                "threads": tc,
                "prefetch": pf,
                "paired_rounds": args.repeat,
            },
            "results": [asdict(r) for r in results],
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
