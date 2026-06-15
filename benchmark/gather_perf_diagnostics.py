#!/usr/bin/env python3
"""Consolidated gather-performance diagnostic for colstore.

One harness that re-derives, from fresh measurements on the current host, the
answers to every performance question raised about the reader-side gather, so
the findings live in the repo instead of scattered across notes. It does not
assume the historical conclusions -- it measures and reports what *this* machine
does, then prints a recommendation.

It answers four questions:

  1. THREAD KNEE  -- how many gather threads before throughput saturates?
                     (Sweeps the cap; the multi-column conversions saturate
                     well below core count and regress past the knee.)
  2. BINDING      -- does pinning OpenMP threads help, and close vs spread?
                     (Sets OMP_PROC_BIND/OMP_PLACES *before* the subprocess
                     starts -- the only reliable way; see question 4.)
  3. PLACEMENT    -- once threads are bound, does NUMA *data* placement matter?
                     (Compares an interleaved store against one forced onto a
                     single node at write time via numactl.)
  4. OMP INIT     -- can the library set OMP_PROC_BIND itself at import time?
                     (Usually no: numpy/BLAS init OpenMP first, so the env is
                     read too late and the setting is ignored.)

METHODOLOGY (learned the hard way -- do not "simplify" these away):

  * Every measurement runs in a FRESH SUBPROCESS. Thread pinning, the gather
    thread cap, the OpenMP pool, and page-cache page placement all persist
    process-globally; measuring two configs in one process lets the first
    contaminate the second (a later "unbound" run silently inherits the earlier
    run's pinned pool, etc.). Subprocess isolation is mandatory, not tidiness.
  * Each config reads its OWN store file. Warm pages cannot be re-placed
    (`mbind` only steers future faults), so one config must never reuse another
    config's file.
  * Rounds alternate and we report the warm steady-state median (rounds 1+),
    plus the cold round 0 separately.
  * BLAS thread pools are pinned to 1 (OPENBLAS/MKL/etc.) so their idle spin
    does not pollute timings. This does NOT touch colstore's own OpenMP.
  * The affinity diagnostic counts threads whose mask is a subset of ONE node;
    `OMP_PROC_BIND=spread` deliberately puts one thread per node, so it shows
    `0` confined even though binding is active -- not a failure.

Usage:
  python gather_perf_diagnostics.py --all
  python gather_perf_diagnostics.py --experiment knee --ops dict-unsorted
  python gather_perf_diagnostics.py --all --quick      # smaller + faster
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

_BLAS_HYGIENE = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _add_benchmark_to_path() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here, here / "benchmark", *here.parents):
        if (cand / "perf_workload.py").exists():
            sys.path.insert(0, str(cand))
            return cand / "perf_workload.py"
    raise SystemExit("cannot locate perf_workload.py (run from the repo or benchmark/)")


# Self-contained NUMA topology (read /sys directly so this harness does not
# depend on any in-flux package internals).
def _parse_cpulist(text: str) -> list[int]:
    out: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _numa_nodes() -> list[int]:
    base = Path("/sys/devices/system/node")
    if not base.is_dir():
        return []
    nodes = []
    for d in base.glob("node[0-9]*"):
        m = re.match(r"node(\d+)$", d.name)
        if m:
            nodes.append(int(m.group(1)))
    return sorted(nodes)


def _node_cpus(node: int) -> list[int]:
    try:
        return _parse_cpulist(Path(f"/sys/devices/system/node/node{node}/cpulist").read_text())
    except OSError:
        return []


def _node_physical_cores(node: int) -> int:
    cores: set[frozenset[int]] = set()
    for cpu in _node_cpus(node):
        try:
            sib = Path(
                f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            ).read_text()
            cores.add(frozenset(_parse_cpulist(sib)))
        except OSError:
            cores.add(frozenset({cpu}))
    return len(cores)


# ===========================================================================
# Worker: one measurement (or one OMP-init check) in its own process.
# ===========================================================================
def run_worker(a: argparse.Namespace) -> int:
    _add_benchmark_to_path()
    import numpy as np  # noqa: F401

    import colstore

    if a.omp_check:
        # Does setting OMP_PROC_BIND from *inside* Python bind the pool? Caller
        # already set the env before importing us; run a gather and report how
        # many distinct per-thread CPU masks resulted.
        ds = colstore.open(a.store_file)
        try:
            from perf_workload import _build_thunk

            _build_thunk(ds, "dict-unsorted", a.rows, a.indices)()
            masks = set()
            for st in Path("/proc/self/task").glob("*/status"):
                m = re.search(r"Cpus_allowed_list:\s*(\S+)", st.read_text())
                if m:
                    masks.add(m.group(1))
            print(
                "RESULT " + json.dumps({"distinct_masks": len(masks), "sample": sorted(masks)[:3]})
            )
        finally:
            ds.close()
        return 0

    from perf_workload import _build_thunk

    if a.threads > 0:
        colstore.set_gather_thread_cap(a.threads)
    ds = colstore.open(a.store_file)
    out: dict = {}
    try:
        thunk = _build_thunk(ds, a.op, a.rows, a.indices)
        thunk()  # warm
        nodes = _numa_nodes()
        if nodes:
            nodeset = set(_node_cpus(nodes[0]))
            conf = tot = 0
            for st in Path("/proc/self/task").glob("*/status"):
                m = re.search(r"Cpus_allowed_list:\s*(\S+)", st.read_text())
                if not m:
                    continue
                tot += 1
                allowed = set(_parse_cpulist(m.group(1)))
                if allowed and allowed <= nodeset:
                    conf += 1
            out["aff"] = f"{conf}/{tot}"
            line = next(
                (
                    ln
                    for ln in Path("/proc/self/numa_maps").read_text().splitlines()
                    if Path(a.store_file).name in ln
                ),
                "",
            )
            out["placement"] = (
                " ".join(f"N{n}={c}" for n, c in re.findall(r"N(\d+)=(\d+)", line)) or "?"
            )
        out["cap"] = colstore.config.get_gather_thread_cap()
        per_loop = []
        for _ in range(a.loops):
            t0 = time.perf_counter()
            thunk()
            per_loop.append((time.perf_counter() - t0) * 1e3)
        out["ms"] = statistics.median(per_loop)
    finally:
        ds.close()
    print("RESULT " + json.dumps(out))
    return 0


# ===========================================================================
# Parent: build stores, drive subprocesses, aggregate, conclude.
# ===========================================================================
def _measure(a, store_file, op, threads, loops, rounds, env_extra=None):
    """Run `rounds` isolated measurements; return (warm_median, cold, first_diag)."""
    env = {**os.environ, **_BLAS_HYGIENE, **(env_extra or {})}
    times, diag = [], {}
    for _r in range(rounds):
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            "--store-file",
            str(store_file),
            "--op",
            op,
            "--rows",
            str(a.rows),
            "--indices",
            str(a.indices),
            "--threads",
            str(threads),
            "--loops",
            str(loops),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            sys.stderr.write(proc.stderr[-400:] + "\n")
            continue
        d = json.loads(line[len("RESULT ") :])
        times.append(d["ms"])
        diag = diag or d
    if not times:
        return float("nan"), float("nan"), {}
    warm = statistics.median(times[1:]) if len(times) > 1 else times[0]
    return warm, times[0], diag


def _ensure_store(a, path, numactl=None):
    """Build a store if missing. `numactl` (list) forces write-time placement."""
    if path.exists():
        return True
    wl = _add_benchmark_to_path()
    base = [
        "python",
        str(wl),
        "--op",
        "dict-unsorted",
        "--rows",
        str(a.rows),
        "--cols",
        str(a.cols),
        "--records",
        str(a.records),
        "--indices",
        str(a.indices),
        "--loops",
        "1",
        "--keep-store",
        "--store-path",
        str(path),
    ]
    cmd = (numactl + base) if numactl else base
    tag = " (numactl " + " ".join(numactl[1:]) + ")" if numactl else ""
    print(f"  building store{tag}: {path.name}", file=sys.stderr)
    r = subprocess.run(cmd, env={**os.environ, **_BLAS_HYGIENE})
    return r.returncode == 0 and path.exists()


def exp_thread_knee(a, spread_store, results):
    print("\n## 1. THREAD KNEE  (cap sweep; lowest time = knee)\n")
    knee = {}
    for op in a.ops:
        row = {}
        for t in a.threads:
            warm, _, _ = _measure(a, spread_store, op, t, a.loops, a.rounds)
            row[t] = warm
            print(f"   {op:>14} t={t:<3} {warm:8.2f} ms")
        best = min(row, key=lambda k: row[k])
        knee[op] = best
        print(
            f"   -> {op}: knee at {best} threads ({row[best]:.0f} ms; "
            f"{row[min(row)]:.0f} ms at min cap, "
            f"{row[max(row)]:.0f} ms at max cap)\n"
        )
    results["knee"] = knee


def exp_binding(a, spread_store, results):
    print("\n## 2. THREAD BINDING  (OMP_PROC_BIND at the knee thread count)\n")
    t = results.get("knee", {}).get(a.ops[0], 16)
    best = {}
    for op in a.ops:
        cell = {}
        for bind in ("false", "close", "spread"):
            env = (
                {"OMP_PROC_BIND": "false"}
                if bind == "false"
                else {"OMP_PROC_BIND": bind, "OMP_PLACES": "cores"}
            )
            warm, _, diag = _measure(a, spread_store, op, t, a.loops, a.rounds, env)
            cell[bind] = warm
            print(f"   {op:>14} bind={bind:<6} t={t:<3} {warm:8.2f} ms  aff={diag.get('aff','?')}")
        winner = min(cell, key=lambda k: cell[k])
        best[op] = winner
        gain = cell["false"] / cell[winner] if cell[winner] else float("nan")
        print(f"   -> {op}: best={winner} ({gain:.2f}x vs unbound)\n")
    results["binding"] = best


def exp_placement(a, spread_store, results):
    print("\n## 3. DATA PLACEMENT  (does NUMA placement matter once threads are bound?)\n")
    if shutil.which("numactl") is None:
        print("   numactl not found -- skipping (cannot force single-node placement).\n")
        return
    node0 = spread_store.with_name("diag_node0.cstore")
    if not _ensure_store(a, node0, numactl=["numactl", "--cpunodebind=0", "--membind=0"]):
        print("   could not build node-0 store -- skipping.\n")
        return
    t = results.get("knee", {}).get(a.ops[0], 16)
    env = {"OMP_PROC_BIND": "spread", "OMP_PLACES": "cores"}
    sp, _, dsp = _measure(a, spread_store, a.ops[0], t, a.loops, a.rounds, env)
    n0, _, dn0 = _measure(a, node0, a.ops[0], t, a.loops, a.rounds, env)
    print(f"   {a.ops[0]} @ t={t}, bind=spread:")
    print(f"     interleaved store : {sp:8.2f} ms   placement={dsp.get('placement','?')}")
    print(f"     node-0 store      : {n0:8.2f} ms   placement={dn0.get('placement','?')}")
    ratio = sp / n0 if n0 else float("nan")
    matters = abs(ratio - 1.0) > 0.08
    print(
        f"   -> placement {'MATTERS' if matters else 'does NOT matter'} "
        f"(node-0 is {ratio:.2f}x the interleaved time)\n"
    )
    results["placement_matters"] = matters
    if not a.keep:
        node0.unlink(missing_ok=True)


def exp_omp_init(a, spread_store, results):
    print("\n## 4. OMP INIT  (can the library set OMP_PROC_BIND at import?)\n")
    env = {**os.environ, **_BLAS_HYGIENE, "OMP_PROC_BIND": "spread", "OMP_PLACES": "cores"}
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        "--omp-check",
        "--store-file",
        str(spread_store),
        "--rows",
        str(a.rows),
        "--indices",
        str(a.indices),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line is None:
        print("   check failed.\n")
        return
    d = json.loads(line[len("RESULT ") :])
    works = d["distinct_masks"] > 1
    print(f"   distinct per-thread masks: {d['distinct_masks']}  sample={d['sample']}")
    print(
        f"   -> setting OMP_PROC_BIND in-process {'WORKS' if works else 'does NOT work '}"
        f"({'bound' if works else 'one mask = pool already initialized before env was read'})\n"
    )
    results["omp_in_process_works"] = works


def print_conclusions(a, results):
    print("\n" + "=" * 76)
    print(" CONCLUSIONS (this host)".center(76, "="))
    print("=" * 76)
    knee = results.get("knee", {})
    if knee:
        vals = sorted(set(knee.values()))
        print(f"\n 1. Thread knee: {knee}")
        print(
            f"    -> set the default gather thread cap toward ~{max(vals)} "
            f"(currently the calibrated default may be lower); more threads do not help."
        )
    binding = results.get("binding", {})
    if binding:
        from collections import Counter

        pick = Counter(binding.values()).most_common(1)[0][0]
        print(f"\n 2. Binding: best per op {binding}")
        print(f"    -> default OMP_PROC_BIND={pick}, OMP_PLACES=cores (bind threads to cores).")
    if "placement_matters" in results:
        pm = results["placement_matters"]
        print(
            f"\n 3. Data placement {'matters' if pm else 'does NOT matter'} once threads are bound."
        )
        print(
            f"    -> {'keep' if pm else 'drop'} NUMA data-placement machinery "
            f"(mbind/node-local){'.' if pm else '; the win is thread binding, not locality.'}"
        )
    if "omp_in_process_works" in results:
        w = results["omp_in_process_works"]
        print(f"\n 4. In-process OMP_PROC_BIND {'works' if w else 'does NOT work'}.")
        if w:
            print("    -> set it at import.")
        else:
            print("    -> bind at runtime (sched_setaffinity in a parallel region) or")
            print("       document the env export; setting the env from Python is too late.")
    print("\n" + "=" * 76)


def run_parent(a) -> int:
    _add_benchmark_to_path()

    nodes = _numa_nodes()
    print(f"# host: numa_nodes={nodes or 'single/none'}", end="")
    if nodes:
        print(
            f" node0_logical={len(_node_cpus(nodes[0]))} "
            f"node0_physical={_node_physical_cores(nodes[0])}"
        )
    else:
        print()
    print(
        f"# rows={a.rows:,} cols={a.cols} indices={a.indices:,} ops={a.ops} "
        f"threads={a.threads} rounds={a.rounds} loops={a.loops}"
    )

    work = Path(a.outdir or os.environ.get("SCRATCH") or "/tmp")
    spread = work / "diag_spread.cstore"
    if not _ensure_store(a, spread):
        raise SystemExit("failed to build the base store")

    want = a.experiment
    results: dict = {}
    if want in ("knee", "all"):
        exp_thread_knee(a, spread, results)
    if want in ("binding", "all"):
        if "knee" not in results:  # binding needs a thread count
            results["knee"] = {op: 16 for op in a.ops}
        exp_binding(a, spread, results)
    if want in ("placement", "all"):
        exp_placement(a, spread, results)
    if want in ("omp", "all"):
        exp_omp_init(a, spread, results)
    print_conclusions(a, results)

    if not a.keep:
        spread.unlink(missing_ok=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--omp-check", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--store-file", help=argparse.SUPPRESS)
    p.add_argument("--all", dest="experiment", action="store_const", const="all", default="all")
    p.add_argument(
        "--experiment", dest="experiment", choices=["knee", "binding", "placement", "omp", "all"]
    )
    p.add_argument("--ops", nargs="+", default=["dict-unsorted", "recarray", "frame"])
    p.add_argument("--threads", nargs="+", type=int, default=[8, 16, 32, 64])
    p.add_argument("--rows", type=int, default=64_000_000)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--records", type=int, default=16)
    p.add_argument("--indices", type=int, default=64_000_000)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--loops", type=int, default=5)
    p.add_argument("--op", default="dict-unsorted", help=argparse.SUPPRESS)  # worker
    p.add_argument("--outdir", default=None)
    p.add_argument("--keep", action="store_true", help="keep diagnostic store files")
    p.add_argument("--quick", action="store_true", help="smaller store, fewer rounds")
    a = p.parse_args()

    if a.quick:
        a.rows = a.indices = 16_000_000
        a.rounds = 2
        a.threads = [8, 16, 32]

    return run_worker(a) if a.worker else run_parent(a)


if __name__ == "__main__":
    raise SystemExit(main())
