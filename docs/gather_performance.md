# Gather performance diagnostics

`benchmark/gather_perf_diagnostics.py` is a single, self-contained harness that
re-derives — from fresh measurements on whatever host it runs on — the answers to
the performance questions about the reader-side gather (`dict`/`recarray`/`frame`
and the scattered fancy-index path). It exists so those answers live in the repo
and can be re-checked on new hardware, rather than being rediscovered from
scratch each time.

It does **not** hard-code the historical conclusions. It measures, then prints a
recommendation for the current machine. If a future CPU behaves differently, the
script will say so.

## What it answers

| # | Question | How it tests |
|---|----------|--------------|
| 1 | **Thread knee** — how many gather threads before throughput saturates? | Sweeps the gather thread cap and reports the fastest count per op. |
| 2 | **Binding** — does pinning OpenMP threads help, and `close` vs `spread`? | Runs each op under `OMP_PROC_BIND=false/close/spread` (set before the process starts) at the knee thread count. |
| 3 | **Placement** — once threads are bound, does NUMA *data* placement matter? | Compares an interleaved store against one forced onto a single node at write time via `numactl`, both read with threads bound. |
| 4 | **OMP init** — can the library set `OMP_PROC_BIND` itself at import? | Sets the env from inside Python, then checks whether the pool actually bound. |

## Usage

Run on a quiet compute node (`salloc` first); login nodes are noisy.

```bash
# everything (default); ~10-20 min at the default 64M-row size
python benchmark/gather_perf_diagnostics.py --all

# one experiment
python benchmark/gather_perf_diagnostics.py --experiment knee
python benchmark/gather_perf_diagnostics.py --experiment binding --ops dict-unsorted

# fast smoke (16M rows, fewer rounds) — for wiring/CI, not for decisions
python benchmark/gather_perf_diagnostics.py --all --quick
```

Useful flags: `--ops`, `--threads "8 16 32 64"`, `--rows`, `--cols`, `--indices`,
`--rounds`, `--loops`, `--outdir` (defaults to `$SCRATCH` or `/tmp`), `--keep`
(retain the diagnostic store files). It prints per-measurement lines as it goes
and a `CONCLUSIONS (this host)` block at the end.

## Methodology — do not "simplify" these away

These were learned by getting them wrong; each one silently corrupts the result
if dropped.

- **Every measurement runs in a fresh subprocess.** The gather thread cap, the
  OpenMP pool's affinity, and the page cache's page placement are all
  *process-global and persistent*. Measuring two configurations in one process
  lets the first contaminate the second — e.g. an "unbound" run that follows a
  pinned run silently inherits the pinned pool and looks fast. Subprocess
  isolation is correctness, not tidiness.
- **Each configuration reads its own store file.** Warm pages cannot be
  re-placed (`mbind` only steers *future* faults), so one policy's file must
  never be reused by another. The script builds a fresh file per placement.
- **Cold vs warm.** Round 0 reflects the requested placement at first fault;
  later rounds reflect whatever the page cache settled on. The harness reports
  the warm steady-state median and treats round 0 separately.
- **BLAS hygiene.** `OPENBLAS_NUM_THREADS=1` (and MKL/NUMEXPR/VECLIB) are set for
  every subprocess so idle BLAS spin does not pollute timings. This does **not**
  touch colstore's own OpenMP.
- **`OMP_PROC_BIND` must be set before the process starts.** The OpenMP runtime
  reads it once at init, and numpy/BLAS typically initialize OpenMP during
  import — so setting it from Python is too late (experiment 4 demonstrates
  this). The binding sweep therefore sets it in the subprocess environment.
- **Reading the affinity diagnostic.** The `aff=` field counts threads whose CPU
  mask is a subset of a *single* node. `OMP_PROC_BIND=spread` deliberately puts
  one thread per node, so it reports `0` confined even though binding is fully
  active. Do not read `spread`'s `0/` as "binding failed"; the timing is the
  evidence.

## Reference findings

These are the results from the development host (a multi-socket, multi-NUMA-node
x86-64 server) at 64M rows × 8 columns. **Treat them as the baseline to confirm
or refute, not as ground truth** — rerun the harness on your target.

1. **Thread knee ≈ 16.** All three multi-column conversions saturate around 16
   gather threads; 32 gives nothing and 64 regresses badly (the
   work-proportional ramp over-provisions for memory-bound work). The
   single-column scatter the original cap was tuned on saturated lower (~8), but
   conversions move ~8× the bytes and want ~2× the threads.
2. **Binding does *not* help — it regresses.** An early sweep suggested
   `OMP_PROC_BIND=spread` was ~1.3× faster, but a fuller placement × binding ×
   cap matrix (binding applied at runtime, the only way the library can) found
   spread binding **24–51% slower** across every placement and cap on the
   multi-node node: for a random-scatter gather you cannot co-locate a thread
   with its data, so spreading only adds remote-access and coherence cost.
   `gather_binding` ships **off** and is retained only as an opt-in lever.
3. **Placement *is* the lever — interleave wins.** With binding off, where the
   pages sit dominates: the best cell was an interleaved store read unbound, and
   the same gather confined to a single node was markedly slower. A separate
   cold-read A/B across contiguous, sorted, and scattered patterns confirmed
   interleave is at least as fast as local for every pattern and never slower.
   The earlier "placement doesn't matter once threads are bound" reading was an
   artifact of binding itself; on the realized (unbound) path, interleave is the
   win. This is the shipped `auto` default.
4. **The library cannot set `OMP_PROC_BIND` at import** on a numpy/BLAS stack
   (all threads showed one full-width mask). Binding must be applied at runtime
   from inside a parallel region, or exported in the environment before launch.

### What the reference findings imply for the code

- Raise the default gather thread cap toward the measured knee for multi-column
  conversions (calibrate per host rather than hard-coding). *Done:* the default
  scales with socket count to reach ~16 on a dual-socket host, and `calibrate()`
  brackets the sweep to 32 in case a wider host's knee is higher.
- Do **not** bind the gather's threads. The fuller placement × binding × cap
  matrix reversed the early binding result (24–51% slower on multi-node), so
  `gather_binding` ships off and is kept only as an opt-in lever.
- **Keep** the NUMA interleave placement path — once binding is off it is the
  win, not removable. The `auto` default interleaves on multi-node (writer-side
  at write time, reader-side at open for cold reads) and is a no-op elsewhere.

If a rerun on new hardware contradicts any of these, the `CONCLUSIONS` block will
report the divergence — investigate before assuming the old defaults still hold.

## Relationship to the other benchmark tools

- `benchmark/run_perf.sh` and `collect_perf_reference.sh` are the lower-level
  `perf`-counter tools (stat/record/annotate, IPC, cache/TLB misses, scaling and
  NUMA sweeps). Use them to understand *why* a number is what it is.
- `gather_perf_diagnostics.py` is the high-level harness that answers the
  decision questions and prints recommendations. Start here; drop to the `perf`
  tools when you need the microarchitectural detail.
