# colstore Performance Optimization Series — Combined Reference Summary

`colstore` is a memory-mapped columnar binary format (`.cstore`) for
structured-array datasets — load and write, not streaming — built on C++17 +
OpenMP gather kernels, Cython bindings, and a Python reader/writer (`ColStore`,
`ColumnView`, `TableView`, with `ds.dict()` / `ds.recarray()` / `ds.frame()`
shortcuts). It deploys on NERSC Perlmutter CPU nodes (dual AMD EPYC 7763,
Zen 3). Priorities: **speed first, then maintainability/extensibility, then
readability.**

This is the consolidated record of every performance-engineering change, keyed
to the merged pull requests (PR numbers in **bold**). It supersedes three
earlier summaries (a six-stage summary and two byte-identical twelve-stage
summaries) — which covered only the multi-record native-read effort, PRs #26–#44
— by adding the **foundational threading/reader/NUMA work that preceded it
(PRs #11–#25)** and the **follow-on optimizations that came after it** (mask-gate
calibration, the bin-reuse parallel-regime fix, parallel strided copy, and the
uniform reciprocal divide). Two items the old summaries listed as *unsolved* were
in fact already implemented (`frame()` per-column BlockManager, PR #24; mask
density-gate calibration, PR #47) and are corrected here.

The "Stage N" labels below are the project's own, from the optimization-series
summary doc (**PR #45**); they name PRs #26–#44. PR-numbered work outside that
range is labeled by its role (foundational, or a follow-on to the stage it
touches).

---

## Contents

- [At-a-glance: every optimization, by PR](#at-a-glance-every-optimization-by-pr)
- [Correctness & robustness fixes](#correctness--robustness-fixes)
- [Methodology](#methodology)
- [Phase 0 — foundational performance work (PRs #11–#25)](#phase-0--foundational-performance-work)
- [Round 1 — native read path + writer (Stages 1–6)](#round-1--native-read-path--writer)
- [Round 2 — selector-side overheads (Stages 7–12)](#round-2--selector-side-overheads)
- [Follow-on optimizations (post-series)](#follow-on-optimizations-post-series)
- [Structural refactor that preserved kernel performance](#structural-refactor-that-preserved-kernel-performance)
- [Net effect](#net-effect)
- [Rejected alternatives](#rejected-alternatives)
- [Open / deferred items](#open--deferred-items)

---

## At-a-glance: every optimization, by PR

Ordered by merge (PR number). Speedups are on the deployment hardware (EPYC 7763)
unless noted, stated **with the regime the win lives in** — most converge to ~1×
(no regression) outside it.

| PR | Phase | Optimization | What changed | Best speedup | Where the win lives |
|----|-------|--------------|--------------|--------------|---------------------|
| **#11** | Foundation | Auto thread tuning | Per-call OpenMP resolution (serial <262K, ~1 thread/1M, cap = cores//2 ∈ [1,8]); cached autotune | **up to ~40×** vs the oversubscribed baseline | Many-core hosts (was 244 threads spin-waiting) |
| **#13** | Foundation | Per-column thread-budget split | `_gather_many` divides the cap across concurrent columns (`cap // n_workers`) | 483 → 188 ms | Multi-column reads on many-core (was 2.8× slower than numpy) |
| **#14** | Foundation | Always use the C++ kernel | Drop the `np.take` sub-threshold delegation; no-alloc `gather_into` entry | **1.5–2× (up to 4×)** | All gather sizes, even single-threaded (numpy re-validates indices) |
| **#23** | Foundation | Reader shortcuts + parallel contiguous copy | `ds.dict/recarray/frame()` direct methods; `_parallel_copy` for large reads | `dict()` **~11.8×** (~14.5 GB/s) | Large contiguous whole-store reads |
| **#24** | Foundation | `frame()` per-column BlockManager | `_make_dataframe_no_consolidate` shares memory, skips pandas consolidation | **5.35×** (closes most of a 10× gap) | `ds.frame()` construction |
| **#25** | Foundation | NUMA page-cache interleave | Writer `set_mempolicy` + reader `mbind` `MPOL_INTERLEAVE` | wall **1.12–1.19×**, CPU **1.58–2.23×** | Multi-socket / multi-NPS reads |
| **#26** | Stage 1 | Native contiguous range copy | `copy_multirecord_range`: binary-search start + one `memcpy`/record (serial) | **5–23×** | Many-small-record layouts |
| **#27** | Stage 1 | Thread-count floor fix | Round-up division + min 2 threads past the 256K threshold (fixes a #11 bug) | **2.0–17.9×** | 300K–20M elements (was floored to 1 thread) |
| **#28** | Stage 2 | Fused branchless fancy gather | `gather_multirecord_typed<T>`: in-register branchless binning, OpenMP over indices | **2.65–3.02×** (1-thread) | Unsorted multi-record fancy reads, native dtype |
| **#29** | Stage 3 | Auto-calibrated prefetch + CLI | Runtime `prefetch_distance` (`auto`), per-host table, cache mgmt, `colstore` CLI | **+20–29%** reclaimed | Removes a fixed-prefetch pessimization on Zen 3 |
| **#30** | Stage 4 | Vectored streaming writer | One `writev()`/record (iovec) replacing C+2 syscalls + flushes | **2.36–2.50×** | 50–200-row records |
| **#31** | Stage 5 | Bin-reuse multi-column gather | Bin once into `int32`, one bins-fed pass/column (`_bins`/`_withbins`) | **1.74–2.68×** | ≥2-column unsorted reads; also cuts CPU ~C-fold |
| **#36** | Stage 6 | Native sorted fancy gather | `gather_multirecord_sorted`: monotone walk, O(K+R); kills the per-record Python loop | **1.85–53.5×** | Sorted multi-record reads; largest at high R |
| **#39** | Stage 7 | Native strided range reads | `gather_multirecord_strided`: arithmetic stream, no index array, O(K+R) both ways | **4.3–4.8× fwd, 5.5–7.1× rev** | `step≠1` multi-record slices |
| **#40** | Stage 8 | Sampled sortedness check | Probe 16 pairs to *reject* unsorted early; exact full pass only on ascending samples | **1.05–1.06×** end-to-end | Unsorted reads at K≈10⁷ (serial-cost removal) |
| **#41** | Stage 9 | Uniform-record fast path | Closed-form `r = idx / rows_per_record`; arithmetic-bin trio | **1.4–5.2×** | Uniform-layout files; largest single-column, high R |
| **#42** | Stage 10 | Record-base precompute | One `record_base[r]` scalar for trailing irregular columns | **1.02–1.19×** (kernel 1.20×) | Irregular multi-column reads, small-to-mid R |
| **#43** | Stage 11 | Zero-copy read API | `copy=False` returns read-only memmap views; O(1) | **1.7–2.3×**, ½ peak RAM | Whole compacted-store reads, native dtype |
| **#44** | Stage 12 | Boolean-mask-native path | `gather_multirecord_mask`: word-at-a-time mask scan, no index materialization | **4–6× (1col), 2–3× (2col), up to 9.1×** | Mask density above gate; peaks on whole-record cuts |
| **#47** | Stage 12 f/u | Mask-density gate calibration | Default 0.15 → 0.0; `mask-density` becomes a per-host calibration target | node **1.35–3.2×** at every density | Recovers wins the container-derived gate left on the table |
| **#69** | Stage 5 fix | Bin-reuse parallel-regime gate | Decline bin-reuse where the concurrent column pool out-fields it | recovers **~3×** (0.32× → parity+) | Moderate K below the scaling knee (K≈1M) |
| **#71** | Reader | Parallel strided-slice copy | Route strided `.array()` through `_parallel_copy` (was forced serial) | 0.87× → parallel | Large strided reads (>16 MiB) |
| **#93** | Stage 9 fix | Uniform reciprocal divide | Replace the per-index `div` with a magic-reciprocal multiply | flips **0.75–0.94× → faster-than-generic** | Uniform route, multi-thread large-K (magnitude pending on-node A/B) |

A large structural refactor (the policy migration, PRs #61–#67) unified the six
gather families onto a templated `gather_core<T, Policy>` while holding kernel
codegen byte-identical; its one performance-relevant result is in
[Structural refactor](#structural-refactor-that-preserved-kernel-performance).
The bulk of PRs #48–#92 are docs/test/benchmark-harness consolidation — not
optimizations — and are omitted here.

---

## Correctness & robustness fixes

Landed alongside the optimization work; equivalence asserted by tests.

| PR | Fix | What it addressed |
|----|-----|-------------------|
| **#21** | Windows wheel support | POSIX-only `fcntl` at import + GCC-only `__builtin_prefetch` blocked MSVC; added `colstore._lock`, the `COLSTORE_PREFETCH` macro (`_mm_prefetch` on MSVC), CI matrix; plus a lock-ordering data-corruption fix in `update` mode |
| **#32** | Alignment-safe loads | Packed columns place typed loads at odd addresses (UB); all kernels load via `load_unaligned` memcpy helper, `-fsanitize=alignment`-clean at parity |
| **#33** | Byte-order-correct fancy reads | Byte-offset gathers delivered raw LE bytes into native outputs; destinations now disk-typed, no-op convert on LE |
| **#34** | Backend semantics (doc) | Multi-record fancy reads never consulted `backend`; contract stated and pinned by a test |
| **#35** | writev zero-progress guard | Zero return with buffers outstanding now raises instead of spinning |
| **#37** | Non-contiguous index arrays | Strided selectors (`[::2]`, `[::-1]`) reached kernels as contiguous int64 (wrong values / segfault); normalized + `_require_c_contiguous` at every Cython entry |
| **#38** | NUMA local-policy test | Test-only spy contamination; the NUMA module (PR #25) is active on the node |
| **#70** | Interleaved compaction timing | Benchmark methodology: time before/after reads interleaved to kill a NUMA first-touch artifact |
| **#83** | check_numa cold A/B | Evict with no reader open so the cold-cache A/B is actually cold |

---

## Methodology

Held at every stage:

- **Premise-verified first.** Each change was profiled or prototyped before
  implementation; several premises were overturned by measurement (e.g. Stage 2:
  the cost was `searchsorted` binning at 73–90%, not the byte-offset temporaries;
  PR #24: `pd.DataFrame._from_arrays(verify_integrity=False)`, the hypothesized
  `frame()` fix, was disproved — it still consolidates).
- **Equivalence asserted, not assumed.** Each stage ships a `benchmark/check_*.py`
  with a correctness gate that runs before timing and a toggle reproducing the
  reference route; byte-identity is hash-checked where applicable.
- **Validated on the deploy hardware.** All verdicts are EPYC 7763. The
  measurement doctrine: **interleaved paired A/B in one process** is the only
  trustworthy comparison on this NUMA machine (sequential separate-process runs
  carry ±50% first-touch variance). Serious runs use `numactl --interleave=all`,
  `OMP_PROC_BIND=close OMP_PLACES=cores`, idle node. Hot-path refactors also
  require disassembly-level codegen verification (PR #93's reciprocal divide was
  confirmed `div`-free by `objdump`).
- **Gated routing.** Every routing change ships behind a measured gate whose
  constant doubles as the benchmark baseline, so the shipped floor is the
  pre-change behavior.

**Test-suite growth:** 311 (pre-Round 1) → 407 (post-Round 1) → 600 (post-Round 2)
→ 617 (mask calibration, #47) → 671–672 (policy migration) → **711 passed /
1 skipped** (reciprocal divide, #93 onward), passing in both default and
`OMP_NUM_THREADS=8` regimes throughout.

**Kernel-name note:** the per-stage C symbols (`colstore_gather_multirecord_*`)
remain the public ABI. After the policy migration their bodies route through
`gather_core<T, Policy>` over six policy structs (`IndexedPolicy`,
`BytesPolicy`, `MultiRecordPolicy`, `MultiRecordBinsPolicy`, `WithBinsPolicy`,
`RBasePolicy`) plus hand-written kernels for the loop-shape-divergent cases
(sorted, strided, mask, range-copy, uniform trio). Every stage below maps onto
current code.

---

## Phase 0 — foundational performance work

Before any native multi-record kernel existed, the gather path was made to
thread sanely, the C++ kernel was made unconditional, the reader-conversion
shortcuts were added, and multi-socket placement was fixed. *(The record-based
multi-record file format itself landed in **PR #16** — the substrate the Round 1
kernels operate on, a feature rather than an optimization.)*

### Auto thread tuning — **PR #11**
- **Problem.** The C++ backend was up to **40× slower than numpy** on many-core
  machines: 244 OpenMP threads on a 128-core box spent ~55% of cycles in barrier
  spin-waits (env-driven oversubscription).
- **Change.** Per-call thread resolution inside the kernel via a `num_threads()`
  clause: serial below a 262K-element threshold, then ~1 thread per 1M elements,
  clamped to a hardware-derived cap (`physical_cores // 2`, clamped `[1, 8]` —
  the bandwidth-bound gather saturates at a small count regardless of core
  count). Plus an opt-in cached autotuner (`autotune.calibrate()`, fingerprinted)
  for the last 10–20%; calibration never runs implicitly, so import stays fast.
- **Result.** Default experience matches a hand-tuned small thread count with no
  setup; the foundation `resolve_thread_count` / `get_gather_thread_cap()` that
  every later stage builds on.

### Per-column thread-budget split — **PR #13**
- **Problem.** `_gather_many` ran each column at the full cap, so N concurrent
  columns fielded N × cap threads (160 on a 20-column read, cap=8). The
  20-column `to_dict` at 10M rows was **483 ms vs numpy's 172 ms**.
- **Change.** Thread a per-call cap through `_gather_one` → `kernels.gather`;
  `_gather_many` divides it: `per_column_cap = max(1, cap // n_workers)`. The
  thread product is bounded by the cap by construction.
- **Result.** 10M × 20-col `to_dict` 483 → **188 ms** (parity with numpy).

### Always use the C++ kernel — **PR #14**
- **Overturned premise.** A prior change delegated to `np.take` when
  `resolve_thread_count` returned 1, assuming numpy's C loop wins at small sizes.
  Measurement overturned it: even genuinely single-threaded, the C++ kernel beats
  `np.take` by **1.5–2×** (up to 4× at 10M sorted indices) — `np.take`
  re-validates every index (~25 ms `sys` in a 42 ms call) while colstore already
  validated upstream. There is no size below which delegating helps.
- **Change.** `kernels.gather` calls a no-allocation `_gather.gather_into` entry
  directly; only byte-order/unsupported-dtype *correctness* fallbacks to numpy
  remain. Adds the `perf_suite.py` regression harness (`--compare baseline.json`,
  10% noise band).

### Reader shortcuts + parallel contiguous copy — **PR #23**
- **Change.** Direct `dict()` / `recarray()` / `frame()` on `ColStoreReader` (no
  `[:]` round-trip), and a parallel memcpy (`_parallel_copy`) for large
  contiguous reads where one core can't saturate the bus.
- **Result.** `ds.dict()` reaches **~11.8×** (~14.5 GB/s) on a 1 GB / 50-column
  store. This is the PR that exposed the `frame()` gap addressed next.

### `frame()` per-column BlockManager — **PR #24**
- **Problem.** `ds.dict()` ran near-bandwidth but `ds.frame()` lagged ~10×
  (697 ms vs 69 ms). Root cause: `pd.DataFrame(dict)` consolidates 50 float64
  columns into one 2D block — a 1 GB extra alloc + memcpy. The gather was fine.
- **Disproved hypothesis.** `pd.DataFrame._from_arrays(verify_integrity=False)`
  skips *validation* but **still consolidates** (~220 ms float64 / ~700 ms mixed,
  matching the default constructor). Per-column BlockManager is required.
- **Change.** `_make_dataframe_no_consolidate` calls
  `create_block_manager_from_column_arrays(..., consolidate=False)` and wraps with
  `DataFrame._from_mgr` — each column its own block, memory shared with the input
  arrays, zero extra copies. Falls back to `pd.DataFrame(dict)` with a warning on
  pandas versions lacking the private API.
- **Result.** **5.35×** over the naive constructor (`check_frame_construction.py`),
  closing most of the dict/frame gap. *(Residual headroom remains — see
  [open items](#open--deferred-items) #1.)*

### NUMA page-cache interleave — **PR #25**
- **Problem.** On multi-socket hosts, page-cache pages landed on whichever node
  serviced the writer's I/O; a wide gather then issued 7/8 of its loads
  cross-socket.
- **Change.** `config.set_numa_policy("auto")` (default): writer-side
  `set_mempolicy(MPOL_INTERLEAVE)` distributes `MAP_SHARED` pages round-robin at
  write time (the warm-cache win); reader-side `mbind(MPOL_INTERLEAVE)` handles
  cold reads of externally-written files (`mbind` can't move already-resident
  warm pages on `MAP_SHARED`). `"local"` opts out (measured ~4% slower under
  interleave at `workers=1`, where a single thread now does 7/8 remote loads).
- **Result.** On a 1 GB / 50-column store: wall **1.19× / 1.12×**, CPU **1.58× /
  2.23×** (`dict()` / `frame()`) — the signature of remote-stall elimination.
  *(Implementation note: `maxnode` must be "bitmap bits + 1", not "max id + 1",
  or the kernel's endmask silently drops the top node.)*

---

## Round 1 — native read path + writer

Round 1 (Stages 1–6, **PRs #26–#36**) made the multi-record read side native
end-to-end and fixed the writer's syscall pattern.

### Stage 1 — Native contiguous multi-record range copy — **PR #26**
- **Problem.** Contiguous range reads spanning records ran a per-record Python
  loop.
- **Change.** C++ `copy_multirecord_range`: binary-search the start record, one
  `memcpy` per overlapping record. Deliberately serial (`memcpy` is already
  bandwidth-bound). Python fallback for extension-less builds.
- **Measured.** 5–23× on many-small-record layouts, → ~1× on few-large; no
  regression where per-record overhead didn't dominate.

### Thread-count floor fix — **PR #27**
- **Problem.** Found during Stage 1 validation: an integer-rounding bug in
  `resolve_thread_count` (from PR #11) floored every gather below 1M elements to
  one thread, dead-coding the 256K threshold.
- **Measured.** Once unclamped: 2.0× at 2 threads (300K–1M), 5.0× at 8 threads
  (5M), 17.9× at 32 threads (20M). Knee ≈ one thread per 1M elements.

### Stage 2 — Fused branchless multi-record fancy gather — **PR #28**
- **Overturned premise.** The unsorted path ran `np.searchsorted` → materialized
  byte offsets → raw kernel. The proposal blamed the offsets; profiling showed
  `searchsorted` was **73–90%**, offsets only 7–11%.
- **Change.** `gather_multirecord_typed<T>`: branchless cmov binary search over
  cumulative boundaries (the branchy `upper_bound` form mispredicts ~50% and is
  ~5× slower), address in registers, one load, OpenMP across indices. Native
  unsorted branch only.
- **Measured.** 2.65–3.02× single-threaded, with thread scaling on top.

### Stage 3 — Auto-calibrated prefetch, cache mgmt, CLI — **PR #29**
- **Problem.** Prefetch look-ahead was a compiled constant
  (`DEFAULT_PREFETCH_DISTANCE = 8`), unreachable from Python.
- **Change.** All 12 extern-C wrappers + 4 Cython entries take `prefetch_distance`
  (`0` disables — new capability; `-1` = compiled default).
  `set_prefetch_distance("auto")` classifies the regime (source-vs-LLC size,
  sortedness) against a fingerprint-cached calibrated table. Calibration uses
  interleaved rounds + a half-vs-half stability warning. Adds cache-clearing API
  and the registry-driven `colstore` CLI.
- **Key finding.** On quiet Zen 3, software prefetch is a *pessimization* for
  unsorted gathers (the OoO engine already saturates MLP). Calibrated table:
  `{resident_unsorted: 0, resident_sorted: 0, dram_unsorted: 0, dram_sorted: 128}`.
  The fixed default of 8 silently cost **20–29%** in three of four regimes.

### Stage 4 — Vectored streaming writer — **PR #30**
- **Problem.** Per record: header + one flushing `tofile()` per column + padding
  = C+2 syscalls. Decomposition: 83% of small-record `write()` time.
- **Change.** Assemble each record as an iovec, emit one `writev()` on the raw fd
  (handling partial writes + `IOV_MAX`, copying only strided views). Non-`writev`
  platforms keep the old path; outputs byte-identical (whole-file-hash asserted).
- **Measured.** 2.50× (50-row × 4-col), 2.36× (200-row), → ~1.0× at the ~2.7 GB/s
  page-cache ceiling; no regression to 10M-row records.

### Stage 5 — Bin-reuse multi-column gather — **PR #31**
- **Problem.** A C-column unsorted read recomputed the identical record binning C
  times. Binning is **87–93%** of kernel cost on the EPYC.
- **Design (owned by measurement).** A fully fused C-column kernel wins serially
  (2.5–4.5×) but **collapses at 8 threads** (1.35–1.57× at C=8) — C load+store
  streams saturate miss handling. Bin-reuse holds 1.90–2.51× at 8 threads.
- **Change.** `gather_multirecord_bins` (kernel + `int32` bins) and `_withbins`
  (sequential bins read, no search). Reader routes {multi-record, ≥2 native
  columns, unsorted} through the pair at full cap.
- **Measured.** 1.74× (R=1000, C=4) → 2.68× (R=10000, C=8) at K=1M; cuts CPU
  ~C-fold. *(A parallel-regime mis-application was later fixed — see
  [follow-ons](#follow-on-optimizations-post-series), PR #69.)*

### Post-Stage-5 review
Four fix PRs (**#32–#35**, see [Correctness & robustness fixes](#correctness--robustness-fixes))
plus one rejection: a **small-K threshold for bin-reuse** — bin-reuse wins at
every K from 4 to 131072 (1.44–1.62×) because it removes per-column dispatch,
which dominates at small K; a threshold would add a regression.

### Stage 6 — Native sorted multi-record fancy gather — **PR #36**
- **Problem.** The sorted path ran O(R log K) partition + a serial per-record
  Python loop building a K-sized `byte_offsets` array. The Python machinery
  *grows with record count* — 13% at R=100 up to 97% at R=10⁵.
- **Change.** `gather_multirecord_sorted`: per-thread binary search of the
  chunk's first record, then a monotone cursor — O(K+R). Per-record state hoisted
  into the boundary-crossing branch; steady-state loop is one compare, one
  multiply-add, one alignment-safe load. The partition, loop, and offsets array
  vanish, and the walk threads. *(Dev note: a per-element-base first cut measured
  0.85× at low R, caught by the stage benchmark; the hoisted form has no
  regression.)*

  | layout               | pipeline | walk kernel | speedup |
  | -------------------- | -------- | ----------- | ------- |
  | R=100, 200K rows/rec | 11.4 ms  | 5.8 ms      | 1.96×   |
  | R=1000, 20K rows/rec | 9.3 ms   | 5.0 ms      | 1.85×   |
  | R=10⁴, 2K rows/rec   | 46.4 ms  | 4.7 ms      | 9.82×   |
  | R=10⁵, 200 rows/rec  | 350.1 ms | 6.5 ms      | 53.49×  |

  Walk time is nearly flat (4.7–6.5 ms) across a 1000× record-count range — the
  O(K+R) signature: load-bound, not bookkeeping-bound, at every R.

*Post-Stage-6 fixes:* **#37** (non-contiguous index arrays) and **#38** (NUMA
local-policy test) — see [Correctness & robustness fixes](#correctness--robustness-fixes).

---

## Round 2 — selector-side overheads

Round 2 (Stages 7–12, **PRs #39–#44**) removed the costs *upstream* of the
now-native kernels — index materialization, sortedness checks, record binning
where the layout permits — and added a zero-copy API.

### Stage 7 — Native strided multi-record range reads — **PR #39**
- **Problem.** `step≠1` slices materialized a full int64 `arange`, paid the O(K)
  sortedness check, and hit the log-R search kernel for negative steps.
- **Change.** `gather_multirecord_strided`: the row stream is synthesized
  arithmetically (no index array), cursor advances monotonically in the step's
  direction, O(K+R) both ways. One kernel serves both directions (the unused
  boundary test is a perfectly predicted not-taken branch — confirmed free, which
  rejected a direction-templated pair).

  | step | K        | speedup    |
  |------|----------|------------|
  | 2    | 10⁷      | 4.32–4.75× |
  | 10   | 2×10⁶    | 1.30–2.96× |
  | 100  | 2×10⁵    | 1.04–1.24× |
  | −1   | 2×10⁷    | 5.61–6.91× |
  | −3   | 6.7×10⁶  | 5.54–7.11× |

  *Benchmark note:* warm each slice with a read of that same slice before timing
  — pages faulted by the other side skew small-K results.

### Stage 8 — Sampled-rejection sortedness detection — **PR #40**
- **Problem.** Routing gates ran the full O(K) non-decreasing check (+ a K−1-byte
  temp) on every fancy selector — pure overhead for unsorted, and serial.
- **Change.** `_indices_are_sorted` probes 16 evenly spaced pairs first. Sampling
  only ever *rejects*: a descent skips the full pass; an ascending sample falls
  through to the exact full pass (correctness unconditional). Skipped below 32768
  elements (full pass beats the ~15 µs sampler; crossover ~23K).
- **Measured.** Check-level 7.8× / 35× / 683× at K = 2×10⁵ / 10⁶ / 10⁷.
  End-to-end 1.05–1.06× at K=10⁷; compounds with every future kernel speedup
  (the removed cost was serial).

### Stage 9 — Uniform-record fast path — **PR #41**
- **Problem.** On uniform-row-count files the search (87–93% of kernel cost) is
  closed-form: `r = idx / rows_per_record`.
- **Change.** `gather_multirecord_uniform` (single column — one division + affine
  address) + a `_uniform_bins`/`_uniform_withbins` pair (first column divides +
  records the int32 bin; later columns read the bin).
  `_detect_uniform_record_layout` (O(R), cached) verifies layout numerically.
- **Rejected.** Per-column division without bins *loses* at R=10³, 4-col (0.85×):
  division × C exceeds one shallow search + C−1 bins reads.

  | R    | rows/rec | read  | K=2×10⁶ | K=10⁷ |
  |------|----------|-------|---------|-------|
  | 10³  | 20000    | 1 col | 1.61×   | 1.44× |
  | 10³  | 20000    | 4 col | 1.12×   | 1.13× |
  | 10⁴  | 2000     | 1 col | 2.48×   | 1.69× |
  | 10⁴  | 2000     | 4 col | 1.32×   | 1.19× |
  | 10⁵  | 200      | 1 col | 5.17×   | 3.03× |
  | 10⁵  | 200      | 4 col | 1.80×   | 1.60× |

  > **These figures predate PR #93.** On the production node the integer division
  > later proved a *net loss* vs the generic route (0.75–0.94×); the reciprocal
  > divide (PR #93) restores the intended win. See
  > [follow-ons](#follow-on-optimizations-post-series).

### Stage 10 — Record-base precompute (irregular multi-column) — **PR #42**
- **Problem.** Trailing irregular columns paid three per-record metadata loads +
  two multiplies per element.
- **Change.** Fold into one per-record scalar
  `record_base[r] = rsb[r] + col_prefix·nrr[r] − rsr[r]·itemsize`, built once per
  column (0.27 ms at R=10⁵). `gather_multirecord_withbins_rbase` address is
  `record_base[bins[i]] + indices[i]·itemsize`. Gated on `n >= n_records`.
- **Measured.** 1.11×/1.19× (R=10³), 1.13×/1.07× (R=10⁴), 1.02×/1.04× (R=10⁵) at
  K=2×10⁶/10⁷; kernel-level 1.20–1.21×.

### Stage 11 — Zero-copy read API (`copy=False`) — **PR #43**
- **Problem.** Materializing whole compacted stores copied every column though
  the bytes already sit contiguous behind `mode="r"` memmaps — doubling peak RAM.
- **Change.** `copy=False` returns read-only views — supported when the store is
  single-record (`colstore.compact`), native dtype, selector None/int/slice;
  everything else raises with the remedy (the flag is a memory guarantee). Views
  pin the mapping via `.base`, surviving `close()`.
- **Measured.** O(1) view creation (~3 µs flat 8 MB→800 MB vs 60 ms to copy
  800 MB); read-and-reduce 1.7–2.3×; **peak resident memory halved.**

### Stage 12 — Boolean-mask-native path — **PR #44**
- **Problem.** Boolean masks lowered to int64 indices immediately
  (`np.flatnonzero`) — O(N) conversion + 8 B/selected-row + per-column sortedness
  + 8 B/element index traffic, all derivable from a 1 B/row mask.
- **Change.** `gather_multirecord_mask` scans the mask directly with a monotone
  walk. Decisive design: **word-at-a-time** (a per-element test lost 0.2–0.6× to
  misprediction at p=0.5) — all-zero words skip, all-ones become a clipped memcpy,
  mixed words compact branchlessly under a per-thread over-store quota; a count
  check guards the output.
- **Routing.** Multi-record native reads above the density gate take the kernel;
  sparse/single-record/non-native lower to `flatnonzero`.

  | selector             | 1 col      | 2 col      |
  |----------------------|------------|------------|
  | density 0.9          | 4.3–5.2×   | 2.7–3.2×   |
  | density 0.5          | 4.2–6.2×   | 2.6–3.0×   |
  | density 0.2          | 3.9–11.6×  | 2.0–2.6×   |
  | density 0.1–0.01     | 0.94–1.07× | 0.97–1.02× |
  | whole-record cut 0.3 | 4.5–8.6×   | 2.9–9.1×   |

  The gate was initially 0.15 (container-derived); **PR #47 later calibrated it
  per host and dropped the default to 0.0** — see below.

*The optimization-series summary doc itself landed as **PR #45**; **PR #46**
refreshed a stale bin-reuse docstring.*

---

## Follow-on optimizations (post-series)

Four optimizations landed after the series summary (PR #45). Two correct routing
decisions that had inverted on the production node once threading became
work-proportional; one extends parallel copy to strided reads; one is the
reciprocal divide.

### Mask-density gate calibration — **PR #47** (Stage 12 follow-on)
- **Finding.** Node calibration measured the mask route winning **1.35–3.2× at
  every density** (halves-stable, picked gate 0.01): `flatnonzero` is serial while
  the mask kernel's passes parallelize, so the container-derived 0.15 gate was
  leaving wins on the table.
- **Change.** `mask-density` becomes a per-host calibration target
  (`colstore calibration run mask-density`), default gate **0.15 → 0.0** (route on
  at every density), resolution precedence explicit > fingerprint-cache > default.
  Calibration also serves the opposite host class — single-core environments
  (0.4–0.6× on sparse masks) get the gate raised or disabled. Retires the
  reader-level constant seam for `config.set_mask_density_gate`.

### Bin-reuse parallel-regime gate — **PR #69** (Stage 5 fix)
- **Regression.** `check_multicolumn_gather` at K=1M showed bin-reuse at **0.32×**
  vs the per-column fallback (expected 1.9–2.5×).
- **Root cause.** Not the kernel (it does ~1.5× *less* CPU work) — memory-level
  parallelism. Bin-reuse runs columns sequentially on one kernel's intra-column
  OpenMP; since `resolve_thread_count` became work-proportional, a single K=1M
  kernel resolves to just **2** of an 8-cap, while the per-column pool fields one
  resolved width *per column* concurrently. The route was tuned before
  work-proportional threading, when a sequential kernel was assumed to claim the
  full cap — true only past the scaling knee.
- **Change.** A gate in `_gather_many_bin_reuse` declines the route (falls through
  to the column pool) only when the concurrent path strictly out-fields the
  sequential one. Serial-regime reads never gate (every existing contract
  untouched); no new concurrency. At K=8M the sequential kernel resolves to the
  full cap and bin-reuse wins again — the route is right *above* the knee.
- **Result.** Recovers the ~3× regression at moderate K while preserving the
  large-K win. *(Absolute wall-time recovery pending on-node A/B.)*

### Parallel strided-slice copy — **PR #71** (reader)
- **Problem.** A large strided `.array()` read was stuck serial (cpu/wall 0.87×)
  while a contiguous read of comparable size parallelized (1.79×) — the strided
  branch did an unconditional single-threaded `np.array(..., copy=True)`.
- **Change.** Generalize `_parallel_contiguous_copy` → `_parallel_copy` (it sizes
  work by *logical* nbytes, correct for a strided view) and route both slice
  branches through it; index with the original slice object (reconstructing from
  `slice.indices()` misreads the `-1` negative-step stop). Thresholds (16 MiB)
  unchanged — only large strided reads gain threads; everything else is
  byte-for-byte identical.

### Uniform reciprocal divide — **PR #93** (Stage 9 fix)
- **The optimization had inverted.** On the node, the uniform route ran
  *slower* than the generic route it was meant to beat — `check_uniform_multirecord`
  **0.87–0.94×**, `run_benchmarks uniform_kernel` **0.75×**, at every size. Since
  the reader auto-selects uniform for uniform layouts, it was defaulting to the
  slower path.
- **Root cause.** `r = idx / rows_per_record` is a runtime 64-bit `div` (~20–40
  cycles, not pipelined; computed twice per iteration in the single-column kernel
  — prefetch + load). The generic route's branch-predicted search over a small
  cache-resident `record_starts` array is faster.
- **Change.** `rows_per_record` is invariant, so it's a textbook
  reciprocal-multiply case. A `UniformDivisor` helper precomputes the magic
  constant once (Granlund–Montgomery, libdivide branchfull form, `__uint128_t` for
  the 128-bit step); each division becomes a 64×64 multiply-high + two shifts
  (~4 cycles). Applied to the single-column and bins-first-column kernels;
  `withbins` already reused `bins[]` and never divided.
- **Verified.** Standalone divider exhaustive test (**124,005,000 checks, 0
  failures**); `objdump` confirms `div/idiv = 0`, `mul = 20/20/16` across the
  three uniform kernels; 711 tests pass.
- **Windows portability.** The fast path is gated `#if defined(__SIZEOF_INT128__)`
  (GCC/Clang — the hot deploy toolchain) with a plain `n / divisor` fallback
  `struct` for MSVC (which has neither `__uint128_t` nor `__builtin_clzll`). The
  fallback is provably correct; Windows pays only a hideable `div`. *(This gate
  is present in the current tree; it was a refinement folded in after PR #93, not
  a separately numbered PR in the provided list.)*
- **Magnitude pending.** The container is single-core, so the multi-thread
  large-K regime where this shows up can't be timed there; the expected flip from
  0.75–0.94× to faster-than-generic **needs an on-node interleaved A/B to confirm
  the magnitude.**

---

## Structural refactor that preserved kernel performance

The policy-migration arc (**PRs #61–#67**: compile-time itemsize dispatcher,
fold the multirecord family onto `gather_core`, stateless policies, make
`gather_core` the sole implementation, collapse wrappers onto `gather_entry`)
was primarily maintainability/compression — it roughly halved the kernel source —
not an optimization. It earns a place for one **performance** result, in
**PR #63**: an earlier **struct-state** policy design was measurably slower —
`restrict` on struct members doesn't survive GCC inlining, and the OpenMP
outliner captures the aggregate behind a context pointer, costing a register. On
`withbins` that meant a per-iteration reload + stack spill, measured **+5.1%**
(128 MB, max threads) / +0.9% (256 MB, 8 threads). The **stateless
parameter-pack** form restored hand-written-identical codegen (verified by
disassembly; reference loop-body instruction counts: indexed 61, bytes 62,
multirecord 128, bins 135, withbins 91, rbase 74). *Do not simplify policies back
into structs.* The public C ABI is unchanged, so every stage above maps onto
current code.

---

## Net effect

The read side is native end-to-end and selector-native, on a threading and
placement foundation that scales. Threading is work-proportional and capped where
bandwidth saturates; the C++ kernel is unconditional (beats `np.take` even
single-threaded); `ds.dict()` runs near bandwidth (~11.8×) and `ds.frame()`
within ~5× of it (per-column BlockManager); multi-socket reads interleave the
page cache (CPU 1.6–2.2×). On top of that: contiguous ranges memcpy (5–23×);
strided slices run an index-free walk (4.3× fwd, 5.5–7.1× rev) and parallelize
when large; unsorted fancy reads run the fused branchless kernel with binning
once per multi-column read, arithmetically (magic-reciprocal) on uniform files
and with precomputed bases on irregular ones; sorted fancy reads run the
load-bound linear walk (1.9–53× with R); boolean masks bypass index
materialization entirely with a per-host-calibrated gate (4–6× single-column, up
to 9.1× on whole-record cuts); the sortedness gate costs microseconds; compacted
stores read zero-copy (half the peak memory). All kernels thread above 256K
elements, load misaligned packed columns safely, honor per-host prefetch
calibration, run `div`-free on the deploy toolchain, and sit behind gates whose
floor is the pre-change behavior. Event-sized record streams write ~2.4× faster.

---

## Rejected alternatives

*Do not re-propose without new evidence.*

| Idea | Why rejected | From |
|------|--------------|------|
| `np.take` delegation below a size threshold | C++ kernel beats it 1.5–2× (up to 4×) even single-threaded; numpy re-validates indices | PR #14 |
| `pd.DataFrame._from_arrays(verify_integrity=False)` for `frame()` | Skips validation but still consolidates (~220–700 ms, no better than the default ctor) | PR #24 |
| NUMA `"local"` as default | ~4% slower than interleave only at `workers=1`; the wide-read win dominates | PR #25 |
| Fused C-column gather kernel | Wins serially (2.5–4.5×) but collapses at 8 threads (1.35–1.57× at C=8); per-thread streams saturate miss handling | Stage 5 |
| Small-K threshold for bin-reuse | Bin-reuse wins at every K from 4–131072 (1.44–1.62×); a threshold adds a regression | Stage 5 review |
| Fixed / unsorted-gather prefetch on Zen 3 | Pessimization; the fixed d8 cost 20–29% | Stage 3 |
| `posix_fallocate` | Measured wash; glibc-emulation hazard on filesystems without native support (older Lustre) | Stage 4 |
| SIMD gather | Bottleneck is outstanding-miss capacity, not throughput; `vpgather` is microcoded on Zen 3. Revisit on other hardware only | Deferred |
| Direction-templated strided pair | The single kernel's unused boundary branch is free (1.19 vs 1.20 ms) | Stage 7 |
| Unconditional sortedness sampling | Below ~23K elements the full pass beats the sampler's fixed overhead | Stage 8 |
| Per-column arithmetic binning without bins | Loses to the generic bins route at R=10³, multi-column (0.85×) | Stage 9 |
| Per-element integer divide in the uniform kernel | 0.75–0.94× vs generic on the node — a non-pipelined `div`; reciprocal multiply replaces it | PR #93 |
| Per-reader `record_base` cache | Per-call build is 0.27 ms vs 60–300 ms kernels; cache buys noise, costs resident state | Stage 10 |
| Run-coalesced sorted gather (materialized indices) | Bandwidth ceiling ~1.0× dense, 0.2–0.9× measured; headroom moved to mask-native | Round 2 |
| Per-element mask test | 0.2–0.6× from misprediction at mid densities; word-at-a-time is shipped | Stage 12 |
| Views invalidated at reader close | Converts a refcounting non-issue into a segfault class on a read-only mapping | Stage 11 |
| Silent copy fallback for `copy=False` | Voids the memory guarantee; unsupported cases raise with the remedy | Stage 11 |
| MSVC reciprocal intrinsics | Silent-corruption risk off the hot deploy target without the GCC/Clang correctness sweep; hideable `div` fallback is provably correct | PR #93 |
| Struct-state gather policies | +5.1% (128 MB, max threads) from lost member-`restrict` + outliner register spill | PR #63 |

---

## Open / deferred items

Priority order; **measure-first** items must not be changed on dev-container
intuition.

| # | Item | Status / trigger |
|---|------|------------------|
| 1 | **`ds.frame()` residual headroom** beyond the per-column BlockManager (PR #24) | **Top target, measure-first.** Per-column BlockManager is *done* and `_from_arrays` is ruled out; remaining ladder is direct memmap-to-BlockManager → Arrow interchange |
| 2 | Multirecord search kernels (`multirecord`/`bins` ~34 ms vs `withbins` ~19 ms) | **Measure-first.** Flamegraph at production size; optimize only if cost is in `bin_record`, else it's at the memory ceiling |
| 3 | rbase routing-threshold A/B (single-column large-K through `MultiRecordPolicy`) | **Measure-first** on EPYC; `check_record_base` harness isolates it |
| 4 | Confirm the PR #93 reciprocal-divide magnitude on-node | Run the interleaved paired A/B the container couldn't (single-core); verify the flip past 0.75–0.94× |
| 5 | Confirm the PR #69 bin-reuse-gate wall-time recovery on-node | Same; the gate verdict is proven, the absolute recovery is not |
| 6 | Per-thread-count prefetch calibration axis | Close with one `sweep_prefetch_distance.py --threads 8` on a quiet node |
| 7 | int32 index kernels | Halves index bandwidth when total rows < 2³¹; premise-check the index-traffic share first (mask-native already removes it where it dominates) |
| 8 | Cold-read (page-fault-bound) paths | Mostly unprofiled (numbers are warm); `drop_pagecache` exists. Profile only if real workloads show page-fault time on the critical path |
| 9 | Multi-record `writev` batching | Remaining tiny-record gap is per-call Python cost; complicates durability semantics |
| 10 | Uniform-trio fold onto `gather_core` | Gated per-kernel on paired A/B (instruction-bound, register-pressure risk); must preserve the `__SIZEOF_INT128__` reciprocal split |
| 11 | Concurrent-withbins shape (running withbins columns *on* the pool) | Considered in PR #69, deferred — adds real concurrency, needs node-side A/B and touches the nested-pool oversubscription hazard |
| 12 | Native record-index scan | Open-latency for very-high-record-count files; only if open time appears in profiles |
| 13 | Format-level column alignment | Future format-version decision, not a retrofit (alignment-safe loads already cover correctness) |
| 14 | Single-record mask-native route | Kept on flatnonzero+fancy to preserve `backend` contract; numpy's boolean indexing leaves little headroom |
| 15 | NUMA follow-ups | `auto` vs `local` for single-threaded readers; writer-side interleave on real Lustre paths |
