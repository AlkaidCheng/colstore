# colstore Performance Optimization Series: Reference Summary

This document summarizes two rounds of performance engineering on
`colstore`, a memory-mapped columnar binary format (C++ gather kernels,
Cython bindings, Python reader/writer). Each stage was delivered as an
independently revertable change set, premise-verified by profiling or a
standalone prototype before implementation, and validated on the deployment
hardware (multi-socket, multi-NUMA-node CPU compute nodes) before merge.
Output equivalence with the replaced code path — byte-identity where
applicable — is asserted by tests in every stage, not assumed; every stage
ships a `benchmark/check_*.py` script with a correctness gate that runs
before timing and a toggle that reproduces the replaced route verbatim.
Rejected design alternatives are recorded with the measurements that
rejected them, so they are not re-proposed without new evidence.

Round 1 (Stages 1–6 plus an external correctness review) made the read side
native end-to-end and fixed the writer's syscall pattern. Round 2 (Stages
7–12) removed the remaining selector-side overheads — index materialization,
sortedness checks, record binning where the layout permits — added a
zero-copy read API, and rejected one proposed optimization on measured
grounds. The test suite grew from 311 tests before Round 1 to 407 after it
and 600 after Round 2, passing in both single-thread and
`OMP_NUM_THREADS=8` regimes at every stage (the deployment node collects a
handful of additional host-conditional tests).

---

# Round 1 — Stages 1–6

## Stage 1 — Native contiguous multi-record range copy

**Branch:** `perf/native-multirecord-range-copy`

**Problem.** Contiguous range reads spanning multiple records went through a
generic per-record Python loop, paying interpreter and dispatch overhead per
record.

**Change.** A C++ `copy_multirecord_range` kernel: binary-search the starting
record, then one `memcpy` per overlapping record. Deliberately serial —
`memcpy` is already memory-bandwidth-bound, so threading adds overhead
without throughput. A Python fallback remains for builds without the
extension.

**Measured (deployment hardware).** 5–23x on many-small-record layouts,
converging to ~1x on few-large-record layouts: the gain lives where
per-record overhead dominated, and there is no regression where it did not.

---

## Threshold fix — work-proportional gather thread count

**Branch:** `fix/resolve-thread-count-floor`

**Problem.** Found during Stage 1 validation: an integer-rounding bug in
`resolve_thread_count` floored every gather below 1M elements to a single
thread, dead-coding the intended 256K-element parallel threshold.

**Change.** Round-up division plus a minimum of 2 threads past the threshold;
behavior below 256K elements is unchanged.

**Measured.** Once unclamped: 2.0x at 2 threads (300K–1M elements), 5.0x at
8 threads (5M), 17.9x at 32 threads (20M). The scaling knee sits at roughly
one thread per 1M elements, confirming the existing `ELEMENTS_PER_THREAD`
constant.

---

## Stage 2 — Fused branchless multi-record fancy gather

**Branch:** `perf/native-fancy-gather`

**Problem and overturned premise.** Unsorted fancy-index reads of
multi-record stores ran a NumPy pipeline: `np.searchsorted` to bin each
index to its record, materialized byte offsets, then a raw gather kernel.
The original optimization proposal blamed the byte-offset materialization;
isolation profiling overturned this — the `searchsorted` binning was 73–90%
of the path, the offset temporaries only 7–11%. Eliminating offsets alone
would have yielded ~1.1x.

**Change.** `gather_multirecord_typed<T>`: per index, a **branchless** binary
search over the cumulative record boundaries (conditional-move based; the
branchy `std::upper_bound` form mispredicts ~50% of comparisons and measured
~5x slower than the branchless form), byte address computed in registers,
one load — no `searchsorted`, no K-sized temporaries, OpenMP-parallel across
indices (which `searchsorted` can never be). Scope is the native-byte-order
unsorted branch only; the sorted path (already 4–11x faster via boundary
partitioning) and non-native hosts are untouched.

**Measured.** 2.65–3.02x single-threaded on the deployment hardware, with
the threshold fix's thread scaling applying on top.

---

## Stage 3 — Auto-calibrated prefetch distance, cache management, CLI

**Branch:** `perf/auto-prefetch-distance` (four commits)

**Problem.** The gather kernels' software-prefetch look-ahead was a compiled
constant (`DEFAULT_PREFETCH_DISTANCE = 8`) unreachable from Python: neither
tunable per host nor disableable. The optimal value is both hardware- and
load-dependent.

**Commit 1 — runtime-configurable distance with `"auto"` default.** All 12
extern-C wrappers and four Cython entry points take `prefetch_distance`
(`> 0` look-ahead; `0` disables — a new capability; `-1` resolves to the
compiled default). `config.set_prefetch_distance("auto")` (the default)
resolves the distance per call by classifying the access regime from two
cheap signals — source size vs last-level-cache size, and index sortedness —
against a per-regime table measured by `autotune.calibrate_prefetch()` and
cached under a hardware fingerprint. Uncalibrated `"auto"` falls back to the
compiled default, so out-of-the-box behavior is unchanged.

**Calibration methodology.** Initial calibration runs on a shared login node
were unstable. Diagnosis: sequential per-candidate repeats let background
load drift penalize individual candidates. The sweep was rebuilt as
interleaved rounds (every candidate timed once per round) with the median
over rounds as the statistic and a half-vs-half pick-disagreement warning as
a built-in stability diagnostic. On a quiet compute node, all four regimes
calibrated halves-stable.

**Key finding.** On the target CPU, software prefetch is a
*pessimization* for unsorted gathers — the out-of-order engine already
saturates memory-level parallelism, and prefetch instructions are pure
overhead. The calibrated table is `{resident_unsorted: 0, resident_sorted:
0, dram_unsorted: 0, dram_sorted: 128}`; the previous fixed default of 8 had
been silently costing 20–29% in three of four regimes. A loaded login node
prefers d32–128 instead: the optimum is load-dependent, which is why it is
measured where jobs run rather than guessed at compile time.

**Commits 2–4 — operational surface.** Idempotent cache-clearing API
(`clear_cached_cap` / `clear_cached_prefetch` / `clear_calibration`, with
in-process reset semantics); harmonized calibration machinery (the thread-cap
calibration migrated to the same interleaved-median methodology, uniform
`rounds=10` on both targets); and an extensible `colstore` console-script
CLI. The CLI is registry-driven: calibration targets are declarative entries
from which `calibration run|show|clear` (including per-target
`--<name>-rounds` options) all render, and command groups own noun
namespaces so the top level stays unambiguous as features grow.
`calibration show` reports whether each cache's hardware fingerprint matches
the current machine — closing a real footgun on clusters where `$HOME` is
shared between login and compute nodes and a stale calibration would
otherwise be silently applied.

---

## Stage 4 — Vectored streaming writer

**Branch:** `perf/vectored-writer`

**Problem.** Per record, the streaming writer issued one buffered write per
piece — record header, one `tofile()` per column (each forcing a flush of
the buffered layer), padding — i.e. C+2 syscalls plus interleaved flushes for
a C-column record. Decomposition attributed 83% of small-record `write()`
time to this machinery (13% to validation, which is API semantics and
untouched).

**Change.** Each record is assembled as an iovec (32-byte header bytes, one
entry per column buffer, alignment padding) and emitted with a single
`writev()` on the raw fd. The implementation handles the two liberties POSIX
reserves for `writev` — partial writes (byte-granularity views, resumed
mid-buffer) and `IOV_MAX` segment limits (chunked across calls) — and copies
only strided column views (the same movement `tofile` performed internally).
Platforms without `os.writev` keep the previous path; the two paths produce
byte-identical files, asserted by whole-file hashing across zero-row
records, padding cases, strided inputs, and update-mode appends.

**Measured (deployment hardware).** 2.50x for 50-row x 4-column records,
2.36x at 200 rows, converging to ~1.0x at the host's ~2.7 GB/s page-cache
ceiling, with no regression at any record size up to single 10M-row records.

**Deferred with data.** `posix_fallocate` preallocation for the one-shot
write path measured a wash (buffered writes do not block on block
allocation), and glibc emulates it by touching every block on filesystems
without native fallocate support — including older Lustre — which would
double the write cost precisely where the library deploys.

---

## Stage 5 — Bin-reuse multi-column gather

**Branch:** `perf/multicolumn-bin-reuse`

**Problem.** A C-column unsorted fancy read ran the Stage-2 fused kernel
independently per column, recomputing the per-index record binning — which
is identical for every column of the read — C times. Profiling on the
deployment hardware showed the binning is **87–93%** of the kernel's cost
(bracketed by two load-only proxies that agree tightly): the CPU's
memory-level parallelism pipelines the independent loads to ~3.5 ns each,
while the log2(R)-step dependent-chain search cannot overlap as well.

**Design decision, owned by hardware measurements.** Two designs were
prototyped standalone and measured on a compute node. A fully fused
C-column kernel (bin once in-register, load all columns per iteration) wins
serially (2.5–4.5x) but collapses at realistic thread counts — its C
concurrent load+store streams per thread saturate the memory subsystem's
miss-handling capacity (1.35–1.57x at C=8 with 8 threads). Bin-reuse (bin
once into an `int32` array, then one bins-provided pass per column) holds
1.90–2.51x at 8 threads, growing with column and record count. Bin-reuse was
shipped; the fused design's rejection is recorded with its numbers, since
serial and container measurements both pointed the other way.

**Change.** A kernel pair — `gather_multirecord_bins` (the Stage-2 kernel
plus an `int32` bins output) and `gather_multirecord_withbins` (per-element
record is a sequential bins read instead of a search; the prefetch
look-ahead likewise reads `bins[i+d]`, retiring the Stage-2 note about the
look-ahead's duplicate search for reused columns). The reader's
`_gather_many` routes `{multi-record, ≥2 native columns, unsorted fancy
index}` reads through the pair, each call at the full thread cap with OpenMP
over indices, amortizing the sortedness check and prefetch resolution once
per read. Sorted selectors, single columns, slices, single-record stores,
and non-native columns (which fall back per column) are untouched.

**Measured.** End-to-end through the public reader API: 1.74x (R=1000, C=4)
to 2.68x (R=10000, C=8) at K=1M. Beyond wall time, the route cuts total CPU
work roughly C-fold on the read's dominant cost, which matters for
allocation accounting independent of latency.

---

## Post-series correctness review (four fix PRs)

An external review of Stages 1–5 was verified claim-by-claim before any fix
was written; one of its proposals was rejected by measurement.

* **Alignment-safe kernel loads** (`fix/alignment-safe-loads`) — the
  substantive finding. Record bodies are packed with no inter-column
  padding, so a column's start is naturally aligned only if every preceding
  column's byte count is a multiple of its alignment; an odd-length `int8`
  column followed by a `float64` column was confirmed to place 8-byte loads
  at odd addresses. The kernels' typed-pointer dereferences at such
  addresses were undefined behavior (tolerated by x86, breakable by a future
  compiler or sanitizer build). All five source-loading kernel families now
  load through a shared fixed-size-`memcpy` helper (`load_unaligned`),
  verified in both directions under `-fsanitize=alignment` (previous code
  traps, fixed code clean) at performance parity.
* **Byte-order-correct fancy reads** (`fix/nonnative-byteorder-fancy-reads`)
  — a pre-existing bug that predates the series: byte-offset gathers
  delivered raw little-endian bytes into native-typed outputs, which would
  be garbage on a big-endian host. Destinations are now disk-typed with a
  no-op-on-LE conversion at the return.
* **Backend semantics** (`docs/backend-semantics-multirecord`) — documented
  rather than changed: multi-record fancy reads have never consulted the
  reader's `backend` parameter (true since multi-record support landed);
  the contract is now stated and pinned by a test.
* **writev zero-progress guard** (`fix/writev-zero-progress-guard`) — a
  zero return with non-empty buffers outstanding now raises instead of
  spinning; the guard distinguishes genuine zero progress from
  exact-boundary consumption, which a naive check misfires on.
* **Rejected: a small-K threshold for the bin-reuse route.** Measured K
  from 4 to 131072: bin-reuse wins at every size (1.44–1.62x, including
  1.44x at K=4), because the route adds no native calls while removing
  per-column checks and dispatch — exactly what dominates at small K. The
  proposed threshold would have introduced a regression.

---

## Stage 6 — Native sorted multi-record fancy gather

**Branch:** `perf/native-sorted-multirecord-gather`

**Problem.** The sorted multi-record fancy path ran a NumPy pipeline: an
O(R log K) boundary partition, a serial per-record Python loop
materializing a K-sized `byte_offsets` array, then the raw byte-offset
kernel. Decomposition shows the Python-side machinery *grows with the
record count* — 13% of the path at R=100, 34% at R=1000, 79% at R=10⁴, 97%
at R=10⁵. Event-organized physics files live at exactly the high-R end.

**Change.** `gather_multirecord_sorted`: each thread binary-searches the
record of the first index in its chunk, then advances the record cursor
monotonically — O(K + R) total versus O(K log R) for the unsorted kernel.
Per-record state (next boundary, record base offset) is hoisted into the
boundary-crossing branch, so the steady-state inner loop is one compare,
one multiply-add, and one alignment-safe load — strictly less per-element
work than the byte-offset kernel it replaces, which reads each precomputed
offset from memory. The partition, the Python loop, and the `byte_offsets`
array all disappear, and the walk threads (the Python loop could not).
Prefetch is issued only for look-aheads within the current record — the
only case computable without walking ahead and the dominant one for dense
sorted reads; `dram_sorted` is the single regime where host calibration
enables prefetch, and the gate keeps it correct there. The
boundary-partition pipeline survives unchanged as the non-native
(big-endian host) fallback and as the equivalence reference in tests.

A development note recorded with the commit: the first cut recomputed the
record base per element and measured 0.85x at low R — a regression caught
by the stage's own benchmark before delivery; the hoisted form shows no
regression in any regime.

**Measured (deployment hardware, K=1M, f8).**

| layout               | pipeline | walk kernel | speedup |
| -------------------- | -------- | ----------- | ------- |
| R=100, 200K rows/rec | 11.4 ms  | 5.8 ms      | 1.96x   |
| R=1000, 20K rows/rec | 9.3 ms   | 5.0 ms      | 1.85x   |
| R=10⁴, 2K rows/rec   | 46.4 ms  | 4.7 ms      | 9.82x   |
| R=10⁵, 200 rows/rec  | 350.1 ms | 6.5 ms      | 53.49x  |

Walk-kernel time is nearly flat (4.7–6.5 ms) across a 1000x range in record
count — the O(K + R) signature: the sorted read is load-bound rather than
bookkeeping-bound at every R.

---

---

## Post-Stage-6 correctness fixes

Two further fixes landed after Stage 6, before Round 2:

* **Non-contiguous index arrays** (`fix/noncontiguous-index-arrays`) — a
  strided fancy selector (e.g. ``rows[::2]`` or ``rows[::-1]``) reached the
  C++ kernels, which interpret the array as a contiguous int64 pointer:
  wrong values for positive strides, out-of-bounds reads (a segfault on
  ``[::-1]``) for negative ones. Fixed at three layers: the view layer
  normalizes fancy selectors with ``np.ascontiguousarray(..., int64)``, and
  every Cython entry validates ALL pointer-interpreted arrays C-contiguous
  via a shared ``_require_c_contiguous`` helper.
* **NUMA local-policy test** (`fix/numa-local-policy-test`) — test-only: a
  spy installed before ``store()``'s implicit open contaminated the
  assertion on NUMA hosts. The NUMA module itself
  (``MPOL_INTERLEAVE`` on file memmaps at open, ``config.set_numa_policy``,
  measured 1.79x on a 1 GB whole-store read on the node) is active on the
  deployment hardware.

---

# Round 2 — Stages 7–12

Round 2 targets what Round 1 left: the selector-handling costs upstream of
the (now native) kernels, and layout-conditional shortcuts the generic
kernels cannot take. The standing conventions carry over unchanged; one
addition is that every routing change ships behind a measured gate whose
constant doubles as the benchmark's baseline seam, so the shipped floor is
the pre-change behavior.

---

## Stage 7 — Native strided multi-record range reads

**Branch:** `perf/native-strided-multirecord-range`

**Problem.** Multi-record slices with ``step != 1`` materialized a full
int64 ``np.arange`` selector, paid the O(K) sortedness check, and went
through the fancy machinery: the sorted walk kernel for positive steps, the
per-element binary-search kernel for negative steps (a descending stream
fails the sortedness check), whose binning deepens with log R.

**Change.** ``gather_multirecord_strided``: the row stream is synthesized
arithmetically inside the kernel — no index array exists at any point — and
the record cursor advances monotonically in the step's direction: O(K + R)
both ways, offsets in registers, OpenMP across output chunks,
same-record-gated prefetch with the look-ahead row formed arithmetically
instead of loaded. One kernel serves both step directions (for a fixed step
sign, the unused boundary test is a perfectly predicted not-taken branch —
confirmed free by kernel-level parity with the sorted kernel on prebuilt
indices, 1.19 vs 1.20 ms, which also rejected a direction-templated kernel
pair). Non-native dtypes keep the arange fallback; unit-step slices keep
the per-record memcpy route; single-record stores are untouched.

**Measured (deployment hardware, full-span slices, 160 MB f8 column,
R = 10³–10⁵).**

| step | K        | speedup    |
|------|----------|------------|
| 2    | 10⁷      | 4.32–4.75x |
| 10   | 2×10⁶    | 1.30–2.96x |
| 100  | 2×10⁵    | 1.04–1.24x |
| −1   | 2×10⁷    | 5.61–6.91x |
| −3   | 6.7×10⁶  | 5.54–7.11x |

The negative-step win grows with R as the replaced path's per-element
binary search deepens with log R while the walk stays O(K + R). At small K
the win reduces to the removed arange + sortedness check. Benchmark note,
recorded as timeless guidance: warm each slice with a read of that same
slice before timing either side — timing one side against pages faulted by
the other skews small-K results.

---

## Stage 8 — Sampled-rejection sortedness detection

**Branch:** `perf/sampled-sortedness-rejection`

**Problem.** Both fancy-routing gates (per-column dispatch and the
multi-column bin-reuse gate) ran the full O(K) non-decreasing check — which
also allocates a K−1-byte comparison temporary — on every fancy selector.
For unsorted selectors the pass is pure overhead, and it is serial, so its
share of a read grows with the gather kernels' thread count.

**Change.** ``_indices_are_sorted`` probes 16 evenly spaced adjacent pairs
first. Sampling only ever **rejects** sortedness: any sampled descent
proves the selector unsorted and skips the full pass; an all-ascending
sample falls through to the full pass, so sorted selectors keep their exact
proof and correctness is unconditional (pinned by a test whose only descent
sits between probe positions). A random unsorted selector skips the full
pass with probability 1 − 2⁻¹⁶. Sampling is skipped below 32768 elements,
where the full pass is cheaper than the sampler's ~15 µs fixed overhead
(measured crossover ~23K elements); unconditional sampling was rejected on
that data. Routing behavior is exactly preserved, including the n == 1
fused-kernel route.

**Measured (deployment hardware).** Check-level on random selectors: 46 µs
/ 208 µs / 4.0 ms at K = 2×10⁵ / 10⁶ / 10⁷ down to ~6 µs (7.8x / 35x /
683x). End-to-end unsorted reads gain 1.05–1.06x at K=10⁷ (the 4.0 ms
serial check against 93–146 ms threaded reads) and show no measurable
change at K=2×10⁶. The win equals the check's serial share and compounds
with every future kernel speedup, since the removed cost was serial.

---

## Stage 9 — Uniform-record fast path

**Branch:** `perf/uniform-record-fast-path`

**Problem.** Record binning is 87–93% of the fused gather kernel's cost on
the deployment hardware (Stage 5 measurement), and on files whose records
all have the same row count the search it performs is computable in closed
form: ``r = idx / rows_per_record``, exact for every index when all records
but the last have the same row count and the last is no larger.

**Change.** Three kernels: ``gather_multirecord_uniform`` (single column —
one integer division and one affine address formula per element; the
partial final record collapses to a single guarded base), and a
``gather_multirecord_uniform_bins`` / ``_uniform_withbins`` pair mirroring
the Stage-5 bins route for multi-column reads — the first column divides
once per element and records the int32 bin; subsequent columns read the bin
(cheaper than re-dividing) and use the affine formula, touching no
per-record metadata. Detection (``_detect_uniform_record_layout``, O(R),
cached per reader) verifies the row counts and the constant body stride
numerically rather than deriving them from format internals; irregular
files keep the generic kernels, and the sorted, strided, and contiguous
routes are untouched (not search-dominated).

**Rejected with data.** Per-column division without bins: at R=10³ it
*loses* to the generic bins route on 4-column reads (0.85x) — division × C
exceeds one shallow search plus C−1 sequential bins reads. The shipped
bins pair dominates both routes at every measured R.

**Measured (deployment hardware, random unsorted selectors, f8).**

| R    | rows/rec | read  | K=2×10⁶ | K=10⁷ |
|------|----------|-------|---------|-------|
| 10³  | 20000    | 1 col | 1.61x   | 1.44x |
| 10³  | 20000    | 4 col | 1.12x   | 1.13x |
| 10⁴  | 2000     | 1 col | 2.48x   | 1.69x |
| 10⁴  | 2000     | 4 col | 1.32x   | 1.19x |
| 10⁵  | 200      | 1 col | 5.17x   | 3.03x |
| 10⁵  | 200      | 4 col | 1.80x   | 1.60x |

The win grows with R (the replaced search deepens with log R while the
division is R-independent) and shrinks somewhat at K=10⁷, where the
wider-threaded kernels are more miss-bound and binning is a smaller share.

---

## Stage 10 — Record-base precompute for the irregular multi-column route

**Branch:** `perf/record-base-precompute`

**Problem.** On irregular files, after the first column fills the record
bins, the generic withbins kernel still pays three per-record metadata
loads (``rsb[r]``, ``nrr[r]``, ``rsr[r]``) plus two multiplies per element,
for every trailing column.

**Change.** All of it is a function of r alone, so it folds into one
per-record scalar, ``record_base[r] = rsb[r] + col_prefix·nrr[r] −
rsr[r]·itemsize``, built once per column as an O(R) vectorized pass
(0.27 ms at R=10⁵ against 60–300 ms kernels). The new
``gather_multirecord_withbins_rbase`` kernel's per-element address is
``record_base[bins[i]] + indices[i]·itemsize`` — one metadata load, one
multiply-add. Gated on ``n >= n_records`` to amortize the build; the first
column is untouched (its cost is the search, which is Stage 9's territory
where the layout permits). A per-reader-per-column cache was rejected: the
per-call build is noise-level, and caching adds R×8 bytes of resident
state per column for no measurable return.

**Measured (deployment hardware, 4-column unsorted reads, irregular
layouts).** 1.11x / 1.19x (R=10³), 1.13x / 1.07x (R=10⁴), 1.02x / 1.04x
(R=10⁵) at K = 2×10⁶ / 10⁷. The win is largest at small-to-mid R: at scale,
the untouched first column's deepening search claims a growing share and
the trailing columns become more outstanding-miss-bound, where out-of-order
execution already overlapped the removed metadata loads. Kernel-level, the
withbins → rbase change is 1.20–1.21x; the gate boundary at K = R shows no
regression.

---

## Rejected — Run-coalesced sorted gather

**Prototype branch (dropped):** `perf/run-coalesced-sorted-gather`

**Hypothesis.** Mask/cut-derived sorted selectors (rich in consecutive-
index runs) could be served faster by memcpy-ing runs than by the
per-element sorted walk.

**Rejection, with the mechanism.** A correctness-complete prototype kernel
lost at every tested density on the dev container: 0.90x at run fraction
0.99, 0.81x on whole-record cuts, 0.22–0.38x at mid/low densities. The
ceiling is structural, not an implementation artifact: the plain sorted
walk on dense selectors already moves ~24 bytes/element (8 B index read +
8 B load + 8 B store) at memory bandwidth, and any coalescing scheme that
still reads the full int64 index array (run detection requires it) moves
the same bytes plus a serially dependent detection chain — at best ~1.0x
when bandwidth-bound, losses when core-bound. The finding redirected the
effort: the headroom for mask-derived reads is never materializing the
indices at all, which became Stage 12.

---

## Stage 11 — Zero-copy read API (`copy=False`)

**Branch:** `feat/zero-copy-reads` (a feature, not a kernel change)

**Problem.** Workflows that materialize whole compacted stores and hand the
arrays to NumPy/Awkward/pandas paid a full copy of every column: the bytes
already sit contiguous in the page cache behind the reader's per-column
``mode="r"`` memmaps, and the copy doubles peak resident memory and
reads+writes every byte before the consumer reads it again.

**Change.** ``ColumnView.array``, ``TableView.dict``, and the whole-store
``ColStoreReader.dict`` take ``copy: bool = True``. ``copy=False`` returns
READ-ONLY ndarray views of the open memmaps — supported exactly when the
store is single-record (``colstore.compact`` produces these), the dtype is
native byte order, and the selector is None / int / slice of any step.
Everything else raises ``ValueError`` with the remedy, never a silent copy:
the flag is a memory guarantee. ``recarray``/``frame`` are unchanged (both
repack by construction); ``copy=True`` is byte-for-byte the previous
behavior. Lifetime: views pin the mapping via ``.base`` and ``close()`` is
reference-dropping, so views remain valid after close — the
invalidate-at-close alternative was rejected as converting a refcounting
non-issue into a segfault class.

**Measured (deployment hardware, f8, warm cache).** View creation is O(1)
in the column size (~3 µs flat from 8 MB to 800 MB, vs 60 ms to copy the
800 MB column). Read-and-reduce gains 1.7–2.3x by removing the copy's
extra read+write pass; independent of timing, peak resident memory is
halved (page-cache-backed and reclaimable instead of a committed second
copy).

---

## Stage 12 — Boolean-mask-native path

**Branch:** `perf/boolean-mask-native`

**Problem.** Boolean masks lowered to int64 indices immediately
(``np.flatnonzero`` in the view layer), paying the O(N) conversion and an
8-bytes-per-selected-row allocation once, then the sortedness pass and
8 bytes/element of index traffic inside the sorted walk — per column. All
of it is derivable from the mask, which is 1 byte per row.

**Change.** ``gather_multirecord_mask`` reads the mask directly with a
monotone record walk: no index array exists, no sortedness check runs, and
run coalescing is free because the mask byte is the datum being scanned
anyway — the property whose absence rejected run coalescing on
materialized indices. The decisive design point is word-at-a-time
processing: a first version with a per-element mask test lost 0.2–0.6x at
mid densities to branch misprediction (the test is a coin flip at p=0.5);
classifying 8 mask bytes at once is predictable in every density regime —
all-zero words skip, runs of all-ones words become one memcpy (clipped at
record boundaries), mixed words compact branchlessly with the transient
over-store confined to the thread's output region by a quota guard. Two
internal passes fix per-thread output offsets, and the kernel verifies the
total count against the caller-sized output (status return; nothing
written on mismatch).

**Routing and gate.** Multi-record reads with native dtypes and mask
density (selected fraction) ≥ 0.15 take the kernel, single- and
multi-column; the gate sits at the measured crossover where per-column
index traffic (8 B per *selected* element) undercuts re-reading the
1 B-per-*row* mask. Sparse masks, single-record stores (preserving the
``backend`` parameter's contract on the resulting fancy reads), and
non-native dtypes lower to ``flatnonzero`` and run the pre-existing paths
unchanged — the density is free to check, since both routes need the
count.

**Measured (deployment hardware, f8, random masks plus a 30% whole-record
cut, R = 10³–10⁵).**

| selector             | 1 col      | 2 col      |
|----------------------|------------|------------|
| density 0.9          | 4.3–5.2x   | 2.7–3.2x   |
| density 0.5          | 4.2–6.2x   | 2.6–3.0x   |
| density 0.2          | 3.9–11.6x  | 2.0–2.6x   |
| density 0.1–0.01     | 0.94–1.07x | 0.97–1.02x |
| whole-record cut 0.3 | 4.5–8.6x   | 2.9–9.1x   |

Below-gate rows measure the fallback against itself (the route declines
there by design), confirming the parity floor; the gate's placement rests
on dev-container data showing the forced kernel at 0.4–0.6x at density
0.01. The whole-record-cut win peaks at R=10⁵ (8.6–9.1x), where runs span
entire records. (Superseded by per-host calibration: the node sweep later
measured the mask route winning at every grid density, and the compiled
default gate is now 0.0 — see the open-items register.)

---

# Net effect

For the motivating workload — multi-column selection-driven reads over
multi-record datasets — the read side is native end-to-end and now also
selector-native. Contiguous ranges memcpy at 5–23x; strided slices run an
index-free walk (up to 4.8x forward, 5.5–7.1x reversed); unsorted fancy
reads run the fused branchless kernel with binning computed once per
multi-column read, arithmetically on uniform-record files (up to 5.2x
single-column), and with per-record bases precomputed for trailing columns
on irregular files; sorted fancy reads run the load-bound linear walk
(1.9–53x with record count); boolean masks at realistic densities bypass
index materialization entirely (4–6x single-column, 2–3x multi-column, up
to 9.1x on whole-record cuts); the sortedness gate costs microseconds
instead of milliseconds; and compacted stores can be consumed zero-copy
(O(1) materialization, half the peak memory). All kernels thread correctly
above 256K elements, load misaligned packed columns safely
(UBSan-verified), honor the per-host prefetch calibration, and every
routing change sits behind a measured gate whose floor is the pre-change
behavior. Streams of event-sized records write ~2.4x faster, and per-host
calibration remains a one-command operation with fingerprint-guarded
caches.

# Rejected alternatives (do not re-propose without new evidence)

* **Fused C-column gather kernel** — wins serially (2.5–4.5x) but collapses
  at 8 threads on the deployment node (1.35–1.57x at C=8); per-thread
  stream counts saturate miss handling. (Stage 5)
* **Argsort + sorted walk + reindex for unsorted selectors** — argsort on K
  int64 costs more than the binning it would replace; measured slower than
  binning the unsorted indices directly. (Stage 2)
* **Small-K threshold for the bin-reuse route** — bin-reuse wins at every K
  from 4 to 131072 (1.44–1.62x); a threshold would add a regression.
* **Fixed software prefetch / prefetch for unsorted gathers** —
  pessimization; the old fixed d8 cost 20–29%. (Stage 3 calibration)
* **`posix_fallocate`** — measured wash, with a glibc-emulation hazard on
  filesystems without native support (older Lustre). (Stage 4)
* **SIMD gather** — the bottleneck is outstanding-miss capacity, not
  instruction throughput; `vpgather` is microcoded on the target microarchitecture.
* **Direction-templated strided kernel pair** — the single kernel's unused
  boundary branch is free (1.19 vs 1.20 ms kernel parity). (Stage 7)
* **Unconditional sortedness sampling** — below ~23K elements the full pass
  is cheaper than the sampler's fixed overhead. (Stage 8)
* **Per-column arithmetic binning without bins** — loses to the generic
  bins route at R=10³ on multi-column reads (0.85x). (Stage 9)
* **Per-reader record_base cache** — the per-call build is 0.27 ms against
  60–300 ms kernels; caching buys noise and costs resident state. (Stage 10)
* **Run-coalesced sorted gather on materialized indices** — bandwidth
  ceiling ~1.0x dense, 0.2–0.9x measured; the headroom moved to the
  mask-native path. (Round 2 rejection)
* **Per-element mask test in the mask kernel** — 0.2–0.6x from branch
  misprediction at mid densities; word-at-a-time classification is the
  shipped form. (Stage 12)
* **Views invalidated at reader close** — converts a refcounting non-issue
  into a segfault class on a read-only mapping. (Stage 11)
* **Silent copy fallback for `copy=False`** — voids the memory guarantee
  the flag exists for; unsupported cases raise with the remedy. (Stage 11)
* **Per-pattern cold-read NUMA placement** — the warm sweep never covered
  cold reads, so a cold A/B of `local` vs `interleave` across contiguous,
  sorted-fancy, and random-scatter selectors on a multi-node host tested
  whether the best placement flips by pattern. Interleave was at least as fast
  for every pattern (1.10x contiguous, ~1.0x within noise otherwise) and never
  slower; no flip, so the single `auto` default (interleave on multi-node)
  stands and no per-pattern mechanism is warranted. (`check_cold_read_placement`)

# Open / deferred items (recorded reasons)

* **Per-thread-count prefetch calibration axis** — close with one
  `--threads 8` sweep on a quiet node, or justify the axis with it.
* **Multi-record `writev` batching** — remaining tiny-record writer gap is
  per-call Python cost; complicates durability semantics.
* **int32 index kernels** — halves index bandwidth for K-large reads when
  total rows < 2³¹; premise-check the index-traffic share first. The
  mask-native path (Stage 12) already removes index traffic entirely for
  the densities where it dominates, narrowing this item's scope.
* **Native record-index scan** — open-latency for very-high-record-count
  files; only if open time shows up in real profiles.
* **Format-level column alignment** — a future format-version decision, not
  a retrofit (alignment-safe loads already solve correctness).
* **Single-record mask-native route** — single-record masks deliberately
  keep the flatnonzero + fancy path to preserve the `backend` parameter's
  contract; numpy's own boolean indexing leaves little headroom there.
* **Mask-density gate refinement** — closed twice over: the gate is a
  per-host calibration target (`colstore calibrate mask-density`) that
  sweeps the density grid through both routes and places the gate at the
  measured crossover (1.05x win margin, monotonicity required; no win
  disables the route), and the deployment-node calibration found the mask
  route winning at every grid density (1.35–3.2x, gate 0.01,
  halves-stable) — the lowered route's serial flatnonzero loses to the
  kernel's parallel passes even sparse. The compiled default is therefore
  0.0 (route on at every density); calibration exists to raise or disable
  the gate on hosts where sparse masks lose, e.g. single-core
  environments.
* **NUMA follow-ups** — the cold `auto` vs `local` comparison across access
  patterns is resolved (see rejected alternatives: interleave wins or ties,
  no per-pattern flip). Still open: a strict single-threaded-reader variant,
  and writer-side interleave scope on real Lustre paths.
