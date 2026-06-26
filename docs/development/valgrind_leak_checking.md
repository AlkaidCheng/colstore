# Valgrind leak checking

Developer tooling for the native gather extension — not part of the installed
package; it lives under `scripts/` and is run from a source checkout.

A Memcheck harness exercises every gather path under Valgrind, then judges the
result on leaks **attributable to colstore's own code** — so the benign
interpreter/dependency teardown that any Python process produces does not drown
out a real leak.

Three pieces plus a suppression file:

| File | Role |
|---|---|
| `run_valgrind.sh` | Driver. Sets the environment, runs Memcheck, calls the analyzer, manages the log. |
| `valgrind_workload.py` | The deterministic workload it runs — writes synthetic stores and touches every gather route, looped so per-call leaks accumulate. |
| `analyze_valgrind.py` | Parses the (large) log, categorises leak records by origin, writes a summary, extracts colstore-attributable records, sets the exit code. |
| `colstore.supp` | Suppressions for known-benign OpenMP/NumPy/loader and CPython import noise. Loaded automatically. |

## Requirements

- **Valgrind** on `PATH` (`module load valgrind` on an HPC cluster with environment modules, or
  `conda install -c conda-forge valgrind`).
- The **native extension built with debug info**, so leak stacks resolve to
  source lines instead of `???`. The `--build` flag below does this
  (`RelWithDebInfo`); otherwise build it yourself first.

## Quick start

From the repository root:

```bash
# build with debug info, then leak-check
scripts/run_valgrind.sh --build

# or, if the extension is already installed with debug info
scripts/run_valgrind.sh
```

Memcheck runs ~20–50× slower than native, so the defaults are deliberately
small (100k rows × 4 cols × 8 records, 3 iterations). The run ends with a
categorised verdict; it exits `1` if — and only if — a leak is attributable to
colstore.

## What it writes

For a log `valgrind-colstore-<timestamp>.log`, the run produces:

- `…-<timestamp>.summary.txt` — the full readable breakdown (totals, records by
  origin, top stacks, verdict).
- `…-<timestamp>.colstore-leaks.txt` — the full text of every colstore-attributable
  leak record. **This is the file to open if the verdict is FAIL.** It says
  `(no leak records attributable to colstore)` on a clean run.
- `…-<timestamp>.log.gz` — the raw Valgrind log, gzip-compressed (kept only as
  evidence; you should rarely need it).

## Options

```
--build               build the package (RelWithDebInfo) before running
--python PATH         interpreter to use (default: python3)
--rows N              rows per synthetic store      (default 100000)
--cols N              columns per store             (default 4)
--records N           records per store             (default 8)
--iterations N        repeat the workload N times   (default 3)
--threads N           gather thread cap; >1 also runs OpenMP under Memcheck
                      (default 1 — serial kernels give the cleanest report)
--track-origins       add --track-origins=yes (origins of uninitialised values)
--gen-suppressions    emit ready-to-vet suppression stanzas (off by default;
                      it greatly enlarges the log)
--top N               show the N most frequent leaking stacks (default 10)
--keep-log            leave the raw log uncompressed
--delete-log          delete the raw log after analysis (keep the summary)
--supp PATH           extra suppression file (repeatable)
--out PATH            log file path
-h, --help            full help
```

A heavier pass with threaded kernels:

```bash
scripts/run_valgrind.sh --iterations 5 --threads 4 --track-origins
```

## Analysing an existing log

The analyzer runs standalone on any saved log, plain or `.gz`:

```bash
scripts/analyze_valgrind.py valgrind-colstore-20260613-195901.log
scripts/analyze_valgrind.py some-run.log.gz --top 20
```

It writes the same `*.summary.txt` / `*.colstore-leaks.txt` files and exits `1`
iff colstore is implicated.

## Why a clean run still reports thousands of "leaks"

On a Python **not built `--with-valgrind`** (conda, most system Pythons), the
interpreter deserialises `.pyc` code objects and interns strings at import and
startup that are never freed before exit. The OS reclaims them, so they are not
real leaks, but Memcheck reports them as "definitely lost" in their thousands.
`colstore.supp` quiets the bulk; the analyzer buckets the rest as
`cpython-import` / `numpy` / `openmp` / … and the verdict ignores them because
none pass through colstore code. A clean run therefore looks like:

```
 PASS: 0 leak records attributable to colstore.
 5368 definite/indirect record(s) are interpreter/dependency teardown …
```

## Use in CI

The non-zero exit on a colstore-attributable leak makes the driver usable as a
gate:

```bash
scripts/run_valgrind.sh --build --delete-log
```
