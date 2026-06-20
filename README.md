# ColStore

A memory-mapped columnar binary format for fast, memory-efficient I/O on
structured arrays. `colstore` lets you write a tabular dataset to a single
`.cstore` file (in one shot or streamed across many records) and then load
arbitrary row/column subsets without materializing the rest. Internally,
columns are stored back-to-back as raw NumPy bytes, reads use `np.memmap`,
and fancy-index gathers run through a parallel C++ kernel (OpenMP + software
prefetching) bound via Cython. Process memory stays bounded by the size of
the output you ask for; the source file is never fully read into RAM.

## Install

```bash
pip install colstore
```

Building from source needs a C++17 compiler and CMake ≥ 3.18. On macOS install
`libomp` (`brew install libomp`) to get the parallel kernel; without it the
build still succeeds but the kernel runs single-threaded.

## Quick start

```python
import colstore

# One-shot write + open. `.cstore` is the canonical extension.
ds = colstore.store(df, "data.cstore")

# Or open an existing file for read.
ds = colstore.open("data.cstore")

# Indexing returns lazy views; no data is read yet.
ds['price']                          # ColumnView
ds[100:200]                          # TableView
ds[100:200, 'price']                 # ColumnView
ds[100:200, ['price', 'qty']]        # TableView
ds[[1, 5, 9], ['price', 'qty']]      # TableView (fancy rows + cols)

# Materialize through one of the materialization methods.
ds['price'].array()                          # 1D ndarray
ds[indices, ['price', 'qty']].dict()         # dict of 1D arrays
ds[indices, ['price', 'qty']].recarray()     # structured ndarray
ds[indices, ['price', 'qty']].frame()        # pandas DataFrame
ds[100:200].dict(copy=False)                 # read-only views, no copy (see Zero-copy reads)
```

## Reader, writer, frame: which to use

colstore has three objects, one per job. Most code only ever needs
`ColStoreReader`.

| | Job | Get one from | Output |
|---|---|---|---|
| **`ColStoreReader`** | Read an existing file | `colstore.open(path)` | NumPy arrays / DataFrames, in memory |
| **`ColStoreWriter`** | Write a *new* file from data you hold | `colstore.create` / `recreate` / `update` (context manager) | a `.cstore` on disk |
| **`ColStoreFrame`** | Derive a *new* file from an existing one | `reader.edit()` | a new `.cstore` on disk, plus a reader for it |

- **`ColStoreReader`** is the read interface. `colstore.open(path)` returns one;
  index it for lazy views (`ds[rows, cols]`) and materialize with `.array()`,
  `.dict()`, `.recarray()`, or `.frame()`. This is what you use to get data out.
- **`ColStoreWriter`** is the write interface for *new* data. You use it through
  the `colstore.create` / `recreate` / `update` context managers to stream
  records into a file. For a single in-memory dataset there is no need to touch
  it directly — `colstore.store(data, path)` writes a dict, record array, or
  DataFrame in one shot. (See **Writing**, below.)
- **`ColStoreFrame`** is the edit interface. `reader.edit()` returns one; update,
  add, drop, rename, or transform its columns and call `.write(path)` to stream
  the result to a new file. It never modifies the source store.

The distinction people miss is **writer vs. frame**: a writer persists data you
are already holding in memory, while a frame derives a new file from one already
on disk by transforming its columns. Starting from raw arrays, reach for a writer
(or `store`); starting from a `.cstore` you want a modified copy of, reach for
`edit()`, which gives you a frame.

## Filtering with `query()`

`query()` selects rows with a pandas-style predicate string and returns a **lazy
view** — only the columns named in the predicate are read to build the row mask;
the selected columns aren't materialized until you call `.frame()` / `.dict()` /
`.recarray()` / `.array()`.

```python
hot = ds.query("energy > 100 and -2.5 < eta < 2.5")    # lazy TableView
hot.frame()                                             # materialize now
ds.query("pt > @cut and region == 'SR'", params={"cut": 30}).dict()
ds.query("flag in (1, 3)", columns=["pt", "eta"]).recarray()
```

The grammar is a strict whitelist evaluated **without `eval`**: column names,
numeric/string/bool literals, comparisons (including chained `a < x < b`), the
boolean operators (`and` / `or` / `not` and `& | ~` — parenthesize the bitwise
forms, which bind tighter than comparison), arithmetic, and `in` / `not in`
membership. `@name` resolves from `params` (the calling frame is never
inspected), and a bool column is a predicate on its own (`ds.query("is_signal")`).
Anything outside the grammar — a function call, an attribute — raises
`colstore.QueryError`. It behaves identically on a single file and a multi-file
dataset.

## Writing

`colstore.store(data, path)` is the one-shot path; it dispatches on the
input type:

```python
import colstore
import numpy as np

# From a dict of 1D arrays.
colstore.store(
    {"x": np.arange(100, dtype=np.float32), "y": np.arange(100, dtype=np.int64)},
    "data.cstore",
)

# From a structured (record) array.
records = np.empty(100, dtype=[("price", np.float32), ("qty", np.int32)])
colstore.store(records, "data.cstore", mode="recreate")

# From a pandas DataFrame.
colstore.store(df, "data.cstore", mode="recreate")
```

`mode="create"` (default) refuses to overwrite; `mode="recreate"` truncates
an existing file.

For multi-record streaming writes (data arriving in batches, or appending
to an existing file), use `colstore.create` / `colstore.recreate` /
`colstore.update`:

```python
# Append-only stream into a new file.
with colstore.create("data.cstore") as f:
    for batch in source:
        f.write({"x": batch.x, "y": batch.y})

# Resume appending to an existing file. Schema is loaded from the manifest
# and every write must match it. Bytes from a crashed prior writer are
# truncated on open.
with colstore.update("data.cstore") as f:
    f.write({"x": more_x, "y": more_y})
```

The writer commits records atomically on `close()` by rewriting a 32-byte
counters block. A reader opening the file mid-write sees only what the
last successful close committed.

The streaming writers (`create` / `recreate` / `update`) and `compact` take an
advisory `flock` for the duration of the write, so a second writer on the same
file fails fast with a clear error instead of corrupting it. On filesystems that
don't implement `flock` (some Lustre, GPFS, and NFS mounts), the lock is skipped
with a one-time warning and the write proceeds — there is no lock to contend for
on such a mount, so concurrent-writer detection is simply unavailable there.

## Compaction

A streaming write produces one record per `write()` call. Reads of
slice and sorted-fancy index patterns scale near-flat with record count,
but unsorted-fancy reads degrade as records accumulate. `colstore.compact`
collapses all records into one, so every read pattern takes the
single-record fast path:

```python
colstore.compact("data.cstore")                 # in-place
colstore.compact("data.cstore", out="new.cstore")  # leave source untouched
```

In-place compaction writes to a sibling temp file and atomically renames
into place; the source is untouched on failure. The byte copy uses
`os.sendfile` on Linux (kernel-space copy, no Python-side surfacing)
and `shutil.copyfileobj` on macOS/Windows; on both paths memory
footprint is bounded by the kernel/I/O buffer (tens of KB) regardless
of file size — files much larger than RAM compact fine.


## Multiple files

A run is often split across many same-schema `.cstore` files. Open them as one
logical table — every read decomposes across the files and is stitched back
together, with no data copied:

```python
ds = colstore.open(["jan.cstore", "feb.cstore", "mar.cstore"])  # a ColStoreDataset
ds.n_rows                                  # sum of the files
ds[1_000:2_000, ["price", "qty"]].dict()   # slices span the files transparently
ds[[5, 1_000_000, 7], "price"].array()     # fancy and boolean selection too
```

The result is a `ColStoreDataset`. It is empty-constructible and growable, and
takes a mix of paths (which it opens and *owns*) and already-open readers or
datasets (which it *borrows* and leaves open):

```python
from colstore import ColStoreDataset

ds = ColStoreDataset()                       # empty; grow it later
ds.append("jan.cstore")                      # opens and owns this file
ds.append(existing_reader)                   # borrows an open reader
ds |= another_reader                         # in-place combine (borrows)

combined = reader_a | reader_b | reader_c    # combine open readers into one dataset
```

A dataset supports everything a single-file reader does — indexing, the lazy
views, `dict()`/`recarray()`/`frame()`, and `edit()` — by delegating to the
per-file readers, so the tuned single-file gather path is reused unchanged; a
one-file dataset costs the same as the bare reader. Closing a dataset closes
only the files it opened, so readers you passed in stay open and remain yours to
close.

To materialize the combination as one physical file, use `concat`:

```python
# Lazy: a dataset over the files, no copy — the same as open([...]).
ds = colstore.concat(["jan.cstore", "feb.cstore"])

# Eager: stream the combined data into one new file, in bounded memory.
reader = colstore.concat(["jan.cstore", "feb.cstore"], out="q1.cstore")
```

The written file reads back on the single-record fast path. See the [dataset
read decomposition](docs/dataset_read_decomposition.svg) diagram for how a read
is split across files and reassembled.

## Introspection

```python
i = colstore.info("data.cstore")
# ColStoreInfo(path='data.cstore', n_rows=1_000_000, n_records=42,
#              columns=[a:<f4, b:<i8], file_size=8_001_232B,
#              needs_compaction=True)

colstore.schema("data.cstore")
# [{'name': 'a', 'dtype': '<f4', 'encoding': 'raw', 'nullable': False}, ...]
```

Both `info` and `schema` read only the file header (no record bodies are
scanned), so they're cheap on multi-GB files.

## Configuration

```python
from colstore import (
    set_max_workers,
    set_default_madvise,
    set_default_backend,
    set_gather_thread_cap,
    calibrate,
)
from colstore import config

set_max_workers(8)                 # parallel gathers across columns
set_default_madvise("sequential")  # OS read-ahead hint for sorted-index reads
set_default_backend("cpp")         # gather kernel: cpp | numpy | numba
set_gather_thread_cap(16)          # threads per gather (default scales with socket count)
config.set_numa_policy("auto")     # page placement: auto (interleave on multi-node) | local
config.set_write_method("auto")    # write fill: auto (pwrite where available) | pwrite | mmap
calibrate()                        # one-time: measure the thread/prefetch knees for this host
```

## Zero-copy reads (`copy=False`)

Materializing a whole store normally copies every column out of the mapping into
fresh arrays — doubling peak memory and reading+writing every byte before you
touch it. When the layout allows it, `copy=False` instead returns **read-only
views over the page cache itself**:

```python
ds = colstore.open("data.cstore")                  # ideally compacted first
d = ds.dict(copy=False)                            # read-only ndarrays backed by the mmap
total = d["energy"].sum()                          # computed straight from page-cache bytes

ds["price"].array(copy=False)                      # one read-only column
ds[100:200, ["price", "qty"]].frame(copy=False)    # a read-only DataFrame aliasing the mmap
```

`copy=False` is a **guarantee, not a hint**: it returns a real view or raises —
never a silent copy. It is supported exactly when the store is **single-record**
(`colstore.compact` produces these), the dtype is **native byte order**, and the
selector is whole-store / an int / a slice of any step. A fancy or boolean
selector needs a gather, which by definition copies, so it raises `ValueError`
with the remedy; the whole-table forms are all-or-nothing and never return a mix
of views and copies.

`array`, `dict`, and `frame` all take `copy=False`; `recarray` always repacks
(it interleaves the columns into one record buffer) and so ignores it. View
creation is O(1) in column size and **halves peak resident memory** — the data
stays page-cache-backed and reclaimable instead of committed to a second buffer.
Views pin the mapping, so they stay valid after `ds.close()`. Because the views
are read-only they cannot corrupt the store; use the default `copy=True` when you
need to mutate. The full contract is in
[Performance &amp; internals](docs/performance.md) §7.

## How reads parallelize

A gather's thread count is decided in two stages. A single-column read runs at
the full gather thread cap; a multi-column read (`dict` / `recarray` / `frame`)
either splits the cap across a per-column pool or runs the columns sequentially
at the full cap, depending on the route taken. Either way the kernel's
`resolve_thread_count` has the final say: it scales the actual thread count with
the number of indices and clamps it to the cap, so small reads stay serial and
only large ones spend the whole budget.

![Gather thread decision flow](docs/assets/gather_thread_decision.svg)

The cap itself defaults to half the physical cores, bounded by a per-socket
allowance so multi-socket hosts (with more memory channels) get a higher
default; `colstore.autotune` refines it per host by measuring the saturation
knee directly.

## NUMA placement

On a multi-node (multi-socket) Linux host, the default `auto` policy interleaves
a store's pages across nodes so the memory controllers share the load; on a
single-node host, or under `local`, it leaves the kernel's first-touch placement
that keeps pages near the reading thread. Placement is decided at the first page
fault, so warm pages cannot be moved — only a cold read (pages not yet resident)
is placed according to the policy.

![NUMA placement decision](docs/assets/numa_placement_decision.svg)

Whether the best cold-read placement depends on the access pattern was the one
case the warm sweep never covered. `benchmark/check_cold_read_placement.py`
measured it on a multi-node host: cold, across contiguous, sorted-fancy, and
random-scatter selectors, interleave was at least as fast as local for every
pattern (1.10× on the contiguous scan, within noise otherwise) and never
slower. The winner never flips to local, so there is nothing for a per-pattern
mechanism to exploit and the single `auto` default stands. Revisit only if a
host or access pattern is found where local beats interleave for cold reads.

Pinning the gather threads (rather than placing pages) is a separate, opt-in
lever that ships **off**: a placement × binding × cap sweep measured spread
binding 24–51% slower on a multi-node host, so `gather_binding` defaults to off
and the realized path is the unbound default pool.

![Gather thread binding status](docs/assets/gather_thread_binding_status.svg)

## How writes reach disk

Every write — `frame.write()`, `concat(..., out=...)`, `write_dataset` — chooses
a path and a fill method:

```
write
│
├─ pure merge?  every output column an unchanged on-disk passthrough
│  │            (e.g. concat of same-schema files, no edits)
│  └─ yes ─► merge copy   : copy each source column's byte ranges straight
│                           into the output (no materialization)
│
└─ no           a transform, a new/dropped/renamed column, an in-memory
   │            or constant column, or a single source
   └────────► streaming write : evaluate each column in bounded-memory
                                 batches, then write each batch out

   both paths fill the destination body with one of:
     pwrite  (default where os.pwrite exists)  large sequential writes
     mmap    (fallback, e.g. Windows)          memory-mapped output
```

**Why the fill method matters.** An `mmap`'d output is dirtied one page at a
time; a parallel filesystem serves that pattern poorly — a 1 GB write faults
~250k pages and runs at a fraction of the device's bandwidth. `pwrite` issues
large contiguous writes instead, which such filesystems serve well: measured ~2×
faster for the streaming write and ~4× for the merge copy on a parallel
filesystem (zero page faults vs ~250k), and faster node-local too. `pwrite` is
therefore the default wherever `os.pwrite` exists; `mmap` remains the fallback on
Windows.

**Controlling it.** The default (`auto`) is right on every filesystem measured so
far. Override only to force a method — to reproduce the `mmap` path, or on a
platform where it is faster:

```python
from colstore import config

config.set_write_method("pwrite")  # auto (default) | pwrite | mmap
```

The merge copy reuses the gather thread budget to copy its byte ranges in
parallel; the streaming write fills one batch at a time within the configured
memory budget. Output is byte-identical across every method.

## On-disk format

![The .cstore on-disk format](docs/assets/file_format.svg)

```
[magic 8B = b"CSTORE\x00\x01"]
[counters 32B: n_records(8) + committed_rows(8) + crc32(4) + reserved(12)]
[manifest_len 8B (u64 little-endian)]
[manifest_json: format_version + columns + manifest_crc32]
[zero-padding to 64-byte alignment]
[record_0 header 32B][record_0 body, padded to 8B]
[record_1 header 32B][record_1 body, padded to 8B]
...
```

The JSON manifest is immutable and carries only the schema; the mutable
record/row counters live in their own fixed-position 32-byte block (with
its own CRC) right after the magic. Each record body holds the columns
back-to-back as raw bytes. A one-shot write produces a single-record file
that reads via a per-column memmap fast path; a streamed write produces a
multi-record file with per-pattern dispatch (contiguous range, sorted
fancy, unsorted fancy).

![Single-record vs multi-record column layout](docs/assets/record_layout.svg)

## Supported dtypes

All fixed-size NumPy dtypes are supported: `float32`/`float64`,
`int8/16/32/64`, `uint8/16/32/64`, `bool`, `datetime64`, `timedelta64`,
and fixed-width strings (`S` bytes and `U` unicode, with the width baked
into the dtype, e.g. `S16` or `U8`). Object dtype (variable-length
strings, Python objects) is rejected at write time — the design point is
zero-copy random access, which requires a fixed stride per row.

## Design philosophy

A few choices shape everything above:

- **Load and write, not stream-compute.** colstore persists a structured array
  and reads arbitrary row/column subsets back fast; it is not a query engine.
  Reads are memory-mapped, so process memory stays bounded by the output you ask
  for, never the file size.
- **Speed first, on the hardware that matters.** The hot paths — every
  access-pattern gather, the SoA→AoS record interleave, the contiguous and merge
  copies — run in C++/OpenMP kernels bound through Cython, dispatched per pattern
  (contiguous range, strided, sorted/unsorted fancy, boolean mask). Performance is
  judged on multi-socket, multi-NUMA compute nodes; wheels stay portable (no
  `-march=native`), with per-host calibration of the thread cap and prefetch
  distance.
- **No optimization ships without measurement.** Every change is gated by an
  interleaved A/B against the path it replaces, asserted output-identical, and
  carries a committed `benchmark/check_*.py`. Rejected and deferred alternatives
  are recorded with the measurement that closed them and a named reopen condition
  (see the [optimization series](docs/optimization_series.md)), so an idea is not
  relitigated without new evidence.
- **Zero-copy where the layout permits.** A compacted, native-byte-order store is
  contiguous on disk, so whole-column reads can hand back read-only views of the
  page cache (`copy=False`) instead of copying. The format is built around a fixed
  stride per row to keep this possible — which is why variable-length object
  dtypes are refused at write time.
- **Correctness is not traded for speed.** Misaligned packed columns load safely
  (UBSan-verified), writes commit atomically via a fixed counters block, a reader
  opening mid-write sees only the last committed state, and `copy=False` is a hard
  guarantee that raises rather than silently copying.

## Documentation

In-depth guides live in [`docs/`](docs/):

- [Performance &amp; internals](docs/performance.md) — the file layout, the kernel
  behind each access pattern, how reads parallelize, NUMA placement, and
  zero-copy.
- [Gather diagnostics](docs/gather_diagnostics.md) — re-measure the thread,
  binding, and placement knobs on your own hardware.
- [Optimization series](docs/optimization_series.md) — the cumulative record of
  every optimization and the measurement behind it.
- [Valgrind leak checking](docs/valgrind_leak_checking.md) — the native-leak
  Memcheck harness under `scripts/`.

The [docs index](docs/README.md) lists everything, including the diagrams.

## License

MIT License - see [LICENSE](LICENSE) file for details.
