---
name: colstore
description: Use this skill whenever a task involves reading, writing, editing, or converting data with colstore — the memory-mapped columnar `.cstore` format for fast, memory-bounded random-access I/O on structured (tabular) arrays. Covers opening files and multi-file datasets, lazy NumPy-style indexing and zero-copy reads, column reductions and statistics (sum / mean / std / var / min / max / count) and membership tests, filtering and projection with query strings and `col()` expressions, deriving new files with the editing frame, one-shot and streaming/shard writes, exporting a selection with `saveas`, format interop (Arrow / Parquet / Feather / JSON / HDF5 / NPZ / ROOT), and performance tuning. Apply it whenever code touches a `.cstore` file or does `import colstore`.
---

# colstore

`colstore` is a memory-mapped columnar binary format (`.cstore`) for fast, memory-bounded
random-access I/O on structured-array datasets. A dataset is a flat table of fixed-width
typed columns. On disk, columns are stored back-to-back as **raw NumPy bytes**; reads are
`np.memmap`, so a gather is a parallel `memcpy` of only the pages you touch — no decode, no
decompression. **Process memory tracks the output you request, never the file size**, so a 1%
slice of a file larger than RAM costs a few pages.

It is a *load-and-store* format (not a streaming query engine, not a database). Reach for it
when you repeatedly read row/column subsets of large numeric tables and want bounded memory and
zero-copy where possible.

## When it fits / when it doesn't

| Good fit | Poor fit |
|---|---|
| Large tables of fixed-width dtypes (ints, floats, bool, fixed `S`/`U` strings, datetimes) | Variable-length / nested / Python-object columns — **rejected at write time** |
| Random-access reads of row/column subsets; whole-column scans | Tiny data where a `.npy`/CSV is simpler |
| Bounded-memory reads of files larger than RAM | Concurrent multi-writer transactional workloads (it's append-only, single-writer) |
| Deriving new files by transforming columns (the editing frame) | In-place row mutation (files are immutable; you derive a new one) |

## Install

```bash
pip install colstore
# optional extras, by need: pandas, arrow (pyarrow), hdf5 (h5py), uproot, numba, progress (tqdm)
pip install "colstore[arrow,pandas]"
```

Requires Python ≥ 3.10. Wheels ship the compiled C++/OpenMP gather kernel; check
`colstore.cpp_available()`.

## Mental model: four objects

| Object | Role | Get one from |
|---|---|---|
| **`ColStoreReader`** | Read an existing file | `colstore.open(path)` / `colstore.store(...)` |
| **`ColStoreDataset`** | Read many same-schema files / a shard dir as one table | `colstore.open([paths])` / `colstore.open(dir)` |
| **`ColStoreWriter`** | Stream a *new* file from data you hold | `colstore.create` / `recreate` / `update` (context managers) |
| **`ColStoreFrame`** | Derive a *new* file from an existing one by transforming columns | `reader.edit()` |

The distinction people miss is **writer vs. frame**: a writer persists arrays you already hold;
a frame derives a new file from one already on disk. Most code only needs `ColStoreReader`.

**File shape.** A file written in one shot (or run through `compact`) is **single-record**: each
column is one contiguous run — this is what makes whole-column reads and zero-copy possible. A
file appended to over time is **multi-record** (one run per append, interleaved); reads still
work but stitch across records and are not zero-copy until compacted.

## Reading

Indexing is lazy and NumPy-style — it returns a view (`ColumnView` / `TableView`); nothing is
read until you materialize.

```python
import colstore

ds = colstore.open("data.cstore")          # ColStoreReader

# --- lazy views (no I/O yet) ---
ds["price"]                                  # ColumnView (one column, by name)
ds[100:200]                                  # TableView (row slice, all columns)
ds[100:200, "price"]                         # ColumnView (rows + one column)
ds[100:200, ["price", "qty"]]                # TableView (rows + columns)
ds[[1, 5, 9], ["price", "qty"]]              # TableView (fancy rows + columns)
ds[mask_bool_array]                          # TableView (boolean row mask)

# views compose -- re-index a view and it equals the direct form:
ds[100:200]["price"]                         # ColumnView  == ds[100:200, "price"]
ds["price"][:100]                            # ColumnView  == ds[:100, "price"]
ds[100:200, ["price", "qty"]][:10]           # TableView (first 10 rows of the selection)
ds[[1, 5, 9]][["price", "qty"]]              # TableView (project a row view's columns)

# --- materialize ---
ds.array("price")                            # 1D ndarray  (== ds["price"].array())
ds[idx, ["price", "qty"]].dict()             # {name: 1D ndarray}
ds[idx, ["price", "qty"]].recarray()         # structured ndarray
ds[100:200].frame()                          # pandas DataFrame (needs pandas)
ds[100:200].dict(copy=False)                 # ZERO-COPY read-only views over the mmap

# --- introspection ---
ds.shape           # (n_rows, n_cols)
ds.n_rows          # row count (a property); ds.count() returns the same total
ds.columns         # ['price', 'qty', ...]
ds.dtypes          # {name: np.dtype}
ds.head(5); ds.tail(5)   # Preview: renders in terminal + notebook
ds.close()         # optional; the mmap is released on GC / context exit
```

**Compute on a column.** A `ColumnView` (`ds["price"]`, or `ds[rows, "price"]`) is an eager
compute handle over its selected rows:

```python
ds["price"].mean()                # also .sum() .min() .max() .std() .var() .count()
np.std(ds["price"], ddof=1)       # NumPy reductions dispatch to the same one-pass scan
ds["price"] * 2                   # operators + ufuncs materialize an ndarray of the rows
ds["price"] + ds["qty"]; np.log(ds["price"])
mask = ds["id"].isin([1, 2, 3])   # membership -> a boolean ndarray ...
ds[mask].recarray()               # ... index the table with it (== ds[ds["id"].isin([1, 2, 3])])
```

Reductions are eager scalar terminals; operators, ufuncs, and `isin` materialize the selected
rows. A *frame* column (`reader.edit()["price"]`) shares these names, but only the elementwise ones
(operators / ufuncs) stay lazy and build an expression there — its reductions are still eager
scalars. See Editing.

**Zero-copy (`copy=False`).** On a single-record, native-byte-order store, `array`/`dict`/
`recarray` with `copy=False` hand back **read-only** ndarrays aliasing the page cache — no copy.
It is a hard guarantee: if the layout can't support it (multi-record, byte-swapped, fancy
indices), it **raises** rather than silently copying. Use it for large sequential reads; `compact`
first if the file is multi-record.

## Filtering and projection

Filter with a query **string** or a composable **`col()` expression**; both stay lazy.

```python
from colstore import col

# query string (parsed eagerly, evaluated lazily):
ds.query("price > 100 and 0 < qty < 50").frame()
ds.query("price > @cut", params={"cut": 30}).dict()          # bind variables

# col() expressions (compose with & | ~, stay lazy):
ds[(col("price") > 100) & (col("region") == "EU")].recarray()
ds.where(col("qty").isin([1, 2, 3]))                          # explicit verb
ds[col("price") > 100, ["price", "qty"]].dict()              # filter rows + project cols

# projection by exact name (chainable, lazy):
ds.select("price", "qty")                                     # keep these columns
ds.drop("notes")                                             # all but these
ds.query("price > 30").select("qty")                         # filter then project
```

Only the predicate columns are read to compute the row mask, and selected columns only when you
materialize. Call `.evaluate()` (or `query(..., lazy=False)`) to resolve the row mask once so a
following `.frame()`/`.dict()` doesn't recompute it. A bad expression raises `QueryError`.

## Editing: derive a new file (`ColStoreFrame`)

`reader.edit()` returns a frame — a deferred expression graph over the store's columns. Each edit
returns a **new** frame (pass `inplace=True` to mutate), so edits branch cheaply off a shared
base. The source file is never modified.

```python
fr = ds.edit()

# transform / add columns (values are expressions built from fr["col"]):
fr = fr.with_columns(pt2=fr["pt"] * 2.0, logp=fr.apply(np.log1p, fr["price"], out_dtype="f8"))
fr = fr.assign(flag=fr["qty"] > 0)           # assign is an alias of with_columns
fr = fr.astype({"pt": "float32"})            # cast dtypes
fr = fr.drop("notes").rename({"pt": "momentum"})

# pandas-style item assignment, in place (the imperative form of with_columns):
fr["pt2"] = fr["pt"] * 2.0                   # value derived from the frame's columns (general form)
fr["s"]   = fr["a"] + fr["b"]                # ... or a scalar; both co-filter with any selection
del fr["flag"]                               # drop a column in place
# A frame-derived expression (or scalar) always works. A raw external array is accepted ONLY on a
# base frame (before any where()/index selection) and must match the store's row count -- after a
# selection the row count is data-dependent, so a bare array is rejected; use an expression there.

# row cuts with an optional cutflow label (and optional weight):
sel = (ds.edit()
       .filter("price > 100", "price cut")
       .filter(col("region") == "EU", "region", weight="w")
       .where(col("qty") > 0))               # where/filter: the same cut (label optional), unlabeled here
sel.report()                                 # cutflow table (raw/weighted survivors per cut)

# aggregations (read only the column needed):
fr.sum("pt"); fr.mean("pt"); fr.std("pt"); fr.var("pt"); fr.min("pt"); fr.max("pt"); fr.count()

# materialize in memory, or stream to a new file:
fr.array("pt"); fr.dict(); fr.recarray(); fr.frame()   # frame() -> pandas DataFrame
new_reader = fr.write("derived.cstore")      # streams the result, returns a reader for it
for batch in fr.iter_batches("256 MiB"):     # bounded-memory streaming; each batch is a ColStoreFrame
    batch.dict()                             # ... materialize it with .dict() / .recarray() / .array(name)
```

## Writing

```python
import numpy as np, colstore

# one-shot: write a single-record file and get a reader back
ds = colstore.store(data, "out.cstore")      # data: dict[str, ndarray] | recarray | DataFrame

# streaming: write record by record (schema locks on the first non-empty write)
with colstore.create("out.cstore") as w:     # create: fail if exists
    for batch in source:                     # recreate: truncate if exists
        w.write({"x": batch.x, "y": batch.y})   # update: append to an existing file
# each w.write(columns) appends one record; all columns must share the same length

colstore.compact("out.cstore")               # collapse a multi-record file → single-record
```

`store(...)` and `compact(...)` take `show_progress=False` to silence the progress bar (the
streaming writers `create/recreate/update` have no progress bar). Writes commit atomically via a
counters block — a reader opening mid-write sees only the last committed state.

## Multiple files and shards

```python
# open many same-schema files (or a shard directory) as one logical table:
ds = colstore.open(["jan.cstore", "feb.cstore"])     # ColStoreDataset, same read API
ds = colstore.open("shards_dir/")                     # a directory of shards

# combine sources — lazy view, or eagerly written in bounded memory:
view   = colstore.concat(["jan.cstore", "feb.cstore"])               # lazy ColStoreReader/Dataset
reader = colstore.concat(["jan.cstore", "feb.cstore"], out="q1.cstore")

# grow a dataset by appending shards (each append is a new file, no rewrite):
colstore.append("shards_dir/", data)                  # one shard
with colstore.appender("shards_dir/") as ap:          # streaming many shards
    for batch in source:
        ap.write(batch)
```

A `ColStoreDataset` exposes the same indexing/query/materialize API as a reader, plus
`.needs_compaction`. Persist a new shard with the module-level `colstore.append(dir, data)` /
`appender` above — they take raw arrays and write a file. (A dataset's own `.append` only extends
the in-memory view with more *files*, not raw data.)

## Format interop

colstore reads/writes other formats through a registry. `open` stays native (magic-byte `.cstore`
only); foreign formats go through explicit verbs.

```python
# export a reader / dataset / view (saveas dispatches by file extension):
ds.arrow()                          # zero-copy pyarrow Table  (== ds.to("arrow"))
ds.to_parquet("out.parquet")        # also .to_feather/.to_json/.to_npz/.to_hdf
ds.saveas("out.parquet")            # by extension or format=...; HDF5 ext is .h5/.hdf5 (not .hdf)
ds[rows, cols].saveas("subset.cstore")     # ".cstore" too -> colstore's own writer
colstore.to_root(ds, "out.root")    # ROOT (needs uproot/ROOT; numeric columns only)

# import a foreign file into a new .cstore and open it (returns a reader):
r = colstore.ingest("in.parquet", "out.cstore")            # format inferred from extension
r = colstore.from_parquet("in.parquet", "out.cstore")      # also from_feather/json/npz/hdf/root
```

Saving to `.cstore` uses colstore's own writer: it streams in bounded memory and raw-copies
unchanged columns, so a whole store — or a multi-file dataset — is copied / merged exactly like
`concat()`, never materialized. `saveas` writes a file and returns `None` (a `.root` target returns
its path); it does *not* hand back a reader — use `ingest` / `from_*`, which import a foreign file
into a new `.cstore` and return one. `ingest` rejects a `.cstore` source (already native — open it
with `colstore.open()`). ROOT export is numeric-only (ROOT branches have no fixed-width string
type), and its default multithreaded write does not preserve row order (pass `multithreading=False`
to keep it). Enumerate what's available at runtime with `colstore.interop.file_formats()` /
`colstore.interop.data_formats()`.

## Configuration & performance

Reads parallelize across an OpenMP kernel. Defaults are good; tune only with measurement.

```python
colstore.max_threads()                 # kernel's available thread count
colstore.set_max_workers(n)            # package-wide threads for multi-column reads
colstore.set_gather_thread_cap(n)      # max OpenMP threads per single gather call
colstore.ensure_calibrated()           # apply a cached per-machine thread cap, else calibrate once
colstore.set_default_backend("cpp")    # "cpp" (default) | "numpy" | "numba"
colstore.set_default_madvise("normal") # mmap advice for new opens (NUMA/large-file tuning)
```

Per-open overrides: `colstore.open(path, backend=..., madvise=..., max_workers=..., mlock=...)`.

**Best practices**
- **`compact` once, read zero-copy many.** A single-record store gives `copy=False` views; a
  multi-record one copies. Compact files you read repeatedly. Check with
  `colstore.info(path).needs_compaction` / `.n_records` (a `ColStoreDataset` has `.needs_compaction`
  too; a `ColStoreReader` does not).
- **Project and filter before materializing.** `ds.query(...).select(...)` reads only the columns
  and rows you keep — the whole point of the format.
- **Use `copy=False` for big sequential reads**; let it raise rather than assume a copy.
- **Reuse a reader** across queries (the mmap and thread pool are shared); don't reopen per access.
- **Bounded memory:** reads never load the whole file; for large derived writes, stream with
  `frame.iter_batches(...)` or `frame.write(...)` instead of materializing.
- **Row count:** `n_rows` is the property; `count()` returns the same scalar total on a reader or
  dataset, the rows a view selects, and the surviving rows on a frame. `shape`/`columns`/`dtypes`
  are properties. Prefer `n_rows`/`count()` over `len()`: `len()` gives the row count on a reader or
  dataset, but the *column* count on a frame, and raises on a view.

**Constraints / gotchas**
- Only fixed-width dtypes. Variable-length, nested, or Python-`object` columns are refused at
  write time — bake them into fixed `S`/`U` strings or split them out first.
- `copy=False` raises on multi-record or byte-swapped stores (compact first).
- Files are immutable; "editing" derives a new file via `frame.write(...)`.
- `update` appends *records* (makes the file multi-record); `append` adds *shards* to a dataset
  dir. Neither rewrites existing data.

## Public API reference

All names are importable from the top-level `colstore` package. Signatures show the most useful
arguments.

**Opening & reading**
- `open(path | [paths]) -> ColStoreReader | ColStoreDataset` — open a file, or several as one dataset.
- `ColStoreReader(path, *, madvise=, mlock=, backend=, max_workers=)` — indexing → `ColumnView`/`TableView`; `.array/.dict/.recarray/.frame(copy=True)`, `.count()`, `.query/.where/.select/.drop`, `.head/.tail`, `.edit()`, `.arrow/.to/.saveas/.to_*`; props `.shape/.n_rows/.columns/.dtypes`.
- `ColumnView`, `TableView` — lazy views that compose (re-index with `view[rows]`, `view[name]`, `view[["a","b"]]`); `.evaluate()` fixes the row mask; `.count()`. A `ColumnView` (one column) materializes with `.array()` only and is an eager compute handle: `.sum/.mean/.min/.max/.std/.var`, operators/ufuncs → `ndarray`, `.isin([...])` → boolean mask. A `TableView` has the table materializers (`.array(name)/.dict/.recarray/.frame`) and `.select/.drop`.
- `info(path) -> ColStoreInfo`, `schema(path) -> list[dict]` — metadata without reading bodies.

**Writing**
- `store(data, path, *, mode="create", batch_size="auto", statistics=False) -> ColStoreReader` — one-shot write + open.
- `create / recreate / update(path) -> ColStoreWriter` — streaming context managers (new / truncate / append).
- `ColStoreWriter.write(columns: dict[str, ndarray])` — append one record; schema locks on first write.
- `compact(path, *, out=None) -> Path` — multi-record → single-record (enables zero-copy).

**Datasets & shards**
- `ColStoreDataset(sources)` — many files as one table; reader API plus `.needs_compaction` (and `.append(source)`, which adds more *files* to the in-memory view — to persist a shard use `colstore.append(dir, data)`).
- `concat(sources, *, out=None, memory_budget=None) -> ColStoreReader | ColStoreDataset` — lazy, or written if `out`.
- `append(directory, data, *, name=, statistics=) -> Path` — add one shard.
- `appender(directory) -> Appender` / `Appender.write(data)`, `.flush()`, `.close()` — stream shards.

**Editing**
- `ColStoreFrame` (via `reader.edit()`) — transform: `.with_columns/.assign/.astype/.drop/.rename/.apply` (or in place: `fr[col] = expr`, `del fr[col]`); row cuts: `.filter/.where/.report` (cutflow); reduce: `.sum/.mean/.std/.var/.min/.max/.count`; materialize: `.array/.dict/.recarray/.frame`; stream: `.iter_batches/.write(path)`. All transforms take `inplace=`.
- `col(name) -> expression` — lazy column reference; compose with `> < == != & | ~`, `.isin([...])`.

**Format interop**
- `ingest(source, dest, *, format=None) -> ColStoreReader` and `from_parquet/from_feather/from_json/from_npz/from_hdf/from_root` — import.
- `saveas(source, dest, *, format=None)`, `to_root(source, path, ...)` — export by extension (returns `None`; a `.root` dest returns its path); a `.cstore` dest uses the native streaming writer (raw-copy / merge like `concat`), foreign formats convert (ROOT is numeric-only). Also `reader.saveas/.to(name)/.arrow()/.to_*`.

**Configuration & diagnostics**
- `set_max_workers/get_max_workers`, `set_gather_thread_cap/get_gather_thread_cap`, `set_default_backend/get_default_backend`, `set_default_madvise/get_default_madvise`.
- `calibrate(*, persist=True, rounds=10) -> int`, `ensure_calibrated() -> int`, `max_threads() -> int`, `cpp_available()`, `numba_available()`, `use_passive_openmp_wait()`.

**Exceptions:** `FormatError` (not a valid `.cstore`), `QueryError` (bad query/expression).

## End-to-end recipes

```python
# 1. Filter a large file and save the selection, in bounded memory.
import colstore
from colstore import col
ds = colstore.open("events.cstore")
(ds.query("pt > 30 and -2.5 < eta < 2.5")   # query strings: comparisons/and/or/not, no func calls
   .select("pt", "eta", "phi")
   .saveas("selected.cstore"))              # streams the selection; or .edit().write(...) for a reader

# 2. Convert Parquet -> .cstore and read a zero-copy column.
r = colstore.from_parquet("big.parquet", "big.cstore")   # single-record, so:
pt = r.array("pt", copy=False)                            # read-only view, no copy

# 3. Build a growable shard dataset, then read it as one table.
for chunk in stream_of_batches():
    colstore.append("dataset/", chunk)
all_rows = colstore.open("dataset/").query("score > 0.9").recarray()

# 4. Derive new columns and a cutflow, then materialize.
fr = colstore.open("data.cstore").edit()
fr = fr.with_columns(pt_gev=fr["pt"] / 1000.0)
fr = fr.filter("pt_gev > 25", "pt").filter(col("region") == "SR", "region")
df = fr.frame()                              # pandas DataFrame (or fr.dict() / fr.recarray())
fr.report()                                  # cutflow: survivors per cut
```
