# Changelog

All notable changes to colstore. The current version is the
[latest release](https://github.com/AlkaidCheng/colstore/releases/latest); this
log documents it and everything since.

## [Unreleased]

### Changed

- [[#266](https://github.com/AlkaidCheng/colstore/pull/266)] Prebuilt Linux wheels now
  require **glibc >= 2.28** (built on `manylinux_2_28` rather than the EOL
  `manylinux2014` / glibc 2.17, which crashed the wheel build under CPython 3.12+).
  Systems on glibc 2.17-2.27 (e.g. CentOS 7) install from the source distribution instead.

### Fixed

- [[#267](https://github.com/AlkaidCheng/colstore/pull/267)] The ROOT and uproot format
  backends are detected reliably again once PyROOT has been imported in the same process.
  Availability was probed with `importlib.util.find_spec`, which *raises* `ValueError`
  (instead of returning a spec) for an already-imported module whose `__spec__` is `None` --
  the state PyROOT's lazy facade leaves behind on some ROOT builds. After the first ROOT read
  or write, every later backend lookup -- including `backend="auto"` and `backend="uproot"` --
  then failed with `ValueError: ROOT.__spec__ is None`. Availability is now resolved without
  importing the module and treats an already-imported module as present.

## [0.5.0] - 2026-07-01

### Added

- [[#262](https://github.com/AlkaidCheng/colstore/pull/262)] A `max_workers` option on
  `colstore.convert` (and `colstore convert --max-workers`) to convert multiple files
  concurrently: `None` (default) is sequential, an `int` is a thread-pool worker count, and
  `"auto"` uses the per-host throughput plateau (`config.get_convert_auto_workers()`, default 8,
  tunable via `set_convert_auto_workers`). It parallelizes a glob/list one-to-one convert and a
  merge's per-file imports; the threads overlap the per-file I/O and decode (the heavy stages
  release the GIL). On a cold parallel filesystem a batch import runs ~3–5× faster; peak memory
  scales with the worker count (each worker holds one file's working set), so pair it with
  `batch_size` to bound the total. A committed benchmark, `benchmark/check_convert_parallel.py`,
  measures the thread/process speedup per format.
- [[#260](https://github.com/AlkaidCheng/colstore/pull/260)] A `colstore convert` command-line
  interface wrapping `colstore.convert`: `colstore convert SOURCE... [-o OUTPUT]` with
  `--format`, `--columns`, `--dtype NAME=DTYPE`, `--rename STEM=NEWSTEM`, `--output-dir`,
  `--batch-size`, `--no-compact`, `--overwrite`, `--on-mismatch`, and `--dry-run` (which prints
  the resolved input → output plan without writing). `convert` also gains a `columns=` parameter
  that converts only the named columns in either direction — projected as a foreign file is read
  on import, selected from the store on export. `convert` now also raises when two inputs would
  resolve to the same output path (previously a silent overwrite) instead of writing one over the
  other, and warns when the same input is listed more than once (honored, but usually a mistake).
- [[#259](https://github.com/AlkaidCheng/colstore/pull/259)] `batch_size` now streams the
  **export** direction too (`colstore.convert(".cstore -> foreign", batch_size=...)` and
  `ds.saveas(dest, batch_size=...)`), so a `.cstore` larger than memory can be written out
  in bounded memory. Parquet writes one row group per batch, Feather one Arrow record batch,
  HDF5 appends to resizable datasets, ROOT streams through its Snapshot, and a `.cstore`
  export maps the budget onto its editing-frame writer. A target with no appendable path —
  JSON / NPZ, the pandas HDF5 backend, or Feather with write options — is written whole with
  a warning, matching the import side's best-effort contract. The largest saving is for the
  formats that otherwise materialize every column first (HDF5 / JSON / NPZ): peak memory
  stays flat as the file grows instead of scaling with it.
- [[#258](https://github.com/AlkaidCheng/colstore/pull/258)] A `batch_size` option on
  `colstore.convert` for bounded-memory streaming import. With `batch_size` set (an `int`
  row count or a `"256 MiB"` byte budget), a foreign file is read in row-batches and
  streamed into the `.cstore` instead of materialized whole. It applies to Parquet,
  Feather, and HDF5 (joining ROOT, which already streamed) when the file's schema is
  fixed-width numeric / temporal with no nulls; a file that cannot stream stably — a
  variable-width string or null column, a pandas *fixed*-format HDF5 (the `to_hdf`
  default), or JSON / NPZ — falls back to a whole-file read with a warning, so
  `batch_size` is best-effort and never silently corrupts or fails mid-stream. `compact`
  (default `True`) collapses the streamed multi-record result into one record.

- [[#255](https://github.com/AlkaidCheng/colstore/pull/255)] An `on_mismatch` policy on
  `colstore.open` for multi-file datasets. The default `"strict"` keeps the existing
  behavior (every file must share one schema, or a `ValueError` is raised);
  `on_mismatch="drop"` opens the files anyway, exposing only the columns common to every
  file with one consistent dtype and warning about the rest. Handy to open a set of files
  where a column's dtype drifted between writes (e.g. `bool` in some, all-null `float64`
  in others) without re-ingesting; reads stay zero-copy since only the dataset's exposed
  schema narrows. Raises only if no column is common to all files.
- [[#254](https://github.com/AlkaidCheng/colstore/pull/254)] A `dtypes` override on
  import. `colstore.convert(src, dest, dtypes={"flag": "bool"})` (and the `from_parquet`
  / `from_feather` / `from_json` / `from_hdf` / `from_npz` sugar) coerces named columns
  to a target dtype as they are read — handy to give a column one dtype across files
  whose schemas differ (e.g. a flag that is `bool` in some files and all-null in others,
  which would otherwise mismatch as `b1` vs `f8` on a multi-file open). An all-null
  column fills the target's zero (`False` / `0`) for a bool / integer dtype and keeps
  `NaN` for a float dtype; a column with real values is cast. An override naming a
  column the file lacks raises `KeyError`.
- [[#247](https://github.com/AlkaidCheng/colstore/pull/247)] `isin()` on a column.
  `ds[name].isin(values)` on a reader/dataset column is eager — a boolean `ndarray` to
  use as a mask (`ds[ds['id'].isin(keep)]`); `frame[name].isin(values)` on a frame column
  is lazy — a boolean expression to assign or compose, with a frame filtered the usual
  way via `where(col('id').isin(...))`. A `set` / `frozenset` is expanded to its members.
  (`col(name).isin(...)` already worked inside queries.)
- [[#246](https://github.com/AlkaidCheng/colstore/pull/246)] `std()` and `var()` column
  reductions, alongside `sum` / `mean` / `min` / `max` / `count`. Available on a
  reader/dataset column (`ds[name].std()`) and a frame column (`frame[name].var()`) as
  eager scalar terminals over the column's selected rows; `np.std(column)` /
  `np.var(column)` dispatch to the same bounded-memory streaming pass. `ddof` sets the
  delta degrees of freedom (`ddof=0` population, `ddof=1` sample). The variance is a
  numerically stable one-pass computation. (median / quantile are intentionally not
  provided — they need the full column in memory.)
- [[#245](https://github.com/AlkaidCheng/colstore/pull/245)] `count()` is now available
  on every object as the scalar row count: a reader/dataset returns its total (the same
  as `n_rows`), a row-selected view the rows it selects (resolving a `col()` / `query`
  predicate), and the editing frame the rows left after its filters. Previously only the
  frame and a single column offered it (`ds[mask].count()` raised `AttributeError`).
- [[#241](https://github.com/AlkaidCheng/colstore/pull/241)] `saveas` now writes
  colstore's own format: `ds[rows, cols].saveas('out.cstore')` (or `format="cstore"`)
  saves a selection to a new `.cstore`, alongside the existing Parquet / Feather /
  JSON / HDF5 / NPZ / ROOT targets. It streams through colstore's own writer (no
  optional backend) without materializing the selection: a whole store — or a
  multi-file dataset — is raw-copied / merged exactly like `concat()`, and a row/column
  selection streams in bounded memory.
- [[#240](https://github.com/AlkaidCheng/colstore/pull/240)] A `TableView` now exposes
  the same column-access surface as the reader: index it by name for a column
  (`ds[rows]['col']` → `ColumnView`) or by a list to narrow it
  (`ds[rows][['a', 'b']]` → `TableView`), and read one column with `array(name)`. The
  shared surface lives in one `_ColumnTable` mixin, so a reader/dataset and a table
  view stay consistent. (A no-argument `array()` stays absent — several columns with
  different dtypes do not pack into one homogeneous array.)
- [[#239](https://github.com/AlkaidCheng/colstore/pull/239)] Elementwise operators and
  NumPy ufuncs on a reader/dataset column view now compute eagerly: `ds['a'] * 2`,
  `ds['a'] + ds['b']`, `ds['a'] > 0`, and `np.log(ds['a'])` materialize the selected
  rows and return an `ndarray` (operators previously raised). This completes the eager
  read surface; reductions stay scalar terminals, and the deferred-expression world
  remains the editing frame (`reader.edit()`).
- [[#237](https://github.com/AlkaidCheng/colstore/pull/237)] Pandas-style terminals
  on a single column. A reader/dataset column (`ds[name]`) and a frame column
  (`frame[name]`) now offer reductions (`sum` / `mean` / `min` / `max` / `count`),
  1-D materialization (`array()`), and the NumPy array interface, each over the
  column's current row selection. Reductions are eager terminals whether spelled as a
  method (`column.mean()`) or a NumPy reduction (`np.sum(column)`); on a frame column,
  elementwise NumPy stays lazy (`np.log(frame[name])` builds an expression). The
  reductions are shared through one `ColumnReductions` mixin and stream in bounded
  memory. `np.asarray(frame)` yields the frame's data as a structured record array
  (rather than its column names).
- [[#231](https://github.com/AlkaidCheng/colstore/pull/231)] `ColStoreFrame.frame()`
  returning a pandas DataFrame, so the editing frame has the same materializers
  (`array` / `dict` / `recarray` / `frame`) as the reader and views.
- [[#229](https://github.com/AlkaidCheng/colstore/pull/229)] Online documentation
  built with Sphinx and published to GitHub Pages.

### Changed

- [[#257](https://github.com/AlkaidCheng/colstore/pull/257)] `ingest` is renamed to
  `convert` and generalized. `convert(source, dest)` now works in both directions —
  foreign → `.cstore` (import, returns a reader), `.cstore` → foreign (export, returns the
  output path), and `.cstore` → `.cstore` (copy / merge) — inferring the direction from the
  extensions; converting between two foreign formats raises. `source` may be a single path,
  a glob, or a list: with many inputs a literal `dest` merges them into one file, a `dest`
  template (`run_{index}.cstore`) or a `rename` dict / callable names them one-to-one, and
  omitting `dest` auto-names each by swapping its extension. `output_dir`, `overwrite` (an
  existing output otherwise raises), and `on_mismatch` (for merges) round out the options.
  The `ingest` name is removed; the `from_*` import shortcuts are unchanged.
- [[#256](https://github.com/AlkaidCheng/colstore/pull/256)] A multi-file dataset no
  longer requires its files to store columns in the same order. `colstore.open([...])`
  now accepts files that share the same column names and dtypes in any order (the dataset
  takes the first file's order); reads are by name, so the result is identical. Column
  order was previously part of the strict schema check and a difference raised
  `ValueError`. A different set of names, or a dtype disagreement, still raises (use
  `on_mismatch="drop"` to open the common columns).
- [[#236](https://github.com/AlkaidCheng/colstore/pull/236)] `TableView` and
  `ColumnView` now repr as a formatted table that fits the terminal width (like a
  pandas DataFrame / Series); `head()` / `tail()` previews fit the window too. The
  reader and dataset keep their compact handle repr.

### Fixed

- [[#261](https://github.com/AlkaidCheng/colstore/pull/261)] HDF5 export with the `h5py`
  backend now preserves column order. HDF5 stores group links alphabetically by default, so a
  `.cstore` exported to `.h5` and read back came out with its columns re-sorted (the values were
  intact, only the order changed). The writer now tracks link creation order (`track_order=True`,
  h5py ≥ 2.9), so the order round-trips; on an older h5py the argument is omitted and the order
  falls back to alphabetical rather than the write failing.
- [[#253](https://github.com/AlkaidCheng/colstore/pull/253)] An all-null column imported
  from a foreign file — a pandas object column that is entirely `NaN`, or an all-null /
  null-typed Arrow column — is now stored as an all-`NaN` `float64` column instead of
  raising "colstore has no null support". Such a column carries no data and no recoverable
  type, so float `NaN` (colstore's in-band missing value) is the natural fixed-width form;
  a column that *mixes* nulls with real values still raises. Affects `convert` /
  `from_parquet` / `from_feather` / `from_json` / `from_hdf`.
- [[#244](https://github.com/AlkaidCheng/colstore/pull/244)] A view is now re-indexable
  by rows, composed onto its current selection. `ds['col'][:100]` (previously
  `TypeError: 'ColumnView' object is not subscriptable`) and `ds[rows, cols][:10]`
  (previously rejected) now work, equal to `ds[:100, 'col']` and `ds[:10, cols]`. A
  slice of a slice stays a slice; `view['col']` / `view[['a', 'b']]` still project
  columns, and `view[rows, cols]` does both.
- [[#243](https://github.com/AlkaidCheng/colstore/pull/243)] Row-selecting (fancy
  index, boolean mask, or strided slice) a fixed-width column wider than 8 bytes —
  e.g. a NumPy `<U3` / `S10` string — on a multi-record or multi-file store no longer
  raises `TypeError: Unsupported element size`. The C++ gather kernels gained a generic
  byte-copy path for any element width; the 1/2/4/8-byte typed fast path is unchanged
  (byte-identical codegen).
- [[#242](https://github.com/AlkaidCheng/colstore/pull/242)] A bare empty row index
  (`ds[[]]`) now selects no rows instead of raising `IndexError`. NumPy types an empty
  Python list as `float64`, which the row-index validator rejected; an empty selection
  is now treated as an empty integer index, the same as `ds[np.array([], dtype=int)]`.
- [[#237](https://github.com/AlkaidCheng/colstore/pull/237)] Converting a column
  expression with `np.asarray` no longer raises the NumPy 2.0 `__array__`
  copy-keyword `DeprecationWarning`.
- [[#235](https://github.com/AlkaidCheng/colstore/pull/235)] The HDF5 and JSON
  importers no longer reject a numeric column whose missing values are an in-band
  sentinel: float `NaN` and datetime/timedelta `NaT` now round-trip, matching the
  Parquet and Feather paths. Genuine out-of-band nulls (object `None`, masked
  nullable dtypes) still raise.

### Removed

- [[#250](https://github.com/AlkaidCheng/colstore/pull/250)] The optional `numba`
  gather backend. `set_default_backend()` and the per-open `backend=` argument now
  accept only `"cpp"` (default) and `"numpy"`; the `numba_available()` function and
  the `numba` install extra are removed. Numba was never the default backend, and
  importing it (with its LLVM stack) was the largest single cost in `import colstore`.
  Removing it — together with no longer scanning installed entry points when
  registering the built-in formats — makes `import colstore` about 2.5x faster, with
  no change to the C++ or NumPy gather paths.
- [[#232](https://github.com/AlkaidCheng/colstore/pull/232)] `FILE_EXTENSION` is no
  longer exported from the top-level `colstore` namespace; it remains available as
  `colstore.format.FILE_EXTENSION`.

## [0.4.0] - 2026-06-25

A major feature release: a lazy editing and query layer, multi-file and
append-shard datasets, a pluggable format-interop framework, and a performance
overhaul on new C++/OpenMP gather kernels. See the
[release notes](https://github.com/AlkaidCheng/colstore/releases/tag/v0.4.0) for
the full detail.

### Added

- **Lazy editing and queries.** `reader.edit()` returns a `ColStoreFrame` that
  derives a new file through a deferred column-expression graph: `col()`
  expressions and an `apply()` escape hatch, reductions (`sum` / `mean` / `min` /
  `max` / `count`), row filtering and projection (`query` / `where` / `select` /
  `drop`), `astype` / `assign`, in-memory materializers or `.write()` to a new
  file, and a labeled, weighted `report()` cutflow.
- **Multi-file and append-shard datasets.** `ColStoreDataset` opens many
  same-schema files — or a directory of shards — as one logical table; `concat()`
  combines sources lazily or into one written file; `append()` / `appender()`
  grow a dataset shard by shard without rewriting it.
- **Format interop (`colstore.interop`).** A zero-copy Apache Arrow bridge plus
  Parquet / Feather / JSON / HDF5 / NPZ / ROOT converters, through `ingest()` /
  `saveas()` / `to_root()` / `from_*`.

### Changed

- **Performance overhaul.** New C++/OpenMP gather kernels across the read and
  write paths, with per-pattern kernel dispatch and zero-copy reads.

---

Releases before 0.4.0 are summarized on the
[GitHub Releases](https://github.com/AlkaidCheng/colstore/releases) page.

[Unreleased]: https://github.com/AlkaidCheng/colstore/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/AlkaidCheng/colstore/releases/tag/v0.5.0
[0.4.0]: https://github.com/AlkaidCheng/colstore/releases/tag/v0.4.0
