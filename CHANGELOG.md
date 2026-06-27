# Changelog

All notable changes to colstore. The current version is the
[latest release](https://github.com/AlkaidCheng/colstore/releases/latest); this
log documents it and everything since.

## [Unreleased]

### Added

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

- [[#236](https://github.com/AlkaidCheng/colstore/pull/236)] `TableView` and
  `ColumnView` now repr as a formatted table that fits the terminal width (like a
  pandas DataFrame / Series); `head()` / `tail()` previews fit the window too. The
  reader and dataset keep their compact handle repr.

### Fixed

- [[#237](https://github.com/AlkaidCheng/colstore/pull/237)] Converting a column
  expression with `np.asarray` no longer raises the NumPy 2.0 `__array__`
  copy-keyword `DeprecationWarning`.
- [[#235](https://github.com/AlkaidCheng/colstore/pull/235)] The HDF5 and JSON
  importers no longer reject a numeric column whose missing values are an in-band
  sentinel: float `NaN` and datetime/timedelta `NaT` now round-trip, matching the
  Parquet and Feather paths. Genuine out-of-band nulls (object `None`, masked
  nullable dtypes) still raise.

### Removed

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

[Unreleased]: https://github.com/AlkaidCheng/colstore/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/AlkaidCheng/colstore/releases/tag/v0.4.0
