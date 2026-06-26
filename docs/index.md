---
sd_hide_title: true
---

# colstore

```{rst-class} lead
A memory-mapped columnar binary format for fast, memory-efficient random-access
I/O on structured arrays.
```

`colstore` writes a tabular dataset to a single `.cstore` file — in one shot or
streamed across many records — and loads arbitrary row/column subsets back
without materializing the rest. Reads are memory-mapped, so process memory stays
bounded by the output you ask for, never the file size; fancy-index and boolean
gathers run through a parallel C++/OpenMP kernel bound via Cython.

```{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`zap` Zero-decode reads
The on-disk column is raw NumPy bytes, so a gather is a parallel `memcpy` of the
pages you touch — no decode, no decompress.
:::

:::{grid-item-card} {octicon}`database` Bounded memory
Reads are `np.memmap`; a 1% slice of a file larger than RAM costs a few pages,
not the whole frame.
:::

:::{grid-item-card} {octicon}`copy` Zero-copy views
A compacted, native-byte-order store hands back read-only ndarrays aliasing the
page cache (`copy=False`).
:::

:::{grid-item-card} {octicon}`stack` Multi-file & shards
Open many same-schema files — or a growable directory of shards — as one logical
table.
:::

:::{grid-item-card} {octicon}`pencil` Lazy edits & queries
`reader.edit()` derives a new file through a deferred column-expression graph,
without touching the source.
:::

:::{grid-item-card} {octicon}`plug` Format interop
A zero-copy Arrow bridge plus Parquet / Feather / JSON / HDF5 / NPZ / ROOT
converters.
:::
```

## Install

```bash
pip install colstore
```

## Quick start

```python
import colstore

ds = colstore.store(df, "data.cstore")     # one-shot write + open
ds = colstore.open("data.cstore")          # or open an existing file

ds["price"]                                 # lazy ColumnView, nothing read yet
ds[[1, 5, 9], ["price", "qty"]].dict()      # materialize a fancy gather
ds[100:200].dict(copy=False)                # zero-copy views over the mmap
```

```{toctree}
:hidden:
:caption: User guide

Performance & internals <guide/performance>
Format interop <guide/interop>
```

```{toctree}
:hidden:
:caption: Reference

API reference <api>
```

```{toctree}
:hidden:
:caption: Development

UX series <development/ux_series>
Optimization series <development/optimization_series>
Gather diagnostics <development/gather_diagnostics>
Valgrind leak checking <development/valgrind_leak_checking>
```
