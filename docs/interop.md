# Interop with external formats

`colstore.interop` exchanges colstore data with external formats in both
directions. Each format is a `Format` object registered under a short name, of
one of two kinds:

- a **`DataFormat`** bridges an **in-memory object** (e.g. Apache Arrow) and is
  selected **by name**;
- a **`FileFormat`** bridges an **on-disk file** (e.g. NPZ, ROOT) and is selected
  **by file extension**.

The two kinds keep object-syntax and path-syntax separate, giving four verbs:

|         | data format (object, by name)          | file format (path, by extension)        |
| ------- | -------------------------------------- | --------------------------------------- |
| export  | `ds.to("arrow")` / `ds.arrow()`        | `ds.saveas("out.npz")` / `colstore.saveas(ds, path)` |
| import  | `colstore.interop.from_object("arrow", obj, dest)` | `colstore.ingest("in.root", dest)`      |

`colstore.open` stays the native, magic-byte reader for `.cstore` files; foreign
files are imported with `colstore.ingest`. Every export verb works on a whole
store, a column, or any selection (`ds[rows, cols].saveas(...)`).

## Discovery

Formats are discovered through packaging **entry points** (groups
`colstore.data_formats` / `colstore.file_formats`), so the registry is not a
hard-coded table and a third party adds a format by declaring an entry point in
its own package — no colstore change. List what is available per kind, and add
one at runtime:

```python
colstore.interop.data_formats()   # frozenset({'arrow'})
colstore.interop.file_formats()   # frozenset({'npz', 'root'})
colstore.interop.register(MyFormat())          # runtime registration
```

A backend (`pyarrow`, PyROOT, uproot) is imported **only when a conversion runs**
— `import colstore` pulls in none of them, and listing formats does not import
their modules. The optional backends install via extras: `colstore[arrow]`,
`colstore[uproot]`.

## Data formats

### Apache Arrow

A column, selection, or whole store exports to Arrow with `to("arrow")` (or the
`arrow()` shorthand):

```python
ds.to("arrow")              # pyarrow.Table, one field per column
ds.arrow()                  # shorthand for to("arrow")
ds["price"].to("arrow")     # pyarrow.Array (a ChunkedArray over many records/files)
ds[:, ["price", "qty"]].to("arrow")
```

A native column is handed over **without copying** — the Arrow values buffer *is*
the memory-mapped file, and the mapping is kept alive for the lifetime of the
Arrow data. Reader, dataset, and view objects also implement the Arrow C Data
Interface (`__arrow_c_array__` / `__arrow_c_stream__`), so an Arrow consumer reads
colstore data directly:

```python
import pyarrow as pa
pa.table(ds)                # zero-copy
pl.from_arrow(ds["price"])  # polars, DuckDB, etc.
```

The whole column on a native store is zero-copy; a row subset, a boolean or
Unicode column, or a non-native (big-endian) file materializes first.
`datetime64` / `timedelta64` columns must be in a second-to-nanosecond unit.
Requires `pyarrow`.

### NumPy and pandas

`np.asarray` / `np.array` materialize any reader, dataset, or view: a single
column becomes a 1-D array; several columns, or a whole store, become a
structured record array (one field per column, `result[name]` is the column).

```python
np.asarray(ds["price"])              # 1-D ndarray
np.asarray(ds)                       # structured record array, one field per column
np.asarray(ds[:, ["price", "qty"]])  # likewise, for the selection
```

For a pandas DataFrame, call `ds.frame()` — it builds the columns directly and
takes `copy=False`. `pd.DataFrame(ds.dict())` is equivalent; passing a reader
itself to `pd.DataFrame(ds)` does not expand it into columns, so prefer
`ds.frame()`.

## File formats

`colstore.ingest(source, dest)` imports a foreign file into a new `.cstore` and
returns an opened reader; `ds.saveas(dest)` (and `colstore.saveas(ds, dest)`)
writes one out. The format is chosen from the path's extension; pass `format=` to
override it. `ingest`'s `dest` must not already exist (`mode="recreate"`
overwrites); `saveas` overwrites its target and rejects a selection with no
columns. Only colstore's fixed-width dtypes round-trip, so a format's
variable-length / nested columns are coerced (fixed-width strings) or rejected.

### NumPy `.npz`

One array per column; round-trips every fixed-width dtype (including fixed-width
strings) with no optional dependency. `compress=True` uses
`numpy.savez_compressed`.

```python
ds.saveas("data.npz")
back = colstore.ingest("data.npz", "data.cstore")
```

### ROOT

[ROOT](https://root.cern/) files convert through one of two interchangeable
**kernels**, chosen with `kernel=`:

- `"ROOT"` — ROOT's own `RDataFrame` (PyROOT; install separately, e.g.
  `conda install -c conda-forge root`);
- `"uproot"` — the pure-Python [uproot](https://uproot.readthedocs.io/) library
  (`pip install colstore[uproot]`);
- `"auto"` (default) — uses PyROOT if importable, otherwise uproot.

The typed entry points are `colstore.from_root` / `colstore.to_root` (`from_root`
returns an opened reader; `to_root` accepts a reader, dataset, or path and returns
the written `pathlib.Path`):

```python
reader = colstore.from_root("events.root", "events.cstore")          # auto kernel
reader = colstore.from_root("events.root:Events", "events.cstore", kernel="uproot")
path = colstore.to_root(ds, "out.root", treename="events")
ds.saveas("out.root", kernel="uproot")                                # file-verb form
```

Only **fixed-size scalar branches** are storable: a branch's storability is read
from its materialized NumPy dtype (1-D, numeric/bool), so both kernels agree
exactly and a jagged (`RVec`, `vector<…>`) or array branch is not stored.
`keep_valid_only` (default `True`) keeps the storable columns and skips the rest
with a warning; `keep_valid_only=False` raises if any column in scope is not
storable. A `columns=` name absent from the tree is always an error. Both
directions stream by `batch_size` (an `int` row count, a byte string like the
default `"512 MiB"`, or `None`). `compression_level` / `compression_algorithm` /
`output_format` / `multithreading` configure the ROOT-kernel write only.

A few behaviors differ between kernels:

- **`int8` / `uint8`** write only through the uproot kernel — ROOT's `Snapshot`
  cannot emit an 8-bit branch, so the ROOT kernel raises (cast to `int16`, or use
  `kernel="uproot"`).
- **Auto-discovered column order** follows branch declaration order (uproot) or
  `GetColumnNames()` / alphabetical (ROOT); pass an explicit `columns=` list for a
  deterministic order.
- **RNTuple** is not auto-detected (auto-detection finds only TTrees); pass
  `treename=` to read one, or to re-read a file written with
  `output_format="rntuple"`.
