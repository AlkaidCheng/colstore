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
colstore.interop.file_formats()   # {'npz', 'parquet', 'feather', 'json', 'hdf5', 'root'}
colstore.interop.register(MyFormat())          # runtime registration
```

A backend (`pyarrow`, PyROOT, uproot, `h5py`, `pandas`) is imported **only when a
conversion runs** — `import colstore` pulls in none of them, and listing formats
does not import their modules. The optional backends install via extras:
`colstore[arrow]` (Parquet / Feather), `colstore[uproot]` (ROOT), `colstore[hdf5]`,
and `colstore[pandas]` (JSON).

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
variable-length **string** columns are coerced to fixed-width, while **nested**
(list / struct) columns, **non-string** object columns, and **null** values are
rejected with a clear error (colstore has no null type — fill or drop nulls
first).

Each file format also has a named shortcut: `ds.to_npz` / `to_parquet` /
`to_feather` / `to_json` / `to_hdf` (export, like `to_root`), and
`colstore.from_npz` / `from_parquet` / `from_feather` / `from_json` / `from_hdf`
(import, like `from_root`) — each is just the matching `saveas(..., format=...)`
/ `ingest(..., format=...)`.

### NumPy `.npz`

One array per column; round-trips every fixed-width dtype (including fixed-width
strings) with no optional dependency. `compress=True` uses
`numpy.savez_compressed`.

```python
ds.saveas("data.npz")        # or ds.to_npz("data.npz")
back = colstore.ingest("data.npz", "data.cstore")   # or colstore.from_npz(...)
```

### Parquet and Feather

Apache Parquet (`.parquet`, `.pq`) and Feather / Arrow IPC (`.feather`) go through
pyarrow, reusing the zero-copy Arrow export. Both preserve dtypes; string columns
widen to fixed-width unicode, list / struct columns are rejected. Requires
`colstore[arrow]`.

```python
ds.to_parquet("data.parquet")
back = colstore.from_parquet("data.parquet", "data.cstore")
ds.to_feather("data.feather")
```

### JSON

Via pandas, with `orient=` (default `"records"`) for the JSON layout. JSON is a
text format and carries no dtypes, so values round-trip but the exact width may
not (e.g. `float32` reads back as `float64`). Requires `colstore[pandas]`.

```python
ds.to_json("data.json")                       # orient="records"
back = colstore.from_json("data.json", "data.cstore")
```

### HDF5

HDF5 (`.h5`, `.hdf5`) is read across writers: a **pandas / PyTables** store
(`DataFrame.to_hdf`) is detected by its `pandas_type` attribute and read with
`pandas.read_hdf`; a **plain h5py** file (one dataset per column, at the root or
under a group) is read with h5py. Export chooses the writer with `backend=`
(`"h5py"` default, no pandas/PyTables needed; or `"pandas"`) and the group with
`key=` (default `"data"`). Requires `colstore[hdf5]` (and pandas + PyTables for
the pandas backend).

```python
ds.to_hdf("data.h5")                          # h5py backend, key="data"
ds.to_hdf("data.h5", backend="pandas", key="table")
back = colstore.from_hdf("data.h5", "data.cstore")        # auto-detects the writer
back = colstore.from_hdf("other.h5", "data.cstore", key="mygroup")
```

### ROOT

[ROOT](https://root.cern/) files convert through one of two interchangeable
**backends**, chosen with `backend=`:

- `"ROOT"` — ROOT's own `RDataFrame` (PyROOT; install separately, e.g.
  `conda install -c conda-forge root`);
- `"uproot"` — the pure-Python [uproot](https://uproot.readthedocs.io/) library
  (`pip install colstore[uproot]`);
- `"auto"` (default) — uses PyROOT if importable, otherwise uproot.

The typed entry points are `colstore.from_root` / `colstore.to_root` (`from_root`
returns an opened reader; `to_root` accepts a reader, dataset, or path and returns
the written `pathlib.Path`):

```python
reader = colstore.from_root("events.root", "events.cstore")          # auto backend
reader = colstore.from_root("events.root:Events", "events.cstore", backend="uproot")
path = colstore.to_root(ds, "out.root", treename="events")
ds.saveas("out.root", backend="uproot")                                # file-verb form
```

A **list of files** (or a `{treename: [files]}` mapping) is read as one combined
dataset over a shared tree — taken from `treename=`, an embedded
`"file.root:tree"`, or the first file's sole tree:

```python
reader = colstore.from_root(["a.root", "b.root"], "all.cstore", treename="events")
reader = colstore.from_root({"events": ["a.root", "b.root"]}, "all.cstore")
```

Only **fixed-size scalar branches** are storable: a branch's storability is read
from its materialized NumPy dtype (1-D, numeric/bool), so both backends agree
exactly and a jagged (`RVec`, `vector<…>`) or array branch is not stored.
`keep_valid_only` (default `True`) keeps the storable columns and skips the rest
with a warning; `keep_valid_only=False` raises if any column in scope is not
storable. A `columns=` name absent from the tree is always an error. Both
directions stream by `batch_size` (an `int` row count, a byte string like the
default `"512 MiB"`, or `None`). `compression_level` / `compression_algorithm` /
`output_format` / `multithreading` configure the ROOT-backend write only.

A few behaviors differ between backends:

- **`int8` / `uint8`** write only through the uproot backend — ROOT's `Snapshot`
  cannot emit an 8-bit branch, so the ROOT backend raises (cast to `int16`, or use
  `backend="uproot"`).
- **Auto-discovered column order** follows branch declaration order (uproot) or
  `GetColumnNames()` / alphabetical (ROOT); pass an explicit `columns=` list for a
  deterministic order.
- **RNTuple** is not auto-detected (auto-detection finds only TTrees); pass
  `treename=` to read one, or to re-read a file written with
  `output_format="rntuple"`.
