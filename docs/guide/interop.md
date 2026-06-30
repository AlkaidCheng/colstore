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
| import  | `colstore.interop.from_object("arrow", obj, dest)` | `colstore.convert("in.root", dest)`     |

`colstore.open` stays the native, magic-byte reader for `.cstore` files; foreign
files are converted with `colstore.convert`. Every export verb works on a whole
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

`colstore.convert(source, dest)` converts a file between colstore's format and
another, inferring the direction from the extensions (one endpoint must be
`.cstore`): a foreign file imports into a new `.cstore` and returns an opened
reader, a `.cstore` exports to a foreign format and returns the output path.
`ds.saveas(dest)` (and `colstore.saveas(ds, dest)`) is the export verb on an
object. The format is chosen from the path's extension; pass `format=` to override
it. An existing output raises unless `overwrite=True`. Only colstore's fixed-width
dtypes round-trip, so a format's variable-length **string** columns are coerced to
fixed-width, while **nested** (list / struct) columns, **non-string** object
columns, and **null** values are rejected with a clear error (colstore has no null
type — fill or drop nulls first; an *all*-null column stores as `float64` NaN).

`convert` also takes the multi-file and bounded-memory options: `source` may be a
glob or list (a literal `dest` merges them, a `{index}` / `{stem}` template or a
`rename` mapping names them one-to-one), and `batch_size` (an `int` row count or a
`"256 MiB"` byte budget) streams a large conversion in bounded memory in **either**
direction — import reads the foreign file in row-batches, export writes it in
row-batches (`ds.saveas(dest, batch_size=...)` streams a write too). A target with no
appendable path is written whole with a warning. See
[Writing a custom file format](#writing-a-custom-file-format) for which schemas can
stream.

Each file format also has a named shortcut: `ds.to_npz` / `to_parquet` /
`to_feather` / `to_json` / `to_hdf` (export, like `to_root`), and
`colstore.from_npz` / `from_parquet` / `from_feather` / `from_json` / `from_hdf`
(import, like `from_root`) — each is just the matching `saveas(..., format=...)`
/ `convert(..., format=...)`.

### NumPy `.npz`

One array per column; round-trips every fixed-width dtype (including fixed-width
strings) with no optional dependency. `compress=True` uses
`numpy.savez_compressed`.

```python
ds.saveas("data.npz")        # or ds.to_npz("data.npz")
back = colstore.convert("data.npz", "data.cstore")  # or colstore.from_npz(...)
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

## Writing a custom file format

A new file format is a `FileFormat` subclass that overrides a few methods. The base
class handles the chores — extension dispatch, the import template, streaming, the
`dtypes` override, compaction, and the fall-backs — so you write only the
format-specific reads and writes, and every format behaves consistently.

Set two class attributes and override the methods for the directions you support:

| direction | override | required for | returns |
| --- | --- | --- | --- |
| export (colstore → file) | `to_file(self, selection, dest, **kw)` | `saveas` / export | `None` |
| streaming export | `stream_export(self, selection, dest, *, batch_size, **kw)` | bounded-memory export | `None` if written, or a reason `str` |
| import (file → colstore) | `read_columns(self, source, *, columns=None, **kw)` | `convert` / import | a `{name: ndarray}` column dict |
| streaming import | `stream_import(self, source, *, columns=None, batch_size, **kw)` | bounded-memory import | a `StreamPlan`, or a reason `str` |

`write_file` and `from_file` are **templates you do not override**. `write_file` calls
`to_file` for a whole-selection write, or, when `batch_size` is set and `stream_export`
returns `None` (it wrote the file), streams the write in bounded memory; a returned reason
string warns and writes whole. `from_file` mirrors it for import: it calls `read_columns`,
or streams when `stream_import` returns a `StreamPlan`, then applies `dtypes`, compacts, and
on a non-streamable file reads whole with a warning. (A format that needs full control over a
direction may override the template directly, as ROOT does.) Register the format with an
[entry point](#discovery) (for a pip-installable package) or
`colstore.interop.register(MyFormat())` at runtime.

### A minimal format

```python
import numpy as np
from colstore.interop import FileFormat, register

class TsvFormat(FileFormat):
    name = "tsv"
    extensions = frozenset({".tsv"})

    def to_file(self, selection, dest, **kwargs):           # export
        data = selection.gather_all()                       # {name: ndarray}
        np.savetxt(dest, np.column_stack(list(data.values())),
                   header="\t".join(data), delimiter="\t")

    def read_columns(self, source, *, columns=None, **kwargs):   # whole-file import
        names = open(source).readline().lstrip("# ").split()
        table = np.loadtxt(source, skiprows=1, delimiter="\t", ndmin=2)
        cols = {name: table[:, i] for i, name in enumerate(names)}
        return cols if columns is None else {n: cols[n] for n in columns}

register(TsvFormat())
```

That is enough for `colstore.convert("data.tsv", "out.cstore")` and `ds.saveas("x.tsv")`
to work. A `convert(..., batch_size=...)` request also works: with no `stream_import`,
the base reads the whole file and warns that streaming is unavailable.

### Adding streaming

To import a large file in bounded memory, override `stream_import` to return a
`StreamPlan` — a lazy iterator of column batches plus the total row count:

```python
from colstore.interop import StreamPlan, resolve_batch_rows

    def stream_import(self, source, *, columns=None, batch_size, **kwargs):
        names = open(source).readline().lstrip("# ").split()
        keep = names if columns is None else columns
        rows = resolve_batch_rows(batch_size, bytes_per_row=8 * len(keep)) or 1 << 20

        def batches():
            reader = np.loadtxt(source, skiprows=1, delimiter="\t", ndmin=2)
            for start in range(0, len(reader), rows):
                chunk = reader[start : start + rows]
                yield {name: chunk[:, names.index(name)] for name in keep}

        return StreamPlan(batches(), total_rows=None)
```

`resolve_batch_rows` turns an `int` row count or a `"256 MiB"` byte budget into rows.
**Every batch must carry the same columns and the same dtypes**, already converted to
fixed-width NumPy columns: the writer locks the schema on the first batch and validates
the rest, so a dtype that drifts between batches — a string whose width is inferred per
batch, or a column that is null in one batch but typed in another — is rejected
mid-stream. So `stream_import` should stream **only a stream-stable schema**: fixed-width
numeric / bool / temporal columns with no nulls. When the file's schema is not
stream-stable, return a **reason string** instead of a `StreamPlan` and the base reads it
whole and warns — for example `return f"column {name!r} is a variable-width string"`.

The Parquet and Feather formats are worked examples of this gate built on the Arrow
helpers in `colstore.interop._stream_import` (`arrow_stream_dtypes` pins the per-column
dtype from the schema or returns a reason; `columns_from_arrow_batch` converts a batch and
rejects a stray null), and HDF5 shows the h5py / pandas variants.

### Adding streaming export

To write a large file in bounded memory, override `stream_export` to pull row-batches from
`selection.iter_batches` and append each to the open file. Return `None` once the file is
written, or a reason string to decline — the base then writes the whole selection via
`to_file` and warns:

```python
    def stream_export(self, selection, dest, *, batch_size, **kwargs):
        if kwargs:                                   # an option this path cannot carry
            return f"streaming export cannot carry {sorted(kwargs)}"
        with open(dest, "w") as handle:
            wrote_header = False
            for batch in selection.iter_batches(batch_size, copy=False):
                if not wrote_header:
                    handle.write("# " + "\t".join(batch) + "\n")
                    wrote_header = True
                np.savetxt(handle, np.column_stack(list(batch.values())), delimiter="\t")
        return None
```

`selection.iter_batches(batch_size, copy=False)` yields `{name: ndarray}` batches sized by
`batch_size`, reusing per-column buffers across batches — so write each batch before drawing
the next, and do not hold it. An empty selection yields one zero-row batch, so the schema is
still written. A target whose writer cannot append (a single-object container like NPZ, or a
write option the incremental writer cannot carry) returns a reason and is written whole. The
Parquet (row-group writer), Feather (Arrow IPC), and HDF5 (resizable datasets) formats are
worked examples; ROOT and the native `.cstore` reuse their already-bounded whole-file writers.
