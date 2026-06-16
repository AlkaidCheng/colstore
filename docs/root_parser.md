# ROOT parser

`colstore.parsers.root` bridges [ROOT](https://root.cern/) files and colstore in
both directions, going through ROOT's `RDataFrame`. It is part of the
`colstore.parsers` package, where each external format lives in its own module
behind a common `Parser` contract.

The two conversion functions are also re-exported at the top level, so
`colstore.from_root` / `colstore.to_root` work after a plain `import colstore`
(equivalent to importing them from `colstore.parsers`). Importing colstore does
not import ROOT — that stays lazy until a conversion runs.

ROOT (PyROOT) is **not** a pip dependency and is **imported lazily** — importing
`colstore.parsers` pulls in nothing heavy, and ROOT is loaded only the first
time a conversion actually runs. Install ROOT separately (for example
`conda install -c conda-forge root`) to use this parser.

## What can be stored

A `.cstore` column is a contiguous block of fixed-width values, so only
**fixed-size scalar branches** convert. Jagged branches (`RVec`, `vector<...>`),
array branches, and `char*` are not storable: they materialize as object arrays
and are skipped (with a warning) when columns are auto-selected, or rejected
with an error when named explicitly. Everything that is stored round-trips
through `numpy`.

## ROOT to colstore

```python
from colstore.parsers import from_root

reader = from_root("events.root", "events.cstore")
```

The source may be a path to a `.root` file, a `{treename: files}` mapping with
exactly one entry, or an existing `RDataFrame`. For a bare path the file's sole
tree is detected automatically; pass `treename=` when a file holds more than one
tree (auto-detection raises rather than guessing). A string path may also embed
the tree directly, following the uproot convention:

```python
reader = from_root("events.root:Events", "events.cstore")
```

URL-scheme and Windows-drive colons (`root://…`, `C:\…`) are not treated as
separators, and the tree name is taken after the last colon. To read a file
whose name genuinely contains a colon, pass it as a `pathlib.Path` (never split)
or use the `{treename: files}` mapping.

Rows are read in batches and each batch is written as one record, so peak memory
stays near one batch rather than the whole file. `batch_size` is the budget per
batch: a byte string such as the default `"512 MiB"` (IEC units — see the
[performance guide](performance.md)), an `int` row count, or `None` for a single
pass. By default the records are compacted into one afterward (`compact=True`)
so subsequent reads take the single-record fast path; pass `compact=False` to
leave the file multi-record.

`RDataFrame.Range` — used for chunked reads — is incompatible with implicit
multithreading, so disable `ROOT.EnableImplicitMT()` before a chunked ingest.

## colstore to ROOT

```python
from colstore.parsers import to_root

rdf = to_root("events.cstore", "events.root", treename="events")
```

`columns` selects which columns to write (in the given order), like an
`RDataFrame.Snapshot` column list; the default writes every column.

The output path is required. The store is read and snapshotted in row chunks —
the first chunk recreates the tree and later chunks append to it — so memory
stays bounded by `batch_size` regardless of file size. An `RDataFrame` over the
freshly written file is returned for immediate use.

Colstore column names that are not valid ROOT branch names (spaces, brackets, or
other symbols, such as `"mg_xsec [fb]"`) are reduced to word characters
(`mg_xsec_fb`) before writing, with a warning that names each change. Names that
collide after sanitizing are disambiguated with a numeric suffix.

## Object interface

The module functions carry the full typed signatures and are the recommended
entry points; `RootParser` is a thin object wrapper for code that dispatches
over formats uniformly:

```python
from colstore.parsers import RootParser

parser = RootParser()              # parser.format_name == "root"
reader = parser.read("events.root", "events.cstore")
rdf = parser.write(reader, "roundtrip.root")
```
