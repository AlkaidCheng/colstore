# ROOT parser

`colstore.parsers.root` bridges [ROOT](https://root.cern/) files and colstore in
both directions, going through ROOT's `RDataFrame`. It is part of the
`colstore.parsers` package, where each external format lives in its own module
behind a common `Parser` contract.

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
from colstore.parsers import root_to_colstore

reader = root_to_colstore("events.root", "events.cstore")
```

The source may be a path to a `.root` file, a `{treename: files}` mapping with
exactly one entry, or an existing `RDataFrame`. For a bare path the file's sole
tree is detected automatically; pass `treename=` when a file holds more than one
tree (auto-detection raises rather than guessing).

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
from colstore.parsers import colstore_to_root

rdf = colstore_to_root("events.cstore", "events.root", treename="events")
```

The output path is required. The store is read and snapshotted in row chunks —
the first chunk recreates the tree and later chunks append to it — so memory
stays bounded by `batch_size` regardless of file size. An `RDataFrame` over the
freshly written file is returned for immediate use.

## Object interface

The module functions carry the full typed signatures and are the recommended
entry points; `RootParser` is a thin object wrapper for code that dispatches
over formats uniformly:

```python
from colstore.parsers import RootParser

parser = RootParser()              # parser.format_name == "root"
reader = parser.to_colstore("events.root", "events.cstore")
rdf = parser.from_colstore(reader, "roundtrip.root")
```
