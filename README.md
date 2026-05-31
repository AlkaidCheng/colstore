# ColStore

A memory-mapped columnar binary format for fast, memory-efficient I/O on
structured arrays. `colstore` lets you write a tabular dataset to a single
`.cstore` file once and then load arbitrary row/column subsets without
materializing the rest. Internally, columns are stored back-to-back as raw
NumPy bytes, reads use `np.memmap`, and fancy-index gathers run through a
parallel C++ kernel (OpenMP + software prefetching) bound via Cython. Process
memory stays bounded by the size of the output you ask for; the source file
is never fully read into RAM.

## Install

```bash
pip install colstore
```

Building from source needs a C++17 compiler and CMake ≥ 3.18. On macOS install
`libomp` (`brew install libomp`) to get the parallel kernel; without it the
build still succeeds but the kernel runs single-threaded.

## Quick start

```python
from colstore import ColStore

# Write and open in one call. `.cstore` is the canonical extension.
ds = ColStore.from_dataframe(df, "data.cstore")

# Indexing returns lazy views; no data is read yet.
ds['price']                          # ColumnView
ds[100:200]                          # TableView
ds[100:200, 'price']                 # ColumnView
ds[100:200, ['price', 'qty']]        # TableView
ds[[1, 5, 9], ['price', 'qty']]      # TableView (fancy rows + cols)

# Materialize through one of the to_* methods.
ds['price'].to_array()                          # 1D ndarray
ds[indices, ['price', 'qty']].to_dict()         # dict of 1D arrays
ds[indices, ['price', 'qty']].to_record()       # structured ndarray
ds[indices, ['price', 'qty']].to_dataframe()    # pandas DataFrame
```

## Writing from other sources

```python
from colstore import ColStore
import numpy as np

# From a dict of 1D arrays.
ColStore.from_dict(
    {"x": np.arange(100, dtype=np.float32), "y": np.arange(100, dtype=np.int64)},
    "data.cstore",
)

# From a structured (record) array.
records = np.empty(100, dtype=[("price", np.float32), ("qty", np.int32)])
ColStore.from_records(records, "data.cstore")
```

Each factory returns an opened `ColStore` ready to read from.

## Configuration

```python
from colstore import set_max_workers, set_default_madvise, set_default_backend

set_max_workers(8)                # parallel gathers across columns
set_default_madvise("sequential") # OS read-ahead hint for sorted-index reads
set_default_backend("cpp")        # gather kernel: cpp | numpy | numba
```

## On-disk format

```
[magic 8B = b"CSTORE\x00\x01"]
[manifest_len 8B (u64 little-endian)]
[manifest_json]
[zero-padding to 64-byte alignment]
[column_0 raw bytes][column_1 raw bytes]...[column_n raw bytes]
```

The manifest is a small JSON object recording `format_version`, `n_rows`,
and per-column `{name, dtype}`. Column dtypes are preserved byte-for-byte;
columns are stored back-to-back with no per-row overhead.

## Supported dtypes

Fixed-size only: `float32`, `float64`, `int8/16/32/64`, `uint8/16/32/64`,
`bool`. Object dtype (strings, Python objects) is rejected at write time —
the design point is zero-copy random access, which requires a fixed stride.

## Layout

```
colstore/
├── pyproject.toml              # scikit-build-core build
├── CMakeLists.txt              # Cython + C++ build
├── include/colstore/
│   └── gather.hpp              # public C++ header
├── src/
│   ├── cpp/gather.cpp          # OpenMP + prefetch kernel
│   ├── cython/_gather.pyx      # dtype-dispatched binding
│   └── colstore/               # Python package
│       ├── __init__.py
│       ├── config.py
│       ├── format.py
│       ├── kernels.py
│       ├── view.py             # ColumnView + TableView
│       └── store.py
└── tests/                      # pytest suite
```

## License

MIT.
