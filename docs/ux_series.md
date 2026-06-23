# colstore Viewing & Editing Features

colstore exposes two surfaces for working with a store: the **reader** (and the
**dataset**, for many files at once) for *viewing* data, and the **frame** for *filtering
and editing* it into a new file. Both share one `col()` expression language. This document
is a reference to their features.

---

## Opening stores

`open`, `concat`, and append accept shell-style glob patterns for file paths:

```python
ds = colstore.open("runs/*.cstore")        # all matching files, naturally sorted
```

Column names are always exact — column selection never takes a wildcard.

---

## Viewing data

A reader or dataset gives lazy views over the file. Indexing returns a view, never a copy;
nothing is read until you materialize a result.

### Filtering rows with `query()` and `col()`

`query()` takes a predicate string and returns a lazy view:

```python
hot = ds.query("energy > 100 and -2.5 < eta < 2.5")
ds.query("pt > @cut and region == 'SR'", params={"cut": 30})   # parameters via @name
```

`col()` builds the same predicates as composable expressions — arithmetic, comparisons,
`.isin()`, combined with `&` `|` `~` — and stays lazy until a terminal:

```python
ds[(col("pt") > 30) & (col("region") == "SR")]
ds.where(col("pt").isin([30, 40, 50]))        # where() is the explicit verb
ds[col("pt") > 30, ["pt", "eta"]]             # filter rows and project columns at once
```

`evaluate()` resolves a predicate to a concrete view.

### Projecting columns with `select()` / `drop()`

```python
ds.select("pt", "eta")                         # keep these columns
ds.drop("weight")                              # all columns except this one
ds.query("pt > 30").select("eta")             # filter rows, then project
```

### Previewing with `head()` / `tail()`

`head()` and `tail()` return a small sample that renders as a table in a notebook and in
the terminal; `config.set_preview_rows(n)` sets how many rows. A still-lazy `query()` view
previews the matching rows without materializing the whole selection.

---

## Editing data

`ds.edit()` (or `view.edit()`) returns a `ColStoreFrame`: a lazy specification of a new
file. A pipeline reads top to bottom:

```python
cf = ds.edit()
cf = cf.assign(ratio=cf["close"] / cf["open"])     # add a derived column
cf = cf.where(col("ratio") > 1.0, "gains")         # filter rows (cuts compose)
cf = cf.astype({"open": "float32"}).drop("noise")  # cast and project
cf = cf.select("close", "open", "ratio")
```

### In place, or a new frame

The most common pitfall. Every frame **method** — `assign`, `where` / `filter`, `select`,
`drop`, `astype` — returns a **new** frame and leaves the original unchanged, so its result
must be captured (or pass `inplace=True`). Item assignment, `cf["name"] = ...`, is the
exception: it always edits the frame **in place**.

```python
cf.where(col("pt") > 30)                 # NO EFFECT — the returned frame is discarded
cf = cf.where(col("pt") > 30)            # correct: capture the result
cf.where(col("pt") > 30, inplace=True)   # or edit this frame in place
cf["pt2"] = cf["pt"] ** 2                # item assignment is always in place
```

### Building columns

Refer to a column with `cf["close"]` and combine it with operators, comparisons, and NumPy
functions (`np.log`, `np.where`, `np.clip`, ufuncs). Attach the result either way — with
`assign(...)`, which returns a new frame, or with `cf["name"] = ...`, which sets it in
place; the value is the same:

```python
cf = cf.assign(ratio=cf["close"] / cf["open"])      # returns a new frame
cf["ratio"] = cf["close"] / cf["open"]              # same value, set in place
cf = cf.assign(logpt=np.log(cf["pt"]),
               tag=np.where(cf["pt"] > 30, 1, 0))   # several columns at once
```

`col("close")` names a column the same way and is interchangeable with `cf["close"]` in
operator, comparison, and NumPy-function expressions (`np.log`, `np.where`, `np.clip`, …) —
`col()` is the usual form inside `where()` filters. Build each expression in one style:
`col()` and `cf[...]` are not combined in a single expression.

`astype(...)` casts columns: `cf.astype({"open": "float32"})`.

### Custom column functions with `apply()`

When a column is awkward to write as a NumPy expression, `apply()` runs your function on
the named columns:

```python
cf["r"] = cf.apply(lambda u, v: np.hypot(u, v), "b", "c")
```

The function receives each column as an array and returns a 1-D array of the same length
(an element-wise computation). The result dtype is detected automatically; pass
`out_dtype=` when the function cannot run on empty input or its dtype depends on the data.

### Filtering and projecting

`where()` (alias `filter()`) takes a `col()` predicate or a query string and cuts rows;
`select()` / `drop()` choose columns. Successive cuts compose. `cf.n_rows` is the number of
selected rows.

---

## Getting data out

Both surfaces share the same terminals, which materialize the selected rows:

```python
ds.dict()              # {name: 1-D ndarray}
ds.recarray()          # one structured ndarray
ds.array("price")      # one column as a 1-D ndarray
ds.frame()             # a pandas DataFrame
ds.to_root("out.root") # a ROOT file
```

On a frame they materialize the edited result the same way (`cf.dict()`, `cf.recarray()`,
`cf.array("ratio")`, `cf.frame()`).

### Reads without copying: `copy=False`

`dict`, `array`, and `frame` accept `copy=False` to return read-only arrays backed
directly by the file, with no copy, when the layout allows it; otherwise the call raises
rather than copying silently. `recarray` always repacks.

```python
d = ds.dict(copy=False)            # read-only arrays over the file
total = d["energy"].sum()
ds[100:200, ["price", "qty"]].frame(copy=False)
```

---

## Streaming large results

When a result is too large to hold at once, stream it a batch at a time.

```python
for batch in cf.iter_batches("256 MiB"):   # each batch is a frame; pick any format
    process(batch.recarray())

cf.write("out.cstore")                      # stream the edited frame to a new file
```

`iter_batches(batch_size)` yields each batch as a materialized frame, so you choose its
output format (`dict` / `recarray` / `write`, or further edits). `batch_size` is a row
count, an IEC string such as `"256 MiB"`, or `None` for the default. `copy=False` makes
each batch a read-only view valid only until the next batch is drawn — for consumers that
finish with a batch before moving on (accumulating a total, writing each batch out).

---

## Cutflow reports

A `where` / `filter` cut can be named and weighted, and `report()` summarizes the cuts:

```python
cf = (ds.edit()
        .where(col("pt") > 30, "trigger", weight="mc_weight")
        .where(col("njets") >= 2, "two jets"))
print(cf.report())     # per-cut entering / passing / efficiency, plus weighted counts
```

`report()` returns a `CutflowReport` that renders as text or HTML and exposes the rows via
`.records()`.

---

## Reductions

Scalar reductions run a full pass over the (optionally filtered) frame:

```python
cf.sum("ratio")    cf.mean("pt")    cf.min("eta")    cf.max("eta")    cf.count()
```

`sum` / `mean` take numeric columns; `min` / `max` also take datetimes; `count()` is the
number of selected rows.
