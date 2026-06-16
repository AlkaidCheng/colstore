# colstore documentation

Supplementary documentation for colstore. The project [README](../README.md)
covers installation, the API, and a tour of the design; the pages here go
deeper.

## Guides

- **[Performance &amp; internals](performance.md)** — how the file is laid out, the
  kernel behind each access pattern, the metadata addressing model, how reads
  are parallelized, NUMA placement, zero-copy, and the practical steps to get
  the best read performance. Start here to understand how colstore works.
- **[Gather diagnostics](gather_diagnostics.md)** — the
  `benchmark/gather_perf_diagnostics.py` harness, which re-derives the thread,
  binding, and placement answers from fresh measurements on whatever host it
  runs on, plus the reference findings from the development hardware.
- **[Optimization series](optimization_series.md)** — the cumulative engineering
  record: every optimization stage across three rounds, each with the
  measurement that justified it and the alternatives that were rejected.
- **[Valgrind leak checking](valgrind_leak_checking.md)** — the developer
  Memcheck harness under `scripts/` that leak-checks the native gather extension
  and reports only leaks attributable to colstore's own code.

## Diagrams

The guides embed these; each also stands alone.

| Diagram | Shows |
|---|---|
| [file_format.svg](file_format.svg) | The on-disk `.cstore` header and record layout. |
| [record_layout.svg](record_layout.svg) | Single-record vs multi-record column layout, and what `compact()` does. |
| [kernel_dispatch.svg](kernel_dispatch.svg) | How a single-column read picks its kernel, in priority order. |
| [gather_thread_decision.svg](gather_thread_decision.svg) | How the gather thread count is decided for single- and multi-column reads. |
| [numa_placement_decision.svg](numa_placement_decision.svg) | The `auto` NUMA page-placement decision and the cold-read verdict. |
| [gather_thread_binding_status.svg](gather_thread_binding_status.svg) | Why reader-side thread binding ships off. |
