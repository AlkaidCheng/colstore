"""Side-by-side perf check: element-indexed vs byte-offset kernel entry points.

The C++ kernel exposes two entry points. The element-indexed
(:func:`colstore._gather.gather`) is the hot path for contiguous gathers;
the byte-offset (:func:`gather_bytes`) serves the multi-record reader. The
two share size-templated inner loops; the only difference is whether the
byte address per element is computed inside the kernel
(``base + indices[i] * itemsize``) or passed in as a pre-computed array.

This script measures both at ``thread_cap=1`` (serial), against ``np.take``
as a reference. It confirms:

* element-indexed (``gather``) matches the historical cost of the per-dtype
  kernel -- no allocation regression from the refactor;
* byte-offset (``gather_bytes``) costs only a little more per call because
  it does one extra array load per element instead of a multiply, but the
  caller must materialize the offsets array separately (a real cost not
  measured by this script).

The takeaway is design rationale, not a runtime decision: use ``gather`` for
contiguous reads, ``gather_bytes`` only when offsets are non-uniform.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from colstore import _gather  # type: ignore[attr-defined]
from colstore.kernels import cpp_available


def _best(fn, repeats: int) -> float:
    fn()
    b = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t)
    return b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rows", type=int, default=20_000_000)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    if not cpp_available():
        raise SystemExit("C++ extension not built.")

    dtype = np.dtype(args.dtype)
    rng = np.random.default_rng(0)
    source = rng.standard_normal(args.source_rows).astype(dtype)
    itemsize = dtype.itemsize

    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    print(
        f"Source: {args.source_rows:,} x {dtype} = "
        f"{args.source_rows * itemsize / 1e6:.1f} MB | "
        f"repeats={args.repeats}, thread_cap=1"
    )
    print()

    for pattern in ("sorted", "unsorted"):
        print(f"---- pattern = {pattern} ----")
        header = (
            f"{'n':>11}  {'np.take ms':>11}  {'gather ms':>11}  "
            f"{'gather_bytes ms':>16}  {'g/np':>6}  {'gb/np':>6}"
        )
        print(header)
        print("-" * len(header))
        for n in sizes:
            if n > args.source_rows:
                continue
            if pattern == "sorted":
                indices = np.sort(rng.choice(args.source_rows, size=n, replace=False)).astype(
                    np.int64
                )
            else:
                indices = rng.permutation(args.source_rows)[:n].astype(np.int64)
            byte_offsets = indices * itemsize

            out_np = np.empty(n, dtype=dtype)
            out_g = np.empty(n, dtype=dtype)
            out_gb = np.empty(n, dtype=dtype)
            _gather.gather(source, indices, out_g, 1)
            _gather.gather_bytes(source, byte_offsets, out_gb, 1)
            np.take(source, indices, out=out_np)
            assert np.array_equal(out_g, out_np)
            assert np.array_equal(out_gb, out_np)

            t_np = _best(
                lambda s=source, i=indices, o=out_np: np.take(s, i, out=o),
                args.repeats,
            )
            t_g = _best(
                lambda s=source, i=indices, o=out_g: _gather.gather(s, i, o, 1),
                args.repeats,
            )
            t_gb = _best(
                lambda s=source, b=byte_offsets, o=out_gb: _gather.gather_bytes(s, b, o, 1),
                args.repeats,
            )
            print(
                f"{n:>11,}  {t_np*1000:>11.3f}  {t_g*1000:>11.3f}  "
                f"{t_gb*1000:>16.3f}  {t_g/t_np:>5.2f}x  {t_gb/t_np:>5.2f}x"
            )
        print()


if __name__ == "__main__":
    main()
