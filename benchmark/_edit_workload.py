"""Synthetic expression graphs for the lazy-edit benchmarks.

Shared builders so the edit benchmarks (CSE, batch size, layout, budget) all
construct the same kinds of workload the same way. Two shapes recur:

  shared      -- many output columns reuse one expensive subexpression of a
                 single leaf; the case common-subexpression elimination is meant
                 to win, since a shared-memo pass computes that subexpression
                 once per row range instead of once per column.
  independent -- every output column is an expensive transform of its own leaf,
                 with nothing shared; the case where there is no CSE to exploit
                 and per-column write locality is the only thing in play.

Keeping construction here means a change to the workload shape lands in one
place for every benchmark. Every transform is built from whitelisted ufuncs and
operators, so the graphs route through the same evaluation path the sink uses.
"""

from __future__ import annotations

import numpy as np

from colstore.frame import Expr, MemoryColumn


def random_array(n: int, dtype: str, seed: int) -> np.ndarray:
    """A reproducible standard-normal column of ``n`` rows in ``dtype``."""
    return np.random.default_rng(seed).standard_normal(n).astype(dtype)


def expensive(base: Expr) -> Expr:
    """A multi-ufunc subexpression of one leaf: the part worth sharing.

    Reuses an inner ``abs(base) + 1`` term across a log and a sqrt, so the node
    has internal sharing of its own on top of being shared between columns.
    """
    shifted = np.abs(base) + 1.0
    return np.log(shifted) * np.sqrt(shifted) + np.sin(base)


def shared_graph(n: int, k: int, *, dtype: str = "f8", seed: int = 0) -> dict[str, Expr]:
    """``k`` output columns that all reuse one expensive subexpression."""
    base = MemoryColumn(random_array(n, dtype, seed))
    core = expensive(base)
    return {f"c{i}": core * float(i + 1) + float(i) for i in range(k)}


def independent_graph(n: int, k: int, *, dtype: str = "f8", seed: int = 0) -> dict[str, Expr]:
    """``k`` output columns, each an expensive transform of its own leaf."""
    return {f"c{i}": expensive(MemoryColumn(random_array(n, dtype, seed + i))) for i in range(k)}


def passthrough_graph(n: int, k: int, *, dtype: str = "f8", seed: int = 0) -> dict[str, Expr]:
    """``k`` plain in-memory columns with no transform: the memcpy-bound shape."""
    return {f"c{i}": MemoryColumn(random_array(n, dtype, seed + i)) for i in range(k)}
