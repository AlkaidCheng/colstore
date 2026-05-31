"""Global configuration for the colstore package.

Settings can be inspected and changed at runtime. They are used as defaults
when constructing :class:`~colstore.ColStore` instances; per-instance overrides
take precedence over these globals.
"""

import os
from typing import Literal

try:
    import psutil

    _physical_cores = psutil.cpu_count(logical=False)
    _DEFAULT_MAX_WORKERS = _physical_cores or os.cpu_count() or 1
except ImportError:
    _DEFAULT_MAX_WORKERS = os.cpu_count() or 1

MadviseOption = Literal["normal", "sequential", "random", "willneed", "dontneed"]
GatherBackend = Literal["cpp", "numpy", "numba"]

_config: dict[str, object] = {
    "max_workers": _DEFAULT_MAX_WORKERS,
    "default_madvise": "sequential",
    "default_backend": "cpp",
}


def get_max_workers() -> int:
    """Return the package-wide default thread count for multi-column reads."""
    return int(_config["max_workers"])


def set_max_workers(n: int) -> None:
    """Set the package-wide thread count for multi-column reads.

    Recommend setting to the number of *physical* CPU cores; hyperthreaded
    logical cores rarely help memory-bound workloads.
    """
    if n < 1:
        raise ValueError(f"max_workers must be >= 1, got {n}.")
    _config["max_workers"] = int(n)


def get_default_madvise() -> MadviseOption | None:
    """Return the default ``madvise`` hint applied to new ``ColStore`` opens."""
    return _config["default_madvise"]  # type: ignore[return-value]


def set_default_madvise(advice: MadviseOption | None) -> None:
    """Set the package-wide default ``madvise`` hint for new ``ColStore`` opens."""
    _config["default_madvise"] = advice


def get_default_backend() -> GatherBackend:
    """Return the default gather backend for new ``ColStore`` opens."""
    return _config["default_backend"]  # type: ignore[return-value]


def set_default_backend(backend: GatherBackend) -> None:
    """Set the default gather backend (``"cpp"``, ``"numpy"``, or ``"numba"``)."""
    if backend not in ("cpp", "numpy", "numba"):
        raise ValueError(
            f"backend must be 'cpp', 'numpy', or 'numba'; got {backend!r}."
        )
    _config["default_backend"] = backend
