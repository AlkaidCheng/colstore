import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProgressBar(Protocol):
    def update(self, n: int = ...) -> Any: ...
    def set_description(self, desc: str | None = ..., refresh: bool = ...) -> Any: ...
    def set_postfix(self, *args: Any, **kwargs: Any) -> Any: ...


class NullProgressBar:
    def update(self, n: int = 1) -> None:
        return None

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        return None

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        return None


@contextmanager
def progress_bar(
    total: int | None = None,
    *,
    desc: str | None = None,
    unit: str = "it",
    enabled: bool = True,
) -> Iterator[ProgressBar]:
    if not enabled:
        yield NullProgressBar()
        return

    try:
        from tqdm.auto import tqdm
    except ImportError:
        warnings.warn(
            "Progress reporting was requested but 'tqdm' is not installed; "
            "continuing without a progress bar. Install the 'progress' extra "
            "(pip install colstore[progress]) to enable it.",
            RuntimeWarning,
            stacklevel=3,
        )
        yield NullProgressBar()
        return

    with tqdm(total=total, desc=desc, unit=unit) as bar:
        yield bar