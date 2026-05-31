"""A small, dependency-light progress-bar abstraction. ..."""

@runtime_checkable
class ProgressBar(Protocol):
    def update(self, n: int = ...) -> Any: ...
    def set_description(self, desc: str | None = ..., refresh: bool = ...) -> Any: ...
    def set_postfix(self, *args: Any, **kwargs: Any) -> Any: ...


class NullProgressBar:
    def update(self, n: int = 1) -> None: ...
    def set_description(self, desc=None, refresh=True) -> None: ...
    def set_postfix(self, *args, **kwargs) -> None: ...


@contextmanager
def progress_bar(total=None, *, desc=None, unit="it", enabled=True) -> Iterator[ProgressBar]:
    if not enabled:
        yield NullProgressBar(); return
    try:
        from tqdm.auto import tqdm
    except ImportError:
        warnings.warn("... install colstore[progress] ...", RuntimeWarning, stacklevel=3)
        yield NullProgressBar(); return
    with tqdm(total=total, desc=desc, unit=unit) as bar:
        yield bar