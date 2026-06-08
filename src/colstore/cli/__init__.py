"""colstore command-line interface.

Invoked as the installed console script: ``colstore <command>``.

The CLI is organized as *command groups*, one module per group under
:mod:`colstore.cli`. Each group module exposes::

    def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None

which adds its (sub)parsers and binds a ``handler`` callable via
``set_defaults(handler=...)``; the handler takes the parsed
:class:`argparse.Namespace` and returns a process exit code. Adding a
command group is one new module plus an entry in
``_COMMAND_GROUP_REGISTRARS`` below. Groups own a noun namespace
(``colstore calibration run``) so top-level names stay unambiguous; a
group may also register a top-level verb alias for its most common
operation (``colstore calibrate``).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from . import calibration

_COMMAND_GROUP_REGISTRARS: tuple[
    Callable[[argparse._SubParsersAction[argparse.ArgumentParser]], None], ...
] = (calibration.register,)


def build_parser() -> argparse.ArgumentParser:
    """Construct the full CLI parser from the registered command groups."""
    parser = argparse.ArgumentParser(
        prog="colstore",
        description="colstore command-line tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for register in _COMMAND_GROUP_REGISTRARS:
        register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)
