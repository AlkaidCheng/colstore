"""The ``convert`` command: convert files between colstore and other formats.

Commands::

    colstore convert SOURCE [SOURCE ...] [-o OUTPUT] [options]

A thin wrapper over :func:`colstore.convert`, so the same rules apply: one endpoint
must be a ``.cstore``, the direction follows the extensions, and ``OUTPUT`` selects
how the inputs are named -- omitted auto-names each input one-to-one, a literal path
merges every input into it, and a ``{index}`` / ``{stem}`` / ``{name}`` / ``{parent}``
template names them one-to-one. ``--dry-run`` prints the resolved input -> output plan
(via :func:`colstore.api.plan_conversions`, the same resolver ``convert`` runs) without
writing anything.
"""

from __future__ import annotations

import argparse
import sys

from .. import api
from ._common import key_value

_ON_MISMATCH_CHOICES = ("strict", "drop")

# Exceptions convert / plan_conversions raise for bad user input, reported as a clean
# "convert: ..." message rather than a traceback. OSError covers missing / existing / not-a-file
# paths; LookupError covers an unknown format and a bad {field} in an --output template;
# OverflowError covers a batch size past the C-long the readers accept.
_USER_ERRORS = (OSError, LookupError, ValueError, TypeError, OverflowError)


def _batch_size(text: str) -> int | str:
    """Parse ``--batch-size``: an all-digit value is a row count, otherwise a byte budget."""
    try:
        return int(text)
    except ValueError:
        return text


def _columns(text: str | None) -> list[str] | None:
    """Split a comma-separated ``--columns`` list into names (``None`` if not given)."""
    if text is None:
        return None
    return [name.strip() for name in text.split(",") if name.strip()]


def _max_workers(text: str) -> int | str:
    """Parse ``--max-workers``: ``auto`` or a positive integer worker count."""
    if text == "auto":
        return "auto"
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer or 'auto', got {text!r}") from None


def _describe_plan(mode: str, groups: list[tuple[list[str], str]], *, verb: str) -> None:
    """Print the resolved input -> output plan (a merge of several inputs, or pairs)."""
    sources, dest = groups[0]
    if mode == "merge" and len(sources) > 1:
        print(f"{verb} (merge) {len(sources)} files into {dest}:")
        for source in sources:
            print(f"    {source}")
        return
    # A one-to-one plan, or a single-input literal dest (a plain conversion).
    pairs = [(sources[0], dest)] if mode == "merge" else [(srcs[0], out) for srcs, out in groups]
    print(f"{verb} {len(pairs)} file(s):")
    for source, output in pairs:
        print(f"    {source} -> {output}")


def _cmd_convert(args: argparse.Namespace) -> int:
    dtypes = dict(args.dtype) if args.dtype else None
    rename = dict(args.rename) if args.rename else None
    columns = _columns(args.columns)
    try:
        mode, groups, _ = api.plan_conversions(
            args.source,
            args.output,
            format=args.format,
            rename=rename,
            output_dir=args.output_dir,
        )
    except _USER_ERRORS as error:
        print(f"convert: {error}", file=sys.stderr)
        return 1

    if args.dry_run:
        _describe_plan(mode, groups, verb="Would convert")
        print("Dry run: nothing was written.")
        return 0

    try:
        result = api.convert(
            args.source,
            args.output,
            format=args.format,
            columns=columns,
            dtypes=dtypes,
            batch_size=args.batch_size,
            compact=not args.no_compact,
            rename=rename,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            on_mismatch=args.on_mismatch,
            max_workers=args.max_workers,
        )
    except _USER_ERRORS as error:
        print(f"convert: {error}", file=sys.stderr)
        return 1

    for opened in result if isinstance(result, list) else [result]:
        close = getattr(opened, "close", None)
        if callable(close):
            close()
    _describe_plan(mode, groups, verb="Converted")
    return 0


def _configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "source",
        nargs="+",
        metavar="SOURCE",
        help="input file(s); a quoted glob (e.g. '*.h5') is expanded by colstore",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT",
        help="output path: omitted auto-names each input, a literal path merges every input "
        "into it, a {index}/{stem}/{name}/{parent} template names them one-to-one",
    )
    parser.add_argument(
        "--format",
        metavar="NAME",
        help="override the foreign format instead of inferring it from the extension",
    )
    parser.add_argument(
        "--columns",
        metavar="COL[,COL...]",
        help="convert only these columns (comma-separated)",
    )
    parser.add_argument(
        "--dtype",
        action="append",
        type=key_value,
        metavar="NAME=DTYPE",
        help="coerce a column to a dtype on import (repeatable), e.g. --dtype flag=bool",
    )
    parser.add_argument(
        "--rename",
        action="append",
        type=key_value,
        metavar="STEM=NEWSTEM",
        help="rename an output by source stem (repeatable, one-to-one only)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="write the outputs into this directory",
    )
    parser.add_argument(
        "--batch-size",
        type=_batch_size,
        metavar="N|SIZE",
        help="stream in bounded memory: a row count (e.g. 100000) or a byte budget "
        "(e.g. '256 MiB')",
    )
    parser.add_argument(
        "--no-compact",
        action="store_true",
        help="keep a streamed import multi-record instead of collapsing it to one record",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output instead of raising",
    )
    parser.add_argument(
        "--on-mismatch",
        choices=_ON_MISMATCH_CHOICES,
        default="strict",
        help="schema reconciliation when merging: 'strict' requires one shared schema, "
        "'drop' keeps the columns common to every input",
    )
    parser.add_argument(
        "--max-workers",
        type=_max_workers,
        default=None,
        metavar="N|auto",
        help="convert this many files concurrently (threads); 'auto' uses the plateau count. "
        "Peak memory scales with the worker count -- pair with --batch-size to bound it",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved input -> output plan without converting",
    )
    parser.set_defaults(handler=_cmd_convert)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the top-level ``convert`` command."""
    parser = subparsers.add_parser(
        "convert",
        help="convert files between colstore and other formats",
        # Show each option's default in --help; the argparse defaults here are the real
        # ones (e.g. on-mismatch=strict, batch-size=None -> whole-file), so users see them.
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _configure_parser(parser)
