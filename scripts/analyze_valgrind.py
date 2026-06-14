#!/usr/bin/env python3
"""Summarize a Valgrind Memcheck log and deliver a leak verdict.

A leak run over a Python process produces a multi-hundred-MB log that is
impractical to read by hand: thousands of leak records, most of them benign
interpreter/dependency teardown on a Python not built ``--with-valgrind``. This
tool streams the log once (so memory stays flat regardless of log size),
classifies each leak record by where it originates, and reports:

  * the headline LEAK SUMMARY totals,
  * a breakdown of definite/indirect records by origin (colstore vs. each
    framework family),
  * the most frequent leaking call stacks,
  * a one-line VERDICT, and
  * the full text of every record attributable to colstore, extracted to a
    small companion file so the raw log rarely needs opening.

It is invoked automatically by ``run_valgrind.sh`` and also runs standalone on
any saved log (plain or ``.gz``):

    scripts/analyze_valgrind.py valgrind-colstore-*.log

Exit code: 0 if no definite/indirect leak is attributable to colstore, else 1.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections import Counter
from pathlib import Path
from typing import IO

# A leak record header, e.g.
#   ==123== 43 bytes in 1 blocks are definitely lost in loss record 6,343 of 36,125
_RECORD_HEADER = re.compile(
    r"^==\d+==\s+([\d,]+) bytes in ([\d,]+) blocks are "
    r"(definitely lost|indirectly lost|possibly lost|still reachable) in loss record"
)
# A LEAK SUMMARY total line, e.g.  ==123==    definitely lost: 1,866,772 bytes in 34,938 blocks
_SUMMARY_LINE = re.compile(
    r"^==\d+==\s+(definitely lost|indirectly lost|possibly lost|still reachable|suppressed):"
    r"\s+([\d,]+) bytes in ([\d,]+) blocks"
)
# The PID-only line that terminates a record's stack.
_RECORD_END = re.compile(r"^==\d+==\s*$")
# A stack frame's symbol: the text after "0x...: ", minus the trailing "(file:line)".
_FRAME_SYM = re.compile(r"0x[0-9A-Fa-f]+:\s+(.+?)(?:\s+\(.*\))?\s*$")
_ERROR_SUMMARY = re.compile(r"ERROR SUMMARY:\s+(\d+)\s+errors")

# Origin classification, highest priority first. colstore is first so a leak that
# passes through colstore code is always attributed to colstore, never hidden
# behind a framework frame lower in the stack.
_ORIGINS: list[tuple[str, re.Pattern[str]]] = [
    (
        "colstore",
        re.compile(
            r"_gather\.cpython|gather\.(?:cpp|hpp)|colstore_gather|gather_multirecord|"
            r"gather_indexed|gather_bytes|gather_core|gather_entry|gather_into|"
            r"copy_multirecord|make_uniform_divisor|uniform_divide"
        ),
    ),
    ("numpy", re.compile(r"numpy/|_multiarray_umath|PyInit__multiarray|\bnpy_|PyArray")),
    ("pandas/cython-ext", re.compile(r"pandas|__pyx_pymod_exec|__Pyx_Import|pytz|dateutil")),
    ("openmp", re.compile(r"libgomp|libomp|\bgomp_|__kmp_|GOMP_")),
    ("loader/tls", re.compile(r"_dl_open|_dl_allocate_tls|/ld-|_dl_init")),
    (
        "cpython-import",
        re.compile(
            r"r_object|read_object|marshal_loads|_imp_exec_builtin|_imp_create_dynamic|"
            r"PyModule_ExecDef|PyImport_ImportModule|init_interp_main|Py_InitializeFromConfig|"
            r"_PyConfig_AsDict|pymain_init|module_exec|PyInit_|PyModule_AddIntConstant|"
            r"unicode_decode_utf8"
        ),
    ),
]
_ACTIONABLE = ("definitely lost", "indirectly lost")


def _int(s: str) -> int:
    return int(s.replace(",", ""))


def _open(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return open(path, "rt", errors="replace")


def _classify(frames: list[str]) -> str:
    blob = "\n".join(frames)
    for name, pattern in _ORIGINS:
        if pattern.search(blob):
            return name
    return "other"


def _signature(frames: list[str], depth: int = 3) -> str:
    """A short top-of-stack signature for grouping near-identical leaks."""
    syms = [f for f in frames if f and f != "UnknownInlinedFun"]
    return " <- ".join(syms[:depth]) if syms else "(no symbols)"


class Stats:
    def __init__(self) -> None:
        # origin -> [records, bytes] for actionable (definite+indirect) records
        self.by_origin: dict[str, list[int]] = {}
        self.summary: dict[str, tuple[int, int]] = {}  # kind -> (bytes, blocks)
        self.signatures: Counter[tuple[str, str]] = Counter()  # (origin, sig) -> count
        self.errors = 0
        self.colstore_records = 0
        self.colstore_bytes = 0
        self.total_records = 0

    def add_record(
        self, kind: str, nbytes: int, frames: list[str], raw: str, leak_fh: IO[str]
    ) -> None:
        if kind not in _ACTIONABLE:
            return
        self.total_records += 1
        origin = _classify(frames)
        rec = self.by_origin.setdefault(origin, [0, 0])
        rec[0] += 1
        rec[1] += nbytes
        self.signatures[(origin, _signature(frames))] += 1
        if origin == "colstore":
            self.colstore_records += 1
            self.colstore_bytes += nbytes
            leak_fh.write(raw)
            leak_fh.write("\n")


def analyze(log: Path, leaks_path: Path) -> Stats:
    stats = Stats()
    in_record = False
    kind = ""
    nbytes = 0
    frames: list[str] = []
    raw: list[str] = []

    def finalize(leak_fh: IO[str]) -> None:
        nonlocal in_record
        if in_record:
            stats.add_record(kind, nbytes, frames, "".join(raw), leak_fh)
        in_record = False

    with _open(log) as fh, open(leaks_path, "w") as leak_fh:
        for line in fh:
            header = _RECORD_HEADER.match(line)
            if header:
                finalize(leak_fh)
                in_record = True
                nbytes = _int(header.group(1))
                kind = header.group(3)
                frames = []
                raw = [line]
                continue
            if in_record:
                raw.append(line)
                if _RECORD_END.match(line):
                    finalize(leak_fh)
                    continue
                m = _FRAME_SYM.search(line)
                if m:
                    frames.append(m.group(1))
                continue
            sm = _SUMMARY_LINE.match(line)
            if sm:
                stats.summary[sm.group(1)] = (_int(sm.group(2)), _int(sm.group(3)))
                continue
            em = _ERROR_SUMMARY.search(line)
            if em:
                stats.errors = int(em.group(1))
        finalize(leak_fh)
    return stats


def render(log: Path, stats: Stats, top: int) -> str:
    bar = "=" * 70
    out: list[str] = [bar, f" Valgrind leak analysis: {log.name}", bar]

    if stats.summary:
        out.append(" Memcheck totals (from LEAK SUMMARY):")
        order = [
            "definitely lost",
            "indirectly lost",
            "possibly lost",
            "still reachable",
            "suppressed",
        ]
        for kind in order:
            if kind in stats.summary:
                b, blk = stats.summary[kind]
                out.append(f"   {kind:<16} {b:>15,} bytes / {blk:>8,} blocks")
        out.append("")

    out.append(" Definite/indirect leak records by origin (actionable = colstore):")
    out.append(f"   {'origin':<18}{'records':>9}{'bytes':>15}")
    for origin in sorted(stats.by_origin, key=lambda o: (o != "colstore", -stats.by_origin[o][0])):
        recs, byts = stats.by_origin[origin]
        flag = "   <- actionable" if origin == "colstore" else ""
        out.append(f"   {origin:<18}{recs:>9,}{byts:>15,}{flag}")
    out.append(f"   {'-' * 42}")
    total_bytes = sum(v[1] for v in stats.by_origin.values())
    out.append(f"   {'total':<18}{stats.total_records:>9,}{total_bytes:>15,}")
    out.append("")

    if stats.signatures:
        out.append(f" Top {top} leaking call stacks (by record count):")
        for (origin, sig), count in stats.signatures.most_common(top):
            sig_short = sig if len(sig) <= 64 else sig[:61] + "..."
            out.append(f"   {count:>7,}  {sig_short}  [{origin}]")
        out.append("")

    out.append(" " + "-" * 16 + " VERDICT " + "-" * 43)
    if stats.colstore_records:
        out.append(
            f" FAIL: {stats.colstore_records:,} leak record(s) "
            f"({stats.colstore_bytes:,} bytes) attributable to colstore."
        )
        out.append(" The full text of each is in the colstore-leaks file below.")
    else:
        framework = stats.total_records
        out.append(" PASS: 0 leak records attributable to colstore.")
        if framework:
            out.append(f" {framework:,} definite/indirect record(s) are interpreter/dependency")
            out.append(" teardown (the OS reclaims them at exit) -- benign on a Python not")
            out.append(" built --with-valgrind. See colstore.supp to quiet them further.")
    out.append(bar)
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Summarize a Valgrind Memcheck log and give a leak verdict."
    )
    p.add_argument("log", type=Path, help="Valgrind log file (plain or .gz)")
    p.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="write the summary here (default: <log>.summary.txt)",
    )
    p.add_argument(
        "--leaks",
        type=Path,
        default=None,
        help="write colstore leak records here (default: <log>.colstore-leaks.txt)",
    )
    p.add_argument("--top", type=int, default=10, help="number of top stacks to show (default 10)")
    p.add_argument("--quiet", action="store_true", help="write files but do not print the summary")
    args = p.parse_args()

    if not args.log.is_file():
        print(f"error: log not found: {args.log}", file=sys.stderr)
        return 2

    stem = args.log.name[:-3] if args.log.suffix == ".gz" else args.log.name
    base = args.log.with_name(stem)
    summary_path = args.summary or base.with_suffix(".summary.txt")
    leaks_path = args.leaks or base.with_suffix(".colstore-leaks.txt")

    stats = analyze(args.log, leaks_path)
    report = render(args.log, stats, args.top)
    summary_path.write_text(report + "\n")
    if not leaks_path.stat().st_size:
        leaks_path.write_text("(no leak records attributable to colstore)\n")

    if not args.quiet:
        print(report)
        print(f"\n wrote summary : {summary_path}")
        print(f" wrote colstore: {leaks_path}")

    return 1 if stats.colstore_records else 0


if __name__ == "__main__":
    raise SystemExit(main())
