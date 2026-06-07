"""The ``calibration`` command group: run, inspect, and clear calibration.

Commands::

    colstore calibration run [TARGET ...]    measure and persist calibration
    colstore calibration show [TARGET ...]   caches and fingerprint status
    colstore calibration clear [TARGET ...]  remove cached calibration
    colstore calibrate [TARGET ...]          top-level alias for ``run``

Calibration targets are declared once in :data:`_TARGETS` below; ``run``,
``show``, and ``clear`` (including their target choices and the per-target
``--<name>-rounds`` options) are all rendered from that registry. Adding a
future calibration is one ``_Target`` entry -- no command logic changes.

Registry order is execution order and encodes dependencies: the thread cap is
calibrated before the prefetch distances because the prefetch sweep is
measured at the configured cap, and the mask-density gate runs last because
its sweep reads through routes whose timing depends on both. Selecting a
subset never reorders execution.
Rerunning simply remeasures and overwrites the caches. Calibration should run
on the hardware the jobs run on; on a cluster, prefer a dedicated compute
node over a shared login node.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .. import autotune


@dataclass(frozen=True)
class _Target:
    """One calibration target, as rendered by run/show/clear.

    ``run`` receives ``(rounds, persist)`` and is expected to print its own
    verbose progress; ``clear`` returns whether a cache file existed;
    ``cache_path``/``load`` back the ``show`` report. Callables go through the
    :mod:`colstore.autotune` module attributes at call time (lambdas, not
    bound references) so tests can monkeypatch them.
    """

    name: str
    summary: str
    run: Callable[[int, bool], object]
    clear: Callable[[], bool]
    cache_path: Callable[[], Path]
    load: Callable[[], object]
    default_rounds: Callable[[], int]


_TARGETS: tuple[_Target, ...] = (
    _Target(
        name="threads",
        summary="gather thread cap",
        run=lambda rounds, persist: autotune.calibrate(
            rounds=rounds, persist=persist, verbose=True
        ),
        clear=lambda: autotune.clear_cached_cap(),
        cache_path=lambda: autotune._cache_path(),
        load=lambda: autotune.load_cached_cap(),
        default_rounds=lambda: autotune._CALIB_ROUNDS,
    ),
    _Target(
        name="prefetch",
        summary="prefetch distances",
        run=lambda rounds, persist: autotune.calibrate_prefetch(
            rounds=rounds, persist=persist, verbose=True
        ),
        clear=lambda: autotune.clear_cached_prefetch(),
        cache_path=lambda: autotune._prefetch_cache_path(),
        load=lambda: autotune.load_cached_prefetch(),
        default_rounds=lambda: autotune._CALIB_ROUNDS,
    ),
    _Target(
        name="mask-density",
        summary="boolean-mask density gate",
        run=lambda rounds, persist: autotune.calibrate_mask_density(
            rounds=rounds, persist=persist, verbose=True
        ),
        clear=lambda: autotune.clear_cached_mask_density(),
        cache_path=lambda: autotune._mask_density_cache_path(),
        load=lambda: autotune.load_cached_mask_density(),
        default_rounds=lambda: autotune._CALIB_ROUNDS,
    ),
)


def _selected(args: argparse.Namespace) -> tuple[_Target, ...]:
    """Targets chosen on the command line, in registry (dependency) order."""
    requested = set(args.targets)
    if not requested:
        return _TARGETS
    return tuple(t for t in _TARGETS if t.name in requested)


def _add_target_positional(parser: argparse.ArgumentParser) -> None:
    names = [t.name for t in _TARGETS]
    parser.add_argument(
        "targets",
        nargs="*",
        choices=[*names, []],  # [] permits the empty default with choices set
        metavar="TARGET",
        help=f"calibration target(s) to act on: {', '.join(names)} (default: all, "
        "always executed in dependency order)",
    )


def _configure_run_parser(parser: argparse.ArgumentParser) -> None:
    _add_target_positional(parser)
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        metavar="N",
        help="interleaved timing rounds for every selected target "
        "(raise on noisy machines); per-target overrides below take precedence",
    )
    for target in _TARGETS:
        parser.add_argument(
            f"--{target.name}-rounds",
            type=int,
            default=None,
            metavar="N",
            help=f"rounds for the {target.summary} sweep " f"(default {target.default_rounds()})",
        )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="apply in-process only; do not write the cache files",
    )
    parser.set_defaults(handler=_cmd_run)


def _rounds_for(target: _Target, args: argparse.Namespace) -> int:
    # argparse normalizes hyphens in option names to underscores.
    per_target = getattr(args, f"{target.name.replace('-', '_')}_rounds")
    if per_target is not None:
        return int(per_target)
    if args.rounds is not None:
        return int(args.rounds)
    return target.default_rounds()


def _cmd_run(args: argparse.Namespace) -> int:
    persist = not args.no_persist
    for target in _selected(args):
        rounds = _rounds_for(target, args)
        print(f"Calibrating {target.summary} (rounds={rounds})...")
        result = target.run(rounds, persist)
        print(f"  -> {result}\n")
    if persist:
        print(f"Calibration cached under: {autotune._cache_dir()}")
    else:
        print("Applied in-process only (--no-persist); nothing was cached.")
    return 0


def _cmd_clear(args: argparse.Namespace) -> int:
    for target in _selected(args):
        state = "removed" if target.clear() else "no cache present"
        print(f"  {target.name}: {state}")
    print("In-process state reset to uncalibrated defaults.")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    print(f"Hardware fingerprint: {autotune._hardware_fingerprint()}")
    for target in _selected(args):
        path = target.cache_path()
        print(f"  {target.name} cache: {path}")
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            print("    (absent)")
            continue
        except (OSError, ValueError) as error:
            print(f"    (unreadable: {error})")
            continue
        matches = payload.get("fingerprint") == autotune._hardware_fingerprint()
        status = "MATCHES this machine" if matches else "DIFFERENT machine -- ignored here"
        print(f"    fingerprint: {status}")
        print(f"    applied value: {target.load()}")
    return 0


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``calibration`` group and the ``calibrate`` alias."""
    group = subparsers.add_parser("calibration", help="manage per-host performance calibration")
    group_sub = group.add_subparsers(dest="calibration_command", required=True)

    run = group_sub.add_parser(
        "run", help="run (or rerun and overwrite) calibration for this machine"
    )
    _configure_run_parser(run)

    show = group_sub.add_parser(
        "show", help="print cache locations, contents, and fingerprint match"
    )
    _add_target_positional(show)
    show.set_defaults(handler=_cmd_show)

    clear = group_sub.add_parser("clear", help="remove this machine's cached calibration")
    _add_target_positional(clear)
    clear.set_defaults(handler=_cmd_clear)

    # Top-level convenience alias for the most common operation. Same
    # arguments, same handler as `calibration run`.
    alias = subparsers.add_parser(
        "calibrate", help="run per-host calibration (alias for 'calibration run')"
    )
    _configure_run_parser(alias)
