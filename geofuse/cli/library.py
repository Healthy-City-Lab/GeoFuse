"""Presets and caches.

::

    geofuse preset list [ENGINE]
    geofuse preset show ENGINE NAME [--json]
    geofuse preset save ENGINE NAME [--from RUN --step STEP] [--cloud-max 20 ...]
    geofuse preset import FILE [--overwrite]
    geofuse preset remove ENGINE NAME
    geofuse cache list [--output-dir DIR ...]
    geofuse cache purge pano | ndvi | fusion | all [--output-dir DIR ...] [--yes]

Presets are the NDVI and GVI tabs' presets: the same built-ins, the same user
library, the same checks. Caches are the ones the GUI's cache panel shows; a
purge never touches job records or results.
"""

from __future__ import annotations

import json
import os
import sys

from geofuse import settings as _settings
from geofuse.cli.commands import (
    EXIT_FAILED,
    EXIT_OK,
    CliError,
    _add_setting_flags,
    _overrides,
    _resolve,
    _state,
)
from geofuse.cli.templates import _value

# ────────────────────────────────────────────────────────────────────
# geofuse preset
# ────────────────────────────────────────────────────────────────────


def _find(engine: str, name: str):
    from geofuse import presets

    entry = presets.find_preset(engine, name)
    if entry is None:
        known = ", ".join(e.name for e in presets.list_presets(engine))
        raise CliError(f"No {engine.upper()} preset named {name!r} (known: {known}).")
    return entry


def _saved_settings(args) -> dict:
    """The settings ``preset save`` stores: a run-file step's, then flags."""
    values: dict = {}
    if args.source:
        from geofuse.runfile import RunFileError, load

        try:
            run = load(args.source)
        except (RunFileError, _settings.SettingError) as exc:
            raise CliError(str(exc)) from exc
        steps = [s for s in run.steps if s.engine == args.engine]
        if args.step:
            steps = [s for s in steps if args.step in (s.key, s.name)]
        if len(steps) != 1:
            names = ", ".join(s.key for s in run.steps if s.engine == args.engine)
            raise CliError(
                f"Name one {args.engine.upper()} step with --step (the run file "
                f"has: {names or 'none'})."
            )
        step = steps[0]
        values = {k: v for k, v in step.settings.items() if k in step.explicit}
    schema = _settings.SCHEMAS[args.engine]
    for override in _overrides(args):
        if override.key not in schema:
            flag = "--" + override.key.replace("_", "-")
            raise CliError(f"{flag} is not a {args.engine.upper()} setting.")
        values[override.key] = override.value
    if not values:
        raise CliError("Nothing to save: give --from RUN --step STEP or setting flags.")
    return values


def cmd_preset(args) -> int:
    from geofuse import presets

    try:
        if args.action == "list":
            for engine in [args.engine] if args.engine else list(presets.ENGINES):
                print(f"{engine.upper()} presets")
                for e in presets.list_presets(engine):
                    kind = "built-in" if e.builtin else "yours"
                    print(f"  {e.name:<42} {kind:<9} {e.description}")
            print(f"\nYour presets are kept in {presets.USER_DIR}.")
            return EXIT_OK
        if args.action == "show":
            entry = _find(args.engine, args.name)
            doc = presets.read_preset(entry.path)
            if args.json:
                print(json.dumps(doc, indent=2, ensure_ascii=False))
                return EXIT_OK
            settings, _ = presets.validate(doc, args.engine)
            print(f"# {entry.name}: {doc.get('description') or ''}".rstrip(": "))
            print(f"# {entry.path}")
            print(f'preset = "{entry.name}"   # or, in a step, the settings:')
            for key, value in settings.items():
                print(f"{key} = {_value(value)}")
            return EXIT_OK
        if args.action == "save":
            doc = presets.make_preset(
                args.engine, args.name, args.description, _saved_settings(args)
            )
            path = presets.save_user_preset(doc, overwrite=args.overwrite)
            print(f"Saved {args.engine.upper()} preset {args.name!r} to {path}.")
            return EXIT_OK
        if args.action == "import":
            doc = presets.read_preset(args.file)
            engine = str(doc.get("engine"))
            presets.validate(doc, engine)
            path = presets.save_user_preset(doc, overwrite=args.overwrite)
            print(f"Imported {engine.upper()} preset {doc.get('name')!r} to {path}.")
            return EXIT_OK
        entry = _find(args.engine, args.name)
        if entry.builtin:
            raise CliError(f"{args.name!r} is a built-in preset; it cannot be removed.")
        os.remove(entry.path)
        print(f"Removed {args.engine.upper()} preset {args.name!r}.")
        return EXIT_OK
    except presets.PresetExists as exc:
        raise CliError(f"{exc} Add --overwrite to replace it.") from exc
    except presets.PresetError as exc:
        raise CliError(str(exc)) from exc


# ────────────────────────────────────────────────────────────────────
# geofuse cache
# ────────────────────────────────────────────────────────────────────

_CACHE_KINDS = {"pano": ("gvi", "gvi_column"), "ndvi": ("ndvi", "ndvi_column")}


def _fusion_output_dirs(args) -> list[str]:
    """The output folders whose fusion caches to show: those named, else the
    GUI's and every one a command-line fusion job wrote to."""
    from geofuse.persistence import caches

    if args.output_dir:
        return [os.path.abspath(d) for d in args.output_dir]
    found = [caches.GUI_OUTPUT_DIR]
    for job in _state(args).jobs():
        out = job.spec.get("output_dir")
        if job.spec.get("kind") == "fusion" and out:
            found.append(os.path.abspath(out))
    return list(dict.fromkeys(found))


def _adopt(args) -> int:
    """Record the NDVI tile workspaces earlier jobs cached before workspaces
    carried records, so runs over a smaller area reuse their tiles."""
    from geofuse.jobs.plans import JobSpec, PlanError, build
    from geofuse.jobs.runners import record_ndvi_cache

    state = _state(args)
    fields = JobSpec.__dataclass_fields__
    for job_id in _resolve(state, args.jobs):
        raw = state.read_spec(job_id) or {}
        label = f"{job_id} ({raw.get('name')})"
        if raw.get("kind") not in ("ndvi", "ndvi_column"):
            print(f"{label}: not an NDVI job; skipped.")
            continue
        spec = JobSpec.from_dict({k: v for k, v in raw.items() if k in fields})
        try:
            sub = build(spec)
            counts = record_ndvi_cache(**sub.kwargs)
        except (PlanError, OSError, ValueError) as exc:
            raise CliError(f"{label}: {exc}") from exc
        summary = ", ".join(f"{n} {what}" for what, n in sorted(counts.items()))
        print(f"{label}: {summary or 'nothing cached'}")
    return EXIT_OK


def cmd_cache(args) -> int:
    from geofuse.persistence import caches

    if args.action == "adopt":
        return _adopt(args)
    dirs = _fusion_output_dirs(args)
    if args.action == "list":
        for key in ("pano", "ndvi"):
            where = caches.pano_db() if key == "pano" else caches.ndvi_tiles()
            print(f"{caches.CACHE_LABELS[key]:<30} {caches.describe_cache(key)}")
            print(f"{'':<30} {where}")
        for folder in dirs:
            line = caches.describe_cache("fusion", output_dir=folder)
            print(f"{caches.CACHE_LABELS['fusion']:<30} {line}")
            print(f"{'':<30} {caches.fusion_cache_dir(folder)}")
        return EXIT_OK

    keys = list(caches.CACHE_LABELS) if args.cache == "all" else [args.cache]
    busy = [
        j
        for j in _state(args).jobs()
        if j.state in ("running", "waiting")
        and any(j.spec.get("kind") in _CACHE_KINDS.get(k, ("fusion",)) for k in keys)
    ]
    if busy and not args.force:
        names = ", ".join(f"{j.id} ({j.name})" for j in busy)
        raise CliError(f"Jobs using this cache are running: {names}. Add --force.")
    targets = [k for k in keys if k != "fusion"] + (
        [("fusion", d) for d in dirs] if "fusion" in keys else []
    )
    shown = [
        caches.CACHE_LABELS[t] if isinstance(t, str) else caches.fusion_cache_dir(t[1])
        for t in targets
    ]
    if not args.yes:
        if not sys.stdin.isatty():
            raise CliError("Add --yes to purge without a prompt.")
        try:
            answer = input(f"Permanently delete: {'; '.join(shown)}? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("Nothing deleted.")
            return EXIT_FAILED
    for target in targets:
        if isinstance(target, str):
            print(f"{caches.CACHE_LABELS[target]}: {caches.purge_cache(target)}")
        else:
            folder = target[1]
            message = caches.purge_cache("fusion", output_dir=folder)
            print(f"{caches.fusion_cache_dir(folder)}: {message}")
    return EXIT_OK


# ────────────────────────────────────────────────────────────────────
# Parser
# ────────────────────────────────────────────────────────────────────


def add_parsers(sub, common) -> None:
    """This module's commands on ``sub``; ``common`` holds the options every
    command takes."""
    from geofuse.presets import ENGINES

    p = sub.add_parser("preset", help="the NDVI and GVI tabs' presets")
    actions = p.add_subparsers(dest="action", metavar="ACTION", required=True)
    a = actions.add_parser("list", parents=[common], help="built-in and yours")
    a.add_argument("engine", nargs="?", choices=ENGINES)
    a = actions.add_parser("show", parents=[common], help="a preset's settings")
    a.add_argument("engine", choices=ENGINES)
    a.add_argument("name")
    a.add_argument("--json", action="store_true", help="the whole preset file")
    a = actions.add_parser(
        "save", parents=[common], help="save settings as one of your presets"
    )
    a.add_argument("engine", choices=ENGINES)
    a.add_argument("name")
    a.add_argument("--description", default="")
    a.add_argument("--from", dest="source", metavar="RUN", help="a run file")
    a.add_argument("--step", help="the step of RUN whose settings to save")
    a.add_argument("--overwrite", action="store_true")
    a.set_defaults(overrides=[])
    _add_setting_flags(a)
    a = actions.add_parser("import", parents=[common], help="add a preset file")
    a.add_argument("file")
    a.add_argument("--overwrite", action="store_true")
    a = actions.add_parser("remove", parents=[common], help="delete one of yours")
    a.add_argument("engine", choices=ENGINES)
    a.add_argument("name")
    p.set_defaults(func=cmd_preset)

    p = sub.add_parser("cache", help="the panorama, NDVI tile and fusion caches")
    actions = p.add_subparsers(dest="action", metavar="ACTION", required=True)
    for name, text in (("list", "what each cache holds"), ("purge", "empty caches")):
        a = actions.add_parser(name, parents=[common], help=text)
        if name == "purge":
            a.add_argument("cache", choices=["pano", "ndvi", "fusion", "all"])
            a.add_argument("--yes", action="store_true", help="do not ask")
            a.add_argument(
                "--force", action="store_true", help="even while jobs use it"
            )
        a.add_argument(
            "--output-dir",
            action="append",
            default=[],
            help="whose fusion cache (repeatable; default: the GUI's and every "
            "command-line fusion job's)",
        )
    a = actions.add_parser(
        "adopt",
        parents=[common],
        help="let runs over a smaller area reuse NDVI tiles these jobs cached",
    )
    a.add_argument("jobs", nargs="+", metavar="JOB")
    p.set_defaults(func=cmd_cache)
