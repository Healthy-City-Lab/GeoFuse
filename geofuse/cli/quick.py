"""One engine without a run file.

::

    geofuse ndvi AREA... --range 2019-05-01:2019-09-30 [--cloud-max 20 ...]
    geofuse gvi AREA... [--capture-mode column --year-column year ...]
    geofuse fusion --target COHORT.gpkg --outcome cesd --ndvi N.tif --gvi G.gpkg

Each builds the one-step run file it stands for, then submits it to the shared
supervisor (``--foreground`` runs it here; ``--plan`` only shows it). The
settings are the same flags the run-file commands take, and ``--save RUN``
writes the run file out, to rerun it or to grow it into a pipeline.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from geofuse import settings as _settings
from geofuse.cli.commands import (
    CliError,
    _add_setting_flags,
    _overrides,
    run_here,
    show_plan,
    submit,
)

# ────────────────────────────────────────────────────────────────────
# Building the run document
# ────────────────────────────────────────────────────────────────────


def _step_name(path: str, fallback: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(path).stem).strip("_-")
    return name or fallback


def _absolute(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _range(text: str) -> list[str]:
    start, sep, end = text.partition(":")
    if not sep:
        raise CliError(f"--range takes START:END in YYYY-MM-DD, not {text!r}.")
    return [start.strip(), end.strip()]


def _channel(values: list[str], flag: str):
    """``--ndvi`` / ``--gvi``: one file, files matched to years by name, or
    ``YEAR=FILE`` pairs."""
    pairs = [v.partition("=") for v in values]
    if all(sep for _, sep, _ in pairs):
        return {year: _absolute(path) for year, _, path in pairs}
    if any(sep for _, sep, _ in pairs):
        raise CliError(f"{flag}: give every file as YEAR=FILE, or none of them.")
    files = [_absolute(v) for v in values]
    return files[0] if len(files) == 1 else files


_FUSION_OPTIONS = (
    "layer",
    "mode",
    "date_column",
    "entity_column",
    "wave_column",
    "area_column",
    "assign_by",
)
_FUSION_COLUMNS = (
    ("outcomes", "outcome", "an outcome column"),
    ("negative_controls", "control", "a negative-control column"),
    ("covariates", "covariate", "a covariate column"),
    ("categorical", "categorical", "a categorical covariate column"),
    ("moderators", "moderator", "an effect-modifier column"),
)


def _step(args) -> dict:
    engine = args.command
    if engine in ("ndvi", "gvi"):
        step: dict = {
            "name": args.name or _step_name(args.study_area[0], engine),
            "study_area": [_absolute(p) for p in args.study_area],
        }
        if args.merge:
            step["merge"] = True
        if getattr(args, "shards", None):
            step["shards"] = args.shards
        if args.year_column:
            step["year_column"] = args.year_column
        if engine == "ndvi":
            if args.range:
                step["ranges"] = [_range(r) for r in args.range]
            if args.date:
                step["specific_dates"] = list(args.date)
    else:
        if bool(args.target) == bool(args.wave):
            raise CliError("Give --target, or a --wave per wave file (two or more).")
        step = {"name": args.name or _step_name(args.target or "fusion", "fusion")}
        if args.target:
            step["target"] = _absolute(args.target)
        else:
            step["waves"] = []
            for i, wave in enumerate(args.wave):
                label, sep, path = wave.partition("=")
                if not sep:
                    label, path = f"wave{i + 1}", wave
                step["waves"].append({"file": _absolute(path), "label": label})
        for key in _FUSION_OPTIONS:
            value = getattr(args, key)
            if value is not None:
                step[key] = value
        if args.band != 1:
            step["band"] = args.band
        for key, flag, _help in _FUSION_COLUMNS:
            if getattr(args, flag):
                step[key] = list(getattr(args, flag))
        step["ndvi"] = _channel(args.ndvi, "--ndvi")
        step["gvi"] = _channel(args.gvi, "--gvi")
    if args.preset:
        step["preset"] = args.preset
    for override in _overrides(args):
        step[override.key] = override.value
    return step


def document(args) -> dict:
    """The run document a shortcut command stands for."""
    step = _step(args)
    return {
        "format": "geofuse-run",
        "version": 1,
        "name": step["name"],
        "output_dir": _absolute(args.output_dir),
        args.command: [step],
    }


def _toml(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    if isinstance(value, dict):
        body = ", ".join(f"{json.dumps(str(k))} = {_toml(v)}" for k, v in value.items())
        return "{ " + body + " }"
    raise CliError(f"Cannot write {value!r} to a run file.")


def dumps(doc: dict) -> str:
    """``doc`` as a run file."""
    lines = [
        f"{key} = {_toml(value)}"
        for key, value in doc.items()
        if not isinstance(value, list)
    ]
    for engine in ("ndvi", "gvi", "fusion"):
        for step in doc.get(engine) or []:
            lines += ["", f"[[{engine}]]"]
            for key, value in step.items():
                if key != "waves":
                    lines.append(f"{key} = {_toml(value)}")
            for wave in step.get("waves") or []:
                lines += ["", f"[[{engine}.waves]]"]
                lines += [f"{k} = {_toml(v)}" for k, v in wave.items()]
    return "\n".join(lines) + "\n"


# ────────────────────────────────────────────────────────────────────
# Command
# ────────────────────────────────────────────────────────────────────


def cmd_quick(args) -> int:
    from geofuse.jobs.plans import PlanError, expand
    from geofuse.runfile import RunFileError, from_document

    doc = document(args)
    try:
        run = from_document(doc, base_dir=os.getcwd(), stem=doc["name"])
        specs = expand(run)
    except (RunFileError, PlanError, _settings.SettingError) as exc:
        raise CliError(str(exc)) from exc
    if args.save:
        Path(args.save).write_text(dumps(doc), encoding="utf-8")
        run.path = _absolute(args.save)
        print(f"Wrote {args.save}")
    if args.plan:
        return show_plan(run, specs)
    if args.foreground:
        return run_here(args, run, specs)
    return submit(args, run, specs)


def _add_fusion_inputs(p) -> None:
    p.add_argument("--target", help="the cohort or area file")
    p.add_argument(
        "--wave",
        action="append",
        metavar="[LABEL=]FILE",
        help="one target file per wave (longitudinal; repeatable)",
    )
    p.add_argument("--layer", help="the target's layer")
    p.add_argument("--band", type=int, default=1, help="a raster target's band")
    for _key, flag, text in _FUSION_COLUMNS:
        p.add_argument(f"--{flag}", action="append", help=f"{text} (repeatable)")
    p.add_argument(
        "--ndvi",
        nargs="+",
        required=True,
        metavar="FILE",
        help="NDVI file(s): one, several matched to years by name, or YEAR=FILE",
    )
    p.add_argument(
        "--gvi", nargs="+", required=True, metavar="FILE", help="GVI file(s), as --ndvi"
    )
    p.add_argument("--mode", choices=["cross-sectional", "year-aware", "longitudinal"])
    for column in ("date", "entity", "wave", "area"):
        p.add_argument(f"--{column}-column")
    p.add_argument("--assign-by", choices=["year", "wave"])


def add_parsers(sub, common) -> None:
    """This module's commands on ``sub`` (which adds ``common`` itself)."""
    for engine, text in (
        ("ndvi", "download NDVI for study areas (no run file needed)"),
        ("gvi", "compute GVI for study areas (no run file needed)"),
        ("fusion", "fit a composite index (no run file needed)"),
    ):
        p = sub.add_parser(engine, help=text)
        if engine == "fusion":
            _add_fusion_inputs(p)
        else:
            p.add_argument("study_area", nargs="+", metavar="AREA", help="vector files")
            p.add_argument(
                "--merge", action="store_true", help="treat the files as one area"
            )
            p.add_argument("--year-column", help="the column of per-row years")
        if engine == "gvi":
            p.add_argument(
                "--shards", type=int, metavar="N", help="split into N jobs + a merge"
            )
        if engine == "ndvi":
            p.add_argument(
                "--range", action="append", metavar="START:END", help="a date range"
            )
            p.add_argument(
                "--date", action="append", metavar="YYYY-MM-DD", help="a single date"
            )
        p.add_argument("--name", help="the step and run name")
        p.add_argument("--preset", help="a preset name or file")
        p.add_argument(
            "-o", "--output-dir", default="output_results", help="where outputs go"
        )
        p.add_argument("--save", metavar="RUN.toml", help="also write the run file")
        how = p.add_mutually_exclusive_group()
        how.add_argument("--plan", action="store_true", help="only show the jobs")
        how.add_argument(
            "--foreground", action="store_true", help="run here until it ends"
        )
        p.set_defaults(func=cmd_quick, overrides=[], only=[])
        _add_setting_flags(p, (engine,), f"{engine.upper()} settings")
