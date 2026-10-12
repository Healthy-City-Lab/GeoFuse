"""Run files: what the command line runs, written in TOML (or JSON).

A run file lists steps — NDVI downloads, GVI runs and fusion fits — each with
its inputs and settings::

    format = "geofuse-run"
    version = 1
    output_dir = "results"

    [defaults.ndvi]                  # every [[ndvi]] step starts from these
    preset = "Landsat TOA, all scenes, water-masked"

    [[ndvi]]
    name = "city"
    study_area = "data/city.gpkg"
    year_column = "year"
    cloud_max = 20

    [[fusion]]
    name = "cesd"
    target = "data/cohort.gpkg"
    outcomes = ["cesd"]
    mode = "year-aware"
    date_column = "interview_date"
    ndvi = "@ndvi.city"              # the output of the step above

A step's settings resolve in a fixed order — the schema default (the GUI's,
:mod:`geofuse.settings`), its preset, the file's ``[defaults.<engine>]``, the
step itself, then command-line overrides — and are checked against the
schema with the offending key named. Unknown keys are refused rather than
ignored, so a typo cannot silently fall back to a default. Relative paths
resolve against the run file's folder; credentials are refused anywhere in
the file.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from . import presets as _presets
from . import settings as _settings

FORMAT = "geofuse-run"
VERSION = 1
ENGINES = ("ndvi", "gvi", "fusion")
FUSION_MODES = ("cross-sectional", "year-aware", "longitudinal")
ASSIGN_BY = ("year", "wave")
CHANNEL_INPUTS = ("ndvi", "gvi")

_STEP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_REFERENCE = re.compile(
    r"^@(?P<engine>ndvi|gvi)\.(?P<step>[A-Za-z0-9][A-Za-z0-9_-]*)"
    r"(?:\[(?P<statistic>[a-z]+)\])?$"
)


class RunFileError(ValueError):
    """A run file that cannot be used, with the reason."""


# ────────────────────────────────────────────────────────────────────
# Step inputs (everything a step names that is not a setting)
# ────────────────────────────────────────────────────────────────────

_COMMON_INPUTS = ("name", "preset", "output_dir", "depends_on")
INPUTS: dict[str, tuple[str, ...]] = {
    "ndvi": _COMMON_INPUTS
    + ("study_area", "merge", "ranges", "specific_dates", "year_column"),
    "gvi": _COMMON_INPUTS + ("study_area", "merge", "year_column", "shards"),
    "fusion": _COMMON_INPUTS
    + (
        "target",
        "layer",
        "band",
        "waves",
        "outcomes",
        "negative_controls",
        "covariates",
        "categorical",
        "moderators",
        "mode",
        "date_column",
        "entity_column",
        "wave_column",
        "area_column",
        "assign_by",
        "ndvi",
        "gvi",
    ),
}
_WAVE_KEYS = ("file", "label", "entity_column", "date_column")


@dataclass(frozen=True)
class Reference:
    """``@<engine>.<step>[<statistic>]``: the output of an earlier step."""

    engine: str
    step: str
    statistic: str | None = None

    @property
    def key(self) -> str:
        return f"{self.engine}.{self.step}"

    def __str__(self) -> str:
        stat = f"[{self.statistic}]" if self.statistic else ""
        return f"@{self.key}{stat}"


@dataclass
class Step:
    """One step of a run file, its inputs checked and its settings resolved."""

    engine: str
    name: str
    inputs: dict[str, Any]
    settings: dict[str, Any]
    output_dir: str
    preset: str | None = None
    #: Settings given explicitly (preset, file or command line), not defaulted.
    explicit: set[str] = field(default_factory=set)
    depends_on: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.engine}.{self.name}"


@dataclass
class RunFile:
    """A checked run file: its steps in an order that respects dependencies."""

    name: str
    output_dir: str
    base_dir: str
    steps: list[Step]
    path: str | None = None

    def step(self, key: str) -> Step:
        for s in self.steps:
            if s.key == key:
                return s
        raise KeyError(key)


# ────────────────────────────────────────────────────────────────────
# Command-line overrides
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Override:
    """A command-line value for a run-file key.

    ``target`` is ``()`` for a top-level key, ``(engine,)`` for every step of
    an engine, ``(engine, step)`` for one step, or ``("*",)`` for every step
    whose engine has the key.
    """

    target: tuple[str, ...]
    key: str
    value: Any


def parse_value(text: str) -> Any:
    """A command-line value read as TOML (``20``, ``true``, ``["a"]``,
    ``2019-05-01``), or as plain text when it is not valid TOML."""
    try:
        return _plain(tomllib.loads(f"v = {text}")["v"])
    except tomllib.TOMLDecodeError:
        return text


def parse_override(spec: str) -> Override:
    """``KEY=VALUE`` with a dotted key: ``output_dir=out``,
    ``ndvi.cloud_max=20`` (every NDVI step) or ``ndvi.city.cloud_max=20``."""
    key, sep, text = spec.partition("=")
    if not sep or not key.strip():
        raise RunFileError(f"--set needs KEY=VALUE, not {spec!r}.")
    parts = tuple(p.strip() for p in key.strip().split("."))
    if len(parts) == 1:
        target: tuple[str, ...] = ()
    elif len(parts) in (2, 3) and parts[0] in ENGINES:
        target = parts[:-1]
    else:
        raise RunFileError(
            f"--set key {key!r} must be KEY, ENGINE.KEY or ENGINE.STEP.KEY "
            f"with ENGINE one of {list(ENGINES)}."
        )
    return Override(target, parts[-1], parse_value(text))


# ────────────────────────────────────────────────────────────────────
# Loading
# ────────────────────────────────────────────────────────────────────


def _plain(value: Any) -> Any:
    """TOML / JSON values with dates as ISO strings, recursively."""
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def read_document(path: str | os.PathLike) -> dict:
    """A run file's raw document (TOML, or JSON for a ``.json`` file)."""
    p = Path(path)
    try:
        if p.suffix.lower() == ".json":
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
        else:
            with open(p, "rb") as f:
                doc = tomllib.load(f)
    except OSError as exc:
        raise RunFileError(f"Could not read the run file: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise RunFileError(f"The run file is not valid TOML: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RunFileError(
            f"The run file is not valid JSON ({exc.msg}, line {exc.lineno})."
        ) from exc
    if not isinstance(doc, dict):
        raise RunFileError("A run file holds a table of keys at its top level.")
    return _plain(doc)


def load(
    path: str | os.PathLike, overrides: Iterable[Override | str] = ()
) -> RunFile:
    """Read, override, check and order the run file at ``path``."""
    p = Path(path).expanduser().resolve()
    run = from_document(
        read_document(p), base_dir=str(p.parent), overrides=overrides, stem=p.stem
    )
    run.path = str(p)
    return run


def from_document(
    doc: Mapping[str, Any],
    *,
    base_dir: str | os.PathLike = ".",
    overrides: Iterable[Override | str] = (),
    stem: str = "run",
) -> RunFile:
    """Check a run-file document (already parsed) and resolve its steps."""
    doc = _plain(dict(doc))
    secret = _presets._find_secret(doc)
    if secret:
        raise RunFileError(
            f"The run file contains a credential ('{secret}'). Run files never "
            "carry API keys or passwords; store them with `geofuse credentials`."
        )
    if doc.get("format", FORMAT) != FORMAT:
        raise RunFileError(f"'format' must be {FORMAT!r}, not {doc.get('format')!r}.")
    version = doc.get("version", VERSION)
    if not isinstance(version, int) or version > VERSION:
        raise RunFileError(
            f"Run file version {version!r} is not supported (this GeoFuse reads "
            f"version {VERSION})."
        )
    known = {"format", "version", "name", "output_dir", "defaults", *ENGINES}
    unknown = sorted(set(doc) - known)
    if unknown:
        raise RunFileError(
            f"Unknown top-level key(s) {unknown}; expected {sorted(known)}."
        )

    overrides = [o if isinstance(o, Override) else parse_override(o) for o in overrides]
    raw_steps = _raw_steps(doc)
    _apply_overrides(doc, raw_steps, overrides)

    base = Path(base_dir).expanduser().resolve()
    run_name = str(doc.get("name") or stem)
    output_dir = _path(doc.get("output_dir") or "output_results", base, "output_dir")
    defaults = doc.get("defaults") or {}
    if not isinstance(defaults, dict) or set(defaults) - set(ENGINES):
        raise RunFileError(
            f"'defaults' holds one table per engine ({list(ENGINES)}), "
            f"not {sorted(defaults) if isinstance(defaults, dict) else defaults!r}."
        )

    steps = [
        _resolve_step(engine, raw, defaults.get(engine) or {}, base, output_dir)
        for engine, raw in raw_steps
    ]
    keys = [s.key for s in steps]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise RunFileError(f"Step names must be unique per engine; repeated: {dupes}.")
    if not steps:
        raise RunFileError("The run file has no [[ndvi]], [[gvi]] or [[fusion]] step.")
    _link(steps)
    return RunFile(
        name=run_name, output_dir=output_dir, base_dir=str(base), steps=_order(steps)
    )


def _raw_steps(doc: Mapping[str, Any]) -> list[tuple[str, dict]]:
    out = []
    for engine in ENGINES:
        tables = doc.get(engine) or []
        if isinstance(tables, dict):
            tables = [tables]
        if not isinstance(tables, list) or not all(isinstance(t, dict) for t in tables):
            raise RunFileError(
                f"'{engine}' must be a list of tables ([[{engine}]] in TOML)."
            )
        for i, table in enumerate(tables):
            raw = dict(table)
            raw.setdefault("name", f"{engine}{i + 1}")
            if not isinstance(raw["name"], str) or not _STEP_NAME.match(raw["name"]):
                raise RunFileError(
                    f"[[{engine}]] step names use letters, digits, '_' and '-', "
                    f"not {raw['name']!r}."
                )
            out.append((engine, raw))
    return out


def _apply_overrides(
    doc: dict, raw_steps: list[tuple[str, dict]], overrides: Sequence[Override]
) -> None:
    for o in overrides:
        if o.target == ():
            if o.key not in ("name", "output_dir"):
                raise RunFileError(
                    f"--set {o.key}: only 'name' and 'output_dir' are top-level; "
                    "use ENGINE.KEY or ENGINE.STEP.KEY for step keys."
                )
            doc[o.key] = o.value
            continue
        engine = o.target[0]
        hits = [
            raw
            for eng, raw in raw_steps
            if (engine == "*" or eng == engine)
            and (len(o.target) == 1 or raw["name"] == o.target[1])
            and (engine != "*" or _knows(eng, o.key))
        ]
        if not hits:
            where = ".".join(o.target)
            raise RunFileError(f"--set {where}.{o.key}: no matching step.")
        for raw in hits:
            raw[o.key] = o.value


def _knows(engine: str, key: str) -> bool:
    return key in INPUTS[engine] or key in _settings.SCHEMAS[engine]


# ────────────────────────────────────────────────────────────────────
# Steps
# ────────────────────────────────────────────────────────────────────


def _path(value: Any, base: Path, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunFileError(f"'{key}' must be a path, not {value!r}.")
    p = Path(os.path.expanduser(value))
    return str(p if p.is_absolute() else (base / p).resolve())


def _paths(value: Any, base: Path, key: str) -> list[str]:
    items = value if isinstance(value, list) else [value]
    if not items:
        raise RunFileError(f"'{key}' needs at least one path.")
    return [_path(v, base, key) for v in items]


def _strings(value: Any, key: str) -> list[str]:
    items = value if isinstance(value, list) else [value]
    if not all(isinstance(v, str) and v for v in items):
        raise RunFileError(f"'{key}' must be a list of column names, not {value!r}.")
    return list(items)


def _iso(value: Any, key: str) -> str:
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError:
        raise RunFileError(
            f"'{key}' needs dates as YYYY-MM-DD, not {value!r}."
        ) from None


def _resolve_settings(
    engine: str, raw: dict, file_defaults: dict, base: Path
) -> tuple[dict, str | None, set[str]]:
    schema = _settings.SCHEMAS[engine]
    values = _settings.defaults(engine)
    explicit: set[str] = set()
    preset = raw.get("preset", file_defaults.get("preset"))
    if preset is not None:
        if engine not in _presets.ENGINES:
            raise RunFileError(f"[[{engine}]] steps have no presets.")
        loaded = _load_preset(engine, preset, base)
        values.update(loaded)
        explicit.update(loaded)
    for source, label in ((file_defaults, f"defaults.{engine}"), (raw, raw["name"])):
        for key, value in source.items():
            if key in INPUTS[engine]:
                continue
            if key not in schema:
                raise RunFileError(
                    f"Unknown key '{key}' in [{label}] ({engine.upper()} inputs: "
                    f"{', '.join(INPUTS[engine])}; settings: see `geofuse config "
                    f"template {engine}`)."
                )
            values[key] = value
            explicit.add(key)
    try:
        return _settings.check(engine, values), preset, explicit
    except _settings.SettingError as exc:
        raise RunFileError(f"[[{engine}]] {raw['name']}: {exc}") from exc


def _load_preset(engine: str, preset: Any, base: Path) -> dict:
    if not isinstance(preset, str) or not preset:
        raise RunFileError(f"'preset' must be a preset name or a path, not {preset!r}.")
    try:
        if preset.lower().endswith(".json") or "/" in preset or "\\" in preset:
            doc = _presets.read_preset(_path(preset, base, "preset"))
            settings, _warnings = _presets.validate(doc, engine)
            return settings
        return _presets.load_settings(engine, preset)
    except _presets.PresetError as exc:
        raise RunFileError(f"Preset {preset!r}: {exc}") from exc


def _resolve_step(
    engine: str, raw: dict, file_defaults: dict, base: Path, run_output: str
) -> Step:
    settings, preset, explicit = _resolve_settings(engine, raw, file_defaults, base)
    name = raw["name"]
    output_dir = (
        _path(raw["output_dir"], base, "output_dir")
        if raw.get("output_dir")
        else run_output
    )
    checker = {"ndvi": _ndvi_inputs, "gvi": _gvi_inputs, "fusion": _fusion_inputs}
    try:
        inputs = checker[engine](raw, settings, base)
    except RunFileError as exc:
        raise RunFileError(f"[[{engine}]] {name}: {exc}") from None
    depends = _strings(raw.get("depends_on") or [], "depends_on")
    if engine == "fusion":
        _fusion_metric(name, inputs["mode"], settings, explicit)
    preset_name = Path(preset).stem if preset and preset.endswith(".json") else preset
    return Step(
        engine=engine,
        name=name,
        inputs=inputs,
        settings=settings,
        output_dir=output_dir,
        preset=preset_name,
        explicit=explicit,
        depends_on=depends,
    )


def _fusion_metric(name: str, mode: str, settings: dict, explicit: set[str]) -> None:
    """Default a longitudinal run to its own metric, then match metric to mode."""
    longitudinal = mode == "longitudinal"
    if longitudinal and "objective_metric" not in explicit:
        settings["objective_metric"] = _settings.LONGITUDINAL_DEFAULT_METRIC
    allowed = (
        _settings.LONGITUDINAL_METRICS
        if longitudinal
        else _settings.CROSS_SECTIONAL_METRICS
    )
    if settings["objective_metric"] not in allowed:
        raise RunFileError(
            f"[[fusion]] {name}: a {mode} run scores with one of {list(allowed)}, "
            f"not {settings['objective_metric']!r}."
        )


def _study_area(raw: dict, base: Path) -> dict:
    if "study_area" not in raw:
        raise RunFileError("'study_area' is required (a path or a list of paths).")
    merge = raw.get("merge", False)
    if not isinstance(merge, bool):
        raise RunFileError(f"'merge' must be true or false, not {merge!r}.")
    return {"study_area": _paths(raw["study_area"], base, "study_area"), "merge": merge}


def _ndvi_inputs(raw: dict, settings: dict, base: Path) -> dict:
    out = _study_area(raw, base)
    ranges = []
    for item in raw.get("ranges") or []:
        if not isinstance(item, list) or len(item) != 2:
            raise RunFileError(
                f"'ranges' holds [start, end] pairs of dates, not {item!r}."
            )
        start, end = _iso(item[0], "ranges"), _iso(item[1], "ranges")
        if start > end:
            raise RunFileError(f"Range {start} → {end} ends before it starts.")
        ranges.append([start, end])
    specific = [_iso(d, "specific_dates") for d in raw.get("specific_dates") or []]
    column = raw.get("year_column")
    if column is not None and (not isinstance(column, str) or not column):
        raise RunFileError(f"'year_column' must be a column name, not {column!r}.")
    if not (ranges or specific or column):
        raise RunFileError(
            "Give at least one date mode: 'ranges', 'specific_dates' or 'year_column'."
        )
    return {**out, "ranges": ranges, "specific_dates": specific, "year_column": column}


def _gvi_inputs(raw: dict, settings: dict, base: Path) -> dict:
    out = _study_area(raw, base)
    column = raw.get("year_column")
    if settings["capture_mode"] == "column":
        if not isinstance(column, str) or not column:
            raise RunFileError("capture_mode 'column' needs 'year_column'.")
    elif column is not None:
        raise RunFileError("'year_column' is only read when capture_mode is 'column'.")
    shards = raw.get("shards", 1)
    if isinstance(shards, bool) or not isinstance(shards, int) or shards < 1:
        raise RunFileError(f"'shards' must be a whole number from 1, not {shards!r}.")
    return {**out, "year_column": column, "shards": shards}


def parse_reference(value: Any) -> Reference | None:
    """The :class:`Reference` a channel source names, or ``None``."""
    if not isinstance(value, str) or not value.startswith("@"):
        return None
    m = _REFERENCE.match(value)
    if not m:
        raise RunFileError(
            f"{value!r} is not a step reference; write @ndvi.STEP, "
            "@ndvi.STEP[statistic] or @gvi.STEP."
        )
    stat = m.group("statistic")
    if stat is not None and (
        m.group("engine") != "ndvi" or stat not in _settings.TEMPORAL_REDUCERS
    ):
        raise RunFileError(
            f"{value!r}: only NDVI references take a statistic, one of "
            f"{list(_settings.TEMPORAL_REDUCERS)}."
        )
    return Reference(m.group("engine"), m.group("step"), stat)


def _channel_source(value: Any, key: str, base: Path) -> Any:
    """A channel's metric files: one path, ``{year: path}``, a list of paths
    (matched to years by filename), or a reference to an earlier step."""
    ref = parse_reference(value)
    if ref is not None:
        if ref.engine != key:
            raise RunFileError(f"'{key}' must reference a [[{key}]] step, not {ref}.")
        return ref
    if isinstance(value, str):
        return _path(value, base, key)
    if isinstance(value, list):
        return _paths(value, base, key)
    if isinstance(value, dict) and value:
        return {str(k): _path(v, base, key) for k, v in value.items()}
    raise RunFileError(
        f"'{key}' must be a path, a list of paths, a {{year = path}} table or "
        f"a step reference, not {value!r}."
    )


def _fusion_inputs(raw: dict, settings: dict, base: Path) -> dict:
    mode = raw.get("mode", "cross-sectional")
    if mode not in FUSION_MODES:
        raise RunFileError(f"'mode' must be one of {list(FUSION_MODES)}, not {mode!r}.")
    target, waves = raw.get("target"), raw.get("waves")
    if (target is None) == (waves is None):
        raise RunFileError("Give exactly one of 'target' or 'waves'.")
    out: dict[str, Any] = {"mode": mode, "target": None, "waves": []}
    if waves is not None:
        if mode != "longitudinal":
            raise RunFileError("'waves' (one target file per wave) is longitudinal.")
        if not isinstance(waves, list) or len(waves) < 2:
            raise RunFileError("'waves' needs a table per wave, at least two.")
        for i, w in enumerate(waves):
            if not isinstance(w, dict) or "file" not in w:
                raise RunFileError(f"waves[{i}] needs at least 'file'.")
            extra = sorted(set(w) - set(_WAVE_KEYS))
            if extra:
                raise RunFileError(f"waves[{i}]: unknown key(s) {extra}.")
            out["waves"].append(
                {
                    "file": _path(w["file"], base, f"waves[{i}].file"),
                    "label": str(w.get("label") or f"wave{i + 1}"),
                    "entity_column": w.get("entity_column") or raw.get("entity_column"),
                    "date_column": w.get("date_column") or raw.get("date_column"),
                }
            )
        labels = [w["label"] for w in out["waves"]]
        if len(set(labels)) != len(labels):
            raise RunFileError(f"Wave labels must be unique, not {labels}.")
    else:
        out["target"] = _path(target, base, "target")

    layer = raw.get("layer")
    if layer is not None and not isinstance(layer, str | int):
        raise RunFileError(f"'layer' must be a name or an index, not {layer!r}.")
    band = raw.get("band", 1)
    if isinstance(band, bool) or not isinstance(band, int) or band < 1:
        raise RunFileError(f"'band' must be a band number from 1, not {band!r}.")
    out.update({"layer": layer, "band": band})

    lists = ("outcomes", "negative_controls", "covariates", "categorical", "moderators")
    for key in lists:
        out[key] = _strings(raw.get(key) or [], key)
    cats = set(out["categorical"])
    out["covariates"] = sorted(set(out["covariates"]) | cats)
    clash = sorted(set(out["outcomes"]) & set(out["negative_controls"]))
    if clash:
        raise RunFileError(f"A column cannot be an outcome and a control: {clash}.")

    for key in ("date_column", "entity_column", "wave_column", "area_column"):
        value = raw.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise RunFileError(f"'{key}' must be a column name, not {value!r}.")
        out[key] = value
    assign_by = raw.get("assign_by", "year")
    if assign_by not in ASSIGN_BY:
        raise RunFileError(f"'assign_by' must be one of {list(ASSIGN_BY)}.")
    out["assign_by"] = assign_by

    if mode == "year-aware" and not out["date_column"]:
        raise RunFileError("mode 'year-aware' needs 'date_column'.")
    if mode == "longitudinal" and not out["waves"]:
        if not (out["entity_column"] and out["date_column"]):
            raise RunFileError(
                "A longitudinal target needs 'entity_column' and 'date_column'."
            )
        if assign_by == "wave" and not out["wave_column"]:
            raise RunFileError("assign_by 'wave' needs 'wave_column'.")
    if mode == "longitudinal":
        for i, w in enumerate(out["waves"]):
            if not (w["entity_column"] and w["date_column"]):
                raise RunFileError(
                    f"waves[{i}] needs 'entity_column' and 'date_column' (on the "
                    "wave or on the step)."
                )

    for key in CHANNEL_INPUTS:
        if key not in raw:
            raise RunFileError(
                f"'{key}' is required: metric files, or a reference to a "
                f"[[{key}]] step."
            )
        out[key] = _channel_source(raw[key], key, base)
        if mode == "cross-sectional" and isinstance(out[key], list | dict):
            raise RunFileError(
                f"A cross-sectional run takes one '{key}' file; per-year files "
                "need mode 'year-aware' or 'longitudinal'."
            )
    return out


# ────────────────────────────────────────────────────────────────────
# Dependencies
# ────────────────────────────────────────────────────────────────────


def _link(steps: list[Step]) -> None:
    by_key = {s.key: s for s in steps}
    for s in steps:
        refs = [
            v for k in CHANNEL_INPUTS if isinstance(v := s.inputs.get(k), Reference)
        ]
        for ref in refs:
            if ref.key not in by_key:
                raise RunFileError(f"{s.key}: {ref} names no step of this run file.")
            if ref.key not in s.depends_on:
                s.depends_on.append(ref.key)
        for dep in s.depends_on:
            if dep not in by_key:
                raise RunFileError(f"{s.key}: depends_on {dep!r} names no step.")


def _order(steps: list[Step]) -> list[Step]:
    """Steps in file order, moved behind anything they depend on."""
    by_key = {s.key: s for s in steps}
    out: list[Step] = []
    state: dict[str, str] = {}

    def visit(s: Step, trail: list[str]) -> None:
        if state.get(s.key) == "done":
            return
        if state.get(s.key) == "open":
            raise RunFileError(f"Steps depend on each other: {' → '.join(trail)}.")
        state[s.key] = "open"
        for dep in s.depends_on:
            visit(by_key[dep], trail + [dep])
        state[s.key] = "done"
        out.append(s)

    for s in steps:
        visit(s, [s.key])
    return out


# ────────────────────────────────────────────────────────────────────
# Year matching (shared with the fusion tab)
# ────────────────────────────────────────────────────────────────────


def match_years_to_filenames(
    file_paths: Sequence[str], years: Sequence[str]
) -> dict[str, str]:
    """``{path: year}`` for filenames that name exactly one year, unambiguously.

    A year matches only where its label appears in the filename bounded by
    non-alphanumerics, so ``2011`` does not match inside ``20115``. The pairing
    is one-to-one in both directions: a filename naming several years is
    ambiguous, and so is a year named by several filenames.
    """
    hits: dict[str, str] = {}
    for path in file_paths:
        stem = os.path.basename(path)
        found = [
            y
            for y in years
            if re.search(rf"(?<![0-9A-Za-z]){re.escape(str(y))}(?![0-9A-Za-z])", stem)
        ]
        if len(found) == 1:
            hits[path] = found[0]
    claims: dict[str, list[str]] = {}
    for path, year in hits.items():
        claims.setdefault(year, []).append(path)
    return {path: year for path, year in hits.items() if len(claims[year]) == 1}
