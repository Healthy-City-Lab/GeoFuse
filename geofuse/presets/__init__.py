"""Named, shareable download settings for the NDVI and GVI tabs.

A preset is a JSON document::

    {"format": "geofuse-preset", "version": 1, "engine": "ndvi" | "gvi",
     "name": ..., "description": ..., "created": "YYYY-MM-DD",
     "geofuse_version": ..., "settings": {...}}

``settings`` holds acquisition and method settings only — never dates, file
paths, study-area data or credentials. Each engine's schema is its
:mod:`geofuse.settings` schema, whose keys, types, choices and bounds match
the tab's widgets; :func:`validate` checks a document against it, refusing
anything malformed (with the key named) and warning about keys it does not
know. Built-in presets ship read-only under ``builtin/<engine>/``; user
presets live in a library folder (by default ``<repo>/presets/<engine>/``,
git-ignored).

Streamlit-free, so the CLI and notebooks can use it too.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .. import settings as _settings

FORMAT = "geofuse-preset"
VERSION = 1
ENGINES = ("ndvi", "gvi")

BUILTIN_DIR = Path(__file__).resolve().parent / "builtin"
USER_DIR = Path(__file__).resolve().parents[2] / "presets"


class PresetError(_settings.SettingError):
    """A preset that cannot be used, with the reason."""


class PresetExists(PresetError):
    """Saving would overwrite a preset of the same name."""


SCHEMAS: dict[str, dict[str, _settings.Setting]] = {
    engine: _settings.SCHEMAS[engine] for engine in ENGINES
}

# A key name like these, or a value shaped like a Google API key, anywhere in
# a document refuses it: presets are shared, credentials must not travel.
_SECRET_KEY = re.compile(r"api[_-]?key|apikey|token|secret|password", re.IGNORECASE)
_SECRET_VALUE = re.compile(r"AIza[0-9A-Za-z_\-]{35}")


def _find_secret(obj: Any, path: str = "") -> str | None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{path}.{k}" if path else str(k)
            if _SECRET_KEY.search(str(k)):
                return here
            found = _find_secret(v, here)
            if found:
                return found
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found = _find_secret(v, f"{path}[{i}]")
            if found:
                return found
    elif isinstance(obj, str) and _SECRET_VALUE.search(obj):
        return path or "value"
    return None


# ────────────────────────────────────────────────────────────────────
# Validation and documents
# ────────────────────────────────────────────────────────────────────


def validate(doc: Any, engine: str) -> tuple[dict, list[str]]:
    """Check a preset document for ``engine``; return ``(settings, warnings)``.

    Raises :class:`PresetError` naming the problem: not a preset, a preset for
    the other engine, a credential anywhere in it, or a setting of the wrong
    type or out of range. Unknown settings are dropped with a warning.
    Settings may be partial; only those present are applied.
    """
    if engine not in ENGINES:
        raise PresetError(f"Unknown engine {engine!r}.")
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise PresetError("This file is not a GeoFuse preset.")
    version = doc.get("version")
    if not isinstance(version, int) or version > VERSION:
        raise PresetError(
            f"Preset version {version!r} is not supported (this GeoFuse reads "
            f"version {VERSION})."
        )
    other = doc.get("engine")
    if other != engine:
        if other in ENGINES:
            raise PresetError(
                f"This is a {other.upper()} preset — load it in the "
                f"{other.upper()} tab."
            )
        raise PresetError(f"Unknown preset engine {other!r}.")
    secret = _find_secret(doc)
    if secret:
        raise PresetError(
            f"The preset contains a credential ('{secret}'); presets never carry "
            "API keys or passwords."
        )
    raw = doc.get("settings")
    if not isinstance(raw, dict):
        raise PresetError("The preset has no 'settings' object.")
    schema = SCHEMAS[engine]
    settings: dict = {}
    warnings: list[str] = []
    for key, value in raw.items():
        field = schema.get(key)
        if field is None:
            warnings.append(f"Unknown setting '{key}' ignored.")
            continue
        if value is None:
            raise PresetError(f"'{key}' needs a value.")
        try:
            settings[key] = field.check(key, value)
        except _settings.SettingError as exc:
            raise PresetError(str(exc)) from exc
    lo, hi = "season_start_month", "season_end_month"
    if lo in settings and hi in settings and settings[lo] > settings[hi]:
        raise PresetError(f"'{lo}' must be on or before '{hi}'.")
    return settings, warnings


def _geofuse_version() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(BUILTIN_DIR), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def make_preset(engine: str, name: str, description: str, settings: dict) -> dict:
    """A validated preset document for ``settings``."""
    name = str(name).strip()
    if not name:
        raise PresetError("A preset needs a name.")
    doc = {
        "format": FORMAT,
        "version": VERSION,
        "engine": engine,
        "name": name,
        "description": str(description or "").strip(),
        "created": date.today().isoformat(),
        "geofuse_version": _geofuse_version(),
        "settings": dict(settings),
    }
    doc["settings"], _ = validate(doc, engine)
    return doc


def read_preset(path: str | os.PathLike) -> dict:
    """Load a preset file's JSON (validated later, against an engine)."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, UnicodeDecodeError) as exc:
        raise PresetError(f"Could not read the preset: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PresetError(
            f"The preset is not valid JSON ({exc.msg}, line {exc.lineno})."
        ) from exc
    if not isinstance(doc, dict):
        raise PresetError("This file is not a GeoFuse preset.")
    return doc


def write_preset(path: str | os.PathLike, doc: dict) -> None:
    """Validate ``doc`` against its own engine and write it as JSON."""
    validate(doc, str(doc.get("engine")))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ────────────────────────────────────────────────────────────────────
# Library
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PresetEntry:
    name: str
    description: str
    path: Path
    builtin: bool


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return slug or "preset"


def _entries(folder: Path, engine: str, builtin: bool) -> list[PresetEntry]:
    out = []
    for path in sorted(folder.glob("*.json")):
        try:
            doc = read_preset(path)
            validate(doc, engine)
        except PresetError:
            continue
        out.append(
            PresetEntry(
                str(doc.get("name") or path.stem),
                str(doc.get("description") or ""),
                path,
                builtin,
            )
        )
    return out


def list_presets(
    engine: str, user_dir: str | os.PathLike | None = None
) -> list[PresetEntry]:
    """Built-in presets first (in file order), then the user's, by name.

    A user preset whose name repeats a built-in's is left out.
    """
    builtins = _entries(BUILTIN_DIR / engine, engine, True)
    taken = {e.name for e in builtins}
    users = _entries(Path(user_dir or USER_DIR) / engine, engine, False)
    users = sorted(
        (e for e in users if e.name not in taken), key=lambda e: e.name.lower()
    )
    return builtins + users


def find_preset(
    engine: str, name: str, user_dir: str | os.PathLike | None = None
) -> PresetEntry | None:
    return next((e for e in list_presets(engine, user_dir) if e.name == name), None)


def save_user_preset(
    doc: dict,
    user_dir: str | os.PathLike | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Write ``doc`` into the user library; refuses to shadow a built-in.

    Raises :class:`PresetExists` when a user preset of that name exists and
    ``overwrite`` is false.
    """
    engine = str(doc.get("engine"))
    validate(doc, engine)
    name = str(doc.get("name") or "").strip()
    if not name:
        raise PresetError("A preset needs a name.")
    if any(e.builtin and e.name == name for e in list_presets(engine, user_dir)):
        raise PresetError(f"'{name}' is a built-in preset; choose another name.")
    folder = Path(user_dir or USER_DIR) / engine
    existing = next(
        (e for e in _entries(folder, engine, False) if e.name == name), None
    )
    if existing is not None and not overwrite:
        raise PresetExists(f"A preset named '{name}' already exists.")
    path = existing.path if existing is not None else folder / f"{_slug(name)}.json"
    write_preset(path, doc)
    return path


def load_settings(
    engine: str, name: str, user_dir: str | os.PathLike | None = None
) -> dict:
    """The validated settings of the named preset."""
    entry = find_preset(engine, name, user_dir)
    if entry is None:
        raise PresetError(f"No {engine.upper()} preset named '{name}'.")
    settings, _ = validate(read_preset(entry.path), engine)
    return settings
