"""The user's GeoFuse configuration (``config.toml``), outside any repository.

::

    state_dir = "D:/geofuse-state"     # where the command line keeps its jobs

    [earthengine]
    project = "my-cloud-project"        # the project NDVI downloads bill

    [limits]                            # what runs at once on this machine
    max_parallel = 4
    gpu = 1
    streetview = 1
    earthengine = 2
    fusion = 1
    idle_minutes = 10

The file lives in the per-user configuration folder (``%LOCALAPPDATA%\\geofuse``
on Windows, ``~/.config/geofuse`` on Linux, Application Support on macOS).
Every key is optional and defaults to the values above. Credentials are not
kept here; see ``geofuse credentials``.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Resource classes capped on one machine, and their defaults. ``max_parallel``
#: and ``idle_minutes`` govern the supervisor; the rest are slots that GUI and
#: command-line jobs share (``geofuse.jobs.locks``).
LIMIT_DEFAULTS: dict[str, int] = {
    "max_parallel": 4,
    "gpu": 1,
    "streetview": 1,
    "earthengine": 2,
    "fusion": 1,
    "idle_minutes": 10,
}
_TABLES = {
    "earthengine": {"project": str},
    "limits": dict.fromkeys(LIMIT_DEFAULTS, int),
}


class ConfigError(ValueError):
    """A configuration file or value that cannot be used, with the key named."""


def config_dir() -> Path:
    try:
        import platformdirs

        return Path(platformdirs.user_config_dir("geofuse", appauthor=False))
    except ImportError:
        return Path.home() / ".geofuse"


def config_path() -> Path:
    return config_dir() / "config.toml"


@dataclass
class Config:
    state_dir: str | None = None
    ee_project: str | None = None
    limits: dict[str, int] = field(default_factory=lambda: dict(LIMIT_DEFAULTS))
    remotes: dict[str, dict] = field(default_factory=dict)
    path: Path | None = None


def _check(doc: dict) -> None:
    for key, value in doc.items():
        if key == "state_dir":
            if not isinstance(value, str) or not value:
                raise ConfigError("'state_dir' must be a folder path.")
        elif key == "remotes":
            if not isinstance(value, dict) or not all(
                isinstance(v, dict) for v in value.values()
            ):
                raise ConfigError("'remotes' holds one table per remote host.")
        elif key in _TABLES:
            if not isinstance(value, dict):
                raise ConfigError(f"'{key}' must be a table ([{key}]).")
            for sub, sub_value in value.items():
                kind = _TABLES[key].get(sub)
                if kind is None:
                    raise ConfigError(
                        f"Unknown key '{key}.{sub}' (known: "
                        f"{', '.join(_TABLES[key])})."
                    )
                if kind is int and (
                    isinstance(sub_value, bool)
                    or not isinstance(sub_value, int)
                    or sub_value < 1
                ):
                    raise ConfigError(f"'{key}.{sub}' must be a whole number from 1.")
                if kind is str and (not isinstance(sub_value, str) or not sub_value):
                    raise ConfigError(f"'{key}.{sub}' must be text.")
        else:
            raise ConfigError(
                f"Unknown key '{key}' (known: state_dir, earthengine, limits, "
                "remotes)."
            )


def read_document(path: str | os.PathLike | None = None) -> dict:
    """The raw configuration document (empty when the file does not exist)."""
    path = Path(path) if path is not None else config_path()
    if not path.is_file():
        return {}
    try:
        with open(path, "rb") as f:
            doc = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Could not read {path}: {exc}") from exc
    _check(doc)
    return doc


def load(path: str | os.PathLike | None = None) -> Config:
    """The user's configuration, every missing key at its default."""
    path = Path(path) if path is not None else config_path()
    doc = read_document(path)
    limits = dict(LIMIT_DEFAULTS)
    limits.update(doc.get("limits") or {})
    return Config(
        state_dir=doc.get("state_dir"),
        ee_project=(doc.get("earthengine") or {}).get("project"),
        limits=limits,
        remotes=dict(doc.get("remotes") or {}),
        path=path,
    )


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ConfigError(f"Cannot write {value!r} to the configuration.")


def dumps(doc: dict) -> str:
    """``doc`` as TOML: top-level values, then tables (one level of nesting,
    plus ``remotes.<name>`` tables)."""
    lines: list[str] = []
    for key, value in doc.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {_toml_value(value)}")
    for key, value in doc.items():
        if not isinstance(value, dict):
            continue
        nested = {k: v for k, v in value.items() if isinstance(v, dict)}
        plain = {k: v for k, v in value.items() if not isinstance(v, dict)}
        if plain:
            lines += ["", f"[{key}]"]
            lines += [f"{k} = {_toml_value(v)}" for k, v in plain.items()]
        for sub, table in nested.items():
            lines += ["", f"[{key}.{sub}]"]
            lines += [f"{k} = {_toml_value(v)}" for k, v in table.items()]
    return "\n".join(lines).lstrip("\n") + "\n"


def set_value(
    dotted_key: str, value: Any, path: str | os.PathLike | None = None
) -> Path:
    """Set ``dotted_key`` (``state_dir``, ``earthengine.project``,
    ``limits.gpu``, ``remotes.<name>.host``…) and write the file; ``None``
    removes the key."""
    path = Path(path) if path is not None else config_path()
    doc = read_document(path)
    parts = dotted_key.split(".")
    node = doc
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"'{dotted_key}' is not a table key.")
    if value is None:
        node.pop(parts[-1], None)
    else:
        node[parts[-1]] = value
    _check(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text(dumps(doc), encoding="utf-8")
    os.replace(tmp, path)
    return path
