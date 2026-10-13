"""Commented run-file templates, written from the settings schema.

Every setting appears commented out at its default with its one-line help, so
a template never drifts from what the toolbox accepts.
"""

from __future__ import annotations

from geofuse import settings as _settings

_HEADER = """\
# GeoFuse run file. Run it with:  geofuse run THIS.toml   (in this terminal)
#                                 geofuse submit THIS.toml (in the background)
# Settings resolve: default < preset < [defaults.<engine>] < step < command line.
# Relative paths are relative to this file. Never put credentials here.
format = "geofuse-run"
version = 1
output_dir = "output_results"
"""

_INPUTS = {
    "ndvi": """\
name = "area"                      # unique among the NDVI steps
study_area = "data/area.gpkg"      # a path, or a list of paths
# merge = true                     # several files as one job
# preset = "Landsat TOA, all scenes, water-masked"
ranges = [[2023-06-01, 2023-09-30]]     # both ends included
# specific_dates = [2023-07-15]         # +/- specific_window_days
# year_column = "year"                  # one composite per year, season months
# chains = [                            # aggregation chains, in place of statistics
#   {name = "f500", steps = ["focal:mean:500", "temporal:mean", "water_mask"]},
# ]
""",
    "gvi": """\
name = "area"
study_area = "data/area.gpkg"
# merge = true
# preset = "Historical capture ± 2 years"
# year_column = "year"             # with capture_mode = "column"
""",
    "fusion": """\
name = "outcome"
target = "data/cohort.gpkg"        # vector (outcome columns) or GeoTIFF (band)
# layer = "participants"           # GeoPackage layer, when there are several
# band = 1                         # raster target band
outcomes = ["outcome"]
# negative_controls = ["grip"]
# covariates = ["age", "income"]
# categorical = ["sex"]            # one-hot encoded; also covariates
# moderators = ["sex"]
mode = "cross-sectional"           # or "year-aware", "longitudinal"
# date_column = "visit_date"       # year-aware and longitudinal
# entity_column = "id"             # longitudinal
# assign_by = "year"               # longitudinal: "year" or "wave"
# wave_column = "wave"             # longitudinal, assign_by = "wave"
# area_column = "fsa"              # longitudinal neighbourhood fixed effects
ndvi = "@ndvi.area"                # a file, {year = file}, [files], or @ndvi.STEP[name]
gvi = "@gvi.area"                  # a file, {year = file}, [files], or @gvi.STEP
""",
}


def _value(value) -> str:
    if value is None:
        return "0"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    return repr(value)


def settings_block(engine: str) -> str:
    """Every setting of ``engine``, commented out at its default."""
    lines = [f"# -- {engine.upper()} settings (defaults shown) --"]
    for key, s in _settings.SCHEMAS[engine].items():
        choices = f" One of {list(s.choices)}." if s.kind == "choice" else ""
        lines.append(f"# {key} = {_value(s.default)}    # {s.help}{choices}")
    return "\n".join(lines) + "\n"


def template(kind: str) -> str:
    """A commented run file: ``ndvi``, ``gvi``, ``fusion`` or ``pipeline``
    (an NDVI download and a GVI run feeding one fusion fit)."""
    if kind == "pipeline":
        parts = [_HEADER]
        for engine in ("ndvi", "gvi", "fusion"):
            parts.append(f"\n[[{engine}]]\n" + _INPUTS[engine])
        parts.append("\n" + settings_block("fusion"))
        return "".join(parts)
    if kind not in _INPUTS:
        raise ValueError(
            f"Unknown template {kind!r}; use ndvi, gvi, fusion or pipeline."
        )
    inputs = _INPUTS[kind]
    if kind == "fusion":
        inputs = inputs.replace('"@ndvi.area"', '"data/ndvi.tif"').replace(
            '"@gvi.area"', '"data/gvi.gpkg"'
        )
    return _HEADER + f"\n[[{kind}]]\n" + inputs + settings_block(kind)
