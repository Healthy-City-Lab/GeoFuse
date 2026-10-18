"""From run-file steps to the jobs the tabs would submit.

Two layers, so a plan can be shown, stored and replayed without holding any
data:

* :func:`expand` turns each step of a :class:`~geofuse.runfile.RunFile` into
  :class:`JobSpec` records — JSON-safe, one per job the GUI would start (an
  NDVI step with two ranges and a year column is three jobs). It reads only
  what it must to decide: year columns, for per-year jobs and for the year
  coverage of a fusion run; layer names and geometry types of fusion
  targets. References to earlier steps resolve to the files those jobs will
  write.
* :func:`build` turns one :class:`JobSpec` into a :class:`Submission`: the
  job record (type, name, params) and the executor call with its arguments,
  exactly as the NDVI, GVI and fusion tabs build them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from typing import Any

from geofuse import runfile as _rf
from geofuse import settings as _settings
from geofuse.jobs.locks import RESOURCES
from geofuse.ndvi_chains import output_stems
from geofuse.vector_io import sanitize_name_for_file

# The fusion tab never downloads metrics itself; it records this window.
_FUSION_NDVI_WINDOW = ("2023-06-01", "2023-09-30")

#: The segmentation model GVI jobs load, as the GVI tab does.
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model",
    "best_model.pth",
)


#: Why a GVI step with ``download_mode = "api"`` cannot start without a key.
NO_STREETVIEW_KEY = (
    'download_mode = "api" needs a Street View API key: store one with '
    "`geofuse credentials set streetview-key`."
)


class PlanError(ValueError):
    """A run file whose jobs cannot be laid out, with the reason."""


@dataclass
class JobSpec:
    """One job of a run, JSON-safe.

    ``options`` holds what differs between jobs of one step (an NDVI date
    range, a GVI capture year, the fusion inputs); ``settings`` is the
    step's resolved settings. ``outputs`` lists what the job will write:
    ``files`` for a single result, ``years`` (``{year: file}``) for a
    per-year job, each keyed by statistic (or aggregation chain) for NDVI.
    """

    step: str
    kind: str
    name: str
    output_dir: str
    sources: list[str]
    merged: bool
    group: str | None
    options: dict[str, Any]
    settings: dict[str, Any]
    preset: str | None
    depends_on: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> JobSpec:
        return cls(**dict(d))


@dataclass
class Submission:
    """What a tab submits for one job: the record and the executor call."""

    record_type: str
    record_name: str
    params: dict[str, Any]
    method: str
    kwargs: dict[str, Any]


# ────────────────────────────────────────────────────────────────────
# Reading what a plan needs from the inputs
# ────────────────────────────────────────────────────────────────────


def column_years(paths: Sequence[str], column: str) -> list[str]:
    """Sorted years found in ``column`` across ``paths`` (as the tabs list them)."""
    import pyogrio

    from geofuse.longitudinal import parse_date_column

    years: set[int] = set()
    for path in paths:
        if not os.path.isfile(path):
            raise PlanError(f"File not found: {path}")
        fields = [str(f) for f in pyogrio.read_info(path)["fields"]]
        if column not in fields:
            raise PlanError(
                f"{os.path.basename(path)} has no column {column!r} "
                f"(columns: {', '.join(fields)})."
            )
        frame = pyogrio.read_dataframe(path, columns=[column], read_geometry=False)
        years |= {int(d.year) for d in parse_date_column(frame[column]).dropna()}
    if not years:
        raise PlanError(f"No parseable years in column {column!r}.")
    return [str(y) for y in sorted(years)]


def _require_files(paths: Sequence[str]) -> None:
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise PlanError("File(s) not found: " + ", ".join(missing))


# ────────────────────────────────────────────────────────────────────
# Expanding steps into jobs
# ────────────────────────────────────────────────────────────────────


def _units(step: _rf.Step) -> list[dict]:
    """One unit per study-area file, or one merged unit (the tabs' merge mode,
    which needs at least two files)."""
    sources = step.inputs["study_area"]
    if step.inputs["merge"] and len(sources) > 1:
        return [{"key": step.name, "sources": list(sources), "merged": True}]
    return [
        {"key": os.path.basename(p), "sources": [p], "merged": False} for p in sources
    ]


def _base(unit: dict) -> str:
    if unit["merged"]:
        return sanitize_name_for_file(unit["key"])
    return os.path.splitext(unit["key"])[0]


#: Each NDVI date mode's statistics and aggregation-chain settings.
_NDVI_MODE_SETTINGS = {
    "range": ("statistics_ranges", "chains_ranges"),
    "specific": ("statistics_specific", "chains_specific"),
    "column": ("statistics_column", "chains_column"),
}


def _ndvi_files(folder: str, output_name: str, settings: dict, mode: str) -> dict:
    """The file each output of one NDVI date mode writes, keyed by statistic,
    or by chain name when the mode has aggregation chains."""
    ext = (
        ".tif"
        if settings["save_geotiff"]
        else ".gpkg" if settings["save_gpkg"] else ".geojson"
    )
    stats, chains = (settings[k] for k in _NDVI_MODE_SETTINGS[mode])
    return {
        key: os.path.join(folder, f"{stem}_ndvi{ext}")
        for key, stem in output_stems(output_name, stats, chains).items()
    }


def _expand_ndvi(step: _rf.Step, today: date) -> list[JobSpec]:
    s, out = step.settings, step.output_dir
    specs = []
    for unit in _units(step):
        base = _base(unit)

        def spec(kind, options, outputs, unit=unit, base=base):
            return JobSpec(
                step=step.key,
                kind=kind,
                name=base,
                output_dir=out,
                sources=unit["sources"],
                merged=unit["merged"],
                group=unit["key"] if unit["merged"] else None,
                options=options,
                settings=dict(s),
                preset=step.preset,
                depends_on=list(step.depends_on),
                resources=list(RESOURCES[kind]),
                outputs=outputs,
            )

        for start, end in step.inputs["ranges"]:
            name = f"{base}_{start.replace('-', '')}_{end.replace('-', '')}"
            specs.append(
                spec(
                    "ndvi",
                    {
                        "mode": "range",
                        "start_date": start,
                        "end_date": end,
                        "output_name": name,
                    },
                    {"files": _ndvi_files(out, name, s, "range")},
                )
            )
        window = s["specific_window_days"]
        for target in step.inputs["specific_dates"]:
            t = date.fromisoformat(target)
            name = f"{base}_{target.replace('-', '')}"
            specs.append(
                spec(
                    "ndvi",
                    {
                        "mode": "specific",
                        "target_date": target,
                        "window_days": window,
                        "start_date": (t - timedelta(days=window)).isoformat(),
                        "end_date": min(t + timedelta(days=window), today).isoformat(),
                        "output_name": name,
                    },
                    {"files": _ndvi_files(out, name, s, "specific")},
                )
            )
        column = step.inputs["year_column"]
        if column:
            folder = os.path.join(out, f"{base}_temporal_ndvi")
            years = column_years(unit["sources"], column)
            by_year = {
                y: _ndvi_files(folder, f"{base}_{y}", s, "column") for y in years
            }
            keys = list(by_year[years[0]])
            specs.append(
                spec(
                    "ndvi_column",
                    {
                        "mode": "column",
                        "date_column": column,
                        "season_start_month": s["season_start_month"],
                        "season_end_month": s["season_end_month"],
                    },
                    {
                        "folder": folder,
                        "years": {
                            key: {y: by_year[y][key] for y in years} for key in keys
                        },
                    },
                )
            )
    return specs


def _shard(specs: list[JobSpec], step: _rf.Step) -> list[JobSpec]:
    """``shards = N``: each job becomes N shard jobs (the sub-step
    ``<step>:shards``) and a merge job, under the step's own key, that waits
    for them and writes the step's outputs."""
    count = int(step.inputs.get("shards") or 1)
    if count < 2:
        return specs
    sub_step = f"{step.key}:shards"
    shards = [
        replace(
            spec,
            step=sub_step,
            name=f"{spec.name} [{i + 1}/{count}]",
            options={**spec.options, "shard": [i, count]},
            outputs={},
        )
        for spec in specs
        for i in range(count)
    ]
    merges = [
        replace(
            spec,
            kind="gvi_merge",
            options={**spec.options, "shards": count, "of": spec.kind},
            depends_on=[sub_step, *spec.depends_on],
            resources=[],
        )
        for spec in specs
    ]
    return shards + merges


def _expand_gvi(step: _rf.Step, today: date) -> list[JobSpec]:
    s, out = step.settings, step.output_dir
    mode = s["capture_mode"]
    specs = []
    for unit in _units(step):
        base = _base(unit)
        ext = "gpkg" if s["save_gpkg"] else "geojson"
        if mode == "column":
            column = step.inputs["year_column"]
            folder = os.path.join(out, f"{base}_temporal_gvi")
            years = column_years(unit["sources"], column)
            kind = "gvi_column"
            options = {"mode": "column", "date_column": column}
            outputs = {
                "folder": folder,
                "years": {
                    y: os.path.join(folder, f"{base}_{y}_gvi.{ext}") for y in years
                },
            }
        else:
            year = None
            if mode == "year":
                year = min(int(s["target_year"] or today.year), today.year)
            kind = "gvi"
            options = {"mode": mode, "target_year": year}
            stem = base if year is None else f"{base}_{year}"
            outputs = {"files": os.path.join(out, f"{stem}_gvi.{ext}")}
        specs.append(
            JobSpec(
                step=step.key,
                kind=kind,
                name=base,
                output_dir=out,
                sources=unit["sources"],
                merged=unit["merged"],
                group=unit["key"] if unit["merged"] else None,
                options=options,
                settings=dict(s),
                preset=step.preset,
                depends_on=list(step.depends_on),
                resources=list(RESOURCES[kind]),
                outputs=outputs,
            )
        )
    return _shard(specs, step)


def _target_info(path: str, layer) -> dict:
    """Whether the target is a vector, its layer, columns and geometry type."""
    from geofuse.vector_io import list_gpkg_layer_names, target_path_is_raster

    _require_files([path])
    if target_path_is_raster(path):
        import rasterio

        with rasterio.open(path) as src:
            return {"vector": False, "bands": int(src.count), "layer": None}
    import pyogrio

    if layer is None and path.lower().endswith((".gpkg", ".zip")):
        layers = list_gpkg_layer_names(path)
        if len(layers) > 1:
            raise PlanError(
                f"{os.path.basename(path)} has several layers {layers}; "
                "name one with 'layer'."
            )
        layer = layers[0] if layers else None
    info = pyogrio.read_info(path, **({} if layer is None else {"layer": layer}))
    return {
        "vector": True,
        "layer": layer,
        "fields": [str(f) for f in info["fields"]],
        "geometry_type": str(info.get("geometry_type") or ""),
    }


def _waves(step: _rf.Step, info: dict) -> list[str]:
    """The years or waves greenery files are assigned to (as the fusion tab
    discovers them)."""
    i = step.inputs
    if i["mode"] == "cross-sectional":
        return []
    if i["mode"] == "year-aware":
        return column_years([i["target"]], i["date_column"])
    if i["waves"]:
        if i["assign_by"] == "wave":
            return [w["label"] for w in i["waves"]]
        years: set[str] = set()
        for w in i["waves"]:
            years.update(column_years([w["file"]], w["date_column"]))
        return sorted(years)
    if i["assign_by"] == "year":
        return column_years([i["target"]], i["date_column"])
    import pyogrio

    frame = pyogrio.read_dataframe(
        i["target"], columns=[i["wave_column"]], read_geometry=False
    )
    return sorted(str(v) for v in frame[i["wave_column"]].dropna().unique())


def _reference_files(
    ref: _rf.Reference, by_step: dict[str, list[JobSpec]], per_year: bool
):
    """The file (or ``{year: file}``) an ``@engine.step`` reference names."""
    jobs = by_step[ref.key]
    if per_year:
        jobs = [j for j in jobs if "years" in j.outputs]
        if len(jobs) != 1:
            raise PlanError(
                f"{ref} must be one per-year job (a year column, one study-area "
                f"unit) to give a file per year; it has {len(jobs)}."
            )
        years = jobs[0].outputs["years"]
    else:
        jobs = [j for j in jobs if "files" in j.outputs]
        if len(jobs) != 1:
            raise PlanError(
                f"{ref} must be exactly one job (one study-area unit, one date "
                f"range or date) to give one file; it has {len(jobs)}."
            )
        years = None
        files = jobs[0].outputs["files"]
    if ref.engine == "gvi":
        if not jobs[0].settings["save_gpkg"]:
            raise PlanError(f"{ref} must save its GeoPackage for fusion to read it.")
        return years if per_year else files
    table = years if per_year else files
    outputs = list(table)
    if ref.statistic is None and len(outputs) != 1:
        raise PlanError(
            f"{ref} writes several outputs {outputs}; name one, e.g. "
            f"{ref}[{outputs[0]}]."
        )
    name = ref.statistic or outputs[0]
    if name not in table:
        raise PlanError(f"{ref} writes no output named {name!r} ({outputs}).")
    return table[name]


def _channel_files(
    step: _rf.Step, key: str, waves: list[str], by_step: dict[str, list[JobSpec]]
):
    source = step.inputs[key]
    per_year = step.inputs["mode"] != "cross-sectional"
    if isinstance(source, _rf.Reference):
        files = _reference_files(source, by_step, per_year)
        _require = False
    else:
        files, _require = source, True
    if not per_year:
        if _require:
            _require_files([files])
        return files
    if isinstance(files, list):
        if _require:
            _require_files(files)
        matched = _rf.match_years_to_filenames(files, waves)
        files = {}
        for path, year in matched.items():
            files[year] = path
    elif _require:
        _require_files(list(files.values()))
    files = {str(k): v for k, v in files.items()}
    missing = [w for w in waves if w not in files]
    if missing:
        raise PlanError(
            f"[[fusion]] {step.name}: no {key.upper()} file for "
            f"{', '.join(missing)} (years / waves found: {', '.join(waves)})."
        )
    return {w: files[w] for w in waves}


def _expand_fusion(step: _rf.Step, by_step: dict[str, list[JobSpec]]) -> list[JobSpec]:
    i = step.inputs
    target = i["target"] or i["waves"][0]["file"]
    info = _target_info(target, i["layer"])
    if info["vector"]:
        missing = [
            c
            for c in (
                i["outcomes"]
                + i["negative_controls"]
                + i["covariates"]
                + i["moderators"]
                + [
                    i[k]
                    for k in (
                        "date_column",
                        "entity_column",
                        "wave_column",
                        "area_column",
                    )
                    if i[k]
                ]
            )
            if c not in info["fields"] and not i["waves"]
        ]
        if missing:
            raise PlanError(
                f"[[fusion]] {step.name}: {os.path.basename(target)} has no "
                f"column(s) {sorted(set(missing))}."
            )
        if not i["outcomes"]:
            raise PlanError(
                f"[[fusion]] {step.name}: a vector target needs 'outcomes'."
            )
    elif i["band"] > info["bands"]:
        raise PlanError(
            f"[[fusion]] {step.name}: the target has {info['bands']} band(s), "
            f"not band {i['band']}."
        )
    waves = _waves(step, info)
    channels = {
        key: _channel_files(step, key, waves, by_step) for key in ("ndvi", "gvi")
    }
    if i["waves"]:
        _require_files([w["file"] for w in i["waves"]])
    options = {
        **{k: v for k, v in i.items() if k not in ("ndvi", "gvi")},
        "target": target,
        "layer": info["layer"],
        "vector": info["vector"],
        "polygon": "polygon" in info.get("geometry_type", "").lower(),
        "waves_found": waves,
        "ndvi_files": channels["ndvi"],
        "gvi_files": channels["gvi"],
    }
    return [
        JobSpec(
            step=step.key,
            kind="fusion",
            name=os.path.splitext(os.path.basename(target))[0],
            output_dir=step.output_dir,
            sources=[target] + [w["file"] for w in i["waves"][1:]],
            merged=False,
            group=None,
            options=options,
            settings=dict(step.settings),
            preset=None,
            depends_on=list(step.depends_on),
            resources=list(RESOURCES["fusion"]),
            outputs={"folder": os.path.join(step.output_dir, "fusion")},
        )
    ]


#: Cluster resource groups: GVI needs a GPU, fusion cores and memory, the
#: rest (NDVI downloads, GVI merges) little of either.
GROUPS = ("gpu", "fusion", "light")


def job_group(spec: JobSpec) -> str:
    if spec.kind in ("gvi", "gvi_column"):
        return "gpu"
    return "fusion" if spec.kind == "fusion" else "light"


def levels(specs: Sequence[JobSpec]) -> list[int]:
    """Each job's dependency level: 0 when it waits for no job in ``specs``,
    else one more than the deepest step it waits for."""
    by_step: dict[str, int] = {}
    out = []
    for spec in specs:
        waits = [by_step[d] for d in spec.depends_on if d in by_step]
        level = 1 + max(waits) if waits else 0
        by_step[spec.step] = max(by_step.get(spec.step, 0), level)
        out.append(level)
    return out


def expand(run: _rf.RunFile, *, today: date | None = None) -> list[JobSpec]:
    """Every job of ``run``, in dependency order (``depends_on`` names steps)."""
    today = today or date.today()
    by_step: dict[str, list[JobSpec]] = {}
    specs: list[JobSpec] = []
    for step in run.steps:
        try:
            if step.engine == "ndvi":
                jobs = _expand_ndvi(step, today)
            elif step.engine == "gvi":
                jobs = _expand_gvi(step, today)
            else:
                jobs = _expand_fusion(step, by_step)
        except PlanError as exc:
            msg = str(exc)
            prefix = f"[[{step.engine}]] {step.name}"
            named = msg if msg.startswith(prefix) else f"{prefix}: {msg}"
            raise PlanError(named) from exc
        by_step[step.key] = jobs
        specs.extend(jobs)
    return specs


# ────────────────────────────────────────────────────────────────────
# Building submissions (what the tabs submit)
# ────────────────────────────────────────────────────────────────────


def _load_unit(spec: JobSpec):
    """``(fname, raw, polygon)`` as the tabs hold a unit: one file or a merged
    group; ``polygon`` is true when any of its files starts with a polygon."""
    from geofuse.vector_io import load_vector_paths, merge_gdfs_wgs84

    loaded = load_vector_paths(spec.sources)
    if len(loaded) != len(spec.sources):
        _require_files(spec.sources)

    def first_is_polygon(gdf) -> bool:
        try:
            return gdf.geometry.iloc[0].geom_type in ("Polygon", "MultiPolygon")
        except Exception:
            return False

    polygon = any(first_is_polygon(g) for _, g in loaded)
    if spec.merged:
        return spec.group, merge_gdfs_wgs84([g for _, g in loaded]), polygon
    return loaded[0][0], loaded[0][1], polygon


def _merge_params(spec: JobSpec, fname: str) -> dict:
    from geofuse.vector_io import file_size_mtime_fingerprint

    if spec.merged:
        return {
            "merged": True,
            "group_name": spec.group,
            "source_files": [os.path.basename(p) for p in spec.sources],
            "source_paths": list(spec.sources),
            "source_fingerprints": [
                file_size_mtime_fingerprint(p) for p in spec.sources
            ],
        }
    return {
        "merged": False,
        "group_name": None,
        "source_files": [fname],
        "source_paths": [],
        "source_fingerprints": [],
    }


def _build_ndvi(spec: JobSpec, ee_project: str | None = None) -> Submission:
    from geofuse.vector_io import file_size_mtime_fingerprint, geometry_sha256

    s, o = spec.settings, spec.options
    fname, raw, _ = _load_unit(spec)
    ds_path = None if spec.merged else spec.sources[0]
    stats_key, chains_key = _NDVI_MODE_SETTINGS[o["mode"]]
    options = {
        "temporal_reducers": list(s[stats_key]),
        "satellite": s["satellite"],
        "coverage_rescue": s["coverage_rescue"],
        "water_mask": s["water_mask"],
        "cloud_handling": s["cloud_handling"],
        "reflectance": s["reflectance"],
        "substitute_2012": s["substitute_2012"],
    }
    if s[chains_key]:
        options["chains"] = [dict(c) for c in s[chains_key]]
    if spec.preset:
        options["preset"] = spec.preset
    common = {
        "cloud_pct": s["cloud_max"],
        "resolution": s["resolution_m"],
        "buffer_m": s["buffer_m"],
    }
    saves = {
        "save_geotiff": s["save_geotiff"],
        "save_gpkg": s["save_gpkg"],
        "save_geojson": s["save_geojson"],
        "save_cluster_tiles": s["save_cluster_tiles"],
    }
    head = {
        "fname": fname,
        "input_path": ds_path,
        "input_fingerprint": file_size_mtime_fingerprint(ds_path),
    }
    tail = {
        "geometry_sha256": geometry_sha256(raw),
        **options,
        **_merge_params(spec, fname),
    }
    if ee_project:
        options = {**options, "ee_project": ee_project}
    data = {"raw": raw}
    if o["mode"] == "column":
        params = {
            **head,
            "mode": "column",
            "date_column": o["date_column"],
            "season_start_month": o["season_start_month"],
            "season_end_month": o["season_end_month"],
            **common,
            **saves,
            **tail,
        }
        kwargs = {
            "fname": fname,
            "dataset_data": data,
            "date_column": o["date_column"],
            "season_start_month": o["season_start_month"],
            "season_end_month": o["season_end_month"],
            **common,
            "output_dir": spec.output_dir,
            **saves,
            **options,
        }
        return Submission(
            "ndvi_column", spec.name, params, "submit_ndvi_column_subprocess", kwargs
        )
    if o["mode"] == "range":
        mode_params = {
            "mode": "range",
            "start_date": o["start_date"],
            "end_date": o["end_date"],
        }
    else:
        mode_params = {
            "mode": "specific",
            "target_date": o["target_date"],
            "window_days": o["window_days"],
        }
    params = {
        **head,
        **mode_params,
        **common,
        "output_name": o["output_name"],
        **saves,
        **tail,
    }
    kwargs = {
        "fname": fname,
        "dataset_data": data,
        "start_date": o["start_date"],
        "end_date": o["end_date"],
        **common,
        "output_name": o["output_name"],
        "output_dir": spec.output_dir,
        **saves,
        **options,
    }
    return Submission("ndvi", spec.name, params, "submit_ndvi_subprocess", kwargs)


def _build_gvi(
    spec: JobSpec, *, model_path: str, api_key: str | None, pano_cache_db_path: str
) -> Submission:
    from geofuse.vector_io import file_size_mtime_fingerprint, geometry_sha256

    s, o = spec.settings, spec.options
    fname, raw, polygon = _load_unit(spec)
    gtype = "poly" if polygon else "point"
    if spec.merged:
        out_fname = f"{sanitize_name_for_file(spec.group)}.gpkg"
        ds_path = None
    else:
        out_fname = fname
        ds_path = spec.sources[0]
    save_debug = s["save_raw"]
    max_year_diff = (
        s["max_year_difference"]
        if s["limit_year_difference"] and o["mode"] != "recent"
        else None
    )
    capture_args: dict[str, Any] = {
        "month_window": (
            [s["month_start"], s["month_end"]] if s["limit_months"] else None
        ),
        "month_policy": s["month_policy"],
        **{key: s[key] for key in _settings.GVI_RUN_OPTIONS},
    }
    if spec.preset:
        capture_args["preset"] = spec.preset
    common = {
        "fname": out_fname,
        "input_path": ds_path,
        "input_fingerprint": file_size_mtime_fingerprint(ds_path),
        "step": s["grid_resolution_m"],
        "buffer": s["buffer_m"],
        "save_panos": save_debug,
        "save_masks": save_debug,
        "save_gpkg": s["save_gpkg"],
        "save_geotiff": s["save_geotiff"],
        "save_geojson": s["save_geojson"],
        "max_year_diff": max_year_diff,
        **capture_args,
        "model_path": model_path,
        "has_api_key": api_key is not None,
        "geometry_sha256": geometry_sha256(raw),
        **_merge_params(spec, fname),
    }
    dataset = {
        "raw": raw,
        "type": gtype,
        "processed": None,
        "meta": None,
        "accumulated": [],
        "results": None,
    }
    saves = {
        "output_dir": spec.output_dir,
        "save_gpkg": s["save_gpkg"],
        "save_geotiff": s["save_geotiff"],
        "save_geojson": s["save_geojson"],
        "pano_cache_db_path": pano_cache_db_path,
    }
    init_args = {"model_path": model_path, "api_key": api_key}
    if o.get("shard"):
        capture_args["shard"] = list(o["shard"])
        common["shard"] = list(o["shard"])
    if o["mode"] == "column":
        params = {**common, "mode": "column", "date_column": o["date_column"]}
        kwargs = {
            "fname": out_fname,
            "dataset_data": dataset,
            "date_column": o["date_column"],
            "init_args": init_args,
            "run_args": {
                "step": s["grid_resolution_m"],
                "buffer": s["buffer_m"],
                "save_panos": save_debug,
                "save_masks": save_debug,
                "max_year_diff": max_year_diff,
                **capture_args,
            },
            **saves,
        }
        return Submission(
            "gvi_column",
            os.path.splitext(out_fname)[0],
            params,
            "submit_gvi_column_subprocess",
            kwargs,
        )
    params = {**common, "target_year": o["target_year"]}
    kwargs = {
        "fname": out_fname,
        "dataset_data": dataset,
        "init_args": init_args,
        "run_args": {
            "step": s["grid_resolution_m"],
            "buffer": s["buffer_m"],
            "save_panos": save_debug,
            "save_masks": save_debug,
            "target_year": o["target_year"],
            "max_year_diff": max_year_diff,
            **capture_args,
        },
        **saves,
    }
    return Submission(
        "gvi", os.path.splitext(out_fname)[0], params, "submit_gvi_subprocess", kwargs
    )


def _build_gvi_merge(spec: JobSpec, **build_args) -> Submission:
    """The merge of a sharded GVI step: the GVI job's inputs, read again."""
    o = spec.options
    gvi = _build_gvi(replace(spec, kind=o["of"]), **build_args)
    kw = gvi.kwargs
    kwargs = {
        "fname": kw["fname"],
        "dataset_data": {
            "raw": kw["dataset_data"]["raw"],
            "type": kw["dataset_data"]["type"],
        },
        "run_args": kw["run_args"],
        "output_dir": spec.output_dir,
        "shards": int(o["shards"]),
        "date_column": o.get("date_column"),
        "save_gpkg": kw["save_gpkg"],
        "save_geotiff": kw["save_geotiff"],
        "save_geojson": kw["save_geojson"],
    }
    params = {**gvi.params, "shards": int(o["shards"])}
    return Submission(
        "gvi_merge", gvi.record_name, params, "submit_gvi_merge_subprocess", kwargs
    )


def fusion_run_config(spec: JobSpec) -> dict[str, Any]:
    """The fusion tab's ``run_config`` for this job (every ``run_fusion``
    setting that is not a file or a runtime object), plus the run options the
    tab leaves at ``run_fusion``'s defaults."""
    from geofuse import cgi_formulas

    s, o = spec.settings, spec.options
    vector, polygon = o["vector"], o["polygon"]
    adjust = s["spatial_adjust_method"] if vector else "none"
    adjusted = adjust != "none"
    blocks = vector and s["spatial_split"]
    form = "linear" if s["index_form"] == "sweep" else s["index_form"]
    covariates = list(o["covariates"])
    cats = set(o["categorical"])
    controls = [c for c in o["negative_controls"] if c not in o["outcomes"]]
    return {
        "buffer_meters": float(max(s["gvi_buffer_max_m"], s["ndvi_buffer_max_m"])),
        "gvi_buffer_min_m": float(s["gvi_buffer_min_m"]),
        "gvi_buffer_max_m": float(s["gvi_buffer_max_m"]),
        "gvi_buffer_step_m": float(s["gvi_buffer_step_m"]),
        "ndvi_buffer_min_m": float(s["ndvi_buffer_min_m"]),
        "ndvi_buffer_max_m": float(s["ndvi_buffer_max_m"]),
        "ndvi_buffer_step_m": float(s["ndvi_buffer_step_m"]),
        "gvi_ladder_spacing": s["gvi_ladder_spacing"],
        "gvi_ladder_rungs": int(s["gvi_ladder_rungs"]),
        "ndvi_ladder_spacing": s["ndvi_ladder_spacing"],
        "ndvi_ladder_rungs": int(s["ndvi_ladder_rungs"]),
        "ndvi_resolution_m": None,
        "gvi_grid_spacing_m": None,
        "n_bins": int(s["n_bins"]),
        "cache_metrics": False,
        "test_size": float(s["test_size"]),
        "objective_metric": s["objective_metric"],
        "residualize_method": s["residualize_method"],
        "search_scoring_method": s["search_scoring_method"],
        "ndvi_start_date": _FUSION_NDVI_WINDOW[0],
        "ndvi_end_date": _FUSION_NDVI_WINDOW[1],
        "ndvi_project_id": None,
        "multi_objective_requested": bool(
            s["multi_objective_requested"] and vector and len(o["outcomes"]) > 1
        ),
        "target_display_name": os.path.basename(o["target"]),
        "resume_existing_study": bool(s["resume_existing_study"]),
        "cgi_formula": cgi_formulas.formula_for(
            _settings.CHANNEL_SETS[s["channel_set"]], form
        ),
        "covariate_columns": covariates,
        "covariate_types": {
            c: ("categorical" if c in cats else "numeric") for c in covariates
        },
        "categorical_missing": s["categorical_missing"],
        "moderator_columns": list(o["moderators"]),
        "negative_controls": controls if vector else [],
        "exposure_iqr": s["exposure_iqr"],
        "standalone_channels": (
            list(_settings.CHANNEL_SETS[s["channel_set"]])
            if s["run_standalones"]
            else []
        ),
        "longitudinal_spec_payload": _longitudinal_payload(spec),
        "cgi_grid_spacing_m": int(s["cgi_grid_spacing_m"]),
        "exposure_kernel": s["exposure_kernel"],
        "exposure_radius_m": float(s["exposure_radius_m"]),
        "exposure_plateau_m": float(s["exposure_plateau_m"]),
        "exposure_sigma_m": float(s["exposure_sigma_m"]),
        "whole_grid_scaling": bool(s["whole_grid_scaling"]) if vector else True,
        "area_balanced_split": bool(s["area_balanced_split"]) if polygon else False,
        "spatial_adjust_method": adjust,
        "spatial_adjust_max_df": int(
            s["spatial_adjust_max_df"]
            if adjusted
            else _settings.FUSION["spatial_adjust_max_df"].default
        ),
        "spatial_adjust_eps_m": s["spatial_adjust_eps_m"] if adjusted else None,
        "spatial_split": bool(blocks),
        "spatial_block_size_m": s["spatial_block_size_m"] if blocks else None,
        "n_spatial_blocks": None,
        "channel_set": s["channel_set"],
        "index_form": s["index_form"],
        "sweep_splits": int(s["sweep_splits"]),
        "discovery_reps": int(s["discovery_reps"]),
        "discovery_shuffles": int(s["discovery_shuffles"]),
        "gain_splits": int(s["gain_splits"]),
        "null_calibration_runs": int(s["null_calibration_runs"]),
        "posterior_draws": int(s["posterior_draws"]),
        "posterior_warmup": int(s["posterior_warmup"] or s["posterior_draws"]),
        "posterior_chains": int(s["posterior_chains"]),
        "radius_prior": s["radius_prior"],
        "gvi_min_coverage": float(s["gvi_min_coverage"]),
        "check_collinearity": bool(s["check_collinearity"]),
        "drop_collinear_channels": bool(
            s["check_collinearity"] and s["drop_collinear_channels"]
        ),
        "vif_threshold": float(s["vif_threshold"]),
    }


#: ``run_fusion`` options the fusion tab does not set; passed only when set
#: away from ``run_fusion``'s own defaults.
_FUSION_EXTRAS = {
    "negative_control_retune": False,
    "report_ci_bootstrap": None,
    "report_effects_bootstrap": 2000,
    "report_effects_permutations": 1000,
    "report_paired_bootstrap": 2000,
}


def _longitudinal_payload(spec: JobSpec) -> dict | None:
    """The tab's ``LongitudinalSpec`` payload for a year-aware or longitudinal
    run; ``None`` for a plain cross-sectional one."""
    from geofuse.longitudinal import LongitudinalSpec, validate_spec

    s, o = spec.settings, spec.options
    if o["mode"] == "cross-sectional":
        return None
    gvi, ndvi = dict(o["gvi_files"]), dict(o["ndvi_files"])
    greenery = {"veg": dict(gvi), "terrain": dict(gvi), "ndvi": ndvi}
    waves = tuple(o["waves_found"])
    if o["mode"] == "year-aware":
        lon = LongitudinalSpec(
            intake_mode="long",
            entity_id_col="_gf_entity",
            wave_labels=waves,
            wave_col="_gf_year",
            date_col=o["date_column"] or "",
            greenery_files=greenery,
            target_files_per_wave={},
            scoring_metric=s["objective_metric"],
            derive_wave_from_date=True,
        )
    else:
        by_year = o["assign_by"] == "year"
        if o["waves"]:
            first = o["waves"][0]
            files = {w["label"]: w["file"] for w in o["waves"]}
            columns = {
                w["label"]: {
                    "entity_col": w["entity_column"],
                    "date_col": w["date_column"],
                }
                for w in o["waves"]
            }
            entity, date_col, wave_col = (
                first["entity_column"],
                first["date_column"],
                None,
            )
        else:
            files, columns = {}, {}
            entity, date_col = o["entity_column"], o["date_column"]
            wave_col = None if by_year else o["wave_column"]
        lon = LongitudinalSpec(
            intake_mode="wide" if o["waves"] else "long",
            entity_id_col=entity or "",
            wave_labels=waves,
            wave_col=wave_col,
            date_col=date_col or "measurement_date",
            greenery_files=greenery,
            target_files_per_wave=files,
            target_columns_per_wave=columns,
            scoring_metric=s["objective_metric"],
            include_time_fixed_effect=bool(s["include_time_fixed_effect"]),
            random_slope_time=bool(s["random_slope_time"]),
            association_target=s["association_target"],
            include_wave_fixed_effects=bool(s["include_wave_fixed_effects"]),
            area_id_col=o["area_column"],
            decline_average_exposure=bool(s["decline_average_exposure"]),
            decline_exposure_change=bool(s["decline_exposure_change"]),
            derive_wave_from_date=by_year,
        )
    errors = validate_spec(lon)
    if errors:
        raise PlanError(
            f"{spec.step}: the run's longitudinal setup is invalid:\n- "
            + "\n- ".join(errors)
        )
    return lon.to_payload()


def _build_fusion(spec: JobSpec) -> Submission:
    from geofuse.vector_io import (
        file_size_mtime_fingerprint,
        geometry_sha256,
        read_vector_path,
    )

    s, o = spec.settings, spec.options
    run_config = fusion_run_config(spec)
    target, vector = o["target"], o["vector"]
    cross = o["mode"] == "cross-sectional"
    veg_path = o["gvi_files"] if cross else None
    ndvi_path = o["ndvi_files"] if cross else None
    sha = None
    if vector:
        layer = o["layer"] if target.lower().endswith((".gpkg", ".zip")) else None
        sha = geometry_sha256(
            read_vector_path(target, layer=layer)
            if layer is not None
            else read_vector_path(target)
        )
    params = {
        **run_config,
        "is_vector_target": vector,
        "outcome_columns": list(o["outcomes"]) if vector else [],
        "target_band": int(o["band"]),
        "target_layer": o["layer"],
        "geometry_sha256": sha,
        "target_path": target,
        "target_fingerprint": file_size_mtime_fingerprint(target),
        "gvi_path": veg_path,
        "gvi_fingerprint": file_size_mtime_fingerprint(veg_path),
        "ndvi_path": ndvi_path,
        "ndvi_fingerprint": file_size_mtime_fingerprint(ndvi_path),
        "has_api_key": False,
        "metric_mode": "Upload Files",
    }
    payload = params["longitudinal_spec_payload"]
    if payload is not None:
        fps: dict[str, dict[str, str]] = {
            "target": {},
            "veg": {},
            "terrain": {},
            "ndvi": {},
        }
        for w, p in (payload.get("target_files_per_wave") or {}).items():
            fps["target"][w] = file_size_mtime_fingerprint(p)
        for ch in ("veg", "terrain", "ndvi"):
            for w, p in (payload.get("greenery_files", {}).get(ch) or {}).items():
                fps[ch][w] = file_size_mtime_fingerprint(p)
        params["longitudinal_spec_payload"] = {**payload, "__file_fingerprints__": fps}
    extras = {
        key: s[key] for key, default in _FUSION_EXTRAS.items() if s[key] != default
    }
    run_kwargs = {
        "target_path": target,
        "target_features_geojson": tuple(o["outcomes"]) if vector else (),
        "target_band": int(o["band"]) if not vector else 1,
        "target_layer": o["layer"] if vector else None,
        "target_cleanup_dir": None,
        "target_cleanup_file": None,
        "veg_path": veg_path,
        "ndvi_path": ndvi_path,
        "output_dir": spec.output_dir,
        **run_config,
        **extras,
    }
    params.update(extras)
    return Submission(
        "fusion",
        spec.name,
        params,
        "submit_fusion_subprocess",
        {"run_kwargs": run_kwargs},
    )


def build(
    spec: JobSpec,
    *,
    model_path: str | None = None,
    api_key: str | None = None,
    pano_cache_db_path: str | None = None,
    ee_project: str | None = None,
) -> Submission:
    """The record and executor call the tab would make for ``spec``.

    ``api_key`` is the Street View key for an ``api`` download (never part of
    a spec); ``model_path`` defaults to the repository's segmentation model;
    ``ee_project`` is the Earth Engine project NDVI jobs initialise with
    (passed to the executor only when set, as the tab never sets one).
    """
    if spec.kind in ("ndvi", "ndvi_column"):
        return _build_ndvi(spec, ee_project)
    if spec.kind in ("gvi", "gvi_column", "gvi_merge"):
        from geofuse.persistence.caches import pano_db

        key = api_key if spec.settings["download_mode"] == "api" else None
        if spec.settings["download_mode"] == "api" and not key:
            raise PlanError(NO_STREETVIEW_KEY)
        builder = _build_gvi_merge if spec.kind == "gvi_merge" else _build_gvi
        return builder(
            spec,
            model_path=model_path or DEFAULT_MODEL_PATH,
            api_key=key,
            pano_cache_db_path=pano_cache_db_path or pano_db(),
        )
    if spec.kind == "fusion":
        return _build_fusion(spec)
    raise PlanError(f"Unknown job kind {spec.kind!r}.")
