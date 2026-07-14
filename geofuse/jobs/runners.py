"""Streamlit-free worker callables for the JobExecutor.

Each runner has the signature ``run_xxx(ctx, *args, **kwargs)``. The first
argument is always a :class:`~geofuse.persistence.job_executor.JobContext`,
used for progress publication, heartbeats, and cancellation polling. Runners
never touch the ``JobStore`` directly and never import ``streamlit``.

These are direct lifts of the workers that used to live in ``ui/tabs/``:

* ``run_gvi``         ← ``_job_worker``         in ``ui/tabs/gvi.py``
* ``run_ndvi``        ← ``_ndvi_worker``        in ``ui/tabs/ndvi.py``
* ``run_ndvi_column`` ← ``_ndvi_column_worker`` in ``ui/tabs/ndvi.py``
* ``run_fusion``      ← ``_fusion_worker``      in ``ui/tabs/fusion.py``
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from geofuse.crs_utils import (
    default_geotiff_creation_options,
    raster_geographic_bounds,
    reproject_geodataframe_to_wgs84,
)
from geofuse.gvi import GVIEngine
from geofuse.jobs import progress_interval_s
from geofuse.jobs.stage_ledger import DONE, RUNNING, SKIPPED, StageLedger
from geofuse.logger import get_logger
from geofuse.longitudinal import (
    GREENERY_CHANNELS,
)
from geofuse.longitudinal import MIXEDLM_METRICS as _LON_MIXEDLM_METRICS
from geofuse.longitudinal import (
    LongitudinalSpec,
)
from geofuse.mixedlm_postscore import (
    compute_post_metrics as _compute_mixedlm_post_metrics,
)
from geofuse.ndvi import NDVIEngine
from geofuse import pdcor as _pdcor_mod
from geofuse.persistence.job_executor import JobContext
from geofuse.raster_sampling import sample_raster_at_features
from geofuse.vision import get_best_device

_log_gvi = get_logger("GVI")
_log_ndvi = get_logger("NDVI")
_log_fusion = get_logger("FUSION")

# ---------------------------------------------------------------------------
# GVI engine cache (replaces @st.cache_resource _get_gvi_engine)
# ---------------------------------------------------------------------------

_engine_lock = threading.Lock()
_engine_cache: dict[tuple[str, str | None], GVIEngine] = {}


def _get_gvi_engine(model_path: str, api_key: str | None) -> GVIEngine:
    """Memoized GVIEngine. Expensive to build (loads model weights)."""
    key = (model_path, api_key)
    with _engine_lock:
        engine = _engine_cache.get(key)
        if engine is None:
            best_device = get_best_device()
            engine = GVIEngine(
                model_path=model_path, device=str(best_device), api_key=api_key
            )
            _engine_cache[key] = engine
        return engine


# ---------------------------------------------------------------------------
# GVI runner
# ---------------------------------------------------------------------------


def run_gvi(
    ctx: JobContext,
    *,
    fname: str,
    dataset_data: dict,
    init_args: dict,
    run_args: dict,
    output_dir: str,
    save_geotiff: bool,
    save_geojson: bool,
    gpu_lock: threading.Lock,
    save_gpkg: bool = True,
) -> dict:
    """Run a GVI analysis. Mirrors the previous ``_job_worker`` in ``ui/tabs/gvi.py``."""
    ctx.progress(status_text="Waiting for GPU...")
    with gpu_lock:
        if ctx.is_cancelled():
            return {"output_paths": []}

        ctx.progress(status_text="Initializing...")
        engine = _get_gvi_engine(init_args["model_path"], init_args.get("api_key"))

    current_accumulated = dataset_data["accumulated"]
    start_idx = len(current_accumulated)
    results_lock = threading.Lock()

    # Throttle progress callbacks via the shared :func:`progress_interval_s`
    # so the JobStore lock stays cheap on big runs. The final point always
    # emits so the bar reaches 100 %.
    _last_progress_t = {"v": 0.0}

    def on_progress(curr: int, total: int) -> None:
        if total <= 0:
            return
        if curr < total:
            now = time.monotonic()
            if now - _last_progress_t["v"] < progress_interval_s(total):
                return
        _last_progress_t["v"] = time.monotonic()
        ctx.progress(
            value=min(curr / total, 1.0),
            status_text=f"Processing ({curr}/{total})",
        )
        ctx.heartbeat()

    def on_result(res) -> None:
        with results_lock:
            dataset_data["accumulated"].append(res)

    def check_cancel() -> bool:
        return ctx.is_cancelled()

    ctx.progress(status_text="Running")

    engine.run_analysis(
        dataset_data["processed"],
        step=run_args["step"],
        folder=output_dir,
        save_panos=run_args["save_panos"],
        save_masks=run_args["save_masks"],
        external_cache=dataset_data["cache_ref"],
        progress_callback=on_progress,
        result_callback=on_result,
        cancel_callback=check_cancel,
        start_index=start_idx,
        target_year=run_args.get("target_year"),
        max_year_diff=run_args.get("max_year_diff"),
    )

    if ctx.is_cancelled():
        return {"output_paths": []}

    ctx.progress(value=1.0, status_text="Writing outputs")

    output_paths, res_df = _write_gvi_outputs(
        accumulated=dataset_data["accumulated"],
        processed_crs=dataset_data["processed"].crs,
        meta=dataset_data.get("meta") or {},
        out_name=os.path.splitext(fname)[0],
        output_dir=output_dir,
        save_gpkg=save_gpkg,
        save_geotiff=save_geotiff,
        save_geojson=save_geojson,
    )
    dataset_data["results"] = res_df
    return {"output_paths": output_paths}


def _write_gvi_outputs(
    *,
    accumulated: list,
    processed_crs,
    meta: dict,
    out_name: str,
    output_dir: str,
    save_gpkg: bool,
    save_geotiff: bool,
    save_geojson: bool,
) -> tuple[list[str], gpd.GeoDataFrame]:
    """Write a GVI result set (GeoPackage / GeoJSON / per-cluster GeoTIFF).

    Shared by :func:`run_gvi` and :func:`run_gvi_column`; returns the written
    paths and the result GeoDataFrame (reprojected to the grid's planar CRS).
    """
    res_df = gpd.GeoDataFrame(accumulated, crs=processed_crs)
    if "orig_index" in res_df.columns:
        res_df.set_index("orig_index", inplace=True)
        res_df.index.name = None
    if res_df.crs is None:
        res_df = res_df.set_crs("EPSG:4326")

    output_paths: list[str] = []
    grid_crs_wkt = meta.get("grid_crs_wkt")
    clusters = meta.get("clusters") or []

    # All GVI outputs land in the toolbox-selected planar CRS so cells stay
    # square in metres across the entire study area. Falls back to EPSG:4326
    # only when the runner is invoked without clustered-grid metadata
    # (direct API callers with point inputs + buffer=0).
    if grid_crs_wkt:
        res_df = res_df.to_crs(grid_crs_wkt)

    if save_gpkg:
        gpkg_path = os.path.join(output_dir, f"{out_name}_gvi.gpkg")
        res_df.to_file(gpkg_path, driver="GPKG", layer="gvi_samples")
        output_paths.append(gpkg_path)

    if save_geojson:
        gj_path = os.path.join(output_dir, f"{out_name}_gvi.geojson")
        reproject_geodataframe_to_wgs84(res_df).to_file(gj_path, driver="GeoJSON")
        output_paths.append(gj_path)
        if len(res_df) > 100_000:
            _log_gvi(
                "WARN",
                f"GeoJSON output is large ({len(res_df):,} points); "
                f"GeoPackage is preferred for re-reading.",
            )

    if save_geotiff:
        if not clusters:
            _log_gvi(
                "WARN",
                "GeoTIFF requested but no cluster metadata is available "
                "(e.g. point input with buffer=0); skipping. Use GeoPackage.",
            )
        else:
            tiles_dir = os.path.join(output_dir, f"{out_name}_gvi_tiles")
            os.makedirs(tiles_dir, exist_ok=True)
            has_cluster_col = "cluster_id" in res_df.columns
            index_entries: list[dict] = []
            for cluster in clusters:
                cid = int(cluster["cluster_id"])
                h = int(cluster["height"])
                w = int(cluster["width"])
                arr_veg = np.full((h, w), np.nan, dtype=np.float32)
                arr_ter = np.full((h, w), np.nan, dtype=np.float32)
                if has_cluster_col:
                    cdf = res_df[res_df["cluster_id"] == cid].dropna(subset=["gvi_veg"])
                    if not cdf.empty:
                        lr = (cdf["row"].to_numpy() - cluster["row_min"]).astype(int)
                        lc = (cdf["col"].to_numpy() - cluster["col_min"]).astype(int)
                        keep = (lr >= 0) & (lr < h) & (lc >= 0) & (lc < w)
                        lr = lr[keep]
                        lc = lc[keep]
                        arr_veg[lr, lc] = cdf["gvi_veg"].to_numpy()[keep]
                        arr_ter[lr, lc] = cdf["gvi_ter"].to_numpy()[keep]
                tile_path = os.path.join(tiles_dir, f"cluster_{cid:04d}.tif")
                # SPARSE_OK=TRUE drops all-NaN blocks — GVI tiles are largely
                # empty along street networks, so this is a significant
                # disk-size win on top of DEFLATE compression.
                with rasterio.open(
                    tile_path,
                    "w",
                    driver="GTiff",
                    height=h,
                    width=w,
                    count=2,
                    dtype=np.float32,
                    crs=grid_crs_wkt,
                    transform=cluster["transform"],
                    nodata=np.nan,
                    **default_geotiff_creation_options(np.float32, sparse=True),
                ) as dst:
                    dst.write(arr_veg, 1)
                    dst.set_band_description(1, "Veg")
                    dst.write(arr_ter, 2)
                    dst.set_band_description(2, "Ter")
                bounds_4326 = raster_geographic_bounds(tile_path)
                index_entries.append(
                    {
                        "cluster_id": cid,
                        "path": os.path.basename(tile_path),
                        "bbox_grid_crs": list(cluster["bbox_grid_crs"]),
                        "bounds_4326": (
                            list(bounds_4326) if bounds_4326 is not None else None
                        ),
                        "height": h,
                        "width": w,
                        "row_min": int(cluster["row_min"]),
                        "col_min": int(cluster["col_min"]),
                    }
                )
            index_path = os.path.join(tiles_dir, "tiles_index.json")
            with open(index_path, "w") as f:
                json.dump(
                    {
                        "grid_crs_wkt": grid_crs_wkt,
                        "step_m": meta.get("step_m"),
                        "anchor_x": meta.get("anchor_x"),
                        "anchor_y": meta.get("anchor_y"),
                        "tiles": index_entries,
                    },
                    f,
                    indent=2,
                )
            output_paths.append(tiles_dir)

    # Sidecar JSON next to the canonical GeoPackage for grid reconstruction.
    if save_gpkg and meta:
        sidecar_path = os.path.join(output_dir, f"{out_name}_gvi.json")
        with open(sidecar_path, "w") as f:
            json.dump(
                {
                    "grid_crs_wkt": grid_crs_wkt,
                    "step_m": meta.get("step_m"),
                    "anchor_x": meta.get("anchor_x"),
                    "anchor_y": meta.get("anchor_y"),
                    "distortion": meta.get("distortion"),
                    "choice_name": meta.get("choice_name"),
                    "n_clusters": len(clusters),
                },
                f,
                indent=2,
            )

    return output_paths, res_df


def run_gvi_column(
    ctx: JobContext,
    *,
    fname: str,
    dataset_data: dict,
    date_column: str,
    init_args: dict,
    run_args: dict,
    output_dir: str,
    save_geotiff: bool,
    save_geojson: bool,
    gpu_lock: threading.Lock,
    save_gpkg: bool = True,
) -> dict:
    """Run one GVI analysis per year present in ``date_column``.

    The input is split by year; each year is processed as a standalone GVI job
    over only that year's features, with ``target_year`` set to that year so
    every sampling point uses the Street View capture nearest it. The grid CRS
    is chosen once from the whole dataset — before the split — so all years'
    grids share one planar CRS and align. Results land in a
    ``{name}_temporal_gvi/`` folder, one set of files per year.
    """
    from geofuse.core import generate_clustered_grid
    from geofuse.crs_utils import select_grid_crs_with_warning
    from geofuse.longitudinal import parse_date_column

    raw = dataset_data["raw"]
    gdf_4326 = (
        raw
        if raw.crs is not None and raw.crs.is_geographic
        else raw.to_crs("EPSG:4326")
    ).copy()
    parsed = parse_date_column(gdf_4326[date_column])
    gdf_4326["_year"] = parsed.dt.year
    gdf_4326 = gdf_4326.dropna(subset=["_year"])
    if gdf_4326.empty:
        raise ValueError(
            f"No valid years could be parsed from column '{date_column}'."
        )
    gdf_4326["_year"] = gdf_4326["_year"].astype(int)

    unique_years = sorted(gdf_4326["_year"].unique())
    n_years = len(unique_years)

    # Grid CRS chosen once from the whole dataset so every per-year grid aligns.
    grid_crs, _distortion, _choice = select_grid_crs_with_warning(
        gdf_4326, _log_gvi, role="Grid CRS"
    )

    base_name = os.path.splitext(fname)[0]
    job_folder = os.path.join(output_dir, f"{base_name}_temporal_gvi")
    os.makedirs(job_folder, exist_ok=True)

    ctx.progress(status_text="Waiting for GPU...")
    with gpu_lock:
        if ctx.is_cancelled():
            return {"output_paths": []}
        ctx.progress(status_text="Initializing...")
        engine = _get_gvi_engine(init_args["model_path"], init_args.get("api_key"))

    step = run_args["step"]
    buffer_m = float(run_args.get("buffer", 0) or 0)
    max_year_diff = run_args.get("max_year_diff")
    output_paths: list[str] = [job_folder]

    def check_cancel() -> bool:
        return ctx.is_cancelled()

    for idx, year in enumerate(unique_years):
        if ctx.is_cancelled():
            return {"output_paths": output_paths}

        year_gdf = gdf_4326[gdf_4326["_year"] == year].drop(columns=["_year"])
        is_poly = year_gdf.geometry.iloc[0].geom_type in ("Polygon", "MultiPolygon")
        if is_poly or buffer_m > 0:
            pts, meta = generate_clustered_grid(
                year_gdf, buffer_m=buffer_m, step_m=float(step), grid_crs=grid_crs
            )
        else:
            pts, meta = year_gdf.copy(), None

        base = idx / max(n_years, 1)
        span = 1.0 / max(n_years, 1)

        def on_progress(curr: int, total: int, _b=base, _s=span, _y=year) -> None:
            if total <= 0:
                return
            ctx.progress(
                value=min(_b + _s * (curr / total), 1.0),
                status_text=f"Year {_y} ({curr}/{total})",
            )
            ctx.heartbeat()

        accumulated: list = []
        results_lock = threading.Lock()

        def on_result(res, _acc=accumulated, _lock=results_lock) -> None:
            with _lock:
                _acc.append(res)

        ctx.progress(status_text=f"Processing year {idx + 1}/{n_years}: {year}")
        engine.run_analysis(
            pts,
            step=step,
            folder=job_folder,
            save_panos=run_args["save_panos"],
            save_masks=run_args["save_masks"],
            external_cache=dataset_data["cache_ref"],
            progress_callback=on_progress,
            result_callback=on_result,
            cancel_callback=check_cancel,
            target_year=int(year),
            max_year_diff=max_year_diff,
        )
        if ctx.is_cancelled():
            return {"output_paths": output_paths}
        if not accumulated:
            _log_gvi("WARN", f"Year {year}: no results produced.")
            continue

        paths, _res_df = _write_gvi_outputs(
            accumulated=accumulated,
            processed_crs=pts.crs,
            meta=meta or {},
            out_name=f"{base_name}_{year}",
            output_dir=job_folder,
            save_gpkg=save_gpkg,
            save_geotiff=save_geotiff,
            save_geojson=save_geojson,
        )
        output_paths.extend(paths)

    ctx.progress(value=1.0, status_text="Completed")
    return {"output_paths": output_paths}


# ---------------------------------------------------------------------------
# NDVI runners
# ---------------------------------------------------------------------------


def _ndvi_on_progress_factory(
    ctx: JobContext, base_offset: float = 0.0, span: float = 1.0
):
    """Build a callback that maps NDVIEngine progress dicts → ``ctx.progress``."""

    def on_ndvi_progress(d: Mapping[str, object]) -> None:
        extras: dict[str, Any] = {}
        prog_value = None
        status_text = None
        if "sub_progress" in d:
            prog_value = base_offset + span * float(d["sub_progress"])  # type: ignore[arg-type]
        if "phase" in d:
            status_text = str(d["phase"])
        if "tiles" in d:
            t = d["tiles"]
            if isinstance(t, tuple) and len(t) == 2:
                k, n = int(t[0]), int(t[1])
                if n > 0:
                    extras["ndvi_tile_bracket"] = f"[{k}/{n}]"
        if d.get("clear_bracket"):
            extras["ndvi_tile_bracket"] = None
        if prog_value is not None or status_text is not None or extras:
            ctx.progress(value=prog_value, status_text=status_text, **extras)
        ctx.heartbeat()

    return on_ndvi_progress


def run_ndvi(
    ctx: JobContext,
    *,
    fname: str,
    dataset_data: dict,
    start_date: str,
    end_date: str,
    cloud_pct: int,
    resolution: int,
    buffer_m: int,
    output_name: str,
    output_dir: str,
    save_geotiff: bool,
    save_geojson: bool,
    save_gpkg: bool = False,
    save_cluster_tiles: bool = False,
    satellite: str = "auto",
    coverage_rescue: bool = True,
) -> dict:
    """Run NDVI for a single date range."""
    from geofuse.crs_utils import buffer_gdf_union_metres

    ctx.progress(status_text="Initializing Earth Engine...")
    engine = NDVIEngine()
    geometry = buffer_gdf_union_metres(dataset_data["raw"], buffer_m)

    on_progress = _ndvi_on_progress_factory(ctx)

    def check_cancel() -> bool:
        return ctx.is_cancelled()

    result = engine.download_and_process(
        geometry=geometry,
        start_date=start_date,
        end_date=end_date,
        output_name=output_name,
        cloud_max=cloud_pct,
        resolution=resolution,
        folder=output_dir,
        cancel_callback=check_cancel,
        ndvi_progress_callback=on_progress,
        write_geotiff=save_geotiff,
        write_geojson=save_geojson,
        write_geopackage=save_gpkg,
        write_cluster_tiles=save_cluster_tiles,
        satellite=satellite,
        coverage_rescue=coverage_rescue,
    )

    if result.get("status") == "cancelled":
        return {"output_paths": []}
    if result.get("status") != "success":
        raise RuntimeError(result.get("message", "NDVI run failed"))

    output_paths: list[str] = []
    tif_path = os.path.join(output_dir, f"{output_name}_ndvi.tif")
    if save_geotiff and os.path.exists(tif_path):
        output_paths.append(tif_path)
    gpkg_path = os.path.join(output_dir, f"{output_name}_ndvi.gpkg")
    if save_gpkg and os.path.exists(gpkg_path):
        output_paths.append(gpkg_path)
    gj_path = os.path.join(output_dir, f"{output_name}_ndvi.geojson")
    if save_geojson and os.path.exists(gj_path):
        output_paths.append(gj_path)
    sidecar_path = os.path.join(output_dir, f"{output_name}_ndvi.json")
    if os.path.exists(sidecar_path):
        output_paths.append(sidecar_path)
    cluster_tiles_dir = os.path.join(output_dir, f"{output_name}_ndvi_tiles")
    if save_cluster_tiles and os.path.isdir(cluster_tiles_dir):
        output_paths.append(cluster_tiles_dir)

    ctx.progress(value=1.0, status_text="Completed")
    return {"output_paths": output_paths}


def run_ndvi_column(
    ctx: JobContext,
    *,
    fname: str,
    dataset_data: dict,
    date_column: str,
    season_start_month: int,
    season_end_month: int,
    cloud_pct: int,
    resolution: int,
    buffer_m: int,
    output_dir: str,
    save_geotiff: bool,
    save_geojson: bool,
    save_gpkg: bool = False,
    save_cluster_tiles: bool = False,
    satellite: str = "auto",
    coverage_rescue: bool = True,
) -> dict:
    """Run one NDVI raster per year present in ``date_column``.

    The input is split by year; each year is processed as a standalone NDVI
    job over only that year's features, composited across the growing-season
    months ``[season_start_month, season_end_month]`` of that year. The export
    CRS is chosen once from the whole dataset — before the split — so every
    year's raster snaps to the same global pixel grid and the outputs align.
    Results land in a ``{name}_temporal_ndvi/`` folder, one set of files per
    year (``{name}_{year}_ndvi.tif`` plus optional GeoPackage/GeoJSON).
    """
    import calendar

    from geofuse.crs_utils import buffer_gdf_union_metres
    from geofuse.longitudinal import parse_date_column

    gdf = dataset_data["raw"].copy()
    parsed = parse_date_column(gdf[date_column])
    gdf["_year"] = parsed.dt.year
    gdf = gdf.dropna(subset=["_year"])
    if gdf.empty:
        raise ValueError(
            f"No valid years could be parsed from column '{date_column}'."
        )
    gdf["_year"] = gdf["_year"].astype(int)

    unique_years = sorted(gdf["_year"].unique())
    n_years = len(unique_years)

    # CRS chosen once from the whole dataset so every per-year raster aligns.
    crs_override = NDVIEngine.compute_export_crs(gdf)

    base_name = os.path.splitext(fname)[0]
    job_folder = os.path.join(output_dir, f"{base_name}_temporal_ndvi")
    os.makedirs(job_folder, exist_ok=True)

    engine = NDVIEngine()
    output_paths: list[str] = [job_folder]

    def check_cancel() -> bool:
        return ctx.is_cancelled()

    for idx, year in enumerate(unique_years):
        if ctx.is_cancelled():
            return {"output_paths": output_paths}

        year_gdf = gdf[gdf["_year"] == year].drop(columns=["_year"])
        geometry = buffer_gdf_union_metres(year_gdf, buffer_m)

        start_d = f"{year:04d}-{season_start_month:02d}-01"
        last_day = calendar.monthrange(int(year), int(season_end_month))[1]
        end_d = f"{year:04d}-{season_end_month:02d}-{last_day:02d}"
        output_name = f"{base_name}_{year}"

        ctx.progress(
            status_text=f"Processing year {idx + 1}/{n_years}: {year} "
            f"({start_d} → {end_d})"
        )
        span = 1.0 / max(n_years, 1)
        base = idx / max(n_years, 1)
        on_progress = _ndvi_on_progress_factory(ctx, base_offset=base, span=span)

        result = engine.download_and_process(
            geometry=geometry,
            start_date=start_d,
            end_date=end_d,
            output_name=output_name,
            cloud_max=cloud_pct,
            resolution=resolution,
            folder=job_folder,
            cancel_callback=check_cancel,
            ndvi_progress_callback=on_progress,
            write_geotiff=save_geotiff,
            write_geojson=save_geojson,
            write_geopackage=save_gpkg,
            write_cluster_tiles=save_cluster_tiles,
            satellite=satellite,
            coverage_rescue=coverage_rescue,
            crs_override=crs_override,
        )
        if result.get("status") == "cancelled":
            return {"output_paths": output_paths}
        if result.get("status") != "success":
            _log_ndvi(
                "WARN", f"Year {year} failed: {result.get('message')}"
            )
            continue

        for suffix, enabled in (
            ("_ndvi.tif", save_geotiff),
            ("_ndvi.gpkg", save_gpkg),
            ("_ndvi.geojson", save_geojson),
            ("_ndvi.json", True),
        ):
            p = os.path.join(job_folder, f"{output_name}{suffix}")
            if enabled and os.path.exists(p):
                output_paths.append(p)
        tiles_dir = os.path.join(job_folder, f"{output_name}_ndvi_tiles")
        if save_cluster_tiles and os.path.isdir(tiles_dir):
            output_paths.append(tiles_dir)

    ctx.progress(value=1.0, status_text="Completed")
    return {"output_paths": output_paths}


# ---------------------------------------------------------------------------
# Fusion runner
# ---------------------------------------------------------------------------


_STUDY_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _build_fusion_study_name(
    target_display_name: str,
    label: str,
    objective_metric: str,
    suffix: str = "",
    *,
    config_fingerprint: str | None = None,
) -> str:
    """Filesystem-safe Optuna ``study_name`` (also the SQLite filename stem).

    ``config_fingerprint`` lets the caller bake an arbitrary short hash
    of the run config (buffer ladders, CGI formula, covariates, …) into
    the name so a config change starts a brand-new study instead of
    silently appending trials to the previous study's SQLite file.
    Without the fingerprint, e.g. a buffer-max change from 800 m to
    1500 m would mix old long-radius trials into the new pool and the
    rerun's reports / robust trials would be contaminated.
    """
    stem = os.path.splitext(target_display_name or "target")[0]
    parts = [stem, str(label), objective_metric]
    if config_fingerprint:
        parts.append(config_fingerprint)
    if suffix:
        parts.append(suffix)
    raw = "__".join(parts)
    return _STUDY_NAME_UNSAFE.sub("_", raw).strip("_") or "fusion_study"


def _fusion_config_fingerprint(
    *,
    buffer_meters: float,
    gvi_buffer_min_m: float,
    gvi_buffer_max_m: float,
    gvi_buffer_step_m: float,
    ndvi_buffer_min_m: float,
    ndvi_buffer_max_m: float,
    ndvi_buffer_step_m: float,
    cgi_formula: str,
    covariate_columns: list[str] | None,
    whole_grid_scaling: bool,
    cgi_grid_spacing_m: float | None,
    area_balanced_split: bool,
    test_size: float,
    covariate_types: dict[str, str] | None = None,
    spatial_split: bool = False,
    spatial_block_size_m: float | None = None,
    n_spatial_blocks: int | None = None,
    normalize_channels: bool = False,
    spatial_adjust_method: str = "none",
    spatial_adjust_max_df: int = 10,
    spatial_adjust_eps_m: float | None = None,
) -> str:
    """8-char hex hash of every setting that changes the search space / split.

    Anything that affects which params are suggested or which entities are
    sampled lives here; purely cosmetic knobs (standalone toggle, bootstrap
    counts) are excluded so they don't fragment the per-job caches.
    """
    import hashlib as _hl

    payload = "|".join(
        [
            f"bm:{buffer_meters:.1f}",
            f"gvi:{gvi_buffer_min_m:.1f}-{gvi_buffer_max_m:.1f}@{gvi_buffer_step_m:.1f}",
            f"ndvi:{ndvi_buffer_min_m:.1f}-{ndvi_buffer_max_m:.1f}@{ndvi_buffer_step_m:.1f}",
            f"fmla:{cgi_formula}",
            f"cov:{','.join(sorted(covariate_columns or []))}",
            f"wg:{int(whole_grid_scaling)}",
            f"grid:{cgi_grid_spacing_m if cgi_grid_spacing_m is not None else 'na'}",
            f"ab:{int(area_balanced_split)}",
            f"ts:{test_size:.3f}",
            f"sp:{int(spatial_split)}",
            f"spb:{spatial_block_size_m if spatial_block_size_m is not None else 'na'}",
            f"spn:{n_spatial_blocks if n_spatial_blocks is not None else 'na'}",
            # Bump when the search space / selection scheme changes so old
            # per-job caches can't pool with a new run.
            "sel:stabsel-cpss-v3",
        ]
        # Only appended when on, so legacy (un-normalized) runs keep their hash.
        + (["nc:1"] if normalize_channels else [])
        # Spatial-confounding adjustment changes the objective, so it splits the
        # cache; only appended when on so legacy runs keep their hash.
        + (
            [
                f"spadj:{spatial_adjust_method}@{int(spatial_adjust_max_df)}@"
                f"{spatial_adjust_eps_m if spatial_adjust_eps_m is not None else 'auto'}"
            ]
            if spatial_adjust_method and spatial_adjust_method != "none"
            else []
        )
        # Categorical covariates change the design matrix (one-hot dummies), so
        # they split the cache; only appended when any covariate is categorical
        # so legacy (all-numeric) runs keep their hash.
        + (
            [
                "covt:"
                + ",".join(
                    f"{k}={v}"
                    for k, v in sorted((covariate_types or {}).items())
                    if str(v).lower() == "categorical"
                )
            ]
            if any(
                str(v).lower() == "categorical"
                for v in (covariate_types or {}).values()
            )
            else []
        )
    )
    return _hl.sha256(payload.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Fusion pipeline / stage-ledger helpers
# ---------------------------------------------------------------------------


# Ordered pipeline steps a fusion run moves through per target outcome. The
# staged-resume ledger (geofuse.jobs.stage_ledger) records each so the monitor
# shows where a run is and a stopped job reports where it left off. Resume itself
# is content-addressed (metric cache, pre-aggregation cache), so re-running the
# same job reuses/resumes each on-disk artifact transparently — the ledger is
# the visibility layer over that durability. The heavy stages map one-to-one
# onto real work: ``optimize`` is the bootstrap stability search (the dominant
# compute, and where the "k/N trials" sub-bar lives), ``evaluate`` scores the
# winning weights once on the held-out test set (fast), and ``report_stats``
# runs the replicate statistics — test-set bootstrap CIs, effect sizes, and
# permutation tests — that form the long tail users otherwise misread as
# "evaluating".
_FUSION_STAGE_STEPS: tuple[tuple[str, str], ...] = (
    ("load_target", "Load target"),
    ("load_metrics", "Load metric maps"),
    ("preaggregate", "Spatial pre-processing"),
    ("split", "Split train / test folds"),
    ("optimize", "Stability selection (bootstrap search)"),
    ("evaluate", "Score held-out test set"),
    ("report_stats", "Bootstrap CIs, effects & permutation tests"),
    ("apply", "Apply fusion weights"),
    ("reports", "Generate reports and composite map"),
)

# Relative wall-time weights for the main progress bar. Bootstrap searches
# dominate a run; the replicate-statistics tail is the next largest cost, while
# I/O and apply stages are comparatively instant. Weighting keeps the ledger-
# derived bar monotonic *and* roughly time-proportional instead of leaping to
# ~50 % the moment the fast setup stages finish.
_FUSION_STAGE_WEIGHTS: dict[str, float] = {
    "load_target": 1.0,
    "load_metrics": 2.0,
    "prepare_longitudinal": 1.0,
    "preaggregate": 3.0,
    "split": 1.0,
    "optimize": 20.0,
    "evaluate": 1.0,
    "report_stats": 8.0,
    "apply": 1.0,
    "mixedlm_postscore": 2.0,
    "reports": 1.0,
}
_FUSION_STANDALONE_SEARCH_WEIGHT = 20.0
_FUSION_STANDALONE_REPORT_WEIGHT = 8.0


def _fusion_stage_weight(stage_key: str) -> float:
    """Relative wall-time weight of a ledger stage key (see _FUSION_STAGE_WEIGHTS).

    Strips the ``"<label>::"`` multi-outcome prefix, then maps standalone
    search / report keys onto their dedicated weights and everything else onto
    the per-step table. Unknown steps default to unit weight.
    """
    step = stage_key.split("::", 1)[1] if "::" in stage_key else stage_key
    if step.startswith("standalone_"):
        return (
            _FUSION_STANDALONE_REPORT_WEIGHT
            if step.endswith("_report")
            else _FUSION_STANDALONE_SEARCH_WEIGHT
        )
    return _FUSION_STAGE_WEIGHTS.get(step, 1.0)

# Mixed-effects fusion inserts an extra step before pre-aggregation: load
# the per-wave target frames (wide intake only) and the per-wave greenery
# files for every channel, then hand them to the engine. The cross-
# sectional pipeline skips this stage.
_FUSION_LONGITUDINAL_STAGE: tuple[str, str] = (
    "prepare_longitudinal",
    "Load longitudinal data",
)

# After the final ``apply`` stage, mixed-effects runs also score every
# robust + top-X% trial + the averaged-composite parameters on the held-out
# test set with all four ``mixedlm_*`` metrics, and write the results to
# ``mixedlm_metrics.csv`` for downstream analysis.
_FUSION_MIXEDLM_POSTSCORE_STAGE: tuple[str, str] = (
    "mixedlm_postscore",
    "Score all MixedLM metrics on robust + top trials",
)


def _load_longitudinal_metric_file(path: str, channel: str) -> Any:
    """Read one per-wave metric file into the layout the engine expects.

    Returns either a GeoDataFrame (vector metric) or the raster-dict layout
    ``{"data", "transform", "crs", "bounds", "width", "height"}`` that
    matches what ``MetricFusionEngine.load_metrics`` builds for the cross-
    sectional case. Multi-band GVI rasters split into ``veg`` + ``terrain``
    point GeoDataFrames per ``GVIEngine`` convention (band 1 = vegetation,
    band 2 = terrain); the caller picks which to keep based on ``channel``.
    """
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.transform import xy

    if path.lower().endswith((".tif", ".tiff")):
        with rasterio.open(path) as src:
            n_bands = src.count
            if channel in ("veg", "terrain") and n_bands >= 2:
                band = 1 if channel == "veg" else 2
                arr = src.read(band)
                transform = src.transform
                crs = src.crs
                h, w = arr.shape
                rows_i, cols_i = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
                xs, ys = xy(
                    transform, rows_i.flatten(), cols_i.flatten(), offset="center"
                )
                vals = arr.flatten()
                mask = ~np.isnan(vals)
                gdf = gpd.GeoDataFrame(
                    {channel: vals[mask]},
                    geometry=gpd.points_from_xy(
                        np.asarray(xs)[mask], np.asarray(ys)[mask]
                    ),
                    crs=crs,
                )
                gdf.attrs["metric_column"] = channel
                return gdf
            data = src.read(1, masked=True)
            return {
                "data": data,
                "transform": src.transform,
                "crs": src.crs,
                "bounds": src.bounds,
                "width": src.width,
                "height": src.height,
            }
    # Vector formats (GPKG / GeoJSON / shapefile / zip)
    gdf = gpd.read_file(path)
    default_col = {"veg": "veg", "terrain": "terrain", "ndvi": "NDVI"}[channel]
    col = (
        default_col
        if default_col in gdf.columns
        else ("value" if "value" in gdf.columns else None)
    )
    if col is None:
        raise ValueError(
            f"Cannot find metric value column in {path} for channel {channel!r}. "
            f"Expected one of [{default_col!r}, 'value']; got columns "
            f"{list(gdf.columns)}."
        )
    gdf.attrs["metric_column"] = col
    return gdf


def _resolve_longitudinal_spec(
    payload: dict | None,
) -> LongitudinalSpec | None:
    """Reconstruct the dataclass from the runner kwarg, returning ``None`` for
    cross-sectional jobs."""
    if not payload:
        return None
    return LongitudinalSpec.from_payload(payload)


def _clean_params(params: dict | None) -> dict:
    """Drop the ``__*__`` stability-selection bookkeeping keys from a params dict."""
    if not params:
        return {}
    return {k: v for k, v in params.items() if not str(k).startswith("__")}


def _study_bundles(
    cgi_bundle: dict, standalones_bundle: dict
) -> list[tuple[str, str, dict]]:
    """``(study_key, display, bundle)`` for the CGI study then each standalone."""
    out: list[tuple[str, str, dict]] = [("cgi", "CGI (combined)", cgi_bundle)]
    for ch in ("veg", "terrain", "ndvi"):
        b = standalones_bundle.get(ch)
        if b:
            out.append((ch, _STANDALONE_CHANNEL_LABELS.get(ch, ch), b))
    return out


def _write_fusion_outputs(
    *,
    output_dir: str,
    label: str,
    multi_outcome: bool,
    objective_metric: str,
    formula_name: str,
    cgi_bundle: dict,
    standalones_bundle: dict,
    aic_bic: dict | None,
    covariate_impact: dict | None,
    collinearity_report: dict | None,
    run_config_record: dict | None,
    log,
) -> list[str]:
    """Persist every test result a fusion job produces to disk.

    Writes, under ``output_dir`` (``study_results/``), a machine-readable
    manifest plus tidy CSVs so each result is recorded both for replay and
    for spreadsheet analysis. ``__<label>`` is appended to every basename in
    multi-outcome runs. Returns the list of files written (best-effort: a
    failed individual write is logged and skipped, never fatal).

    Files (per outcome):

    - ``run_config.json`` — every setting the job ran with (fidelity record).
    - ``results_summary.json`` — nested manifest: each study's params, test
      score + CI, direction, subset scores, and stability stats, plus the
      CGI-vs-standalone AIC/BIC verdict, covariate-impact summary, and the
      collinearity report.
    - ``test_scores.csv`` — one headline row per study (test score, CI,
      direction).
    - ``scores.csv`` — long form: study × subset (train/val/test/all) ×
      score/score_raw/n.
    - ``parameters.csv`` — long form: study × param → value.
    - ``stability_cells.csv`` — the ranked weight cells per study.
    - ``stability_bootstraps.csv`` — the per-bootstrap leaderboard per study.
    - ``covariate_impact.csv`` — per-covariate effects (when covariates set).
    """
    import json

    import numpy as np
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    sfx = f"__{label}" if multi_outcome else ""
    written: list[str] = []
    studies = _study_bundles(cgi_bundle, standalones_bundle)

    def _path(name: str) -> str:
        return os.path.join(output_dir, name)

    def _f(v: Any) -> float | None:
        try:
            fv = float(v)
            return fv if np.isfinite(fv) else None
        except (TypeError, ValueError):
            return None

    def _ci(bundle: dict) -> dict:
        return (bundle.get("test_results") or {}).get("test_ci") or {}

    def _emit_json(name: str, payload: Any) -> None:
        try:
            p = _path(name)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            written.append(p)
        except Exception as exc:  # pragma: no cover - disk-IO guard
            log("WARN", f"[{label}] Could not write {name}: {exc}")

    def _emit_csv(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        try:
            p = _path(name)
            pd.DataFrame(rows).to_csv(p, index=False)
            written.append(p)
        except Exception as exc:  # pragma: no cover - disk-IO guard
            log("WARN", f"[{label}] Could not write {name}: {exc}")

    # ── run_config.json (fidelity record) ─────────────────────────────
    if run_config_record is not None:
        _emit_json(f"run_config{sfx}.json", run_config_record)

    # ── test_scores.csv ───────────────────────────────────────────────
    test_rows: list[dict] = []
    for key, _disp, b in studies:
        tr = b.get("test_results") or {}
        ci = _ci(b)
        test_rows.append(
            {
                "study": key,
                "metric": objective_metric,
                "test_score": _f(tr.get("test_score")),
                "ci_lower": _f(ci.get("lower")),
                "ci_upper": _f(ci.get("upper")),
                "direction": (
                    int(b["direction_sign"])
                    if b.get("direction_sign") is not None
                    else None
                ),
            }
        )
    _emit_csv(f"test_scores{sfx}.csv", test_rows)

    # ── scores.csv (every subset) ─────────────────────────────────────
    score_rows: list[dict] = []
    for key, _disp, b in studies:
        subsets = b.get("subset_scores") or {}
        for subset in ("train", "val", "test", "all"):
            block = subsets.get(subset) or {}
            if not block:
                continue
            score_rows.append(
                {
                    "study": key,
                    "subset": subset,
                    "metric": objective_metric,
                    "score": _f(block.get("score")),
                    "score_raw": _f(block.get("score_raw")),
                    "n": block.get("n"),
                }
            )
    _emit_csv(f"scores{sfx}.csv", score_rows)

    # ── parameters.csv (long form) ────────────────────────────────────
    param_rows: list[dict] = []
    for key, _disp, b in studies:
        params = _clean_params(b.get("averaged_params") or b.get("best_params"))
        for pname, pval in params.items():
            param_rows.append({"study": key, "param": pname, "value": pval})
    _emit_csv(f"parameters{sfx}.csv", param_rows)

    # ── stability_cells.csv + stability_bootstraps.csv ────────────────
    cell_rows: list[dict] = []
    bs_rows: list[dict] = []
    for key, _disp, b in studies:
        summ = b.get("stability_summary") or {}
        for rank, c in enumerate(summ.get("cell_stats") or [], start=1):
            row: dict = {"study": key, "rank": rank}
            for wk, wv in (c.get("weights") or {}).items():
                row[wk] = wv
            row["count"] = c.get("count")
            row["q_worst"] = _f(c.get("q_worst"))
            row["median"] = _f(c.get("median"))
            row["selection_probability"] = _f(c.get("selection_probability"))
            cell_rows.append(row)
        for entry in summ.get("per_bootstrap_summary") or []:
            row = {
                "study": key,
                "bootstrap": entry.get("bootstrap"),
                "n_trials": entry.get("n_trials"),
                "top_oob": _f(entry.get("top_oob")),
                "median_oob": _f(entry.get("median_oob")),
                "min_oob": _f(entry.get("min_oob")),
                "max_oob": _f(entry.get("max_oob")),
            }
            for pk, pv in (entry.get("top_params") or {}).items():
                row[pk] = pv
            bs_rows.append(row)
    _emit_csv(f"stability_cells{sfx}.csv", cell_rows)
    _emit_csv(f"stability_bootstraps{sfx}.csv", bs_rows)

    # ── covariate_impact.csv ──────────────────────────────────────────
    if covariate_impact and covariate_impact.get("per_covariate"):
        _emit_csv(
            f"covariate_impact{sfx}.csv",
            [dict(r) for r in covariate_impact["per_covariate"]],
        )

    # ── decline_terms.csv (longitudinal exposure × time) ──────────────
    decline_terms = (cgi_bundle or {}).get("decline_terms") or None
    if decline_terms and decline_terms.get("terms"):
        _emit_csv(
            f"decline_terms{sfx}.csv", [dict(r) for r in decline_terms["terms"]]
        )

    # ── results_summary.json (master manifest) ────────────────────────
    studies_manifest: dict[str, dict] = {}
    for key, disp, b in studies:
        tr = b.get("test_results") or {}
        summ = b.get("stability_summary") or {}
        studies_manifest[key] = {
            "display": disp,
            "channel": b.get("channel", "cgi"),
            "params": _clean_params(b.get("averaged_params") or b.get("best_params")),
            "test_score": _f(tr.get("test_score")),
            "test_ci": {
                "lower": _f(_ci(b).get("lower")),
                "upper": _f(_ci(b).get("upper")),
                "method": _ci(b).get("method", "percentile"),
            },
            "direction": (
                int(b["direction_sign"])
                if b.get("direction_sign") is not None
                else None
            ),
            "subset_scores": b.get("subset_scores") or {},
            "stability": {
                k: summ.get(k)
                for k in (
                    "q_worst",
                    "median",
                    "count",
                    "selection_probability",
                    "worst_quantile",
                    "n_bootstraps",
                    "n_trials_per_bootstrap",
                    "n_total_trials",
                    "higher_is_better",
                )
            },
        }

    cov_summary = None
    if covariate_impact:
        cov_summary = {
            k: covariate_impact.get(k)
            for k in (
                "r2_full",
                "r2_cgi_only",
                "r2_lift_from_covariates",
                "cgi_coef",
                "cgi_std_err",
                "n",
            )
        }
        cov_summary["per_covariate"] = covariate_impact.get("per_covariate") or []

    manifest = {
        "outcome": label,
        "objective_metric": objective_metric,
        "cgi_formula": formula_name,
        "studies": studies_manifest,
        "cgi_vs_standalone_aic_bic": aic_bic,
        "covariate_impact": cov_summary,
        "decline_terms": (cgi_bundle or {}).get("decline_terms"),
        "collinearity": collinearity_report,
    }
    _emit_json(f"results_summary{sfx}.json", manifest)

    # ── aic_bic.json + collinearity.json (standalone copies) ──────────
    if aic_bic is not None:
        _emit_json(f"aic_bic{sfx}.json", aic_bic)
    if collinearity_report:
        _emit_json(f"collinearity{sfx}.json", collinearity_report)

    log(
        "OK",
        f"[{label}] Wrote {len(written)} result file(s) to {output_dir}.",
    )
    return written


# Human labels for the standalone channels surfaced in ledger stages and
# logs. Keys match the engine's ``greenery_channel`` values.
_STANDALONE_CHANNEL_LABELS: dict[str, str] = {
    "veg": "Vegetation",
    "terrain": "Terrain",
    "ndvi": "NDVI",
}


def _fusion_stage_key(label: str, step: str, *, multi: bool) -> str:
    """Stage-ledger key for ``step`` under outcome ``label`` (label-scoped if multi)."""
    return f"{label}::{step}" if multi else step


def _build_fusion_ledger(
    labels: list[str],
    *,
    multi: bool,
    standalone_channels: list[str] | None = None,
    longitudinal: bool = False,
    mixedlm_postscore: bool = False,
) -> StageLedger:
    """Fresh ledger covering every (outcome, step) pair in run order.

    For each outcome the CGI pipeline (`_FUSION_STAGE_STEPS`) lands first, then
    two stages per enabled standalone metric — the stability search and the
    test scoring / reporting that follows it. Standalones reuse the already-
    built split + pre-aggregation cache. When ``longitudinal`` is true an extra
    ``prepare_longitudinal`` stage is inserted between ``load_metrics`` and
    ``preaggregate`` to cover per-wave file loading. The MixedLM
    post-score stage is only added when ``mixedlm_postscore`` is true
    (a longitudinal study whose scoring metric is actually a
    ``mixedlm_*`` one — year-aware cross-sectional studies sit on a
    spec too but score with OLS so they skip the post-score step).
    """
    standalones = list(standalone_channels or [])
    steps: list[tuple[str, str]] = []
    for label in labels:
        for step_key, step_label in _FUSION_STAGE_STEPS:
            key = _fusion_stage_key(label, step_key, multi=multi)
            disp = f"[{label}] {step_label}" if multi else step_label
            steps.append((key, disp))
            # Slot the longitudinal-prep stage in right after load_metrics so
            # the monitor reads top-to-bottom in actual execution order.
            if longitudinal and step_key == "load_metrics":
                lon_key_raw, lon_label = _FUSION_LONGITUDINAL_STAGE
                lon_key = _fusion_stage_key(label, lon_key_raw, multi=multi)
                lon_disp = f"[{label}] {lon_label}" if multi else lon_label
                steps.append((lon_key, lon_disp))
            # And the post-score stage right after ``apply`` so the
            # multi-metric CSV is written before any standalone studies
            # take over the engine state.
            if mixedlm_postscore and step_key == "apply":
                ps_key_raw, ps_label = _FUSION_MIXEDLM_POSTSCORE_STAGE
                ps_key = _fusion_stage_key(label, ps_key_raw, multi=multi)
                ps_disp = f"[{label}] {ps_label}" if multi else ps_label
                steps.append((ps_key, ps_disp))
        for ch in standalones:
            ch_lbl = _STANDALONE_CHANNEL_LABELS.get(ch, ch)
            search_key = _fusion_stage_key(label, f"standalone_{ch}", multi=multi)
            report_key = _fusion_stage_key(
                label, f"standalone_{ch}_report", multi=multi
            )
            search_step = f"Standalone {ch_lbl} stability selection"
            report_step = f"Standalone {ch_lbl} test scoring & reports"
            steps.append(
                (search_key, f"[{label}] {search_step}" if multi else search_step)
            )
            steps.append(
                (report_key, f"[{label}] {report_step}" if multi else report_step)
            )
    return StageLedger.from_steps(steps)


def _stability_summary(params: dict) -> dict:
    """Lift the stability-selection diagnostics out of a winning-params dict.

    ``bootstrap_stability_selection`` stashes its bookkeeping under ``__``-
    prefixed keys (so the "Final params" panel strips them). This surfaces the
    ones the results UI shows as a plain summary dict.
    """

    def g(key: str, default: Any = None) -> Any:
        return params.get(key, default)

    return {
        "q_worst": g("__cell_q_worst__"),
        "median": g("__cell_median__"),
        "count": g("__cell_count__"),
        "selection_probability": g("__cell_selection_probability__"),
        "worst_quantile": g("__worst_quantile__"),
        # Automated threshold calibration (Bodinier).
        "stability_score": g("__stability_score__"),
        "selection_threshold": g("__selection_threshold__"),
        "selection_size_k": g("__selection_size_k__"),
        "n_candidate_cells": g("__n_candidate_cells__"),
        "n_stably_selected": g("__n_stably_selected__"),
        "pfer": g("__pfer__"),
        "pfer_controlled": g("__pfer_controlled__"),
        "n_bootstraps": g("__n_bootstraps__"),
        "n_trials_per_bootstrap": g("__n_trials_per_bootstrap__"),
        "n_total_trials": g("__n_total_trials__"),
        "higher_is_better": g("__higher_is_better__"),
        "cell_stats": g("__cell_stats__", []),
        "winning_cell": g("__winning_cell__"),
        "winning_cell_oob_scores": g("__winning_cell_oob_scores__", []),
        "per_bootstrap_summary": g("__per_bootstrap_summary__", []),
        "trial_history": g("__trial_history__", []),
        # Stage-2 (radius sub-cell) diagnostics.
        "radius_cell_q_worst": g("__radius_cell_q_worst__"),
        "radius_cell_median": g("__radius_cell_median__"),
        "radius_cell_count": g("__radius_cell_count__"),
        "radius_bin_m": g("__radius_bin_m__"),
        "radius_cell_stats": g("__radius_cell_stats__", []),
    }


def _direction_sign(engine: Any, params: dict, metric: str) -> int:
    """``+1`` / ``-1`` sign of the greenery↔outcome relationship on the test set.

    Distance correlation is unsigned, so the report needs a separate direction
    indicator. Returns ``+1`` (neutral) on any failure.
    """
    from .. import objective_scoring as _scoring

    try:
        res = engine.evaluate_on_test(
            params=params, metric=metric, return_predictions=True
        )
        target = np.asarray(res.get("targets"), dtype=np.float64)
        pred = np.asarray(res.get("predictions"), dtype=np.float64)
        cov = res.get("covariates")
        return int(
            _scoring.relationship_sign(
                target,
                pred,
                cov,
                residualize_method=getattr(engine, "residualize_method", "linear"),
            )
        )
    except Exception:
        return 1


def _compare_cgi_vs_standalone(
    engine: Any,
    standalones_bundle: dict,
    cgi_params: dict,
    metric: str,
    *,
    longitudinal: bool,
    log,
) -> dict | None:
    """AIC/BIC verdict: is CGI justified over the best single standalone channel?

    The best standalone is the channel with the strongest whole-data (``all``)
    score (direction-aware). The full (3-channel) and reduced (best-channel)
    models are fit on the whole dataset's per-entity channel design built at the
    CGI winning aggregation params, so the verdict is on the same ``all`` slice
    as the paired objective comparison. Returns ``None`` when no standalone
    qualifies or the design / fit fails.
    """
    from .. import mixed_effects_scoring as _me
    from .. import objective_scoring as _scoring

    chans = ["veg", "terrain", "ndvi"]
    higher_is_better = (
        metric in _scoring.HIGHER_IS_BETTER or metric in _me.HIGHER_IS_BETTER
    )
    scored: list[tuple[str, float]] = []
    for ch in chans:
        b = standalones_bundle.get(ch)
        if not b:
            continue
        ss = (b.get("subset_scores") or {}).get("all") or {}
        ts = ss.get("score")
        if ts is None or not np.isfinite(float(ts)):
            continue
        scored.append((ch, float(ts)))
    if not scored:
        return None
    best_ch = max(scored, key=lambda kv: kv[1] if higher_is_better else -kv[1])[0]
    best_idx = chans.index(best_ch)

    try:
        design = engine.build_channel_design(cgi_params, subset="all")
    except Exception as exc:
        log("WARN", f"AIC/BIC channel design failed: {exc}")
        return None

    X = design["channels"]
    target = design["target"]
    cov = design["covariates"]
    try:
        if longitudinal and design.get("entity_id") is not None:
            return _me.compare_models_aic_bic_mixedlm(
                target,
                X,
                chans,
                best_idx,
                design["entity_id"],
                design["years_since_baseline"],
                covariates=cov,
            )
        return _scoring.compare_models_aic_bic(
            target, X, chans, best_idx, covariates=cov
        )
    except Exception as exc:
        log("WARN", f"AIC/BIC comparison failed: {exc}")
        return None


def _jsonsafe_results(obj, _depth: int = 0):
    """Recursively convert a fusion results payload to JSON-serializable types.

    Heavy or non-serializable values (DataFrames, ndarrays, per-trial pools)
    are dropped — they're reproducible from the on-disk artifacts — so the
    compact bundle written beside the job can rehydrate the results view after
    a Streamlit restart without pinning engines in memory.
    """
    if _depth > 12:
        return None
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            if k in (
                "composite_df",
                "robust_trials",
                "per_trial_test",
                "all_completed_trials",
            ):
                continue
            out[str(k)] = _jsonsafe_results(v, _depth + 1)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [_jsonsafe_results(v, _depth + 1) for v in obj]
    try:
        import pandas as _pd

        if isinstance(obj, (_pd.DataFrame, _pd.Series)):
            return None
    except Exception:
        pass
    try:
        s = str(obj)
        return s if len(s) <= 2000 else None
    except Exception:
        return None


def run_fusion(
    ctx: JobContext,
    *,
    target_path: str,
    target_features_geojson,
    target_band,
    target_layer,
    target_cleanup_dir: str | None,
    target_cleanup_file: str | None,
    buffer_meters: float,
    gvi_buffer_min_m: float,
    gvi_buffer_max_m: float,
    gvi_buffer_step_m: float,
    ndvi_buffer_min_m: float,
    ndvi_buffer_max_m: float,
    ndvi_buffer_step_m: float,
    ndvi_resolution_m: float | None = None,
    gvi_grid_spacing_m: float | None = None,
    n_bins: int,
    veg_path: str | None = None,
    terrain_path: str | None = None,
    ndvi_path: str | None = None,
    cache_metrics: bool = False,
    test_size: float,
    objective_metric: str,
    gvi_api_key: str | None = None,
    ndvi_start_date: str | None = None,
    ndvi_end_date: str | None = None,
    ndvi_project_id: str | None = None,
    multi_objective_requested: bool = False,
    output_dir: str,
    target_display_name: str = "target",
    resume_existing_study: bool = True,
    cgi_formula: str = "weighted_average",
    covariate_columns: list[str] | None = None,
    covariate_types: dict[str, str] | None = None,
    standalone_channels: list[str] | None = None,
    longitudinal_spec_payload: dict | None = None,
    cgi_grid_spacing_m: float | None = None,
    whole_grid_scaling: bool = False,
    area_balanced_split: bool = False,
    normalize_channels: bool = False,
    spatial_adjust_method: str = "none",
    spatial_adjust_max_df: int = 10,
    spatial_adjust_eps_m: float | None = None,
    residualize_method: str = "linear",
    n_bootstraps: int = 20,
    n_trials_per_bootstrap: int = 50,
    weight_bin_pct: int = 10,
    min_cell_count: int = 3,
    worst_quantile: float = 0.10,
    max_pfer: float = 1.0,
    spatial_split: bool = False,
    spatial_block_size_m: float | None = None,
    n_spatial_blocks: int | None = None,
    check_collinearity: bool = False,
    vif_threshold: float = 10.0,
    report_ci_bootstrap: int | None = None,
    report_effects_bootstrap: int = 2000,
    report_effects_permutations: int = 1000,
    report_paired_bootstrap: int = 2000,
) -> dict:
    """Run a fusion job: stability-selection tuning + held-out test scoring.

    For CGI (and each enabled standalone channel) the engine draws
    ``n_bootstraps`` resamples of the train+val pool, runs an
    ``n_trials_per_bootstrap``-trial RandomSampler search per resample, and
    selects the weight cell with the best worst-quantile out-of-bag score.
    ``max_pfer`` caps the calibrated selection size so the reported PFER bound
    stays under it (a non-positive value disables the cap). The winning params
    are then scored once on the held-out test split with a percentile bootstrap
    CI. When standalones are enabled, both a whole-data (``all``) paired
    objective comparison and an ``all``-data AIC/BIC comparison report whether
    CGI is justified over the best single channel.

    The ``report_*`` knobs set the reporting replicate budgets: the test-set
    CI bootstrap (``report_ci_bootstrap``; ``None`` picks 2,000 for the O(n²)
    ``partial_distance_corr`` objective and 10,000 otherwise — percentile CIs
    are stable well below the higher figure), the effects bootstrap /
    permutation counts, and the paired CGI-vs-standalone bootstrap."""
    try:
        # Imported here (not at module load) so the runner module stays light
        # and the fusion engine is only pulled into the process that runs the
        # job — the spawn child, where fusion now executes.
        from geofuse.fusion import MetricFusionEngine

        targets = list(target_features_geojson) if target_features_geojson else [None]
        n_t = max(len(targets), 1)
        multi_outcome = len([t for t in targets if t is not None]) > 1

        if multi_objective_requested and multi_outcome:
            _log_fusion(
                "WARN",
                "Multi-objective optimization run requested — joint study not "
                "implemented yet; running separate single-objective studies per outcome.",
            )

        _log_fusion("INFO", f"Starting fusion job {ctx.job_id} ({n_t} target run(s))")

        # Validate the standalone request up front so a typo doesn't slip
        # through to the ledger and engine. ``None`` and empty list both mean
        # "CGI only" (the legacy behaviour).
        standalones: list[str] = list(standalone_channels or [])
        for _ch in standalones:
            if _ch not in ("veg", "terrain", "ndvi"):
                raise ValueError(
                    f"standalone_channels entries must be one of "
                    f"'veg','terrain','ndvi'; got {_ch!r}."
                )
        if standalones:
            _log_fusion(
                "INFO",
                f"Standalone single-metric studies enabled: {', '.join(standalones)}",
            )

        # Test-CI replicate budget: explicit value wins; otherwise 2,000 for
        # the O(n²) partial-distance-correlation objective, 10,000 otherwise.
        report_ci_n = (
            int(report_ci_bootstrap)
            if report_ci_bootstrap is not None
            else (2_000 if objective_metric == "partial_distance_corr" else 10_000)
        )

        # Mixed-effects / longitudinal mode. Reconstructed once up front so
        # spec errors surface before any heavy compute. Cross-sectional jobs
        # pass ``longitudinal_spec_payload=None`` (the default) and follow the
        # legacy pipeline unchanged.
        longitudinal_spec = _resolve_longitudinal_spec(longitudinal_spec_payload)
        if longitudinal_spec is not None:
            _log_fusion(
                "INFO",
                f"Mixed-effects fusion enabled: intake_mode={longitudinal_spec.intake_mode!r}, "
                f"waves={list(longitudinal_spec.wave_labels)}, "
                f"scoring_metric={longitudinal_spec.scoring_metric!r}.",
            )

        by_target: dict = {}
        output_paths: list[str] = []

        # Pre-compute every outcome label so the staged-resume ledger can list
        # all (outcome, step) stages up front; the monitor then shows pending
        # stages before the runner reaches them.
        all_labels = [
            (t if t is not None else f"raster_band_{target_band}") for t in targets
        ]
        ordered_labels: list[str] = list(all_labels)
        mixedlm_postscore_enabled = (
            longitudinal_spec is not None
            and longitudinal_spec.scoring_metric in _LON_MIXEDLM_METRICS
        )
        ledger = _build_fusion_ledger(
            all_labels,
            multi=multi_outcome,
            standalone_channels=standalones,
            longitudinal=longitudinal_spec is not None,
            mixedlm_postscore=mixedlm_postscore_enabled,
        )
        ctx.update_stage_ledger(ledger.to_dict())

        # Settings snapshot written verbatim into each job's ``study_results``
        # so the run is reproducible from disk (the job store records the same
        # config, but the JSON travels with the artifacts). Secrets and
        # infra-only handles (API key, project id, file paths, engine class)
        # are intentionally excluded.
        run_config_record: dict[str, Any] = {
            "objective_metric": objective_metric,
            "residualize_method": str(residualize_method),
            "cgi_formula": cgi_formula,
            "covariate_columns": list(covariate_columns or []),
            "standalone_channels": list(standalones),
            "test_size": float(test_size),
            "n_bins": n_bins,
            "n_bootstraps": int(n_bootstraps),
            "n_trials_per_bootstrap": int(n_trials_per_bootstrap),
            "weight_bin_pct": int(weight_bin_pct),
            "min_cell_count": int(min_cell_count),
            "worst_quantile": float(worst_quantile),
            "max_pfer": float(max_pfer),
            "buffer_meters": buffer_meters,
            "gvi_buffer_min_m": gvi_buffer_min_m,
            "gvi_buffer_max_m": gvi_buffer_max_m,
            "gvi_buffer_step_m": gvi_buffer_step_m,
            "ndvi_buffer_min_m": ndvi_buffer_min_m,
            "ndvi_buffer_max_m": ndvi_buffer_max_m,
            "ndvi_buffer_step_m": ndvi_buffer_step_m,
            "ndvi_resolution_m": ndvi_resolution_m,
            "gvi_grid_spacing_m": gvi_grid_spacing_m,
            "cgi_grid_spacing_m": cgi_grid_spacing_m,
            "whole_grid_scaling": bool(whole_grid_scaling),
            "area_balanced_split": bool(area_balanced_split),
            "normalize_channels": bool(normalize_channels),
            "spatial_split": bool(spatial_split),
            "spatial_block_size_m": spatial_block_size_m,
            "n_spatial_blocks": n_spatial_blocks,
            "check_collinearity": bool(check_collinearity),
            "vif_threshold": float(vif_threshold),
            "report_ci_bootstrap": int(report_ci_n),
            "report_effects_bootstrap": int(report_effects_bootstrap),
            "report_effects_permutations": int(report_effects_permutations),
            "report_paired_bootstrap": int(report_paired_bootstrap),
            "cache_metrics": bool(cache_metrics),
            "resume_existing_study": bool(resume_existing_study),
            "ndvi_start_date": ndvi_start_date,
            "ndvi_end_date": ndvi_end_date,
            "multi_objective_requested": bool(multi_objective_requested),
            "longitudinal_spec": longitudinal_spec_payload,
            "target_display_name": target_display_name,
            "selection_method": "bootstrap_stability_selection",
        }

        def stage(key: str, status: str, message: str = "") -> None:
            """Record a stage transition in the ledger and persist it."""
            ledger.set_status(
                key,
                status,
                progress=1.0 if status == DONE else 0.0,
                message=message or None,
            )
            ctx.update_stage_ledger(ledger.to_dict())

        # Per-stage monotonic-time gate for ``stage_progress``. ``update_stage_
        # ledger`` writes SQLite synchronously (unlike the memory-only
        # ``update_progress``), so per-trial fractional updates must be
        # throttled or they hammer the store.
        _stage_prog_t: dict[str, float] = {}

        def stage_progress(key: str, frac: float, message: str = "") -> None:
            """Advance a *running* stage's fractional progress, throttled to ~1/s.

            The final tick (``frac >= 1``) always writes so the ledger row lands
            on its true endpoint; intermediate ticks are gated at 1 s per stage.
            """
            now = time.monotonic()
            if frac < 1.0 and (now - _stage_prog_t.get(key, 0.0)) < 1.0:
                return
            _stage_prog_t[key] = now
            ledger.mark_progress(key, frac, message)
            ctx.update_stage_ledger(ledger.to_dict())

        cache_dir = os.path.join(output_dir, "fusion_cache")

        # Per-run artifact folder so reruns don't overwrite each other.
        from datetime import datetime as _dt

        _job_short = ctx.job_id.replace("-", "")[:8] if ctx.job_id else "anon"
        job_stamp = _dt.now().strftime("%Y%m%dT%H%M%S")
        job_artifacts_root = os.path.join(
            output_dir, "fusion", f"{job_stamp}__{_job_short}"
        )
        os.makedirs(job_artifacts_root, exist_ok=True)

        # Cross-sectional metric sources (cropped veg / terrain / NDVI frames)
        # depend only on the shared target extent + buffer, so they're loaded
        # once and adopted by every later outcome's engine — no N-fold re-read.
        # ``None`` until the first outcome loads them; longitudinal mode routes
        # per-wave files instead and leaves this unused.
        shared_metric_data: tuple | None = None

        for ti, target_feature in enumerate(targets):
            if ctx.is_cancelled():
                return {"output_paths": output_paths}

            label = all_labels[ti]
            prefix = f"[{label}] " if n_t > 1 else ""

            def skey(step: str, _label: str = label) -> str:
                return _fusion_stage_key(_label, step, multi=multi_outcome)

            def prog(local: float) -> float:
                return (ti + local) / n_t

            # Stage keys belonging to this outcome, for the ledger-derived main
            # progress bar. For multi-outcome runs keys carry a "<label>::"
            # prefix; single-outcome runs own the whole ledger.
            _label_prefix = f"{label}::" if multi_outcome else None
            label_stage_keys = [
                s.key
                for s in ledger.stages
                if _label_prefix is None or s.key.startswith(_label_prefix)
            ]
            _label_total_weight = sum(
                _fusion_stage_weight(k) for k in label_stage_keys
            )

            def prog_ledger() -> float:
                """Main-bar value derived from this outcome's ledger state.

                ``(finished + running·progress)`` weighted by
                ``_fusion_stage_weight`` over the outcome's total weight, mapped
                into the outcome's slice via ``prog``. Monotonic by construction
                — stages only advance and a running stage's fraction only grows.
                """
                if _label_total_weight <= 0:
                    return prog(0.0)
                done = 0.0
                for k in label_stage_keys:
                    st_ = ledger.get(k)
                    if st_ is None:
                        continue
                    w = _fusion_stage_weight(k)
                    if st_.status in (DONE, SKIPPED):
                        done += w
                    elif st_.status == RUNNING:
                        done += w * st_.progress
                return prog(done / _label_total_weight)

            ctx.progress(
                value=prog_ledger(),
                status_text=f"{prefix}Initializing fusion engine...",
            )

            # If the current outcome was also picked as a covariate (only
            # possible when several outcomes share a covariate list), drop it
            # for *this* outcome's run — a column can't predict itself.
            user_covs = list(covariate_columns or [])
            outcome_covs = [c for c in user_covs if c != target_feature]
            if outcome_covs != user_covs:
                _log_fusion(
                    "INFO",
                    f"[{label}] Dropping covariate(s) that match this outcome: "
                    f"{sorted(set(user_covs) - set(outcome_covs))}",
                )

            engine = MetricFusionEngine(
                target_file=target_path,
                target_feature=target_feature,
                target_band=target_band,
                target_layer=target_layer,
                buffer_meters=buffer_meters,
                gvi_buffer_min_m=gvi_buffer_min_m,
                gvi_buffer_max_m=gvi_buffer_max_m,
                gvi_buffer_step_m=gvi_buffer_step_m,
                ndvi_buffer_min_m=ndvi_buffer_min_m,
                ndvi_buffer_max_m=ndvi_buffer_max_m,
                ndvi_buffer_step_m=ndvi_buffer_step_m,
                n_bins=n_bins,
                cache_dir=cache_dir,
                cgi_formula=cgi_formula,
                covariate_columns=outcome_covs,
                covariate_types=dict(covariate_types or {}),
                longitudinal_spec=longitudinal_spec,
                cgi_grid_spacing_m=cgi_grid_spacing_m,
                whole_grid_scaling=whole_grid_scaling,
                area_balanced_split=area_balanced_split,
                normalize_channels=normalize_channels,
                spatial_adjust_method=spatial_adjust_method,
                spatial_adjust_max_df=spatial_adjust_max_df,
                spatial_adjust_eps_m=spatial_adjust_eps_m,
                residualize_method=residualize_method,
            )

            ctx.progress(
                value=prog_ledger(), status_text=f"{prefix}Loading target data..."
            )
            stage(skey("load_target"), RUNNING)
            engine.load_target()
            stage(skey("load_target"), DONE)

            if not veg_path:
                ctx.progress(
                    value=prog_ledger(),
                    status_text=f"{prefix}Downloading GVI Vegetation data...",
                )
            elif not terrain_path:
                ctx.progress(
                    value=prog_ledger(),
                    status_text=f"{prefix}Downloading GVI Terrain data...",
                )
            elif not ndvi_path:
                ctx.progress(
                    value=prog_ledger(),
                    status_text=f"{prefix}Downloading NDVI satellite data...",
                )
            else:
                ctx.progress(
                    value=prog_ledger(),
                    status_text=f"{prefix}Loading provided metric files...",
                )

            if ctx.is_cancelled():
                return {"output_paths": output_paths}

            last_update_time = {"veg": 0.0, "terrain": 0.0}

            def gvi_progress_callback(component, curr, total):
                current_time = time.time()
                if (
                    current_time - last_update_time.get(component, 0) < 0.5
                    and curr != total
                ):
                    return
                last_update_time[component] = current_time
                gvi_progress = {
                    "component": component,
                    "current": curr,
                    "total": total,
                    "percent": 100 * curr / total if total > 0 else 0,
                }
                ctx.set_extra(gvi_progress=gvi_progress)
                ctx.heartbeat()

            def cancel_check():
                return ctx.is_cancelled()

            stage(skey("load_metrics"), RUNNING)
            if longitudinal_spec is None:
                if shared_metric_data is None:
                    engine.load_metrics(
                        veg_file=veg_path,
                        terrain_file=terrain_path,
                        ndvi_file=ndvi_path,
                        cache_metrics=cache_metrics,
                        gvi_api_key=gvi_api_key,
                        ndvi_start_date=ndvi_start_date,
                        ndvi_end_date=ndvi_end_date,
                        ndvi_project_id=ndvi_project_id,
                        progress_callback=gvi_progress_callback,
                        cancel_callback=cancel_check,
                        ndvi_resolution_m=ndvi_resolution_m,
                        gvi_grid_spacing_m=gvi_grid_spacing_m,
                    )
                    shared_metric_data = (
                        engine.veg_data,
                        engine.terrain_data,
                        engine.ndvi_data,
                    )
                else:
                    # Every outcome shares the same target extent + metric
                    # paths, so reuse the first outcome's cropped frames instead
                    # of re-reading them from disk.
                    engine.adopt_metric_data(
                        *shared_metric_data,
                        ndvi_resolution_m=ndvi_resolution_m,
                        gvi_grid_spacing_m=gvi_grid_spacing_m,
                    )
                    _log_fusion(
                        "INFO",
                        f"[{label}] Reusing metric data loaded for the first "
                        "outcome (skipped re-reading veg / terrain / NDVI).",
                    )
            else:
                # Longitudinal mode bypasses ``load_metrics`` (whose path
                # builds one cross-sectional source per channel); the per-
                # wave files are loaded in the prepare_longitudinal stage
                # below and pushed via ``set_longitudinal_metric_data``.
                _log_fusion(
                    "INFO",
                    f"[{label}] Longitudinal mode: per-wave metric files will "
                    "load in the prepare_longitudinal stage.",
                )
            stage(skey("load_metrics"), DONE)

            # Per-wave file resolution + engine injection. Wide-mode also
            # loads N per-wave target frames here. Long-mode skips the wide
            # frames branch and uses the single ``load_target`` result.
            if longitudinal_spec is not None:
                stage(skey("prepare_longitudinal"), RUNNING)
                ctx.progress(
                    value=prog_ledger(),
                    status_text=f"{prefix}Loading per-wave files...",
                )
                # Year-aware cross-sectional
                if longitudinal_spec.derive_wave_from_date:
                    from geofuse.longitudinal import parse_date_column as _parse_date

                    tgdf = engine.target_gdf
                    if tgdf is None:
                        raise RuntimeError(
                            "derive_wave_from_date requires load_target() to have run."
                        )
                    date_col = longitudinal_spec.date_col
                    if date_col not in tgdf.columns:
                        raise ValueError(
                            f"derive_wave_from_date: date column {date_col!r} "
                            f"is not present in the target frame."
                        )
                    parsed_dates = _parse_date(tgdf[date_col])
                    wave_col_name = longitudinal_spec.wave_col or "_gf_year"
                    entity_col_name = longitudinal_spec.entity_id_col
                    tgdf[wave_col_name] = parsed_dates.dt.year.astype("Int64").astype(
                        str
                    )
                    tgdf[entity_col_name] = np.arange(len(tgdf)).astype(str)
                    _log_fusion(
                        "INFO",
                        f"[{label}] Synthesised wave column {wave_col_name!r} "
                        f"and entity column {entity_col_name!r} for year-aware "
                        "cross-sectional run.",
                    )
                if longitudinal_spec.intake_mode == "wide":
                    wide_frames: list[tuple[str, gpd.GeoDataFrame]] = []
                    for wave_label in longitudinal_spec.wave_labels:
                        wpath = longitudinal_spec.target_files_per_wave[wave_label]
                        wide_frames.append((wave_label, gpd.read_file(wpath)))
                    engine.set_longitudinal_wave_frames(wide_frames)
                    _log_fusion(
                        "INFO",
                        f"[{label}] Loaded {len(wide_frames)} per-wave target "
                        "files (wide intake).",
                    )
                for ch in GREENERY_CHANNELS:
                    per_wave: dict[str, Any] = {}
                    for wave_label, fp in longitudinal_spec.greenery_files[ch].items():
                        per_wave[wave_label] = _load_longitudinal_metric_file(fp, ch)
                    engine.set_longitudinal_metric_data(ch, per_wave)
                    n_unique = len({id(v) for v in per_wave.values()})
                    _log_fusion(
                        "INFO",
                        f"[{label}] Loaded {ch}: {len(per_wave)} wave(s), "
                        f"{n_unique} unique source(s).",
                    )
                stage(skey("prepare_longitudinal"), DONE)
                if ctx.is_cancelled():
                    return {"output_paths": output_paths}

            # Sample materialization + the on-disk pre-aggregation cache form one
            # "spatial pre-processing" stage; prepare_fusion_data feeds the cache.
            stage(skey("preaggregate"), RUNNING)
            ctx.progress(
                value=prog_ledger(),
                status_text=(
                    f"{prefix}Preparing fusion samples "
                    "(can take a while on large polygon targets)..."
                ),
            )
            fusion_df = engine.prepare_fusion_data()
            if ctx.is_cancelled():
                return {"output_paths": output_paths}

            # Mandatory spatial pre-processing (first compute step): build or
            # reuse the on-disk per-(entity, radius) stat cache so every Optuna
            # trial is a fast column read instead of recomputing buffer
            # aggregations. Resumable across cancels/crashes.
            _last_pct = {"v": -1}

            def preaggr_progress(current: int, total: int) -> None:
                pct = (current * 100) // max(1, total)
                if pct == _last_pct["v"]:
                    return
                _last_pct["v"] = pct
                ctx.set_extra(
                    preaggr_progress={
                        "current": current,
                        "total": total,
                        "percent": pct,
                    }
                )
                stage_progress(skey("preaggregate"), pct / 100.0)
                ctx.progress(
                    value=prog_ledger(),
                    status_text=(
                        f"{prefix}Spatial pre-processing: "
                        f"{current:,}/{total:,} grid cells ({pct}%)"
                    ),
                )
                ctx.heartbeat()

            completed = engine.precompute_aggregations(
                progress_callback=preaggr_progress,
                cancel_callback=cancel_check,
            )
            # Clear the dedicated preaggr_progress sub-bar so it doesn't linger
            # past this stage in the monitor.
            ctx.set_extra(preaggr_progress=None)
            if not completed or ctx.is_cancelled():
                return {"output_paths": output_paths}
            stage(skey("preaggregate"), DONE)

            # Optional iterative-VIF collinearity check on the CGI grid
            # pixel values. Disabled channels get pinned to weight 0 in
            # every subsequent trial (the engine threads
            # ``self._disabled_channels`` into ``formula.suggest_params``).
            if check_collinearity:
                try:
                    # Pass ``fusion_df`` explicitly — this stage runs
                    # before ``split_data``, so ``engine.train_val_data``
                    # and ``engine.test_data`` are still ``None``.
                    collinearity_report = engine.check_channel_collinearity(
                        data=fusion_df,
                        vif_threshold=float(vif_threshold),
                    )
                    if engine._disabled_channels:
                        _log_fusion(
                            "WARN",
                            f"[{label}] Collinearity check dropped: "
                            f"{sorted(engine._disabled_channels)}. "
                            f"VIFs (initial): "
                            f"{dict(zip(collinearity_report['channels_in'], collinearity_report['initial_vifs']))}.",
                        )
                    else:
                        _log_fusion(
                            "OK",
                            f"[{label}] Collinearity check passed (max VIF ≤ "
                            f"{vif_threshold}).",
                        )
                except Exception as exc:
                    _log_fusion(
                        "WARN",
                        f"[{label}] Collinearity check failed: {exc}. "
                        "Continuing with all three channels enabled.",
                    )

            # Config fingerprint keys the per-job caches; ``suffix`` forces a
            # fresh artifact namespace when the user opts out of resume.
            suffix = (
                ""
                if resume_existing_study
                else datetime.now().strftime("%Y%m%dT%H%M%S")
            )
            config_fp = _fusion_config_fingerprint(
                buffer_meters=float(buffer_meters),
                gvi_buffer_min_m=float(gvi_buffer_min_m),
                gvi_buffer_max_m=float(gvi_buffer_max_m),
                gvi_buffer_step_m=float(gvi_buffer_step_m),
                ndvi_buffer_min_m=float(ndvi_buffer_min_m),
                ndvi_buffer_max_m=float(ndvi_buffer_max_m),
                ndvi_buffer_step_m=float(ndvi_buffer_step_m),
                cgi_formula=cgi_formula,
                covariate_columns=outcome_covs,
                covariate_types=dict(covariate_types or {}),
                whole_grid_scaling=bool(whole_grid_scaling),
                cgi_grid_spacing_m=cgi_grid_spacing_m,
                area_balanced_split=bool(area_balanced_split),
                test_size=float(test_size),
                spatial_split=bool(spatial_split),
                spatial_block_size_m=spatial_block_size_m,
                n_spatial_blocks=n_spatial_blocks,
                normalize_channels=bool(normalize_channels),
                spatial_adjust_method=spatial_adjust_method,
                spatial_adjust_max_df=spatial_adjust_max_df,
                spatial_adjust_eps_m=spatial_adjust_eps_m,
            )

            def _standalone_study_name(ch: str) -> str:
                ch_suffix = ch if resume_existing_study else f"{ch}_{suffix}"
                return _build_fusion_study_name(
                    target_display_name=target_display_name,
                    label=label,
                    objective_metric=objective_metric,
                    suffix=ch_suffix,
                    config_fingerprint=config_fp,
                )

            # One held-out test split + the train+val pool that stability
            # selection resamples.
            stage(skey("split"), RUNNING)
            ctx.progress(
                value=prog_ledger(),
                status_text=f"{prefix}Splitting data into train / val / test...",
            )
            engine.split_data(
                fusion_df=fusion_df,
                test_size=test_size,
                random_state=42,
                spatial_split=bool(spatial_split),
                spatial_block_size_m=spatial_block_size_m,
                n_spatial_blocks=n_spatial_blocks,
            )
            stage(skey("split"), DONE)

            # Partial distance correlation is O(n²) in entities even with the
            # cached-side scorer (each trial still builds the composite's
            # distance matrix), and above the cached-path budget it falls back
            # to the stock estimator entirely. Plain distance_corr uses the
            # fast O(n log n) estimator and is unaffected. Warn rather than
            # cap so the metric stays exact.
            if objective_metric == "partial_distance_corr" and outcome_covs:
                try:
                    n_entities = (
                        int(fusion_df["polygon_id"].nunique())
                        if "polygon_id" in fusion_df.columns
                        else int(len(fusion_df))
                    )
                except Exception:
                    n_entities = 0
                if n_entities > 5000:
                    _log_fusion(
                        "WARN",
                        f"[{label}] partial_distance_corr scores are O(n²) over "
                        f"{n_entities:,} entities with covariates — trials and "
                        "reporting CIs will take noticeably longer"
                        + (
                            " (and the cached fast path is disabled at this "
                            "entity count)"
                            if n_entities > _pdcor_mod.MAX_CACHE_N
                            else ""
                        )
                        + ". 'distance_corr' (fast) or 'spearman' are cheaper "
                        "covariate-aware objectives if runtime matters.",
                    )

            # ``n_trials_per_bootstrap`` is the CGI (target) per-bootstrap budget.
            # A standalone single channel explores a far smaller stability-
            # selection space — one weight axis (``slots`` 10 %-bins) vs the CGI's
            # main-weight simplex (``weight_cell_count`` cells) — so it scales DOWN
            # by that cell ratio to match the CGI's per-cell trial density instead
            # of over-sampling its tiny search.
            from .. import cgi_formulas as _cgi_formulas

            _cgi_cells = max(
                1, _cgi_formulas.weight_cell_count(cgi_formula, int(weight_bin_pct))
            )
            _standalone_cells = max(1, 100 // int(weight_bin_pct))
            cgi_trials_per_bootstrap = int(n_trials_per_bootstrap)
            standalone_trials_per_bootstrap = max(
                1,
                round(int(n_trials_per_bootstrap) * _standalone_cells / _cgi_cells),
            )
            if standalone_trials_per_bootstrap != cgi_trials_per_bootstrap:
                _log_fusion(
                    "INFO",
                    f"[{label}] CGI uses {int(n_bootstraps)}×{cgi_trials_per_bootstrap} "
                    f"trials over {_cgi_cells} weight cells; each standalone scales "
                    f"down to {int(n_bootstraps)}×{standalone_trials_per_bootstrap} "
                    f"({_standalone_cells} cells / {_cgi_cells} = "
                    f"×{_standalone_cells / _cgi_cells:.2f}).",
                )

            def _study_progress_cb(study_label: str, stage_key: str | None = None):
                """Per-trial callback → live caption, trial bar, and stage row.

                Throttled to ~0.4 s (always fires on the final trial) so the
                job card shows "<study>: k / N trials" without flooding the
                store. When ``stage_key`` is given it also advances that
                running search stage's ledger fraction (via ``stage_progress``,
                which self-throttles the synchronous SQLite write) and steps the
                ledger-derived main bar.
                """
                state = {"t": 0.0}

                def _cb(done: int, total: int) -> None:
                    now = time.monotonic()
                    if done < total and (now - state["t"]) < 0.4:
                        return
                    state["t"] = now
                    pct = (100.0 * done / total) if total else 0.0
                    if stage_key is not None:
                        stage_progress(
                            stage_key,
                            (done / total) if total else 0.0,
                            f"{done:,}/{total:,} trials",
                        )
                    ctx.progress(
                        value=prog_ledger() if stage_key is not None else None,
                        status_text=f"{prefix}{study_label}: {done:,}/{total:,} trials",
                        fusion_study_progress={
                            "study": study_label,
                            "current": int(done),
                            "total": int(total),
                            "percent": float(pct),
                        },
                    )

                return _cb

            # A non-positive cap means "no PFER cap" — pass None so the
            # calibration is free to grow the selection size K.
            max_pfer_arg = None if float(max_pfer) <= 0 else float(max_pfer)

            # ── Stability selection: the dominant compute ──
            # Headline params: the stability-selection winning weight cell on
            # the full train+val pool (params averaged within the cell). The
            # "CGI: k/N trials" sub-bar and the running-stage fraction both live
            # here, so the running stage and the trial counter agree.
            stage(skey("optimize"), RUNNING)
            ctx.progress(
                value=prog_ledger(),
                status_text=(
                    f"{prefix}Stability selection "
                    f"({int(n_bootstraps)}×{cgi_trials_per_bootstrap})..."
                ),
            )
            headline_params = engine.bootstrap_stability_selection(
                metric=objective_metric,
                n_bootstraps=int(n_bootstraps),
                n_trials_per_bootstrap=cgi_trials_per_bootstrap,
                weight_bin_pct=int(weight_bin_pct),
                min_cell_count=int(min_cell_count),
                worst_quantile=float(worst_quantile),
                max_pfer=max_pfer_arg,
                spatial_resample=bool(spatial_split),
                seed=42,
                cancel_callback=cancel_check,
                progress_callback=_study_progress_cb("CGI", skey("optimize")),
            )
            engine.best_params = dict(headline_params)
            cgi_stability_summary = _stability_summary(headline_params)
            # Kept for the results bundle's schema. Stability selection has no
            # master Optuna study, so there is no explicit best/robust trial
            # pool — the winning cell is the aggregate over bootstrap resamples.
            best_params: dict = {}
            robust_trials: list = []
            if ctx.is_cancelled():
                return {"output_paths": output_paths}
            stage(skey("optimize"), DONE)

            # ── Score the held-out test set: fast, the honest "evaluate" ──
            stage(skey("evaluate"), RUNNING)
            ctx.progress(
                value=prog_ledger(),
                status_text=f"{prefix}Scoring held-out test set...",
            )
            test_results = engine.evaluate_on_test(
                params=headline_params, metric=objective_metric
            )
            stage(skey("evaluate"), DONE)

            # ── Replicate statistics: the long tail ──
            # Test-set percentile bootstrap CI, relationship direction, and the
            # held-out effect sizes / permutation p-value — thousands of
            # replicates, named for what it is instead of hiding under
            # "evaluate".
            stage(
                skey("report_stats"),
                RUNNING,
                f"Test CI: {report_ci_n:,} bootstrap replicates...",
            )
            ctx.progress(
                value=prog_ledger(),
                status_text=(
                    f"{prefix}Bootstrap CIs, effects & permutation tests "
                    f"({report_ci_n:,} replicates)..."
                ),
            )
            # Independent held-out effect size + percentile bootstrap CI.
            try:
                cgi_test_ci = engine.bootstrap_test_score_ci(
                    headline_params,
                    objective_metric,
                    n_bootstrap=report_ci_n,
                    ci_level=0.95,
                    method="percentile",
                    seed=42,
                )
                test_results["test_ci"] = cgi_test_ci
            except Exception as exc:
                _log_fusion("WARN", f"[{label}] Bootstrap CI failed: {exc}")

            # Direction of the greenery↔outcome relationship (distance
            # correlation is unsigned, so the sign is reported separately).
            cgi_direction = _direction_sign(engine, headline_params, objective_metric)

            # Held-out test effect (the headline) plus a descriptive whole-data
            # figure, per-subset bootstrap CIs, and the held-out permutation
            # p-value — all in the objective metric's own units.
            cgi_effects: dict | None = None
            try:
                cgi_effects = engine.evaluate_effects(
                    headline_params,
                    objective_metric,
                    n_bootstrap=int(report_effects_bootstrap),
                    n_perm=int(report_effects_permutations),
                    seed=42,
                )
                if cgi_effects and cgi_effects.get("test"):
                    _t = cgi_effects["test"]
                    _a = cgi_effects.get("all") or {}
                    _log_fusion(
                        "OK",
                        f"[{label}] Held-out test {objective_metric}="
                        f"{_t.get('score')} "
                        f"[{_t.get('lower')}, {_t.get('upper')}], "
                        f"p={_t.get('p_value')}; whole-data (in-sample) "
                        f"{_a.get('score')}.",
                    )
            except Exception as exc:
                _log_fusion("WARN", f"[{label}] Whole-data effects failed: {exc}")

            # No master Optuna study in stability mode, so there is no per-trial
            # test sidecar to build.
            cgi_per_trial_test: dict[int, dict[str, float]] = {}
            stage(skey("report_stats"), DONE)

            stage(skey("apply"), RUNNING)
            ctx.progress(
                value=prog_ledger(), status_text=f"{prefix}Applying fusion weights..."
            )
            composite_df = engine.apply_fusion()
            stage(skey("apply"), DONE)

            # ── Post-hoc multi-metric reporting (MixedLM scoring only) ──
            # Re-score every robust + top-20% trial + the averaged-composite
            # parameters on the test set with all four mixedlm_* metrics so
            # the user can compare metric agreement across the trial pool.
            # The CSV basename includes the outcome label so multi-outcome
            # runs don't overwrite each other. Skipped when the
            # longitudinal spec is acting only as a per-year file-routing
            # key with an OLS scorer — there are no MixedLM metrics to
            # report.
            if mixedlm_postscore_enabled:
                stage(skey("mixedlm_postscore"), RUNNING)
                ctx.progress(
                    value=prog_ledger(),
                    status_text=(
                        f"{prefix}Scoring all MixedLM metrics on robust trials..."
                    ),
                )
                postscore_dir = os.path.join(job_artifacts_root, "study_results")
                csv_basename = (
                    f"mixedlm_metrics__{label}.csv"
                    if multi_outcome
                    else "mixedlm_metrics.csv"
                )
                try:
                    _compute_mixedlm_post_metrics(
                        engine,
                        postscore_dir,
                        winning_params=headline_params,
                        csv_basename=csv_basename,
                        log=_log_fusion,
                    )
                except Exception as exc:
                    _log_fusion(
                        "WARN",
                        f"[{label}] Post-hoc MixedLM scoring failed: {exc}",
                    )
                stage(skey("mixedlm_postscore"), DONE)

            # ── Standalone single-metric stability searches ─────────────────
            # One bootstrap stability search per enabled channel, reusing the
            # same engine, the already-built per-(entity, radius) cache, and the
            # train+val/test split.
            #
            # Each search overwrites ``engine.best_params`` /
            # ``engine._active_greenery_channel`` with the standalone's, so we
            # snapshot the CGI state up front and restore it after the loop.
            cgi_best_value = float(
                headline_params.get("__cell_q_worst__", float("nan"))
            )
            cgi_best_params = engine.best_params

            standalones_bundle: dict[str, dict] = {}
            for ch in standalones:
                if ctx.is_cancelled():
                    return {"output_paths": output_paths}
                ch_disp = _STANDALONE_CHANNEL_LABELS.get(ch, ch)
                stage(skey(f"standalone_{ch}"), RUNNING)
                ctx.progress(
                    value=prog_ledger(),
                    status_text=(
                        f"{prefix}Standalone {ch_disp} stability selection "
                        f"({int(n_bootstraps)}×{int(n_trials_per_bootstrap)})..."
                    ),
                )
                ch_study_name = _standalone_study_name(ch)
                # Pin the active channel so _objective treats the trial's
                # composite as this channel's normalized value, then run the
                # same bootstrap stability search as CGI. The channel stays
                # pinned through the composite write below; the CGI state is
                # restored after the loop.
                engine._active_greenery_channel = ch
                ch_best = engine.bootstrap_stability_selection(
                    metric=objective_metric,
                    n_bootstraps=int(n_bootstraps),
                    n_trials_per_bootstrap=int(standalone_trials_per_bootstrap),
                    weight_bin_pct=int(weight_bin_pct),
                    min_cell_count=int(min_cell_count),
                    worst_quantile=float(worst_quantile),
                    max_pfer=max_pfer_arg,
                    spatial_resample=bool(spatial_split),
                    seed=42,
                    cancel_callback=cancel_check,
                    progress_callback=_study_progress_cb(
                        f"Standalone: {ch_disp}", skey(f"standalone_{ch}")
                    ),
                )
                engine.best_params = dict(ch_best)
                ch_headline_params = dict(ch_best)
                ch_stability_summary = _stability_summary(ch_best)
                stage(skey(f"standalone_{ch}"), DONE)

                # Test scoring, CIs, subset scores, composite TIFF, and the
                # optional MixedLM post-score form this channel's report stage.
                stage(skey(f"standalone_{ch}_report"), RUNNING)
                ctx.progress(
                    value=prog_ledger(),
                    status_text=f"{prefix}Standalone {ch_disp} test scoring & reports...",
                )
                ch_test = engine.evaluate_on_test(
                    params=ch_headline_params, metric=objective_metric
                )
                try:
                    ch_test_ci = engine.bootstrap_test_score_ci(
                        ch_headline_params,
                        objective_metric,
                        n_bootstrap=report_ci_n,
                        ci_level=0.95,
                        method="percentile",
                        seed=42,
                    )
                    ch_test["test_ci"] = ch_test_ci
                except Exception as exc:
                    _log_fusion(
                        "WARN",
                        f"[{label}] Standalone {ch} bootstrap CI failed: {exc}",
                    )
                ch_direction = _direction_sign(
                    engine, ch_headline_params, objective_metric
                )

                # Per-subset scores (train / val / test / all) for this
                # standalone, computed while ``_active_greenery_channel`` is
                # still pinned to ``ch`` so the composite is the single-channel
                # value the study optimized. Gives the standalone the same
                # subset-score parity the CGI study has.
                ch_subset_scores: dict | None = None
                try:
                    ch_subset_scores = engine.compute_subset_scores(
                        params=ch_headline_params,
                        metric=objective_metric,
                    )
                except Exception as exc:
                    _log_fusion(
                        "WARN",
                        f"[{label}] Standalone {ch} subset-score "
                        f"computation failed: {exc}",
                    )

                # Composite TIFF for this standalone (no master study → no
                # per-trial plots). Written into its own subdirectory.
                ch_report_dir = os.path.join(
                    job_artifacts_root, "study_results", f"standalone_{ch}"
                )
                ch_averaged_params: dict = dict(ch_headline_params)
                ch_composite_path = os.path.join(
                    job_artifacts_root, f"composite_greenery_{ch}.tif"
                )
                try:
                    engine.generate_composite_greenery_map(
                        output_path=ch_composite_path,
                    )
                    _log_fusion(
                        "OK",
                        f"[{label}] Standalone {ch}: composite TIFF written to "
                        f"{ch_report_dir}.",
                    )
                except Exception as exc:
                    _log_fusion(
                        "WARN",
                        f"[{label}] Standalone {ch} composite generation "
                        f"failed: {exc}",
                    )

                ch_best_value = float(ch_best.get("__cell_q_worst__", float("nan")))
                standalones_bundle[ch] = {
                    "channel": ch,
                    "best_params": ch_best,
                    "averaged_params": ch_averaged_params,
                    "best_value": ch_best_value,
                    "robust_trials": [],
                    "all_completed_trials": [],
                    "per_trial_test": {},
                    "test_results": ch_test,
                    "subset_scores": ch_subset_scores,
                    "stability_summary": ch_stability_summary,
                    "direction_sign": ch_direction,
                    "objective_metric": objective_metric,
                    "study_name": ch_study_name,
                    "report_dir": ch_report_dir,
                }
                # Post-hoc all-4 MixedLM metrics for this standalone, written
                # to a per-channel CSV beside the CGI one. The engine's
                # ``_active_greenery_channel`` is still set to ``ch`` here
                # so ``evaluate_on_test`` builds the right composite. Skipped
                # in OLS-scoring longitudinal mode (same gate as the CGI
                # post-score block above).
                if mixedlm_postscore_enabled:
                    ps_dir = os.path.join(job_artifacts_root, "study_results")
                    ps_basename = (
                        f"mixedlm_metrics__{label}__{ch}.csv"
                        if multi_outcome
                        else f"mixedlm_metrics__{ch}.csv"
                    )
                    try:
                        _compute_mixedlm_post_metrics(
                            engine,
                            ps_dir,
                            winning_params=ch_best,
                            csv_basename=ps_basename,
                            log=_log_fusion,
                        )
                    except Exception as exc:
                        _log_fusion(
                            "WARN",
                            f"[{label}] Standalone {ch} post-hoc MixedLM "
                            f"scoring failed: {exc}",
                        )
                stage(skey(f"standalone_{ch}_report"), DONE)

            # Restore the engine to its CGI state so downstream code that reads
            # ``engine.best_params`` / ``engine._active_greenery_channel`` sees
            # the CGI run.
            if standalones:
                engine.best_params = cgi_best_params
                engine._active_greenery_channel = "cgi"

            # ── AIC/BIC: is CGI justified over the best standalone channel? ──
            cgi_vs_standalone_aic_bic: dict | None = None
            if standalones:
                cgi_vs_standalone_aic_bic = _compare_cgi_vs_standalone(
                    engine,
                    standalones_bundle,
                    headline_params,
                    objective_metric,
                    longitudinal=longitudinal_spec is not None,
                    log=_log_fusion,
                )
                if cgi_vs_standalone_aic_bic and cgi_vs_standalone_aic_bic.get("ok"):
                    _log_fusion(
                        "OK",
                        f"[{label}] CGI vs best standalone "
                        f"({cgi_vs_standalone_aic_bic['best_channel']}): "
                        f"ΔBIC={cgi_vs_standalone_aic_bic['delta_bic']:.1f} → "
                        f"{cgi_vs_standalone_aic_bic['verdict']}.",
                    )

            # Paired bootstrap objective difference (CGI − standalone) on the
            # full dataset, in the metric's own units — computed for every
            # standalone channel as one family and Holm-corrected across it, so
            # comparing CGI against several channels doesn't inflate
            # significance. The headline verdict is the AIC/BIC best channel;
            # AIC/BIC stays a secondary report.
            cgi_vs_standalone_paired: dict | None = None
            cgi_vs_standalone_paired_family: list[dict] = []
            if standalones and standalones_bundle:
                for ch in [
                    c for c in ("veg", "terrain", "ndvi") if c in standalones_bundle
                ]:
                    ch_bundle = standalones_bundle[ch]
                    ch_params = (
                        ch_bundle.get("averaged_params")
                        or ch_bundle.get("best_params")
                        or {}
                    )
                    try:
                        pd_res = engine.paired_objective_difference(
                            headline_params,
                            ch_params,
                            ch,
                            objective_metric,
                            n_bootstrap=int(report_paired_bootstrap),
                            seed=42,
                        )
                    except Exception as exc:
                        _log_fusion(
                            "WARN", f"[{label}] Paired difference ({ch}) failed: {exc}"
                        )
                        pd_res = None
                    if pd_res:
                        cgi_vs_standalone_paired_family.append(pd_res)

            if cgi_vs_standalone_paired_family:
                from .. import statistical_testing as _stats_mod

                holm = _stats_mod.holm_bonferroni(
                    [d.get("p_value") for d in cgi_vs_standalone_paired_family]
                )
                for d, hp in zip(cgi_vs_standalone_paired_family, holm):
                    d["p_value_holm"] = None if hp != hp else float(hp)
                    d["family_size"] = len(cgi_vs_standalone_paired_family)
                best_ch = (cgi_vs_standalone_aic_bic or {}).get("best_channel")
                cgi_vs_standalone_paired = next(
                    (
                        d
                        for d in cgi_vs_standalone_paired_family
                        if d.get("standalone_channel") == best_ch
                    ),
                    None,
                ) or max(
                    cgi_vs_standalone_paired_family,
                    key=lambda d: d.get("observed_diff", float("-inf")),
                )
                _log_fusion(
                    "OK",
                    f"[{label}] CGI vs standalones (family of "
                    f"{len(cgi_vs_standalone_paired_family)}): headline "
                    f"`{cgi_vs_standalone_paired.get('standalone_channel')}` "
                    f"Δ={cgi_vs_standalone_paired.get('observed_diff'):.4f}, "
                    f"p={cgi_vs_standalone_paired.get('p_value'):.4g}, "
                    f"Holm p={cgi_vs_standalone_paired.get('p_value_holm')}.",
                )

            # ── Composite GeoTIFF (raster write) ──
            # Runs after the standalone stages so they can advance the ledger
            # first. Failures here don't kill the run — the composite_df is
            # already captured in ``bundle``.
            stage(skey("reports"), RUNNING)
            ctx.progress(
                value=prog_ledger(),
                status_text=f"{prefix}Generating reports + composite GeoTIFF...",
            )
            report_dir = os.path.join(job_artifacts_root, "study_results")
            cgi_composite_path = os.path.join(
                job_artifacts_root, "composite_greenery.tif"
            )
            averaged_params: dict | None = dict(headline_params)
            try:
                # No master study to plot trials from — generate the composite
                # TIFF directly from the stability-selection winning params.
                engine.generate_composite_greenery_map(
                    output_path=cgi_composite_path,
                )
                _log_fusion(
                    "OK",
                    f"[{label}] Composite TIFF written to {job_artifacts_root}.",
                )
            except Exception as exc:
                _log_fusion(
                    "WARN",
                    f"[{label}] Composite generation failed: {exc}",
                )
            stage(skey("reports"), DONE)

            cgi_subset_scores: dict | None = None
            try:
                cgi_subset_scores = engine.compute_subset_scores(
                    params=averaged_params or best_params,
                    metric=objective_metric,
                )
            except Exception as exc:
                _log_fusion(
                    "WARN",
                    f"[{label}] CGI subset-score computation failed: {exc}",
                )

            covariate_impact: dict | None = None
            try:
                covariate_impact = engine.compute_covariate_impact(
                    params=averaged_params or best_params,
                    metric=objective_metric,
                )
            except Exception as exc:
                _log_fusion(
                    "WARN",
                    f"[{label}] Covariate impact computation failed: {exc}",
                )

            # Longitudinal exposure–decline terms (greenery × time) on the
            # winning composite — the overall slope plus the optional
            # between/within decomposition selected in the spec.
            decline_terms: dict | None = None
            if longitudinal_spec is not None:
                try:
                    decline_terms = engine.compute_decline_terms(
                        params=averaged_params or best_params
                    )
                except Exception as exc:
                    _log_fusion(
                        "WARN",
                        f"[{label}] Decline-terms computation failed: {exc}",
                    )

            bundle = {
                "best_params": best_params,
                "averaged_params": averaged_params,
                "best_value": cgi_best_value,
                "robust_trials": robust_trials,
                "per_trial_test": cgi_per_trial_test,
                "composite_df": composite_df,
                "objective_metric": objective_metric,
                "test_results": test_results,
                "subset_scores": cgi_subset_scores,
                "covariate_impact": covariate_impact,
                "decline_terms": decline_terms,
                "target_feature": target_feature,
                # Run details persisted so the results panel survives a disk
                # reload (when the live engine is gone): user-facing covariate
                # names + their types, the formula, and the target / outcome.
                "covariate_columns": list(
                    getattr(engine, "_covariate_columns_user", None) or outcome_covs
                ),
                "covariate_types": dict(getattr(engine, "covariate_types", {}) or {}),
                "covariate_dummy_map": dict(
                    getattr(engine, "_covariate_dummy_map", {}) or {}
                ),
                "cgi_formula": cgi_formula,
                "target_display_name": target_display_name,
                "outcome_label": target_feature or target_display_name,
                "standalones": standalones_bundle,
                "artifacts_dir": job_artifacts_root,
                "composite_path": cgi_composite_path,
                "report_dir": report_dir,
                # ``None`` when the collinearity check wasn't requested.
                "collinearity_report": getattr(engine, "_collinearity_report", None),
                "disabled_channels": sorted(
                    getattr(engine, "_disabled_channels", set()) or set()
                ),
                # Stability-selection diagnostics, the held-out direction sign,
                # and the AIC/BIC verdict vs the best standalone (when run).
                "stability_summary": cgi_stability_summary,
                "direction_sign": cgi_direction,
                "cgi_vs_standalone_aic_bic": cgi_vs_standalone_aic_bic,
                # Held-out test effect (headline) + descriptive whole-data CI +
                # held-out permutation p-value, and the paired objective
                # difference vs each standalone (headline = best channel; the
                # family carries Holm-corrected p-values).
                "cgi_effects": cgi_effects,
                "cgi_vs_standalone_paired": cgi_vs_standalone_paired,
                "cgi_vs_standalone_paired_family": cgi_vs_standalone_paired_family,
            }
            by_target[label] = bundle

            # Persist every test result (CGI + standalones) to disk: a
            # machine-readable manifest plus tidy CSVs. Best-effort — a write
            # failure is logged, never fatal to the run.
            try:
                result_files = _write_fusion_outputs(
                    output_dir=report_dir,
                    label=label,
                    multi_outcome=multi_outcome,
                    objective_metric=objective_metric,
                    formula_name=cgi_formula,
                    cgi_bundle=bundle,
                    standalones_bundle=standalones_bundle,
                    aic_bic=cgi_vs_standalone_aic_bic,
                    covariate_impact=covariate_impact,
                    collinearity_report=bundle.get("collinearity_report"),
                    run_config_record=run_config_record,
                    log=_log_fusion,
                )
                output_paths.extend(result_files)
            except Exception as exc:
                _log_fusion(
                    "WARN",
                    f"[{label}] Could not write fusion result files: {exc}",
                )

            ctx.progress(value=prog(1.0))

        sole_label = ordered_labels[0]
        results_payload: dict[str, Any] = {
            "mode": "multi" if multi_outcome else "single",
            "ordered_labels": ordered_labels,
            "by_target": by_target,
            "multi_objective_requested": bool(multi_objective_requested)
            and multi_outcome,
        }
        if not multi_outcome:
            results_payload.update(by_target[sole_label])

        # Fusion runs in a spawn child, so live engine objects (unpicklable and
        # large) never cross the process boundary. Ship only the JSON-safe
        # results — the same payload persisted below — with no live engine. The
        # results view hydrates from this exactly as it does from disk after a
        # Streamlit restart: the composite viewer reads its GeoTIFFs from disk
        # and simply omits the live-engine target overlay.
        jsonsafe_payload = _jsonsafe_results(results_payload)
        ctx.set_extra(
            engine=None,
            engines_by_target={},
            results=jsonsafe_payload,
            artifacts_dir=job_artifacts_root,
        )

        # Compact on-disk results bundle so a completed run re-opens after a
        # Streamlit restart (``rec.extra`` isn't persisted, but ``output_paths``
        # is). Best-effort — a write failure never fails the run.
        try:
            bundle_json_path = os.path.join(job_artifacts_root, "results_bundle.json")
            with open(bundle_json_path, "w", encoding="utf-8") as _bf:
                json.dump(jsonsafe_payload, _bf, default=str)
            output_paths.append(bundle_json_path)
        except Exception as exc:
            _log_fusion("WARN", f"Could not write results bundle JSON: {exc}")

        ctx.progress(value=1.0, status_text="Completed")
        return {"output_paths": output_paths}

    finally:
        if target_cleanup_dir:
            shutil.rmtree(target_cleanup_dir, ignore_errors=True)
        if target_cleanup_file and os.path.isfile(target_cleanup_file):
            try:
                os.remove(target_cleanup_file)
            except OSError:
                pass
