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
from geofuse.jobs.stage_ledger import DONE, RUNNING, StageLedger
from geofuse.logger import get_logger
from geofuse.ndvi import NDVIEngine
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
    )

    if ctx.is_cancelled():
        return {"output_paths": []}

    ctx.progress(value=1.0, status_text="Writing outputs")

    res_df = gpd.GeoDataFrame(
        dataset_data["accumulated"], crs=dataset_data["processed"].crs
    )
    if "orig_index" in res_df.columns:
        res_df.set_index("orig_index", inplace=True)
        res_df.index.name = None
    if res_df.crs is None:
        res_df = res_df.set_crs("EPSG:4326")

    out_name = os.path.splitext(fname)[0]
    output_paths: list[str] = []
    meta = dataset_data.get("meta") or {}
    grid_crs_wkt = meta.get("grid_crs_wkt")
    clusters = meta.get("clusters") or []

    # All GVI outputs land in the toolbox-selected planar CRS so cells stay
    # square in metres across the entire study area. Falls back to EPSG:4326
    # only when the runner is invoked without clustered-grid metadata
    # (direct API callers with point inputs + buffer=0).
    if grid_crs_wkt:
        res_df = res_df.to_crs(grid_crs_wkt)
    dataset_data["results"] = res_df

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
    save_features_samples: bool = False,
    sample_radius_m: float = 0.0,
    sample_stat: str = "mean",
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

    # Sample-at-features uses the *un-buffered* raw input — the user wants
    # NDVI at their original locations, not at the buffered download AOI.
    sample_at = dataset_data["raw"] if save_features_samples else None

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
        sample_at_features=sample_at,
        sample_radius_m=sample_radius_m,
        sample_stat=sample_stat,
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
    samples_gpkg = os.path.join(output_dir, f"{output_name}_ndvi_at_features.gpkg")
    if save_features_samples and os.path.exists(samples_gpkg):
        output_paths.append(samples_gpkg)

    ctx.progress(value=1.0, status_text="Completed")
    return {"output_paths": output_paths}


def run_ndvi_column(
    ctx: JobContext,
    *,
    fname: str,
    dataset_data: dict,
    date_column: str,
    window_days: int,
    cloud_pct: int,
    resolution: int,
    buffer_m: int,
    output_dir: str,
    save_geotiff: bool,
    save_geojson: bool,
    save_gpkg: bool = False,
) -> dict:
    """Run NDVI extraction per feature using a date column."""
    from geofuse.crs_utils import buffer_gdf_union_metres

    gdf = dataset_data["raw"].copy()
    gdf["_parsed_date"] = pd.to_datetime(gdf[date_column], errors="coerce")
    gdf = gdf.dropna(subset=["_parsed_date"])
    if gdf.empty:
        raise ValueError("No valid dates found in the selected column.")

    unique_dates = sorted(gdf["_parsed_date"].dt.date.unique())
    n_dates = len(unique_dates)
    engine = NDVIEngine()
    base_extent = gpd.GeoDataFrame(
        {"geometry": [gdf.geometry.union_all()]}, crs=gdf.crs
    )
    full_extent = buffer_gdf_union_metres(base_extent, buffer_m)
    all_results: list[gpd.GeoDataFrame] = []
    output_paths: list[str] = []
    base_name = fname.replace(".geojson", "")

    for idx, target_date in enumerate(unique_dates):
        if ctx.is_cancelled():
            return {"output_paths": output_paths}

        start_d = target_date - timedelta(days=window_days)
        end_d = target_date + timedelta(days=window_days)
        date_str = target_date.strftime("%Y%m%d")
        tmp_name = f"{base_name}_{date_str}_tmp"

        ctx.progress(status_text=f"Processing date {idx + 1}/{n_dates}: {target_date}")

        span = 1.0 / max(n_dates, 1)
        base = idx / max(n_dates, 1)
        on_progress = _ndvi_on_progress_factory(ctx, base_offset=base, span=span)

        def check_cancel() -> bool:
            return ctx.is_cancelled()

        result = engine.download_and_process(
            geometry=full_extent,
            start_date=start_d.isoformat(),
            end_date=end_d.isoformat(),
            output_name=tmp_name,
            cloud_max=cloud_pct,
            resolution=resolution,
            folder=output_dir,
            cancel_callback=check_cancel,
            ndvi_progress_callback=on_progress,
            write_geotiff=True,
            write_geojson=False,
        )
        if result.get("status") == "cancelled":
            return {"output_paths": output_paths}
        if result.get("status") != "success":
            _log_ndvi(
                "WARN", f"Column run for {target_date} failed: {result.get('message')}"
            )
            continue

        tif_path = os.path.join(output_dir, f"{tmp_name}_ndvi.tif")
        if not os.path.exists(tif_path):
            continue

        # Delegate the exact-pixel sample to the shared helper — it
        # handles the planar-CRS reprojection, nodata sentinel, and
        # NaN filtering uniformly with every other raster consumer.
        date_gdf = gdf[gdf["_parsed_date"].dt.date == target_date].copy()
        out = sample_raster_at_features(
            tif_path,
            date_gdf,
            band=1,
            radius_m=0.0,
            stat="mean",
            value_column="NDVI",
        )
        out["ndvi_date"] = target_date.isoformat()
        all_results.append(out)

        if not save_geotiff and os.path.isfile(tif_path):
            os.remove(tif_path)
        elif save_geotiff:
            output_paths.append(tif_path)

    if all_results:
        merged = gpd.GeoDataFrame(
            pd.concat(all_results, ignore_index=True), crs=all_results[0].crs
        )
        merged = merged.drop(columns=["_parsed_date"], errors="ignore")
        if save_geojson:
            gj_path = os.path.join(output_dir, f"{base_name}_temporal_ndvi.geojson")
            reproject_geodataframe_to_wgs84(merged).to_file(gj_path, driver="GeoJSON")
            output_paths.append(gj_path)
        if save_gpkg:
            gpkg_path = os.path.join(output_dir, f"{base_name}_temporal_ndvi.gpkg")
            merged.to_file(gpkg_path, driver="GPKG", layer="ndvi_samples")
            output_paths.append(gpkg_path)
        dataset_data["results"] = merged

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
) -> str:
    """Filesystem-safe Optuna ``study_name`` (also the SQLite filename stem)."""
    stem = os.path.splitext(target_display_name or "target")[0]
    parts = [stem, str(label), objective_metric]
    if suffix:
        parts.append(suffix)
    raw = "__".join(parts)
    return _STUDY_NAME_UNSAFE.sub("_", raw).strip("_") or "fusion_study"


# Ordered pipeline steps a fusion run moves through per target outcome. The
# staged-resume ledger (geofuse.jobs.stage_ledger) records each so the monitor
# shows where a run is and a stopped job reports where it left off. Resume itself
# is content-addressed (metric cache, pre-aggregation cache, Optuna study), so
# re-running the same job reuses/resumes each on-disk artifact transparently —
# the ledger is the visibility layer over that durability.
_FUSION_STAGE_STEPS: tuple[tuple[str, str], ...] = (
    ("load_target", "Load target"),
    ("load_metrics", "Load metric maps"),
    ("preaggregate", "Spatial pre-processing"),
    ("split", "Split train / test folds"),
    ("optimize", "Optimize CGI study"),
    ("robust", "Filter robust trials"),
    ("evaluate", "Evaluate on test"),
    ("apply", "Apply fusion weights"),
)

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
) -> StageLedger:
    """Fresh ledger covering every (outcome, step) pair in run order.

    For each outcome the 8-step CGI pipeline (`_FUSION_STAGE_STEPS`) lands
    first, then one stage per enabled standalone metric — those reuse the
    already-built split + pre-aggregation cache, so each is a single
    optimize/robust/evaluate burst that's compact enough to fit in one
    ledger row.
    """
    standalones = list(standalone_channels or [])
    steps: list[tuple[str, str]] = []
    for label in labels:
        for step_key, step_label in _FUSION_STAGE_STEPS:
            key = _fusion_stage_key(label, step_key, multi=multi)
            disp = f"[{label}] {step_label}" if multi else step_label
            steps.append((key, disp))
        for ch in standalones:
            key = _fusion_stage_key(label, f"standalone_{ch}", multi=multi)
            ch_lbl = _STANDALONE_CHANNEL_LABELS.get(ch, ch)
            disp_step = f"Standalone {ch_lbl} study"
            disp = f"[{label}] {disp_step}" if multi else disp_step
            steps.append((key, disp))
    return StageLedger.from_steps(steps)


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
    ndvi_resolution_m: float | None,
    gvi_grid_spacing_m: float | None,
    n_bins: int,
    veg_path: str | None,
    terrain_path: str | None,
    ndvi_path: str | None,
    cache_metrics: bool,
    test_size: float,
    k_folds: int,
    n_trials: int,
    n_startup_trials: int,
    objective_metric: str,
    pruner_type: str,
    sampler_type: str,
    gvi_api_key: str | None,
    ndvi_start_date: str | None,
    ndvi_end_date: str | None,
    ndvi_project_id: str | None,
    multi_objective_requested: bool,
    output_dir: str,
    MetricFusionEngine,
    target_display_name: str = "target",
    resume_existing_study: bool = True,
    cgi_formula: str = "weighted_average",
    covariate_columns: list[str] | None = None,
    standalone_channels: list[str] | None = None,
) -> dict:
    """Run fusion optimization. Mirrors the previous ``_fusion_worker``."""
    try:
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

        by_target: dict = {}
        engines_by_target: dict = {}
        output_paths: list[str] = []

        # Pre-compute every outcome label so the staged-resume ledger can list
        # all (outcome, step) stages up front; the monitor then shows pending
        # stages before the runner reaches them.
        all_labels = [
            (t if t is not None else f"raster_band_{target_band}") for t in targets
        ]
        ordered_labels: list[str] = list(all_labels)
        ledger = _build_fusion_ledger(
            all_labels, multi=multi_outcome, standalone_channels=standalones
        )
        ctx.update_stage_ledger(ledger.to_dict())

        def stage(key: str, status: str, message: str = "") -> None:
            """Record a stage transition in the ledger and persist it."""
            ledger.set_status(
                key,
                status,
                progress=1.0 if status == DONE else 0.0,
                message=message or None,
            )
            ctx.update_stage_ledger(ledger.to_dict())

        cache_dir = os.path.join(output_dir, "fusion_cache")

        for ti, target_feature in enumerate(targets):
            if ctx.is_cancelled():
                return {"output_paths": output_paths}

            label = all_labels[ti]
            prefix = f"[{label}] " if n_t > 1 else ""

            def skey(step: str, _label: str = label) -> str:
                return _fusion_stage_key(_label, step, multi=multi_outcome)

            def prog(local: float) -> float:
                return (ti + local) / n_t

            ctx.progress(
                value=prog(0.05),
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
            )

            ctx.progress(value=prog(0.1), status_text=f"{prefix}Loading target data...")
            stage(skey("load_target"), RUNNING)
            engine.load_target()
            stage(skey("load_target"), DONE)

            if not veg_path:
                ctx.progress(
                    value=prog(0.15),
                    status_text=f"{prefix}Downloading GVI Vegetation data...",
                )
            elif not terrain_path:
                ctx.progress(
                    value=prog(0.20),
                    status_text=f"{prefix}Downloading GVI Terrain data...",
                )
            elif not ndvi_path:
                ctx.progress(
                    value=prog(0.25),
                    status_text=f"{prefix}Downloading NDVI satellite data...",
                )
            else:
                ctx.progress(
                    value=prog(0.15),
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
            stage(skey("load_metrics"), DONE)

            # Sample materialization + the on-disk pre-aggregation cache form one
            # "spatial pre-processing" stage; prepare_fusion_data feeds the cache.
            stage(skey("preaggregate"), RUNNING)
            ctx.progress(
                value=prog(0.28),
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
                ctx.progress(
                    value=prog(0.30 + 0.04 * pct / 100),
                    status_text=(
                        f"{prefix}Spatial pre-processing: "
                        f"{current:,}/{total:,} entities ({pct}%)"
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

            stage(skey("split"), RUNNING)
            ctx.progress(
                value=prog(0.34),
                status_text=f"{prefix}Splitting data into train/val/test folds...",
            )
            engine.split_data(
                fusion_df=fusion_df,
                test_size=test_size,
                k_folds=k_folds,
                random_state=42,
            )
            stage(skey("split"), DONE)

            stage(skey("optimize"), RUNNING)
            ctx.progress(
                value=prog(0.35),
                status_text=f"{prefix}Optimizing ({n_trials} trials)...",
            )
            study_dir = os.path.join(output_dir, "fusion_studies")
            # When the user opts out of resume, suffix the study name with a
            # timestamp so a fresh SQLite file is created instead of attaching
            # to the existing one.
            suffix = (
                ""
                if resume_existing_study
                else datetime.now().strftime("%Y%m%dT%H%M%S")
            )
            study_name = _build_fusion_study_name(
                target_display_name=target_display_name,
                label=label,
                objective_metric=objective_metric,
                suffix=suffix,
            )
            best_params = engine.optimize_fusion(
                n_trials=n_trials,
                n_startup_trials=n_startup_trials,
                objective_metric=objective_metric,
                pruner_type=pruner_type if pruner_type != "none" else None,
                sampler_type=sampler_type,
                seed=42,
                show_progress=False,
                study_name=study_name,
                study_dir=study_dir,
                cancel_callback=cancel_check,
            )
            if ctx.is_cancelled():
                return {"output_paths": output_paths}
            stage(skey("optimize"), DONE)

            stage(skey("robust"), RUNNING)
            ctx.progress(
                value=prog(0.85), status_text=f"{prefix}Filtering robust trials..."
            )
            robust_trials = engine.get_robust_trials(
                method="auto", p_threshold=0.05, tolerance=0.1, min_trials=10
            )
            stage(skey("robust"), DONE)

            stage(skey("evaluate"), RUNNING)
            ctx.progress(
                value=prog(0.9), status_text=f"{prefix}Evaluating on test set..."
            )
            test_results = engine.evaluate_on_test(
                params=best_params, metric=objective_metric
            )
            stage(skey("evaluate"), DONE)

            stage(skey("apply"), RUNNING)
            ctx.progress(
                value=prog(0.95), status_text=f"{prefix}Applying fusion weights..."
            )
            composite_df = engine.apply_fusion()
            stage(skey("apply"), DONE)

            # ── Standalone single-metric studies ────────────────────────────
            # One Optuna study per enabled channel, reusing the same engine,
            # the already-built per-(entity, radius) cache, and the train/val/
            # test split. Each gets its own study SQLite file (suffix = the
            # channel name) so trials don't pool with the CGI study.
            #
            # Each ``optimize_fusion`` call replaces ``engine.study`` /
            # ``engine.best_params`` / ``engine._active_greenery_channel`` with
            # the standalone's, so we snapshot the CGI state up front and
            # restore it after the loop. The results UI reads
            # ``engine.study.trials`` etc. on the returned engine and expects
            # the CGI study there.
            cgi_study = engine.study
            cgi_best_value = engine.study.best_value if engine.study else None
            cgi_best_params = engine.best_params

            standalones_bundle: dict[str, dict] = {}
            for ch in standalones:
                if ctx.is_cancelled():
                    return {"output_paths": output_paths}
                ch_disp = _STANDALONE_CHANNEL_LABELS.get(ch, ch)
                stage(skey(f"standalone_{ch}"), RUNNING)
                ctx.progress(
                    value=prog(0.95),
                    status_text=(
                        f"{prefix}Standalone {ch_disp} study ({n_trials} trials)..."
                    ),
                )
                ch_suffix = ch if resume_existing_study else f"{ch}_{suffix}"
                ch_study_name = _build_fusion_study_name(
                    target_display_name=target_display_name,
                    label=label,
                    objective_metric=objective_metric,
                    suffix=ch_suffix,
                )
                ch_best = engine.optimize_fusion(
                    n_trials=n_trials,
                    n_startup_trials=n_startup_trials,
                    objective_metric=objective_metric,
                    pruner_type=pruner_type if pruner_type != "none" else None,
                    sampler_type=sampler_type,
                    seed=42,
                    show_progress=False,
                    study_name=ch_study_name,
                    study_dir=study_dir,
                    cancel_callback=cancel_check,
                    greenery_channel=ch,
                )
                if ctx.is_cancelled():
                    return {"output_paths": output_paths}
                ch_robust = engine.get_robust_trials(
                    method="auto", p_threshold=0.05, tolerance=0.1, min_trials=10
                )
                ch_test = engine.evaluate_on_test(
                    params=ch_best, metric=objective_metric
                )
                standalones_bundle[ch] = {
                    "best_params": ch_best,
                    "best_value": engine.study.best_value,
                    "robust_trials": ch_robust,
                    "test_results": ch_test,
                    "objective_metric": objective_metric,
                    "study_name": ch_study_name,
                }
                stage(skey(f"standalone_{ch}"), DONE)

            # Restore the engine to its CGI-study state so downstream UI code
            # that reads ``engine.study.trials`` / ``engine.best_params`` /
            # ``engine._active_greenery_channel`` sees the combined run.
            if standalones:
                engine.study = cgi_study
                engine.best_params = cgi_best_params
                engine._active_greenery_channel = "cgi"

            bundle = {
                "best_params": best_params,
                "best_value": cgi_best_value,
                "robust_trials": robust_trials,
                "composite_df": composite_df,
                "objective_metric": objective_metric,
                "test_results": test_results,
                "target_feature": target_feature,
                "standalones": standalones_bundle,
            }
            by_target[label] = bundle
            engines_by_target[label] = engine
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

        ctx.set_extra(
            engine=None if multi_outcome else engines_by_target[sole_label],
            engines_by_target=engines_by_target,
            results=results_payload,
        )
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
