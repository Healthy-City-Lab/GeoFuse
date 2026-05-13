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

from geofuse.gvi import GVIEngine
from geofuse.logger import get_logger
from geofuse.ndvi import NDVIEngine
from geofuse.persistence.job_executor import JobContext
from geofuse.vision import get_best_device

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

    def on_progress(curr: int, total: int) -> None:
        if total <= 0:
            return
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
    dataset_data["results"] = res_df

    out_name = os.path.splitext(fname)[0]
    output_paths: list[str] = []

    if save_geojson:
        gj_path = os.path.join(output_dir, f"{out_name}_gvi.geojson")
        res_df.to_file(gj_path, driver="GeoJSON")
        output_paths.append(gj_path)

    if dataset_data["meta"] and save_geotiff:
        from rasterio.transform import rowcol

        meta = dataset_data["meta"]
        arr_veg = np.full((meta["height"], meta["width"]), np.nan, dtype=np.float32)
        arr_ter = np.full((meta["height"], meta["width"]), np.nan, dtype=np.float32)
        valid = res_df.dropna(subset=["gvi_veg"])
        if not valid.empty:
            rows, cols = rowcol(
                meta["transform"],
                valid.geometry.x.values,
                valid.geometry.y.values,
            )
            rows = np.clip(rows, 0, meta["height"] - 1)
            cols = np.clip(cols, 0, meta["width"] - 1)
            arr_veg[rows, cols] = valid["gvi_veg"].values
            arr_ter[rows, cols] = valid["gvi_ter"].values
        tif_path = os.path.join(output_dir, f"{out_name}_gvi.tif")
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=meta["height"],
            width=meta["width"],
            count=2,
            dtype=np.float32,
            crs=meta["crs"],
            transform=meta["transform"],
            nodata=np.nan,
        ) as dst:
            dst.write(arr_veg, 1)
            dst.set_band_description(1, "Veg")
            dst.write(arr_ter, 2)
            dst.set_band_description(2, "Ter")
        output_paths.append(tif_path)

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
    )

    if result.get("status") == "cancelled":
        return {"output_paths": []}
    if result.get("status") != "success":
        raise RuntimeError(result.get("message", "NDVI run failed"))

    output_paths: list[str] = []
    tif_path = os.path.join(output_dir, f"{output_name}_ndvi.tif")
    if save_geotiff and os.path.exists(tif_path):
        output_paths.append(tif_path)
    gj_path = os.path.join(output_dir, f"{output_name}_ndvi.geojson")
    if save_geojson and os.path.exists(gj_path):
        output_paths.append(gj_path)

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
            _log_ndvi("WARN", f"Column run for {target_date} failed: {result.get('message')}")
            continue

        tif_path = os.path.join(output_dir, f"{tmp_name}_ndvi.tif")
        if not os.path.exists(tif_path):
            continue

        date_gdf = gdf[gdf["_parsed_date"].dt.date == target_date].copy()
        with rasterio.open(tif_path) as src:
            for row_idx, row in date_gdf.iterrows():
                geom = row.geometry
                pt = geom if geom.geom_type == "Point" else geom.centroid
                try:
                    r, c = src.index(pt.x, pt.y)
                    window = rasterio.windows.Window(c, r, 1, 1)
                    val = src.read(1, window=window)
                    ndvi_val = float(val[0][0]) if val.size > 0 else np.nan
                    if ndvi_val == -9999:
                        ndvi_val = np.nan
                except Exception:
                    ndvi_val = np.nan
                date_gdf.at[row_idx, "NDVI"] = ndvi_val
                date_gdf.at[row_idx, "ndvi_date"] = target_date.isoformat()

        all_results.append(date_gdf)
        if not save_geotiff and os.path.isfile(tif_path):
            os.remove(tif_path)
        elif save_geotiff:
            output_paths.append(tif_path)

    if all_results:
        merged = gpd.GeoDataFrame(
            pd.concat(all_results, ignore_index=True), crs=gdf.crs
        )
        merged = merged.drop(columns=["_parsed_date"], errors="ignore")
        out_path = os.path.join(output_dir, f"{base_name}_temporal_ndvi.geojson")
        if save_geojson:
            merged.to_file(out_path, driver="GeoJSON")
            output_paths.append(out_path)
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
    pre_aggregate: bool = False,
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

        by_target: dict = {}
        engines_by_target: dict = {}
        ordered_labels: list[str] = []
        output_paths: list[str] = []

        cache_dir = os.path.join(output_dir, "fusion_cache")

        for ti, target_feature in enumerate(targets):
            if ctx.is_cancelled():
                return {"output_paths": output_paths}

            label = (
                target_feature
                if target_feature is not None
                else f"raster_band_{target_band}"
            )
            ordered_labels.append(label)
            prefix = f"[{label}] " if n_t > 1 else ""

            def prog(local: float) -> float:
                return (ti + local) / n_t

            ctx.progress(
                value=prog(0.05),
                status_text=f"{prefix}Initializing fusion engine...",
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
            )

            ctx.progress(value=prog(0.1), status_text=f"{prefix}Loading target data...")
            engine.load_target()

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

            ctx.progress(
                value=prog(0.28),
                status_text=(
                    f"{prefix}Preparing fusion samples + splitting data "
                    "(can take a while on large polygon targets)..."
                ),
            )
            engine.split_data(test_size=test_size, k_folds=k_folds, random_state=42)

            # Optional spatial pre-processing: pre-aggregate per-point × radius
            # × stat lookup table so every Optuna trial is a numpy.take.
            if pre_aggregate:
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
                            f"{current:,}/{total:,} points ({pct}%)"
                        ),
                    )
                    ctx.heartbeat()

                completed = engine.precompute_aggregations(
                    progress_callback=preaggr_progress,
                    cancel_callback=cancel_check,
                )
                # Clear the dedicated preaggr_progress sub-bar so it doesn't
                # linger past this stage in the monitor.
                ctx.set_extra(preaggr_progress=None)
                if not completed or ctx.is_cancelled():
                    return {"output_paths": output_paths}

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

            ctx.progress(
                value=prog(0.85), status_text=f"{prefix}Filtering robust trials..."
            )
            robust_trials = engine.get_robust_trials(
                method="auto", p_threshold=0.05, tolerance=0.1, min_trials=10
            )

            ctx.progress(
                value=prog(0.9), status_text=f"{prefix}Evaluating on test set..."
            )
            test_results = engine.evaluate_on_test(
                params=best_params, metric=objective_metric
            )

            ctx.progress(
                value=prog(0.95), status_text=f"{prefix}Applying fusion weights..."
            )
            composite_df = engine.apply_fusion()

            bundle = {
                "best_params": best_params,
                "best_value": engine.study.best_value,
                "robust_trials": robust_trials,
                "composite_df": composite_df,
                "objective_metric": objective_metric,
                "test_results": test_results,
                "target_feature": target_feature,
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
