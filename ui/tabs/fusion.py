import base64
import glob
import io
import os
import tempfile
import threading
import time
import uuid
from datetime import date, datetime

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from PIL import Image as PILImage
from shapely.geometry import box as shapely_box
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_folium import st_folium

try:
    from geofuse.fusion import MetricFusionEngine as _MetricFusionEngine
except ImportError:
    _MetricFusionEngine = None


# ---------------------------------------------------------------------------
# Metric-file helpers
# ---------------------------------------------------------------------------


def _scan_metric_files(output_dir: str, suffix: str) -> list:
    """Return sorted list of (label, path) for all *_{suffix}.geojson in output_dir."""
    pattern = os.path.join(output_dir, f"*_{suffix}.geojson")
    return [(os.path.basename(p), p) for p in sorted(glob.glob(pattern))]


def _compute_buffered_extent(
    tmp_target_path: str,
    is_geojson: bool,
    buffer_meters: float,
) -> "gpd.GeoDataFrame | None":
    """Compute the buffered target extent as a GeoDataFrame in EPSG:4326."""
    try:
        if is_geojson:
            gdf = gpd.read_file(tmp_target_path)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            else:
                gdf = gdf.to_crs("EPSG:4326")
        else:  # GeoTIFF
            with rasterio.open(tmp_target_path) as src:
                b = src.bounds
                src_crs = src.crs
            gdf = gpd.GeoDataFrame(
                {"geometry": [shapely_box(b.left, b.bottom, b.right, b.top)]},
                crs=src_crs,
            ).to_crs("EPSG:4326")

        utm_crs = gdf.estimate_utm_crs()
        gdf_utm = gdf.to_crs(utm_crs)
        buffered_geom = gdf_utm.geometry.union_all().buffer(buffer_meters)
        return gpd.GeoDataFrame({"geometry": [buffered_geom]}, crs=utm_crs).to_crs(
            "EPSG:4326"
        )
    except Exception:
        return None


def _check_coverage(metric_path: str, buffered_gdf: "gpd.GeoDataFrame") -> bool:
    """Return True if the metric file's spatial extent fully covers buffered_gdf."""
    try:
        if metric_path.lower().endswith((".tif", ".tiff")):
            with rasterio.open(metric_path) as src:
                metric_box = shapely_box(*src.bounds)
                metric_crs = src.crs
        else:
            mdf = gpd.read_file(metric_path)
            metric_box = shapely_box(*mdf.total_bounds)
            metric_crs = mdf.crs

        target_geom = buffered_gdf.to_crs(metric_crs).geometry.union_all()
        return metric_box.covers(target_geom)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Background worker (module-level)
# ---------------------------------------------------------------------------


def _fusion_worker(
    job_id,
    target_path,
    target_feature,
    target_band,
    buffer_meters,
    gvi_buffer_min_m,
    gvi_buffer_max_m,
    gvi_buffer_step_m,
    ndvi_buffer_min_m,
    ndvi_buffer_max_m,
    ndvi_buffer_step_m,
    ndvi_resolution_m,
    gvi_grid_spacing_m,
    n_bins,
    veg_path,
    terrain_path,
    ndvi_path,
    cache_metrics,
    test_size,
    k_folds,
    n_trials,
    n_startup_trials,
    objective_metric,
    pruner_type,
    sampler_type,
    gvi_api_key,
    ndvi_start_date,
    ndvi_end_date,
    ndvi_project_id,
    output_dir,
    job_tracker_dict,
    MetricFusionEngine,
):
    try:
        job_tracker_dict[job_id]["status"] = "Initializing fusion engine..."
        job_tracker_dict[job_id]["progress"] = 0.05

        print(f"[FUSION] Starting fusion job {job_id}")
        print(
            "[FUSION] Buffer ladders (UI; engine uses buffer_meters=max until wired): "
            f"GVI [{gvi_buffer_min_m}, {gvi_buffer_max_m}] step={gvi_buffer_step_m} m, "
            f"NDVI [{ndvi_buffer_min_m}, {ndvi_buffer_max_m}] step={ndvi_buffer_step_m} m"
        )
        if ndvi_resolution_m is not None:
            print(f"[FUSION] NDVI export resolution (pending engine): {ndvi_resolution_m} m")
        if gvi_grid_spacing_m is not None:
            print(f"[FUSION] GVI sampling grid spacing (pending engine): {gvi_grid_spacing_m} m")

        engine = MetricFusionEngine(
            target_file=target_path,
            target_feature=target_feature,
            target_band=target_band,
            buffer_meters=buffer_meters,
            n_bins=n_bins,
            cache_dir=os.path.join(output_dir, "fusion_cache"),
        )

        print("[FUSION] Engine initialized successfully")

        job_tracker_dict[job_id]["status"] = "Loading target data..."
        job_tracker_dict[job_id]["progress"] = 0.1
        engine.load_target()

        print("[FUSION] Target loaded successfully")
        print("[FUSION] Starting metric loading/download phase")
        print(
            f"[FUSION] veg_path={veg_path}, terrain_path={terrain_path}, "
            f"ndvi_path={ndvi_path}"
        )

        if not veg_path:
            job_tracker_dict[job_id]["status"] = "Downloading GVI Vegetation data..."
            job_tracker_dict[job_id]["progress"] = 0.15
            print("[FUSION] Will auto-download GVI vegetation")
        elif not terrain_path:
            job_tracker_dict[job_id]["status"] = "Downloading GVI Terrain data..."
            job_tracker_dict[job_id]["progress"] = 0.20
            print("[FUSION] Will auto-download GVI terrain")
        elif not ndvi_path:
            job_tracker_dict[job_id]["status"] = "Downloading NDVI satellite data..."
            job_tracker_dict[job_id]["progress"] = 0.25
            print("[FUSION] Will auto-download NDVI")
        elif veg_path and terrain_path and ndvi_path:
            job_tracker_dict[job_id]["status"] = "Loading provided metric files..."
            job_tracker_dict[job_id]["progress"] = 0.15
            print("[FUSION] Loading from provided files")

        if job_tracker_dict[job_id]["cancel"]:
            job_tracker_dict[job_id]["status"] = "Cancelled"
            return

        last_update_time = {"veg": 0, "terrain": 0}

        def gvi_progress_callback(component, curr, total):
            current_time = time.time()
            if (
                current_time - last_update_time.get(component, 0) < 0.5
                and curr != total
            ):
                return
            last_update_time[component] = current_time
            job_tracker_dict[job_id]["gvi_progress"] = {
                "component": component,
                "current": curr,
                "total": total,
                "percent": 100 * curr / total if total > 0 else 0,
            }

        def cancel_check():
            return job_tracker_dict[job_id]["cancel"]

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
        )

        print("[FUSION] Metrics loaded successfully")

        job_tracker_dict[job_id]["status"] = "Splitting data..."
        job_tracker_dict[job_id]["progress"] = 0.3
        engine.split_data(test_size=test_size, k_folds=k_folds, random_state=42)

        job_tracker_dict[job_id]["status"] = f"Optimizing ({n_trials} trials)..."
        job_tracker_dict[job_id]["progress"] = 0.35

        best_params = engine.optimize_fusion(
            n_trials=n_trials,
            n_startup_trials=n_startup_trials,
            objective_metric=objective_metric,
            pruner_type=pruner_type if pruner_type != "none" else None,
            sampler_type=sampler_type,
            seed=42,
            show_progress=False,
        )

        job_tracker_dict[job_id]["status"] = "Filtering robust trials..."
        job_tracker_dict[job_id]["progress"] = 0.85
        robust_trials = engine.get_robust_trials(
            method="auto", p_threshold=0.05, tolerance=0.1, min_trials=10
        )

        job_tracker_dict[job_id]["status"] = "Evaluating on test set..."
        job_tracker_dict[job_id]["progress"] = 0.9
        test_results = engine.evaluate_on_test(
            params=best_params, metric=objective_metric
        )

        job_tracker_dict[job_id]["status"] = "Applying fusion weights..."
        job_tracker_dict[job_id]["progress"] = 0.95
        composite_df = engine.apply_fusion()

        job_tracker_dict[job_id]["engine"] = engine
        job_tracker_dict[job_id]["results"] = {
            "best_params": best_params,
            "best_value": engine.study.best_value,
            "robust_trials": robust_trials,
            "composite_df": composite_df,
            "objective_metric": objective_metric,
            "test_results": test_results,
        }
        job_tracker_dict[job_id]["status"] = "Completed"
        job_tracker_dict[job_id]["progress"] = 1.0

    except InterruptedError:
        job_tracker_dict[job_id]["status"] = "Cancelled"
        job_tracker_dict[job_id]["progress"] = 0.0
        print(f"[FUSION] Job {job_id} cancelled by user")

    except Exception as e:
        import traceback

        job_tracker_dict[job_id]["status"] = f"Error: {str(e)}"
        job_tracker_dict[job_id]["error_detail"] = traceback.format_exc()


# ---------------------------------------------------------------------------
# Tab render entry point
# ---------------------------------------------------------------------------


def render(output_dir: str) -> None:
    st.header("Metric Fusion & Optimization")
    st.markdown(
        "Optimize weighted fusion of GVI and NDVI metrics against target outcomes."
    )

    MetricFusionEngine = _MetricFusionEngine

    if MetricFusionEngine is None:
        st.error("MetricFusionEngine module not found in geofuse/fusion.py")
        return

    if "fusion_engine" not in st.session_state:
        st.session_state.fusion_engine = None
    if "fusion_results" not in st.session_state:
        st.session_state.fusion_results = None

    # =========================================================================
    # ROW 1: Configuration (Left) | Preview (Right)
    # =========================================================================
    col_fusion_left, col_fusion_right = st.columns([1, 1])

    tmp_target_path = None
    is_geojson = False
    is_tiff = False
    target_feature = None
    target_band = 1

    with col_fusion_left:
        st.subheader("Target Configuration")

        target_file = st.file_uploader(
            "Upload Target File (GeoJSON or GeoTIFF)",
            type=["geojson", "json", "tif", "tiff"],
            key="fusion_target_upload",
        )

        if target_file:
            is_geojson = target_file.name.lower().endswith((".geojson", ".json"))
            is_tiff = target_file.name.lower().endswith((".tif", ".tiff"))

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(target_file.name)[1]
            ) as tmp:
                tmp.write(target_file.getvalue())
                tmp_target_path = tmp.name

            if is_geojson:
                try:
                    preview_gdf = gpd.read_file(tmp_target_path)
                    numeric_cols = preview_gdf.select_dtypes(
                        include=[np.number]
                    ).columns.tolist()
                    st.info(f"📍 Detected: **GeoJSON** with {len(preview_gdf)} points")
                    target_feature = st.selectbox(
                        "Target Attribute (Outcome Variable)",
                        options=numeric_cols,
                        help="Numeric outcome column.",
                        key="fusion_target_attribute",
                    )
                except Exception as e:
                    st.error(f"Error loading GeoJSON: {e}")
                    tmp_target_path = None

            elif is_tiff:
                try:
                    with rasterio.open(tmp_target_path) as src:
                        n_bands = src.count
                        st.info(f"🗺️ Detected: **GeoTIFF** with {n_bands} band(s)")
                        target_band = st.number_input(
                            "Select Target Band",
                            min_value=1,
                            max_value=n_bands,
                            value=1,
                            help="Band treated as the outcome surface.",
                            key="fusion_target_band",
                        )
                except Exception as e:
                    st.error(f"Error loading GeoTIFF: {e}")
                    tmp_target_path = None

        st.divider()
        st.subheader("Metric Configuration")

        st.markdown("**GVI buffer exploration (m)**")
        col_bgvi_a, col_bgvi_b, col_bgvi_c = st.columns(3)
        with col_bgvi_a:
            gvi_buffer_min_m = st.number_input(
                "GVI minimum buffer",
                min_value=50,
                max_value=4900,
                value=100,
                step=50,
                help="Smallest GVI radius searched (m).",
                key="fusion_gvi_buffer_min",
            )
        with col_bgvi_b:
            gvi_buffer_max_m = st.number_input(
                "GVI maximum buffer",
                min_value=100,
                max_value=5000,
                value=1500,
                step=50,
                help="Largest GVI radius (m); extent padding uses max with NDVI.",
                key="fusion_gvi_buffer_max",
            )
        with col_bgvi_c:
            gvi_buffer_step_m = st.number_input(
                "GVI buffer step",
                min_value=10,
                max_value=500,
                value=50,
                step=10,
                help="Radius discretization (m).",
                key="fusion_gvi_buffer_step",
            )

        st.markdown("**NDVI buffer exploration (m)**")
        col_bndvi_a, col_bndvi_b, col_bndvi_c = st.columns(3)
        with col_bndvi_a:
            ndvi_buffer_min_m = st.number_input(
                "NDVI minimum buffer",
                min_value=50,
                max_value=4900,
                value=100,
                step=50,
                help="Smallest NDVI radius searched (m).",
                key="fusion_ndvi_buffer_min",
            )
        with col_bndvi_b:
            ndvi_buffer_max_m = st.number_input(
                "NDVI maximum buffer",
                min_value=100,
                max_value=5000,
                value=1500,
                step=50,
                help="Largest NDVI radius (m); extent padding uses max with GVI.",
                key="fusion_ndvi_buffer_max",
            )
        with col_bndvi_c:
            ndvi_buffer_step_m = st.number_input(
                "NDVI buffer step",
                min_value=10,
                max_value=500,
                value=50,
                step=10,
                help="Radius discretization (m).",
                key="fusion_ndvi_buffer_step",
            )

        buffer_extent_m = float(max(gvi_buffer_max_m, ndvi_buffer_max_m))

        metric_mode = st.radio(
            "Metric Source",
            options=["Use Loaded Results", "Upload Files", "Auto-Download"],
            horizontal=True,
            help="Use existing outputs, upload rasters/vectors, or fetch metrics at run time.",
            key="fusion_metric_source",
        )

        ndvi_auto_start = date(2023, 6, 1)
        ndvi_auto_end = date(2023, 9, 30)
        cache_metrics = False

        gvi_path = None
        ndvi_path = None
        gvi_api_key_input = ""

        if metric_mode == "Use Loaded Results":
            # Scan output folder for all pre-computed files
            all_gvi_files = _scan_metric_files(output_dir, "gvi")
            all_ndvi_files = _scan_metric_files(output_dir, "ndvi")

            # Compute buffered target extent for spatial filtering (if target loaded)
            buffered_extent = None
            if tmp_target_path:
                buffered_extent = _compute_buffered_extent(
                    tmp_target_path, is_geojson, buffer_extent_m
                )

            def _filter_by_coverage(file_list, bext):
                """Return (covering, non_covering) label lists."""
                if bext is None:
                    return [lbl for lbl, _ in file_list], []
                covering, non_covering = [], []
                for lbl, path in file_list:
                    (covering if _check_coverage(path, bext) else non_covering).append(
                        lbl
                    )
                return covering, non_covering

            gvi_covering, gvi_outside = _filter_by_coverage(
                all_gvi_files, buffered_extent
            )
            ndvi_covering, ndvi_outside = _filter_by_coverage(
                all_ndvi_files, buffered_extent
            )

            col_gvi_sel, col_ndvi_sel = st.columns(2)

            with col_gvi_sel:
                if not all_gvi_files:
                    st.info("No GVI results found in the output folder.")
                else:
                    if buffered_extent is not None:
                        st.caption(
                            f"🌿 GVI: {len(gvi_covering)} cover target"
                            + (f", {len(gvi_outside)} outside" if gvi_outside else "")
                        )
                    options_gvi = (
                        [None]
                        + gvi_covering
                        + (
                            ["── outside target ──"] + gvi_outside
                            if gvi_outside
                            else []
                        )
                    )
                    gvi_selection = st.selectbox(
                        "🌿 Select GVI Result",
                        options=options_gvi,
                        format_func=lambda x: (
                            "(Optional — will auto-download)" if x is None else x
                        ),
                        key="fusion_gvi_select",
                    )
                    if gvi_selection and not gvi_selection.startswith("──"):
                        gvi_path = os.path.join(output_dir, gvi_selection)
                        if not os.path.exists(gvi_path):
                            st.warning("⚠️ File not found on disk.")
                            gvi_path = None
                        elif (
                            buffered_extent is not None and gvi_selection in gvi_outside
                        ):
                            st.warning(
                                "⚠️ This result does not fully cover the buffered "
                                "target area — spatial alignment may be incomplete."
                            )
                        else:
                            st.success(f"✓ {gvi_selection}")

            with col_ndvi_sel:
                if not all_ndvi_files:
                    st.info("No NDVI results found in the output folder.")
                else:
                    if buffered_extent is not None:
                        st.caption(
                            f"🛰️ NDVI: {len(ndvi_covering)} cover target"
                            + (f", {len(ndvi_outside)} outside" if ndvi_outside else "")
                        )
                    options_ndvi = (
                        [None]
                        + ndvi_covering
                        + (
                            ["── outside target ──"] + ndvi_outside
                            if ndvi_outside
                            else []
                        )
                    )
                    ndvi_selection = st.selectbox(
                        "🛰️ Select NDVI Result",
                        options=options_ndvi,
                        format_func=lambda x: (
                            "(Optional — will auto-download)" if x is None else x
                        ),
                        key="fusion_ndvi_select",
                    )
                    if ndvi_selection and not ndvi_selection.startswith("──"):
                        ndvi_path = os.path.join(output_dir, ndvi_selection)
                        if not os.path.exists(ndvi_path):
                            st.warning("⚠️ File not found on disk.")
                            ndvi_path = None
                        elif (
                            buffered_extent is not None
                            and ndvi_selection in ndvi_outside
                        ):
                            st.warning(
                                "⚠️ This result does not fully cover the buffered "
                                "target area — spatial alignment may be incomplete."
                            )
                        else:
                            st.success(f"✓ {ndvi_selection}")

        elif metric_mode == "Upload Files":
            col_gvi_up, col_ndvi_up = st.columns(2)

            with col_gvi_up:
                gvi_file = st.file_uploader(
                    "🌿 Upload GVI File",
                    type=["geojson", "json", "tif", "tiff"],
                    key="fusion_gvi_upload",
                )
                if gvi_file:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(gvi_file.name)[1]
                    ) as tmp:
                        tmp.write(gvi_file.getvalue())
                        gvi_path = tmp.name
                    st.success(f"✓ Uploaded: {gvi_file.name}")

            with col_ndvi_up:
                ndvi_file = st.file_uploader(
                    "🛰️ Upload NDVI File",
                    type=["geojson", "json", "tif", "tiff"],
                    key="fusion_ndvi_upload",
                )
                if ndvi_file:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(ndvi_file.name)[1]
                    ) as tmp:
                        tmp.write(ndvi_file.getvalue())
                        ndvi_path = tmp.name
                    st.success(f"✓ Uploaded: {ndvi_file.name}")

        else:  # Auto-Download
            col_ad1, col_ad2 = st.columns(2)
            with col_ad1:
                ndvi_auto_start = st.date_input(
                    "NDVI Start Date",
                    value=date(2023, 6, 1),
                    help="Composite interval start (auto-download NDVI).",
                    key="fusion_ndvi_start_date",
                )
            with col_ad2:
                ndvi_auto_end = st.date_input(
                    "NDVI End Date",
                    value=date(2023, 9, 30),
                    help="Composite interval end (auto-download NDVI).",
                    key="fusion_ndvi_end_date",
                )
            cache_metrics = st.checkbox(
                "Cache Metrics to Disk",
                value=True,
                help="Persist fetched metrics under the output fusion cache.",
                key="fusion_cache_metrics",
            )
            gvi_api_key_input = st.text_input(
                "Street View API Key (optional)",
                type="password",
                help="Optional Google Street View key; leave blank for built-in access.",
                key="fusion_streetview_api_key",
            )

        if metric_mode != "Auto-Download":
            ndvi_auto_start = date(2023, 6, 1)
            ndvi_auto_end = date(2023, 9, 30)
            cache_metrics = False

        ndvi_resolution_m = None
        gvi_grid_spacing_m = None

        if metric_mode == "Auto-Download":
            st.markdown("**Metric generation settings**")
            ndvi_resolution_m = st.number_input(
                "NDVI satellite resolution (m)",
                min_value=5.0,
                max_value=100.0,
                value=10.0,
                step=5.0,
                help="Target pixel size for NDVI export (Earth Engine).",
                key="fusion_ndvi_satellite_resolution",
            )
            gvi_grid_spacing_m = st.number_input(
                "GVI sampling grid spacing (m)",
                min_value=10.0,
                max_value=500.0,
                value=50.0,
                step=10.0,
                help="Spacing for street-view sample points on the grid.",
                key="fusion_gvi_sampling_grid_spacing",
            )

        st.divider()
        st.subheader("Optimization Settings")

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            objective_metric = st.selectbox(
                "Objective Metric",
                options=["pearson", "spearman", "r2", "rmse", "mutual_info"],
                index=0,
                help="Quantity maximized or minimized across CV folds.",
                key="fusion_objective_metric",
            )
            n_trials = st.number_input(
                "Total Trials",
                min_value=50,
                max_value=1000,
                value=300,
                step=50,
                help="Number of Optuna trials.",
                key="fusion_n_trials",
            )
            optimizer = st.selectbox(
                "Optimizer",
                options=["TPE", "CMA-ES", "Random"],
                index=0,
                help="Hyperparameter search sampler.",
                key="fusion_optimizer",
            )

        with col_opt2:
            n_startup_trials = st.number_input(
                "Random Startup Trials",
                min_value=10,
                max_value=500,
                value=150,
                step=10,
                help="Uniformly random trials before the main sampler.",
                key="fusion_n_startup",
            )
            pruner_type = st.selectbox(
                "Pruner",
                options=["median", "hyperband", "successive_halving", "none"],
                index=0,
                help="Early stopping rule for unpromising trials.",
                key="fusion_pruner",
            )

        col_split1, col_split2, col_split3 = st.columns(3)
        with col_split1:
            test_size = st.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.5,
                value=0.3,
                step=0.05,
                help="Held-out evaluation fraction.",
                key="fusion_test_size",
            )
        with col_split2:
            k_folds = st.number_input(
                "K-Fold CV",
                min_value=3,
                max_value=10,
                value=5,
                help="Cross-validation folds on the non-test subset.",
                key="fusion_k_folds",
            )
        with col_split3:
            n_bins = st.number_input(
                "Stratification Bins",
                min_value=3,
                max_value=10,
                value=5,
                help="Quantile bins for stratified train/test split.",
                key="fusion_stratification_bins",
            )

    with col_fusion_right:
        st.subheader("Target Preview")

        if target_file and tmp_target_path:
            m_fusion_preview = folium.Map(location=[51.0447, -114.0719], zoom_start=10)

            try:
                if is_geojson:
                    preview_gdf = gpd.read_file(tmp_target_path)
                    if preview_gdf.crs is None:
                        preview_gdf.set_crs("EPSG:4326", inplace=True)
                    else:
                        preview_gdf = preview_gdf.to_crs("EPSG:4326")

                    if target_feature and target_feature in preview_gdf.columns:
                        vals = preview_gdf[target_feature].dropna()
                        if len(vals) > 0:
                            vmin, vmax = vals.min(), vals.max()
                            for _, row in preview_gdf.iterrows():
                                if pd.notna(row[target_feature]):
                                    norm_val = (
                                        (row[target_feature] - vmin) / (vmax - vmin)
                                        if vmax > vmin
                                        else 0.5
                                    )
                                    color = (
                                        f"#{int(255*(1-norm_val)):02x}"
                                        f"{int(255*norm_val):02x}00"
                                    )
                                    folium.CircleMarker(
                                        location=[row.geometry.y, row.geometry.x],
                                        radius=5,
                                        color=color,
                                        fill=True,
                                        fill_opacity=0.7,
                                        popup=(
                                            f"{target_feature}: "
                                            f"{row[target_feature]:.3f}"
                                        ),
                                    ).add_to(m_fusion_preview)
                    else:
                        folium.GeoJson(preview_gdf).add_to(m_fusion_preview)

                    bounds = preview_gdf.total_bounds
                    m_fusion_preview.fit_bounds(
                        [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
                    )

                elif is_tiff:
                    with rasterio.open(tmp_target_path) as src:
                        arr = src.read(target_band)
                        bounds_native = src.bounds
                        src_crs = src.crs

                        from rasterio.warp import transform_bounds

                        bounds_4326 = transform_bounds(
                            src_crs, "EPSG:4326", *bounds_native
                        )

                        valid_data = arr[(arr != src.nodata) & ~np.isnan(arr)]
                        if len(valid_data) > 0:
                            vmin, vmax = np.percentile(valid_data, [2, 98])
                            norm_data = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
                            cmap = plt.get_cmap("RdYlGn")
                            colored = cmap(norm_data)
                            mask = (arr == src.nodata) | np.isnan(arr)
                            colored[..., 3] = np.where(mask, 0, 0.7)

                            img_bytes = (colored * 255).astype(np.uint8)
                            im = PILImage.fromarray(img_bytes)
                            buff = io.BytesIO()
                            im.save(buff, format="PNG")
                            img_url = (
                                f"data:image/png;base64,"
                                f"{base64.b64encode(buff.getvalue()).decode()}"
                            )

                            folium.raster_layers.ImageOverlay(
                                image=img_url,
                                bounds=[
                                    [bounds_4326[1], bounds_4326[0]],
                                    [bounds_4326[3], bounds_4326[2]],
                                ],
                                opacity=0.7,
                            ).add_to(m_fusion_preview)

                            m_fusion_preview.fit_bounds(
                                [
                                    [bounds_4326[1], bounds_4326[0]],
                                    [bounds_4326[3], bounds_4326[2]],
                                ]
                            )

                st_folium(
                    m_fusion_preview,
                    width="100%",
                    height=400,
                    key="fusion_preview_map",
                )

            except Exception as e:
                st.error(f"Preview error: {e}")
        else:
            st.info("Upload a target file to preview")

    # =========================================================================
    # ROW 2: Run Button & Progress
    # =========================================================================
    st.divider()

    col_run1, col_run2, col_run3 = st.columns([2, 1, 1])
    with col_run1:
        run_fusion = st.button(
            "🚀 Run Fusion Optimization", type="primary", use_container_width=True
        )
    with col_run2:
        if st.session_state.fusion_results:
            if st.button(
                "📊 Export Results", use_container_width=True, key="fusion_export"
            ):
                result_df = st.session_state.fusion_results["composite_df"]
                export_gdf = gpd.GeoDataFrame(
                    result_df,
                    geometry=gpd.points_from_xy(
                        result_df.index % 100, result_df.index // 100
                    ),
                    crs="EPSG:4326",
                )
                export_path = os.path.join(
                    output_dir,
                    f"fusion_composite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson",
                )
                export_gdf.to_file(export_path, driver="GeoJSON")
                st.success(f"Exported to: {export_path}")
    with col_run3:
        if st.session_state.fusion_engine:
            if st.button("🔄 Reset", use_container_width=True, key="fusion_reset"):
                st.session_state.fusion_engine = None
                st.session_state.fusion_results = None
                st.rerun()

    if run_fusion:
        if not target_file:
            st.error("❌ Please upload a target file")
        elif is_geojson and not target_feature:
            st.error("❌ Please select a target attribute for the GeoJSON target")
        elif gvi_buffer_min_m > gvi_buffer_max_m or ndvi_buffer_min_m > ndvi_buffer_max_m:
            st.error(
                "❌ Each modality's minimum buffer must be less than or equal to its maximum buffer."
            )
        else:
            if metric_mode == "Use Loaded Results" and not gvi_path and not ndvi_path:
                st.error(
                    "❌ No metrics selected. Please select GVI/NDVI results or "
                    "switch to Auto-Download mode."
                )
            else:
                with st.expander("Configuration Summary", expanded=True):
                    st.write(f"**Target:** {target_file.name}")
                    if is_geojson:
                        st.write(f"**Feature:** {target_feature}")
                    else:
                        st.write(f"**Band:** {target_band}")
                    st.write(
                        f"**GVI buffers (m):** {gvi_buffer_min_m} – {gvi_buffer_max_m} "
                        f"(step {gvi_buffer_step_m})"
                    )
                    st.write(
                        f"**NDVI buffers (m):** {ndvi_buffer_min_m} – {ndvi_buffer_max_m} "
                        f"(step {ndvi_buffer_step_m})"
                    )
                    st.write(f"**Extent padding (m):** {buffer_extent_m}")
                    if ndvi_resolution_m is not None:
                        st.write(f"**NDVI resolution (m):** {ndvi_resolution_m}")
                    if gvi_grid_spacing_m is not None:
                        st.write(f"**GVI grid spacing (m):** {gvi_grid_spacing_m}")
                    st.write(
                        f"**GVI Source:** "
                        f"{'✓ ' + os.path.basename(gvi_path) if gvi_path else '📥 Auto-download'}"
                    )
                    st.write(
                        f"**NDVI Source:** "
                        f"{'✓ ' + os.path.basename(ndvi_path) if ndvi_path else '📥 Auto-download'}"
                    )
                    st.write(
                        f"**Optimization:** {n_trials} trials, {k_folds}-fold CV, "
                        f"{test_size*100:.0f}% test set"
                    )

                job_id = f"fusion_{uuid.uuid4().hex[:8]}"

                if "jobs" not in st.session_state:
                    st.session_state.jobs = {}

                st.session_state.jobs[job_id] = {
                    "name": f"Fusion: {target_file.name}",
                    "status": "Starting...",
                    "progress": 0.0,
                    "cancel": False,
                    "type": "fusion",
                }

                gvi_api_key = (
                    (gvi_api_key_input or None)
                    if metric_mode == "Auto-Download"
                    else None
                )
                ndvi_project_id = None

                thread = threading.Thread(
                    target=_fusion_worker,
                    args=(
                        job_id,
                        tmp_target_path,
                        target_feature if is_geojson else None,
                        target_band if is_tiff else 1,
                        buffer_extent_m,
                        gvi_buffer_min_m,
                        gvi_buffer_max_m,
                        gvi_buffer_step_m,
                        ndvi_buffer_min_m,
                        ndvi_buffer_max_m,
                        ndvi_buffer_step_m,
                        ndvi_resolution_m,
                        gvi_grid_spacing_m,
                        n_bins,
                        gvi_path,
                        None,  # terrain_path — separate from veg in load_metrics
                        ndvi_path,
                        cache_metrics,
                        test_size,
                        k_folds,
                        n_trials,
                        n_startup_trials,
                        objective_metric,
                        pruner_type,
                        optimizer,
                        gvi_api_key,
                        ndvi_auto_start.isoformat(),
                        ndvi_auto_end.isoformat(),
                        ndvi_project_id,
                        output_dir,
                        st.session_state.jobs,
                        MetricFusionEngine,
                    ),
                    daemon=True,
                )
                add_script_run_ctx(thread)
                thread.start()

                st.success("✅ Fusion job started! Check sidebar for progress.")

    # Check for completed fusion jobs and load results
    if "jobs" in st.session_state:
        for job_id, job_data in st.session_state.jobs.items():
            if (
                job_data.get("type") == "fusion"
                and job_data.get("status") == "Completed"
                and "results" in job_data
                and st.session_state.fusion_results is None
            ):
                st.session_state.fusion_engine = job_data.get("engine")
                st.session_state.fusion_results = job_data["results"]

    # =========================================================================
    # ROW 3: Results Display
    # =========================================================================
    if st.session_state.fusion_results:
        st.divider()
        st.subheader("Optimization Results")

        results = st.session_state.fusion_results
        engine = st.session_state.fusion_engine

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

        total_weight = (
            results["best_params"]["veg_weight"]
            + results["best_params"]["terrain_weight"]
            + results["best_params"]["ndvi_weight"]
        )

        with col_m1:
            veg_pct = (
                (results["best_params"]["veg_weight"] / total_weight * 100)
                if total_weight > 0
                else 0
            )
            st.metric("Vegetation Weight", f"{veg_pct:.1f}%")

        with col_m2:
            terrain_pct = (
                (results["best_params"]["terrain_weight"] / total_weight * 100)
                if total_weight > 0
                else 0
            )
            st.metric("Terrain Weight", f"{terrain_pct:.1f}%")

        with col_m3:
            ndvi_pct = (
                (results["best_params"]["ndvi_weight"] / total_weight * 100)
                if total_weight > 0
                else 0
            )
            st.metric("NDVI Weight", f"{ndvi_pct:.1f}%")

        with col_m4:
            metric_name = results["objective_metric"].upper()
            st.metric(f"Best {metric_name}", f"{results['best_value']:.4f}")

        with col_m5:
            if results["robust_trials"]:
                st.metric(
                    "Robust Trials",
                    f"{len(results['robust_trials'])}/{len(engine.study.trials)}",
                )
            else:
                st.metric("Total Trials", len(engine.study.trials))

        col_detail1, col_detail2 = st.columns(2)

        with col_detail1:
            st.markdown("**Best Trial Details**")
            best_trial = engine.study.best_trial

            info_data = {
                "Trial Number": best_trial.number,
                "Buffer Distance": f"{engine.buffer_meters}m",
                "Veg Weight (raw)": results["best_params"]["veg_weight"],
                "Terrain Weight (raw)": results["best_params"]["terrain_weight"],
                "NDVI Weight (raw)": results["best_params"]["ndvi_weight"],
                "Veg Radius": f"{results['best_params'].get('veg_radius', 'N/A')}m",
                "Terrain Radius": (
                    f"{results['best_params'].get('terrain_radius', 'N/A')}m"
                ),
                "NDVI Radius": f"{results['best_params'].get('ndvi_radius', 'N/A')}m",
                f"Train {metric_name}": (
                    f"{best_trial.user_attrs.get('train_score_mean', 'N/A')}"
                ),
                f"Val {metric_name}": (
                    f"{best_trial.user_attrs.get('val_score_mean', 'N/A')}"
                ),
            }

            if "train_pvalue" in best_trial.user_attrs:
                info_data["Train p-value"] = (
                    f"{best_trial.user_attrs['train_pvalue']:.4e}"
                )
            if "val_pvalue" in best_trial.user_attrs:
                info_data["Val p-value"] = f"{best_trial.user_attrs['val_pvalue']:.4e}"

            st.json(info_data)

        with col_detail2:
            st.markdown("**Optimization History**")

            trial_values = [t.value for t in engine.study.trials if t.value is not None]
            trial_numbers = [
                t.number for t in engine.study.trials if t.value is not None
            ]

            if trial_values:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(trial_numbers, trial_values, alpha=0.6, linewidth=0.5)

                running_best = []
                current_best = (
                    -np.inf if engine.study.direction.name == "MAXIMIZE" else np.inf
                )
                for val in trial_values:
                    if engine.study.direction.name == "MAXIMIZE":
                        current_best = max(current_best, val)
                    else:
                        current_best = min(current_best, val)
                    running_best.append(current_best)

                ax.plot(
                    trial_numbers,
                    running_best,
                    color="red",
                    linewidth=2,
                    label="Best",
                )
                ax.set_xlabel("Trial")
                ax.set_ylabel(f"{metric_name}")
                ax.set_title("Optimization Progress")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

        if results["robust_trials"]:
            st.divider()
            st.markdown("**Robust Trials (Statistically Significant)**")

            robust_data = []
            for t in results["robust_trials"][:10]:
                robust_data.append(
                    {
                        "Trial": t.number,
                        "GVI %": (
                            f"{(t.params['gvi_weight'] / (t.params['gvi_weight'] + t.params['ndvi_weight']) * 100):.1f}"
                        ),
                        "NDVI %": (
                            f"{(t.params['ndvi_weight'] / (t.params['gvi_weight'] + t.params['ndvi_weight']) * 100):.1f}"
                        ),
                        f"Train {metric_name}": (
                            f"{t.user_attrs.get('train_score', 0):.4f}"
                        ),
                        f"Test {metric_name}": (
                            f"{t.user_attrs.get('test_score', 0):.4f}"
                        ),
                        "Train p": f"{t.user_attrs.get('train_pvalue', 1):.4e}",
                        "Test p": f"{t.user_attrs.get('test_pvalue', 1):.4e}",
                    }
                )

            st.dataframe(robust_data, use_container_width=True)
