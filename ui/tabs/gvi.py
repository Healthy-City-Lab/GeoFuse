import base64
import glob
import io
import os
import threading
import uuid
from datetime import datetime

import folium
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
from helpers import apply_buffer_m, generate_raster_grid, load_clean_gdf
from PIL import Image as PILImage
from rasterio.transform import array_bounds
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_folium import st_folium

from geofuse.gvi import GVIEngine
from geofuse.vision import get_best_device

# ---------------------------------------------------------------------------
# Background worker (module-level)
# ---------------------------------------------------------------------------


def _job_worker(
    job_id, fname, dataset_data, init_args, run_args, output_dir, job_tracker_dict
):
    gpu_lock = _get_gpu_lock()
    try:
        job_tracker_dict[job_id]["status"] = "Waiting for GPU..."
        with gpu_lock:
            if job_tracker_dict[job_id]["cancel"]:
                job_tracker_dict[job_id]["status"] = "Cancelled"
                return

            job_tracker_dict[job_id]["status"] = "Initializing..."
            engine = _get_gvi_engine(init_args["model_path"], init_args.get("api_key"))

            current_accumulated = dataset_data["accumulated"]
            start_idx = len(current_accumulated)

            def on_progress(curr, total):
                job_tracker_dict[job_id]["progress"] = min(curr / total, 1.0)
                job_tracker_dict[job_id]["status"] = f"Processing ({curr}/{total})"

            def on_result(res):
                dataset_data["accumulated"].append(res)

            def check_cancel():
                return job_tracker_dict[job_id]["cancel"]

            job_tracker_dict[job_id]["status"] = "Running"

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

            if job_tracker_dict[job_id]["cancel"]:
                job_tracker_dict[job_id]["status"] = "Cancelled"
            else:
                job_tracker_dict[job_id]["status"] = "Completed"
                job_tracker_dict[job_id]["progress"] = 1.0

                res_df = gpd.GeoDataFrame(
                    dataset_data["accumulated"], crs=dataset_data["processed"].crs
                )
                if "orig_index" in res_df.columns:
                    res_df.set_index("orig_index", inplace=True)
                    res_df.index.name = None
                dataset_data["results"] = res_df

                out_name = os.path.splitext(fname)[0]
                res_df.to_file(
                    os.path.join(output_dir, f"{out_name}_gvi.geojson"),
                    driver="GeoJSON",
                )

                if dataset_data["meta"]:
                    from rasterio.transform import rowcol

                    meta = dataset_data["meta"]
                    arr_veg = np.full(
                        (meta["height"], meta["width"]), np.nan, dtype=np.float32
                    )
                    arr_ter = np.full(
                        (meta["height"], meta["width"]), np.nan, dtype=np.float32
                    )
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

    except Exception as e:
        job_tracker_dict[job_id]["status"] = f"Error: {str(e)}"
        print(f"Job Failed: {e}")


# ---------------------------------------------------------------------------
# Cached resources (defined at module level so cache keys are stable)
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_gpu_lock():
    return threading.Lock()


@st.cache_resource
def _get_gvi_engine(model_path, api_key):
    best_device = get_best_device()
    st.info(f"🚀 Using device: {best_device}")
    return GVIEngine(model_path=model_path, device=str(best_device), api_key=api_key)


# ---------------------------------------------------------------------------
# Tab render entry point
# ---------------------------------------------------------------------------


def render(output_dir: str, parent_dir: str) -> None:
    st.header("GVI Sourcing")

    # --- SESSION STATE ---
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}
    if "master_cache" not in st.session_state:
        st.session_state.master_cache = {}
    if "jobs" not in st.session_state:
        st.session_state.jobs = {}
    if "inspector_select_key" not in st.session_state:
        st.session_state.inspector_select_key = None

    # --- SIDEBAR JOB MONITOR ---
    def callback_dismiss_job(jid):
        if jid in st.session_state.jobs:
            del st.session_state.jobs[jid]

    def callback_cancel_job(jid):
        if jid in st.session_state.jobs:
            st.session_state.jobs[jid]["cancel"] = True

    @st.fragment(run_every=1)
    def show_job_monitor_fragment():
        st.header("Job Monitor")

        job_keys = list(st.session_state.jobs.keys())
        if not job_keys:
            st.info("No active jobs.")
        else:
            for jid in job_keys:
                job = st.session_state.jobs[jid]
                with st.container(border=True):
                    job_type = job.get("type", "gvi")
                    job_name = job.get("name", job.get("fname", "Unknown"))

                    c1, c2 = st.columns([7, 3])
                    c1.markdown(f"**{job_name}**")

                    if job_type == "fusion":
                        c2.caption("🔀 Fusion | Optimizing")
                    elif job_type == "ndvi":
                        c2.caption(f"🛰️ NDVI | {job.get('start_time', '')}")
                    else:
                        c2.caption(
                            f"{job.get('task', 'Job')} | {job.get('start_time', '')}"
                        )

                    st.progress(job["progress"])
                    st.caption(f"Status: {job['status']}")

                    if "gvi_progress" in job and job["gvi_progress"]:
                        gvi = job["gvi_progress"]
                        st.progress(
                            gvi["percent"] / 100,
                            text=f"{gvi['current']:,} / {gvi['total']:,}",
                        )

                    if "error_detail" in job:
                        with st.expander("Error Details"):
                            st.code(job["error_detail"])

                    is_running = (
                        job["status"]
                        in [
                            "Queued",
                            "Initializing...",
                            "Initializing fusion engine...",
                            "Loading target data...",
                            "Loading metrics (may auto-download)...",
                            "Downloading GVI Vegetation data...",
                            "Downloading GVI Terrain data...",
                            "Downloading NDVI satellite data...",
                            "Loading provided metric files...",
                            "Splitting data...",
                            "Waiting for GPU...",
                            "Running",
                        ]
                        or "Processing" in job["status"]
                        or "Optimizing" in job["status"]
                        or "Downloading" in job["status"]
                    )

                    if is_running:
                        st.button(
                            "Cancel",
                            key=f"cancel_{jid}",
                            on_click=callback_cancel_job,
                            args=(jid,),
                        )
                    else:
                        st.button(
                            "🗑️",
                            key=f"del_{jid}",
                            on_click=callback_dismiss_job,
                            args=(jid,),
                        )

    with st.sidebar:
        show_job_monitor_fragment()

    # =========================================================================
    # ROW 1: INPUTS & SUBMISSION (Left) | INPUT PREVIEW MAP (Right)
    # =========================================================================
    col_top_left, col_top_right = st.columns(2)

    with col_top_left:
        st.subheader("Input Configuration")
        mode = st.radio(
            "Download Mode",
            ["Package (Scraper)", "API (Street View)"],
            horizontal=True,
            key="gvi_download_mode",
        )
        api_key = (
            st.text_input(
                "Street View API Key",
                type="password",
                autocomplete="off",
                help="Optional Google Street View key; Will fall back to built-in access if not provided.",
                key="gvi_google_api_key",
            )
            if mode == "API (Street View)"
            else None
        )
        gvi_res = st.slider("Grid Resolution (m)", 20, 500, 50, key="gvi_res")
        gvi_buffer = st.slider(
            "Download Buffer (m)",
            min_value=0,
            max_value=2000,
            value=0,
            step=50,
            key="gvi_buffer",
            help="Expand the study area boundary outward by this many metres before generating the sampling grid.",
        )
        save_debug = st.checkbox(
            "Save Raw Images & Masks", value=False, key="gvi_save_debug"
        )

        uploaded_files = st.file_uploader(
            "Upload Study Areas", accept_multiple_files=True, key="gvi_up"
        )

        if uploaded_files is not None:
            current_names = [f.name for f in uploaded_files]
            for k in list(st.session_state.datasets.keys()):
                ds = st.session_state.datasets[k]
                if ds.get("type") == "restored":
                    continue
                if k not in current_names:
                    del st.session_state.datasets[k]
            for f in uploaded_files:
                if f.name not in st.session_state.datasets:
                    try:
                        raw = load_clean_gdf(f)
                        gtype = (
                            "poly"
                            if raw.geometry.iloc[0].geom_type
                            in ["Polygon", "MultiPolygon"]
                            else "point"
                        )
                        st.session_state.datasets[f.name] = {
                            "raw": raw,
                            "processed": None,
                            "accumulated": [],
                            "results": None,
                            "meta": None,
                            "type": gtype,
                        }
                    except Exception as e:
                        st.error(f"Error: {e}")

        if uploaded_files == []:
            for k in list(st.session_state.datasets.keys()):
                if st.session_state.datasets[k].get("type") != "restored":
                    del st.session_state.datasets[k]

        if st.session_state.datasets:
            if st.button("Generate Sampling Grids", key="gvi_gen_grids"):
                with st.spinner("Processing..."):
                    for d in st.session_state.datasets.values():
                        if d.get("type") == "restored":
                            continue
                        if d["type"] == "poly":
                            pts, meta = generate_raster_grid(
                                apply_buffer_m(d["raw"], gvi_buffer), gvi_res
                            )
                            d["processed"] = pts
                            d["meta"] = meta
                        else:
                            d["processed"] = d["raw"].copy()
                            d["meta"] = None
                        d["accumulated"] = []
                        d["results"] = None
                    st.success("Grids generated!")

        st.divider()
        if st.button("🚀 Run GVI Analysis", type="primary", key="gvi_run"):
            if not st.session_state.datasets:
                st.warning("Upload at least one study area to get started.")
            else:
                model_path = os.path.join(
                    parent_dir, "geofuse", "model", "best_model.pth"
                )
                started = False
                for fname, d in st.session_state.datasets.items():
                    if d.get("type") == "restored":
                        continue

                    if d.get("processed") is None:
                        if d["type"] == "poly":
                            pts, meta = generate_raster_grid(
                                apply_buffer_m(d["raw"], gvi_buffer), gvi_res
                            )
                            d["processed"] = pts
                            d["meta"] = meta
                        else:
                            d["processed"] = d["raw"].copy()
                            d["meta"] = None
                        d["accumulated"] = []
                        d["results"] = None

                    existing = [
                        j
                        for j, v in st.session_state.jobs.items()
                        if v["fname"] == fname
                        and v["status"]
                        in ["Running", "Waiting for GPU...", "Initializing..."]
                    ]
                    if existing:
                        continue

                    job_id = str(uuid.uuid4())[:8]
                    st.session_state.jobs[job_id] = {
                        "fname": fname,
                        "task": "GVI",
                        "start_time": datetime.now().strftime("%H:%M:%S"),
                        "progress": 0.0,
                        "status": "Queued",
                        "cancel": False,
                        "handoff_complete": False,
                    }

                    init_args = {"model_path": model_path, "api_key": api_key}
                    run_args = {
                        "step": gvi_res,
                        "save_panos": save_debug,
                        "save_masks": save_debug,
                    }
                    d["cache_ref"] = st.session_state.master_cache

                    t = threading.Thread(
                        target=_job_worker,
                        args=(
                            job_id,
                            fname,
                            d,
                            init_args,
                            run_args,
                            output_dir,
                            st.session_state.jobs,
                        ),
                    )
                    add_script_run_ctx(t)
                    t.start()
                    started = True

                if started:
                    st.success("Analysis started. Monitor progress in the sidebar.")
                else:
                    st.info("All study areas are already running or completed.")

    with col_top_right:
        st.subheader("Study Area Preview")
        m_input = folium.Map(location=[51.0447, -114.0719], zoom_start=11)

        all_bounds = []
        for fname, d in st.session_state.datasets.items():
            if d.get("type") == "restored":
                continue
            if d.get("raw") is not None:
                folium.GeoJson(
                    d["raw"],
                    name=f"{fname} (study area)",
                    style_function=lambda x: {
                        "color": "#1a73e8",
                        "weight": 2,
                        "fill": False,
                    },
                ).add_to(m_input)
                all_bounds.append(d["raw"].total_bounds)
                if gvi_buffer > 0:
                    buf_gdf = apply_buffer_m(d["raw"], gvi_buffer)
                    folium.GeoJson(
                        buf_gdf,
                        name=f"{fname} (buffer)",
                        style_function=lambda x: {
                            "color": "#f4910c",
                            "weight": 2,
                            "fillOpacity": 0.07,
                            "dashArray": "6 4",
                        },
                    ).add_to(m_input)
                    all_bounds.append(buf_gdf.total_bounds)
            if d.get("processed") is not None and not d["processed"].empty:
                preview = d["processed"].iloc[:1000]
                if preview.geometry.iloc[0].geom_type == "Point":
                    for _, row in preview.iterrows():
                        folium.CircleMarker(
                            [row.geometry.y, row.geometry.x],
                            radius=1,
                            color="red",
                            fill=True,
                            fill_opacity=0.6,
                        ).add_to(m_input)
                else:
                    folium.GeoJson(
                        preview,
                        style_function=lambda x: {"color": "red", "weight": 1},
                    ).add_to(m_input)

        if all_bounds:
            min_x = min([b[0] for b in all_bounds])
            min_y = min([b[1] for b in all_bounds])
            max_x = max([b[2] for b in all_bounds])
            max_y = max([b[3] for b in all_bounds])
            m_input.fit_bounds([[min_y, min_x], [max_y, max_x]])

        st_folium(
            m_input, width="100%", height=500, key="map_input", returned_objects=[]
        )

    st.divider()

    # =========================================================================
    # ROW 2: RESULT INSPECTOR (Left) | RESULT PREVIEW MAP (Right)
    # =========================================================================
    col_btm_left, col_btm_right = st.columns(2)

    with col_btm_left:
        st.subheader("Result Inspector")

        if st.button("🔄 Scan Output Folder", key="gvi_scan_folder"):
            found_files = glob.glob(os.path.join(output_dir, "*_gvi.geojson"))
            count = 0
            for p in found_files:
                base_name = os.path.basename(p).replace("_gvi.geojson", "")
                tif_path = os.path.join(output_dir, f"{base_name}_gvi.tif")
                if not os.path.exists(tif_path):
                    continue

                if base_name not in st.session_state.datasets:
                    try:
                        gdf = gpd.read_file(p)
                        if gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326":
                            gdf = gdf.to_crs("EPSG:4326")
                        with rasterio.open(tif_path) as src:
                            meta = {
                                "transform": src.transform,
                                "width": src.width,
                                "height": src.height,
                                "crs": src.crs,
                            }
                        raw_geom = gdf.geometry.union_all().envelope
                        raw_gdf = gpd.GeoDataFrame(
                            {"geometry": [raw_geom]}, crs=gdf.crs
                        )
                        st.session_state.datasets[base_name] = {
                            "raw": raw_gdf,
                            "processed": None,
                            "accumulated": [],
                            "results": gdf,
                            "meta": meta,
                            "type": "restored",
                        }
                        count += 1
                    except Exception as e:
                        print(f"Error: {e}")
            if count > 0:
                st.success(f"Loaded {count} result(s) from the output folder.")
            else:
                st.info("No results found in the output folder.")

        completed_ds = [
            k
            for k, v in st.session_state.datasets.items()
            if v.get("type") == "restored"
        ]

        options = ["All Regions"] + completed_ds

        selected_option = st.selectbox(
            "Select Result",
            options,
            index=None,
            placeholder="Choose a result...",
            key="inspector_select_key",
        )

        raster_layer = st.radio(
            "Background Raster",
            ["Vegetation", "Terrain"],
            horizontal=True,
            key="gvi_raster_layer",
        )
        r_opacity = st.slider("Layer Opacity", 0.0, 1.0, 0.7, key="gvi_layer_opacity")
        show_points = st.checkbox(
            "Show Sample Points", value=False, key="gvi_show_points"
        )

    with col_btm_right:
        st.subheader("Results Preview")
        m_result = folium.Map(location=[51.0447, -114.0719], zoom_start=11)

        if selected_option:
            if selected_option == "All Regions":
                targets = completed_ds
            else:
                targets = [selected_option]

            res_bounds = []

            for ds_name in targets:
                if ds_name not in st.session_state.datasets:
                    continue
                ds = st.session_state.datasets[ds_name]

                if ds.get("meta"):
                    meta = ds["meta"]
                    left, bottom, right, top = array_bounds(
                        meta["height"], meta["width"], meta["transform"]
                    )

                    from rasterio.warp import transform_bounds

                    meta_crs = meta.get("crs", "EPSG:4326")
                    if meta_crs != "EPSG:4326":
                        left, bottom, right, top = transform_bounds(
                            meta_crs, "EPSG:4326", left, bottom, right, top
                        )

                    folium.Rectangle(
                        bounds=[[bottom, left], [top, right]],
                        color="grey",
                        weight=1,
                        fill=False,
                        popup=f"{ds_name} Extent",
                    ).add_to(m_result)
                    res_bounds.append([left, bottom, right, top])

                    from rasterio.transform import rowcol

                    meta = ds["meta"]
                    arr = np.full((meta["height"], meta["width"]), np.nan)

                    col_name = "gvi_ter" if "Terrain" in raster_layer else "gvi_veg"
                    res = ds["results"].dropna(subset=[col_name])

                    if not res.empty:
                        rows, cols = rowcol(
                            meta["transform"],
                            res.geometry.x.values,
                            res.geometry.y.values,
                        )
                        mask_idx = (
                            (rows >= 0)
                            & (rows < meta["height"])
                            & (cols >= 0)
                            & (cols < meta["width"])
                        )
                        arr[rows[mask_idx], cols[mask_idx]] = res[col_name].values[
                            mask_idx
                        ]

                        cmap_name = "OrRd" if "Terrain" in raster_layer else "Greens"
                        try:
                            cmap = matplotlib.colormaps[cmap_name]
                        except (AttributeError, KeyError):
                            cmap = plt.get_cmap(cmap_name)

                        norm_data = np.clip((arr - 0) / 0.6, 0, 1)
                        colored = cmap(norm_data)
                        colored[..., 3] = np.where(np.isnan(arr), 0, r_opacity)
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
                            bounds=[[bottom, left], [top, right]],
                            opacity=r_opacity,
                            interactive=False,
                        ).add_to(m_result)

                if show_points:
                    gdf_viz = ds["results"].copy()
                    if "gvi_veg" in gdf_viz.columns:
                        gdf_viz["gvi_veg"] = gdf_viz["gvi_veg"].round(4)
                    if "gvi_ter" in gdf_viz.columns:
                        gdf_viz["gvi_ter"] = gdf_viz["gvi_ter"].round(4)

                    valid_pts = gdf_viz.dropna(subset=["gvi_veg"])
                    folium.GeoJson(
                        valid_pts,
                        marker=folium.Circle(
                            radius=20,
                            fill_color="green",
                            fill_opacity=0.8,
                            color=None,
                        ),
                        tooltip=folium.GeoJsonTooltip(
                            fields=["gvi_veg", "gvi_ter"],
                            aliases=["Veg Index:", "Ter Index:"],
                        ),
                    ).add_to(m_result)

            if res_bounds:
                min_x = min([b[0] for b in res_bounds])
                min_y = min([b[1] for b in res_bounds])
                max_x = max([b[2] for b in res_bounds])
                max_y = max([b[3] for b in res_bounds])
                m_result.fit_bounds([[min_y, min_x], [max_y, max_x]])

        st_folium(
            m_result, width="100%", height=500, key="map_result", returned_objects=[]
        )
