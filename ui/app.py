import base64
import glob
import io
import json
import os
import sys
import threading
import time
import uuid
from datetime import date, datetime, timedelta

import folium
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from PIL import Image as PILImage
from rasterio.transform import array_bounds
from shapely.geometry import Point

# --- STREAMLIT RUNTIME CONTEXT ---
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_folium import st_folium

from geofuse.gvi import GVIEngine
from geofuse.ndvi import NDVIEngine
from geofuse.vision import get_best_device

# --- 1. GLOBAL PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- 2. GDAL ENVIRONMENT FIX ---
if "GDAL_DATA" not in os.environ:
    conda_prefix = sys.prefix
    gdal_data_path = os.path.join(conda_prefix, "Library", "share", "gdal")
    if os.path.exists(gdal_data_path):
        os.environ["GDAL_DATA"] = gdal_data_path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from geofuse.fusion import FusionOptimizer
except ImportError:
    FusionOptimizer = None

st.set_page_config(page_title="GeoFuse Toolbox", layout="wide")

# --- CSS: Remove padding for tighter layout ---
st.markdown(
    """
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    iframe { width: 100% !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("GeoFuse: Multimodal Environmental Profiling")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Job Monitor", "NDVI Sourcing", "GVI Sourcing", "Fusion & Optimization"]
)

output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# -----------------------------------------------------------------------------
# TAB 1: HPC JOB MONITOR
# -----------------------------------------------------------------------------
with tab1:
    st.header("HPC Job Monitor")
    col1, col2 = st.columns([1, 3])
    with col1:
        job_id = st.text_input("Enter Job ID (e.g., 42591):")
        auto_refresh = st.checkbox("Auto-refresh (2s)", value=True)
    with col2:
        if job_id:
            status_file = os.path.join("logs", f"{job_id}_status.json")
            if os.path.exists(status_file):
                placeholder = st.empty()
                while True:
                    try:
                        with open(status_file, "r") as f:
                            data = json.load(f)
                    except:
                        time.sleep(0.5)
                        continue
                    with placeholder.container():
                        st.subheader(f"Stage: {data.get('stage', 'Unknown')}")
                        st.progress(data.get("progress", 0))
                        if data.get("metrics"):
                            st.json(data.get("metrics"))
                    if not auto_refresh or data.get("progress", 0) >= 100:
                        break
                    time.sleep(2)
            else:
                st.info("Waiting for job...")


# --- SHARED HELPERS ---
@st.cache_data
def load_clean_gdf(file_obj):
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as tmp:
        tmp.write(file_obj.getvalue())
        tmp_path = tmp.name
    try:
        gdf = gpd.read_file(tmp_path)
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        else:
            gdf = gdf.to_crs("EPSG:4326")
        for col in gdf.columns:
            if (
                pd.api.types.is_datetime64_any_dtype(gdf[col])
                or gdf[col].dtype == "object"
            ):
                try:
                    gdf[col] = gdf[col].astype(str)
                except:
                    gdf = gdf.drop(columns=[col])
        return gdf
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def generate_raster_grid(gdf_4326, spacing_meters):
    from rasterio.transform import from_bounds, xy

    minx, miny, maxx, maxy = gdf_4326.total_bounds
    center_lat = (miny + maxy) / 2.0
    lat_rad = np.radians(center_lat)
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
    res_x = spacing_meters / m_per_deg_lon
    res_y = spacing_meters / m_per_deg_lat
    width = int(np.ceil((maxx - minx) / res_x))
    height = int(np.ceil((maxy - miny) / res_y))
    transform = from_bounds(
        minx, miny, minx + (width * res_x), miny + (height * res_y), width, height
    )
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    xs, ys = xy(transform, rows.flatten(), cols.flatten(), offset="center")
    df = pd.DataFrame({"row": rows.flatten(), "col": cols.flatten(), "x": xs, "y": ys})
    gdf_grid = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:4326"
    )
    gdf_clipped = gpd.sjoin(gdf_grid, gdf_4326, how="inner", predicate="intersects")
    return gdf_clipped, {
        "transform": transform,
        "width": width,
        "height": height,
        "crs": "EPSG:4326",
    }


# -----------------------------------------------------------------------------
# TAB 2: NDVI CONFIGURATOR
# -----------------------------------------------------------------------------
with tab2:
    st.header("Satellite Data Configuration (NDVI)")

    if "ndvi_datasets" not in st.session_state:
        st.session_state.ndvi_datasets = {}
    if "ndvi_inspector_select" not in st.session_state:
        st.session_state.ndvi_inspector_select = None

    def ndvi_worker(
        job_id,
        fname,
        dataset_data,
        start_date,
        end_date,
        cloud_pct,
        resolution,
        job_tracker_dict,
    ):
        try:
            job_tracker_dict[job_id]["status"] = "Initializing GEE..."
            engine = NDVIEngine()
            job_tracker_dict[job_id]["status"] = "Downloading & Processing..."
            result = engine.download_and_process(
                geometry=dataset_data["raw"],
                start_date=start_date,
                end_date=end_date,
                output_name=fname.replace(".geojson", ""),
                cloud_max=cloud_pct,
                resolution=resolution,
                folder=output_dir,
            )
            if result["status"] == "success":
                job_tracker_dict[job_id]["status"] = "Completed"
                job_tracker_dict[job_id]["progress"] = 1.0
            else:
                job_tracker_dict[job_id]["status"] = f"Error: {result['message']}"
        except Exception as e:
            job_tracker_dict[job_id]["status"] = f"Error: {str(e)}"

    col_ndvi_top_left, col_ndvi_top_right = st.columns(2)
    with col_ndvi_top_left:
        st.subheader("1. Job Configuration")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input("Start Date", value=date(2023, 6, 1))
        with col_d2:
            end_date = st.date_input("End Date", value=date(2023, 9, 30))
        cloud_pct = st.slider("Max Cloud Coverage (%)", 0, 100, 10)
        resolution = st.number_input("Resolution (m)", value=10, min_value=10)
        ndvi_files = st.file_uploader(
            "Upload Catchment Area (GeoJSON)", accept_multiple_files=True, key="ndvi_up"
        )

        if ndvi_files is not None:
            current_names = [f.name for f in ndvi_files]
            for k in list(st.session_state.ndvi_datasets.keys()):
                ds = st.session_state.ndvi_datasets[k]
                if ds.get("type") == "restored":
                    continue
                if k not in current_names:
                    del st.session_state.ndvi_datasets[k]
            for f in ndvi_files:
                if f.name not in st.session_state.ndvi_datasets:
                    try:
                        raw = load_clean_gdf(f)
                        st.session_state.ndvi_datasets[f.name] = {
                            "raw": raw,
                            "processed": None,
                            "results": None,
                            "meta": None,
                            "type": "input",
                        }
                    except Exception as e:
                        st.error(f"Error: {e}")
        if ndvi_files == []:
            for k in list(st.session_state.ndvi_datasets.keys()):
                if st.session_state.ndvi_datasets[k].get("type") != "restored":
                    del st.session_state.ndvi_datasets[k]

        st.divider()
        if st.button("Run NDVI Analysis", type="primary"):
            if not st.session_state.ndvi_datasets:
                st.warning("No data uploaded.")
            else:
                if "jobs" not in st.session_state:
                    st.session_state.jobs = {}
                for fname, d in st.session_state.ndvi_datasets.items():
                    if d.get("type") == "restored":
                        continue
                    job_id = str(uuid.uuid4())[:8]
                    st.session_state.jobs[job_id] = {
                        "fname": fname,
                        "task": "NDVI",
                        "start_time": datetime.now().strftime("%H:%M:%S"),
                        "progress": 0.0,
                        "status": "Queued",
                        "cancel": False,
                        "handoff_complete": False,
                    }
                    t = threading.Thread(
                        target=ndvi_worker,
                        args=(
                            job_id,
                            fname,
                            d,
                            start_date,
                            end_date,
                            cloud_pct,
                            resolution,
                            st.session_state.jobs,
                        ),
                    )
                    add_script_run_ctx(t)
                    t.start()
                st.success("NDVI Jobs started! Check Sidebar.")

    with col_ndvi_top_right:
        st.subheader("Input Preview")
        m_ndvi_input = folium.Map(location=[51.0447, -114.0719], zoom_start=10)
        all_bounds = []
        for fname, d in st.session_state.ndvi_datasets.items():
            if d.get("type") == "restored":
                continue
            if d.get("raw") is not None:
                folium.GeoJson(
                    d["raw"],
                    name=f"{fname} (AOI)",
                    style_function=lambda x: {"color": "blue", "fill": False},
                ).add_to(m_ndvi_input)
                all_bounds.append(d["raw"].total_bounds)
        if all_bounds:
            min_x = min([b[0] for b in all_bounds])
            min_y = min([b[1] for b in all_bounds])
            max_x = max([b[2] for b in all_bounds])
            max_y = max([b[3] for b in all_bounds])
            m_ndvi_input.fit_bounds([[min_y, min_x], [max_y, max_x]])
        st_folium(
            m_ndvi_input,
            width="100%",
            height=500,
            key="map_ndvi_input",
            returned_objects=[],
        )

    st.divider()

    col_ndvi_btm_left, col_ndvi_btm_right = st.columns(2)
    with col_ndvi_btm_left:
        st.subheader("2. Result Inspector")
        if st.button("🔄 Scan Output Folder (NDVI)"):
            found_files = glob.glob(os.path.join(output_dir, "*_ndvi.geojson"))
            count = 0
            for p in found_files:
                base_name = os.path.basename(p).replace("_ndvi.geojson", "")
                tif_path = os.path.join(output_dir, f"{base_name}_ndvi.tif")
                if not os.path.exists(tif_path):
                    continue
                if base_name not in st.session_state.ndvi_datasets:
                    try:
                        gdf = gpd.read_file(p)
                        with rasterio.open(tif_path) as src:
                            meta = {
                                "transform": src.transform,
                                "width": src.width,
                                "height": src.height,
                                "crs": src.crs,
                            }

                        # Fix AOI geom for map
                        if gdf.crs.to_epsg() != 4326:
                            raw_geom = (
                                gdf.to_crs(epsg=4326).geometry.union_all().envelope
                            )
                        else:
                            raw_geom = gdf.geometry.union_all().envelope

                        raw_gdf = gpd.GeoDataFrame(
                            {"geometry": [raw_geom]}, crs="EPSG:4326"
                        )
                        st.session_state.ndvi_datasets[base_name] = {
                            "raw": raw_gdf,
                            "processed": None,
                            "results": gdf,
                            "meta": meta,
                            "type": "restored",
                        }
                        count += 1
                    except Exception as e:
                        print(f"Error loading {base_name}: {e}")
            if count > 0:
                st.success(f"Loaded {count} NDVI results.")
            else:
                st.info("No valid NDVI results found.")

        completed_ds = [
            k
            for k, v in st.session_state.ndvi_datasets.items()
            if v.get("type") == "restored"
        ]
        options = ["All Regions"] + completed_ds
        selected_opt = st.selectbox(
            "Select NDVI Result",
            options,
            index=None,
            placeholder="Select a Result...",
            key="ndvi_inspector_select",
        )
        r_opacity = st.slider("Raster Opacity", 0.0, 1.0, 0.7, key="ndvi_op")
        show_points = st.checkbox("Show Validation Points (Blue Dots)", value=False)
        val_container = st.empty()

    with col_ndvi_btm_right:
        st.subheader("Results Preview")
        m_ndvi_result = folium.Map(location=[51.0447, -114.0719], zoom_start=10)

        if selected_opt:
            targets = completed_ds if selected_opt == "All Regions" else [selected_opt]
            res_bounds = []

            for ds_name in targets:
                if ds_name not in st.session_state.ndvi_datasets:
                    continue
                ds = st.session_state.ndvi_datasets[ds_name]

                tif_path = os.path.join(output_dir, f"{ds_name}_ndvi.tif")
                if os.path.exists(tif_path):
                    try:
                        with rasterio.open(tif_path) as src:
                            # 1. READ RAW (Standardized EPSG:4326)
                            arr = src.read(1)

                            # 2. GET BOUNDS
                            # src.bounds -> (min_lon, min_lat, max_lon, max_lat)
                            bounds = src.bounds
                            # Folium -> [[min_lat, min_lon], [max_lat, max_lon]]
                            folium_bounds = [
                                [bounds.bottom, bounds.left],
                                [bounds.top, bounds.right],
                            ]

                            res_bounds.append(
                                [bounds.left, bounds.bottom, bounds.right, bounds.top]
                            )

                            # 3. RENDER
                            norm_data = np.clip((arr - (-0.2)) / (1.0 - (-0.2)), 0, 1)
                            cmap = plt.get_cmap("RdYlGn")
                            colored = cmap(norm_data)

                            mask = (arr == -9999) | np.isnan(arr) | (arr == 0)
                            colored[..., 3] = np.where(mask, 0, r_opacity)

                            img_bytes = (colored * 255).astype(np.uint8)
                            im = PILImage.fromarray(img_bytes)
                            buff = io.BytesIO()
                            im.save(buff, format="PNG")
                            img_url = f"data:image/png;base64,{base64.b64encode(buff.getvalue()).decode()}"

                            folium.raster_layers.ImageOverlay(
                                image=img_url,
                                bounds=folium_bounds,
                                opacity=r_opacity,
                                interactive=True,
                            ).add_to(m_ndvi_result)

                            # Extent Box
                            folium.Rectangle(
                                bounds=folium_bounds, color="red", weight=2, fill=False
                            ).add_to(m_ndvi_result)

                    except Exception as e:
                        print(f"Viz Error {ds_name}: {e}")

                # 4. RENDER POINTS (Validation)
                if show_points:
                    if ds.get("results") is not None:
                        try:
                            gdf_pts = ds["results"]
                            if len(gdf_pts) > 5000:
                                st.warning(f"{ds_name}: Showing 5000 sample points.")
                                gdf_pts = gdf_pts.sample(5000)

                            folium.GeoJson(
                                gdf_pts,
                                marker=folium.Circle(
                                    radius=1, color="blue", fill=True, fill_opacity=1
                                ),
                                tooltip=folium.GeoJsonTooltip(
                                    fields=["NDVI"], aliases=["Val:"]
                                ),
                            ).add_to(m_ndvi_result)
                        except Exception as e:
                            st.error(f"Error loading points: {e}")

            if res_bounds:
                min_x = min([b[0] for b in res_bounds])
                min_y = min([b[1] for b in res_bounds])
                max_x = max([b[2] for b in res_bounds])
                max_y = max([b[3] for b in res_bounds])
                m_ndvi_result.fit_bounds([[min_y, min_x], [max_y, max_x]])

        map_data = st_folium(
            m_ndvi_result,
            width="100%",
            height=500,
            key="map_ndvi_result",
            returned_objects=["last_clicked"],
        )

        # Click Logic
        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lon = map_data["last_clicked"]["lng"]
            val_found = False
            search_targets = (
                completed_ds
                if (selected_opt == "All Regions" or selected_opt is None)
                else [selected_opt]
            )

            for ds_name in search_targets:
                if ds_name not in st.session_state.ndvi_datasets:
                    continue
                tif_path = os.path.join(output_dir, f"{ds_name}_ndvi.tif")
                if os.path.exists(tif_path):
                    with rasterio.open(tif_path) as src:
                        try:
                            # 1. Simple lookup (Source is 4326)
                            r, c = src.index(lon, lat)
                            window = rasterio.windows.Window(c, r, 1, 1)
                            val = src.read(1, window=window)
                            if val.size > 0:
                                pixel_val = val[0][0]
                                if (
                                    pixel_val != -9999
                                    and not np.isnan(pixel_val)
                                    and pixel_val != 0
                                ):
                                    val_container.info(
                                        f"**{ds_name}**: NDVI = {pixel_val:.4f} at ({lat:.4f}, {lon:.4f})"
                                    )
                                    val_found = True
                                    break
                        except:
                            pass
            if not val_found:
                val_container.info(
                    f"Clicked at ({lat:.4f}, {lon:.4f}) - No valid data."
                )

# -----------------------------------------------------------------------------
# TAB 3: GVI CONFIGURATOR (2x2 Grid Layout)
# -----------------------------------------------------------------------------
with tab3:
    st.header("Street View & Vision Pipeline")

    # --- GLOBAL RESOURCES & LOCKS ---
    @st.cache_resource
    def get_gpu_lock():
        return threading.Lock()

    gpu_lock = get_gpu_lock()

    @st.cache_resource
    def get_gvi_engine(model_path, api_key):
        # Auto-selects best device: CUDA > MPS > CPU
        best_device = get_best_device()
        st.info(f"🚀 Using device: {best_device}")
        return GVIEngine(
            model_path=model_path, device=str(best_device), api_key=api_key
        )

    # --- SESSION STATE ---
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}
    if "master_cache" not in st.session_state:
        st.session_state.master_cache = {}
    if "jobs" not in st.session_state:
        st.session_state.jobs = {}

    if "inspector_select_key" not in st.session_state:
        st.session_state.inspector_select_key = None

    def generate_raster_grid(gdf_4326, spacing_meters):
        # (This function is already defined in global scope above, reusing)
        from rasterio.transform import from_bounds, xy

        minx, miny, maxx, maxy = gdf_4326.total_bounds
        center_lat = (miny + maxy) / 2.0
        lat_rad = np.radians(center_lat)
        m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad)
        m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
        res_x = spacing_meters / m_per_deg_lon
        res_y = spacing_meters / m_per_deg_lat
        width = int(np.ceil((maxx - minx) / res_x))
        height = int(np.ceil((maxy - miny) / res_y))
        transform = from_bounds(
            minx, miny, minx + (width * res_x), miny + (height * res_y), width, height
        )
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        xs, ys = xy(transform, rows.flatten(), cols.flatten(), offset="center")
        df = pd.DataFrame(
            {"row": rows.flatten(), "col": cols.flatten(), "x": xs, "y": ys}
        )
        gdf_grid = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:4326"
        )
        gdf_clipped = gpd.sjoin(gdf_grid, gdf_4326, how="inner", predicate="intersects")
        return gdf_clipped, {
            "transform": transform,
            "width": width,
            "height": height,
            "crs": "EPSG:4326",
        }

    # --- THREAD WORKER ---
    def job_worker(job_id, fname, dataset_data, init_args, run_args, job_tracker_dict):
        try:
            job_tracker_dict[job_id]["status"] = "Waiting for GPU..."
            with gpu_lock:
                if job_tracker_dict[job_id]["cancel"]:
                    job_tracker_dict[job_id]["status"] = "Cancelled"
                    return

                job_tracker_dict[job_id]["status"] = "Initializing..."
                # Extract model_path and api_key (device is auto-selected)
                engine = get_gvi_engine(
                    init_args["model_path"], init_args.get("api_key")
                )

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
                        import rasterio
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

    # --- SIDEBAR LOGIC ---
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
                    c1, c2 = st.columns([7, 3])
                    c1.markdown(f"**{job['fname']}**")
                    c2.caption(f"{job.get('task','Job')} | {job.get('start_time','')}")
                    st.progress(job["progress"])
                    st.caption(f"Status: {job['status']}")

                    is_running = (
                        job["status"]
                        in [
                            "Queued",
                            "Initializing...",
                            "Waiting for GPU...",
                            "Running",
                        ]
                        or "Processing" in job["status"]
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

    # --- ROW 1 LEFT: Job Settings ---
    with col_top_left:
        st.subheader("1. Job Configuration")
        mode = st.radio(
            "Download Mode", ["Package (Scraper)", "API (Google Key)"], horizontal=True
        )
        api_key = (
            st.text_input("Google API Key", type="password")
            if mode == "API (Google Key)"
            else None
        )
        gvi_res = st.slider("Grid Resolution (meters)", 20, 500, 50)
        save_debug = st.checkbox("Save Raw Images & Masks?", value=False)

        uploaded_files = st.file_uploader(
            "Upload Study Areas", accept_multiple_files=True, key="gvi_up"
        )

        # Sync Uploads
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

        # Buttons
        if st.session_state.datasets:
            if st.button("Generate Sampling Grids"):
                with st.spinner("Processing..."):
                    for d in st.session_state.datasets.values():
                        if d.get("type") == "restored":
                            continue
                        if d["type"] == "poly":
                            pts, meta = generate_raster_grid(d["raw"], gvi_res)
                            d["processed"] = pts
                            d["meta"] = meta
                        else:
                            d["processed"] = d["raw"].copy()
                            d["meta"] = None
                        d["accumulated"] = []
                        d["results"] = None
                    st.success("Grids generated!")

        st.divider()
        if st.button("🚀 Start Batch Analysis", type="primary"):
            if not st.session_state.datasets:
                st.warning("No data.")
            else:
                model_path = os.path.join(
                    parent_dir, "geofuse", "model", "best_model.pth"
                )
                started = False
                for fname, d in st.session_state.datasets.items():
                    if d.get("type") == "restored":
                        continue

                    # --- AUTO-GENERATE GRID IF MISSING ---
                    if d.get("processed") is None:
                        if d["type"] == "poly":
                            pts, meta = generate_raster_grid(d["raw"], gvi_res)
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

                    init_args = {
                        "model_path": model_path,
                        # Device auto-selected by engine (CUDA > MPS > CPU)
                        "api_key": api_key,
                    }
                    run_args = {
                        "step": gvi_res,
                        "save_panos": save_debug,
                        "save_masks": save_debug,
                    }
                    d["cache_ref"] = st.session_state.master_cache

                    t = threading.Thread(
                        target=job_worker,
                        args=(
                            job_id,
                            fname,
                            d,
                            init_args,
                            run_args,
                            st.session_state.jobs,
                        ),
                    )
                    add_script_run_ctx(t)
                    t.start()
                    started = True

                if started:
                    st.success("Jobs started!")
                else:
                    st.info("No new jobs.")

    # --- ROW 1 RIGHT: Input Preview Map ---
    with col_top_right:
        st.subheader("Input Preview")
        m_input = folium.Map(location=[51.0447, -114.0719], zoom_start=11)

        all_bounds = []
        for fname, d in st.session_state.datasets.items():
            # STRICT FILTER: Show only input datasets (exclude restored)
            if d.get("type") == "restored":
                continue

            if d.get("raw") is not None:
                folium.GeoJson(
                    d["raw"],
                    name=f"{fname} (AOI)",
                    style_function=lambda x: {"color": "blue", "fill": False},
                ).add_to(m_input)
                all_bounds.append(d["raw"].total_bounds)
            if d.get("processed") is not None and not d["processed"].empty:
                preview = d["processed"].iloc[:1000]
                # SAFE RENDER: Check if Geometry is Point (Avoiding Polygon 'y' crash)
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
                    # Render Polygons if grids are not points
                    folium.GeoJson(
                        preview, style_function=lambda x: {"color": "red", "weight": 1}
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

    # --- ROW 2 LEFT: Inspector Controls ---
    with col_btm_left:
        st.subheader("2. Result Inspector")

        if st.button("🔄 Scan Output Folder for Completed Jobs"):
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

                        # Use union_all() instead of deprecated unary_union
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
                st.success(f"Loaded {count} results.")
            else:
                st.info("No valid paired results found.")

        # STRICT FILTER: Only show scanned (restored) datasets in the dropdown
        completed_ds = [
            k
            for k, v in st.session_state.datasets.items()
            if v.get("type") == "restored"
        ]

        # Add All Regions option
        options = ["All Regions"] + completed_ds

        selected_option = st.selectbox(
            "Select Area to Inspect",
            options,
            index=None,
            placeholder="Select a Result...",
            key="inspector_select_key",
        )

        # REMOVED POINTS OPTION, ONLY RASTER SELECTION NOW
        raster_layer = st.radio(
            "Background Raster", ["Vegetation", "Terrain"], horizontal=True
        )
        r_opacity = st.slider("Layer Opacity", 0.0, 1.0, 0.7)

    # --- ROW 2 RIGHT: Result Preview Map ---
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

                # DRAW RASTER BOX
                if ds.get("meta"):
                    meta = ds["meta"]
                    left, bottom, right, top = array_bounds(
                        meta["height"], meta["width"], meta["transform"]
                    )

                    folium.Rectangle(
                        bounds=[[bottom, left], [top, right]],
                        color="grey",
                        weight=1,
                        fill=False,
                        popup=f"{ds_name} Extent",
                    ).add_to(m_result)
                    res_bounds.append([left, bottom, right, top])

                    # 1. RENDER RASTER
                    import base64
                    import io

                    from PIL import Image as PILImage
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
                        img_url = f"data:image/png;base64,{base64.b64encode(buff.getvalue()).decode()}"

                        folium.raster_layers.ImageOverlay(
                            image=img_url,
                            bounds=[[bottom, left], [top, right]],
                            opacity=r_opacity,
                            interactive=False,
                        ).add_to(m_result)

                # 2. RENDER POINTS (ALWAYS ON)
                gdf_viz = ds["results"].copy()
                if "gvi_veg" in gdf_viz.columns:
                    gdf_viz["gvi_veg"] = gdf_viz["gvi_veg"].round(4)
                if "gvi_ter" in gdf_viz.columns:
                    gdf_viz["gvi_ter"] = gdf_viz["gvi_ter"].round(4)

                valid_pts = gdf_viz.dropna(subset=["gvi_veg"])
                folium.GeoJson(
                    valid_pts,
                    marker=folium.Circle(
                        radius=20, fill_color="green", fill_opacity=0.8, color=None
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

# -----------------------------------------------------------------------------
# TAB 4: FUSION DEMO
# -----------------------------------------------------------------------------
# TODO: FUSION_UI_INTEGRATION - Integrate with GVI/NDVI results for seamless workflow
# TODO: FUSION_AUTO_MERGE - Auto-merge GVI and NDVI outputs by spatial join
# TODO: FUSION_VISUALIZATION - Add correlation plots and weight sensitivity analysis

with tab4:
    st.header("Composite Metric Fusion")
    st.markdown(
        "Interactive demonstration of how weights affect the outcome correlation."
    )

    if FusionOptimizer is None:
        st.error("FusionOptimizer module not found in geofuse/fusion.py")
    else:
        uploaded_csv = st.file_uploader("Upload Merged Data (CSV)", type="csv")

        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            st.write("Data Preview:", df.head())

            # Attempt to guess columns
            num_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()

            target = st.selectbox("Target Outcome Variable", num_cols)
            features = st.multiselect(
                "Environmental Features",
                num_cols,
                default=[c for c in num_cols if "NDVI" in c or "GVI" in c],
            )

            if features and target:
                col_opt, col_man = st.columns(2)

                with col_man:
                    st.subheader("Manual Weighting")
                    weights = {}
                    for f in features:
                        weights[f] = st.slider(f"Weight: {f}", 0.0, 1.0, 0.5, key=f)

                    df["Manual_CGI"] = 0
                    for f, w in weights.items():
                        df["Manual_CGI"] += df[f] * w

                    # Simple correlation
                    corr = df["Manual_CGI"].corr(df[target])
                    st.metric("Pearson Correlation", f"{corr:.3f}")

                with col_opt:
                    st.subheader("AI Optimization (Optuna)")
                    n_trials = st.number_input("Optimization Trials", 50, 500, 100)

                    if st.button("Run Optimization"):
                        with st.spinner("Optimizing..."):
                            opt = FusionOptimizer(df, target, features)
                            best_params = opt.run_optimization(
                                total_trials=n_trials, random_trials=20
                            )

                            st.success("Optimization Complete!")
                            st.json(best_params)

                            df_opt = opt.apply_best_weights()
                            best_corr = df_opt["CGI"].corr(df_opt[target])
                            st.metric("Optimized Correlation", f"{best_corr:.3f}")

                            delta = best_corr - corr
                            st.markdown(f"**Improvement over manual:** {delta:+.3f}")
