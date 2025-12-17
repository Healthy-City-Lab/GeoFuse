import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import json
import os
import sys
import time
from datetime import date
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt

from geofuse.ndvi import NDVIEngine
from geofuse.gvi import GVIEngine

# --- 1. GLOBAL PATH SETUP (Fixes NameError) ---
# Get the folder containing this script (GeoFuse/ui)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (GeoFuse/)
parent_dir = os.path.dirname(current_dir)

# Add the project root to sys.path so we can import 'geofuse' modules
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- 2. GDAL ENVIRONMENT FIX (Fixes Terminal Warnings) ---
# On Windows Conda, GDAL often forgets where its data files are. We help it.
if "GDAL_DATA" not in os.environ:
    # Typical path: C:\Users\...\envs\geofuse\Library\share\gdal
    conda_prefix = sys.prefix
    gdal_data_path = os.path.join(conda_prefix, "Library", "share", "gdal")

    if os.path.exists(gdal_data_path):
        os.environ["GDAL_DATA"] = gdal_data_path
    else:
        # Fallback check
        print(f"[WARN] Could not find GDAL data at {gdal_data_path}")

# FIX: Allow multiple OpenMP runtimes to coexist (silences the crash)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Assuming FusionOptimizer exists in geofuse/fusion.py based on your repo structure
try:
    from geofuse.fusion import FusionOptimizer
except ImportError:
    FusionOptimizer = None

st.set_page_config(page_title="GeoFuse Toolbox", layout="wide")
st.title("GeoFuse: Multimodal Environmental Profiling")

# Tabs for the different modules
tab1, tab2, tab3, tab4 = st.tabs(
    ["Job Monitor", "NDVI Sourcing", "GVI Sourcing", "Fusion & Optimization"]
)

# Shared Output Directory
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# -----------------------------------------------------------------------------
# TAB 1: HPC JOB MONITOR
# -----------------------------------------------------------------------------
with tab1:
    st.header("HPC Job Monitor")
    st.markdown("Track the progress of remote HPC jobs via their Job ID.")

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
                    except json.JSONDecodeError:
                        time.sleep(0.5)
                        continue

                    with placeholder.container():
                        st.subheader(f"Stage: {data.get('stage', 'Unknown')}")
                        prog = data.get("progress", 0)
                        st.progress(prog)

                        metrics = data.get("metrics", {})
                        if metrics:
                            st.json(metrics)

                        log_file = os.path.join("logs", f"{job_id}.txt")
                        if os.path.exists(log_file):
                            with st.expander("Live Logs (Tail)", expanded=True):
                                with open(log_file, "r") as lf:
                                    lines = lf.readlines()[-15:]
                                    st.code("".join(lines))

                    if not auto_refresh or prog >= 100:
                        break
                    time.sleep(2)
            else:
                st.info("Waiting for job to start or invalid Job ID...")
        else:
            st.info("Enter a Job ID to begin monitoring.")

# -----------------------------------------------------------------------------
# TAB 2: NDVI CONFIGURATOR
# -----------------------------------------------------------------------------
with tab2:
    st.header("Satellite Data Configuration")
    col_a, col_b = st.columns([1, 2])

    with col_a:
        target_date = st.date_input("Target Date", value=date(2023, 7, 15))
        tolerance = st.number_input("Cloud Tolerance (Days)", value=15, min_value=1)
        cloud_max = st.slider("Max Cloud Cover (%)", 0, 30, 10)
        ndvi_file = st.file_uploader(
            "Upload Catchment Area (Shapefile/GeoJSON)", key="ndvi_up"
        )

        if st.button("Run NDVI Analysis", type="primary"):
            if ndvi_file:
                # Save temp file
                temp_path = os.path.join(output_dir, "temp_aoi_ndvi.geojson")
                with open(temp_path, "wb") as f:
                    f.write(ndvi_file.getbuffer())

                status_box = st.empty()
                status_box.text("Initializing Earth Engine...")

                try:
                    engine = NDVIEngine()
                    output_tif = os.path.join(output_dir, f"ndvi_{target_date}.tif")

                    status_box.text("Downloading Satellite Data...")
                    success = engine.export_geotiff(
                        temp_path, str(target_date), output_tif, tolerance=tolerance
                    )

                    if success:
                        status_box.success(f"Saved to: {output_tif}")
                    else:
                        status_box.error("No images found for this date/location.")

                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please upload a shapefile first.")

    with col_b:
        # Map Visualization
        m = folium.Map(location=[51.0447, -114.0719], zoom_start=10)

        if ndvi_file:
            try:
                ndvi_file.seek(0)
                gdf = gpd.read_file(ndvi_file)
                gdf_wgs84 = gdf.to_crs(epsg=4326)
                centroid = gdf_wgs84.geometry.centroid.iloc[0]
                m = folium.Map(location=[centroid.y, centroid.x], zoom_start=11)
                folium.GeoJson(gdf_wgs84).add_to(m)
            except Exception as e:
                st.error(f"Visualization Error: {e}")

        st_folium(m, width=800, height=500)

# -----------------------------------------------------------------------------
# TAB 3: GVI CONFIGURATOR
# -----------------------------------------------------------------------------
with tab3:
    st.header("Street View & Vision Pipeline")

    col_a, col_b = st.columns([1, 2])
    m_gvi = folium.Map(location=[51.0447, -114.0719], zoom_start=11)

    with col_a:
        mode = st.radio("Download Mode", ["Package (Scraper)", "API (Google Key)"])
        api_key = (
            st.text_input("Google API Key", type="password")
            if mode == "API (Google Key)"
            else None
        )

        gvi_res = st.slider("Grid Resolution (meters)", 20, 500, 50)
        save_debug = st.checkbox("Save Raw Images?", value=False)
        gvi_file = st.file_uploader("Upload Study Area (GeoJSON/SHP)", key="gvi_up")

        # --- DATA LOADER ---
        @st.cache_data(show_spinner=False)
        def process_uploaded_file(uploaded_file):
            temp_path = os.path.join(output_dir, "temp_aoi_gvi.geojson")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                gdf = gpd.read_file(temp_path)
                import pandas as pd

                for col in gdf.columns:
                    if gdf[
                        col
                    ].dtype == "object" or pd.api.types.is_datetime64_any_dtype(
                        gdf[col]
                    ):
                        try:
                            gdf[col] = gdf[col].astype(str)
                        except:
                            gdf = gdf.drop(columns=[col])

                if gdf.crs and gdf.crs.is_geographic:
                    utm_crs = gdf.estimate_utm_crs()
                    gdf_metric = gdf.to_crs(utm_crs)
                else:
                    gdf_metric = gdf

                return gdf_metric, gdf.to_crs(epsg=4326), gdf.total_bounds
            except Exception as e:
                return None, None, None

        gdf_metric = None
        if gvi_file:
            gdf_metric, gdf_vis, bounds = process_uploaded_file(gvi_file)
            if gdf_vis is not None:
                folium.GeoJson(
                    gdf_vis,
                    name="AOI",
                    style_function=lambda x: {"fillColor": "#3388ff"},
                ).add_to(m_gvi)
                m_gvi.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        # --- PREVIEW GRID BUTTON ---
        if "show_preview" not in st.session_state:
            st.session_state.show_preview = False

        if st.button("👁️ Preview Sampling Grid"):
            st.session_state.show_preview = True

        if st.session_state.show_preview and gdf_metric is not None:
            # Replicate Backend Logic: Pixel Centers
            minx, miny, maxx, maxy = gdf_metric.total_bounds

            # Dimensions
            width = int(np.ceil((maxx - minx) / gvi_res))
            height = int(np.ceil((maxy - miny) / gvi_res))

            cols = np.arange(width)
            rows = np.arange(height)

            preview_points = []
            for r in rows:
                y_center = maxy - (r + 0.5) * gvi_res
                for c in cols:
                    x_center = minx + (c + 0.5) * gvi_res
                    if gdf_metric.contains(Point(x_center, y_center)).any():
                        preview_points.append(Point(x_center, y_center))

            if len(preview_points) > 0:
                gdf_prev = gpd.GeoDataFrame(
                    geometry=preview_points, crs=gdf_metric.crs
                ).to_crs(epsg=4326)
                for _, row in gdf_prev.iterrows():
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=2,
                        color="red",
                        fill=True,
                        fill_opacity=1,
                    ).add_to(m_gvi)

        # --- RUN ANALYSIS ---
        if "gvi_results" not in st.session_state:
            st.session_state.gvi_results = None

        if st.button("🚀 Run GVI Analysis", type="primary"):
            model_path = os.path.join(parent_dir, "geofuse", "model", "best_model.pth")
            if gdf_metric is not None:
                try:
                    engine = GVIEngine(
                        model_path=model_path, device="cuda", api_key=api_key
                    )
                    with st.spinner(f"Analyzing {gvi_res}m grid..."):
                        results = engine.run_analysis(
                            gdf_metric,
                            step=gvi_res,
                            folder=output_dir,
                            save_panos=save_debug,
                            save_masks=save_debug,
                        )
                        st.session_state.gvi_results = results
                    if not results.empty:
                        st.success("Processing Complete!")
                except Exception as e:
                    st.error(f"Failed: {e}")

        # --- DISPLAY RESULTS ---
        if st.session_state.gvi_results is not None:
            gdf_res = st.session_state.gvi_results
            col_d1, col_d2 = st.columns(2)

            res_path = os.path.join(output_dir, "final_gvi_results.geojson")
            tif_path = os.path.join(output_dir, "gvi_distribution.tif")

            gdf_out = gdf_res.to_crs(epsg=4326)
            gdf_out.to_file(res_path, driver="GeoJSON")
            col_d1.download_button(
                "⬇️ Results (GeoJSON)", open(res_path, "rb"), "gvi_points.geojson"
            )

            if os.path.exists(tif_path):
                col_d2.download_button(
                    "⬇️ Distribution Map (GeoTIFF)", open(tif_path, "rb"), "gvi_map.tif"
                )

            st.subheader("Map Visualization")
            viz_layer = st.radio(
                "Select Layer:",
                [
                    "Points (Vector)",
                    "Vegetation Heatmap (Raster)",
                    "Terrain Heatmap (Raster)",
                ],
                horizontal=True,
            )

            # 1. Vector Layer
            if viz_layer == "Points (Vector)":
                # Only show non-null values
                valid_points = gdf_out.dropna(subset=["gvi_veg"])
                folium.GeoJson(
                    valid_points,
                    name="GVI Points",
                    tooltip=folium.GeoJsonTooltip(
                        fields=["gvi_veg", "gvi_ter"], aliases=["Veg:", "Ter:"]
                    ),
                    style_function=lambda x: {
                        "color": (
                            "#00FF00" if x["properties"]["gvi_veg"] > 0 else "#808080"
                        ),
                        "radius": 3,
                        "fillOpacity": 0.8,
                    },
                ).add_to(m_gvi)

            # 2. Raster Layer (With Reprojection)
            elif "Raster" in viz_layer and os.path.exists(tif_path):
                import rasterio
                from rasterio.warp import (
                    calculate_default_transform,
                    reproject,
                    Resampling,
                )
                import matplotlib.pyplot as plt

                band_idx = 1 if "Vegetation" in viz_layer else 2

                # Use updated Matplotlib API
                colormap = (
                    plt.colormaps["Greens"] if band_idx == 1 else plt.colormaps["OrRd"]
                )

                with rasterio.open(tif_path) as src:
                    # Reproject from UTM to WGS84 for Folium
                    transform, width, height = calculate_default_transform(
                        src.crs, "EPSG:4326", src.width, src.height, *src.bounds
                    )

                    data = np.zeros((height, width), dtype=np.float32)

                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs="EPSG:4326",
                        resampling=Resampling.nearest,
                    )

                    # Normalize & Colorize
                    data_norm = (data - 0) / (0.6 - 0)
                    data_norm = np.clip(data_norm, 0, 1)
                    colored_data = colormap(data_norm)

                    # Alpha masking: 0.9 Opacity for data, 0 for NoData
                    mask_nan = (data == src.nodata) | np.isnan(data) | (data == -1)
                    colored_data[..., 3] = np.where(mask_nan, 0, 0.9)

                    # Convert to PNG
                    from PIL import Image as PILImage
                    import io, base64

                    img_bytes = (colored_data * 255).astype(np.uint8)
                    im = PILImage.fromarray(img_bytes)
                    buff = io.BytesIO()
                    im.save(buff, format="PNG")
                    img_url = f"data:image/png;base64,{base64.b64encode(buff.getvalue()).decode()}"

                    # Calculate Bounds for Folium
                    from rasterio.transform import array_bounds

                    w, s, e, n = array_bounds(height, width, transform)
                    image_bounds = [[s, w], [n, e]]

                    folium.raster_layers.ImageOverlay(
                        image=img_url, bounds=image_bounds, opacity=0.9, name=viz_layer
                    ).add_to(m_gvi)

    with col_b:
        st_folium(m_gvi, width=800, height=500)

# -----------------------------------------------------------------------------
# TAB 4: FUSION DEMO
# -----------------------------------------------------------------------------
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
