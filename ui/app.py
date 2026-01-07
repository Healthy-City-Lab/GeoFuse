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
import matplotlib
import matplotlib.pyplot as plt

from geofuse.ndvi import NDVIEngine
from geofuse.gvi import GVIEngine

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
                            with st.expander("Live Logs", expanded=True):
                                with open(log_file, "r") as lf:
                                    st.code("".join(lf.readlines()[-15:]))

                    if not auto_refresh or prog >= 100:
                        break
                    time.sleep(2)
            else:
                st.info("Waiting for job to start...")

# -----------------------------------------------------------------------------
# TAB 2: NDVI CONFIGURATOR
# -----------------------------------------------------------------------------
with tab2:
    st.header("Satellite Data Configuration")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        target_date = st.date_input("Target Date", value=date(2023, 7, 15))
        tolerance = st.number_input("Cloud Tolerance (Days)", value=15, min_value=1)
        ndvi_file = st.file_uploader("Upload Catchment Area", key="ndvi_up")

        if st.button("Run NDVI Analysis", type="primary"):
            if ndvi_file:
                temp_path = os.path.join(output_dir, "temp_aoi_ndvi.geojson")
                with open(temp_path, "wb") as f:
                    f.write(ndvi_file.getbuffer())

                status = st.empty()
                status.text("Initializing Earth Engine...")
                try:
                    engine = NDVIEngine()
                    output_tif = os.path.join(output_dir, f"ndvi_{target_date}.tif")
                    status.text("Downloading Satellite Data...")
                    if engine.export_geotiff(
                        temp_path, str(target_date), output_tif, tolerance=tolerance
                    ):
                        status.success(f"Saved to: {output_tif}")
                    else:
                        status.error("No images found.")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col_b:
        m = folium.Map(location=[51.0447, -114.0719], zoom_start=10)
        if ndvi_file:
            try:
                ndvi_file.seek(0)
                gdf = gpd.read_file(ndvi_file).to_crs(epsg=4326)
                b = gdf.total_bounds
                m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
                folium.GeoJson(gdf).add_to(m)
            except Exception as e:
                st.error(f"Map Error: {e}")
        st_folium(m, width=800, height=500)

# -----------------------------------------------------------------------------
# TAB 3: GVI CONFIGURATOR (Optimized)
# -----------------------------------------------------------------------------
with tab3:
    st.header("Street View & Vision Pipeline")
    col_a, col_b = st.columns([1, 2])
    m_gvi = folium.Map(location=[51.0447, -114.0719], zoom_start=11)

    @st.cache_data
    def load_and_clean_gdf(file_obj):
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

            # Sanitize dates/objects
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

        centroid = gdf_4326.geometry.centroid.iloc[0]
        lat_rad = np.radians(centroid.y)
        m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad)
        m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)

        res_x = spacing_meters / m_per_deg_lon
        res_y = spacing_meters / m_per_deg_lat

        minx, miny, maxx, maxy = gdf_4326.total_bounds
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

        grid_meta = {
            "transform": transform,
            "width": width,
            "height": height,
            "crs": "EPSG:4326",
        }
        return gdf_clipped, grid_meta

    with col_a:
        mode = st.radio("Download Mode", ["Package (Scraper)", "API (Google Key)"])
        api_key = (
            st.text_input("Google API Key", type="password")
            if mode == "API (Google Key)"
            else None
        )
        gvi_res = st.slider("Grid Resolution (meters)", 20, 500, 50)
        save_debug = st.checkbox("Save Raw Images & Masks?", value=False)
        gvi_file = st.file_uploader("Upload Study Area", key="gvi_up")

        if "gdf_points" not in st.session_state:
            st.session_state.gdf_points = None
            st.session_state.grid_meta = None

        if gvi_file:
            try:
                raw_gdf = load_and_clean_gdf(gvi_file)
                folium.GeoJson(
                    raw_gdf,
                    name="AOI",
                    style_function=lambda x: {"fillColor": "none", "color": "blue"},
                ).add_to(m_gvi)
                b = raw_gdf.total_bounds
                m_gvi.fit_bounds([[b[1], b[0]], [b[3], b[2]]])

                if st.button("Generate Sampling Grid"):
                    with st.spinner("Calculating grid..."):
                        pts, meta = generate_raster_grid(raw_gdf, gvi_res)
                        st.session_state.gdf_points = pts
                        st.session_state.grid_meta = meta
                        st.success(f"Generated {len(pts)} sampling points.")
            except Exception as e:
                st.error(f"Error: {e}")

        # PREVIEW (Visually limited to 1000, but logic uses FULL data)
        if st.session_state.gdf_points is not None:
            total_pts = len(st.session_state.gdf_points)
            if st.checkbox(f"Show Preview Points ({total_pts})", value=True):
                limit = 1000
                preview_df = (
                    st.session_state.gdf_points.iloc[:limit]
                    if total_pts > limit
                    else st.session_state.gdf_points
                )
                for _, row in preview_df.iterrows():
                    folium.CircleMarker(
                        [row.geometry.y, row.geometry.x],
                        radius=2,
                        color="red",
                        fill=True,
                        fill_opacity=1,
                    ).add_to(m_gvi)
                if total_pts > limit:
                    st.caption(
                        f"⚠️ Displaying first {limit} points only (Analysis uses all {total_pts})."
                    )

        # RUN ANALYSIS
        if "gvi_results" not in st.session_state:
            st.session_state.gvi_results = None

        if st.button("🚀 Run GVI Analysis", type="primary"):
            if st.session_state.gdf_points is None:
                st.warning("Please generate grid points first.")
            else:
                try:
                    # 1. Prepare Data (FULL DATASET, NO LIMITS)
                    gdf_analysis = st.session_state.gdf_points.copy()

                    # 2. Run Engine (No UTM conversion needed for Points mode)
                    model_path = os.path.join(
                        parent_dir, "geofuse", "model", "best_model.pth"
                    )
                    engine = GVIEngine(
                        model_path=model_path, device="cuda", api_key=api_key
                    )

                    with st.spinner(f"Analyzing {len(gdf_analysis)} points..."):
                        # FIX: Pass save_masks explicitly
                        results_gdf = engine.run_analysis(
                            gdf_analysis,
                            step=gvi_res,
                            folder=output_dir,
                            save_panos=save_debug,
                            save_masks=save_debug,  # <--- FIXED: Masks now save
                        )

                        # Merge results back to session state
                        # Ensure columns exist
                        if "gvi_veg" not in st.session_state.gdf_points:
                            st.session_state.gdf_points["gvi_veg"] = np.nan
                            st.session_state.gdf_points["gvi_ter"] = np.nan

                        # Use index to update the master dataframe
                        st.session_state.gdf_points.update(results_gdf)
                        st.session_state.gvi_results = (
                            st.session_state.gdf_points.copy()
                        )

                        # --- RESTORED: SAVE OUTPUT FILES ---
                        # A. Save GeoJSON
                        json_path = os.path.join(output_dir, "gvi_results.geojson")
                        st.session_state.gvi_results.to_file(
                            json_path, driver="GeoJSON"
                        )

                        # B. Save Multi-Band GeoTIFF (Reconstructed from Grid Meta)
                        meta = st.session_state.grid_meta
                        if meta:
                            import rasterio

                            # Build arrays from point data
                            arr_veg = np.full(
                                (meta["height"], meta["width"]),
                                np.nan,
                                dtype=np.float32,
                            )
                            arr_ter = np.full(
                                (meta["height"], meta["width"]),
                                np.nan,
                                dtype=np.float32,
                            )

                            # Map values to grid indices
                            # Filter only valid results that were returned
                            res = st.session_state.gvi_results.dropna(
                                subset=["gvi_veg"]
                            )
                            rows = res["row"].astype(int).values
                            cols = res["col"].astype(int).values

                            arr_veg[rows, cols] = res["gvi_veg"].values
                            arr_ter[rows, cols] = res["gvi_ter"].values

                            tif_path = os.path.join(output_dir, "gvi_distribution.tif")
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
                                dst.set_band_description(1, "Vegetation GVI")
                                dst.write(arr_ter, 2)
                                dst.set_band_description(2, "Terrain GVI")

                    st.success(f"Analysis Complete! Saved to {output_dir}")

                except Exception as e:
                    st.error(f"Failed: {e}")
                    st.exception(e)

        # --- DISPLAY RESULTS (ROBUST VERSION) ---
        if st.session_state.gvi_results is not None:
            gdf_res = st.session_state.gvi_results
            meta = st.session_state.grid_meta

            viz_layer = st.radio(
                "Layer:",
                ["Points", "Vegetation Raster", "Terrain Raster"],
                horizontal=True,
            )

            if viz_layer == "Points":
                # Filter out points where GVI calculation failed (no street view)
                valid_pts = gdf_res.dropna(subset=["gvi_veg"])
                st.info(f"Points with Data: {len(valid_pts)} / {len(gdf_res)}")

                folium.GeoJson(
                    valid_pts,
                    name="GVI Points",
                    marker=folium.Circle(
                        radius=3, fill_color="green", fill_opacity=1, color=None
                    ),
                    # UPDATED: Added Terrain to tooltip
                    tooltip=folium.GeoJsonTooltip(
                        fields=["gvi_veg", "gvi_ter"],
                        aliases=["Veg:", "Ter:"],
                        localize=True,
                    ),
                ).add_to(m_gvi)

            elif "Raster" in viz_layer:
                from rasterio.transform import array_bounds, rowcol
                import io, base64
                from PIL import Image as PILImage
                import matplotlib.pyplot as plt

                # --- NEW: Opacity Slider ---
                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    r_opacity = st.slider("Raster Opacity", 0.0, 1.0, 0.7, step=0.1)

                # 1. Initialize Empty Grid
                arr = np.full((meta["height"], meta["width"]), np.nan)
                col_name = "gvi_veg" if "Vegetation" in viz_layer else "gvi_ter"

                # 2. ROBUST FILL: Convert Coordinates -> Row/Col indices
                res = gdf_res.dropna(subset=[col_name])

                if not res.empty:
                    # Inverse Transform: Map Lat/Lon back to Grid Indices
                    # This ensures visual alignment matches spatial reality 100%
                    rows, cols = rowcol(
                        meta["transform"], res.geometry.x.values, res.geometry.y.values
                    )

                    rows = np.array(rows)
                    cols = np.array(cols)
                    vals = res[col_name].values

                    # Safety Check
                    valid_mask = (
                        (rows >= 0)
                        & (rows < meta["height"])
                        & (cols >= 0)
                        & (cols < meta["width"])
                    )

                    arr[rows[valid_mask], cols[valid_mask]] = vals[valid_mask]

                # 3. Visual Styling
                try:
                    cmap = (
                        matplotlib.colormaps["Greens"]
                        if "Vegetation" in viz_layer
                        else matplotlib.colormaps["OrRd"]
                    )
                except:
                    cmap = (
                        plt.cm.get_cmap("Greens")
                        if "Vegetation" in viz_layer
                        else plt.cm.get_cmap("OrRd")
                    )

                norm_data = np.clip((arr - 0) / 0.6, 0, 1)
                colored = cmap(norm_data)

                # Apply Opacity Slider
                mask = np.isnan(arr)
                colored[..., 3] = np.where(mask, 0, r_opacity)

                # 4. Create Image
                img_bytes = (colored * 255).astype(np.uint8)
                im = PILImage.fromarray(img_bytes)
                buff = io.BytesIO()
                im.save(buff, format="PNG")
                img_url = f"data:image/png;base64,{base64.b64encode(buff.getvalue()).decode()}"

                # 5. Define Bounds
                left, bottom, right, top = array_bounds(
                    meta["height"], meta["width"], meta["transform"]
                )

                folium.raster_layers.ImageOverlay(
                    image=img_url,
                    bounds=[[bottom, left], [top, right]],
                    opacity=r_opacity,
                    name=viz_layer,
                    interactive=False,
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
