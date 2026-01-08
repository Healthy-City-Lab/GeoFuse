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
                status.text("Running Engine...")
                try:
                    engine = NDVIEngine()
                    out = os.path.join(output_dir, f"ndvi_{target_date}.tif")
                    if engine.export_geotiff(
                        temp_path, str(target_date), out, tolerance=tolerance
                    ):
                        status.success(f"Saved: {out}")
                    else:
                        status.error("No images found.")
                except Exception as e:
                    st.error(str(e))
    with col_b:
        m = folium.Map(location=[51.0447, -114.0719], zoom_start=10)
        if ndvi_file:
            try:
                ndvi_file.seek(0)
                gdf = gpd.read_file(ndvi_file).to_crs(epsg=4326)
                minx, miny, maxx, maxy = gdf.total_bounds
                center_y = (miny + maxy) / 2
                center_x = (minx + maxx) / 2
                m.fit_bounds([[miny, minx], [maxy, maxx]])
                folium.GeoJson(gdf).add_to(m)
            except Exception as e:
                st.error(f"Map Error: {e}")
        st_folium(m, width=800, height=500, returned_objects=[])

# -----------------------------------------------------------------------------
# TAB 3: GVI CONFIGURATOR (Batch Processing)
# -----------------------------------------------------------------------------
with tab3:
    st.header("Street View & Vision Pipeline")
    col_a, col_b = st.columns([1, 2])
    m_gvi = folium.Map(location=[51.0447, -114.0719], zoom_start=11)

    # --- SESSION STATE INITIALIZATION ---
    # Structure: "datasets" = { filename: { "raw": gdf, "processed": gdf, "accumulated": list, "results": gdf, ... } }
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}
    if "master_cache" not in st.session_state:
        st.session_state.master_cache = {}
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

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

    # --- CACHED ENGINE LOADER ---
    @st.cache_resource
    def get_gvi_engine(model_path, device, api_key):
        return GVIEngine(model_path=model_path, device=device, api_key=api_key)

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

    with col_a:
        mode = st.radio(
            "Download Mode",
            ["Package (Scraper)", "API (Google Key)"],
            disabled=st.session_state.is_processing,
        )
        api_key = (
            st.text_input(
                "Google API Key",
                type="password",
                disabled=st.session_state.is_processing,
            )
            if mode == "API (Google Key)"
            else None
        )
        gvi_res = st.slider(
            "Grid Resolution (meters)",
            20,
            500,
            50,
            disabled=st.session_state.is_processing,
        )
        save_debug = st.checkbox(
            "Save Raw Images & Masks?",
            value=False,
            disabled=st.session_state.is_processing,
        )

        # 1. UNIFIED FILE MANAGER
        uploaded_files = st.file_uploader(
            "Upload Study Areas",
            accept_multiple_files=True,
            key="gvi_up",
            disabled=st.session_state.is_processing,
        )

        if uploaded_files:
            current_names = [f.name for f in uploaded_files]
            # Remove Deleted
            for k in list(st.session_state.datasets.keys()):
                if k not in current_names:
                    del st.session_state.datasets[k]
            # Add New
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
                        # Init with 'accumulated' list for partial results
                        st.session_state.datasets[f.name] = {
                            "raw": raw,
                            "processed": None,
                            "accumulated": [],
                            "results": None,
                            "meta": None,
                            "type": gtype,
                        }
                    except Exception as e:
                        st.error(f"Error loading {f.name}: {e}")
        else:
            st.session_state.datasets = {}

        # 2. GRID GENERATION
        if st.session_state.datasets:
            st.caption(f"Loaded {len(st.session_state.datasets)} datasets.")

            if st.button(
                "Generate Sampling Grids for All",
                disabled=st.session_state.is_processing,
            ):
                with st.spinner("Processing geometries..."):
                    for fname, d in st.session_state.datasets.items():
                        if d["type"] == "poly":
                            pts, meta = generate_raster_grid(d["raw"], gvi_res)
                            d["processed"] = pts
                            d["meta"] = meta
                        else:
                            d["processed"] = d["raw"].copy()
                            d["meta"] = None

                        # Reset results on new grid generation
                        d["accumulated"] = []
                        d["results"] = None

                    st.success("Grids generated!")

            # Draw Layers
            all_bounds = [
                d["raw"].total_bounds for d in st.session_state.datasets.values()
            ]
            if all_bounds:
                b = [
                    min(x[0] for x in all_bounds),
                    min(x[1] for x in all_bounds),
                    max(x[2] for x in all_bounds),
                    max(x[3] for x in all_bounds),
                ]
                m_gvi.fit_bounds([[b[1], b[0]], [b[3], b[2]]])

                for fname, d in st.session_state.datasets.items():
                    folium.GeoJson(
                        d["raw"],
                        name=f"{fname} (AOI)",
                        style_function=lambda x: {
                            "color": "blue",
                            "fill": False,
                            "weight": 2,
                        },
                    ).add_to(m_gvi)

                    if d["processed"] is not None:
                        preview = d["processed"].iloc[:1000]
                        for _, row in preview.iterrows():
                            folium.CircleMarker(
                                [row.geometry.y, row.geometry.x],
                                radius=1,
                                color="red",
                                fill=True,
                                fill_opacity=0.6,
                            ).add_to(m_gvi)

    # 3. RUN BATCH ANALYSIS
    if st.button(
        "🚀 Run Batch Analysis", type="primary", disabled=st.session_state.is_processing
    ):
        if not st.session_state.datasets:
            st.warning("No datasets loaded.")
        else:
            # RESET for fresh run (only clear if user manually clicked Run)
            for d in st.session_state.datasets.values():
                d["results"] = None
                d["accumulated"] = []  # Clear previous accumulated

            st.session_state.is_processing = True
            st.rerun()

    # --- PROCESSING LOOP ---
    if st.session_state.is_processing:
        if st.button("🛑 Abort Processing"):
            st.session_state.is_processing = False
            st.rerun()

        try:
            with st.status("Batch Analysis in Progress...", expanded=True) as status:
                model_path = os.path.join(
                    parent_dir, "geofuse", "model", "best_model.pth"
                )
                engine = get_gvi_engine(
                    model_path=model_path, device="cuda", api_key=api_key
                )

                for i, (fname, d) in enumerate(st.session_state.datasets.items()):
                    # Skip if marked as fully done
                    if d["results"] is not None:
                        st.info(f"Skipping {fname} (Completed)")
                        continue

                    if d["processed"] is None:
                        st.warning(f"Skipping {fname} (No grid generated)")
                        continue

                    st.write(f"Analyzing **{fname}**...")

                    # --- RESUME LOGIC ---
                    # 1. Determine where to start based on saved partials
                    current_count = len(d["accumulated"])
                    total_points = len(d["processed"])

                    # 2. Callback to save data IMMEDIATELY
                    def save_point_callback(data_dict):
                        d["accumulated"].append(data_dict)

                    # 3. UI Progress
                    file_progress = st.progress(0, text="Starting...")

                    def update_bar(curr, total):
                        pct = min(curr / total, 1.0)
                        file_progress.progress(
                            pct, text=f"Processing Point {curr}/{total}"
                        )

                    # 4. Call Engine with Start Index
                    # If current_count > 0, engine skips that many
                    engine.run_analysis(
                        d["processed"],
                        step=gvi_res,
                        folder=output_dir,
                        save_panos=save_debug,
                        save_masks=save_debug,
                        external_cache=st.session_state.master_cache,
                        progress_callback=update_bar,
                        result_callback=save_point_callback,  # Pass the saver
                        start_index=current_count,  # Pass the resume index
                    )

                    file_progress.empty()

                    # 5. Build Final DataFrame from ALL accumulated data
                    if d["accumulated"]:
                        results = gpd.GeoDataFrame(
                            d["accumulated"], crs=d["processed"].crs
                        )
                        if "orig_index" in results.columns:
                            results.set_index("orig_index", inplace=True)
                            results.index.name = None

                        st.session_state.datasets[fname]["results"] = results

                        # Save to Disk
                        out_name = os.path.splitext(fname)[0]
                        results.to_file(
                            os.path.join(output_dir, f"{out_name}_gvi.geojson"),
                            driver="GeoJSON",
                        )

                        if d["meta"]:
                            import rasterio
                            from rasterio.transform import rowcol

                            meta = d["meta"]
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

                            valid = results.dropna(subset=["gvi_veg"])
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

                status.update(
                    label="Batch Analysis Complete!", state="complete", expanded=False
                )
            st.success(f"Outputs saved to {output_dir}")

        except Exception as e:
            st.error(f"Analysis failed: {e}")

        st.session_state.is_processing = False
        st.rerun()

    # 4. RESULTS INSPECTOR
    st.divider()
    st.subheader("🔍 Results Inspector")

    completed_ds = [
        k for k, v in st.session_state.datasets.items() if v["results"] is not None
    ]

    if completed_ds:
        options = ["All Regions"] + completed_ds
        selected_option = st.selectbox("Select Area to Inspect", options)

        if selected_option == "All Regions":
            targets = completed_ds
        else:
            targets = [selected_option]

        viz_layer = st.radio(
            "Visualization Layer",
            ["Points", "Vegetation Raster", "Terrain Raster"],
            horizontal=True,
            key="viz_radio",
        )

        r_opacity = 0.7
        if "Raster" in viz_layer:
            r_opacity = st.slider("Opacity", 0.0, 1.0, 0.7)

        for ds_name in targets:
            ds = st.session_state.datasets[ds_name]

            if viz_layer == "Points":
                valid_pts = ds["results"].dropna(subset=["gvi_veg"])
                if len(targets) == 1:
                    st.caption(f"Showing {len(valid_pts)} points for {ds_name}.")

                folium.GeoJson(
                    valid_pts,
                    name=f"{ds_name} Points",
                    marker=folium.Circle(
                        radius=3, fill_color="green", fill_opacity=1, color=None
                    ),
                    tooltip=folium.GeoJsonTooltip(
                        fields=["gvi_veg", "gvi_ter", "pano_id"],
                        aliases=["Veg:", "Ter:", "ID:"],
                    ),
                ).add_to(m_gvi)

            elif "Raster" in viz_layer:
                if not ds["meta"]:
                    if len(targets) == 1:
                        st.warning(f"No grid for {ds_name} (Point input).")
                    continue

                from rasterio.transform import array_bounds, rowcol
                import io, base64
                from PIL import Image as PILImage

                meta = ds["meta"]
                arr = np.full((meta["height"], meta["width"]), np.nan)
                col_name = "gvi_veg" if "Vegetation" in viz_layer else "gvi_ter"

                res = ds["results"].dropna(subset=[col_name])
                if not res.empty:
                    rows, cols = rowcol(
                        meta["transform"], res.geometry.x.values, res.geometry.y.values
                    )
                    mask_idx = (
                        (rows >= 0)
                        & (rows < meta["height"])
                        & (cols >= 0)
                        & (cols < meta["width"])
                    )
                    arr[rows[mask_idx], cols[mask_idx]] = res[col_name].values[mask_idx]

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
                    mask = np.isnan(arr)
                    colored[..., 3] = np.where(mask, 0, r_opacity)

                    img_bytes = (colored * 255).astype(np.uint8)
                    im = PILImage.fromarray(img_bytes)
                    buff = io.BytesIO()
                    im.save(buff, format="PNG")
                    img_url = f"data:image/png;base64,{base64.b64encode(buff.getvalue()).decode()}"

                    left, bottom, right, top = array_bounds(
                        meta["height"], meta["width"], meta["transform"]
                    )
                    folium.raster_layers.ImageOverlay(
                        image=img_url,
                        bounds=[[bottom, left], [top, right]],
                        opacity=r_opacity,
                        interactive=False,
                        name=f"{ds_name} Raster",
                    ).add_to(m_gvi)

    with col_b:
        st_folium(m_gvi, width=800, height=500, returned_objects=[])

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
