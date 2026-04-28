import glob
import io
import os
import threading
import uuid
from datetime import date, datetime, timedelta

import base64
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from PIL import Image as PILImage
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_folium import st_folium

from geofuse.ndvi import NDVIEngine
from helpers import load_clean_gdf


# ---------------------------------------------------------------------------
# Background workers (module-level so they can be pickled / called in threads)
# ---------------------------------------------------------------------------


def _ndvi_worker(
    job_id,
    fname,
    dataset_data,
    start_date,
    end_date,
    cloud_pct,
    resolution,
    output_name,
    output_dir,
    job_tracker_dict,
):
    try:
        job_tracker_dict[job_id]["status"] = "Initializing Earth Engine..."
        engine = NDVIEngine()
        job_tracker_dict[job_id]["status"] = "Downloading and processing..."
        result = engine.download_and_process(
            geometry=dataset_data["raw"],
            start_date=start_date,
            end_date=end_date,
            output_name=output_name,
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


def _ndvi_column_worker(
    job_id,
    fname,
    dataset_data,
    date_column,
    window_days,
    cloud_pct,
    resolution,
    output_dir,
    job_tracker_dict,
):
    """Extract NDVI for each entity using a per-entity date from an attribute column."""
    try:
        gdf = dataset_data["raw"].copy()
        gdf["_parsed_date"] = pd.to_datetime(gdf[date_column], errors="coerce")
        gdf = gdf.dropna(subset=["_parsed_date"])
        if gdf.empty:
            job_tracker_dict[job_id]["status"] = (
                "Error: No valid dates found in the selected column."
            )
            return

        unique_dates = sorted(gdf["_parsed_date"].dt.date.unique())
        n_dates = len(unique_dates)
        engine = NDVIEngine()
        full_extent = gpd.GeoDataFrame(
            {"geometry": [gdf.geometry.union_all()]}, crs=gdf.crs
        )
        all_results = []

        for idx, target_date in enumerate(unique_dates):
            if job_tracker_dict[job_id]["cancel"]:
                job_tracker_dict[job_id]["status"] = "Cancelled"
                return

            start_d = target_date - timedelta(days=window_days)
            end_d = target_date + timedelta(days=window_days)
            date_str = target_date.strftime("%Y%m%d")
            base_name = fname.replace(".geojson", "")
            tmp_name = f"{base_name}_{date_str}_tmp"

            job_tracker_dict[job_id]["status"] = (
                f"Processing date {idx + 1}/{n_dates}: {target_date}"
            )
            job_tracker_dict[job_id]["progress"] = (idx + 0.5) / n_dates

            result = engine.download_and_process(
                geometry=full_extent,
                start_date=start_d.isoformat(),
                end_date=end_d.isoformat(),
                output_name=tmp_name,
                cloud_max=cloud_pct,
                resolution=resolution,
                folder=output_dir,
            )
            if result["status"] != "success":
                print(f"[NDVI column] {target_date} failed: {result.get('message')}")
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

        if all_results:
            merged = gpd.GeoDataFrame(
                pd.concat(all_results, ignore_index=True), crs=gdf.crs
            )
            merged = merged.drop(columns=["_parsed_date"], errors="ignore")
            base_name = fname.replace(".geojson", "")
            out_path = os.path.join(
                output_dir, f"{base_name}_temporal_ndvi.geojson"
            )
            merged.to_file(out_path, driver="GeoJSON")
            dataset_data["results"] = merged
            job_tracker_dict[job_id]["status"] = "Completed"
            job_tracker_dict[job_id]["progress"] = 1.0
        else:
            job_tracker_dict[job_id]["status"] = (
                "Completed — no valid NDVI data could be extracted."
            )
            job_tracker_dict[job_id]["progress"] = 1.0
    except Exception as e:
        job_tracker_dict[job_id]["status"] = f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# Tab render entry point
# ---------------------------------------------------------------------------


def render(output_dir: str) -> None:
    st.header("NDVI Sourcing")

    if "ndvi_datasets" not in st.session_state:
        st.session_state.ndvi_datasets = {}
    if "ndvi_inspector_select" not in st.session_state:
        st.session_state.ndvi_inspector_select = None
    if "ndvi_date_configs" not in st.session_state:
        st.session_state.ndvi_date_configs = {}

    # --- LAYOUT ---
    col_ndvi_top_left, col_ndvi_top_right = st.columns(2)
    with col_ndvi_top_left:
        st.subheader("1. Input & Date Configuration")

        cloud_pct = st.slider("Maximum Cloud Coverage (%)", 0, 100, 10)
        resolution = st.number_input("Resolution (m)", value=10, min_value=10)
        ndvi_files = st.file_uploader(
            "Upload Study Areas (GeoJSON)",
            accept_multiple_files=True,
            key="ndvi_up",
        )

        # Sync uploaded files with session state
        if ndvi_files is not None:
            current_names = [f.name for f in ndvi_files]
            for k in list(st.session_state.ndvi_datasets.keys()):
                ds = st.session_state.ndvi_datasets[k]
                if ds.get("type") == "restored":
                    continue
                if k not in current_names:
                    del st.session_state.ndvi_datasets[k]
                    st.session_state.ndvi_date_configs.pop(k, None)
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
                        st.error(f"Failed to load {f.name}: {e}")
        if ndvi_files == []:
            for k in list(st.session_state.ndvi_datasets.keys()):
                if st.session_state.ndvi_datasets[k].get("type") != "restored":
                    del st.session_state.ndvi_datasets[k]
                    st.session_state.ndvi_date_configs.pop(k, None)

        ndvi_input_datasets = {
            k: v
            for k, v in st.session_state.ndvi_datasets.items()
            if v.get("type") != "restored"
        }

        if ndvi_input_datasets:
            st.markdown("**Date Configuration**")
            for fname, d in ndvi_input_datasets.items():
                if fname not in st.session_state.ndvi_date_configs:
                    st.session_state.ndvi_date_configs[fname] = {
                        "use_ranges": True,
                        "use_specific": False,
                        "use_column": False,
                        "ranges": [(date(2023, 6, 1), date(2023, 9, 30))],
                        "specific_dates": [date(2023, 7, 15)],
                        "window_days_specific": 30,
                        "window_days_column": 30,
                    }
                cfg = st.session_state.ndvi_date_configs[fname]
                # Migrate old single-mode format
                if "mode" in cfg and "use_ranges" not in cfg:
                    old_mode = cfg.pop("mode")
                    cfg["use_ranges"] = old_mode == "Date Range"
                    cfg["use_specific"] = old_mode == "Specific Date"
                    cfg["use_column"] = old_mode == "Use Attribute Column"
                    cfg.setdefault("ranges", [(date(2023, 6, 1), date(2023, 9, 30))])
                    cfg.setdefault("specific_dates", [date(2023, 7, 15)])
                    cfg["window_days_specific"] = cfg.pop("window_days", 30)
                    cfg.setdefault("window_days_column", 30)

                today = date.today()

                with st.expander(fname, expanded=True):
                    chk_c1, chk_c2, chk_c3 = st.columns(3)
                    use_ranges = chk_c1.checkbox(
                        "Date Range(s)",
                        value=cfg.get("use_ranges", True),
                        key=f"ndvi_use_ranges_{fname}",
                    )
                    use_specific = chk_c2.checkbox(
                        "Specific Date(s)",
                        value=cfg.get("use_specific", False),
                        key=f"ndvi_use_specific_{fname}",
                    )
                    use_column = chk_c3.checkbox(
                        "Attribute Column",
                        value=cfg.get("use_column", False),
                        key=f"ndvi_use_col_{fname}",
                    )
                    cfg["use_ranges"] = use_ranges
                    cfg["use_specific"] = use_specific
                    cfg["use_column"] = use_column

                    if not any([use_ranges, use_specific, use_column]):
                        st.warning("Select at least one date mode.")

                    # ── Date Range(s) ─────────────────────────────────────────
                    if use_ranges:
                        st.markdown("**Date Range(s)**")
                        st.caption("One output file is produced per range.")
                        remove_idx = None
                        for i, (s, e) in enumerate(cfg["ranges"]):
                            stored_s = st.session_state.get(
                                f"ndvi_rs_{fname}_{i}", s
                            )
                            stored_e = st.session_state.get(
                                f"ndvi_re_{fname}_{i}", e
                            )
                            range_invalid = stored_s >= stored_e
                            s_help = (
                                "Start date is on or after the end date."
                                if range_invalid
                                else None
                            )
                            e_help = (
                                "End date is on or before the start date."
                                if range_invalid
                                else None
                            )
                            c1, c2, c3 = st.columns([4, 4, 1])
                            new_s = c1.date_input(
                                "Start",
                                value=s,
                                key=f"ndvi_rs_{fname}_{i}",
                                max_value=today,
                                label_visibility="collapsed",
                                help=s_help,
                            )
                            new_e = c2.date_input(
                                "End",
                                value=e,
                                key=f"ndvi_re_{fname}_{i}",
                                max_value=today,
                                label_visibility="collapsed",
                                help=e_help,
                            )
                            cfg["ranges"][i] = (new_s, new_e)
                            if new_s >= new_e:
                                st.markdown(
                                    '<p style="color:#ff4b4b;font-size:0.78em;'
                                    'margin:0 0 4px 0;">'
                                    "⚠ End date must be after start date.</p>",
                                    unsafe_allow_html=True,
                                )
                            if len(cfg["ranges"]) > 1:
                                if c3.button(
                                    "✕",
                                    key=f"ndvi_rrem_{fname}_{i}",
                                    help="Remove this range",
                                ):
                                    remove_idx = i
                        if remove_idx is not None:
                            old_n = len(cfg["ranges"])
                            cfg["ranges"].pop(remove_idx)
                            for j in range(remove_idx, old_n):
                                st.session_state.pop(f"ndvi_rs_{fname}_{j}", None)
                                st.session_state.pop(f"ndvi_re_{fname}_{j}", None)
                            st.rerun()
                        if st.button("Add Date Range", key=f"ndvi_radd_{fname}"):
                            cfg["ranges"].append((date(today.year, 1, 1), today))
                            st.rerun()

                    # ── Specific Date(s) ──────────────────────────────────────
                    if use_specific:
                        if use_ranges:
                            st.divider()
                        st.markdown("**Specific Date(s)**")
                        cfg["window_days_specific"] = st.number_input(
                            "Composite window (± days)",
                            min_value=7,
                            max_value=180,
                            value=cfg.get("window_days_specific", 30),
                            key=f"ndvi_win_s_{fname}",
                        )
                        st.caption(
                            "An NDVI composite is built from imagery within ± the "
                            "window above. One output file is produced per date."
                        )
                        remove_idx = None
                        for i, d_val in enumerate(cfg["specific_dates"]):
                            c1, c2 = st.columns([9, 1])
                            new_d = c1.date_input(
                                "Date of interest",
                                value=d_val,
                                key=f"ndvi_sd_{fname}_{i}",
                                max_value=today,
                                label_visibility="collapsed",
                            )
                            cfg["specific_dates"][i] = new_d
                            if len(cfg["specific_dates"]) > 1:
                                if c2.button(
                                    "✕",
                                    key=f"ndvi_srem_{fname}_{i}",
                                    help="Remove this date",
                                ):
                                    remove_idx = i
                        if remove_idx is not None:
                            old_n = len(cfg["specific_dates"])
                            cfg["specific_dates"].pop(remove_idx)
                            for j in range(remove_idx, old_n):
                                st.session_state.pop(f"ndvi_sd_{fname}_{j}", None)
                            st.rerun()
                        if st.button("Add Date", key=f"ndvi_sadd_{fname}"):
                            cfg["specific_dates"].append(today)
                            st.rerun()

                    # ── Attribute Column ──────────────────────────────────────
                    if use_column:
                        if use_ranges or use_specific:
                            st.divider()
                        st.markdown("**Attribute Column**")
                        attr_cols = [
                            c for c in d["raw"].columns if c.lower() != "geometry"
                        ]
                        if attr_cols:
                            col_sel = st.selectbox(
                                "Date attribute column",
                                attr_cols,
                                key=f"ndvi_col_{fname}",
                            )
                            cfg["window_days_column"] = st.number_input(
                                "Composite window (± days)",
                                min_value=7,
                                max_value=180,
                                value=cfg.get("window_days_column", 30),
                                key=f"ndvi_win_c_{fname}",
                            )
                            st.caption(
                                f"NDVI is sampled per entity using its date from "
                                f"'{col_sel}'. Produces a single merged output file."
                            )
                        else:
                            st.warning("No attribute columns found in this file.")

        st.divider()

        if st.button("🚀 Run NDVI Analysis", type="primary"):
            if not ndvi_input_datasets:
                st.warning(
                    "No input areas loaded. Upload at least one GeoJSON file."
                )
            else:
                if "jobs" not in st.session_state:
                    st.session_state.jobs = {}
                jobs_started = 0
                validation_errors = []

                for fname, d in ndvi_input_datasets.items():
                    cfg = st.session_state.ndvi_date_configs.get(fname, {})
                    base_name = fname.replace(".geojson", "")

                    use_ranges = st.session_state.get(
                        f"ndvi_use_ranges_{fname}", cfg.get("use_ranges", True)
                    )
                    use_specific = st.session_state.get(
                        f"ndvi_use_specific_{fname}", cfg.get("use_specific", False)
                    )
                    use_column = st.session_state.get(
                        f"ndvi_use_col_{fname}", cfg.get("use_column", False)
                    )

                    if not any([use_ranges, use_specific, use_column]):
                        validation_errors.append(
                            f"{fname}: No date mode is enabled."
                        )
                        continue

                    # --- Date Range jobs ---
                    if use_ranges:
                        for i, (s_def, e_def) in enumerate(
                            cfg.get(
                                "ranges", [(date(2023, 6, 1), date(2023, 9, 30))]
                            )
                        ):
                            start_d = st.session_state.get(
                                f"ndvi_rs_{fname}_{i}", s_def
                            )
                            end_d = st.session_state.get(
                                f"ndvi_re_{fname}_{i}", e_def
                            )
                            if start_d >= end_d:
                                validation_errors.append(
                                    f"{fname}: Date range {i + 1} — "
                                    "start date must be before end date."
                                )
                                continue
                            output_name = (
                                f"{base_name}_"
                                f"{start_d.strftime('%Y%m%d')}_"
                                f"{end_d.strftime('%Y%m%d')}"
                            )
                            job_id = str(uuid.uuid4())[:8]
                            st.session_state.jobs[job_id] = {
                                "fname": fname,
                                "name": f"{base_name} ({start_d} → {end_d})",
                                "task": "NDVI",
                                "type": "ndvi",
                                "start_time": datetime.now().strftime("%H:%M:%S"),
                                "progress": 0.0,
                                "status": "Queued",
                                "cancel": False,
                            }
                            t = threading.Thread(
                                target=_ndvi_worker,
                                args=(
                                    job_id,
                                    fname,
                                    d,
                                    start_d.isoformat(),
                                    end_d.isoformat(),
                                    cloud_pct,
                                    resolution,
                                    output_name,
                                    output_dir,
                                    st.session_state.jobs,
                                ),
                            )
                            add_script_run_ctx(t)
                            t.start()
                            jobs_started += 1

                    # --- Specific Date jobs ---
                    if use_specific:
                        window_days = st.session_state.get(
                            f"ndvi_win_s_{fname}",
                            cfg.get("window_days_specific", 30),
                        )
                        for i, d_def in enumerate(
                            cfg.get("specific_dates", [date(2023, 7, 15)])
                        ):
                            target_date = st.session_state.get(
                                f"ndvi_sd_{fname}_{i}", d_def
                            )
                            start_d = target_date - timedelta(days=window_days)
                            end_d = min(
                                target_date + timedelta(days=window_days),
                                date.today(),
                            )
                            output_name = (
                                f"{base_name}_{target_date.strftime('%Y%m%d')}"
                            )
                            job_id = str(uuid.uuid4())[:8]
                            st.session_state.jobs[job_id] = {
                                "fname": fname,
                                "name": f"{base_name} (near {target_date})",
                                "task": "NDVI",
                                "type": "ndvi",
                                "start_time": datetime.now().strftime("%H:%M:%S"),
                                "progress": 0.0,
                                "status": "Queued",
                                "cancel": False,
                            }
                            t = threading.Thread(
                                target=_ndvi_worker,
                                args=(
                                    job_id,
                                    fname,
                                    d,
                                    start_d.isoformat(),
                                    end_d.isoformat(),
                                    cloud_pct,
                                    resolution,
                                    output_name,
                                    output_dir,
                                    st.session_state.jobs,
                                ),
                            )
                            add_script_run_ctx(t)
                            t.start()
                            jobs_started += 1

                    # --- Attribute Column job ---
                    if use_column:
                        date_col = st.session_state.get(f"ndvi_col_{fname}")
                        window_days = st.session_state.get(
                            f"ndvi_win_c_{fname}",
                            cfg.get("window_days_column", 30),
                        )
                        if not date_col:
                            validation_errors.append(
                                f"{fname}: No date column selected."
                            )
                        else:
                            job_id = str(uuid.uuid4())[:8]
                            st.session_state.jobs[job_id] = {
                                "fname": fname,
                                "name": f"{base_name} (by column: {date_col})",
                                "task": "NDVI",
                                "type": "ndvi",
                                "start_time": datetime.now().strftime("%H:%M:%S"),
                                "progress": 0.0,
                                "status": "Queued",
                                "cancel": False,
                            }
                            t = threading.Thread(
                                target=_ndvi_column_worker,
                                args=(
                                    job_id,
                                    fname,
                                    d,
                                    date_col,
                                    window_days,
                                    cloud_pct,
                                    resolution,
                                    output_dir,
                                    st.session_state.jobs,
                                ),
                            )
                            add_script_run_ctx(t)
                            t.start()
                            jobs_started += 1

                for err in validation_errors:
                    st.error(err)

                if jobs_started:
                    st.success(
                        f"{jobs_started} job(s) started. "
                        "Monitor progress in the sidebar."
                    )
                elif not validation_errors:
                    st.info("No new jobs were submitted.")

    with col_ndvi_top_right:
        st.subheader("Area of Interest Preview")
        m_ndvi_input = folium.Map(location=[51.0447, -114.0719], zoom_start=10)
        all_bounds = []
        for fname, d in st.session_state.ndvi_datasets.items():
            if d.get("type") == "restored":
                continue
            if d.get("raw") is not None:
                folium.GeoJson(
                    d["raw"],
                    name=fname,
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
        if st.button("Scan Output Folder"):
            found_files = glob.glob(os.path.join(output_dir, "*_ndvi.geojson"))
            count = 0
            for p in found_files:
                base_name = os.path.basename(p).replace("_ndvi.geojson", "")
                tif_path = os.path.join(output_dir, f"{base_name}_ndvi.tif")
                has_tif = os.path.exists(tif_path)
                if base_name not in st.session_state.ndvi_datasets:
                    try:
                        gdf = gpd.read_file(p)
                        meta = None
                        if has_tif:
                            with rasterio.open(tif_path) as src:
                                meta = {
                                    "transform": src.transform,
                                    "width": src.width,
                                    "height": src.height,
                                    "crs": src.crs,
                                }
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
                st.success(f"Loaded {count} result(s) from the output folder.")
            else:
                st.info("No processed results found in the output folder.")

        completed_ds = [
            k
            for k, v in st.session_state.ndvi_datasets.items()
            if v.get("type") == "restored"
        ]
        options = ["All Regions"] + completed_ds
        selected_opt = st.selectbox(
            "Select result",
            options,
            index=None,
            placeholder="Choose a result...",
            key="ndvi_inspector_select",
        )
        r_opacity = st.slider("Raster Opacity", 0.0, 1.0, 0.7, key="ndvi_op")
        show_points = st.checkbox("Show Sample Points", value=False)
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
                            arr = src.read(1)

                            from rasterio.warp import transform_bounds

                            bounds_native = src.bounds
                            src_crs = src.crs
                            bounds_4326 = transform_bounds(
                                src_crs, "EPSG:4326", *bounds_native
                            )
                            folium_bounds = [
                                [bounds_4326[1], bounds_4326[0]],
                                [bounds_4326[3], bounds_4326[2]],
                            ]
                            res_bounds.append(
                                [
                                    bounds_4326[0],
                                    bounds_4326[1],
                                    bounds_4326[2],
                                    bounds_4326[3],
                                ]
                            )

                            norm_data = np.clip((arr - (-0.2)) / (1.0 - (-0.2)), 0, 1)
                            cmap = plt.get_cmap("RdYlGn")
                            colored = cmap(norm_data)
                            mask = (arr == -9999) | np.isnan(arr) | (arr == 0)
                            colored[..., 3] = np.where(mask, 0, r_opacity)

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
                                bounds=folium_bounds,
                                opacity=r_opacity,
                                interactive=True,
                            ).add_to(m_ndvi_result)
                            folium.Rectangle(
                                bounds=folium_bounds,
                                color="red",
                                weight=2,
                                fill=False,
                            ).add_to(m_ndvi_result)
                    except Exception as e:
                        print(f"Viz Error {ds_name}: {e}")

                if show_points:
                    if ds.get("results") is not None:
                        try:
                            gdf_pts = ds["results"]
                            if len(gdf_pts) > 5000:
                                st.warning(
                                    f"{ds_name}: Displaying a 5,000-point sample."
                                )
                                gdf_pts = gdf_pts.sample(5000)
                            tooltip_fields = (
                                ["NDVI", "ndvi_date"]
                                if "ndvi_date" in gdf_pts.columns
                                else ["NDVI"]
                            )
                            tooltip_aliases = (
                                ["NDVI:", "Date:"]
                                if "ndvi_date" in gdf_pts.columns
                                else ["NDVI:"]
                            )
                            folium.GeoJson(
                                gdf_pts,
                                marker=folium.Circle(
                                    radius=1,
                                    color="blue",
                                    fill=True,
                                    fill_opacity=1,
                                ),
                                tooltip=folium.GeoJsonTooltip(
                                    fields=tooltip_fields,
                                    aliases=tooltip_aliases,
                                ),
                            ).add_to(m_ndvi_result)
                        except Exception as e:
                            st.error(f"Error rendering sample points: {e}")

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
                                        f"**{ds_name}**: NDVI = {pixel_val:.4f} "
                                        f"at ({lat:.4f}, {lon:.4f})"
                                    )
                                    val_found = True
                                    break
                        except Exception:
                            pass
            if not val_found:
                val_container.info(
                    f"No valid NDVI data at ({lat:.4f}, {lon:.4f})."
                )
