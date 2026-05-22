import base64
import gc
import glob
import io
import os
from datetime import date, timedelta

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
from helpers import (
    RESTART_SESSION_KEY,
    apply_buffer_m,
    load_vector_upload_sessions,
    render_job_restart_panel,
)
from map_preview import (
    add_black_point_layer,
    add_study_area_layers,
    add_uniform_point_layer,
    trim_point_gdf_for_display,
)
from PIL import Image as PILImage
from shapely.geometry import box as shapely_box
from streamlit_folium import st_folium

from geofuse.crs_utils import reproject_geodataframe_to_wgs84
from geofuse.jobs.runners import run_ndvi, run_ndvi_column
from geofuse.vector_io import geometry_sha256

# ---------------------------------------------------------------------------
# Tab render entry point
# ---------------------------------------------------------------------------


def _ndvi_scan_outputs(output_dir: str) -> dict[str, dict]:
    """Discover NDVI result files and return ``{base_name: dataset_dict}``.

    Pure function — no Streamlit calls — so it can run safely in a background
    thread while workers keep processing.
    """
    from rasterio.warp import transform_bounds

    base_names: set[str] = set()
    for pat in ("*_ndvi.tif", "*_ndvi.tiff", "*_ndvi.geojson", "*_ndvi.gpkg"):
        for p in glob.glob(os.path.join(output_dir, pat)):
            stem = os.path.basename(p).rsplit(".", 1)[0]
            base_names.add(stem.removesuffix("_ndvi"))

    found: dict[str, dict] = {}
    for base_name in sorted(base_names):
        tif_path = next(
            (
                os.path.join(output_dir, f"{base_name}_ndvi{ext}")
                for ext in (".tif", ".tiff")
                if os.path.isfile(os.path.join(output_dir, f"{base_name}_ndvi{ext}"))
            ),
            None,
        )
        gpkg_path = os.path.join(output_dir, f"{base_name}_ndvi.gpkg")
        geojson_path = os.path.join(output_dir, f"{base_name}_ndvi.geojson")

        try:
            meta: dict | None = None
            b = crs = None
            if tif_path:
                with rasterio.open(tif_path) as src:
                    meta = {
                        "transform": src.transform,
                        "width": src.width,
                        "height": src.height,
                        "crs": src.crs,
                    }
                    b = src.bounds
                    crs = src.crs

            results_gdf = None
            if os.path.isfile(gpkg_path):
                results_gdf = reproject_geodataframe_to_wgs84(gpd.read_file(gpkg_path))
            elif os.path.isfile(geojson_path):
                results_gdf = reproject_geodataframe_to_wgs84(
                    gpd.read_file(geojson_path)
                )

            if results_gdf is not None and not results_gdf.empty:
                raw_geom = results_gdf.geometry.union_all().envelope
                raw_gdf = gpd.GeoDataFrame({"geometry": [raw_geom]}, crs="EPSG:4326")
            elif tif_path:
                w, s, e, n = transform_bounds(crs, "EPSG:4326", *b)
                raw_gdf = gpd.GeoDataFrame(
                    {"geometry": [shapely_box(w, s, e, n)]},
                    crs="EPSG:4326",
                )
            else:
                continue

            found[base_name] = {
                "raw": raw_gdf,
                "processed": None,
                "results": results_gdf,
                "meta": meta,
                "type": "restored",
            }
        except Exception as e:
            print(f"Error loading {base_name}: {e}")
    return found


def _ndvi_restart_summary_lines(p: dict) -> list[str]:
    mode = p.get("mode", "?")
    lines = [
        f"**Original file:** `{p.get('fname', '?')}`",
        f"**Mode:** {mode}",
    ]
    if mode == "range":
        lines.append(
            f"**Date range:** {p.get('start_date', '?')} → " f"{p.get('end_date', '?')}"
        )
    elif mode == "specific":
        lines.append(
            f"**Target date:** {p.get('target_date', '?')} "
            f"(±{p.get('window_days', '?')} d)"
        )
    elif mode == "column":
        lines.append(
            f"**Date column:** `{p.get('date_column', '?')}` "
            f"(±{p.get('window_days', '?')} d)"
        )
    lines.append(
        f"**Cloud max:** {p.get('cloud_pct', '?')}% · "
        f"**Resolution:** {p.get('resolution', '?')} m · "
        f"**Buffer:** {p.get('buffer_m', '?')} m"
    )
    lines.append(
        f"**Outputs:** GeoTIFF={bool(p.get('save_geotiff'))} · "
        f"GeoPackage={bool(p.get('save_gpkg'))} · "
        f"GeoJSON={bool(p.get('save_geojson'))}"
    )
    return lines


def _render_ndvi_restart_panel(store, executor, output_dir) -> None:
    """Show the restart workflow when the user clicked ↻ on an NDVI job."""
    job_id = st.session_state.get(RESTART_SESSION_KEY)
    if not job_id:
        return
    rec = store.get(job_id)
    if rec is None:
        return
    if rec.type == "gvi":
        st.info(
            "A restart is pending for a GVI job. Switch to the **GVI "
            "Sourcing** tab to complete it."
        )
        return
    if rec.type not in ("ndvi", "ndvi_column"):
        return

    p = rec.params or {}

    def _on_confirm(gdf, fname_new: str, _extras: dict) -> None:
        fname = fname_new
        # Stage the verified GDF in the NDVI tab's dataset registry.
        try:
            gtype = (
                "poly"
                if gdf.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]
                else "point"
            )
        except Exception:
            gtype = "point"
        if "ndvi_datasets" not in st.session_state:
            st.session_state.ndvi_datasets = {}
        st.session_state.ndvi_datasets[fname] = {"raw": gdf, "type": gtype}

        new_params = dict(p)
        new_params["geometry_sha256"] = geometry_sha256(gdf)
        new_params["restart_of"] = rec.id
        new_params["fname"] = fname

        base_name = os.path.splitext(fname)[0]
        record = store.submit(type=rec.type, name=base_name, params=new_params)

        common = {
            "fname": fname,
            "dataset_data": st.session_state.ndvi_datasets[fname],
            "cloud_pct": int(p.get("cloud_pct", 10)),
            "resolution": int(p.get("resolution", 10)),
            "buffer_m": int(p.get("buffer_m", 0)),
            "output_dir": output_dir,
            "save_geotiff": bool(p.get("save_geotiff", True)),
            "save_geojson": bool(p.get("save_geojson", False)),
            "save_gpkg": bool(p.get("save_gpkg", False)),
        }

        if rec.type == "ndvi":
            executor.submit_runner(
                record,
                run_ndvi,
                start_date=str(p.get("start_date", "")),
                end_date=str(p.get("end_date", "")),
                output_name=str(p.get("output_name", base_name)),
                **common,
            )
        else:  # ndvi_column
            executor.submit_runner(
                record,
                run_ndvi_column,
                date_column=str(p.get("date_column", "")),
                window_days=int(p.get("window_days", 30)),
                **common,
            )

    render_job_restart_panel(
        rec,
        accept_types=[
            "geojson",
            "json",
            "gpkg",
            "shp",
            "dbf",
            "shx",
            "prj",
            "cpg",
            "zip",
        ],
        summary_lines=_ndvi_restart_summary_lines(p),
        extra_inputs_renderer=None,
        on_confirm=_on_confirm,
    )


def _ndvi_size_hint(buffer_m: int, resolution_m: int) -> None:
    """Inline banner: estimated pixel count and GeoTIFF tile expectations.

    Driven by the bbox of all uploaded NDVI datasets (already in EPSG:4326),
    buffered by ``buffer_m``. Cheap heuristic — uses planar approximation at
    each dataset's centroid latitude.
    """
    datasets = st.session_state.get("ndvi_datasets") or {}
    if resolution_m <= 0:
        return
    bbox_minx = bbox_miny = float("inf")
    bbox_maxx = bbox_maxy = float("-inf")
    total_area_m2 = 0.0
    have_data = False
    for d in datasets.values():
        raw = d.get("raw") if isinstance(d, dict) else None
        if raw is None or len(raw) == 0:
            continue
        minx, miny, maxx, maxy = raw.total_bounds
        if not np.isfinite([minx, miny, maxx, maxy]).all():
            continue
        have_data = True
        bbox_minx = min(bbox_minx, minx)
        bbox_miny = min(bbox_miny, miny)
        bbox_maxx = max(bbox_maxx, maxx)
        bbox_maxy = max(bbox_maxy, maxy)
        cent_lat = (miny + maxy) / 2.0
        m_per_deg_lat = 111000.0
        m_per_deg_lon = 111000.0 * float(np.cos(np.radians(cent_lat)))
        # bbox area in metres (NDVI rasterizes the bbox of the buffered geom)
        width_m = (maxx - minx) * m_per_deg_lon + 2 * max(buffer_m, 0)
        height_m = (maxy - miny) * m_per_deg_lat + 2 * max(buffer_m, 0)
        total_area_m2 += max(0.0, width_m) * max(0.0, height_m)
    if not have_data:
        return

    est_cells = int(total_area_m2 / (resolution_m * resolution_m))
    span_lon = bbox_maxx - bbox_minx
    span_lat = bbox_maxy - bbox_miny
    msgs: list[str] = []
    if est_cells > 0:
        msgs.append(f"Estimated NDVI pixels: ~{est_cells:,}")
    if est_cells > 1_000_000:
        msgs.append(
            "Vector output of every pixel would be very large — prefer GeoTIFF, "
            "or use GeoPackage instead of GeoJSON if you need vector samples."
        )
    if span_lon > 6.0 or span_lat > 6.0:
        msgs.append(
            f"Study area spans {span_lon:.1f}° lon × {span_lat:.1f}° lat — "
            "Earth Engine downloads will be tiled automatically."
        )
    if msgs:
        st.info(" · ".join(msgs))


def render(output_dir: str) -> None:
    st.header("NDVI")

    if "ndvi_datasets" not in st.session_state:
        st.session_state.ndvi_datasets = {}
    if "ndvi_inspector_select" not in st.session_state:
        st.session_state.ndvi_inspector_select = None
    if "ndvi_date_configs" not in st.session_state:
        st.session_state.ndvi_date_configs = {}

    from services import get_job_executor, get_job_store

    _render_ndvi_restart_panel(get_job_store(), get_job_executor(), output_dir)

    st.subheader("Input Configuration")

    ndvi_files = st.file_uploader(
        "Upload Study Areas",
        accept_multiple_files=True,
        type=["geojson", "json", "gpkg", "shp", "dbf", "shx", "prj", "cpg", "zip"],
        key="ndvi_up",
        help=(
            "GeoJSON, GeoPackage, or zip. For Shapefile, select all parts together "
            "(.shp, .dbf, .shx; add .prj when you have it)."
        ),
    )

    # Sync uploaded files with session state
    if ndvi_files is not None:
        loaded = load_vector_upload_sessions(ndvi_files)
        logical_names = [name for name, _ in loaded]
        for k in list(st.session_state.ndvi_datasets.keys()):
            ds = st.session_state.ndvi_datasets[k]
            if ds.get("type") == "restored":
                continue
            if k not in logical_names:
                del st.session_state.ndvi_datasets[k]
                st.session_state.ndvi_date_configs.pop(k, None)
        for fname, raw in loaded:
            if fname not in st.session_state.ndvi_datasets:
                try:
                    st.session_state.ndvi_datasets[fname] = {
                        "raw": raw,
                        "processed": None,
                        "results": None,
                        "meta": None,
                        "type": "input",
                    }
                except Exception as e:
                    st.error(f"Failed to load {fname}: {e}")
        gc.collect()
    if ndvi_files == []:
        for k in list(st.session_state.ndvi_datasets.keys()):
            if st.session_state.ndvi_datasets[k].get("type") != "restored":
                del st.session_state.ndvi_datasets[k]
                st.session_state.ndvi_date_configs.pop(k, None)
        gc.collect()

    ndvi_input_datasets = {
        k: v
        for k, v in st.session_state.ndvi_datasets.items()
        if v.get("type") != "restored"
    }

    # ── Settings + map (form prevents per-keystroke reruns on sliders) ──────
    with st.form("ndvi_job_form"):
        ndvi_buf_preview = int(st.session_state.get("ndvi_buffer", 0))
        fc_ndvi_l, fc_ndvi_r = st.columns(2)
        with fc_ndvi_l:
            with st.container(border=True):
                st.slider(
                    "Maximum Cloud Coverage (%)",
                    0,
                    100,
                    10,
                    key="ndvi_cloud",
                    help=(
                        "Cloud mask threshold for Earth Engine. Together with resolution "
                        "and buffer, these apply when you press Run below."
                    ),
                )
                st.number_input(
                    "Resolution (m)",
                    value=10,
                    min_value=10,
                    key="ndvi_res",
                    help="Target pixel size for the NDVI raster export.",
                )
                st.slider(
                    "Download Buffer (m)",
                    min_value=0,
                    max_value=2000,
                    value=0,
                    step=50,
                    key="ndvi_buffer",
                    help="Expand the study area outward by this distance (metres) before download.",
                )
        with fc_ndvi_r:
            st.subheader("Study Area Preview")
            m_ndvi_input = folium.Map(location=[51.0447, -114.0719], zoom_start=10)
            all_bounds = []
            for fname, d in st.session_state.ndvi_datasets.items():
                if d.get("type") == "restored":
                    continue
                if d.get("raw") is not None:
                    add_study_area_layers(
                        m_ndvi_input,
                        d["raw"],
                        study_name=fname,
                        buffer_m=ndvi_buf_preview,
                        buffer_name=f"{fname} (buffer)",
                    )
                    all_bounds.append(d["raw"].total_bounds)
                    if ndvi_buf_preview > 0:
                        all_bounds.append(
                            apply_buffer_m(d["raw"], ndvi_buf_preview).total_bounds
                        )
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
        # Submit in the form so the buffer/cloud/res changes are committed before run.
        st.form_submit_button(
            "Apply Settings",
            use_container_width=False,
            help="Commit slider values before adjusting dates below.",
        )

    # ── Date configuration + output format + run ─────────────────────────────
    # These are outside the form because "Add / Remove" date buttons cannot live
    # inside a Streamlit form. They sit immediately below the settings+map
    # section so the page still reads top-to-bottom as one workflow.
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
                    remove_idx = None
                    for i, (s, e) in enumerate(cfg["ranges"]):
                        stored_s = st.session_state.get(f"ndvi_rs_{fname}_{i}", s)
                        stored_e = st.session_state.get(f"ndvi_re_{fname}_{i}", e)
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
                    if st.button(
                        "Add Date Range",
                        key=f"ndvi_radd_{fname}",
                        help="Each range row produces its own NDVI output file.",
                    ):
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
                        help=(
                            "Composite uses imagery within ± this many days around "
                            "each date of interest. One output file per date row."
                        ),
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
                    attr_cols = [c for c in d["raw"].columns if c.lower() != "geometry"]
                    if attr_cols:
                        col_sel = st.selectbox(
                            "Date attribute column",
                            attr_cols,
                            key=f"ndvi_col_{fname}",
                            help=(
                                "Per-feature dates from this column; composite uses "
                                "the window below. Produces one merged output for the layer."
                            ),
                        )
                        cfg["window_days_column"] = st.number_input(
                            "Composite window (± days)",
                            min_value=7,
                            max_value=180,
                            value=cfg.get("window_days_column", 30),
                            key=f"ndvi_win_c_{fname}",
                            help="± day window around each feature date for the Earth Engine composite.",
                        )
                    else:
                        st.warning("No attribute columns found in this file.")

    _ndvi_size_hint(
        int(st.session_state.get("ndvi_buffer", 0)),
        int(st.session_state.get("ndvi_res", 10)),
    )

    oc_ndvi_a, oc_ndvi_b, oc_ndvi_c = st.columns(3)
    with oc_ndvi_a:
        st.checkbox(
            "Save GeoTIFF",
            value=True,
            key="ndvi_out_geotiff",
            help="Raster NDVI surface (primary format for NDVI).",
        )
    with oc_ndvi_b:
        st.checkbox(
            "Save GeoPackage",
            value=False,
            key="ndvi_out_gpkg",
            help=(
                "Vector samples in EPSG:4326, single file, readable by every "
                "modern GIS. Recommended over GeoJSON for large outputs."
            ),
        )
    with oc_ndvi_c:
        st.checkbox(
            "Save GeoJSON",
            value=False,
            key="ndvi_out_geojson",
            help="Compatibility option only. Slow to read past ~100k points.",
        )
    run = st.button(
        "🚀 Run NDVI Analysis",
        type="primary",
        use_container_width=True,
        key="ndvi_run_btn",
    )

    cloud_pct = int(st.session_state.get("ndvi_cloud", 10))
    resolution = int(st.session_state.get("ndvi_res", 10))
    buffer_m = int(st.session_state.get("ndvi_buffer", 0))
    ndvi_out_ok = (
        st.session_state.get("ndvi_out_geotiff", True)
        or st.session_state.get("ndvi_out_gpkg", False)
        or st.session_state.get("ndvi_out_geojson", False)
    )

    if run:
        if not ndvi_out_ok:
            st.error("Select at least one output format.")
        elif not ndvi_input_datasets:
            st.warning("Upload at least one study area to get started.")
        else:
            from services import get_job_executor, get_job_store

            store = get_job_store()
            executor = get_job_executor()
            save_gt = st.session_state.get("ndvi_out_geotiff", True)
            save_gj = st.session_state.get("ndvi_out_geojson", False)
            save_gp = st.session_state.get("ndvi_out_gpkg", False)
            jobs_started = 0
            validation_errors = []

            for fname, d in ndvi_input_datasets.items():
                cfg = st.session_state.ndvi_date_configs.get(fname, {})
                # Strip *any* extension (.geojson / .shp / .gpkg / .zip / …)
                # so the monitor title is just the file stem.
                base_name = os.path.splitext(fname)[0]

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
                    validation_errors.append(f"{fname}: No date mode is enabled.")
                    continue

                # --- Date Range jobs ---
                if use_ranges:
                    for i, (s_def, e_def) in enumerate(
                        cfg.get("ranges", [(date(2023, 6, 1), date(2023, 9, 30))])
                    ):
                        start_d = st.session_state.get(f"ndvi_rs_{fname}_{i}", s_def)
                        end_d = st.session_state.get(f"ndvi_re_{fname}_{i}", e_def)
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
                        record = store.submit(
                            type="ndvi",
                            name=base_name,
                            params={
                                "fname": fname,
                                "mode": "range",
                                "start_date": start_d.isoformat(),
                                "end_date": end_d.isoformat(),
                                "cloud_pct": cloud_pct,
                                "resolution": resolution,
                                "buffer_m": buffer_m,
                                "output_name": output_name,
                                "save_geotiff": save_gt,
                                "save_gpkg": save_gp,
                                "save_geojson": save_gj,
                                "geometry_sha256": geometry_sha256(d["raw"]),
                            },
                        )
                        executor.submit_runner(
                            record,
                            run_ndvi,
                            fname=fname,
                            dataset_data=d,
                            start_date=start_d.isoformat(),
                            end_date=end_d.isoformat(),
                            cloud_pct=cloud_pct,
                            resolution=resolution,
                            buffer_m=buffer_m,
                            output_name=output_name,
                            output_dir=output_dir,
                            save_geotiff=save_gt,
                            save_gpkg=save_gp,
                            save_geojson=save_gj,
                        )
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
                        output_name = f"{base_name}_{target_date.strftime('%Y%m%d')}"
                        record = store.submit(
                            type="ndvi",
                            name=base_name,
                            params={
                                "fname": fname,
                                "mode": "specific",
                                "target_date": target_date.isoformat(),
                                "window_days": window_days,
                                "cloud_pct": cloud_pct,
                                "resolution": resolution,
                                "buffer_m": buffer_m,
                                "output_name": output_name,
                                "save_geotiff": save_gt,
                                "save_gpkg": save_gp,
                                "save_geojson": save_gj,
                                "geometry_sha256": geometry_sha256(d["raw"]),
                            },
                        )
                        executor.submit_runner(
                            record,
                            run_ndvi,
                            fname=fname,
                            dataset_data=d,
                            start_date=start_d.isoformat(),
                            end_date=end_d.isoformat(),
                            cloud_pct=cloud_pct,
                            resolution=resolution,
                            buffer_m=buffer_m,
                            output_name=output_name,
                            output_dir=output_dir,
                            save_geotiff=save_gt,
                            save_gpkg=save_gp,
                            save_geojson=save_gj,
                        )
                        jobs_started += 1

                # --- Attribute Column job ---
                if use_column:
                    date_col = st.session_state.get(f"ndvi_col_{fname}")
                    window_days = st.session_state.get(
                        f"ndvi_win_c_{fname}",
                        cfg.get("window_days_column", 30),
                    )
                    if not date_col:
                        validation_errors.append(f"{fname}: No date column selected.")
                    else:
                        record = store.submit(
                            type="ndvi_column",
                            name=base_name,
                            params={
                                "fname": fname,
                                "mode": "column",
                                "date_column": date_col,
                                "window_days": window_days,
                                "cloud_pct": cloud_pct,
                                "resolution": resolution,
                                "buffer_m": buffer_m,
                                "save_geotiff": save_gt,
                                "save_gpkg": save_gp,
                                "save_geojson": save_gj,
                                "geometry_sha256": geometry_sha256(d["raw"]),
                            },
                        )
                        executor.submit_runner(
                            record,
                            run_ndvi_column,
                            fname=fname,
                            dataset_data=d,
                            date_column=date_col,
                            window_days=window_days,
                            cloud_pct=cloud_pct,
                            resolution=resolution,
                            buffer_m=buffer_m,
                            output_dir=output_dir,
                            save_geotiff=save_gt,
                            save_gpkg=save_gp,
                            save_geojson=save_gj,
                        )
                        jobs_started += 1

            for err in validation_errors:
                st.error(err)

            if jobs_started:
                st.success(
                    f"{jobs_started} job(s) started. "
                    "Monitor progress in the sidebar Job Monitor."
                )
            elif not validation_errors:
                st.info("No new jobs were submitted.")

    st.divider()

    col_ndvi_btm_left, col_ndvi_btm_right = st.columns(2)
    with col_ndvi_btm_left:
        st.subheader("Result Inspector")
        scan_row_l, scan_row_r = st.columns([11, 1])
        with scan_row_l:
            ndvi_scan_clicked = st.button(
                "🔄 Scan Output Folder",
                key="ndvi_scan_folder",
                use_container_width=True,
            )
        with scan_row_r:
            ndvi_scan_spinner_slot = st.empty()

        if ndvi_scan_clicked:
            import threading as _threading

            holder: dict = {}

            def _scan_worker(out_dir: str, target: dict) -> None:
                try:
                    target["result"] = _ndvi_scan_outputs(out_dir)
                except Exception as exc:  # noqa: BLE001
                    target["error"] = exc

            with ndvi_scan_spinner_slot:
                with st.spinner("​"):
                    t = _threading.Thread(
                        target=_scan_worker, args=(output_dir, holder), daemon=True
                    )
                    t.start()
                    t.join()

            err = holder.get("error")
            if err is not None:
                st.error(f"Scan failed: {err}")
            else:
                result = holder.get("result") or {}
                count = 0
                for base_name, dataset_dict in result.items():
                    if base_name not in st.session_state.ndvi_datasets:
                        st.session_state.ndvi_datasets[base_name] = dataset_dict
                        count += 1
                if count > 0:
                    st.success(f"Loaded {count} result(s) from the output folder.")
                else:
                    st.info(
                        "No new NDVI results found "
                        "(looked for *_ndvi.tif, *_ndvi.tiff, *_ndvi.gpkg, "
                        "*_ndvi.geojson)."
                    )

        completed_ds = [
            k
            for k, v in st.session_state.ndvi_datasets.items()
            if v.get("type") == "restored"
        ]
        options = ["All Regions"] + completed_ds
        selected_opt = st.selectbox(
            "Select Result",
            options,
            index=None,
            placeholder="Choose a result...",
            key="ndvi_inspector_select",
        )
        r_opacity = st.slider("Layer Opacity", 0.0, 1.0, 0.7, key="ndvi_op")

        def _ndvi_inspector_has_geojson(sel: str | None) -> bool:
            if not completed_ds:
                return False
            if sel is None:
                return False
            if sel == "All Regions":
                return all(
                    st.session_state.ndvi_datasets[k].get("results") is not None
                    for k in completed_ds
                )
            return (
                st.session_state.ndvi_datasets.get(sel, {}).get("results") is not None
            )

        ndvi_pts_ok = _ndvi_inspector_has_geojson(selected_opt)
        if not ndvi_pts_ok and st.session_state.get("ndvi_show_points"):
            st.session_state.ndvi_show_points = False
        show_points = st.checkbox(
            "Show Sample Points",
            value=False,
            key="ndvi_show_points",
            disabled=not ndvi_pts_ok,
            help=(
                "Requires a GeoJSON next to this GeoTIFF (same base name, "
                "_ndvi.geojson). Enable Save GeoJSON when running NDVI or add the file."
                if not ndvi_pts_ok
                else None
            ),
        )
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
                rendered_from_tif = False
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
                            rendered_from_tif = True
                    except Exception as e:
                        print(f"Viz Error {ds_name}: {e}")

                # GeoPackage / GeoJSON fallback: black points, no tooltips.
                if not rendered_from_tif and ds.get("results") is not None:
                    pts = ds["results"]
                    if pts is not None and not pts.empty:
                        l, btm, r, t = pts.total_bounds
                        res_bounds.append([l, btm, r, t])
                        add_black_point_layer(
                            m_ndvi_result,
                            pts,
                            radius=6,
                            fill_opacity=r_opacity,
                            layer_name=f"{ds_name} NDVI samples",
                        )

                if show_points and ds.get("results") is not None:
                    try:
                        gdf_pts, trim_msg = trim_point_gdf_for_display(ds["results"])
                        if trim_msg:
                            st.warning(f"{ds_name}: {trim_msg}")
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
                        add_uniform_point_layer(
                            m_ndvi_result,
                            gdf_pts,
                            tooltip_fields=tooltip_fields,
                            tooltip_aliases=tooltip_aliases,
                            geojson_marker=folium.Circle(
                                radius=1,
                                color="blue",
                                fill=True,
                                fill_opacity=1,
                            ),
                            cluster_circle_radius=4,
                            cluster_color="blue",
                            cluster_fill_color="blue",
                            cluster_fill_opacity=1.0,
                            layer_name="NDVI sample points",
                        )
                    except Exception as e:
                        st.error(f"Error rendering sample points: {e}")

            # Only fit bounds when the selection changes — otherwise the
            # user's pan/zoom would be reset on every script rerun.
            last_sel = st.session_state.get("_ndvi_inspector_last_sel")
            if res_bounds and selected_opt != last_sel:
                min_x = min([b[0] for b in res_bounds])
                min_y = min([b[1] for b in res_bounds])
                max_x = max([b[2] for b in res_bounds])
                max_y = max([b[3] for b in res_bounds])
                m_ndvi_result.fit_bounds([[min_y, min_x], [max_y, max_x]])
            st.session_state["_ndvi_inspector_last_sel"] = selected_opt

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
                val_container.info(f"No valid NDVI data at ({lat:.4f}, {lon:.4f}).")
