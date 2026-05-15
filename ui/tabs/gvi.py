import base64
import gc
import glob
import io
import os

import folium
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
from helpers import (
    apply_buffer_m,
    generate_clustered_grid,
    load_vector_upload_sessions,
)
from map_preview import (
    add_study_area_layers,
    add_uniform_point_layer,
    trim_point_gdf_for_display,
)
from PIL import Image as PILImage
from rasterio.transform import array_bounds
from shapely.geometry import box as shapely_box
from streamlit_folium import st_folium

from geofuse.crs_utils import reproject_geodataframe_to_wgs84
from geofuse.jobs.runners import run_gvi


def _gvi_output_tif_path(output_dir: str, base_name: str) -> str | None:
    """Resolve ``{base_name}_gvi.tif`` or ``.tiff`` on disk."""
    for ext in (".tif", ".tiff"):
        p = os.path.join(output_dir, f"{base_name}_gvi{ext}")
        if os.path.isfile(p):
            return p
    return None


def _gvi_dataset_uses_raster_grid(dataset: dict, buffer_m: float) -> bool:
    """Regular grid (GeoTIFF-capable) vs raw point features only."""
    if dataset.get("type") == "poly":
        return True
    return dataset.get("type") == "point" and buffer_m > 0


def _gvi_upload_signature(uploaded_files) -> tuple[tuple[str, int], ...] | None:
    """Stable fingerprint for the file uploader selection (None = widget not committed)."""
    if uploaded_files is None:
        return None
    return tuple((str(f.name), int(getattr(f, "size", 0) or 0)) for f in uploaded_files)


def _gvi_discard_heavy_dataset_fields() -> None:
    """Drop grid / result GeoDataFrames from this tab's ``datasets`` only; then GC."""
    for d in st.session_state.datasets.values():
        if d.get("type") == "restored":
            continue
        d["processed"] = None
        d["meta"] = None
        d["accumulated"] = []
        d["results"] = None
    gc.collect()


def _gvi_size_hint(buffer_m: int, step_m: int) -> None:
    """Show an inline banner with rough sample-count and CRS-choice expectations.

    Driven by previously-committed widget values (form widgets don't commit
    until submit), so the message updates on each form interaction cycle.
    """
    if step_m <= 0:
        return
    bbox_minx = bbox_miny = float("inf")
    bbox_maxx = bbox_maxy = float("-inf")
    total_area_m2 = 0.0
    have_data = False
    for d in st.session_state.datasets.values():
        if d.get("type") == "restored":
            continue
        raw = d.get("raw")
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
        if d.get("type") == "point":
            n = len(raw)
            disk = np.pi * (max(buffer_m, 0)) ** 2
            total_area_m2 += n * disk
        else:
            try:
                deg2 = float(raw.geometry.area.sum())
                total_area_m2 += deg2 * m_per_deg_lat * m_per_deg_lon
            except Exception:
                pass
    if not have_data:
        return

    est_pts = int(total_area_m2 / (step_m * step_m))
    span_lon = bbox_maxx - bbox_minx
    span_lat = bbox_maxy - bbox_miny
    msgs: list[str] = []
    if est_pts > 0:
        msgs.append(f"Estimated sample points: ~{est_pts:,}")
    if est_pts > 100_000:
        msgs.append(
            "GeoJSON is slow to read at this scale — keep GeoPackage on as your primary output."
        )
    if span_lon > 6.0 or span_lat > 6.0:
        msgs.append(
            f"Study area spans {span_lon:.1f}° lon × {span_lat:.1f}° lat — grid math will use Lambert Conformal Conic; outputs stay in WGS84."
        )
    if msgs:
        st.info(" · ".join(msgs))


def _gvi_materialize_grids_if_missing(gvi_buffer: int, gvi_res: int) -> None:
    """Set ``processed`` / ``meta`` for datasets that still need a grid."""
    gc.collect()
    for d in st.session_state.datasets.values():
        if d.get("type") == "restored":
            continue
        if d.get("processed") is not None:
            continue
        if _gvi_dataset_uses_raster_grid(d, gvi_buffer):
            pts, meta = generate_clustered_grid(
                d["raw"], buffer_m=float(gvi_buffer), step_m=float(gvi_res)
            )
            d["processed"] = pts
            d["meta"] = meta
        else:
            d["processed"] = d["raw"].copy()
            d["meta"] = None
        d["accumulated"] = []
        d["results"] = None


# ---------------------------------------------------------------------------
# Live dataset registry (per-session, in addition to the process-level store)
# ---------------------------------------------------------------------------
#
# Job records persisted in SQLite carry only JSON-serializable parameters.
# The live GeoDataFrames they operate on stay in ``st.session_state.datasets``
# (per-session) and are looked up by ``fname`` from ``record.params``.


# ---------------------------------------------------------------------------
# Tab render entry point
# ---------------------------------------------------------------------------


def render(output_dir: str, parent_dir: str) -> None:
    st.header("GVI Sourcing")

    # --- SESSION STATE ---
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}
    if "gvi_inspector_select" not in st.session_state:
        st.session_state.gvi_inspector_select = None

    # JobStore + executor + PanoCache are process-level singletons (see ui/services.py).
    # We import them lazily here to keep tab modules free of side-effect imports.
    from services import (
        ansi_log_lines_to_html,
        get_job_executor,
        get_job_store,
        get_pano_cache,
    )

    from geofuse.logger import get_job_log_lines

    store = get_job_store()
    executor = get_job_executor()
    pano_cache = get_pano_cache()

    # --- SIDEBAR JOB MONITOR ---
    def callback_dismiss_job(jid):
        # Permanently remove from SQLite history as well so the entry doesn't
        # reappear next time the page is loaded.
        store.purge(jid)

    def callback_cancel_job(jid):
        store.request_cancel(jid)

    _ACTIVE = {"queued", "running"}
    _TERMINAL = {"completed", "error", "cancelled", "interrupted"}
    _EMOJI = {
        "fusion": "🔀",
        "ndvi": "🛰️",
        "ndvi_column": "🛰️",
        "gvi": "🌳",
    }
    _TERMINAL_LABELS = {
        "completed": "✅ Completed",
        "cancelled": "🚫 Cancelled",
        "interrupted": "⏸️ Interrupted (process restarted)",
        "error": "❌ Error",
    }

    def _render_details(rec) -> None:
        """Key/value summary of job parameters inside the Details expander."""
        p = rec.params or {}
        if rec.type == "gvi":
            st.write(f"**Grid step:** {p.get('step', '?')} m")
            st.write(f"**Buffer:** {p.get('buffer', '?')} m")
            st.write(
                f"**Save panos / masks:** "
                f"{bool(p.get('save_panos'))} / {bool(p.get('save_masks'))}"
            )
            st.write(
                f"**Outputs:** GeoPackage={bool(p.get('save_gpkg', True))} · "
                f"GeoTIFF={bool(p.get('save_geotiff'))} · "
                f"GeoJSON={bool(p.get('save_geojson'))}"
            )
            st.write(
                f"**Street View API key:** {'yes' if p.get('has_api_key') else 'no'}"
            )
        elif rec.type in ("ndvi", "ndvi_column"):
            mode = p.get("mode", "?")
            st.write(f"**Mode:** {mode}")
            if mode == "range":
                st.write(
                    f"**Date range:** {p.get('start_date', '?')} → "
                    f"{p.get('end_date', '?')}"
                )
            elif mode == "specific":
                st.write(
                    f"**Target date:** {p.get('target_date', '?')} "
                    f"(window ±{p.get('window_days', '?')} d)"
                )
            elif mode == "column":
                st.write(
                    f"**Date column:** {p.get('date_column', '?')} "
                    f"(window ±{p.get('window_days', '?')} d)"
                )
            st.write(f"**Cloud max:** {p.get('cloud_pct', '?')}%")
            st.write(f"**Resolution:** {p.get('resolution', '?')} m")
            st.write(f"**Buffer:** {p.get('buffer_m', '?')} m")
            st.write(
                f"**Outputs:** GeoTIFF={bool(p.get('save_geotiff'))} · "
                f"GeoPackage={bool(p.get('save_gpkg'))} · "
                f"GeoJSON={bool(p.get('save_geojson'))}"
            )
        elif rec.type == "fusion":
            st.write(
                f"**Trials:** {p.get('n_trials', '?')} "
                f"(startup {p.get('n_startup_trials', '?')})"
            )
            st.write(f"**Objective:** {p.get('objective_metric', '?')}")
            st.write(
                f"**Sampler:** {p.get('sampler_type', '?')} · "
                f"**Pruner:** {p.get('pruner_type', '?')}"
            )
            outcomes = p.get("outcome_columns") or []
            st.write(
                f"**Outcomes:** {len(outcomes)}{' — ' + ', '.join(outcomes) if outcomes else ''}"
            )
            st.write(f"**Resume study:** {bool(p.get('resume_existing_study', True))}")
            st.write(f"**Pre-aggregation:** {bool(p.get('pre_aggregate', False))}")
        if rec.output_paths:
            st.write("**Output files:**")
            for path in rec.output_paths:
                st.code(path, language=None)
        if rec.submitted_at:
            st.caption(f"Submitted at: {rec.submitted_at}")
        if rec.completed_at:
            st.caption(f"Completed at: {rec.completed_at}")

    def _render_logs(rec_id: str) -> None:
        lines = get_job_log_lines(rec_id)
        if not lines:
            st.caption("(no log output captured yet)")
            return
        st.markdown(ansi_log_lines_to_html(lines), unsafe_allow_html=True)

    @st.fragment(run_every=1)
    def show_job_monitor_fragment():
        st.header("Job Monitor")

        h = store.health()
        st.caption(
            f"Active: {h['active']} · Stuck: {h['stuck']} · "
            f"Errors (1h): {h['errored_last_hour']}"
        )

        all_recs = store.list_all()
        active = sorted(
            [r for r in all_recs if r.status in _ACTIVE],
            key=lambda r: r.submitted_at or r.id,
        )
        terminal = sorted(
            [r for r in all_recs if r.status in _TERMINAL],
            key=lambda r: r.completed_at or r.updated_at or "",
            reverse=True,
        )
        ordered = active + terminal

        if not ordered:
            st.info("No active jobs.")
            return

        for rec in ordered:
            with st.container(border=True):
                emoji = _EMOJI.get(rec.type, "•")
                # Title: emoji + clean name only — no extension, no params.
                st.markdown(f"### {emoji} {rec.name}")

                # Primary progress bar (no tile-bracket overlay here; the
                # bracket appears as its own row below for clarity).
                st.progress(float(rec.progress))

                # Status line. Terminal jobs show the canonical label;
                # active jobs show whatever the worker reported last.
                if rec.status in _TERMINAL:
                    label = _TERMINAL_LABELS.get(rec.status, rec.status.capitalize())
                else:
                    label = rec.status_text or rec.status.capitalize()
                st.caption(label)

                bracket = rec.extra.get("ndvi_tile_bracket")
                if rec.type in ("ndvi", "ndvi_column") and bracket:
                    st.caption(f"NDVI tile: {bracket}")

                gvi_progress = rec.extra.get("gvi_progress")
                if gvi_progress:
                    st.progress(
                        gvi_progress["percent"] / 100,
                        text=(
                            f"{gvi_progress['current']:,} / "
                            f"{gvi_progress['total']:,}"
                        ),
                    )

                preaggr_progress = rec.extra.get("preaggr_progress")
                if preaggr_progress:
                    st.progress(
                        preaggr_progress["percent"] / 100,
                        text=(
                            f"Spatial pre-processing: "
                            f"{preaggr_progress['current']:,} / "
                            f"{preaggr_progress['total']:,}"
                        ),
                    )

                # Collapsible details (parameters)
                with st.expander("Details", expanded=False):
                    _render_details(rec)

                # Collapsible per-job log
                with st.expander("Logs", expanded=False):
                    _render_logs(rec.id)

                # Error detail block stays in its own expander when present
                error_detail = rec.extra.get("error_detail")
                if rec.error:
                    with st.expander("Error trace"):
                        st.code(error_detail or rec.error)

                # Action button
                if rec.status in _ACTIVE:
                    st.button(
                        "Cancel",
                        key=f"cancel_{rec.id}",
                        on_click=callback_cancel_job,
                        args=(rec.id,),
                    )
                else:
                    if rec.status == "interrupted":
                        st.caption(
                            "Interrupted on restart — re-upload the source "
                            "dataset and re-submit from the form above."
                        )
                    st.button(
                        "🗑️",
                        key=f"del_{rec.id}",
                        on_click=callback_dismiss_job,
                        args=(rec.id,),
                    )

    with st.sidebar:
        show_job_monitor_fragment()

    st.subheader("Input Configuration")

    uploaded_files = st.file_uploader(
        "Upload Study Areas",
        accept_multiple_files=True,
        type=["geojson", "json", "gpkg", "shp", "dbf", "shx", "prj", "cpg", "zip"],
        key="gvi_up",
        help=(
            "GeoJSON, GeoPackage, or a zipped archive. For Esri Shapefile, select "
            "all components in one go (at minimum .shp, .dbf, .shx; include .prj when available)."
        ),
    )

    if uploaded_files is not None:
        sig_new = _gvi_upload_signature(uploaded_files)
        sig_prev = st.session_state.get("_gvi_prev_upload_sig")
        if sig_prev is not None and sig_new is not None and sig_prev != sig_new:
            _gvi_discard_heavy_dataset_fields()
        st.session_state._gvi_prev_upload_sig = sig_new

        loaded = load_vector_upload_sessions(uploaded_files)
        logical_names = [name for name, _ in loaded]
        for k in list(st.session_state.datasets.keys()):
            ds = st.session_state.datasets[k]
            if ds.get("type") == "restored":
                continue
            if k not in logical_names:
                del st.session_state.datasets[k]
        for fname, raw in loaded:
            if fname not in st.session_state.datasets:
                try:
                    gtype = (
                        "poly"
                        if raw.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]
                        else "point"
                    )
                    st.session_state.datasets[fname] = {
                        "raw": raw,
                        "processed": None,
                        "accumulated": [],
                        "results": None,
                        "meta": None,
                        "type": gtype,
                    }
                except Exception as e:
                    st.error(f"Error loading {fname}: {e}")

        gc.collect()

    if uploaded_files == []:
        for k in list(st.session_state.datasets.keys()):
            if st.session_state.datasets[k].get("type") != "restored":
                del st.session_state.datasets[k]
        gc.collect()

    with st.form("gvi_job_form"):
        gvi_buf_preview = int(st.session_state.get("gvi_buffer", 0))
        fc_gvi_l, fc_gvi_r = st.columns(2)
        with fc_gvi_l:
            with st.container(border=True):
                st.radio(
                    "Download Mode",
                    ["Package (Scraper)", "API (Street View)"],
                    horizontal=True,
                    key="gvi_download_mode",
                    help=(
                        "Grid spacing, download buffer, and debug export apply when you "
                        "press Generate Sampling Grids or Run below. The study-area "
                        "preview map does not update from these controls until you run "
                        "one of those actions."
                    ),
                )
                mode_sel = st.session_state.get(
                    "gvi_download_mode", "Package (Scraper)"
                )
                if mode_sel == "API (Street View)":
                    st.text_input(
                        "Street View API Key",
                        type="password",
                        autocomplete="off",
                        help="Optional Google Street View key; falls back to built-in access if empty.",
                        key="gvi_google_api_key",
                    )

                st.slider(
                    "Grid Resolution (m)",
                    min_value=20,
                    max_value=500,
                    value=50,
                    step=5,
                    key="gvi_res",
                    help="Spacing for sampling points in the generated grid (metres).",
                )
                st.slider(
                    "Download Buffer (m)",
                    min_value=0,
                    max_value=2000,
                    value=0,
                    step=50,
                    key="gvi_buffer",
                    help=(
                        "Expand the study area outward by this distance (metres) before "
                        "building the sampling grid."
                    ),
                )
                st.checkbox(
                    "Save Raw Images & Masks",
                    value=False,
                    key="gvi_save_debug",
                    help="Keep downloaded panoramas and segmentation masks under the output folder.",
                )

        with fc_gvi_r:
            st.subheader("Study Area Preview")
            show_sampling_grid = st.session_state.get(
                "gvi_preview_sampling_grid", False
            )
            m_input = folium.Map(location=[51.0447, -114.0719], zoom_start=11)

            all_bounds = []
            for fname, d in st.session_state.datasets.items():
                if d.get("type") == "restored":
                    continue
                if d.get("raw") is not None:
                    add_study_area_layers(
                        m_input,
                        d["raw"],
                        study_name=f"{fname} (study area)",
                        buffer_m=gvi_buf_preview,
                        buffer_name=f"{fname} (buffer)",
                    )
                    all_bounds.append(d["raw"].total_bounds)
                    if gvi_buf_preview > 0:
                        all_bounds.append(
                            apply_buffer_m(d["raw"], gvi_buf_preview).total_bounds
                        )
                if (
                    show_sampling_grid
                    and d.get("processed") is not None
                    and not d["processed"].empty
                ):
                    if d.get("meta") is not None:
                        add_uniform_point_layer(
                            m_input,
                            d["processed"],
                            tooltip_fields=[],
                            tooltip_aliases=[],
                            geojson_marker=folium.Circle(
                                radius=3,
                                color="#4a148c",
                                weight=1,
                                fill=True,
                                fill_opacity=0.85,
                            ),
                            cluster_threshold=0,
                            cluster_circle_radius=5,
                            cluster_color="#4a148c",
                            cluster_fill_color="#9c27b0",
                            cluster_fill_opacity=0.82,
                            layer_name=f"{fname} sampling grid",
                        )

            if all_bounds:
                min_x = min([b[0] for b in all_bounds])
                min_y = min([b[1] for b in all_bounds])
                max_x = max([b[2] for b in all_bounds])
                max_y = max([b[3] for b in all_bounds])
                m_input.fit_bounds([[min_y, min_x], [max_y, max_x]])

            st_folium(
                m_input, width="100%", height=500, key="map_input", returned_objects=[]
            )

        _gvi_size_hint(
            int(st.session_state.get("gvi_buffer", 0)),
            int(st.session_state.get("gvi_res", 50)),
        )

        oc_gvi_a, oc_gvi_b, oc_gvi_c = st.columns(3)
        with oc_gvi_a:
            st.checkbox(
                "Save GeoPackage",
                value=True,
                key="gvi_out_gpkg",
                help=(
                    "Recommended. Single-file vector samples (EPSG:4326) "
                    "readable by every modern GIS. Scales to country-scale "
                    "runs and supports sparse cluster layouts without voids."
                ),
            )
        with oc_gvi_b:
            st.checkbox(
                "Save GeoTIFF (per-cluster tiles)",
                value=False,
                key="gvi_out_geotiff",
                help=(
                    "Optional. Writes one GeoTIFF tile per buffered cluster "
                    "into a *_gvi_tiles/ folder, in the auto-selected planar "
                    "CRS (no resampling). Skipped if no clusters are defined."
                ),
            )
        with oc_gvi_c:
            st.checkbox(
                "Save GeoJSON",
                value=False,
                key="gvi_out_geojson",
                help=(
                    "Compatibility option only. Slow to read past ~100k "
                    "points; prefer GeoPackage for large national runs."
                ),
            )

        st.checkbox(
            "Show Sampling Grid on Map",
            value=False,
            key="gvi_preview_sampling_grid",
            help=(
                "Draw generated sampling grid points on the preview map. Turn off for large "
                "grids to keep the browser responsive."
            ),
        )
        gen_row_l, gen_row_r = st.columns([11, 1])
        with gen_row_l:
            gen = st.form_submit_button(
                "Generate Sampling Grids",
                use_container_width=True,
                key="gvi_gen_sampling_grids",
            )
        with gen_row_r:
            gen_action_spinner = st.empty()
        run_row_l, run_row_r = st.columns([11, 1])
        with run_row_l:
            run = st.form_submit_button(
                "🚀 Run GVI Analysis",
                type="primary",
                use_container_width=True,
                key="gvi_run_analysis",
            )
        with run_row_r:
            run_action_spinner = st.empty()

        gvi_buffer_for_gen = int(st.session_state.get("gvi_buffer", 0))
        gvi_res_for_gen = int(st.session_state.get("gvi_res", 50))

        if gen:
            if not st.session_state.datasets:
                st.warning("Upload at least one study area first.")
            else:
                _gvi_discard_heavy_dataset_fields()
                distortion_msgs: list[str] = []
                with gen_action_spinner:
                    with st.spinner("\u200b"):
                        for fname_g, d in st.session_state.datasets.items():
                            if d.get("type") == "restored":
                                continue
                            if _gvi_dataset_uses_raster_grid(d, gvi_buffer_for_gen):
                                pts, meta = generate_clustered_grid(
                                    d["raw"],
                                    buffer_m=float(gvi_buffer_for_gen),
                                    step_m=float(gvi_res_for_gen),
                                )
                                d["processed"] = pts
                                d["meta"] = meta
                                dist = float(meta.get("distortion", 0.0) or 0.0)
                                if dist > 0.02:
                                    distortion_msgs.append(
                                        f"{fname_g}: planar CRS "
                                        f"{meta.get('choice_name', '?')} \u2014 "
                                        f"distortion ~{dist * 100:.1f}% across "
                                        f"the extent. Outputs stay in WGS84; "
                                        f"distances may drift across far-apart "
                                        f"clusters."
                                    )
                            else:
                                d["processed"] = d["raw"].copy()
                                d["meta"] = None
                            d["accumulated"] = []
                            d["results"] = None
                for msg in distortion_msgs:
                    st.warning(msg)
                st.success("Grids generated!")
                gc.collect()
                st.rerun()

        elif run:
            gvi_out_ok_form = (
                st.session_state.get("gvi_out_gpkg", True)
                or st.session_state.get("gvi_out_geotiff", False)
                or st.session_state.get("gvi_out_geojson", False)
            )
            if gvi_out_ok_form and st.session_state.datasets:
                with run_action_spinner:
                    with st.spinner("\u200b"):
                        _gvi_materialize_grids_if_missing(
                            gvi_buffer_for_gen, gvi_res_for_gen
                        )

    gvi_buffer = int(st.session_state.get("gvi_buffer", 0))
    gvi_res = int(st.session_state.get("gvi_res", 50))
    gvi_out_ok = (
        st.session_state.get("gvi_out_gpkg", True)
        or st.session_state.get("gvi_out_geotiff", False)
        or st.session_state.get("gvi_out_geojson", False)
    )

    if run:
        if not gvi_out_ok:
            st.error("Select at least one output format.")
        elif not st.session_state.datasets:
            st.warning("Upload at least one study area to get started.")
        else:
            save_gp = st.session_state.get("gvi_out_gpkg", True)
            save_gt = st.session_state.get("gvi_out_geotiff", False)
            save_gj = st.session_state.get("gvi_out_geojson", False)
            save_debug = st.session_state.get("gvi_save_debug", False)
            mode = st.session_state.get("gvi_download_mode", "Package (Scraper)")
            api_key = None
            if mode == "API (Street View)":
                k = st.session_state.get("gvi_google_api_key", "")
                api_key = k if k else None

            model_path = os.path.join(parent_dir, "geofuse", "model", "best_model.pth")
            started = False

            # Identity tuple for an in-flight job — only an *identical*
            # resubmission is blocked. Changing resolution, buffer, or any
            # output flag produces a new signature and a new job.
            def _gvi_signature(p: dict) -> tuple:
                return (
                    p.get("fname"),
                    p.get("step"),
                    p.get("buffer"),
                    p.get("save_panos"),
                    p.get("save_masks"),
                    p.get("save_gpkg"),
                    p.get("save_geotiff"),
                    p.get("save_geojson"),
                    p.get("has_api_key"),
                )

            for fname, d in st.session_state.datasets.items():
                if d.get("type") == "restored":
                    continue

                job_params = {
                    "fname": fname,
                    "step": gvi_res,
                    "buffer": gvi_buffer,
                    "save_panos": save_debug,
                    "save_masks": save_debug,
                    "save_gpkg": save_gp,
                    "save_geotiff": save_gt,
                    "save_geojson": save_gj,
                    "model_path": model_path,
                    "has_api_key": api_key is not None,
                }
                sig = _gvi_signature(job_params)

                # Skip only if an identical submission is still active.
                duplicate = [
                    r
                    for r in store.list_active()
                    if r.type == "gvi" and _gvi_signature(r.params) == sig
                ]
                if duplicate:
                    continue

                d["cache_ref"] = pano_cache

                record = store.submit(
                    type="gvi",
                    name=os.path.splitext(fname)[0],
                    params=job_params,
                )
                executor.submit_runner(
                    record,
                    run_gvi,
                    fname=fname,
                    dataset_data=d,
                    init_args={"model_path": model_path, "api_key": api_key},
                    run_args={
                        "step": gvi_res,
                        "save_panos": save_debug,
                        "save_masks": save_debug,
                    },
                    output_dir=output_dir,
                    save_gpkg=save_gp,
                    save_geotiff=save_gt,
                    save_geojson=save_gj,
                    gpu_lock=executor.gpu_lock,
                )
                started = True

            if started:
                st.success("Analysis started. Monitor progress in the sidebar.")
            else:
                st.info("All study areas are already running or completed.")

    st.divider()

    col_btm_left, col_btm_right = st.columns(2)

    with col_btm_left:
        st.subheader("Result Inspector")

        if st.button("🔄 Scan Output Folder", key="gvi_scan_folder"):
            import json as _json

            from rasterio.warp import transform_bounds

            # Discover every base name by union of GPKG / GeoJSON / single TIF
            # / per-cluster tiles folder. The new canonical output is GPKG +
            # sidecar JSON, but legacy single-TIF outputs are still supported.
            base_names: set[str] = set()
            for pat in (
                "*_gvi.gpkg",
                "*_gvi.geojson",
                "*_gvi.tif",
                "*_gvi.tiff",
            ):
                for p in glob.glob(os.path.join(output_dir, pat)):
                    stem = os.path.basename(p).rsplit(".", 1)[0]
                    base_names.add(stem.removesuffix("_gvi"))
            for tiles_dir in glob.glob(os.path.join(output_dir, "*_gvi_tiles")):
                base_names.add(os.path.basename(tiles_dir).removesuffix("_gvi_tiles"))

            count = 0
            for base_name in sorted(base_names):
                if base_name in st.session_state.datasets:
                    continue

                gpkg_path = os.path.join(output_dir, f"{base_name}_gvi.gpkg")
                gj_path = os.path.join(output_dir, f"{base_name}_gvi.geojson")
                sidecar_path = os.path.join(output_dir, f"{base_name}_gvi.json")
                single_tif: str | None = None
                for ext in (".tif", ".tiff"):
                    p = os.path.join(output_dir, f"{base_name}_gvi{ext}")
                    if os.path.isfile(p):
                        single_tif = p
                        break
                tiles_dir = os.path.join(output_dir, f"{base_name}_gvi_tiles")
                has_tiles = os.path.isdir(tiles_dir)

                try:
                    results = None
                    if os.path.isfile(gpkg_path):
                        results = reproject_geodataframe_to_wgs84(
                            gpd.read_file(gpkg_path)
                        )
                    elif os.path.isfile(gj_path):
                        results = reproject_geodataframe_to_wgs84(
                            gpd.read_file(gj_path)
                        )

                    meta: dict | None = None
                    if single_tif:
                        with rasterio.open(single_tif) as src:
                            meta = {
                                "transform": src.transform,
                                "width": src.width,
                                "height": src.height,
                                "crs": src.crs,
                            }
                    elif os.path.isfile(sidecar_path):
                        with open(sidecar_path) as f:
                            sc = _json.load(f)
                        meta = {
                            "grid_crs_wkt": sc.get("grid_crs_wkt"),
                            "step_m": sc.get("step_m"),
                            "anchor_x": sc.get("anchor_x"),
                            "anchor_y": sc.get("anchor_y"),
                            "tiles_dir": tiles_dir if has_tiles else None,
                            "n_clusters": sc.get("n_clusters"),
                        }
                    elif has_tiles:
                        meta = {"tiles_dir": tiles_dir}

                    if results is not None and not results.empty:
                        raw_geom = results.geometry.union_all().envelope
                        raw_gdf = gpd.GeoDataFrame(
                            {"geometry": [raw_geom]}, crs="EPSG:4326"
                        )
                    elif single_tif:
                        with rasterio.open(single_tif) as src:
                            b = src.bounds
                            crs = src.crs
                        w, s, e, n = transform_bounds(crs, "EPSG:4326", *b)
                        raw_gdf = gpd.GeoDataFrame(
                            {"geometry": [shapely_box(w, s, e, n)]},
                            crs="EPSG:4326",
                        )
                    else:
                        # Nothing readable for this base name.
                        continue

                    st.session_state.datasets[base_name] = {
                        "raw": raw_gdf,
                        "processed": None,
                        "accumulated": [],
                        "results": results,
                        "meta": meta,
                        "type": "restored",
                    }
                    count += 1
                except Exception as e:
                    print(f"Error scanning {base_name}: {e}")
            if count > 0:
                st.success(f"Loaded {count} result(s) from the output folder.")
            else:
                st.info(
                    "No GVI results found in the output folder "
                    "(looked for *_gvi.gpkg, *_gvi.geojson, *_gvi.tif, "
                    "*_gvi_tiles/)."
                )

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
            key="gvi_inspector_select",
        )

        raster_layer = st.radio(
            "Background Raster",
            ["Vegetation", "Terrain"],
            horizontal=True,
            key="gvi_raster_layer",
        )
        r_opacity = st.slider("Layer Opacity", 0.0, 1.0, 0.7, key="gvi_layer_opacity")

        def _gvi_inspector_has_geojson(sel: str | None) -> bool:
            if not completed_ds:
                return False
            if sel is None:
                return False
            if sel == "All Regions":
                return all(
                    st.session_state.datasets[k].get("results") is not None
                    for k in completed_ds
                )
            return st.session_state.datasets.get(sel, {}).get("results") is not None

        gvi_pts_ok = _gvi_inspector_has_geojson(selected_option)
        if not gvi_pts_ok and st.session_state.get("gvi_show_points"):
            st.session_state.gvi_show_points = False
        show_points = st.checkbox(
            "Show Sample Points",
            value=False,
            key="gvi_show_points",
            disabled=not gvi_pts_ok,
            help=(
                "Requires a GeoJSON next to this GeoTIFF (same base name, "
                "_gvi.geojson). Enable Save GeoJSON when running GVI or add the file."
                if not gvi_pts_ok
                else None
            ),
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

                meta = ds.get("meta") or {}
                has_single_raster = (
                    meta.get("transform") is not None
                    and meta.get("height") is not None
                    and meta.get("width") is not None
                )

                # Always show an extent rectangle for the dataset, derived from
                # the legacy raster meta when available, otherwise from the
                # raw envelope produced by Scan Output Folder.
                if has_single_raster:
                    from rasterio.warp import transform_bounds

                    left, bottom, right, top = array_bounds(
                        meta["height"], meta["width"], meta["transform"]
                    )
                    meta_crs = meta.get("crs", "EPSG:4326")
                    if meta_crs != "EPSG:4326":
                        left, bottom, right, top = transform_bounds(
                            meta_crs, "EPSG:4326", left, bottom, right, top
                        )
                elif ds.get("raw") is not None and not ds["raw"].empty:
                    left, bottom, right, top = ds["raw"].total_bounds
                else:
                    left = bottom = right = top = None

                if left is not None:
                    folium.Rectangle(
                        bounds=[[bottom, left], [top, right]],
                        color="grey",
                        weight=1,
                        fill=False,
                        popup=f"{ds_name} Extent",
                    ).add_to(m_result)
                    res_bounds.append([left, bottom, right, top])

                # Raster overlay is only available for legacy single-TIF
                # outputs. New GeoPackage-only outputs render as points only
                # (toggle Show Sample Points below).
                if has_single_raster:
                    from rasterio.transform import rowcol

                    arr = np.full((meta["height"], meta["width"]), np.nan)
                    col_name = "gvi_ter" if "Terrain" in raster_layer else "gvi_veg"
                    if ds.get("results") is not None:
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

                    tif_path_ds = _gvi_output_tif_path(output_dir, ds_name)
                    if not np.any(np.isfinite(arr)) and tif_path_ds:
                        with rasterio.open(tif_path_ds) as src:
                            bi = 2 if "Terrain" in raster_layer else 1
                            r = src.read(bi).astype(np.float64)
                            nodata = src.nodata
                        if r.shape == (meta["height"], meta["width"]):
                            arr = r
                            if nodata is not None:
                                arr = np.where(arr == nodata, np.nan, arr)

                    if np.any(np.isfinite(arr)):
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

                if show_points and ds.get("results") is not None:
                    try:
                        gdf_viz = ds["results"].copy()
                        gdf_viz, trim_msg = trim_point_gdf_for_display(gdf_viz)
                        if trim_msg:
                            st.warning(f"{ds_name}: {trim_msg}")
                        if "gvi_veg" in gdf_viz.columns:
                            gdf_viz["gvi_veg"] = gdf_viz["gvi_veg"].round(4)
                        if "gvi_ter" in gdf_viz.columns:
                            gdf_viz["gvi_ter"] = gdf_viz["gvi_ter"].round(4)

                        valid_pts = gdf_viz.dropna(subset=["gvi_veg"])
                        add_uniform_point_layer(
                            m_result,
                            valid_pts,
                            tooltip_fields=["gvi_veg", "gvi_ter"],
                            tooltip_aliases=["Veg Index:", "Ter Index:"],
                            geojson_marker=folium.Circle(
                                radius=20,
                                fill_color="green",
                                fill_opacity=0.8,
                                color=None,
                            ),
                            cluster_circle_radius=6,
                            cluster_color="#1a7f37",
                            cluster_fill_color="green",
                            cluster_fill_opacity=0.8,
                            layer_name=f"{ds_name} GVI sample points",
                        )
                    except Exception as e:
                        st.error(f"Error rendering sample points: {e}")

            if res_bounds:
                min_x = min([b[0] for b in res_bounds])
                min_y = min([b[1] for b in res_bounds])
                max_x = max([b[2] for b in res_bounds])
                max_y = max([b[3] for b in res_bounds])
                m_result.fit_bounds([[min_y, min_x], [max_y, max_x]])

        st_folium(
            m_result, width="100%", height=500, key="map_result", returned_objects=[]
        )
