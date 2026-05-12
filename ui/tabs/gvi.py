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
from helpers import apply_buffer_m, generate_raster_grid, load_vector_upload_sessions
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


def _gvi_geotiff_only_points_blocked() -> bool:
    """True when Run would error: GeoTIFF-only with point study areas and no buffer."""
    save_gt = st.session_state.get("gvi_out_geotiff", True)
    save_gj = st.session_state.get("gvi_out_geojson", True)
    gbuf = int(st.session_state.get("gvi_buffer", 0))
    return bool(
        save_gt
        and not save_gj
        and any(
            d.get("type") == "point" and gbuf <= 0
            for d in st.session_state.datasets.values()
            if d.get("type") != "restored"
        )
    )


def _gvi_materialize_grids_if_missing(gvi_buffer: int, gvi_res: int) -> None:
    """Set ``processed`` / ``meta`` for datasets that still need a grid."""
    gc.collect()
    for d in st.session_state.datasets.values():
        if d.get("type") == "restored":
            continue
        if d.get("processed") is not None:
            continue
        if _gvi_dataset_uses_raster_grid(d, gvi_buffer):
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
    if "master_cache" not in st.session_state:
        # In-memory pano cache for the current Streamlit process.
        # Feature 1 will replace this with a SQLite-backed PanoCache singleton.
        st.session_state.master_cache = {}
    if "gvi_inspector_select" not in st.session_state:
        st.session_state.gvi_inspector_select = None

    # JobStore + executor are process-level singletons (see ui/services.py).
    # We import them lazily here to keep tab modules free of side-effect imports.
    from services import get_job_executor, get_job_store

    store = get_job_store()
    executor = get_job_executor()

    # --- SIDEBAR JOB MONITOR ---
    def callback_dismiss_job(jid):
        store.dismiss(jid)

    def callback_cancel_job(jid):
        store.request_cancel(jid)

    _ACTIVE = {"queued", "running"}
    _TERMINAL = {"completed", "error", "cancelled", "interrupted"}

    @st.fragment(run_every=1)
    def show_job_monitor_fragment():
        st.header("Job Monitor")

        h = store.health()
        st.caption(
            f"Active: {h['active']} · Stuck: {h['stuck']} · "
            f"Errors (1h): {h['errored_last_hour']}"
        )

        records = sorted(
            store.list_all(),
            key=lambda r: r.updated_at or "",
            reverse=True,
        )
        if not records:
            st.info("No active jobs.")
            return

        for rec in records:
            with st.container(border=True):
                c1, c2 = st.columns([7, 3])
                c1.markdown(f"**{rec.name}**")
                if rec.type == "fusion":
                    c2.caption("🔀 Fusion")
                elif rec.type in ("ndvi", "ndvi_column"):
                    c2.caption("🛰️ NDVI")
                else:
                    c2.caption("🌳 GVI")

                bracket = rec.extra.get("ndvi_tile_bracket")
                if rec.type in ("ndvi", "ndvi_column") and bracket:
                    st.progress(float(rec.progress), text=str(bracket))
                else:
                    st.progress(float(rec.progress))

                # Terminal status overrides the last in-progress text so the user
                # doesn't see e.g. "Processing (26/425)" after a cancel.
                if rec.status in _TERMINAL:
                    terminal_labels = {
                        "completed": "Completed",
                        "cancelled": "Cancelled",
                        "interrupted": "Interrupted (process restarted)",
                        "error": rec.error or "Error",
                    }
                    status_label = terminal_labels.get(
                        rec.status, rec.status.capitalize()
                    )
                else:
                    status_label = rec.status_text or rec.status.capitalize()
                st.caption(status_label)

                gvi_progress = rec.extra.get("gvi_progress")
                if gvi_progress:
                    st.progress(
                        gvi_progress["percent"] / 100,
                        text=f"{gvi_progress['current']:,} / {gvi_progress['total']:,}",
                    )

                error_detail = rec.extra.get("error_detail")
                if rec.error:
                    with st.expander("Error Details"):
                        st.code(error_detail or rec.error)

                if rec.status in _ACTIVE:
                    st.button(
                        "Cancel",
                        key=f"cancel_{rec.id}",
                        on_click=callback_cancel_job,
                        args=(rec.id,),
                    )
                elif rec.status == "interrupted":
                    st.caption(
                        "Interrupted on restart. Re-upload the source "
                        "dataset and re-submit from the form above."
                    )
                    st.button(
                        "🗑️",
                        key=f"del_{rec.id}",
                        on_click=callback_dismiss_job,
                        args=(rec.id,),
                    )
                else:
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

        oc_gvi_a, oc_gvi_b = st.columns(2)
        with oc_gvi_a:
            st.checkbox(
                "Save GeoTIFF",
                value=True,
                key="gvi_out_geotiff",
                help="Raster GVI surface. At least one of GeoTIFF or GeoJSON must stay on to run.",
            )
        with oc_gvi_b:
            st.checkbox(
                "Save GeoJSON",
                value=True,
                key="gvi_out_geojson",
                help="Vector sample points with attributes. At least one output format must stay on.",
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
                with gen_action_spinner:
                    with st.spinner("\u200b"):
                        for d in st.session_state.datasets.values():
                            if d.get("type") == "restored":
                                continue
                            if _gvi_dataset_uses_raster_grid(d, gvi_buffer_for_gen):
                                pts, meta = generate_raster_grid(
                                    apply_buffer_m(d["raw"], gvi_buffer_for_gen),
                                    gvi_res_for_gen,
                                )
                                d["processed"] = pts
                                d["meta"] = meta
                            else:
                                d["processed"] = d["raw"].copy()
                                d["meta"] = None
                            d["accumulated"] = []
                            d["results"] = None
                st.success("Grids generated!")
                gc.collect()
                st.rerun()

        elif run:
            gvi_out_ok_form = st.session_state.get(
                "gvi_out_geotiff", True
            ) or st.session_state.get("gvi_out_geojson", True)
            if (
                gvi_out_ok_form
                and st.session_state.datasets
                and not _gvi_geotiff_only_points_blocked()
            ):
                with run_action_spinner:
                    with st.spinner("\u200b"):
                        _gvi_materialize_grids_if_missing(
                            gvi_buffer_for_gen, gvi_res_for_gen
                        )

    gvi_buffer = int(st.session_state.get("gvi_buffer", 0))
    gvi_res = int(st.session_state.get("gvi_res", 50))
    gvi_out_ok = st.session_state.get("gvi_out_geotiff", True) or st.session_state.get(
        "gvi_out_geojson", True
    )

    if run:
        if not gvi_out_ok:
            st.error("Select at least one output format.")
        elif not st.session_state.datasets:
            st.warning("Upload at least one study area to get started.")
        else:
            save_gt = st.session_state.get("gvi_out_geotiff", True)
            save_gj = st.session_state.get("gvi_out_geojson", True)
            save_debug = st.session_state.get("gvi_save_debug", False)
            mode = st.session_state.get("gvi_download_mode", "Package (Scraper)")
            api_key = None
            if mode == "API (Street View)":
                k = st.session_state.get("gvi_google_api_key", "")
                api_key = k if k else None

            geotiff_only_with_points = (
                save_gt
                and not save_gj
                and any(
                    d.get("type") == "point" and gvi_buffer <= 0
                    for d in st.session_state.datasets.values()
                    if d.get("type") != "restored"
                )
            )
            if geotiff_only_with_points:
                st.error(
                    "GeoTIFF-only output needs a raster sampling grid. For point "
                    "study areas, set Download Buffer (m) above zero so buffers "
                    "define the grid extent, enable Save GeoJSON, or use polygon "
                    "study areas."
                )
            else:
                model_path = os.path.join(
                    parent_dir, "geofuse", "model", "best_model.pth"
                )
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

                    d["cache_ref"] = st.session_state.master_cache

                    record = store.submit(
                        type="gvi",
                        name=f"{fname} (GVI, step={gvi_res}m, buf={gvi_buffer}m)",
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
            from rasterio.warp import transform_bounds

            tif_paths: dict[str, str] = {}
            for pat in (
                os.path.join(output_dir, "*_gvi.tif"),
                os.path.join(output_dir, "*_gvi.tiff"),
            ):
                for p in glob.glob(pat):
                    tif_paths[os.path.basename(p)] = p
            count = 0
            for basename in sorted(tif_paths.keys()):
                tif_path = tif_paths[basename]
                stem = basename.rsplit(".", 1)[0]
                base_name = stem.removesuffix("_gvi")
                geojson_path = os.path.join(output_dir, f"{base_name}_gvi.geojson")
                has_geojson = os.path.isfile(geojson_path)
                if base_name not in st.session_state.datasets:
                    try:
                        with rasterio.open(tif_path) as src:
                            meta = {
                                "transform": src.transform,
                                "width": src.width,
                                "height": src.height,
                                "crs": src.crs,
                            }
                            b = src.bounds
                            crs = src.crs
                        if has_geojson:
                            gdf = reproject_geodataframe_to_wgs84(
                                gpd.read_file(geojson_path)
                            )
                            raw_geom = gdf.geometry.union_all().envelope
                            raw_gdf = gpd.GeoDataFrame(
                                {"geometry": [raw_geom]}, crs="EPSG:4326"
                            )
                            results = gdf
                        else:
                            w, s, e, n = transform_bounds(crs, "EPSG:4326", *b)
                            raw_gdf = gpd.GeoDataFrame(
                                {"geometry": [shapely_box(w, s, e, n)]},
                                crs="EPSG:4326",
                            )
                            results = None
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
                        print(f"Error: {e}")
            if count > 0:
                st.success(f"Loaded {count} result(s) from the output folder.")
            else:
                st.info(
                    "No GVI GeoTIFF results found in the output folder "
                    "(files named *_gvi.tif or *_gvi.tiff)."
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
