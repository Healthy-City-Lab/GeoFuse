import base64
import glob
import html
import io
import os
from datetime import date, datetime
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
from branca.element import MacroElement
from helpers import (
    FUSION_TARGET_UPLOAD_TYPES,
    RESTART_SESSION_KEY,
    materialize_uploaded_dataset,
    sanitize_gdf_attributes_for_json,
)
from jinja2 import Template
from map_preview import add_mixed_geojson_preview, add_outcome_colored_geometry_layer
from PIL import Image as PILImage
from shapely.geometry import box as shapely_box
from streamlit_folium import st_folium

try:
    from geofuse.fusion import MetricFusionEngine as _MetricFusionEngine
except ImportError:
    _MetricFusionEngine = None

from geofuse import cgi_formulas as _cgi_formulas
from geofuse.crs_utils import buffer_gdf_union_metres, reproject_geodataframe_to_wgs84
from geofuse.jobs.runners import run_fusion
from geofuse.vector_io import (
    geometry_sha256,
    list_gpkg_layer_names,
    read_vector_path,
    vector_format_from_path,
)

_FUSION_OUTCOME_ADD_PLACEHOLDER = "— Select column —"


class _FusionVerticalScaleControl(MacroElement):
    """Leaflet control: vertical red→yellow→green strip with numeric bounds."""

    _template = Template("""
{% macro script(this, kwargs) %}
    var {{ this.get_name() }}_vsc = L.control({position: 'topright'});
    {{ this.get_name() }}_vsc.onAdd = function (map) {
        var d = L.DomUtil.create('div', 'gf-fusion-vscale leaflet-bar');
        d.style.background = 'rgba(255,255,255,0.78)';
        d.style.padding = '6px 8px';
        d.style.borderRadius = '4px';
        d.style.border = '2px solid rgba(0,0,0,0.12)';
        d.innerHTML = {{ this.inner_html|tojson }};
        L.DomEvent.disableClickPropagation(d);
        L.DomEvent.disableScrollPropagation(d);
        return d;
    };
    {{ this.get_name() }}_vsc.addTo({{ this._parent.get_name() }});
{% endmacro %}
""")

    def __init__(self, inner_html: str):
        super().__init__()
        self._name = "FusionVScale"
        self.inner_html = inner_html


def _fusion_append_outcome_callback() -> None:
    """Append the chosen GeoJSON outcome column and reset the add widget."""
    pick = st.session_state.get("fusion_add_outcome_column")
    if not pick or pick == _FUSION_OUTCOME_ADD_PLACEHOLDER:
        return
    if "fusion_outcome_columns" not in st.session_state:
        st.session_state.fusion_outcome_columns = []
    if pick not in st.session_state.fusion_outcome_columns:
        st.session_state.fusion_outcome_columns.append(pick)
    st.session_state.fusion_add_outcome_column = _FUSION_OUTCOME_ADD_PLACEHOLDER


def _fusion_resolve_active_bundle():
    """Return (per-outcome bundle dict, engine) for the outcome selected in results UI."""
    fr = st.session_state.get("fusion_results")
    if not fr:
        return None, None
    engines_map = st.session_state.get("fusion_engines_by_target") or {}
    if fr.get("mode") == "multi":
        labels = fr.get("ordered_labels") or []
        pk = st.session_state.get("fusion_results_outcome_pick")
        if not pk and labels:
            pk = labels[0]
        if not pk or pk not in fr.get("by_target", {}):
            return None, None
        return fr["by_target"][pk], engines_map.get(pk)
    return fr, st.session_state.get("fusion_engine")


def _geojson_geometry_summary(gdf: gpd.GeoDataFrame) -> str:
    """Short human-readable geometry description for UI (not only points)."""
    n = len(gdf)
    if n == 0:
        return "0 features"
    vc = gdf.geometry.geom_type.value_counts()
    if len(vc) == 1:
        t = vc.index[0]
        plural_map = {
            "Point": ("point", "points"),
            "MultiPoint": ("multi-point feature", "multi-point features"),
            "LineString": ("line", "lines"),
            "MultiLineString": ("multi-line", "multi-lines"),
            "Polygon": ("polygon", "polygons"),
            "MultiPolygon": ("multi-polygon", "multi-polygons"),
            "GeometryCollection": (
                "geometry collection",
                "geometry collections",
            ),
        }
        singular, plural = plural_map.get(t, (t.lower(), t.lower() + "s"))
        return f"{n} {singular if n == 1 else plural}"
    return f"{n} features (mixed geometry types)"


def _fusion_rdygn_hex(t: float) -> str:
    """Map t in [0, 1] to hex color red → yellow → green."""
    t = float(np.clip(t, 0.0, 1.0))
    if t <= 0.5:
        u = t * 2.0
        r, g, b = 255, int(255 * u), 0
    else:
        u = (t - 0.5) * 2.0
        r, g = int(255 * (1.0 - u)), 255
        b = 0
    return f"#{r:02x}{g:02x}{b:02x}"


def _fusion_value_to_rdygn_hex(v: float, vmin: float, vmax: float) -> str:
    """Fill color for outcome preview (uniform green when vmax <= vmin)."""
    if not np.isfinite(v) or not np.isfinite(vmin) or not np.isfinite(vmax):
        return "#22aa44"
    if vmax <= vmin:
        return "#22aa44"
    return _fusion_rdygn_hex((v - vmin) / (vmax - vmin))


def _fusion_vertical_scale_inner_html(vmin: float, vmax: float) -> str:
    """HTML fragment for the in-map vertical legend body."""
    vmin_s = html.escape(f"{vmin:.4g}")
    vmax_s = html.escape(f"{vmax:.4g}")
    if vmax > vmin:
        bar_bg = "linear-gradient(to top, #ff0000 0%, #ffff00 50%, #00ff00 100%)"
    else:
        bar_bg = "#22aa44"
    return (
        '<div aria-label="Outcome value scale" '
        'style="display:flex;flex-direction:row;align-items:stretch;gap:6px;'
        'height:min(200px,36vh);max-height:240px;box-sizing:border-box;">'
        '<div style="display:flex;flex-direction:column;justify-content:space-between;'
        "text-align:right;font-size:11px;line-height:1.15;color:#222;"
        'min-width:2.2rem;flex-shrink:0;">'
        f"<span>{vmax_s}</span><span>{vmin_s}</span></div>"
        '<div title="High (top) to low (bottom)" '
        'style="width:12px;border-radius:2px;border:1px solid rgba(0,0,0,0.3);'
        f'background:{bar_bg};flex-shrink:0;"></div></div>'
    )


def _add_fusion_vertical_scale_to_map(m: folium.Map, vmin: float, vmax: float) -> None:
    _FusionVerticalScaleControl(_fusion_vertical_scale_inner_html(vmin, vmax)).add_to(m)


def _add_outcome_geometry_preview(
    m: folium.Map,
    preview_gdf: gpd.GeoDataFrame,
    preview_feature: str,
) -> bool:
    """Draw outcome-colored geometries and attach the vertical scale; return False if nothing drawn."""
    vals = preview_gdf[preview_feature].dropna()
    if len(vals) == 0:
        return False
    vmin, vmax = float(vals.min()), float(vals.max())

    def _to_hex(v: float) -> str:
        return _fusion_value_to_rdygn_hex(v, vmin, vmax)

    ok = add_outcome_colored_geometry_layer(
        m,
        preview_gdf,
        value_column=preview_feature,
        value_to_hex=_to_hex,
    )
    if ok:
        _add_fusion_vertical_scale_to_map(m, vmin, vmax)
    return ok


# ---------------------------------------------------------------------------
# Metric-file helpers
# ---------------------------------------------------------------------------


def _scan_metric_files(output_dir: str, suffix: str) -> list:
    """Return sorted (basename, path) for raster metrics ``*_{suffix}.tif`` / ``.tiff``.

    Coverage checks use fast raster bounds only. Matching GeoJSON files duplicate the
    same grids and are excluded from this scan.
    """
    seen: dict[str, str] = {}
    for ext in (".tif", ".tiff"):
        pattern = os.path.join(output_dir, f"*_{suffix}{ext}")
        for p in glob.glob(pattern):
            seen[os.path.basename(p)] = p
    return [(bn, seen[bn]) for bn in sorted(seen.keys())]


def _compute_buffered_extent(
    tmp_target_path: str,
    is_vector_target: bool,
    buffer_meters: float,
    vector_layer: str | int | None = None,
) -> "gpd.GeoDataFrame | None":
    """Compute the buffered target extent as a GeoDataFrame in EPSG:4326."""
    try:
        if is_vector_target:
            kwargs: dict = {}
            if vector_layer is not None and Path(tmp_target_path).suffix.lower() in (
                ".gpkg",
                ".zip",
            ):
                kwargs["layer"] = vector_layer
            gdf = reproject_geodataframe_to_wgs84(
                gpd.read_file(tmp_target_path, **kwargs)
            )
        else:
            with rasterio.open(tmp_target_path) as src:
                b = src.bounds
                src_crs = src.crs
            gdf = gpd.GeoDataFrame(
                {"geometry": [shapely_box(b.left, b.bottom, b.right, b.top)]},
                crs=src_crs,
            ).to_crs("EPSG:4326")

        return buffer_gdf_union_metres(gdf, buffer_meters).to_crs("EPSG:4326")
    except Exception:
        return None


def _check_coverage(metric_path: str, buffered_gdf: "gpd.GeoDataFrame") -> bool:
    """Return True if the metric raster's extent fully covers buffered_gdf."""
    try:
        if not metric_path.lower().endswith((".tif", ".tiff")):
            return False
        with rasterio.open(metric_path) as src:
            metric_box = shapely_box(*src.bounds)
            metric_crs = src.crs

        target_geom = buffered_gdf.to_crs(metric_crs).geometry.union_all()
        return metric_box.covers(target_geom)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Restart workflow
# ---------------------------------------------------------------------------


def _fusion_restart_summary_lines(p: dict) -> list[str]:
    """Read-only summary of the original job's config shown above the re-run form."""
    covs = p.get("covariate_columns") or []
    standalones = p.get("standalone_channels") or []
    lines = [
        f"**Target:** `{p.get('target_display_name', '?')}`",
        f"**Outcomes:** {', '.join(p.get('outcome_columns') or []) or '—'}",
        f"**CGI formula:** `{p.get('cgi_formula') or 'weighted_average'}`",
        f"**Covariates:** {', '.join(covs) if covs else '—'}",
        f"**Standalone metrics:** " f"{', '.join(standalones) if standalones else '—'}",
        f"**Trials:** {p.get('n_trials', '?')} "
        f"(startup {p.get('n_startup_trials', '?')})",
        f"**Objective:** {p.get('objective_metric', '?')} · "
        f"**Sampler:** {p.get('sampler_type', '?')} · "
        f"**Pruner:** {p.get('pruner_type', '?')}",
        f"**GVI buffers (m):** {p.get('gvi_buffer_min_m', '?')} – "
        f"{p.get('gvi_buffer_max_m', '?')} (step {p.get('gvi_buffer_step_m', '?')})",
        f"**NDVI buffers (m):** {p.get('ndvi_buffer_min_m', '?')} – "
        f"{p.get('ndvi_buffer_max_m', '?')} (step {p.get('ndvi_buffer_step_m', '?')})",
        f"**Metric source:** {p.get('metric_mode', '?')}",
    ]
    return lines


def _submit_fusion_restart(
    store,
    executor,
    rec,
    p: dict,
    target_mat,
    output_dir: str,
    veg_path: str | None,
    ndvi_path: str | None,
    api_key: str | None,
) -> None:
    """Resubmit a fusion job with identical params; on-disk caches resume.

    The Optuna study (by ``study_name``), the per-job pre-aggregation cache (by
    fingerprint), and the metric-download cache are all content-addressed, so a
    same-config resubmit picks up where the previous run left off — the new job
    just walks the stage ledger, with stages whose underlying caches are
    populated completing near-instantly.
    """
    if _MetricFusionEngine is None:
        raise RuntimeError("MetricFusionEngine is unavailable; cannot restart.")

    new_params = dict(p)
    new_params["restart_of"] = rec.id
    new_params["resume_existing_study"] = True
    new_params["has_api_key"] = api_key is not None

    is_vector = bool(p.get("is_vector_target"))
    outcome_columns = list(p.get("outcome_columns") or [])
    job_target_band = int(p.get("target_band", 1))

    new_rec = store.submit(type="fusion", name=rec.name, params=new_params)
    executor.submit_runner(
        new_rec,
        run_fusion,
        target_path=target_mat.path,
        target_features_geojson=(tuple(outcome_columns) if is_vector else ()),
        target_band=job_target_band if not is_vector else 1,
        target_layer=p.get("target_layer") if is_vector else None,
        target_cleanup_dir=target_mat.cleanup_dir,
        target_cleanup_file=target_mat.cleanup_file,
        buffer_meters=float(p.get("buffer_meters", 0.0)),
        gvi_buffer_min_m=float(p.get("gvi_buffer_min_m", 100)),
        gvi_buffer_max_m=float(p.get("gvi_buffer_max_m", 1500)),
        gvi_buffer_step_m=float(p.get("gvi_buffer_step_m", 50)),
        ndvi_buffer_min_m=float(p.get("ndvi_buffer_min_m", 100)),
        ndvi_buffer_max_m=float(p.get("ndvi_buffer_max_m", 1500)),
        ndvi_buffer_step_m=float(p.get("ndvi_buffer_step_m", 50)),
        ndvi_resolution_m=p.get("ndvi_resolution_m"),
        gvi_grid_spacing_m=p.get("gvi_grid_spacing_m"),
        n_bins=int(p.get("n_bins", 5)),
        veg_path=veg_path,
        terrain_path=None,
        ndvi_path=ndvi_path,
        cache_metrics=bool(p.get("cache_metrics", False)),
        test_size=float(p.get("test_size", 0.3)),
        k_folds=int(p.get("k_folds", 5)),
        n_trials=int(p.get("n_trials", 300)),
        n_startup_trials=int(p.get("n_startup_trials", 150)),
        objective_metric=p.get("objective_metric", "pearson"),
        pruner_type=p.get("pruner_type", "median"),
        sampler_type=p.get("sampler_type", "TPE"),
        gvi_api_key=api_key,
        ndvi_start_date=p.get("ndvi_start_date") or date(2023, 6, 1).isoformat(),
        ndvi_end_date=p.get("ndvi_end_date") or date(2023, 9, 30).isoformat(),
        ndvi_project_id=None,
        multi_objective_requested=bool(p.get("multi_objective_requested")),
        output_dir=output_dir,
        MetricFusionEngine=_MetricFusionEngine,
        target_display_name=p.get("target_display_name") or "target",
        resume_existing_study=True,
        cgi_formula=str(p.get("cgi_formula") or "weighted_average"),
        covariate_columns=list(p.get("covariate_columns") or []),
        standalone_channels=list(p.get("standalone_channels") or []),
    )


def _render_fusion_restart_panel(store, executor, output_dir: str) -> bool:
    """Restart workflow for a stopped fusion job.

    Returns True when the panel handled the active restart session — the caller
    should ``return`` and skip the rest of the tab so the user finishes the
    restart flow before submitting anything else.
    """
    job_id = st.session_state.get(RESTART_SESSION_KEY)
    if not job_id:
        return False
    rec = store.get(job_id)
    if rec is None or rec.type != "fusion":
        return False

    p = rec.params or {}
    is_vector = bool(p.get("is_vector_target"))
    metric_mode = p.get("metric_mode") or "Use Loaded Results"
    had_api_key = bool(p.get("has_api_key"))
    expected_hash = p.get("geometry_sha256")

    with st.expander(f"↻ Restart fusion job: {rec.name or rec.id}", expanded=True):
        st.caption(
            "Re-upload the original target file. The Optuna study, "
            "pre-aggregation cache, and metric downloads are all content-"
            "addressed, so a same-config restart resumes where the previous "
            "run stopped."
        )
        for line in _fusion_restart_summary_lines(p):
            st.write(line)

        cancel_col, _ = st.columns([1, 4])
        with cancel_col:
            if st.button("Cancel restart", key=f"f_restart_cancel_{rec.id}"):
                st.session_state[RESTART_SESSION_KEY] = None
                st.rerun()

        target_uploads = st.file_uploader(
            f"Re-upload target (original: `{p.get('target_display_name', '?')}`)",
            accept_multiple_files=True,
            type=FUSION_TARGET_UPLOAD_TYPES,
            key=f"f_restart_target_{rec.id}",
        )
        if not target_uploads:
            st.info("Select the original target file(s) to continue.")
            return True

        try:
            target_mat = materialize_uploaded_dataset(target_uploads)
        except ValueError as e:
            st.error(str(e))
            return True

        if is_vector:
            try:
                rv_kwargs: dict = {}
                target_layer = p.get("target_layer")
                if target_layer is not None and Path(
                    target_mat.path
                ).suffix.lower() in (
                    ".gpkg",
                    ".zip",
                ):
                    rv_kwargs["layer"] = target_layer
                gdf_re = read_vector_path(target_mat.path, **rv_kwargs)
            except Exception as e:
                st.error(f"Failed to read target: {e}")
                return True
            actual_hash = geometry_sha256(gdf_re)
            if expected_hash:
                if actual_hash != expected_hash:
                    st.error(
                        "Hash mismatch — uploaded target is not the original.\n\n"
                        f"  Expected: `{expected_hash[:16]}…`\n"
                        f"  Got:      `{actual_hash[:16]}…`"
                    )
                    return True
                st.success("Target geometry verified against the original.")
            else:
                st.toast(
                    "Original geometry hash was not recorded — skipping verification.",
                    icon="⚠️",
                )
        else:
            st.toast("Raster target — content hash not verified.", icon="ℹ️")

        # Re-resolve metric files. Loaded-results paths live under the output
        # folder and are stable; uploaded files were tempdir-scoped and gone;
        # auto-download skips both (the runner re-fetches via the disk cache).
        veg_path_re: str | None = None
        ndvi_path_re: str | None = None
        if metric_mode == "Upload Files":
            st.markdown("**Re-upload metric files (original temp uploads are gone)**")
            col_g, col_n = st.columns(2)
            with col_g:
                gvi_up = st.file_uploader(
                    "🌿 GVI File",
                    accept_multiple_files=True,
                    type=FUSION_TARGET_UPLOAD_TYPES,
                    key=f"f_restart_gvi_{rec.id}",
                )
                if gvi_up:
                    try:
                        veg_path_re = materialize_uploaded_dataset(gvi_up).path
                    except ValueError as e:
                        st.error(f"GVI: {e}")
                        return True
            with col_n:
                ndvi_up = st.file_uploader(
                    "🛰️ NDVI File",
                    accept_multiple_files=True,
                    type=FUSION_TARGET_UPLOAD_TYPES,
                    key=f"f_restart_ndvi_{rec.id}",
                )
                if ndvi_up:
                    try:
                        ndvi_path_re = materialize_uploaded_dataset(ndvi_up).path
                    except ValueError as e:
                        st.error(f"NDVI: {e}")
                        return True
        elif metric_mode == "Use Loaded Results":
            gvi_bn = p.get("gvi_basename")
            ndvi_bn = p.get("ndvi_basename")
            if gvi_bn and p.get("gvi_under_output_dir"):
                cand = os.path.join(output_dir, gvi_bn)
                if os.path.exists(cand):
                    veg_path_re = cand
                    st.success(f"✓ Reusing GVI file from output folder: `{gvi_bn}`")
                else:
                    st.warning(
                        f"⚠️ GVI file `{gvi_bn}` is no longer in the output folder; "
                        "the runner will auto-download (cache reused when present)."
                    )
            if ndvi_bn and p.get("ndvi_under_output_dir"):
                cand = os.path.join(output_dir, ndvi_bn)
                if os.path.exists(cand):
                    ndvi_path_re = cand
                    st.success(f"✓ Reusing NDVI file from output folder: `{ndvi_bn}`")
                else:
                    st.warning(
                        f"⚠️ NDVI file `{ndvi_bn}` is no longer in the output folder; "
                        "the runner will auto-download (cache reused when present)."
                    )
        # Auto-Download: leave both paths None; runner re-runs the download
        # against the metric cache (cache_metrics flag preserved in params).

        api_key: str | None = None
        if had_api_key:
            st.caption(
                "Original job used the Street View API. Re-supply the API key "
                "(secrets are not persisted between runs)."
            )
            api_key = (
                st.text_input(
                    "Street View API Key",
                    type="password",
                    autocomplete="off",
                    key=f"f_restart_apikey_{rec.id}",
                )
                or None
            )

        if st.button(
            "Verify & re-run",
            type="primary",
            key=f"f_restart_confirm_{rec.id}",
        ):
            try:
                _submit_fusion_restart(
                    store,
                    executor,
                    rec,
                    p,
                    target_mat,
                    output_dir,
                    veg_path_re,
                    ndvi_path_re,
                    api_key,
                )
            except Exception as e:
                st.error(f"Re-submission failed: {e}")
                return True
            st.session_state[RESTART_SESSION_KEY] = None
            st.success("Restart submitted. Monitor progress in the sidebar.")
            st.rerun()
    return True


# ---------------------------------------------------------------------------
# Tab render entry point
# ---------------------------------------------------------------------------


def render(output_dir: str) -> None:
    st.header("Metric Fusion & Optimization")

    MetricFusionEngine = _MetricFusionEngine

    if MetricFusionEngine is None:
        st.error("MetricFusionEngine module not found in geofuse/fusion.py")
        return

    # Surface the restart workflow if the user clicked ↻ on a fusion job in the
    # sidebar monitor. The panel takes over the tab until cancelled or
    # confirmed — same pattern as the GVI / NDVI tabs.
    from services import get_job_executor, get_job_store

    _restart_store = get_job_store()
    _restart_executor = get_job_executor()
    if _render_fusion_restart_panel(_restart_store, _restart_executor, output_dir):
        return

    if "fusion_engine" not in st.session_state:
        st.session_state.fusion_engine = None
    if "fusion_results" not in st.session_state:
        st.session_state.fusion_results = None
    if "fusion_outcome_columns" not in st.session_state:
        st.session_state.fusion_outcome_columns = []
    if "fusion_engines_by_target" not in st.session_state:
        st.session_state.fusion_engines_by_target = {}

    # =========================================================================
    # ROW 1: Configuration (Left) | Preview (Right)
    # =========================================================================
    col_fusion_left, col_fusion_right = st.columns([1, 1])

    tmp_target_path = None
    target_mat = None
    is_vector_target = False
    is_raster_target = False
    target_layer_for_engine: str | int | None = None
    preview_vector_gdf: gpd.GeoDataFrame | None = None
    target_outcome_columns: list = []
    target_band = int(st.session_state.get("fusion_target_band", 1))
    multi_objective_requested = False
    target_display_name: str | None = None

    with col_fusion_left:
        st.subheader("Target Configuration")

        target_uploads = st.file_uploader(
            "Upload Target File",
            accept_multiple_files=True,
            type=FUSION_TARGET_UPLOAD_TYPES,
            key="fusion_target_upload",
            help=(
                "Vector: GeoJSON, GeoPackage, shapefile sidecars (.shp, .dbf, .shx, "
                ".prj), or vector zip. Raster: GeoTIFF or a raster zip. Select all "
                "shapefile parts in one batch."
            ),
        )

        if target_uploads:
            try:
                target_mat = materialize_uploaded_dataset(target_uploads)
            except ValueError as e:
                st.error(str(e))
                target_mat = None

            if target_mat is not None:
                tmp_target_path = target_mat.path
                is_vector_target = target_mat.is_vector
                is_raster_target = not target_mat.is_vector
                target_display_name = target_mat.display_name
                sig = tuple(sorted((f.name, f.size) for f in target_uploads))
                if st.session_state.get("fusion_target_upload_sig") != sig:
                    st.session_state.fusion_target_upload_sig = sig
                    st.session_state.fusion_outcome_columns = []

                if is_vector_target:
                    try:
                        layers: list[str] = []
                        suf = Path(tmp_target_path).suffix.lower()
                        if suf in (".gpkg", ".zip"):
                            try:
                                layers = list_gpkg_layer_names(tmp_target_path)
                            except Exception:
                                layers = []
                        if len(layers) > 1:
                            target_layer_for_engine = st.selectbox(
                                "Target layer",
                                options=layers,
                                key="fusion_target_gpkg_layer",
                                help="GeoPackage or archive layer containing geometries and attributes.",
                            )
                        elif len(layers) == 1:
                            target_layer_for_engine = layers[0]

                        rv_kwargs: dict = {}
                        if target_layer_for_engine is not None and suf in (
                            ".gpkg",
                            ".zip",
                        ):
                            rv_kwargs["layer"] = target_layer_for_engine
                        preview_vector_gdf = read_vector_path(
                            tmp_target_path, **rv_kwargs
                        )
                        numeric_cols = preview_vector_gdf.select_dtypes(
                            include=[np.number]
                        ).columns.tolist()
                        fmt = vector_format_from_path(tmp_target_path)
                        layer_note = (
                            f" — layer **{layers[0]}**" if len(layers) == 1 else ""
                        )
                        st.info(
                            f"📍 Detected: **{fmt}** — "
                            f"{_geojson_geometry_summary(preview_vector_gdf)}"
                            f"{layer_note}"
                        )
                        if st.session_state.fusion_outcome_columns:
                            for i, col in enumerate(
                                st.session_state.fusion_outcome_columns
                            ):
                                row_l, row_r = st.columns([4, 1])
                                with row_l:
                                    st.text(f"Outcome {i + 1}: {col}")
                                with row_r:
                                    if st.button(
                                        "❌",
                                        key=f"fusion_outcome_remove_{i}",
                                        help="Remove this outcome column",
                                    ):
                                        st.session_state.fusion_outcome_columns.pop(i)
                                        st.rerun()

                        remaining = [
                            c
                            for c in numeric_cols
                            if c not in st.session_state.fusion_outcome_columns
                        ]
                        if remaining:
                            st.selectbox(
                                "Add outcome column",
                                options=[_FUSION_OUTCOME_ADD_PLACEHOLDER] + remaining,
                                key="fusion_add_outcome_column",
                                on_change=_fusion_append_outcome_callback,
                                help=(
                                    "Add each numeric outcome in order; after each "
                                    "choice the list updates for the next column. "
                                    "Each selection is used as an optimization target."
                                ),
                            )
                        elif not numeric_cols:
                            st.warning("No numeric columns found in this target.")
                        elif not st.session_state.fusion_outcome_columns:
                            st.warning("No numeric columns available to add.")

                        target_outcome_columns = list(
                            st.session_state.fusion_outcome_columns
                        )
                        if len(target_outcome_columns) > 1:
                            multi_objective_requested = st.checkbox(
                                "Multi-objective optimization run",
                                value=False,
                                help=(
                                    "When enabled, requests a joint optimization across all "
                                    "selected outcomes. Full multi-objective fusion is not "
                                    "available yet; runs stay sequential until implemented."
                                ),
                                key="fusion_multi_objective_run",
                            )
                    except Exception as e:
                        st.error(f"Error loading vector target: {e}")
                        tmp_target_path = None
                        target_mat = None
                        preview_vector_gdf = None
                        is_vector_target = False
                        is_raster_target = False

                elif is_raster_target:
                    try:
                        with rasterio.open(tmp_target_path) as src:
                            n_bands = src.count
                        st.info(f"🗺️ Detected: **GeoTIFF** with {n_bands} band(s)")
                        target_band = st.number_input(
                            "Outcome band",
                            min_value=1,
                            max_value=n_bands,
                            value=min(
                                int(st.session_state.get("fusion_target_band", 1)),
                                n_bands,
                            ),
                            help="Raster band used as the outcome surface.",
                            key="fusion_target_band",
                        )
                    except Exception as e:
                        st.error(f"Error loading GeoTIFF: {e}")
                        tmp_target_path = None
                        target_mat = None
                        is_raster_target = False

    with col_fusion_right:
        st.subheader("Target Preview")

        if target_uploads and tmp_target_path:
            m_fusion_preview = folium.Map(location=[51.0447, -114.0719], zoom_start=10)

            try:
                if is_vector_target and preview_vector_gdf is not None:
                    preview_gdf = sanitize_gdf_attributes_for_json(preview_vector_gdf)
                    numeric_cols = preview_gdf.select_dtypes(
                        include=[np.number]
                    ).columns.tolist()
                    geom_only = "— Outline only —"
                    preview_pick = st.selectbox(
                        "Preview value column",
                        options=[geom_only] + numeric_cols,
                        key="fusion_preview_value_column",
                        help="Colour map features by this numeric column, or outline only.",
                    )
                    preview_feature = (
                        None if preview_pick == geom_only else preview_pick
                    )

                    if preview_feature and preview_feature in preview_gdf.columns:
                        vals = preview_gdf[preview_feature].dropna()
                        if len(vals) > 0:
                            if not _add_outcome_geometry_preview(
                                m_fusion_preview,
                                preview_gdf,
                                preview_feature,
                            ):
                                add_mixed_geojson_preview(m_fusion_preview, preview_gdf)
                        else:
                            add_mixed_geojson_preview(m_fusion_preview, preview_gdf)
                    else:
                        add_mixed_geojson_preview(m_fusion_preview, preview_gdf)

                    bounds = preview_gdf.total_bounds
                    m_fusion_preview.fit_bounds(
                        [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
                    )

                elif is_raster_target:
                    with rasterio.open(tmp_target_path) as src:
                        n_bands_preview = src.count
                    preview_band = st.number_input(
                        "Preview band",
                        min_value=1,
                        max_value=max(1, n_bands_preview),
                        value=min(
                            int(st.session_state.get("fusion_target_band", 1)),
                            n_bands_preview,
                        ),
                        help="Band shown on the map (can differ from the outcome band on the left).",
                        key="fusion_preview_raster_band",
                    )
                    with rasterio.open(tmp_target_path) as src:
                        n_bands = src.count
                        pb = int(min(max(1, preview_band), n_bands))
                        arr = src.read(pb)
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
                            _add_fusion_vertical_scale_to_map(
                                m_fusion_preview, float(vmin), float(vmax)
                            )

                st_folium(
                    m_fusion_preview,
                    width="100%",
                    height=400,
                    key="fusion_preview_map",
                    returned_objects=[],
                )

            except Exception as e:
                st.error(f"Preview error: {e}")
        else:
            st.info("Upload a target file to preview")

    # =========================================================================
    # Full width: metric source and optimization (below the split preview row)
    # =========================================================================
    st.divider()
    st.subheader("Metric Configuration")
    with st.container(border=True):

        metric_mode = st.radio(
            "Metric Source",
            options=["Use Loaded Results", "Upload Files", "Auto-Download"],
            horizontal=True,
            help=(
                "Use outputs already in the output folder, upload GVI/NDVI files, or "
                "download metrics at run time. Buffer ladders, loaded-result picks, "
                "auto-download fields, and optimization apply when you press "
                "Run Fusion Optimization. Uploading new metric files still refreshes the app."
            ),
            key="fusion_metric_source",
        )

        gvi_path = None
        ndvi_path = None

        if metric_mode == "Upload Files":
            col_gvi_up, col_ndvi_up = st.columns(2)

            with col_gvi_up:
                gvi_uploads = st.file_uploader(
                    "🌿 Upload GVI File",
                    accept_multiple_files=True,
                    type=FUSION_TARGET_UPLOAD_TYPES,
                    key="fusion_gvi_upload",
                    help="GeoTIFF or vector metric. Shapefile requires all sidecars in one selection.",
                )
                if gvi_uploads:
                    try:
                        gvi_ds = materialize_uploaded_dataset(gvi_uploads)
                        gvi_path = gvi_ds.path
                        st.success(f"✓ Loaded: {gvi_ds.display_name}")
                    except ValueError as e:
                        st.error(str(e))
                        gvi_path = None

            with col_ndvi_up:
                ndvi_uploads = st.file_uploader(
                    "🛰️ Upload NDVI File",
                    accept_multiple_files=True,
                    type=FUSION_TARGET_UPLOAD_TYPES,
                    key="fusion_ndvi_upload",
                    help="GeoTIFF or vector metric. Shapefile requires all sidecars in one selection.",
                )
                if ndvi_uploads:
                    try:
                        ndvi_ds = materialize_uploaded_dataset(ndvi_uploads)
                        ndvi_path = ndvi_ds.path
                        st.success(f"✓ Loaded: {ndvi_ds.display_name}")
                    except ValueError as e:
                        st.error(str(e))
                        ndvi_path = None

        st.markdown("**GVI buffer exploration (m)**")
        col_bgvi_a, col_bgvi_b, col_bgvi_c = st.columns(3)
        with col_bgvi_a:
            gvi_buffer_min_m = st.number_input(
                "GVI minimum buffer",
                min_value=50,
                max_value=4900,
                value=100,
                step=50,
                help=(
                    "Smallest GVI radius searched (m). Used with the NDVI buffer ladder "
                    "for fusion; values apply when you run optimization below."
                ),
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

        if metric_mode == "Use Loaded Results":
            all_gvi_files = _scan_metric_files(output_dir, "gvi")
            all_ndvi_files = _scan_metric_files(output_dir, "ndvi")

            buffered_extent = None
            if tmp_target_path:
                buffered_extent = _compute_buffered_extent(
                    tmp_target_path,
                    is_vector_target,
                    buffer_extent_m,
                    target_layer_for_engine if is_vector_target else None,
                )

            def _filter_by_coverage(file_list, bext):
                # Return (covering, non_covering) label lists.
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
                    st.info(
                        "No GVI GeoTIFF results found in the output folder "
                        "(files named *_gvi.tif)."
                    )
                else:
                    gvi_help = (
                        "GeoTIFFs named *_gvi.tif in the output folder. "
                        "Leave unset to auto-download if needed."
                    )
                    if buffered_extent is not None:
                        gvi_help += (
                            f" {len(gvi_covering)} file(s) fully cover the buffered "
                            f"target extent."
                        )
                        if gvi_outside:
                            gvi_help += (
                                f" {len(gvi_outside)} file(s) do not cover that extent."
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
                        help=gvi_help,
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
                    st.info(
                        "No NDVI GeoTIFF results found in the output folder "
                        "(files named *_ndvi.tif)."
                    )
                else:
                    ndvi_help = (
                        "GeoTIFFs named *_ndvi.tif in the output folder. "
                        "Leave unset to auto-download if needed."
                    )
                    if buffered_extent is not None:
                        ndvi_help += (
                            f" {len(ndvi_covering)} file(s) fully cover the buffered "
                            f"target extent."
                        )
                        if ndvi_outside:
                            ndvi_help += f" {len(ndvi_outside)} file(s) do not cover that extent."
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
                        help=ndvi_help,
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

        ndvi_auto_start = date(2023, 6, 1)
        ndvi_auto_end = date(2023, 9, 30)
        cache_metrics = False
        ndvi_resolution_m = None
        gvi_grid_spacing_m = None

        if metric_mode == "Auto-Download":
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
            st.text_input(
                "Street View API Key (optional)",
                type="password",
                autocomplete="off",
                help="Optional Google Street View key; leave blank for built-in access. Masked input with autocomplete disabled (some browsers may still offer to save).",
                key="fusion_streetview_api_key",
            )
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

    st.subheader("Optimization Settings")

    # Numeric attribute columns the user can pick as covariates. Outcomes are
    # excluded because a column can't predict itself — same defensive check the
    # runner applies per-outcome at submit time. Only available for vector
    # targets (raster targets have no attribute table).
    available_covariates: list[str] = []
    if is_vector_target and preview_vector_gdf is not None:
        numeric_attr_cols = preview_vector_gdf.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        outcome_set = set(target_outcome_columns)
        available_covariates = [c for c in numeric_attr_cols if c not in outcome_set]

    with st.form("fusion_metric_run"):
        with st.container(border=True):

            col_cgi1, col_cgi2 = st.columns([1, 2])
            with col_cgi1:
                cgi_formula = st.selectbox(
                    "CGI Formula",
                    options=["weighted_average", "synergy"],
                    index=0,
                    help=(
                        "**weighted_average** — three weights on min-max-"
                        "normalized veg / terrain / NDVI (sum = 100). "
                        "**synergy** — three-metric generalisation of Wang et "
                        "al. 2026: seven weights (sum = 1) plus three powers "
                        "on the main NDVI / Veg / Terrain terms only "
                        "(interactions stay plain). The optimizer searches "
                        "whichever parameter shape you pick."
                    ),
                    key="fusion_cgi_formula",
                )
            with col_cgi2:
                if is_vector_target:
                    objective_metric_now = st.session_state.get(
                        "fusion_objective_metric", "pearson"
                    )
                    cov_help = (
                        "Additional numeric attribute columns to **control for** "
                        "when scoring the CGI's predictive power. With "
                        "covariates the score becomes the greenery term's "
                        "*partial* contribution (partial correlation, "
                        "incremental R², or full-model RMSE depending on the "
                        "objective metric). "
                        "**`mutual_info` ignores covariates by design** — "
                        "conditional MI is hard to estimate from binned data, "
                        "so the score stays the raw greenery↔outcome MI "
                        "regardless of what you pick here. Outcomes you "
                        "already selected are filtered out (a column can't "
                        "predict itself); duplicate selections are de-duplicated."
                    )
                    covariate_columns = st.multiselect(
                        "Covariates (control variables)",
                        options=available_covariates,
                        default=[],
                        help=cov_help,
                        key="fusion_covariate_columns",
                        disabled=not available_covariates,
                    )
                    if not available_covariates:
                        st.caption(
                            "_No numeric attribute columns available outside the "
                            "chosen outcomes._"
                        )
                    if objective_metric_now == "mutual_info" and covariate_columns:
                        st.caption(
                            "ℹ️ The selected covariates will be **ignored** while "
                            "the objective metric is `mutual_info`."
                        )
                else:
                    covariate_columns = []
                    st.caption(
                        "_Covariates are only supported for vector targets — "
                        "raster targets have no attribute table to draw from._"
                    )

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
                resume_existing_study = st.checkbox(
                    "Resume previous study if exists",
                    value=True,
                    key="fusion_resume_study",
                    help=(
                        "When on, re-running with the same target + outcome + "
                        "objective metric loads the existing SQLite study under "
                        "output_results/fusion_studies/ and runs only the "
                        "remaining trials. Turn off to start a fresh study."
                    ),
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

            run_standalones = st.checkbox(
                "Also optimize each metric on its own (NDVI / Vegetation / Terrain)",
                value=False,
                key="fusion_run_standalones",
                help=(
                    "Adds three single-metric Optuna studies alongside the "
                    "combined CGI run, searching only its radius + aggregation. "
                    "Reuses the same train/val/test split and the per-job "
                    "pre-aggregation cache."
                ),
            )

        st.divider()
        _fus_run_spacer, _fus_run_col = st.columns([2.2, 1])
        with _fus_run_col:
            fusion_run_clicked = st.form_submit_button(
                "🚀 Run Fusion Optimization",
                type="primary",
                use_container_width=True,
                key="fusion_form_run_submit",
            )

    if metric_mode != "Auto-Download":
        ndvi_auto_start = date(2023, 6, 1)
        ndvi_auto_end = date(2023, 9, 30)
        cache_metrics = False
        ndvi_resolution_m = None
        gvi_grid_spacing_m = None

    # =========================================================================
    # Run controls and progress
    # =========================================================================
    st.divider()

    col_run2, col_run3 = st.columns([1, 1])
    with col_run2:
        if st.session_state.fusion_results:
            if st.button(
                "📊 Export Results", use_container_width=True, key="fusion_export"
            ):
                bundle, _eng = _fusion_resolve_active_bundle()
                if bundle is None or bundle.get("composite_df") is None:
                    st.error("No composite table available to export.")
                else:
                    result_df = bundle["composite_df"]
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
        if st.session_state.fusion_results:
            if st.button("🔄 Reset", use_container_width=True, key="fusion_reset"):
                st.session_state.fusion_engine = None
                st.session_state.fusion_results = None
                st.session_state.fusion_engines_by_target = {}
                st.rerun()

    if fusion_run_clicked:
        if not target_uploads or not tmp_target_path:
            st.error("❌ Please upload a target file")
        elif is_vector_target and not target_outcome_columns:
            st.error(
                "❌ Add at least one outcome column from the vector target attributes."
            )
        elif (
            gvi_buffer_min_m > gvi_buffer_max_m or ndvi_buffer_min_m > ndvi_buffer_max_m
        ):
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
                fusion_multi_objective = (
                    multi_objective_requested
                    if is_vector_target and len(target_outcome_columns) > 1
                    else False
                )
                with st.expander("Configuration Summary", expanded=True):
                    st.write(f"**Target:** {target_display_name or 'unknown'}")
                    if is_vector_target:
                        st.write(
                            "**Outcomes:** "
                            + ", ".join(f"`{c}`" for c in target_outcome_columns)
                        )
                        if len(target_outcome_columns) > 1:
                            st.write(
                                "**Multi-objective optimization run:** "
                                f"{'Yes' if fusion_multi_objective else 'No'} "
                                "(joint optimization not available yet)"
                            )
                    else:
                        st.write(
                            "**Outcome band:** "
                            f"{int(st.session_state.get('fusion_target_band', target_band))}"
                        )
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

                from services import get_job_executor, get_job_store

                store = get_job_store()
                executor = get_job_executor()

                gvi_api_key = (
                    (st.session_state.get("fusion_streetview_api_key") or None)
                    if metric_mode == "Auto-Download"
                    else None
                )
                ndvi_project_id = None
                job_target_band = int(
                    st.session_state.get("fusion_target_band", target_band)
                )

                # Persist the full re-runnable config so the fusion restart
                # panel can resubmit a stopped job with identical settings.
                # Temp paths (target_path, uploaded metric temp files) and the
                # API key are intentionally omitted — re-upload / re-supply on
                # restart. For metric files chosen from the output folder
                # ("Use Loaded Results"), we store the basename + a flag and
                # re-join against the current output_dir at restart time.
                def _under_dir(path: str | None, base: str) -> bool:
                    if not path:
                        return False
                    return os.path.dirname(os.path.abspath(path)).rstrip(
                        os.sep
                    ) == os.path.abspath(base).rstrip(os.sep)

                target_geom_sha = (
                    geometry_sha256(preview_vector_gdf)
                    if is_vector_target and preview_vector_gdf is not None
                    else None
                )

                fusion_record = store.submit(
                    type="fusion",
                    name=os.path.splitext(target_display_name)[0],
                    params={
                        "target_display_name": target_display_name,
                        "is_vector_target": is_vector_target,
                        "outcome_columns": list(target_outcome_columns),
                        "target_band": job_target_band,
                        "target_layer": target_layer_for_engine,
                        "geometry_sha256": target_geom_sha,
                        "n_trials": n_trials,
                        "n_startup_trials": n_startup_trials,
                        "objective_metric": objective_metric,
                        "sampler_type": optimizer,
                        "pruner_type": pruner_type,
                        "multi_objective_requested": fusion_multi_objective,
                        "resume_existing_study": resume_existing_study,
                        # CGI formula + covariate columns persist into rec.params
                        # so the restart panel can re-run with the same config.
                        "cgi_formula": cgi_formula,
                        "covariate_columns": list(covariate_columns or []),
                        # Standalone single-metric studies, expanded from the
                        # single UI checkbox into the explicit channel list the
                        # runner expects. Empty list = CGI only.
                        "standalone_channels": (
                            ["veg", "terrain", "ndvi"] if run_standalones else []
                        ),
                        # Spatial / sampling config — recreates the same engine
                        # build and pre-aggregation cache fingerprint on restart.
                        "buffer_meters": float(buffer_extent_m),
                        "gvi_buffer_min_m": float(gvi_buffer_min_m),
                        "gvi_buffer_max_m": float(gvi_buffer_max_m),
                        "gvi_buffer_step_m": float(gvi_buffer_step_m),
                        "ndvi_buffer_min_m": float(ndvi_buffer_min_m),
                        "ndvi_buffer_max_m": float(ndvi_buffer_max_m),
                        "ndvi_buffer_step_m": float(ndvi_buffer_step_m),
                        "ndvi_resolution_m": ndvi_resolution_m,
                        "gvi_grid_spacing_m": gvi_grid_spacing_m,
                        "n_bins": int(n_bins),
                        "cache_metrics": bool(cache_metrics),
                        "test_size": float(test_size),
                        "k_folds": int(k_folds),
                        "ndvi_start_date": ndvi_auto_start.isoformat(),
                        "ndvi_end_date": ndvi_auto_end.isoformat(),
                        # Metric source + how to re-resolve metric files on
                        # restart. Temp uploads can't be re-resolved without
                        # the original session; the restart panel asks the
                        # user to re-upload in that case.
                        "metric_mode": metric_mode,
                        "gvi_basename": (
                            os.path.basename(gvi_path) if gvi_path else None
                        ),
                        "ndvi_basename": (
                            os.path.basename(ndvi_path) if ndvi_path else None
                        ),
                        "gvi_under_output_dir": _under_dir(gvi_path, output_dir),
                        "ndvi_under_output_dir": _under_dir(ndvi_path, output_dir),
                        "has_api_key": bool(gvi_api_key),
                    },
                )
                executor.submit_runner(
                    fusion_record,
                    run_fusion,
                    target_path=tmp_target_path,
                    target_features_geojson=(
                        tuple(target_outcome_columns) if is_vector_target else ()
                    ),
                    target_band=job_target_band if is_raster_target else 1,
                    target_layer=(
                        target_layer_for_engine if is_vector_target else None
                    ),
                    target_cleanup_dir=(target_mat.cleanup_dir if target_mat else None),
                    target_cleanup_file=(
                        target_mat.cleanup_file if target_mat else None
                    ),
                    buffer_meters=buffer_extent_m,
                    gvi_buffer_min_m=gvi_buffer_min_m,
                    gvi_buffer_max_m=gvi_buffer_max_m,
                    gvi_buffer_step_m=gvi_buffer_step_m,
                    ndvi_buffer_min_m=ndvi_buffer_min_m,
                    ndvi_buffer_max_m=ndvi_buffer_max_m,
                    ndvi_buffer_step_m=ndvi_buffer_step_m,
                    ndvi_resolution_m=ndvi_resolution_m,
                    gvi_grid_spacing_m=gvi_grid_spacing_m,
                    n_bins=n_bins,
                    veg_path=gvi_path,
                    terrain_path=None,
                    ndvi_path=ndvi_path,
                    cache_metrics=cache_metrics,
                    test_size=test_size,
                    k_folds=k_folds,
                    n_trials=n_trials,
                    n_startup_trials=n_startup_trials,
                    objective_metric=objective_metric,
                    pruner_type=pruner_type,
                    sampler_type=optimizer,
                    gvi_api_key=gvi_api_key,
                    ndvi_start_date=ndvi_auto_start.isoformat(),
                    ndvi_end_date=ndvi_auto_end.isoformat(),
                    ndvi_project_id=ndvi_project_id,
                    multi_objective_requested=fusion_multi_objective,
                    output_dir=output_dir,
                    MetricFusionEngine=MetricFusionEngine,
                    target_display_name=target_display_name,
                    resume_existing_study=resume_existing_study,
                    cgi_formula=cgi_formula,
                    covariate_columns=list(covariate_columns or []),
                    standalone_channels=(
                        ["veg", "terrain", "ndvi"] if run_standalones else []
                    ),
                )

                st.success("✅ Fusion job started! Check sidebar for progress.")

    # Pull completed fusion results from the JobStore into session state for display.
    from services import get_job_store as _get_fusion_store

    _fusion_store = _get_fusion_store()
    for rec in _fusion_store.list_terminal():
        if (
            rec.type == "fusion"
            and rec.status == "completed"
            and rec.extra.get("results") is not None
            and st.session_state.fusion_results is None
        ):
            st.session_state.fusion_engine = rec.extra.get("engine")
            st.session_state.fusion_engines_by_target = (
                rec.extra.get("engines_by_target") or {}
            )
            st.session_state.fusion_results = rec.extra["results"]
            break

    # =========================================================================
    # ROW 3: Results Display
    # =========================================================================
    if st.session_state.fusion_results:
        st.divider()
        st.subheader("Optimization Results")

        results = st.session_state.fusion_results
        if results.get("mode") == "multi":
            st.selectbox(
                "Select outcome",
                options=results["ordered_labels"],
                key="fusion_results_outcome_pick",
            )

        results_view, engine = _fusion_resolve_active_bundle()
        if engine is None or results_view is None:
            st.warning(
                "Optimization details are not available for the selected outcome."
            )
        else:
            metric_name = results_view["objective_metric"].upper()
            best_params = results_view.get("best_params") or {}

            # Formula introspection. Legacy results from before the formula
            # registry don't carry the attribute → fall back to weighted_average
            # so old studies still render with the original three weights.
            formula_name = getattr(engine, "cgi_formula", "weighted_average")
            try:
                formula = _cgi_formulas.get_formula(formula_name)
            except ValueError:
                formula = _cgi_formulas.get_formula("weighted_average")

            covariates_used = list(getattr(engine, "covariate_columns", []) or [])

            # ── Top tile row ───────────────────────────────────────────────
            # For weighted_average we keep the legacy 3-weight tiles so the
            # display matches the user's mental model. For synergy a 7-weight
            # tile row would be unreadable, so we report the dominant main
            # weight + main-term power range instead, and let the JSON / table
            # below carry the full breakdown.
            if formula.name == _cgi_formulas.WEIGHTED_AVERAGE:
                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                total_weight = sum(
                    float(best_params.get(k, 0)) for k in formula.weight_keys
                )
                with col_m1:
                    pct = (
                        100.0 * float(best_params.get("veg_weight", 0)) / total_weight
                        if total_weight > 0
                        else 0.0
                    )
                    st.metric("Vegetation Weight", f"{pct:.1f}%")
                with col_m2:
                    pct = (
                        100.0
                        * float(best_params.get("terrain_weight", 0))
                        / total_weight
                        if total_weight > 0
                        else 0.0
                    )
                    st.metric("Terrain Weight", f"{pct:.1f}%")
                with col_m3:
                    pct = (
                        100.0 * float(best_params.get("ndvi_weight", 0)) / total_weight
                        if total_weight > 0
                        else 0.0
                    )
                    st.metric("NDVI Weight", f"{pct:.1f}%")
                with col_m4:
                    st.metric(
                        f"Best {metric_name}", f"{results_view['best_value']:.4f}"
                    )
                with col_m5:
                    if results_view["robust_trials"]:
                        st.metric(
                            "Robust Trials",
                            f"{len(results_view['robust_trials'])}/{len(engine.study.trials)}",
                        )
                    else:
                        st.metric("Total Trials", len(engine.study.trials))
            else:
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("CGI Formula", formula.name)
                with col_m2:
                    main_weights = {
                        k: float(best_params.get(k, 0))
                        for k in formula.main_weight_keys
                    }
                    dom_key = max(main_weights, key=main_weights.get)
                    dom_label = dom_key.removeprefix("w_").upper()
                    st.metric(
                        "Dominant Main Term",
                        f"{dom_label} ({main_weights[dom_key]*100:.1f}%)",
                    )
                with col_m3:
                    st.metric(
                        f"Best {metric_name}", f"{results_view['best_value']:.4f}"
                    )
                with col_m4:
                    if results_view["robust_trials"]:
                        st.metric(
                            "Robust Trials",
                            f"{len(results_view['robust_trials'])}/{len(engine.study.trials)}",
                        )
                    else:
                        st.metric("Total Trials", len(engine.study.trials))

            # Formula label + the list of controlled covariates surfaced
            # below the tile row so the user can read what the score means.
            st.caption(
                f"**Formula:** `{formula.name}` · **Covariates:** "
                + (
                    ", ".join(f"`{c}`" for c in covariates_used)
                    if covariates_used
                    else "_none_"
                )
                + (
                    "  ·  ℹ️ `mutual_info` ignores covariates"
                    if results_view["objective_metric"] == "mutual_info"
                    and covariates_used
                    else ""
                )
            )

            col_detail1, col_detail2 = st.columns(2)

            with col_detail1:
                st.markdown("**Best Trial Details**")
                best_trial = engine.study.best_trial

                info_data: dict = {
                    "Trial Number": best_trial.number,
                    "Buffer Distance": f"{engine.buffer_meters}m",
                }
                # Formula-driven weight + power dump, then shared spatial params,
                # then the train/val scores + p-values.
                for k in formula.weight_keys:
                    info_data[f"{k} (raw)"] = best_params.get(k, "N/A")
                for k in formula.power_keys:
                    info_data[k] = best_params.get(k, "N/A")
                for k in ("veg_radius", "terrain_radius", "ndvi_radius"):
                    info_data[k] = f"{best_params.get(k, 'N/A')}m"
                if covariates_used:
                    info_data["Covariates controlled"] = covariates_used

                info_data[f"Train {metric_name}"] = (
                    f"{best_trial.user_attrs.get('train_score_mean', 'N/A')}"
                )
                info_data[f"Val {metric_name}"] = (
                    f"{best_trial.user_attrs.get('val_score_mean', 'N/A')}"
                )

                if "train_pvalue" in best_trial.user_attrs:
                    info_data["Train p-value"] = (
                        f"{best_trial.user_attrs['train_pvalue']:.4e}"
                    )
                if "val_pvalue" in best_trial.user_attrs:
                    info_data["Val p-value"] = (
                        f"{best_trial.user_attrs['val_pvalue']:.4e}"
                    )

                st.json(info_data)

            with col_detail2:
                st.markdown("**Optimization History**")

                trial_values = [
                    t.value for t in engine.study.trials if t.value is not None
                ]
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

            if results_view["robust_trials"]:
                st.divider()
                st.markdown("**Robust Trials (Statistically Significant)**")

                # One column per formula weight (renormalised %), plus the
                # main-term powers (synergy only). Train / test scores +
                # p-values stay shared. Building columns from
                # ``formula.weight_keys`` / ``power_keys`` means the table
                # can't drift away from the registry as new formulas land.
                def _wlabel(key: str) -> str:
                    # ``ndvi_weight`` → "NDVI", ``w_ndvi_veg`` → "NDVI·VEG".
                    cleaned = key.removeprefix("w_").removesuffix("_weight")
                    return cleaned.replace("_", "·").upper() + " %"

                def _plabel(key: str) -> str:
                    # ``ndvi_power`` → "NDVI p".
                    return key.removesuffix("_power").upper() + " p"

                robust_data: list[dict] = []
                for t in results_view["robust_trials"][:10]:
                    row: dict = {"Trial": t.number}
                    weight_total = sum(
                        float(t.params.get(k, 0)) for k in formula.weight_keys
                    )
                    for k in formula.weight_keys:
                        row[_wlabel(k)] = (
                            f"{100.0 * float(t.params.get(k, 0)) / weight_total:.1f}"
                            if weight_total > 0
                            else "0.0"
                        )
                    for k in formula.power_keys:
                        row[_plabel(k)] = f"{float(t.params.get(k, 1.0)):.2f}"
                    row[f"Train {metric_name}"] = (
                        f"{t.user_attrs.get('train_score', 0):.4f}"
                    )
                    row[f"Test {metric_name}"] = (
                        f"{t.user_attrs.get('test_score', 0):.4f}"
                    )
                    row["Train p"] = f"{t.user_attrs.get('train_pvalue', 1):.4e}"
                    row["Test p"] = f"{t.user_attrs.get('test_pvalue', 1):.4e}"
                    robust_data.append(row)

                st.dataframe(robust_data, use_container_width=True)

            # ── CGI vs single-metric standalones (when enabled) ─────────────
            # One row per study (CGI + each enabled standalone). Headline
            # numbers come straight from the bundle the runner built so this
            # block stays Streamlit-only — no engine access required.
            standalones = results_view.get("standalones") or {}
            if standalones:
                st.divider()
                st.markdown("**CGI vs Standalone Single-Metric Studies**")

                def _fmt_score(v) -> str:
                    try:
                        return f"{float(v):.4f}"
                    except (TypeError, ValueError):
                        return "—"

                def _fmt_pval(v) -> str:
                    if v is None:
                        return "—"
                    try:
                        return f"{float(v):.4e}"
                    except (TypeError, ValueError):
                        return "—"

                cgi_test = results_view.get("test_results") or {}
                cmp_rows: list[dict] = [
                    {
                        "Study": "CGI (combined)",
                        f"CV val {metric_name}": _fmt_score(
                            results_view.get("best_value")
                        ),
                        f"Test {metric_name}": _fmt_score(cgi_test.get("test_score")),
                        "Test p": _fmt_pval(cgi_test.get("test_pvalue")),
                        "Robust trials": len(results_view.get("robust_trials") or []),
                    }
                ]
                channel_display = {
                    "veg": "Vegetation",
                    "terrain": "Terrain",
                    "ndvi": "NDVI",
                }
                for ch_key, ch_bundle in standalones.items():
                    ch_test = ch_bundle.get("test_results") or {}
                    cmp_rows.append(
                        {
                            "Study": (
                                channel_display.get(ch_key, ch_key) + " (standalone)"
                            ),
                            f"CV val {metric_name}": _fmt_score(
                                ch_bundle.get("best_value")
                            ),
                            f"Test {metric_name}": _fmt_score(
                                ch_test.get("test_score")
                            ),
                            "Test p": _fmt_pval(ch_test.get("test_pvalue")),
                            "Robust trials": len(ch_bundle.get("robust_trials") or []),
                        }
                    )
                st.dataframe(cmp_rows, use_container_width=True)
