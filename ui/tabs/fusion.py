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
import pandas as pd
import rasterio
import streamlit as st
from branca.element import MacroElement
from helpers import (
    FUSION_TARGET_UPLOAD_TYPES,
    RESTART_SESSION_KEY,
    file_size_mtime_fingerprint,
    materialize_uploaded_dataset,
    path_drift_status,
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

# Canonical display labels for greenery channels. Internal keys stay
# ``veg``/``terrain``/``ndvi`` everywhere in code and JSON payloads;
# anything user-visible (UI labels, captions, log lines surfaced to
# users) goes through this mapping so the spelling is uniform.
_CHANNEL_DISPLAY = {"veg": "Vegetation", "terrain": "Terrain", "ndvi": "NDVI"}


class _FusionVerticalScaleControl(MacroElement):
    """Leaflet control: vertical red→yellow→green strip with numeric bounds."""

    _template = Template(
        """
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
"""
    )

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
    lon_payload = p.get("longitudinal_spec_payload") or None
    k_folds = int(p.get("k_folds", 5))
    cv_label = f"{k_folds}-fold CV" if k_folds > 1 else "single split (no CV)"
    lines = [
        f"**Target:** `{p.get('target_display_name', '?')}`",
        f"**Outcomes:** {', '.join(p.get('outcome_columns') or []) or '—'}",
        f"**CGI formula:** `{p.get('cgi_formula') or 'weighted_average'}`",
        f"**Covariates:** {', '.join(covs) if covs else '—'}",
        f"**Standalone metrics:** "
        f"{', '.join(_CHANNEL_DISPLAY.get(s, s) for s in standalones) if standalones else '—'}",
        f"**Trials:** {p.get('n_trials', '?')} "
        f"(startup {p.get('n_startup_trials', '?')}, {cv_label})",
        f"**Objective:** {p.get('objective_metric', '?')} · "
        f"**Sampler:** {p.get('sampler_type', '?')}",
        f"**GVI buffers (m):** {p.get('gvi_buffer_min_m', '?')} – "
        f"{p.get('gvi_buffer_max_m', '?')} (step {p.get('gvi_buffer_step_m', '?')})",
        f"**NDVI buffers (m):** {p.get('ndvi_buffer_min_m', '?')} – "
        f"{p.get('ndvi_buffer_max_m', '?')} (step {p.get('ndvi_buffer_step_m', '?')})",
    ]
    if lon_payload:
        derived = lon_payload.get("derive_wave_from_date")
        descriptor = (
            "year-aware cross-sectional (OLS scorer)"
            if derived
            else f"`{lon_payload.get('intake_mode')}` intake"
        )
        lines.append(
            f"**Year/wave-aware:** {descriptor}, "
            f"waves={lon_payload.get('wave_labels')}, "
            f"scoring=`{lon_payload.get('scoring_metric')}`"
        )
    cgi_grid = p.get("cgi_grid_spacing_m")
    if cgi_grid is not None:
        scaling_scope = "whole-grid" if p.get("whole_grid_scaling") else "per-fold"
        split_kind = (
            "area-balanced" if p.get("area_balanced_split") else "count-balanced"
        )
        lines.append(
            f"**Polygon scoring:** per-pixel CGI · pixel size "
            f"{cgi_grid} m · {scaling_scope} scaling · {split_kind} split"
        )
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
        n_bins=int(p.get("n_bins", 5)),
        veg_path=veg_path,
        ndvi_path=ndvi_path,
        test_size=float(p.get("test_size", 0.3)),
        val_size=float(p.get("val_size", 0.25)),
        k_folds=int(p.get("k_folds", 5)),
        n_trials=int(p.get("n_trials", 300)),
        n_startup_trials=int(p.get("n_startup_trials", 150)),
        objective_metric=p.get("objective_metric", "pearson"),
        sampler_type=p.get("sampler_type", "TPE"),
        output_dir=output_dir,
        MetricFusionEngine=_MetricFusionEngine,
        target_display_name=p.get("target_display_name") or "target",
        resume_existing_study=True,
        cgi_formula=str(p.get("cgi_formula") or "weighted_average"),
        covariate_columns=list(p.get("covariate_columns") or []),
        standalone_channels=list(p.get("standalone_channels") or []),
        longitudinal_spec_payload=p.get("longitudinal_spec_payload"),
        cgi_grid_spacing_m=(
            float(p["cgi_grid_spacing_m"])
            if p.get("cgi_grid_spacing_m") is not None
            else None
        ),
        whole_grid_scaling=bool(p.get("whole_grid_scaling", False)),
        area_balanced_split=bool(p.get("area_balanced_split", False)),
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
    expected_hash = p.get("geometry_sha256")

    with st.expander(f"↻ Restart fusion job: {rec.name or rec.id}", expanded=True):
        st.caption(
            "The Optuna study, pre-aggregation cache, and metric downloads "
            "are all content-addressed, so a same-config restart resumes "
            "where the previous run stopped. Files that still match the "
            "recorded fingerprint don't need to be re-supplied."
        )
        for line in _fusion_restart_summary_lines(p):
            st.write(line)

        # Silent-restart fast path: every recorded input file is still at
        # its original absolute path with the same fingerprint. Skip every
        # uploader and rerun directly from the stored paths.
        rec_target_path = p.get("target_path")
        rec_target_fp = p.get("target_fingerprint", "")
        if (
            rec_target_path
            and path_drift_status(rec_target_path, rec_target_fp) == "ok"
        ):
            st.success(
                f"✓ Target verified at `{rec_target_path}` — no re-upload " "needed."
            )
            from file_picker import path_to_dataset

            target_mat = path_to_dataset(rec_target_path)
            cancel_col, rerun_col = st.columns([1, 1])
            with cancel_col:
                if st.button(
                    "Cancel restart",
                    key=f"f_restart_cancel_silent_{rec.id}",
                    width="stretch",
                ):
                    st.session_state[RESTART_SESSION_KEY] = None
                    st.rerun()
            with rerun_col:
                if st.button(
                    "Re-run",
                    type="primary",
                    key=f"f_restart_confirm_silent_{rec.id}",
                    width="stretch",
                ):
                    try:
                        _submit_fusion_restart(
                            store,
                            executor,
                            rec,
                            p,
                            target_mat,
                            output_dir,
                            p.get("gvi_path"),
                            p.get("ndvi_path"),
                        )
                    except Exception as e:
                        st.error(f"Re-submission failed: {e}")
                        return True
                    st.session_state[RESTART_SESSION_KEY] = None
                    st.success("Restart submitted. Monitor progress in the sidebar.")
                    st.rerun()
            return True

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

        # Re-resolve metric files. The drift fast path reuses the original
        # absolute paths when their size+mtime still match; otherwise the
        # user re-uploads. Loaded-results / auto-download metric modes were
        # removed from the submission UI, so the only fallback path is
        # re-upload.
        veg_path_re: str | None = None
        ndvi_path_re: str | None = None

        rec_gvi_path = p.get("gvi_path")
        rec_gvi_fp = p.get("gvi_fingerprint", "")
        rec_ndvi_path = p.get("ndvi_path")
        rec_ndvi_fp = p.get("ndvi_fingerprint", "")

        col_g, col_n = st.columns(2)
        with col_g:
            if rec_gvi_path and path_drift_status(rec_gvi_path, rec_gvi_fp) == "ok":
                veg_path_re = rec_gvi_path
                st.success(
                    f"✓ Reusing GVI file from original path: "
                    f"`{os.path.basename(rec_gvi_path)}`"
                )
            else:
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
            if rec_ndvi_path and path_drift_status(rec_ndvi_path, rec_ndvi_fp) == "ok":
                ndvi_path_re = rec_ndvi_path
                st.success(
                    f"✓ Reusing NDVI file from original path: "
                    f"`{os.path.basename(rec_ndvi_path)}`"
                )
            else:
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

        # Mixed-effects restart: validate every per-wave file path against the
        # fingerprint recorded at submit time and prompt the user to re-supply
        # any that have moved or changed. Resubmit blocks until all drift is
        # resolved so the cache can't index against stale-content files.
        lon_payload = p.get("longitudinal_spec_payload") or None
        lon_payload_for_submit = lon_payload
        lon_restart_blocked = False
        if lon_payload:
            stored_fps = lon_payload.get("__file_fingerprints__") or {}
            wave_labels = list(lon_payload.get("wave_labels") or [])
            target_files = dict(lon_payload.get("target_files_per_wave") or {})
            greenery_files = {
                ch: dict(per_wave or {})
                for ch, per_wave in (lon_payload.get("greenery_files") or {}).items()
            }

            def _drift_status(path: str, expected_fp: str) -> str:
                if not path:
                    return "no-path"
                if not os.path.isfile(path):
                    return "missing"
                if expected_fp and file_size_mtime_fingerprint(path) != expected_fp:
                    return "modified"
                return "ok"

            # Walk every (group, wave) entry once to discover drift.
            groups: list[tuple[str, str, dict[str, str]]] = []
            if lon_payload.get("intake_mode") == "wide":
                groups.append(("target", "Per-wave target files", target_files))
            for ch_key, ch_lbl in (
                ("veg", "Vegetation per-wave files"),
                ("terrain", "Terrain per-wave files"),
                ("ndvi", "NDVI per-wave files"),
            ):
                groups.append((ch_key, ch_lbl, greenery_files.get(ch_key) or {}))

            drift_items: list[tuple[str, str, str, str, str]] = []
            for group_key, lbl, files in groups:
                exp_group = stored_fps.get(group_key) or {}
                for wave in wave_labels:
                    path = files.get(wave) or ""
                    status = _drift_status(path, exp_group.get(wave, ""))
                    if status != "ok":
                        drift_items.append((group_key, lbl, wave, path, status))

            if drift_items:
                st.markdown("**Mixed-effects: per-wave files need to be re-supplied**")
                st.caption(
                    "Each entry below either moved, was edited, or never "
                    "existed at the recorded path. Paste the current absolute "
                    "path to resolve. The restart is blocked until every drift "
                    "is fixed."
                )
                resolved_overrides: dict[tuple[str, str], str] = {}
                all_resolved = True
                for group_key, lbl, wave, orig_path, status in drift_items:
                    status_icon = (
                        "❌"
                        if status == "missing"
                        else "⚠️" if status == "modified" else "•"
                    )
                    key_id = f"f_restart_lon_{group_key}_{wave}_{rec.id}"
                    new_path = st.text_input(
                        f"{status_icon} {lbl} — wave `{wave}` ({status})",
                        value=orig_path,
                        key=key_id,
                        help=(
                            f"Original path: `{orig_path or '(none)'}`. "
                            "Paste an absolute path to the replacement file."
                        ),
                    )
                    if new_path and os.path.isfile(new_path):
                        resolved_overrides[(group_key, wave)] = new_path
                    else:
                        all_resolved = False

                if not all_resolved:
                    st.warning("Resolve every drifted file above before re-running.")
                    lon_restart_blocked = True
                else:
                    # Build a fresh payload with updated paths + fingerprints
                    # for the storage layer's next restart cycle.
                    updated_payload = dict(lon_payload)
                    new_target_files = dict(target_files)
                    new_greenery_files = {
                        ch: dict(per_wave) for ch, per_wave in greenery_files.items()
                    }
                    for (gk, wv), new_p in resolved_overrides.items():
                        if gk == "target":
                            new_target_files[wv] = new_p
                        else:
                            new_greenery_files.setdefault(gk, {})[wv] = new_p
                    updated_payload["target_files_per_wave"] = new_target_files
                    updated_payload["greenery_files"] = new_greenery_files
                    new_fps = {
                        gk: dict(stored_fps.get(gk) or {})
                        for gk in ("target", "veg", "terrain", "ndvi")
                    }
                    for (gk, wv), new_p in resolved_overrides.items():
                        new_fps.setdefault(gk, {})[wv] = file_size_mtime_fingerprint(
                            new_p
                        )
                    updated_payload["__file_fingerprints__"] = new_fps
                    lon_payload_for_submit = updated_payload
                    st.success(f"All {len(drift_items)} drifted file(s) resolved.")
            else:
                st.success("Mixed-effects per-wave files verified.")

        cancel_col, rerun_col = st.columns([1, 1])
        with cancel_col:
            if st.button(
                "Cancel restart",
                key=f"f_restart_cancel_{rec.id}",
                width="stretch",
            ):
                st.session_state[RESTART_SESSION_KEY] = None
                st.rerun()
        with rerun_col:
            rerun_clicked = st.button(
                "Verify & re-run",
                type="primary",
                key=f"f_restart_confirm_{rec.id}",
                disabled=lon_restart_blocked,
                width="stretch",
            )
        if rerun_clicked:
            # Substitute the validated payload into ``p`` so the existing
            # ``_submit_fusion_restart`` path picks it up via ``p.get(...)``.
            p_for_submit = dict(p)
            if lon_payload_for_submit is not None:
                p_for_submit["longitudinal_spec_payload"] = lon_payload_for_submit
            try:
                _submit_fusion_restart(
                    store,
                    executor,
                    rec,
                    p_for_submit,
                    target_mat,
                    output_dir,
                    veg_path_re,
                    ndvi_path_re,
                )
            except Exception as e:
                st.error(f"Re-submission failed: {e}")
                return True
            st.session_state[RESTART_SESSION_KEY] = None
            st.success("Restart submitted. Monitor progress in the sidebar.")
            st.rerun()
    return True


# ---------------------------------------------------------------------------
# Section panels (new layout)
# ---------------------------------------------------------------------------

# Public set of cross-sectional objective metrics offered by the OLS scorer
# (covariate-aware partial correlation / incremental R² / RMSE / MI).
_CROSS_METRICS: tuple[str, ...] = (
    "pearson",
    "spearman",
    "r2",
    "rmse",
    "mutual_info",
)
# MixedLM scoring metrics — mirror the engine's MIXEDLM_METRICS so they can
# round-trip through the spec without an explicit import.
_MIXEDLM_METRICS: tuple[str, ...] = (
    "mixedlm_tstat",
    "mixedlm_marginal_r2",
    "mixedlm_lr",
    "mixedlm_coef",
)

# Run-mode placeholder is encoded as None in session-state so the rest of the
# form stays hidden until the user picks a mode (no default).
_FUSION_RUN_MODE_CROSS = "Cross-sectional"
_FUSION_RUN_MODE_LON = "Mixed-effects (longitudinal)"
_FUSION_RUN_MODE_OPTIONS = (_FUSION_RUN_MODE_CROSS, _FUSION_RUN_MODE_LON)


def _date_parseable_columns(gdf: gpd.GeoDataFrame) -> list[str]:
    """Return column names whose values parse as dates via the engine's parser.

    The longitudinal module's :func:`parse_date_column` accepts full ISO,
    year+month, year-only strings, integer years, and native datetime dtypes.
    A column is offered when at least 80 % of its sampled non-null values
    parse successfully — same liberal threshold used today by the outcome
    picker. Only the first 200 non-null rows are sampled per column so the
    scan stays fast on large targets.
    """
    import warnings as _warnings

    try:
        from geofuse.longitudinal import parse_date_column
    except ImportError:
        return []
    out: list[str] = []
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=UserWarning)
        for col in gdf.columns:
            if col == "geometry":
                continue
            s = gdf[col]
            non_null = s.dropna().head(200)
            if not len(non_null):
                continue
            try:
                parsed = parse_date_column(non_null)
            except Exception:
                continue
            if parsed.notna().mean() >= 0.8:
                out.append(col)
    return out


def _discover_years_from_date_column(gdf: gpd.GeoDataFrame, date_col: str) -> list[str]:
    """Return unique years in the date column as sorted string labels."""
    try:
        from geofuse.longitudinal import parse_date_column
    except ImportError:
        return []
    parsed = parse_date_column(gdf[date_col])
    years = sorted(int(d.year) for d in parsed.dropna().unique())
    return [str(y) for y in years]


def _render_optimization_setup_panel(
    preview_gdf: gpd.GeoDataFrame | None,
    target_outcome_columns: list[str],
    *,
    target_file_entries: list[dict] | None = None,
) -> dict | None:
    """Render the optimization-setup section and return the collected state.

    Returns a dict with:
      - ``run_mode`` (str | None)
      - ``is_longitudinal`` (bool)
      - ``intake_mode`` (str | None — only for longitudinal)
      - ``date_col`` (str | None)
      - ``entity_id_col`` (str | None)
      - ``wave_col`` (str | None — long intake)
      - ``discovered_waves`` (list[str]) — empty when none discovered yet
      - ``wide_files`` (list[dict]) — one entry per wide-mode file:
        ``{"path": str, "wave_label": str, "entity_col": str, "date_col": str}``
      - ``cross_sectional_date_on`` (bool)
      - ``mixedlm_random_slope`` (bool)
      - ``mixedlm_time_fixed`` (bool)

    ``target_file_entries`` is the ordered list produced by Section 1's
    multi-file picker (``[{"path", "label"}]``); wide-mode longitudinal
    reads its per-wave paths and labels from there. Long vs wide intake
    is auto-selected from ``len(target_file_entries)``: 1 ⇒ long, 2+ ⇒
    wide.

    Returns ``None`` until the user picks a run mode (no default selection).
    """
    st.subheader("Optimization Setup")

    date_candidates = (
        _date_parseable_columns(preview_gdf) if preview_gdf is not None else []
    )
    all_cols = (
        [c for c in preview_gdf.columns if c != "geometry"]
        if preview_gdf is not None
        else []
    )
    id_candidates = [c for c in all_cols if c not in target_outcome_columns]

    run_mode = st.radio(
        "Run mode",
        options=_FUSION_RUN_MODE_OPTIONS,
        index=None,
        horizontal=True,
        key="fusion_run_mode",
        help="Cross-sectional uses OLS scoring; mixed-effects uses MixedLM with per-entity random effects.",
    )
    if run_mode is None:
        st.caption("_Pick a run mode to configure the rest of the form._")
        return None

    is_longitudinal = run_mode == _FUSION_RUN_MODE_LON

    state: dict = {
        "run_mode": run_mode,
        "is_longitudinal": is_longitudinal,
        "intake_mode": None,
        "date_col": None,
        "entity_id_col": None,
        "wave_col": None,
        "discovered_waves": [],
        "wide_files": [],
        "cross_sectional_date_on": False,
        "mixedlm_random_slope": True,
        "mixedlm_time_fixed": True,
    }

    entries = list(target_file_entries or [])
    n_files = len(entries)

    if not is_longitudinal:
        if n_files > 1:
            st.warning(
                f"Cross-sectional mode uses only the first target file "
                f"(`{entries[0]['label']}`). The remaining "
                f"{n_files - 1} file(s) will be ignored — switch to "
                f"mixed-effects (longitudinal) to use all of them."
            )
        # ── Cross-sectional ──────────────────────────────────────────────
        date_on = st.checkbox(
            "Date column available?",
            value=False,
            key="fusion_cross_date_on",
            help="Enables per-year greenery-file routing. The date column never enters the regression.",
        )
        state["cross_sectional_date_on"] = date_on
        if date_on:
            if not date_candidates:
                st.warning(
                    "No date-parseable columns found in the target. "
                    "Turn the toggle off, or pick a target with a date column."
                )
            else:
                date_col = st.selectbox(
                    "Date column",
                    options=date_candidates,
                    key="fusion_cross_date_col",
                    help="Accepts ISO dates, year+month, year-only, or numeric years.",
                )
                state["date_col"] = date_col
                if preview_gdf is not None and date_col:
                    years = _discover_years_from_date_column(preview_gdf, date_col)
                    state["discovered_waves"] = years
                    if years:
                        st.caption(
                            "Discovered years: " + ", ".join(f"`{y}`" for y in years)
                        )
                    else:
                        st.warning("No parseable dates in the selected column.")
        return state

    # ── Longitudinal ────────────────────────────────────────────────────
    # Intake mode is auto-selected from how many files are in Section 1:
    # one file ⇒ long format (single table with a wave column), two or
    # more ⇒ wide format (one file per wave, joined on the entity id).
    if n_files <= 1:
        intake_mode = "long"
    else:
        intake_mode = "wide"
    state["intake_mode"] = intake_mode
    st.caption(f"Intake mode: **{intake_mode}** ")

    if intake_mode == "long":
        if preview_gdf is None:
            st.info("Upload a target file first to populate the column pickers.")
            return state
        lc1, lc2 = st.columns(2)
        with lc1:
            if id_candidates:
                state["entity_id_col"] = st.selectbox(
                    "Entity ID column",
                    options=id_candidates,
                    key="fusion_lon_entity_id_col",
                    help="Unique entity identifier column.",
                )
            else:
                st.warning("No candidate ID columns found in the target.")
        with lc2:
            if date_candidates:
                state["date_col"] = st.selectbox(
                    "Date column",
                    options=date_candidates,
                    key="fusion_lon_date_col",
                    help="Used to derive `years_since_baseline` per entity.",
                )
            else:
                st.warning("No date-parseable columns found in the target.")
        wave_candidates = [
            c
            for c in all_cols
            if c not in (state["entity_id_col"], state["date_col"])
            and c not in target_outcome_columns
        ]
        if wave_candidates:
            state["wave_col"] = st.selectbox(
                "Wave column",
                options=wave_candidates,
                key="fusion_lon_wave_col",
                help="Wave labels populate the per-channel file-assignment grid.",
            )
            if state["wave_col"]:
                waves = sorted(
                    str(v) for v in preview_gdf[state["wave_col"]].dropna().unique()
                )
                state["discovered_waves"] = waves
                if waves:
                    st.caption(
                        "Discovered waves: " + ", ".join(f"`{w}`" for w in waves)
                    )
        else:
            st.warning("No candidate wave columns found in the target.")
        return state

    # ── Wide / multi-file intake ────────────────────────────────────────
    if not entries:
        st.warning(
            "Pick two or more target files in Section 1 to populate the "
            "per-wave column mapping below."
        )
        state["wide_files"] = []
        state["discovered_waves"] = []
        return state

    st.caption(
        "Each file contributes its own rows; they are joined on the "
        "entity-id column at load time. Wave labels come from Section 1 "
        "(reorder there to change baseline-first ordering); pick the "
        "entity-ID and date columns for each file below."
    )

    from file_picker import path_to_widget_id

    wide_files: list[dict] = []
    for i, entry in enumerate(entries):
        path = entry["path"]
        wave_label = entry["label"]
        wid = path_to_widget_id(path)
        with st.container(border=True):
            header = f"Wave {i + 1} — **{wave_label}**"
            if i == 0:
                header += "  _(baseline)_"
            st.markdown(header, help=path)
            file_cols: list[str] = []
            file_date_cands: list[str] = []
            if not os.path.isfile(path):
                st.error(f"Path no longer exists: `{path}`")
                continue
            try:
                file_gdf = read_vector_path(path)
                file_cols = [c for c in file_gdf.columns if c != "geometry"]
                file_date_cands = _date_parseable_columns(file_gdf)
            except Exception as exc:
                st.error(f"Could not read wave {i + 1}: {exc}")
                continue
            ec1, ec2 = st.columns(2)
            with ec1:
                entity_col = st.selectbox(
                    "Entity ID column",
                    options=file_cols or ["—"],
                    key=f"fusion_lon_wide_entity__{wid}",
                    disabled=not file_cols,
                )
            with ec2:
                date_col = st.selectbox(
                    "Date column",
                    options=file_date_cands or ["—"],
                    key=f"fusion_lon_wide_date__{wid}",
                    disabled=not file_date_cands,
                )
            if wave_label and file_cols and file_date_cands:
                wide_files.append(
                    {
                        "path": path,
                        "wave_label": wave_label,
                        "entity_col": entity_col,
                        "date_col": date_col,
                    }
                )

    state["wide_files"] = wide_files
    state["discovered_waves"] = [f["wave_label"] for f in wide_files]
    if state["discovered_waves"]:
        st.caption(
            "Discovered waves: "
            + ", ".join(f"`{w}`" for w in state["discovered_waves"])
        )
    return state


# Channel display config for the metric-assignment panel.
_FUSION_METRIC_CHANNELS: tuple[tuple[str, str, str], ...] = (
    ("ndvi", "🛰️ NDVI", "ndvi"),
    ("gvi", "🌿 GVI", "gvi"),
)


def _render_metric_assignment_panel(
    discovered_waves: list[str],
    year_aware: bool,
) -> dict | None:
    """Per-channel buffer ladder + per-file year/wave assignment.

    Parameters
    ----------
    discovered_waves
        Ordered list of wave labels (longitudinal) or year strings (cross-
        sectional + date-on). When empty, the panel falls back to one file
        per channel (no per-file multi-select).
    year_aware
        True when ``discovered_waves`` should drive a per-file multi-select.
        False = simple one-file-per-channel mode.

    Returns
    -------
    dict with keys (or ``None`` if upload coverage is incomplete):
      - ``gvi_files``: list of ``(path, list[wave_or_None])``
      - ``ndvi_files``: list of ``(path, list[wave_or_None])``
      - ``gvi_buffer_min/max/step``: int
      - ``ndvi_buffer_min/max/step``: int
      - ``coverage_complete``: bool — True iff every wave is covered
        exactly once per channel (always True in non-year-aware mode when
        each channel has exactly one file)
      - ``coverage_errors``: list[str] — human-readable error messages
    """
    st.subheader("Metric File Assignment")

    state: dict = {
        "gvi_files": [],
        "ndvi_files": [],
        "coverage_complete": True,
        "coverage_errors": [],
    }

    for ch_key, ch_label, ch_short in _FUSION_METRIC_CHANNELS:
        st.markdown(f"#### {ch_label}")
        with st.container(border=True):
            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                bmin = st.number_input(
                    f"{ch_short.upper()} minimum buffer",
                    min_value=50,
                    max_value=4900,
                    value=int(
                        st.session_state.get(f"fusion_{ch_short}_buffer_min", 100)
                    ),
                    step=50,
                    key=f"fusion_{ch_short}_buffer_min",
                    help=f"Smallest {ch_short.upper()} radius (m) searched by Optuna.",
                )
            with bcol2:
                bmax = st.number_input(
                    f"{ch_short.upper()} maximum buffer",
                    min_value=100,
                    max_value=5000,
                    value=int(
                        st.session_state.get(f"fusion_{ch_short}_buffer_max", 1000)
                    ),
                    step=50,
                    key=f"fusion_{ch_short}_buffer_max",
                    help=f"Largest {ch_short.upper()} radius (m). Extent padding uses the max.",
                )
            with bcol3:
                bstep = st.number_input(
                    f"{ch_short.upper()} buffer step",
                    min_value=10,
                    max_value=500,
                    value=int(
                        st.session_state.get(f"fusion_{ch_short}_buffer_step", 100)
                    ),
                    step=10,
                    key=f"fusion_{ch_short}_buffer_step",
                    help="Radius discretisation (m).",
                )
            state[f"{ch_key}_buffer_min"] = int(bmin)
            state[f"{ch_key}_buffer_max"] = int(bmax)
            state[f"{ch_key}_buffer_step"] = int(bstep)

            # Files repeater
            count_key = f"fusion_{ch_short}_file_count"
            if count_key not in st.session_state:
                st.session_state[count_key] = 1
            n_files = st.session_state[count_key]
            from file_picker import FT_VECTOR_OR_RASTER, pick_file_path

            sibling_assignments: dict[int, list] = {
                j: list(st.session_state.get(f"fusion_{ch_short}_assign_{j}", []))
                for j in range(n_files)
            }

            channel_files: list[tuple[str, list]] = []
            for i in range(n_files):
                col_up, col_rm = st.columns([5, 1])
                with col_up:
                    file_path = pick_file_path(
                        f"{ch_label} file {i + 1}",
                        key=f"fusion_{ch_short}_path_{i}",
                        file_types=FT_VECTOR_OR_RASTER,
                        help_text=(
                            "Pick a GeoTIFF or vector metric file from disk. "
                            "The path picker bypasses Streamlit's upload limit "
                            "so national-scale rasters are supported directly."
                        ),
                    )
                with col_rm:
                    if i > 0 and st.button(
                        "❌",
                        key=f"fusion_{ch_short}_rm_{i}",
                        help=f"Remove this {ch_short.upper()} file row",
                    ):
                        st.session_state[count_key] -= 1
                        st.rerun()
                if file_path and not os.path.isfile(file_path):
                    st.error(f"Path no longer exists: `{file_path}`")
                    file_path = None
                assigned: list = []
                if year_aware and discovered_waves:
                    my_current = set(sibling_assignments.get(i, []))
                    taken_by_others = set().union(
                        *(
                            set(waves)
                            for j, waves in sibling_assignments.items()
                            if j != i
                        )
                    )
                    visible_options = [
                        w
                        for w in discovered_waves
                        if w not in taken_by_others or w in my_current
                    ]
                    assign_key = f"fusion_{ch_short}_assign_{i}"
                    # Stale session_state can hold a wave that's no longer
                    # in the visible options (sibling claimed it last run).
                    # Prune those before rendering so Streamlit doesn't
                    # error on the invalid value.
                    if assign_key in st.session_state:
                        st.session_state[assign_key] = [
                            w
                            for w in st.session_state[assign_key]
                            if w in visible_options
                        ]
                    assigned = st.multiselect(
                        f"{ch_label} file {i + 1} — applies to year(s) / wave(s)",
                        options=visible_options,
                        key=assign_key,
                        help="Years already assigned to another file are hidden.",
                    )
                    # Update the snapshot so the NEXT row's option list
                    # reflects this row's freshly-rendered selection
                    # (otherwise late rows lag by one rerun).
                    sibling_assignments[i] = list(assigned)
                if file_path:
                    channel_files.append((file_path, assigned))

            if st.button(
                f"+ Add another {ch_short.upper()} file",
                key=f"fusion_{ch_short}_add",
            ):
                st.session_state[count_key] += 1
                st.rerun()

            state[f"{ch_key}_files"] = channel_files

            # Coverage check + caption.
            if year_aware and discovered_waves:
                covered: dict[str, int] = dict.fromkeys(discovered_waves, 0)
                for _, waves in channel_files:
                    for w in waves:
                        if w in covered:
                            covered[w] += 1
                missing = [w for w, n in covered.items() if n == 0]
                duplicates = [w for w, n in covered.items() if n > 1]
                bits = []
                for w in discovered_waves:
                    n = covered.get(w, 0)
                    if n == 1:
                        bits.append(f"`{w}` ✓")
                    elif n == 0:
                        bits.append(f"`{w}` ❌ unassigned")
                    else:
                        bits.append(f"`{w}` ⚠️ ×{n}")
                st.caption("Coverage: " + "  ".join(bits))
                if missing:
                    state["coverage_complete"] = False
                    state["coverage_errors"].append(
                        f"{ch_label}: no file assigned to " + ", ".join(missing)
                    )
                if duplicates:
                    state["coverage_complete"] = False
                    state["coverage_errors"].append(
                        f"{ch_label}: more than one file assigned to "
                        + ", ".join(duplicates)
                    )
            else:
                if len(channel_files) != 1:
                    state["coverage_complete"] = False
                    state["coverage_errors"].append(
                        f"{ch_label}: exactly one file is required "
                        f"(got {len(channel_files)})."
                    )

    return state


def _render_study_details_panel(
    is_longitudinal: bool,
    available_covariates: list[str],
    is_polygon_target: bool = False,
) -> dict:
    """Final form section: CGI formula, covariates, objective metric, …

    Returns the collected widget values; the caller wraps this in
    ``st.form`` and reads the submit click separately.
    """
    st.subheader("Study Details")

    # ── Row 1: CGI formula + covariates ──────────────────────────────────
    col_cgi1, col_cgi2 = st.columns([1, 2])
    with col_cgi1:
        cgi_formula = st.selectbox(
            "CGI Formula",
            options=["weighted_average", "synergy"],
            index=0,
            help=(
                "**weighted_average** — three weights on Vegetation / "
                "Terrain / NDVI summing to 100. "
                "**synergy** — seven weights summing to 100 plus three "
                "powers on the main channel terms."
            ),
            key="fusion_cgi_formula",
        )
    with col_cgi2:
        if available_covariates:
            covariate_columns = st.multiselect(
                "Covariates (control variables)",
                options=available_covariates,
                default=st.session_state.get("fusion_covariate_columns", []),
                key="fusion_covariate_columns",
                help=(
                    "Numeric attribute columns to control for. With "
                    "covariates the score becomes the greenery term's "
                    "partial contribution. `mutual_info` ignores covariates."
                ),
            )
        else:
            covariate_columns = []
            st.caption(
                "_No numeric attribute columns available for covariates "
                "(raster target or no spare numeric columns)._"
            )

    # ── Row 2: Objective metric + trials + optimizer ────────────────────
    metric_options = list(_MIXEDLM_METRICS if is_longitudinal else _CROSS_METRICS)
    default_metric = metric_options[0]
    prior = st.session_state.get("fusion_objective_metric")
    if prior not in metric_options:
        st.session_state["fusion_objective_metric"] = default_metric

    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1:
        objective_metric = st.selectbox(
            "Objective metric",
            options=metric_options,
            key="fusion_objective_metric",
            help="Quantity Optuna optimises per trial (maximised; RMSE minimised).",
        )
    with col_o2:
        n_trials = st.number_input(
            "Total trials",
            min_value=50,
            max_value=1000,
            value=int(st.session_state.get("fusion_n_trials", 100)),
            step=50,
            key="fusion_n_trials",
            help="Number of Optuna trials.",
        )
    with col_o3:
        optimizer = st.selectbox(
            "Optimizer",
            options=["TPE", "CMA-ES", "Random"],
            index=0,
            key="fusion_optimizer",
            help="Hyperparameter search sampler.",
        )

    # ── Row 3: startup, k-fold toggle + slider, test, bins ──────────────
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        n_startup_trials = st.number_input(
            "Random startup trials",
            min_value=10,
            max_value=500,
            value=int(st.session_state.get("fusion_n_startup", 50)),
            step=10,
            key="fusion_n_startup",
            help="Uniformly random trials before the main sampler engages.",
        )
    with col_s2:
        use_cv = st.checkbox(
            "Use k-fold cross-validation",
            value=bool(st.session_state.get("fusion_use_cv", True)),
            key="fusion_use_cv",
            help="On: k models per trial. Off: single train/val split, ~k× faster.",
        )
    with col_s4:
        test_size = st.slider(
            "Test set size",
            min_value=0.1,
            max_value=0.5,
            value=float(st.session_state.get("fusion_test_size", 0.25)),
            step=0.05,
            key="fusion_test_size",
            help="Held-out evaluation fraction of the whole dataset.",
        )
    k_folds = int(st.session_state.get("fusion_k_folds", 5))
    with col_s3:
        if use_cv:
            k_folds = st.number_input(
                "K-Fold CV",
                min_value=3,
                max_value=10,
                value=int(st.session_state.get("fusion_k_folds", 5)),
                key="fusion_k_folds",
                help="CV folds on the non-test subset.",
            )
            val_size_whole = (1.0 - float(test_size)) / max(1, int(k_folds))
        else:
            val_default = float(st.session_state.get("fusion_val_size", 0.25))
            val_size_whole = st.slider(
                "Validation set size",
                min_value=0.05,
                max_value=0.5,
                value=val_default,
                step=0.05,
                key="fusion_val_size",
                help="Validation fraction of the whole dataset (single-split mode).",
            )

    col_info = st.columns(1)
    with col_info[0]:
        train_pct = max(0.0, 1.0 - float(test_size) - float(val_size_whole))
        if use_cv:
            st.caption(
                f"_Test = {float(test_size)*100:.0f}% held out; "
                f"remaining {(1.0 - float(test_size))*100:.0f}% is split "
                f"into {int(k_folds)} folds "
                f"(~{val_size_whole*100:.0f}% val + "
                f"~{(train_pct)*100:.0f}% train per fold)._"
            )
        else:
            st.caption(
                f"_Train ≈ {train_pct*100:.0f}% · Val = "
                f"{val_size_whole*100:.0f}% · Test = "
                f"{float(test_size)*100:.0f}% of the whole dataset._"
            )

    n_bins = st.number_input(
        "Stratification bins",
        min_value=3,
        max_value=10,
        value=int(st.session_state.get("fusion_stratification_bins", 5)),
        key="fusion_stratification_bins",
        help="Quantile bins for the stratified train/test split.",
    )

    # ── Longitudinal-only mixed-effects toggles ─────────────────────────
    mixedlm_random_slope = True
    mixedlm_time_fixed = True
    if is_longitudinal:
        mc1, mc2 = st.columns(2)
        with mc1:
            mixedlm_random_slope = st.checkbox(
                "Random slope on time per entity",
                value=bool(st.session_state.get("fusion_lon_random_slope", True)),
                key="fusion_lon_random_slope",
                help="Switches RE structure to `(1 + years_since_baseline | entity)`.",
            )
        with mc2:
            mixedlm_time_fixed = st.checkbox(
                "Include `years_since_baseline` as fixed effect",
                value=bool(st.session_state.get("fusion_lon_include_time_fixed", True)),
                key="fusion_lon_include_time_fixed",
                help="Adds `+ years_since_baseline` to the fixed-effect design.",
            )

    # ── Resume + standalones ────────────────────────────────────────────
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        resume_existing_study = st.checkbox(
            "Resume previous study if exists",
            value=bool(st.session_state.get("fusion_resume_study", True)),
            key="fusion_resume_study",
            help="Reuses the existing SQLite study and runs only remaining trials.",
        )
    with col_r2:
        run_standalones = st.checkbox(
            "Also optimize each metric on its own (NDVI / Vegetation / Terrain)",
            value=bool(st.session_state.get("fusion_run_standalones", False)),
            key="fusion_run_standalones",
            help="Adds three single-metric Optuna studies alongside the combined CGI run.",
        )

    # ── Polygon scoring (per-pixel CGI) ─────────────────────────────────
    cgi_grid_spacing_m = 50
    whole_grid_scaling = True
    area_balanced_split = True
    if is_polygon_target:
        st.markdown("**Polygon scoring**")
        col_p1, col_p2 = st.columns([2, 1])
        with col_p1:
            cgi_grid_spacing_m = st.select_slider(
                "CGI grid pixel size (m)",
                options=list(range(25, 525, 25)),
                value=int(st.session_state.get("fusion_cgi_grid_spacing_m", 50)),
                key="fusion_cgi_grid_spacing_m",
                help="Per-pixel CGI grid spacing inside the polygon union. Smaller = higher fidelity, bigger cache.",
            )
        with col_p2:
            whole_grid_scaling = st.checkbox(
                "Whole-grid scaling",
                value=bool(st.session_state.get("fusion_whole_grid_scaling", True)),
                key="fusion_whole_grid_scaling",
                help="Skip per-channel scaling; only the composite raster is normalised.",
            )
        area_balanced_split = st.checkbox(
            "Area-balanced stratified split",
            value=bool(st.session_state.get("fusion_area_balanced_split", True)),
            key="fusion_area_balanced_split",
            help="Balance polygon area (not count) across train / val / test within each quartile.",
        )

    return {
        "cgi_formula": cgi_formula,
        "covariate_columns": list(covariate_columns or []),
        "objective_metric": objective_metric,
        "n_trials": int(n_trials),
        "n_startup_trials": int(n_startup_trials),
        "optimizer": optimizer,
        "use_cv": bool(use_cv),
        "k_folds": int(k_folds),
        "test_size": float(test_size),
        "val_size": float(val_size_whole),
        "n_bins": int(n_bins),
        "mixedlm_random_slope": bool(mixedlm_random_slope),
        "mixedlm_time_fixed": bool(mixedlm_time_fixed),
        "resume_existing_study": bool(resume_existing_study),
        "run_standalones": bool(run_standalones),
        "cgi_grid_spacing_m": int(cgi_grid_spacing_m),
        "whole_grid_scaling": bool(whole_grid_scaling),
        "area_balanced_split": bool(area_balanced_split),
    }


# ---------------------------------------------------------------------------
# Tab render entry point
# ---------------------------------------------------------------------------


def _render_fusion_results_section(output_dir: str) -> None:
    """Render the optimization results overview when one is loaded.

    Renders nothing when ``st.session_state.fusion_results`` is unset, so
    callers can invoke it unconditionally. Lives independently of the
    configuration UI so the user can browse a loaded run's results even
    when no fresh study is being set up.
    """
    if "fusion_results" not in st.session_state:
        st.session_state.fusion_results = None
    if "fusion_engine" not in st.session_state:
        st.session_state.fusion_engine = None
    if "fusion_engines_by_target" not in st.session_state:
        st.session_state.fusion_engines_by_target = {}

    if not st.session_state.fusion_results:
        return

    # Darken the results-panel background + shrink metric tile fonts via CSS.
    st.markdown(
        """
<style>
div[data-testid="stVerticalBlock"]:has(> div > div > div[data-fusion-results-anchor]) {
    background-color: rgba(38, 39, 48, 0.55);
    border-radius: 8px;
    padding: 12px 16px;
    margin-top: 8px;
}
div[data-testid="stVerticalBlock"]:has(> div > div > div[data-fusion-results-anchor])
    div[data-testid="stMetric"] {
    padding: 2px 6px;
}
div[data-testid="stVerticalBlock"]:has(> div > div > div[data-fusion-results-anchor])
    div[data-testid="stMetricLabel"] {
    font-size: 11px;
    color: #c0c0c8;
}
div[data-testid="stVerticalBlock"]:has(> div > div > div[data-fusion-results-anchor])
    div[data-testid="stMetricValue"] {
    font-size: 18px;
    line-height: 1.2;
}
</style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            "<div data-fusion-results-anchor></div>",
            unsafe_allow_html=True,
        )
        _render_fusion_results_body(output_dir)


def _render_covariate_impact(results_view: dict, metric_name: str) -> None:
    """Show per-covariate effect direction + importance + lift over CGI-only."""
    impact = results_view.get("covariate_impact")
    if not impact:
        return

    st.divider()
    st.markdown("**Covariate impact**")
    st.caption(
        "Two OLS models fit on the full dataset using the averaged top-20 % "
        "params: **Full** = `target ~ CGI + covariates`, **CGI-only** = "
        "`target ~ CGI`. Coefficients show each covariate's effect direction "
        "and magnitude in the full model; partial R² is the variance only "
        "that covariate explains (drop in R² when it's removed from Full)."
    )

    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("Full model R²", f"{impact.get('r2_full', 0):.4f}")
    with summary_cols[1]:
        st.metric("CGI-only R²", f"{impact.get('r2_cgi_only', 0):.4f}")
    with summary_cols[2]:
        st.metric(
            "Lift from covariates",
            f"{impact.get('r2_lift_from_covariates', 0):.4f}",
        )
    st.caption(
        f"Sample size: n = {impact.get('n', 0)} · CGI coefficient = "
        f"{impact.get('cgi_coef', 0):.4f} "
        f"(SE = {impact.get('cgi_std_err', 0):.4f})"
    )

    rows = impact.get("per_covariate") or []
    if not rows:
        return

    df = pd.DataFrame(rows)
    df = df[
        ["covariate", "direction", "coef", "std_err", "t_stat", "pvalue", "partial_r2"]
    ]
    df = df.sort_values("partial_r2", ascending=False).reset_index(drop=True)
    st.dataframe(df, width="stretch")

    try:
        import plotly.express as _px

        bar_df = df.copy()
        bar_df["abs_coef"] = bar_df["coef"].abs()
        fig = _px.bar(
            bar_df,
            x="covariate",
            y="coef",
            color="direction",
            color_discrete_map={
                "positive": "#2ca02c",
                "negative": "#d62728",
                "—": "#7f7f7f",
            },
            title="Covariate coefficients (full model)",
        )
        fig.update_layout(margin=dict(l=60, r=20, t=60, b=80))
        st.plotly_chart(fig, width="stretch")
    except Exception:
        pass


def _render_trial_distribution_viewer(
    results_view: dict, engine, metric_name: str, standalones: dict
) -> None:
    """Box-plot CI viewer for per-trial objective values.

    Lets the user pick studies, trial pools (all / robust), and subsets
    (train / val / test if recorded per trial) to compare distributions
    side-by-side. Pulls train/val from each trial's CV-fold means stored
    in ``user_attrs`` and test from the trial-level test attr (when
    present); skips subsets that aren't recorded per trial.
    """
    study = getattr(engine, "study", None)
    if study is None or not study.trials:
        return

    st.divider()
    st.markdown("**Per-trial objective distributions**")
    st.caption(
        "Distributions are built from per-trial CV-fold means recorded on "
        "every trial. **test** is computed only for the robust trial set "
        "(per-trial test scoring is expensive), so the test box will hold "
        "≤ robust-trial-count points even when the **all completed** pool "
        "is selected."
    )

    try:
        import optuna as _optuna
        import plotly.express as _px
    except Exception:
        st.info("Plotly + Optuna required for the distribution viewer.")
        return

    # ── Selectors ─────────────────────────────────────────────────────
    study_labels: list[tuple[str, dict, Any]] = [
        ("CGI (combined)", results_view, study)
    ]
    for ch_key, ch_bundle in (standalones or {}).items():
        study_labels.append(
            (f"{_CHANNEL_DISPLAY.get(ch_key, ch_key)} (standalone)", ch_bundle, None)
        )

    sel_cols = st.columns([2, 1, 1])
    with sel_cols[0]:
        picked_studies = st.multiselect(
            "Studies",
            options=[lbl for lbl, _, _ in study_labels],
            default=[study_labels[0][0]],
            key="fusion_distrib_studies",
        )
    with sel_cols[1]:
        pool_pick = st.selectbox(
            "Trial pool",
            options=["robust", "all completed"],
            key="fusion_distrib_pool",
        )
    with sel_cols[2]:
        subset_picks = st.multiselect(
            "Subsets",
            options=["train", "val", "test"],
            default=["train", "val"],
            key="fusion_distrib_subsets",
        )

    if not picked_studies or not subset_picks:
        st.info("Pick at least one study and one subset.")
        return

    # ── Build the long-format dataframe ───────────────────────────────
    rows: list[dict] = []
    for lbl in picked_studies:
        bundle = next(b for l, b, _ in study_labels if l == lbl)
        if lbl.startswith("CGI"):
            trial_pool = (
                bundle.get("robust_trials") or []
                if pool_pick == "robust"
                else [
                    t
                    for t in study.trials
                    if t.state == _optuna.trial.TrialState.COMPLETE
                ]
            )
        else:
            # Standalones are scored in a separate study and the engine
            # is rebound to CGI before results are returned, so the
            # runner snapshots all completed trials into the bundle for
            # the "all completed" pool.
            trial_pool = (
                bundle.get("robust_trials") or []
                if pool_pick == "robust"
                else bundle.get("all_completed_trials")
                or bundle.get("robust_trials")
                or []
            )

        # ``FrozenTrial.set_user_attr`` only mutates the in-memory copy;
        # for test scores the runner also writes a sidecar dict keyed by
        # ``trial.number`` so the value survives a fresh materialisation
        # of ``study.trials`` from storage.
        per_trial_test = bundle.get("per_trial_test") or {}
        for t in trial_pool:
            for subset in subset_picks:
                v: float | None = None
                if subset == "train":
                    v = t.user_attrs.get("train_score_mean")
                elif subset == "val":
                    v = t.user_attrs.get("val_score_mean")
                elif subset == "test":
                    rec = per_trial_test.get(int(t.number))
                    if rec is not None:
                        v = rec.get("test_score")
                    if v is None:
                        v = t.user_attrs.get("test_score")
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "Study": lbl,
                        "Subset": subset,
                        "Value": fv,
                    }
                )

    if not rows:
        st.info("No matching per-trial values were recorded for this selection.")
        return

    df = pd.DataFrame(rows)
    fig = _px.box(
        df,
        x="Study",
        y="Value",
        color="Subset",
        points="outliers",
        title=f"Per-trial {metric_name} distributions ({pool_pick} trials)",
    )
    fig.update_layout(margin=dict(l=60, r=20, t=60, b=80))
    st.plotly_chart(fig, width="stretch")

    # Summary stats per (Study, Subset) for users who want exact numbers.
    with st.expander("Show distribution stats", expanded=False):
        stats = (
            df.groupby(["Study", "Subset"])["Value"]
            .agg(["count", "mean", "std", "min", "max"])
            .round(4)
            .reset_index()
        )
        st.dataframe(stats, width="stretch")


def _render_composite_map_viewer(results_view: dict, engine) -> None:
    """Render the target outcome + selected study composite TIFFs together.

    The user picks which composite GeoTIFFs to display alongside the
    target outcome (drawn from the loaded polygon / point file). All
    composite subplots share a single colorbar fixed at [0, 1]; the
    target gets its own colorbar because its units differ. The figure
    is rendered at 300 dpi and laid out so the subplot count fills a
    near-square grid with minimal wasted space.
    """
    artifacts_dir = results_view.get("artifacts_dir")
    if not artifacts_dir or not os.path.isdir(artifacts_dir):
        return

    # Discover composite TIFFs in the job folder: CGI + each standalone.
    composite_options: list[tuple[str, str]] = []
    cgi_path = os.path.join(artifacts_dir, "composite_greenery.tif")
    if os.path.isfile(cgi_path):
        composite_options.append(("CGI (combined)", cgi_path))
    for ch in ("veg", "terrain", "ndvi"):
        path = os.path.join(artifacts_dir, f"composite_greenery_{ch}.tif")
        if os.path.isfile(path):
            composite_options.append(
                (f"{_CHANNEL_DISPLAY.get(ch, ch)} (standalone)", path)
            )

    if not composite_options:
        return

    st.divider()
    st.markdown("**Composite map viewer**")
    st.caption(
        "Pick one or more composite GeoTIFFs to display next to the "
        "target outcome. All composite subplots share the [0, 1] colorbar."
    )

    label_to_path = {label: path for label, path in composite_options}
    default_pick = [composite_options[0][0]]
    picks = st.multiselect(
        "Studies to plot",
        options=[label for label, _ in composite_options],
        default=default_pick,
        key="fusion_map_picks",
    )

    if not picks:
        return

    try:
        import rasterio
    except Exception:
        st.warning("rasterio not available; cannot render maps.")
        return

    # Read each composite TIFF + the target geometries once.
    composites: list[tuple[str, np.ndarray, Any, Any]] = []
    common_crs = None
    for label in picks:
        path = label_to_path[label]
        try:
            with rasterio.open(path) as src:
                arr = src.read(1, masked=True)
                composites.append((label, arr, src.transform, src.crs))
                if common_crs is None:
                    common_crs = src.crs
        except Exception as exc:
            st.warning(f"Could not read {label}: {exc}")

    if not composites:
        return

    target_gdf = None
    target_feature = results_view.get("target_feature")
    try:
        if engine is not None and engine.target_polygons_gdf is not None:
            target_gdf = engine.target_polygons_gdf.copy()
        elif engine is not None and engine.target_gdf is not None:
            target_gdf = engine.target_gdf.copy()
        if target_gdf is not None and common_crs is not None:
            target_gdf = target_gdf.to_crs(common_crs)
    except Exception as exc:
        st.warning(f"Could not reproject target: {exc}")
        target_gdf = None

    # Figure layout: target + N composites packed into a near-square grid
    # that minimises wasted slots.
    total = len(composites) + (1 if target_gdf is not None else 0)
    import math as _math

    ncols = max(1, int(_math.ceil(_math.sqrt(total))))
    nrows = max(1, int(_math.ceil(total / ncols)))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 4.5 * nrows),
        dpi=300,
        constrained_layout=True,
    )
    flat_axes = np.atleast_1d(axes).ravel().tolist()

    panel_idx = 0
    # ── Target panel ─────────────────────────────────────────────────
    if target_gdf is not None and target_feature is not None:
        ax = flat_axes[panel_idx]
        panel_idx += 1
        try:
            if target_feature in target_gdf.columns:
                vals = (
                    target_gdf[target_feature]
                    .astype(float)
                    .replace([np.inf, -np.inf], np.nan)
                )
                target_gdf.plot(
                    column=vals,
                    ax=ax,
                    cmap="RdYlGn",
                    legend=True,
                    legend_kwds={"shrink": 0.7},
                    missing_kwds={"color": "#cccccc"},
                )
                ax.set_title(f"Target: {target_feature}", fontsize=10)
            else:
                target_gdf.boundary.plot(ax=ax, color="black", linewidth=0.5)
                ax.set_title("Target geometry", fontsize=10)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Target render failed:\n{exc}", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # ── Composite panels (shared colorbar at [0, 1]) ─────────────────
    last_im = None
    for label, arr, transform, crs in composites:
        ax = flat_axes[panel_idx]
        panel_idx += 1
        try:
            data = np.ma.masked_invalid(arr)
            height, width = data.shape
            from rasterio.transform import array_bounds as _ab

            left, bottom, right, top = _ab(height, width, transform)
            last_im = ax.imshow(
                data,
                cmap="RdYlGn",
                vmin=0.0,
                vmax=1.0,
                extent=(left, right, bottom, top),
                origin="upper",
                interpolation="nearest",
            )
            if target_gdf is not None:
                try:
                    target_gdf.boundary.plot(ax=ax, color="black", linewidth=0.3)
                except Exception:
                    pass
        except Exception as exc:
            ax.text(0.5, 0.5, f"Render failed:\n{exc}", ha="center", va="center")
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # Hide unused axes (when total < ncols*nrows).
    for ax in flat_axes[panel_idx:]:
        ax.set_visible(False)

    # One shared colorbar for every composite panel.
    if last_im is not None:
        composite_axes = flat_axes[(1 if target_gdf is not None else 0) : panel_idx]
        if composite_axes:
            cbar = fig.colorbar(
                last_im,
                ax=composite_axes,
                shrink=0.7,
                fraction=0.04,
                pad=0.02,
            )
            cbar.set_label("Composite (0–1)")

    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _render_optuna_plot_explorer(engine, results_view: dict, metric_name: str) -> None:
    """Live Optuna visualisation picker.

    Renders all standard ``optuna.visualization`` plots from the in-memory
    study. The user picks the plot type, the trial pool (all completed
    trials vs the robust subset), and — when applicable — the parameter
    axes. Returns silently if the engine doesn't carry a study (e.g.
    legacy bundles).
    """
    study = getattr(engine, "study", None)
    if study is None:
        return

    try:
        import optuna as _optuna
        from optuna.visualization import (
            plot_contour,
            plot_edf,
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
            plot_rank,
            plot_slice,
            plot_timeline,
        )
    except Exception:
        return

    completed = [
        t for t in study.trials if t.state == _optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        return

    robust_trials = results_view.get("robust_trials") or []

    st.divider()
    st.markdown("**Interactive Plots**")

    plot_options = [
        "Optimization history",
        "Parameter importances",
        "Parallel coordinates",
        "Slice",
        "Contour",
        "Rank",
        "EDF",
        "Timeline",
    ]
    selector_cols = st.columns([2, 2])
    with selector_cols[0]:
        plot_pick = st.selectbox(
            "Plot type",
            options=plot_options,
            key="fusion_optuna_plot_pick",
        )
    with selector_cols[1]:
        pool_options = ["All completed trials"]
        if robust_trials:
            pool_options.append(f"Robust only ({len(robust_trials)})")
        pool_pick = st.selectbox(
            "Trial pool",
            options=pool_options,
            key="fusion_optuna_pool_pick",
        )

    if pool_pick.startswith("Robust"):
        sub_study = _optuna.create_study(
            direction=study.direction, sampler=study.sampler
        )
        for t in robust_trials:
            sub_study.add_trial(t)
        target_study = sub_study
    else:
        target_study = study

    # Discover the union of parameter names across the picked pool so
    # axis selectors only offer params that actually exist in the data
    # being plotted.
    available_params: list[str] = sorted(
        {p for t in target_study.trials for p in t.params}
    )

    params_axes_needed = plot_pick in {
        "Parallel coordinates",
        "Slice",
        "Contour",
        "Rank",
    }

    selected_params: list[str] | None = None
    if params_axes_needed and available_params:
        default_axes = available_params[: min(3, len(available_params))]
        selected_params = st.multiselect(
            "Parameters to include",
            options=available_params,
            default=default_axes,
            key=f"fusion_optuna_params_{plot_pick}",
        )
        if not selected_params:
            st.info("Pick at least one parameter to draw this plot.")
            return

    try:
        if plot_pick == "Optimization history":
            fig = plot_optimization_history(target_study)
        elif plot_pick == "Parameter importances":
            fig = plot_param_importances(target_study)
        elif plot_pick == "Parallel coordinates":
            fig = plot_parallel_coordinate(target_study, params=selected_params)
            # The y-axis label ("Objective Value" by default) sits in the
            # very left margin and Optuna's default layout clips it.
            # Add explicit padding so every label is visible regardless
            # of window width.
            fig.update_layout(
                margin=dict(l=120, r=80, t=70, b=60),
                width=None,
            )
        elif plot_pick == "Slice":
            fig = plot_slice(target_study, params=selected_params)
        elif plot_pick == "Contour":
            if not selected_params or len(selected_params) < 2:
                st.info("Contour needs at least two parameters.")
                return
            fig = plot_contour(target_study, params=selected_params)
        elif plot_pick == "Rank":
            fig = plot_rank(target_study, params=selected_params)
        elif plot_pick == "EDF":
            fig = plot_edf(target_study)
        elif plot_pick == "Timeline":
            fig = plot_timeline(target_study)
        else:
            return
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:
        st.warning(f"Could not render {plot_pick}: {exc}")


def _render_fusion_results_body(output_dir: str) -> None:
    """Inner body of the results panel — kept separate so the styled
    container above stays readable."""
    head_col, clear_col = st.columns([4, 1])
    with head_col:
        st.subheader("Optimization Results")
    with clear_col:
        if st.button(
            "Clear",
            key="fusion_results_clear",
            width="stretch",
            help="Unload these results from the panel (does not delete files).",
        ):
            st.session_state.fusion_engine = None
            st.session_state.fusion_results = None
            st.session_state.fusion_engines_by_target = {}
            st.rerun()

    results = st.session_state.fusion_results
    if results.get("mode") == "multi":
        st.selectbox(
            "Select outcome",
            options=results["ordered_labels"],
            key="fusion_results_outcome_pick",
        )

    results_view, engine = _fusion_resolve_active_bundle()
    if engine is None or results_view is None:
        st.warning("Optimization details are not available for the selected outcome.")
        return

    metric_name = results_view["objective_metric"].upper()
    best_params = results_view.get("best_params") or {}
    averaged_params_raw = results_view.get("averaged_params") or {}
    final_params = {
        k: v for k, v in averaged_params_raw.items() if not k.startswith("__")
    } or best_params

    # Formula introspection. Legacy results from before the formula
    # registry don't carry the attribute → fall back to weighted_average
    # so old studies still render with the original three weights.
    formula_name = getattr(engine, "cgi_formula", "weighted_average")
    try:
        formula = _cgi_formulas.get_formula(formula_name)
    except ValueError:
        formula = _cgi_formulas.get_formula("weighted_average")

    covariates_used = list(getattr(engine, "covariate_columns", []) or [])

    # ── Top summary: averaged top-20% params, grouped by category ──────
    # The composite GeoTIFF is built from these averaged params, so the
    # overview reports them rather than the single best trial. Four
    # rows in order: formula + headline scores · weights · radii · aggregators.
    _CHANNEL_LABELS = _CHANNEL_DISPLAY
    _FORMULA_DISPLAY = {
        "weighted_average": "Weighted Average",
        "synergy": "Synergy",
    }

    def _agg_label(stat: str | None, percentile: int | float | None) -> str:
        if stat == "mean":
            return "Mean"
        if stat == "median":
            return "Median"
        if stat == "percentile":
            try:
                p = int(round(float(percentile)))
            except (TypeError, ValueError):
                return "Percentile"
            suffix = "th"
            if p % 100 not in (11, 12, 13):
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(p % 10, "th")
            return f"{p}{suffix} percentile"
        return "—"

    # ── Row 1: formula · train/val/test (avg top-20%) · robust ratio ────
    subset_scores_view = results_view.get("subset_scores") or {}

    def _subset_score(name: str) -> float | None:
        block = subset_scores_view.get(name) or {}
        v = block.get("score")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    train_subset_score = _subset_score("train")
    val_subset_score = _subset_score("val")
    test_subset_score = _subset_score("test")
    formula_display = _FORMULA_DISPLAY.get(formula.name, formula.name.title())

    headline_cols = st.columns(5)
    with headline_cols[0]:
        st.metric("CGI Formula", formula_display)
    for col, label, score in (
        (headline_cols[1], "Train", train_subset_score),
        (headline_cols[2], "Val", val_subset_score),
        (headline_cols[3], "Test", test_subset_score),
    ):
        with col:
            if score is not None:
                st.metric(
                    f"{label} {metric_name} (avg top-20%)",
                    f"{score:.4f}",
                )
            else:
                st.metric(f"{label} {metric_name} (avg top-20%)", "—")
    with headline_cols[4]:
        if results_view["robust_trials"]:
            st.metric(
                "Robust Trials",
                f"{len(results_view['robust_trials'])}/{len(engine.study.trials)}",
            )
        else:
            st.metric("Total Trials", len(engine.study.trials))

    # ── Row 2: weights ─────────────────────────────────────────────────
    if formula.name == _cgi_formulas.WEIGHTED_AVERAGE:
        st.markdown("**Weights (averaged top-20%)**")
        weight_cols = st.columns(len(formula.weight_keys))
        total_weight = sum(float(final_params.get(k, 0)) for k in formula.weight_keys)
        for col, key in zip(weight_cols, formula.weight_keys):
            ch = key.removesuffix("_weight")
            label = _CHANNEL_LABELS.get(ch, ch.upper())
            pct = (
                100.0 * float(final_params.get(key, 0)) / total_weight
                if total_weight > 0
                else 0.0
            )
            with col:
                st.metric(label, f"{pct:.1f}%")
    else:
        st.markdown("**Weights and powers (averaged top-20%)**")
        all_keys = list(formula.weight_keys) + list(formula.power_keys)
        groups = [all_keys[i : i + 4] for i in range(0, len(all_keys), 4)]
        for group in groups:
            cols = st.columns(len(group))
            for col, key in zip(cols, group):
                val = float(final_params.get(key, 0))
                if key in formula.power_keys:
                    pretty = key.removesuffix("_power").upper() + " power"
                    with col:
                        st.metric(pretty, f"{val:.2f}")
                else:
                    pretty = key.removeprefix("w_").replace("_", "·").upper() + " %"
                    with col:
                        st.metric(pretty, f"{val:.1f}")

    # ── Row 3: radii ───────────────────────────────────────────────────
    st.markdown("**Radii (m)**")
    radii_cols = st.columns(3)
    for col, key in zip(radii_cols, ("veg_radius", "terrain_radius", "ndvi_radius")):
        ch = key.removesuffix("_radius")
        label = _CHANNEL_LABELS.get(ch, ch.upper())
        with col:
            try:
                v = int(round(float(final_params.get(key, 0))))
                st.metric(label, f"{v} m")
            except (TypeError, ValueError):
                st.metric(label, "—")

    # ── Row 4: aggregators ─────────────────────────────────────────────
    st.markdown("**Aggregators**")
    agg_cols = st.columns(2)
    with agg_cols[0]:
        st.metric(
            "GVI (Vegetation + Terrain)",
            _agg_label(
                final_params.get("streetview_stat"),
                final_params.get("streetview_percentile"),
            ),
        )
    with agg_cols[1]:
        st.metric(
            "NDVI",
            _agg_label(
                final_params.get("ndvi_stat"),
                final_params.get("ndvi_percentile"),
            ),
        )

    st.caption(
        "**Covariates:** "
        + (
            ", ".join(f"`{c}`" for c in covariates_used)
            if covariates_used
            else "_none_"
        )
        + (
            "  ·  ℹ️ `mutual_info` ignores covariates"
            if results_view["objective_metric"] == "mutual_info" and covariates_used
            else ""
        )
    )
    artifacts_dir = results_view.get("artifacts_dir")
    if artifacts_dir:
        st.caption(f"📁 Job artifacts: `{artifacts_dir}`")

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
            info_data[k] = best_params.get(k, "N/A")
        if covariates_used:
            info_data["Covariates controlled"] = covariates_used

        info_data[f"Train {metric_name}"] = (
            f"{best_trial.user_attrs.get('train_score_mean', 'N/A')}"
        )
        info_data[f"Val {metric_name}"] = (
            f"{best_trial.user_attrs.get('val_score_mean', 'N/A')}"
        )

        if "train_pvalue" in best_trial.user_attrs:
            info_data["Train p-value"] = f"{best_trial.user_attrs['train_pvalue']:.4e}"
        if "val_pvalue" in best_trial.user_attrs:
            info_data["Val p-value"] = f"{best_trial.user_attrs['val_pvalue']:.4e}"

        st.json(info_data)

    with col_detail2:
        averaged_params = results_view.get("averaged_params") or {}
        if averaged_params:
            st.markdown("**Final (averaged top-20%) Parameters**")
            clean_avg = {
                k: v for k, v in averaged_params.items() if not k.startswith("__")
            }
            st.json(clean_avg)
            n_top = averaged_params.get("__n_top_trials__")
            n_robust = averaged_params.get("__n_robust_trials__")
            if n_top is not None and n_robust is not None:
                st.caption(
                    f"Ensemble of the top {n_top} trials from "
                    f"{n_robust} robust trials. These are the params "
                    "used to render the composite GeoTIFF."
                )
        else:
            st.caption("_No averaged top-20% parameters were recorded for this run._")

    # ── Interactive Optuna plot viewer ─────────────────────────────────
    # The user picks a plot type and (optionally) a study filter +
    # parameter axes; the plot is rendered live from the in-memory study
    # so it's free to explore without regenerating files on disk.
    _render_optuna_plot_explorer(engine, results_view, metric_name)

    # ── Robust trials browser (CGI + per-standalone) ─────────────────
    # One scrollable dataframe per study. The selector lets the user
    # pivot between the combined CGI run and each standalone single-
    # metric study without leaving the results panel.
    study_options: list[tuple[str, dict, Any]] = [
        (
            "CGI (combined)",
            results_view,
            engine,
        )
    ]
    standalones_for_robust = results_view.get("standalones") or {}
    for _ch, _ch_bundle in standalones_for_robust.items():
        study_options.append(
            (
                f"{_CHANNEL_DISPLAY.get(_ch, _ch)} (standalone)",
                _ch_bundle,
                engine,  # standalone study object isn't pickled separately
            )
        )

    if any(opt[1].get("robust_trials") for opt in study_options):
        st.divider()
        st.markdown("**Robust Trials (Statistically Significant)**")

        picked = st.selectbox(
            "Study",
            options=[opt[0] for opt in study_options],
            key="fusion_robust_study_pick",
        )
        picked_view = next(opt[1] for opt in study_options if opt[0] == picked)
        picked_robust = picked_view.get("robust_trials") or []

        def _wlabel(key: str) -> str:
            cleaned = key.removeprefix("w_").removesuffix("_weight")
            return cleaned.replace("_", "·").upper() + " %"

        def _plabel(key: str) -> str:
            return key.removesuffix("_power").upper() + " p"

        if not picked_robust:
            st.info("No robust trials in this study.")
        else:
            # Standalones never suggest weights for non-active channels;
            # show 100 % on the active one and "—" for inapplicable columns.
            picked_channel: str | None = picked_view.get("channel")
            active_weight_keys: set[str] = set()
            if picked_channel == "veg":
                active_weight_keys = {"veg_weight", "w_veg"}
            elif picked_channel == "terrain":
                active_weight_keys = {"terrain_weight", "w_ter"}
            elif picked_channel == "ndvi":
                active_weight_keys = {"ndvi_weight", "w_ndvi"}

            def _radius_applies(radius_key: str) -> bool:
                if picked_channel is None:
                    return True
                return radius_key == f"{picked_channel}_radius"

            def _stat_applies(stat_key: str) -> bool:
                if picked_channel is None:
                    return True
                if picked_channel == "ndvi":
                    return stat_key == "ndvi_stat"
                return stat_key == "streetview_stat"

            robust_data: list[dict] = []
            for t in picked_robust:
                row: dict = {"Trial": t.number}
                if picked_channel is None:
                    weight_total = sum(
                        float(t.params.get(k, 0)) for k in formula.weight_keys
                    )
                    for k in formula.weight_keys:
                        row[_wlabel(k)] = (
                            round(100.0 * float(t.params.get(k, 0)) / weight_total, 1)
                            if weight_total > 0
                            else 0.0
                        )
                else:
                    # Standalone: synthesize 100/0 weights so the table
                    # mirrors the actual scoring (single channel only).
                    for k in formula.weight_keys:
                        row[_wlabel(k)] = 100.0 if k in active_weight_keys else 0.0
                for k in formula.power_keys:
                    row[_plabel(k)] = round(float(t.params.get(k, 1.0)), 2)
                for k in ("veg_radius", "terrain_radius", "ndvi_radius"):
                    label = k.replace("_", " ")
                    if k in t.params and _radius_applies(k):
                        row[label] = t.params[k]
                    elif _radius_applies(k):
                        row[label] = None
                for k in ("streetview_stat", "ndvi_stat"):
                    label = k.replace("_", " ")
                    if k in t.params and _stat_applies(k):
                        row[label] = t.params[k]
                    elif _stat_applies(k):
                        row[label] = "—"
                for k in ("streetview_percentile", "ndvi_percentile"):
                    label = k.replace("_", " ")
                    stat_key = (
                        "ndvi_stat" if k == "ndvi_percentile" else "streetview_stat"
                    )
                    if (
                        k in t.params
                        and _stat_applies(stat_key)
                        and t.params.get(stat_key) == "percentile"
                    ):
                        row[label] = t.params[k]
                train_mean = t.user_attrs.get("train_score_mean")
                val_mean = t.user_attrs.get("val_score_mean")
                train_pv = t.user_attrs.get("train_pvalue_mean")
                val_pv = t.user_attrs.get("val_pvalue_mean")
                row[f"Train {metric_name}"] = (
                    round(float(train_mean), 4) if train_mean is not None else None
                )
                row[f"Val {metric_name}"] = (
                    round(float(val_mean), 4) if val_mean is not None else None
                )
                row["Train p"] = float(train_pv) if train_pv is not None else None
                row["Val p"] = float(val_pv) if val_pv is not None else None
                robust_data.append(row)

            st.caption(
                f"Showing all {len(robust_data)} robust trials. "
                "Sort by clicking column headers; scroll inside the table "
                "to see more rows."
            )
            st.dataframe(
                pd.DataFrame(robust_data),
                width="stretch",
                height=420,
            )

    standalones = results_view.get("standalones") or {}
    if standalones and any(b.get("subset_scores") for b in standalones.values()):
        st.divider()
        st.markdown("**CGI vs Standalone Single-Metric Studies**")

        # Subset / value multi-select; scores are pre-cached in the bundle.
        subset_cols = st.columns([1, 1])
        with subset_cols[0]:
            subset_picks = st.multiselect(
                "Data subsets",
                options=["train", "val", "test", "all"],
                default=["test"],
                key="fusion_subset_pick",
                help="Scores recorded per study with the averaged top-20% params.",
            )
        with subset_cols[1]:
            value_picks = st.multiselect(
                "Values",
                options=["score", "pvalue", "n"],
                default=["score"],
                key="fusion_subset_value_pick",
                help="Rendered as separate charts (different scales).",
            )

        if not subset_picks or not value_picks:
            st.info("Pick at least one subset and one value to compare.")
        else:

            def _study_label(key: str) -> str:
                if key == "cgi":
                    return "CGI (combined)"
                return f"{_CHANNEL_DISPLAY.get(key, key)} (standalone)"

            all_studies: list[tuple[str, dict]] = [("cgi", results_view)]
            for ch_key, ch_bundle in standalones.items():
                all_studies.append((ch_key, ch_bundle))

            try:
                import plotly.express as _px
            except Exception:
                _px = None

            # One grouped-bar chart per value (different scales can't share an axis).
            for value_pick in value_picks:
                rows: list[dict] = []
                for sk, bundle in all_studies:
                    for subset_pick in subset_picks:
                        sub = (bundle.get("subset_scores") or {}).get(subset_pick) or {}
                        val = sub.get(value_pick)
                        try:
                            fval = float(val) if val is not None else None
                        except (TypeError, ValueError):
                            fval = None
                        rows.append(
                            {
                                "Study": _study_label(sk),
                                "Subset": subset_pick,
                                "Value": fval,
                            }
                        )

                df = pd.DataFrame(rows)
                y_title = {
                    "score": f"{metric_name}",
                    "pvalue": "p-value",
                    "n": "n entities",
                }[value_pick]

                if _px is not None:
                    fig = _px.bar(
                        df,
                        x="Study",
                        y="Value",
                        color="Subset",
                        barmode="group",
                        title=y_title,
                        labels={"Value": y_title},
                    )
                    fig.update_layout(
                        margin=dict(l=60, r=20, t=60, b=60),
                        legend_title_text="Subset",
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    # Fallback: pivot to wide and use streamlit's
                    # native bar chart, which auto-groups by columns.
                    wide = df.pivot(index="Study", columns="Subset", values="Value")
                    st.markdown(f"_{y_title}_")
                    st.bar_chart(wide, width="stretch")

            # Compact textual summary so users can read off exact values.
            with st.expander("Show exact values", expanded=False):
                summary: list[dict] = []
                for sk, bundle in all_studies:
                    for subset_pick in subset_picks:
                        sub = (bundle.get("subset_scores") or {}).get(subset_pick) or {}
                        row: dict = {
                            "Study": _study_label(sk),
                            "Subset": subset_pick,
                        }
                        for value_pick in value_picks:
                            v = sub.get(value_pick)
                            try:
                                fv = float(v) if v is not None else None
                            except (TypeError, ValueError):
                                fv = None
                            if fv is None:
                                row[value_pick] = "—"
                            elif value_pick == "pvalue":
                                row[value_pick] = f"{fv:.4e}"
                            elif value_pick == "n":
                                row[value_pick] = int(fv)
                            else:
                                row[value_pick] = round(fv, 4)
                        summary.append(row)
                st.dataframe(pd.DataFrame(summary), width="stretch")

    # ── Covariate impact panel ────────────────────────────────────────
    _render_covariate_impact(results_view, metric_name)

    # ── Per-trial CI box plot viewer ──────────────────────────────────
    _render_trial_distribution_viewer(results_view, engine, metric_name, standalones)

    # ── Composite map viewer ──────────────────────────────────────────
    _render_composite_map_viewer(results_view, engine)

    # ── Mixed-effects post-hoc metrics CSV viewer ─────────────────────
    # The post-score stage writes one CSV per study (CGI + each
    # standalone) under the job's ``study_results/`` directory. Pull
    # the artifacts dir from the bundle so we look in the right per-
    # job folder; fall back to the legacy shared location for older
    # bundles that don't carry one.
    try:
        bundle_artifacts = results_view.get("artifacts_dir")
        if bundle_artifacts:
            _study_root = os.path.join(bundle_artifacts, "study_results")
        else:
            _study_root = os.path.join(output_dir, "fusion", "study_results")
        _active_label = (
            st.session_state.get("fusion_results_outcome_pick")
            if results.get("mode") == "multi"
            else None
        )

        def _match_outcome(fname: str) -> bool:
            if _active_label is None:
                return f"__{_active_label}" not in fname
            return f"__{_active_label}" in fname

        _csv_paths: list[str] = []
        if os.path.isdir(_study_root):
            for _fn in sorted(os.listdir(_study_root)):
                if not _fn.startswith("mixedlm_metrics") or not _fn.endswith(".csv"):
                    continue
                if not _match_outcome(_fn):
                    continue
                _csv_paths.append(os.path.join(_study_root, _fn))

        if _csv_paths:
            st.divider()
            st.markdown("**Mixed-effects: all metrics across trial pools**")
            st.caption(
                "Each pool's per-trial rows are followed by "
                "`__mean__`, `__ci_lo__`, `__ci_hi__`, and `__n__` "
                "summary rows."
            )

            def _tab_label(path: str) -> str:
                base = os.path.basename(path).replace(".csv", "")
                if _active_label:
                    base = base.replace(
                        f"mixedlm_metrics__{_active_label}", "CGI"
                    ).replace(f"__{_active_label}__", "__")
                else:
                    base = base.replace("mixedlm_metrics", "CGI")
                return base.replace("__", " ").strip() or "CGI"

            _tab_labels = [_tab_label(p) for p in _csv_paths]
            _tabs = st.tabs(_tab_labels)
            for _tab, _path in zip(_tabs, _csv_paths):
                with _tab:
                    st.caption(f"Source: `{_path}`")
                    st.dataframe(pd.read_csv(_path), width="stretch")
    except Exception as _exc:  # pragma: no cover -- UI-only guard
        st.warning(f"Could not read mixedlm_metrics CSVs: {_exc}")


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
    # Loaded-results overview
    # =========================================================================
    # Render the results section first so it survives the configuration-UI
    # early returns below. The user loads a completed job's bundle via the
    # **Load results** button in the sidebar Job Monitor; once loaded, this
    # block keeps showing it across page refreshes and across changes to
    # the configuration form below.
    _render_fusion_results_section(output_dir)

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

        from file_picker import FT_VECTOR_OR_RASTER, path_to_dataset, pick_ordered_files

        target_file_entries = pick_ordered_files(
            "Pick Target File(s)",
            key="fusion_target_paths",
            file_types=FT_VECTOR_OR_RASTER,
            help_text=(
                "Pick one file for cross-sectional or long-format longitudinal "
                "runs, or multiple files (one per wave) for wide-format "
                "longitudinal. Drag-free reorder via ▲/▼ — first file is the "
                "baseline and drives the preview, outcome and covariate pickers."
            ),
        )
        target_picked_path: str | None = (
            target_file_entries[0]["path"] if target_file_entries else None
        )

        if target_picked_path and os.path.isfile(target_picked_path):
            target_mat = path_to_dataset(target_picked_path)
            tmp_target_path = target_mat.path
            is_vector_target = target_mat.is_vector
            is_raster_target = not target_mat.is_vector
            target_display_name = target_mat.display_name
            sig = (target_mat.path, os.path.getsize(target_mat.path))
            if st.session_state.get("fusion_target_upload_sig") != sig:
                st.session_state.fusion_target_upload_sig = sig
                st.session_state.fusion_outcome_columns = []

            if target_mat is not None:

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
                                help="Each selection becomes one optimization target.",
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
                                help="Joint multi-objective optimization (currently runs sequentially).",
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

        if target_picked_path and tmp_target_path:
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
    # SECTION 2 — Optimization setup
    # =========================================================================
    st.divider()
    opt_state = _render_optimization_setup_panel(
        preview_vector_gdf if is_vector_target else None,
        target_outcome_columns,
        target_file_entries=target_file_entries,
    )
    if opt_state is None:
        # No run mode picked yet — bail out so downstream sections don't
        # render with empty discovery state.
        return
    is_longitudinal = opt_state["is_longitudinal"]
    year_aware = is_longitudinal or opt_state["cross_sectional_date_on"]

    # =========================================================================
    # SECTION 3 — Metric file assignment
    # =========================================================================
    st.divider()
    metric_state = _render_metric_assignment_panel(
        opt_state["discovered_waves"],
        year_aware=year_aware,
    )
    if metric_state is None:
        return

    # Legacy variables submission still reads — derived from metric_state.
    gvi_buffer_min_m = metric_state["gvi_buffer_min"]
    gvi_buffer_max_m = metric_state["gvi_buffer_max"]
    gvi_buffer_step_m = metric_state["gvi_buffer_step"]
    ndvi_buffer_min_m = metric_state["ndvi_buffer_min"]
    ndvi_buffer_max_m = metric_state["ndvi_buffer_max"]
    ndvi_buffer_step_m = metric_state["ndvi_buffer_step"]
    buffer_extent_m = float(max(gvi_buffer_max_m, ndvi_buffer_max_m))

    # Auto-download mode is no longer exposed; defaults reproduce the
    # legacy non-Auto-Download path.
    ndvi_auto_start = date(2023, 6, 1)
    ndvi_auto_end = date(2023, 9, 30)
    cache_metrics = False
    ndvi_resolution_m = None
    gvi_grid_spacing_m = None

    # =========================================================================
    # SECTION 4 — Study details (form)
    # =========================================================================
    st.divider()

    # Numeric attribute columns the user can pick as covariates
    available_covariates: list[str] = []
    if is_vector_target and preview_vector_gdf is not None:
        numeric_attr_cols = preview_vector_gdf.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        outcome_set = set(target_outcome_columns)
        wide_files = opt_state.get("wide_files") or []
        if is_longitudinal and opt_state.get("intake_mode") == "wide" and wide_files:
            common: set[str] | None = None
            for wf in wide_files:
                try:
                    _frame = gpd.read_file(wf["path"], rows=64)
                except Exception:
                    continue
                _nums = set(_frame.select_dtypes(include=[np.number]).columns)
                common = _nums if common is None else common & _nums
            if common is not None:
                numeric_attr_cols = sorted(common & set(numeric_attr_cols)) or sorted(
                    common
                )
        available_covariates = [c for c in numeric_attr_cols if c not in outcome_set]

    # Only render polygon-only controls when the target carries polygons.
    is_polygon_target_ui = False
    if is_vector_target and preview_vector_gdf is not None:
        try:
            gt = preview_vector_gdf.geometry.geom_type
            is_polygon_target_ui = bool(gt.isin(["Polygon", "MultiPolygon"]).any())
        except Exception:
            is_polygon_target_ui = False

    with st.container():
        study_state = _render_study_details_panel(
            is_longitudinal=is_longitudinal,
            available_covariates=available_covariates,
            is_polygon_target=is_polygon_target_ui,
        )
        st.divider()
        _fus_run_spacer, _fus_run_col = st.columns([2.2, 1])
        with _fus_run_col:
            fusion_run_clicked = st.button(
                "🚀 Run Fusion Optimization",
                type="primary",
                width="stretch",
                key="fusion_form_run_submit",
            )

    # Pull study-state values into the legacy local names the submission
    # block below still reads.
    cgi_formula = study_state["cgi_formula"]
    covariate_columns = study_state["covariate_columns"]
    objective_metric = study_state["objective_metric"]
    n_trials = study_state["n_trials"]
    n_startup_trials = study_state["n_startup_trials"]
    optimizer = study_state["optimizer"]
    k_folds = study_state["k_folds"]
    test_size = study_state["test_size"]
    val_size = study_state.get("val_size", 0.25)
    n_bins = study_state["n_bins"]
    resume_existing_study = study_state["resume_existing_study"]
    run_standalones = study_state["run_standalones"]
    pruner_type = "none"  # UI removed; engine accepts NopPruner via "none"
    lon_random_slope = study_state["mixedlm_random_slope"]
    lon_include_time_fixed = study_state["mixedlm_time_fixed"]
    cgi_grid_spacing_m_param = study_state.get("cgi_grid_spacing_m")
    whole_grid_scaling_param = bool(study_state.get("whole_grid_scaling", False))
    area_balanced_split_param = bool(study_state.get("area_balanced_split", False))

    # =========================================================================
    # Run controls and progress
    # =========================================================================
    st.divider()

    col_run2, col_run3 = st.columns([1, 1])
    with col_run2:
        if st.session_state.fusion_results:
            if st.button(
                "Export per-entity CGI",
                width="stretch",
                key="fusion_export",
                help="One row per target entity with sampled channels + composite CGI.",
            ):
                bundle, _eng = _fusion_resolve_active_bundle()
                if bundle is None or bundle.get("composite_df") is None:
                    st.error("No composite table available to export.")
                elif _eng is None:
                    st.error("Engine state unavailable; cannot attach geometries.")
                else:
                    result_df = bundle["composite_df"].copy()
                    artifacts_dir = bundle.get("artifacts_dir") or output_dir
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

                    geom_attached = False
                    export_path: str | None = None

                    if (
                        "polygon_id" in result_df.columns
                        and _eng.target_polygons_gdf is not None
                    ):
                        polys = _eng.target_polygons_gdf.reset_index(drop=True).copy()
                        polys["polygon_id"] = polys.index
                        keep_attr_cols = [
                            c for c in polys.columns if c == "geometry"
                        ] + ["polygon_id"]
                        result_gdf = polys[keep_attr_cols].merge(
                            result_df, on="polygon_id", how="inner"
                        )
                        export_path = os.path.join(
                            artifacts_dir,
                            f"fusion_composite_{ts}.gpkg",
                        )
                        result_gdf.to_file(export_path, driver="GPKG", layer="cgi")
                        geom_attached = True
                    elif _eng.target_gdf is not None and len(_eng.target_gdf) == len(
                        result_df
                    ):
                        base = _eng.target_gdf.reset_index(drop=True).copy()
                        for col in result_df.columns:
                            base[col] = result_df[col].reset_index(drop=True).values
                        export_path = os.path.join(
                            artifacts_dir,
                            f"fusion_composite_{ts}.gpkg",
                        )
                        base.to_file(export_path, driver="GPKG", layer="cgi")
                        geom_attached = True

                    if not geom_attached:
                        export_path = os.path.join(
                            artifacts_dir,
                            f"fusion_composite_{ts}.csv",
                        )
                        result_df.to_csv(export_path, index=False)

                    st.success(f"Exported {len(result_df)} rows to: `{export_path}`")
                    st.caption(
                        "Columns: `polygon_id`, `target`, `veg` / `terrain` / "
                        "`ndvi`, `composite`, `n_samples`. Geometries joined "
                        "from the loaded target file."
                    )
    with col_run3:
        if st.session_state.fusion_results:
            if st.button("🔄 Reset", width="stretch", key="fusion_reset"):
                st.session_state.fusion_engine = None
                st.session_state.fusion_results = None
                st.session_state.fusion_engines_by_target = {}
                st.rerun()

    if fusion_run_clicked:
        # ── Basic validation ─────────────────────────────────────────────
        if not target_picked_path or not tmp_target_path:
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
        elif not metric_state["coverage_complete"]:
            for _err in metric_state["coverage_errors"]:
                st.error(f"❌ {_err}")
        else:
            # ── Build LongitudinalSpec (or None) ─────────────────────────
            from geofuse.longitudinal import LongitudinalSpec as _LonSpec
            from geofuse.longitudinal import validate_spec as _validate_lon_spec

            def _per_wave_file_map(
                channel_files: list[tuple[str, list]],
            ) -> dict[str, str]:
                """Flatten [(path, [waves])] to {wave: path}."""
                out: dict[str, str] = {}
                for fp, assigned in channel_files:
                    for w in assigned:
                        out[str(w)] = fp
                return out

            longitudinal_spec_payload: dict | None = None
            veg_path: str | None = None
            ndvi_path: str | None = None
            spec_errs: list[str] = []

            if is_longitudinal:
                intake = opt_state["intake_mode"]
                wave_labels = tuple(opt_state["discovered_waves"])
                gvi_per_wave = _per_wave_file_map(metric_state["gvi_files"])
                ndvi_per_wave = _per_wave_file_map(metric_state["ndvi_files"])
                # GVI multi-band raster (or vector) fills both veg + terrain
                # channels for each wave; the engine's per-channel loader
                # picks the right band based on the channel name.
                greenery_files = {
                    "veg": dict(gvi_per_wave),
                    "terrain": dict(gvi_per_wave),
                    "ndvi": dict(ndvi_per_wave),
                }
                if intake == "wide":
                    wide_files = opt_state["wide_files"]
                    if len(wide_files) < 1:
                        spec_errs.append(
                            "Wide-mode longitudinal requires at least one per-wave file."
                        )
                    if wide_files:
                        # Use the first file's column choices as canonical;
                        # the runner reads each per-wave file as-is and the
                        # engine joins on these names.
                        canonical_entity = wide_files[0]["entity_col"]
                        canonical_date = wide_files[0]["date_col"]
                        mismatches = [
                            wf["wave_label"]
                            for wf in wide_files[1:]
                            if wf["entity_col"] != canonical_entity
                            or wf["date_col"] != canonical_date
                        ]
                        if mismatches:
                            st.warning(
                                "⚠️ Wide-mode files use mixed column names; "
                                "first file's columns ({}, {}) are canonical. "
                                "Mismatched waves: {}".format(
                                    canonical_entity,
                                    canonical_date,
                                    ", ".join(mismatches),
                                )
                            )
                        target_files_per_wave = {
                            wf["wave_label"]: wf["path"] for wf in wide_files
                        }
                        entity_id_col = canonical_entity
                        date_col_eff = canonical_date
                    else:
                        target_files_per_wave = {}
                        entity_id_col = ""
                        date_col_eff = ""
                    wave_col_eff: str | None = None
                else:
                    target_files_per_wave = {}
                    entity_id_col = opt_state["entity_id_col"] or ""
                    date_col_eff = opt_state["date_col"] or ""
                    wave_col_eff = opt_state["wave_col"]

                spec = _LonSpec(
                    intake_mode=intake,  # type: ignore[arg-type]
                    entity_id_col=entity_id_col,
                    wave_labels=wave_labels,
                    wave_col=wave_col_eff,
                    date_col=date_col_eff or "measurement_date",
                    greenery_files=greenery_files,
                    target_files_per_wave=target_files_per_wave,
                    scoring_metric=objective_metric,
                    include_time_fixed_effect=lon_include_time_fixed,
                    random_slope_time=lon_random_slope,
                    derive_wave_from_date=False,
                )
                spec_errs += _validate_lon_spec(spec)
                if spec_errs:
                    st.error(
                        "Mixed-effects spec is invalid:\n- " + "\n- ".join(spec_errs)
                    )
                    st.stop()
                longitudinal_spec_payload = spec.to_payload()

            elif opt_state["cross_sectional_date_on"]:
                # Year-aware cross-sectional: same per-wave cache pipeline
                # as longitudinal, but with an OLS scoring metric and a
                # synthesised entity/year column derived in the runner.
                wave_labels = tuple(opt_state["discovered_waves"])
                gvi_per_wave = _per_wave_file_map(metric_state["gvi_files"])
                ndvi_per_wave = _per_wave_file_map(metric_state["ndvi_files"])
                greenery_files = {
                    "veg": dict(gvi_per_wave),
                    "terrain": dict(gvi_per_wave),
                    "ndvi": dict(ndvi_per_wave),
                }
                spec = _LonSpec(
                    intake_mode="long",  # type: ignore[arg-type]
                    entity_id_col="_gf_entity",
                    wave_labels=wave_labels,
                    wave_col="_gf_year",
                    date_col=opt_state["date_col"] or "",
                    greenery_files=greenery_files,
                    target_files_per_wave={},
                    scoring_metric=objective_metric,
                    derive_wave_from_date=True,
                )
                spec_errs = _validate_lon_spec(spec)
                if spec_errs:
                    st.error(
                        "Year-aware cross-sectional spec is invalid:\n- "
                        + "\n- ".join(spec_errs)
                    )
                    st.stop()
                longitudinal_spec_payload = spec.to_payload()

            else:
                # Plain cross-sectional: one file per channel goes directly
                # to the runner's veg_path / ndvi_path; no spec.
                if metric_state["gvi_files"]:
                    veg_path = metric_state["gvi_files"][0][0]
                if metric_state["ndvi_files"]:
                    ndvi_path = metric_state["ndvi_files"][0][0]

            # ── Configuration summary ───────────────────────────────────
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
                    f"**Run mode:** {opt_state['run_mode']}"
                    + (
                        f" · intake = `{opt_state['intake_mode']}`"
                        if is_longitudinal
                        else ""
                    )
                )
                if opt_state["discovered_waves"]:
                    st.write(
                        "**Discovered years / waves:** "
                        + ", ".join(f"`{w}`" for w in opt_state["discovered_waves"])
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
                st.write(
                    f"**Optimization:** {n_trials} trials, "
                    f"{f'{k_folds}-fold CV' if k_folds > 1 else 'single split (no CV)'}, "
                    f"{test_size*100:.0f}% test set"
                )

            from services import get_job_executor, get_job_store

            store = get_job_store()
            executor = get_job_executor()

            job_target_band = int(
                st.session_state.get("fusion_target_band", target_band)
            )
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
                    "cgi_formula": cgi_formula,
                    "covariate_columns": list(covariate_columns or []),
                    "standalone_channels": (
                        ["veg", "terrain", "ndvi"] if run_standalones else []
                    ),
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
                    "val_size": float(val_size),
                    "k_folds": int(k_folds),
                    "ndvi_start_date": ndvi_auto_start.isoformat(),
                    "ndvi_end_date": ndvi_auto_end.isoformat(),
                    "target_path": tmp_target_path,
                    "target_fingerprint": file_size_mtime_fingerprint(tmp_target_path),
                    # Canonical metric paths persisted under the UI-side
                    # names (``gvi_*`` / ``ndvi_*``); the runner reads them
                    # as ``veg_path`` / ``ndvi_path`` (legacy naming).
                    "gvi_path": veg_path,
                    "gvi_fingerprint": file_size_mtime_fingerprint(veg_path),
                    "ndvi_path": ndvi_path,
                    "ndvi_fingerprint": file_size_mtime_fingerprint(ndvi_path),
                    "has_api_key": False,
                    "metric_mode": "Upload Files",
                    "longitudinal_spec_payload": longitudinal_spec_payload,
                    "cgi_grid_spacing_m": (
                        int(cgi_grid_spacing_m_param)
                        if cgi_grid_spacing_m_param is not None and is_polygon_target_ui
                        else None
                    ),
                    "whole_grid_scaling": (
                        whole_grid_scaling_param if is_polygon_target_ui else False
                    ),
                    "area_balanced_split": (
                        area_balanced_split_param if is_polygon_target_ui else False
                    ),
                },
            )
            # Attach per-wave file fingerprints to the spec payload so the
            # restart panel can detect drift on individual per-wave files.
            if longitudinal_spec_payload is not None:
                _fps: dict[str, dict[str, str]] = {
                    "target": {},
                    "veg": {},
                    "terrain": {},
                    "ndvi": {},
                }
                for _w, _p in (
                    longitudinal_spec_payload.get("target_files_per_wave") or {}
                ).items():
                    _fps["target"][_w] = file_size_mtime_fingerprint(_p)
                for _ch in ("veg", "terrain", "ndvi"):
                    for _w, _p in (
                        longitudinal_spec_payload.get("greenery_files", {}).get(_ch)
                        or {}
                    ).items():
                        _fps[_ch][_w] = file_size_mtime_fingerprint(_p)
                fusion_record.params.setdefault(
                    "longitudinal_spec_payload", longitudinal_spec_payload
                )
                fusion_record.params["longitudinal_spec_payload"][
                    "__file_fingerprints__"
                ] = _fps
            executor.submit_runner(
                fusion_record,
                run_fusion,
                target_path=tmp_target_path,
                target_features_geojson=(
                    tuple(target_outcome_columns) if is_vector_target else ()
                ),
                target_band=job_target_band if is_raster_target else 1,
                target_layer=(target_layer_for_engine if is_vector_target else None),
                target_cleanup_dir=(target_mat.cleanup_dir if target_mat else None),
                target_cleanup_file=(target_mat.cleanup_file if target_mat else None),
                buffer_meters=buffer_extent_m,
                gvi_buffer_min_m=gvi_buffer_min_m,
                gvi_buffer_max_m=gvi_buffer_max_m,
                gvi_buffer_step_m=gvi_buffer_step_m,
                ndvi_buffer_min_m=ndvi_buffer_min_m,
                ndvi_buffer_max_m=ndvi_buffer_max_m,
                ndvi_buffer_step_m=ndvi_buffer_step_m,
                n_bins=n_bins,
                veg_path=veg_path,
                ndvi_path=ndvi_path,
                test_size=test_size,
                val_size=float(val_size),
                k_folds=k_folds,
                n_trials=n_trials,
                n_startup_trials=n_startup_trials,
                objective_metric=objective_metric,
                sampler_type=optimizer,
                output_dir=output_dir,
                MetricFusionEngine=MetricFusionEngine,
                target_display_name=target_display_name,
                resume_existing_study=resume_existing_study,
                cgi_formula=cgi_formula,
                covariate_columns=list(covariate_columns or []),
                standalone_channels=(
                    ["veg", "terrain", "ndvi"] if run_standalones else []
                ),
                longitudinal_spec_payload=longitudinal_spec_payload,
                cgi_grid_spacing_m=(
                    float(cgi_grid_spacing_m_param)
                    if cgi_grid_spacing_m_param is not None and is_polygon_target_ui
                    else None
                ),
                whole_grid_scaling=(
                    whole_grid_scaling_param if is_polygon_target_ui else False
                ),
                area_balanced_split=(
                    area_balanced_split_param if is_polygon_target_ui else False
                ),
            )

            st.success("✅ Fusion job started! Check sidebar for progress.")
    # Result-loading is explicit — use the **Load results** button on a
    # completed job card in the sidebar Job Monitor to populate the
    # results panel. The previous auto-load-first-terminal-job behaviour
    # was removed at the user's request so they can choose which run to
    # inspect (or none at all).
