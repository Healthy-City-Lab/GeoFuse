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
    lon_payload = p.get("longitudinal_spec_payload") or None
    k_folds = int(p.get("k_folds", 5))
    cv_label = f"{k_folds}-fold CV" if k_folds > 1 else "single split (no CV)"
    lines = [
        f"**Target:** `{p.get('target_display_name', '?')}`",
        f"**Outcomes:** {', '.join(p.get('outcome_columns') or []) or '—'}",
        f"**CGI formula:** `{p.get('cgi_formula') or 'weighted_average'}`",
        f"**Covariates:** {', '.join(covs) if covs else '—'}",
        f"**Standalone metrics:** " f"{', '.join(standalones) if standalones else '—'}",
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
                    use_container_width=True,
                ):
                    st.session_state[RESTART_SESSION_KEY] = None
                    st.rerun()
            with rerun_col:
                if st.button(
                    "Re-run",
                    type="primary",
                    key=f"f_restart_confirm_silent_{rec.id}",
                    use_container_width=True,
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
                use_container_width=True,
            ):
                st.session_state[RESTART_SESSION_KEY] = None
                st.rerun()
        with rerun_col:
            rerun_clicked = st.button(
                "Verify & re-run",
                type="primary",
                key=f"f_restart_confirm_{rec.id}",
                disabled=lon_restart_blocked,
                use_container_width=True,
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
        help=(
            "**Cross-sectional** — one observation per entity; OLS-based "
            "partial correlation / incremental R² / RMSE / MI. "
            "**Mixed-effects (longitudinal)** — entities measured at "
            "multiple time points; CGI scored via "
            "`statsmodels.MixedLM` with random intercept (+ optional "
            "random slope on time) per entity. Pick a mode to reveal the "
            "rest of the form."
        ),
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

    if not is_longitudinal:
        # ── Cross-sectional ──────────────────────────────────────────────
        date_on = st.checkbox(
            "Date column available?",
            value=False,
            key="fusion_cross_date_on",
            help=(
                "Turn on if the target carries a measurement-date column. "
                "When on, distinct measurement years are discovered and "
                "drive the per-channel metric-file assignment below, so "
                "entities measured in different years can sample greenery "
                "from the right per-year file. The year column is only a "
                "metric-file routing key — it never enters the regression."
            ),
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
                    help=(
                        "Column carrying each row's measurement date. "
                        "Accepts ISO (`2010-01-15`), year+month (`2010-01`), "
                        "year-only strings (`2010`), or numeric years."
                    ),
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
    intake_mode = st.radio(
        "Intake mode",
        options=("long", "wide"),
        index=0,
        horizontal=True,
        key="fusion_lon_intake",
        format_func=lambda x: (
            "Long-format target" if x == "long" else "Wide / multi-file"
        ),
        help=(
            "**long** — one target file with one row per (entity, wave) "
            "and an explicit wave column. **wide** — N target files, one "
            "per wave, joined on a shared entity-id column."
        ),
    )
    state["intake_mode"] = intake_mode

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
                    help=(
                        "Column that uniquely identifies each entity "
                        "(e.g. participant ID). Must be present in the "
                        "long-format target."
                    ),
                )
            else:
                st.warning("No candidate ID columns found in the target.")
        with lc2:
            if date_candidates:
                state["date_col"] = st.selectbox(
                    "Date column",
                    options=date_candidates,
                    key="fusion_lon_date_col",
                    help=(
                        "Per-row measurement date. Used to derive "
                        "`years_since_baseline` per entity."
                    ),
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
                help=(
                    "Column carrying each row's wave label. Distinct "
                    "values populate the per-channel file-assignment grid."
                ),
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
    st.caption(
        "Upload one target file per wave. Each file contributes its own "
        "rows; they are joined on the entity-id column at load time. "
        "Wave labels default to each file's basename and can be edited."
    )
    if "fusion_lon_wide_count" not in st.session_state:
        st.session_state.fusion_lon_wide_count = 1
    n_wide = st.session_state.fusion_lon_wide_count
    from file_picker import FT_VECTOR_OR_RASTER, pick_file_path

    wide_files: list[dict] = []
    for i in range(n_wide):
        with st.container(border=True):
            wc1, wc2 = st.columns([3, 1])
            with wc1:
                file_path = pick_file_path(
                    f"Wave file {i + 1}",
                    key=f"fusion_lon_wide_path_{i}",
                    file_types=FT_VECTOR_OR_RASTER,
                    help_text=(
                        "Pick a per-wave target file from disk. The path "
                        "picker bypasses Streamlit's upload limit so "
                        "large per-wave files are supported directly."
                    ),
                )
            with wc2:
                if i > 0 and st.button(
                    "❌",
                    key=f"fusion_lon_wide_rm_{i}",
                    help="Remove this wave file",
                ):
                    st.session_state.fusion_lon_wide_count -= 1
                    st.rerun()
            default_label = ""
            if file_path:
                default_label = os.path.splitext(os.path.basename(file_path))[0]
            wave_label = st.text_input(
                "Wave label",
                value=st.session_state.get(f"fusion_lon_wide_label_{i}", default_label),
                key=f"fusion_lon_wide_label_{i}",
                help="Wave identifier; ordered by appearance, baseline first.",
            )
            file_cols: list[str] = []
            file_date_cands: list[str] = []
            if file_path:
                if not os.path.isfile(file_path):
                    st.error(f"Path no longer exists: `{file_path}`")
                    file_path = None
                else:
                    try:
                        file_gdf = read_vector_path(file_path)
                        file_cols = [c for c in file_gdf.columns if c != "geometry"]
                        file_date_cands = _date_parseable_columns(file_gdf)
                    except Exception as exc:
                        st.error(f"Could not read wave {i + 1}: {exc}")
            ec1, ec2 = st.columns(2)
            with ec1:
                entity_col = st.selectbox(
                    "Entity ID column",
                    options=file_cols or ["—"],
                    key=f"fusion_lon_wide_entity_{i}",
                    disabled=not file_cols,
                )
            with ec2:
                date_col = st.selectbox(
                    "Date column",
                    options=file_date_cands or ["—"],
                    key=f"fusion_lon_wide_date_{i}",
                    disabled=not file_date_cands,
                )
            if file_path and wave_label and file_cols and file_date_cands:
                wide_files.append(
                    {
                        "path": file_path,
                        "wave_label": wave_label,
                        "entity_col": entity_col,
                        "date_col": date_col,
                    }
                )

    if st.button(
        "+ Add another wave file",
        key="fusion_lon_wide_add",
        help="Add a row for one more wave's target file.",
    ):
        st.session_state.fusion_lon_wide_count += 1
        st.rerun()

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
                        st.session_state.get(f"fusion_{ch_short}_buffer_max", 1500)
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
                        st.session_state.get(f"fusion_{ch_short}_buffer_step", 50)
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
                        help=(
                            "Years / waves whose entities should sample "
                            "greenery from this file. Years already "
                            "claimed by another file in this channel "
                            "are hidden from the list."
                        ),
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
                covered: dict[str, int] = {w: 0 for w in discovered_waves}
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
                "**weighted_average** — three weights on min-max-"
                "normalized veg / terrain / NDVI (sum = 100). "
                "**synergy** — three-metric generalisation of Wang et "
                "al. 2026: seven weights (sum = 100) plus three powers "
                "on the main NDVI / Veg / Terrain terms only."
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
                    "Additional numeric attribute columns to control for "
                    "when scoring the CGI's predictive power. With "
                    "covariates the score becomes the greenery term's "
                    "*partial* contribution (partial correlation, "
                    "incremental R², or full-model RMSE). "
                    "`mutual_info` ignores covariates by design."
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
            help=(
                "Quantity Optuna maximises (or minimises for RMSE) per "
                "trial. Options switch automatically with the run mode: "
                "OLS-based metrics for cross-sectional; MixedLM-based "
                "metrics for longitudinal."
            ),
        )
    with col_o2:
        n_trials = st.number_input(
            "Total trials",
            min_value=50,
            max_value=1000,
            value=int(st.session_state.get("fusion_n_trials", 300)),
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
            value=int(st.session_state.get("fusion_n_startup", 150)),
            step=10,
            key="fusion_n_startup",
            help="Uniformly random trials before the main sampler engages.",
        )
    with col_s2:
        use_cv = st.checkbox(
            "Use k-fold cross-validation",
            value=bool(st.session_state.get("fusion_use_cv", True)),
            key="fusion_use_cv",
            help=(
                "When on, each trial fits k models on a stratified k-fold "
                "split. When off, a single train/val split is used and "
                "each trial fits one model — ~k× faster, no fold-variance."
            ),
        )
    with col_s3:
        if use_cv:
            k_folds = st.number_input(
                "K-Fold CV",
                min_value=3,
                max_value=10,
                value=int(st.session_state.get("fusion_k_folds", 5)),
                key="fusion_k_folds",
                help=(
                    "Cross-validation folds on the non-test subset. The "
                    "test set is always carved off first via the "
                    "**Test set size** slider regardless of this toggle."
                ),
            )
        else:
            k_folds = 1
            st.caption(
                "_Single train/val split on the non-test subset; "
                "test set still held out._"
            )
    with col_s4:
        test_size = st.slider(
            "Test set size",
            min_value=0.1,
            max_value=0.5,
            value=float(st.session_state.get("fusion_test_size", 0.3)),
            step=0.05,
            key="fusion_test_size",
            help="Held-out evaluation fraction.",
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
                help=(
                    "Switches the random-effects structure from "
                    "`(1 | entity)` to `(1 + years_since_baseline | "
                    "entity)`. Costs more fit iterations but lets each "
                    "entity's trajectory have its own slope."
                ),
            )
        with mc2:
            mixedlm_time_fixed = st.checkbox(
                "Include `years_since_baseline` as fixed effect",
                value=bool(st.session_state.get("fusion_lon_include_time_fixed", True)),
                key="fusion_lon_include_time_fixed",
                help=(
                    "Adds `+ years_since_baseline` to the fixed-effect "
                    "design. Keep on unless you want any global temporal "
                    "trend to load onto the greenery coefficient."
                ),
            )

    # ── Resume + standalones ────────────────────────────────────────────
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        resume_existing_study = st.checkbox(
            "Resume previous study if exists",
            value=bool(st.session_state.get("fusion_resume_study", True)),
            key="fusion_resume_study",
            help=(
                "When on, re-running with the same target + outcome + "
                "objective metric loads the existing SQLite study and "
                "runs only the remaining trials."
            ),
        )
    with col_r2:
        run_standalones = st.checkbox(
            "Also optimize each metric on its own (NDVI / Vegetation / Terrain)",
            value=bool(st.session_state.get("fusion_run_standalones", False)),
            key="fusion_run_standalones",
            help=(
                "Adds three single-metric Optuna studies alongside the "
                "combined CGI run, searching only its radius + aggregation."
            ),
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
        "n_bins": int(n_bins),
        "mixedlm_random_slope": bool(mixedlm_random_slope),
        "mixedlm_time_fixed": bool(mixedlm_time_fixed),
        "resume_existing_study": bool(resume_existing_study),
        "run_standalones": bool(run_standalones),
    }


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

        from file_picker import FT_VECTOR_OR_RASTER, path_to_dataset, pick_file_path

        target_picked_path = pick_file_path(
            "Pick Target File",
            key="fusion_target_path",
            file_types=FT_VECTOR_OR_RASTER,
            help_text=(
                "Vector: GeoJSON, GeoPackage, shapefile (.shp with sidecars in "
                "the same folder), or vector zip. Raster: GeoTIFF. The toolbox "
                "reads from this path lazily — bytes are not copied into "
                "memory until preview, sampling, or compute needs them."
            ),
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

    with st.form("fusion_metric_run"):
        study_state = _render_study_details_panel(
            is_longitudinal=is_longitudinal,
            available_covariates=available_covariates,
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
    n_bins = study_state["n_bins"]
    resume_existing_study = study_state["resume_existing_study"]
    run_standalones = study_state["run_standalones"]
    pruner_type = "none"  # UI removed; engine accepts NopPruner via "none"
    lon_random_slope = study_state["mixedlm_random_slope"]
    lon_include_time_fixed = study_state["mixedlm_time_fixed"]

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
            from geofuse.longitudinal import GREENERY_CHANNELS as _LON_CHANNELS
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

            # ── Mixed-effects post-hoc metrics (when present) ────────────────
            # The post-score stage writes one CSV per study (CGI + each
            # standalone) into ``output_results/fusion/study_results/``.
            # Naming:
            #   single outcome: ``mixedlm_metrics.csv``,
            #                   ``mixedlm_metrics__<channel>.csv``
            #   multi outcome:  ``mixedlm_metrics__<outcome>.csv``,
            #                   ``mixedlm_metrics__<outcome>__<channel>.csv``
            # Show one tab per available file so users can compare studies.
            try:
                _study_root = os.path.join(output_dir, "fusion", "study_results")
                _active_label = (
                    st.session_state.get("fusion_results_outcome_pick")
                    if results.get("mode") == "multi"
                    else None
                )

                def _match_outcome(fname: str) -> bool:
                    # Single-outcome run: only the no-outcome-prefix files
                    # apply. Multi-outcome run: keep files whose name carries
                    # the active outcome label.
                    if _active_label is None:
                        return f"__{_active_label}" not in fname
                    return f"__{_active_label}" in fname

                _csv_paths: list[str] = []
                if os.path.isdir(_study_root):
                    for _fn in sorted(os.listdir(_study_root)):
                        if not _fn.startswith("mixedlm_metrics") or not _fn.endswith(
                            ".csv"
                        ):
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
                        # Trim the outcome prefix in multi-mode so the tabs
                        # read like "CGI / Veg / Terrain / NDVI" instead of
                        # "<outcome> / <outcome>__veg / …".
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
                            st.dataframe(pd.read_csv(_path), use_container_width=True)
            except Exception as _exc:  # pragma: no cover -- UI-only guard
                st.warning(f"Could not read mixedlm_metrics CSVs: {_exc}")
