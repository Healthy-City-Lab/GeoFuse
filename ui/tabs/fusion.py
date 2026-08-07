import base64
import glob
import html
import io
import math
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

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
from geofuse import mixed_effects_scoring as _mes
from geofuse.crs_utils import buffer_gdf_union_metres, reproject_geodataframe_to_wgs84
from geofuse.vector_io import (
    geometry_sha256,
    list_gpkg_layer_names,
    read_vector_path,
    vector_format_from_path,
)


def _file_sig(path: str) -> tuple:
    """(size, mtime_ns) fingerprint so a cache entry invalidates on file change."""
    try:
        s = os.stat(path)
        return (int(s.st_size), int(s.st_mtime_ns))
    except OSError:
        return (0, 0)


@st.cache_data(show_spinner=False, max_entries=8)
def _read_vector_cached(
    path: str, sig: tuple, layer: str | int | None = None
) -> "gpd.GeoDataFrame":
    """``read_vector_path`` memoized by (path, file fingerprint, layer).

    Streamlit reruns the whole tab on every widget edit; without this the target
    file was re-read and reprojected each keystroke — the main cause of the tab
    stalling on large targets. ``sig`` is part of the key so an edited file is
    re-read.
    """
    return (
        read_vector_path(path, layer=layer)
        if layer is not None
        else read_vector_path(path)
    )


def _read_vector_for_ui(
    path: str, layer: str | int | None = None
) -> "gpd.GeoDataFrame":
    """Cached target read for the tab (keyed by the file's current fingerprint)."""
    return _read_vector_cached(path, _file_sig(path), layer)


@st.cache_data(show_spinner=False, max_entries=16)
def _read_vector_head(path: str, sig: tuple) -> "gpd.GeoDataFrame":
    """A 64-row peek, cached so wide-mode reruns don't re-read for schema/dtypes."""
    return gpd.read_file(path, rows=64)


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


# ────────────────────────────────────────────────────────────────────
# Metric-file helpers
# ────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────
# Restart workflow
# ────────────────────────────────────────────────────────────────────


def _fusion_preaggr_cache_files(p: dict, output_dir: str) -> list[str]:
    """Reusable greenery pre-aggregation cache files.

    The greenery cache is keyed on the greenery-file set + grid, not the target,
    so it is shared across jobs and cannot be attributed to one target. Every
    ``greenery-*.npz`` unit is listed — clearing them frees recomputable
    intermediates (they rebuild on the next run). Output files and result
    artefacts are never included.
    """
    greenery_dir = os.path.join(output_dir, "fusion_cache", "greenery")
    if not os.path.isdir(greenery_dir):
        return []
    return sorted(
        os.path.join(greenery_dir, n)
        for n in os.listdir(greenery_dir)
        if n.startswith("greenery-") and n.endswith(".npz")
    )


def _purge_fusion_preaggr_cache(paths: list[str]) -> tuple[int, int]:
    """Delete the given cache files; returns ``(removed, bytes_freed)``."""
    removed = 0
    freed = 0
    for path in paths:
        try:
            size = os.path.getsize(path)
            os.remove(path)
            removed += 1
            freed += size
        except OSError:
            continue
    return removed, freed


def _consume_fusion_restart(store) -> None:
    """Seed the setup form when a re-run was confirmed in the job monitor.

    The confirmation happens on the job card itself, so by the time the key
    arrives here the user has already said yes: seed and let the normal flow
    render the pre-filled form. The toast is the only acknowledgement — the
    filled form speaks for itself.
    """
    job_id = st.session_state.get(RESTART_SESSION_KEY)
    if not job_id:
        return
    rec = store.get(job_id)
    if rec is None or rec.type != "fusion":
        return

    st.session_state[RESTART_SESSION_KEY] = None
    _seed_fusion_form(rec.params or {})
    st.toast(f"Loaded settings from job {rec.name or rec.id}.", icon="🔄")


# ────────────────────────────────────────────────────────────────────
# Section panels (new layout)
# ────────────────────────────────────────────────────────────────────

# Public set of cross-sectional objective metrics offered by the OLS scorer
# (covariate-aware distance correlation / partial rank correlation /
# incremental R² / normalized RMSE / MI).
_CROSS_METRICS: tuple[str, ...] = (
    "partial_distance_corr",
    "distance_corr",
    "spearman",
    "r2",
    "nrmse",
    "mutual_info",
    "logit_tstat",
    "logit_coef",
)
# Friendly labels for the objective-metric picker.
_CROSS_METRIC_LABELS: dict[str, str] = {
    "partial_distance_corr": "Partial distance correlation (linear + nonlinear, covariate-aware)",
    "distance_corr": "Distance correlation (fast; linear covariate adjustment)",
    "spearman": "Partial rank correlation (Spearman)",
    "r2": "Incremental R²",
    "nrmse": "Normalized RMSE (lower is better)",
    "mutual_info": "Mutual information (ignores covariates)",
    "logit_tstat": "Logistic Wald |z| (binary outcome)",
    "logit_coef": "Logistic log-odds ratio (binary outcome)",
    "mixedlm_tstat": "Mixed-effects |t| (panel)",
    "mixedlm_marginal_r2": "Mixed-effects marginal R² (panel)",
    "mixedlm_lr": "Mixed-effects likelihood ratio (panel)",
    "mixedlm_coef": "Mixed-effects coefficient (panel)",
    "gee_logit_tstat": "GEE logistic Wald |z| (binary panel)",
    "gee_logit_coef": "GEE logistic log-odds ratio (binary panel)",
}
# Metrics that require a two-valued outcome. Mirror of
# ``objective_scoring.BINARY_ONLY_METRICS`` plus the GEE panel pair.
_BINARY_ONLY_METRICS: frozenset[str] = frozenset(
    {"logit_tstat", "logit_coef", "gee_logit_tstat", "gee_logit_coef"}
)
# Metrics that condition on covariates intrinsically, so the residualization
# picker doesn't apply. Mirror of ``objective_scoring.RESIDUALIZE_IGNORED``.
_RESIDUALIZE_IGNORED_METRICS: frozenset[str] = frozenset(
    {"partial_distance_corr", "mutual_info"}
)
# MixedLM scoring metrics — mirror the engine's MIXEDLM_METRICS so they can
# round-trip through the spec without an explicit import.
_MIXEDLM_METRICS: tuple[str, ...] = (
    "mixedlm_tstat",
    "mixedlm_marginal_r2",
    "mixedlm_lr",
    "mixedlm_coef",
    # Binary outcomes on a panel: MixedLM is Gaussian-only, so these route to
    # GEE with an exchangeable working correlation clustered on the entity.
    "gee_logit_tstat",
    "gee_logit_coef",
)

# Authoritative list of every ``run_fusion`` setting that isn't a file path or
# runtime object. The submit path records exactly these (under the same names)
# and the restart path replays exactly these — so a re-run can never silently
# substitute a default for a forgotten setting. Keep in sync with the
# ``run_fusion`` signature and the ``run_config`` dict built at submit time.
_FUSION_RUN_CONFIG_KEYS: tuple[str, ...] = (
    "buffer_meters",
    "gvi_buffer_min_m",
    "gvi_buffer_max_m",
    "gvi_buffer_step_m",
    "ndvi_buffer_min_m",
    "ndvi_buffer_max_m",
    "ndvi_buffer_step_m",
    "ndvi_resolution_m",
    "gvi_grid_spacing_m",
    "n_bins",
    "cache_metrics",
    "test_size",
    "objective_metric",
    "residualize_method",
    "search_scoring_method",
    "ndvi_start_date",
    "ndvi_end_date",
    "ndvi_project_id",
    "multi_objective_requested",
    "target_display_name",
    "resume_existing_study",
    "cgi_formula",
    "covariate_columns",
    "covariate_types",
    "moderator_columns",
    "exposure_iqr",
    "standalone_channels",
    "longitudinal_spec_payload",
    "cgi_grid_spacing_m",
    "whole_grid_scaling",
    "area_balanced_split",
    "normalize_channels",
    "spatial_adjust_method",
    "spatial_adjust_max_df",
    "spatial_adjust_eps_m",
    "spatial_split",
    "spatial_block_size_m",
    "n_spatial_blocks",
    "n_bootstraps",
    "n_trials_per_bootstrap",
    "weight_bin_pct",
    "weight_refine_bin_pct",
    "min_cell_count",
    "worst_quantile",
    "max_pfer",
    "check_collinearity",
    "vif_threshold",
)

# Recorded param -> form widget key. Re-run seeds these so the normal setup
# form comes up filled in; anything not listed keeps its own default.
_FUSION_PARAM_TO_WIDGET: dict[str, str] = {
    "cgi_formula": "fusion_cgi_formula",
    "objective_metric": "fusion_objective_metric",
    "residualize_method": "fusion_residualize_method",
    "search_scoring_method": "fusion_search_scoring_method",
    "test_size": "fusion_test_size",
    "n_bins": "fusion_stratification_bins",
    "n_bootstraps": "fusion_n_bootstraps",
    "n_trials_per_bootstrap": "fusion_n_trials_per_bootstrap",
    "weight_bin_pct": "fusion_weight_bin_pct",
    "weight_refine_bin_pct": "fusion_weight_refine_bin_pct",
    "max_pfer": "fusion_max_pfer",
    "check_collinearity": "fusion_check_collinearity",
    "vif_threshold": "fusion_vif_threshold",
    "cgi_grid_spacing_m": "fusion_cgi_grid_spacing_m",
    "whole_grid_scaling": "fusion_whole_grid_scaling",
    "area_balanced_split": "fusion_area_balanced_split",
    "normalize_channels": "fusion_normalize_channels",
    "spatial_split": "fusion_spatial_split",
    "spatial_block_size_m": "fusion_spatial_block_size_m",
    "spatial_adjust_method": "fusion_spatial_adjust_method",
    "spatial_adjust_max_df": "fusion_spatial_adjust_max_df",
    "spatial_adjust_eps_m": "fusion_spatial_adjust_eps_m",
    "resume_existing_study": "fusion_resume_study",
    "multi_objective_requested": "fusion_multi_objective_run",
    "gvi_buffer_min_m": "fusion_gvi_buffer_min",
    "gvi_buffer_max_m": "fusion_gvi_buffer_max",
    "gvi_buffer_step_m": "fusion_gvi_buffer_step",
    "ndvi_buffer_min_m": "fusion_ndvi_buffer_min",
    "ndvi_buffer_max_m": "fusion_ndvi_buffer_max",
    "ndvi_buffer_step_m": "fusion_ndvi_buffer_step",
}

# Widgets that are integer-typed ``st.number_input`` / slider controls; the
# recorded params hold floats, and Streamlit refuses a float value on an
# int widget.
_FUSION_SEED_INT_WIDGETS: frozenset[str] = frozenset(
    {
        "fusion_gvi_buffer_min",
        "fusion_gvi_buffer_max",
        "fusion_gvi_buffer_step",
        "fusion_ndvi_buffer_min",
        "fusion_ndvi_buffer_max",
        "fusion_ndvi_buffer_step",
        "fusion_stratification_bins",
        "fusion_n_bootstraps",
        "fusion_n_trials_per_bootstrap",
        "fusion_weight_bin_pct",
        "fusion_cgi_grid_spacing_m",
        "fusion_spatial_block_size_m",
        "fusion_spatial_adjust_max_df",
        "fusion_spatial_adjust_eps_m",
    }
)

# Longitudinal spec field -> form widget key.
_FUSION_LON_TO_WIDGET: dict[str, str] = {
    "entity_id_col": "fusion_lon_entity_id_col",
    "wave_col": "fusion_lon_wave_col",
    "date_col": "fusion_lon_date_col",
    "association_target": "fusion_lon_assoc_target",
    "random_slope_time": "fusion_lon_random_slope",
    "include_time_fixed_effect": "fusion_lon_include_time_fixed",
    "include_wave_fixed_effects": "fusion_lon_wave_fe",
    "decline_average_exposure": "fusion_lon_decline_between",
    "decline_exposure_change": "fusion_lon_decline_within",
}


def _seed_fusion_form(p: dict) -> None:
    """Copy a finished job's settings into the form's session state.

    Re-run is a pre-filled form, not a replay: the user lands on the normal
    setup with everything from the original job already in place and submits
    with the usual button, tweaking whatever they want on the way.
    """
    for param, widget in _FUSION_PARAM_TO_WIDGET.items():
        if p.get(param) is not None:
            val = p[param]
            if widget in _FUSION_SEED_INT_WIDGETS:
                val = int(val)
            st.session_state[widget] = val
    st.session_state["fusion_run_standalones"] = bool(p.get("standalone_channels"))

    types = dict(p.get("covariate_types") or {})
    cols = list(p.get("covariate_columns") or [])
    cat = [c for c in cols if str(types.get(c, "")).lower() == "categorical"]
    # Categorical membership implies covariate membership, so the numeric list
    # holds only what isn't tagged.
    st.session_state["fusion_covariate_columns"] = [c for c in cols if c not in cat]
    st.session_state["fusion_covariate_categorical"] = cat
    # Effect modifiers are their own list — a moderator need not be a covariate,
    # so it cannot be recovered from the two lists above.
    st.session_state["fusion_moderator_columns"] = list(
        p.get("moderator_columns") or []
    )
    st.session_state["fusion_exposure_iqr"] = float(p.get("exposure_iqr") or 0.0)

    lon = p.get("longitudinal_spec_payload") or {}
    is_longitudinal = bool(lon) and str(
        lon.get("scoring_metric", "")
    ).startswith("mixedlm")
    st.session_state["fusion_run_mode"] = (
        _FUSION_RUN_MODE_LON if is_longitudinal else _FUSION_RUN_MODE_CROSS
    )
    for field, widget in _FUSION_LON_TO_WIDGET.items():
        if lon.get(field) is not None:
            st.session_state[widget] = lon[field]
    st.session_state["fusion_lon_area_col"] = lon.get("area_id_col") or "(none)"

    # ── Wave / year source ──────────────────────────────────────
    # Greenery is keyed either to the calendar years found in the date column
    # or to an explicit wave column, and each intake mode reads that choice
    # from a different widget.
    by_year = bool(lon.get("derive_wave_from_date"))
    if lon:
        st.session_state["fusion_lon_assign_mode"] = (
            _FUSION_ASSIGN_BY_YEAR if by_year else _FUSION_ASSIGN_BY_WAVE
        )
    if not is_longitudinal:
        # A cross-sectional run only carries a spec when it routes greenery by
        # measurement year; without one the date control stays off.
        st.session_state["fusion_cross_date_on"] = bool(lon)
        if lon.get("date_col"):
            st.session_state["fusion_cross_date_col"] = lon["date_col"]

    # Wide intake picks the entity / date column per file, so those widgets are
    # keyed by path. Files whose own choice was not recorded fall back to the
    # canonical pair the spec carries.
    from file_picker import path_to_widget_id

    per_file_cols = lon.get("target_columns_per_wave") or {}
    for wave, path in (lon.get("target_files_per_wave") or {}).items():
        cols = per_file_cols.get(wave) or {}
        wid = path_to_widget_id(path)
        entity_col = cols.get("entity_col") or lon.get("entity_id_col")
        date_col = cols.get("date_col") or lon.get("date_col")
        if entity_col:
            st.session_state[f"fusion_lon_wide_entity__{wid}"] = entity_col
        if date_col:
            st.session_state[f"fusion_lon_wide_date__{wid}"] = date_col

    # ── Metric files + their year assignment ────────────────────
    # Longitudinal runs record the greenery sources per (channel, wave), not
    # as the single ``gvi_path`` / ``ndvi_path`` a cross-sectional run uses, so
    # the pickers are rebuilt from that map. Veg and terrain share one GVI file.
    waves = list(lon.get("wave_labels") or [])
    for short, channels in (("gvi", ("veg", "terrain")), ("ndvi", ("ndvi",))):
        by_path: dict[str, list[str]] = {}
        for ch in channels:
            for wave, path in (lon.get("greenery_files") or {}).get(ch, {}).items():
                if wave not in by_path.setdefault(path, []):
                    by_path[path].append(wave)
        if not by_path:
            continue
        order = sorted(by_path, key=lambda p_: waves.index(by_path[p_][0])
                       if by_path[p_] and by_path[p_][0] in waves else 0)
        st.session_state[f"fusion_{short}_paths"] = order
        st.session_state[f"fusion_{short}_year_assign"] = {
            path: [w for w in waves if w in by_path[path]] or by_path[path]
            for path in order
        }

    # Cross-sectional runs carry one file per channel instead.
    for short, key in (("gvi", "gvi_path"), ("ndvi", "ndvi_path")):
        if not st.session_state.get(f"fusion_{short}_paths") and p.get(key):
            st.session_state[f"fusion_{short}_paths"] = [p[key]]

    # ── Target files ────────────────────────────────────────────
    # Wide intake has one target file per wave, keyed by its own labels (which
    # need not be the greenery wave labels). Recorded in picker order, so the
    # first stays the baseline. ``target_path`` alone is just that baseline.
    per_wave_targets = lon.get("target_files_per_wave") or {}
    if per_wave_targets:
        st.session_state["fusion_target_paths"] = list(per_wave_targets.values())
        # The picker's per-file label inputs carry the wave labels, so custom
        # labels must round-trip too (they key ``target_files_per_wave``).
        for wave, path in per_wave_targets.items():
            st.session_state[
                f"fusion_target_paths__label__{path_to_widget_id(path)}"
            ] = str(wave)
    elif p.get("target_path"):
        st.session_state["fusion_target_paths"] = [p["target_path"]]

    # ── Outcome columns ─────────────────────────────────────────
    # The target picker clears this list whenever the first file's signature
    # differs from the one it last saw, so the signature is seeded alongside
    # it — read from the picker's own first entry, which is what that check
    # compares against, rather than from the recorded baseline path.
    outcomes = list(p.get("outcome_columns") or [])
    picked = st.session_state.get("fusion_target_paths") or []
    if outcomes:
        st.session_state["fusion_outcome_columns"] = outcomes
        if picked and os.path.isfile(picked[0]):
            st.session_state["fusion_target_upload_sig"] = (
                picked[0],
                os.path.getsize(picked[0]),
            )


def _keep_valid(key: str, options, *, multi: bool = False) -> None:
    """Drop a held selection the current options no longer offer.

    Seeded and pinned values outlive the file they came from, and a widget
    handed a value outside its options either raises or silently reports a
    column that isn't there. A single-select drops its key entirely so the
    caller's default can take over; a multiselect keeps whatever still fits.
    """
    if key not in st.session_state:
        return
    val = st.session_state[key]
    if multi:
        st.session_state[key] = [v for v in (val or []) if v in options]
    elif val not in options:
        del st.session_state[key]


def _default(key: str, value, options=None, *, multi: bool = False):
    """Set a widget's first-run value, then let session state own it.

    Passing ``value=`` / ``index=`` alongside a ``key`` whose state was written
    through the session-state API makes Streamlit render a warning box into the
    layout, and the argument is ignored anyway. Seeding the key beforehand says
    the same thing without either problem.

    Pass ``options`` for a picker whose choices come from the data: stale picks
    are pruned first so the default fills the gap they leave, never the other
    way round. An emptied multiselect is left empty — deselecting everything is
    a choice. Returns the value in force.
    """
    if options is not None:
        _keep_valid(key, options, multi=multi)
    if key not in st.session_state:
        st.session_state[key] = value
    return st.session_state[key]


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
    years = sorted({int(d.year) for d in parsed.dropna()})
    return [str(y) for y in years]


def _cached_years_for_file(path: str, date_col: str) -> list[str]:
    """Years present in ``path``'s date column, memoized per (path, mtime, col).

    The wide-mode panel re-renders on every interaction; without this the
    per-wave files would be re-read from disk each time just to list years.
    """
    if not path or not date_col or not os.path.isfile(path):
        return []
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return []
    store = st.session_state.setdefault("_fusion_year_cache", {})
    key = (path, mtime, date_col)
    hit = store.get(key)
    if hit is not None:
        return list(hit)
    try:
        years = _discover_years_from_date_column(_read_vector_for_ui(path), date_col)
    except Exception:
        years = []
    store[key] = list(years)
    return list(years)


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
        "association_target": "level",
        "include_wave_fixed_effects": True,
        "area_id_col": None,
        "decline_average_exposure": False,
        "decline_exposure_change": False,
        # True when greenery files are assigned per calendar year rather than
        # per wave file; rows then route by their own measurement date.
        "assign_by_year": False,
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
        # ── Cross-sectional ─────────────────────────────────
        _default("fusion_cross_date_on", False)
        date_on = st.checkbox(
            "Date column available?",
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
                _keep_valid("fusion_cross_date_col", date_candidates)
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

    # ── Longitudinal ────────────────────────────────────────────
    # Intake mode is auto-selected from how many files are in Section 1:
    # one file ⇒ long format (single table with a wave column), two or
    # more ⇒ wide format (one file per wave, joined on the entity id).
    if n_files <= 1:
        intake_mode = "long"
    else:
        intake_mode = "wide"
    state["intake_mode"] = intake_mode
    st.caption(f"Intake mode: **{intake_mode}** ")

    _assign_opts = [_FUSION_ASSIGN_BY_YEAR, _FUSION_ASSIGN_BY_WAVE]
    _default("fusion_lon_assign_mode", _FUSION_ASSIGN_BY_YEAR, _assign_opts)
    assign_mode = st.radio(
        "Assign greenery files per",
        options=_assign_opts,
        horizontal=True,
        key="fusion_lon_assign_mode",
        help=(
            "**Measurement year** — greenery is assigned to the calendar years "
            "found in the date column, and every observation reads the file for "
            "the year it was actually measured (works even when one wave file "
            "spans a year boundary). **Wave file** — one greenery file per wave, "
            "the previous behaviour; use it when an entity has two measurements "
            "in the same calendar year."
        ),
    )
    assign_by_year = assign_mode == _FUSION_ASSIGN_BY_YEAR
    state["assign_by_year"] = assign_by_year

    if intake_mode == "long":
        if preview_gdf is None:
            st.info("Upload a target file first to populate the column pickers.")
            return state
        lc1, lc2 = st.columns(2)
        with lc1:
            if id_candidates:
                _keep_valid("fusion_lon_entity_id_col", id_candidates)
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
                _keep_valid("fusion_lon_date_col", date_candidates)
                state["date_col"] = st.selectbox(
                    "Date column",
                    options=date_candidates,
                    key="fusion_lon_date_col",
                    help="Used to derive `years_since_baseline` per entity.",
                )
            else:
                st.warning("No date-parseable columns found in the target.")
        if assign_by_year:
            # Years drive the greenery assignment; the wave column is not
            # needed (intake derives each row's wave from its date).
            if state["date_col"]:
                years = _discover_years_from_date_column(
                    preview_gdf, state["date_col"]
                )
                state["discovered_waves"] = years
                if years:
                    st.caption(
                        "Discovered years: " + ", ".join(f"`{y}`" for y in years)
                    )
                else:
                    st.warning("No parseable dates in the selected column.")
            return state

        wave_candidates = [
            c
            for c in all_cols
            if c not in (state["entity_id_col"], state["date_col"])
            and c not in target_outcome_columns
        ]
        if wave_candidates:
            _keep_valid("fusion_lon_wave_col", wave_candidates)
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

    # ── Wide / multi-file intake ────────────────────────────────
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
                file_gdf = _read_vector_for_ui(path)
                file_cols = [c for c in file_gdf.columns if c != "geometry"]
                file_date_cands = _date_parseable_columns(file_gdf)
            except Exception as exc:
                st.error(f"Could not read wave {i + 1}: {exc}")
                continue
            ec1, ec2 = st.columns(2)
            with ec1:
                _keep_valid(f"fusion_lon_wide_entity__{wid}", file_cols or ["—"])
                entity_col = st.selectbox(
                    "Entity ID column",
                    options=file_cols or ["—"],
                    key=f"fusion_lon_wide_entity__{wid}",
                    disabled=not file_cols,
                )
            with ec2:
                _keep_valid(f"fusion_lon_wide_date__{wid}", file_date_cands or ["—"])
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
    if assign_by_year:
        # Union of the calendar years across every file's own date column: a
        # file that straddles a year boundary contributes both of its years.
        year_set: set[str] = set()
        for wf in wide_files:
            year_set.update(_cached_years_for_file(wf["path"], wf["date_col"]))
        state["discovered_waves"] = sorted(year_set)
        if state["discovered_waves"]:
            st.caption(
                "Discovered years across all files: "
                + ", ".join(f"`{y}`" for y in state["discovered_waves"])
            )
        elif wide_files:
            st.warning(
                "No parseable dates found in the selected date columns — pick "
                "a date column for each file, or switch to per-wave-file "
                "assignment."
            )
    else:
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

_UNASSIGNED_HEADER = "Unassigned"

# Greenery is keyed either to the calendar years in the date column or to
# an explicit wave column.
_FUSION_ASSIGN_BY_YEAR = "Measurement year"
_FUSION_ASSIGN_BY_WAVE = "Wave file"


def _container_headers(file_paths: list[str]) -> list[str]:
    """Display name per file — basename, disambiguated only when it repeats.

    The chip list above already carries the full path as a tooltip, so the
    header stays short; two files sharing a basename get their parent folder
    prefixed so the drop targets remain tellable apart.
    """
    bases = [os.path.basename(p) for p in file_paths]
    dupes = {b for b in bases if bases.count(b) > 1}
    out: list[str] = []
    for p, b in zip(file_paths, bases):
        if b in dupes:
            parent = os.path.basename(os.path.dirname(p)) or os.path.dirname(p)
            out.append(f"{parent}/{b}" if parent else b)
        else:
            out.append(b)
    return out


def _render_coverage_chips(covered: dict[str, int], waves: list[str]) -> None:
    """One chip per year/wave: the label plus an assigned/unassigned emoji.

    Each chip is a single ``nowrap`` unit so a label can never end up on a
    different line from its own status marker.
    """
    chips = []
    for w in waves:
        n = int(covered.get(w, 0))
        mark = "✅" if n == 1 else ("❌" if n == 0 else "⚠️")
        chips.append(
            "<span style='display:inline-block;white-space:nowrap;"
            "padding:2px 8px;margin:2px 6px 2px 0;border-radius:10px;"
            "border:1px solid rgba(128,128,128,0.35);font-size:0.85em;'>"
            f"{w}&nbsp;{mark}</span>"
        )
    st.markdown("".join(chips), unsafe_allow_html=True)



# Drop zones are laid out as a responsive grid inside the component, so a
# decade of waves across several files stays on a couple of rows instead of one
# full-width row per file. Headers wrap rather than truncate — the whole point
# is knowing which file a year is being dropped onto.
_SORTABLE_STYLE = """
.sortable-component {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
    gap: 0.5rem;
    align-items: start;
}
.sortable-container {
    background: rgba(128, 128, 128, 0.07);
    border: 1px solid rgba(128, 128, 128, 0.28);
    border-radius: 8px;
    padding: 0.35rem 0.45rem 0.5rem;
    min-width: 0;
}
.sortable-container:first-child {
    border-style: dashed;
    border-color: rgba(214, 122, 39, 0.75);
}
.sortable-container-header {
    font-size: 0.78rem;
    font-weight: 600;
    line-height: 1.25;
    padding: 0.15rem 0.1rem 0.35rem;
    white-space: normal;
    overflow-wrap: anywhere;
    border-bottom: 1px solid rgba(128, 128, 128, 0.25);
    margin-bottom: 0.35rem;
}
.sortable-container-body {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    min-height: 2.1rem;
}
.sortable-item {
    font-size: 0.78rem;
    padding: 0.1rem 0.45rem;
    margin: 0;
    border-radius: 5px;
}
"""


def _match_waves_to_filenames(
    file_paths: list[str], waves: list[str]
) -> dict[str, str]:
    """``{path: wave}`` for filenames that name exactly one wave, unambiguously.

    A wave is matched only when its label appears in the filename bounded by
    non-alphanumerics, so ``2011`` does not match inside ``20115``. The pairing
    has to be one-to-one in both directions: a filename naming several waves is
    ambiguous, and a wave named by several filenames is too. Everything else is
    left for the user to drag.
    """
    hits: dict[str, list[str]] = {}
    for path in file_paths:
        stem = os.path.basename(path)
        found = [
            w
            for w in waves
            if re.search(rf"(?<![0-9A-Za-z]){re.escape(str(w))}(?![0-9A-Za-z])", stem)
        ]
        if len(found) == 1:
            hits[path] = found[0]

    claims: dict[str, list[str]] = {}
    for path, wave in hits.items():
        claims.setdefault(wave, []).append(path)
    return {
        path: wave for path, wave in hits.items() if len(claims[wave]) == 1
    }


def _autofill_year_assignment(
    file_paths: list[str], waves: list[str], stored: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Fill unambiguous filename matches into an assignment, in place.

    Only files with nothing assigned yet are touched, and only with a wave no
    other file already holds — a choice the user made by hand always wins.
    """
    taken = {w for ws in stored.values() for w in ws}
    for path, wave in _match_waves_to_filenames(file_paths, waves).items():
        if stored.get(path) or wave in taken:
            continue
        stored[path] = [wave]
        taken.add(wave)
    return stored


def _render_year_fallback(
    ch_short: str,
    file_paths: list[str],
    waves: list[str],
    headers: list[str],
    stored: dict[str, list[str]],
    state_key: str,
) -> dict[str, list[str]]:
    """One multiselect per file, used when the sortables component is missing.

    Each file's options hide the years another file already claimed, so the
    assign-once rule still holds.
    """
    from file_picker import path_to_widget_id

    assignment: dict[str, list[str]] = {}
    for path, head in zip(file_paths, headers):
        taken = {w for q, ws in stored.items() if q != path for w in ws}
        options = [w for w in waves if w not in taken]
        wkey = f"fusion_{ch_short}_assign__{path_to_widget_id(path)}"
        if wkey in st.session_state:
            st.session_state[wkey] = [
                w for w in st.session_state[wkey] if w in options
            ]
        picked = st.multiselect(head, options=options, key=wkey)
        assignment[path] = list(picked)
        stored[path] = list(picked)
    st.session_state[state_key] = assignment
    return assignment


def _render_year_assignment(
    ch_short: str,
    file_paths: list[str],
    waves: list[str],
) -> dict[str, list[str]]:
    """Assign years/waves to files by dragging. Returns ``{path: [wave, ...]}``.

    Years are the draggable items and files are the drop zones, so a year can
    sit in exactly one place and assign-once coverage is structural rather than
    validated afterwards.
    """
    state_key = f"fusion_{ch_short}_year_assign"
    stored: dict[str, list[str]] = dict(st.session_state.get(state_key, {}))
    # Drop files and years that have since disappeared, so a stale assignment
    # can't resurrect a removed file or an outdated year.
    stored = {
        p: [w for w in ws if w in waves]
        for p, ws in stored.items()
        if p in file_paths
    }
    # A newly added file whose name names exactly one year is assigned for the
    # user; anything ambiguous stays in the unassigned pool.
    stored = _autofill_year_assignment(file_paths, waves, stored)
    headers = _container_headers(file_paths)

    try:
        from streamlit_sortables import sort_items
    except Exception:
        return _render_year_fallback(
            ch_short, file_paths, waves, headers, stored, state_key
        )

    claimed = {w for ws in stored.values() for w in ws}
    containers = [
        {
            "header": f"{_UNASSIGNED_HEADER} — drag onto a file",
            "items": [w for w in waves if w not in claimed],
        }
    ]
    for idx, (path, head) in enumerate(zip(file_paths, headers), start=1):
        containers.append({"header": f"{idx}. {head}", "items": stored.get(path, [])})

    # Remount when the file list or year set changes: a custom component keyed
    # on a stable key misbehaves when its item set shifts underneath.
    sig = abs(hash((tuple(file_paths), tuple(waves)))) % (10**9)
    result = sort_items(
        containers,
        multi_containers=True,
        custom_style=_SORTABLE_STYLE,
        key=f"fusion_{ch_short}_sort_{sig}",
    )
    assignment = {
        file_paths[idx - 1]: list(bucket.get("items", []))
        for idx, bucket in enumerate(result)
        if idx > 0
    }
    st.session_state[state_key] = assignment

    # The zone headers carry the file name; the full path only fits here.
    with st.expander(f"Full paths ({len(file_paths)} file(s))", expanded=False):
        for idx, path in enumerate(file_paths, start=1):
            years = assignment.get(path) or []
            span = f"{years[0]}–{years[-1]}" if len(years) > 1 else (years[0] if years else "—")
            st.caption(f"**{idx}.** `{path}` · {len(years)} year(s): {span}")
    return assignment


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
            _default(f"fusion_{ch_short}_buffer_min", 100)
            _default(f"fusion_{ch_short}_buffer_max", 1000)
            _default(f"fusion_{ch_short}_buffer_step", 100)
            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                bmin = st.number_input(
                    f"{ch_short.upper()} minimum buffer",
                    min_value=50,
                    max_value=4900,
                    step=50,
                    key=f"fusion_{ch_short}_buffer_min",
                    help=f"Smallest {ch_short.upper()} radius (m) searched by Optuna.",
                )
            with bcol2:
                bmax = st.number_input(
                    f"{ch_short.upper()} maximum buffer",
                    min_value=100,
                    max_value=5000,
                    step=50,
                    key=f"fusion_{ch_short}_buffer_max",
                    help=f"Largest {ch_short.upper()} radius (m). Extent padding uses the max.",
                )
            with bcol3:
                bstep = st.number_input(
                    f"{ch_short.upper()} buffer step",
                    min_value=10,
                    max_value=500,
                    step=10,
                    key=f"fusion_{ch_short}_buffer_step",
                    help="Radius discretisation (m).",
                )
            state[f"{ch_key}_buffer_min"] = int(bmin)
            state[f"{ch_key}_buffer_max"] = int(bmax)
            state[f"{ch_key}_buffer_step"] = int(bstep)

            # One dialog, any number of files: each pick appends to the kept
            # list and renders as its own chip (basename shown, full path as
            # the chip's tooltip).
            from file_picker import FT_VECTOR_OR_RASTER, pick_multiple_paths

            picked_paths = [
                p
                for p in pick_multiple_paths(
                    f"{ch_label} files",
                    key=f"fusion_{ch_short}_paths",
                    file_types=FT_VECTOR_OR_RASTER,
                    help_text=(
                        "Pick one or more GeoTIFF / vector metric files from "
                        "disk — select several at once in the dialog. The path "
                        "picker bypasses Streamlit's upload limit so "
                        "national-scale rasters are supported directly."
                    ),
                )
                if os.path.isfile(p)
            ]

            # Year/wave assignment: years are the draggable items and files are
            # the drop targets, so each year lands on exactly one file.
            assignment: dict[str, list[str]] = {}
            if year_aware and discovered_waves and picked_paths:
                assignment = _render_year_assignment(
                    ch_short, picked_paths, list(discovered_waves)
                )

            channel_files: list[tuple[str, list]] = [
                (p, list(assignment.get(p, []))) for p in picked_paths
            ]
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
                _render_coverage_chips(covered, discovered_waves)
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
    is_vector_target: bool = False,
    categorical_candidates: list[str] | None = None,
) -> dict:
    """Final form section: CGI formula, covariates, objective metric, …

    Returns the collected widget values; the caller wraps this in
    ``st.form`` and reads the submit click separately.
    """
    st.subheader("Study Details")

    # ── Row 1: CGI formula + covariates ─────────────────────────
    col_cgi1, col_cgi2 = st.columns([1, 2])
    with col_cgi1:
        _default("fusion_cgi_formula", "weighted_average")
        cgi_formula = st.selectbox(
            "CGI Formula",
            options=["weighted_average", "synergy"],
            help=(
                "**weighted_average** — three weights on Vegetation / "
                "Terrain / NDVI summing to 100. "
                "**synergy** — seven weights summing to 100 plus three "
                "powers on the main channel terms."
            ),
            key="fusion_cgi_formula",
        )
    with col_cgi2:
        categorical_candidates = list(categorical_candidates or [])
        if available_covariates:
            _keep_valid("fusion_covariate_columns", available_covariates, multi=True)
            numeric_pick = st.multiselect(
                "Covariates (control variables)",
                options=available_covariates,
                key="fusion_covariate_columns",
                help=(
                    "Attribute columns to control for. With covariates the "
                    "score becomes the greenery term's partial contribution. "
                    "`mutual_info` ignores covariates."
                ),
            )
            # Options are the full static column list, so this doesn't depend on
            # the selection above and stays form-safe.
            detected = ", ".join(categorical_candidates[:8]) or "none"
            _keep_valid(
                "fusion_covariate_categorical", available_covariates, multi=True
            )
            categorical_pick = st.multiselect(
                "Categorical covariates (one-hot encoded)",
                options=available_covariates,
                key="fusion_covariate_categorical",
                help=(
                    "One-hot encoded (drop-first) and entered as dummy controls. "
                    "Listing a column here is enough — it does not also need to "
                    "be in the covariate list. Tag numeric-coded categories "
                    "(e.g. an education band stored as 1–6) here too. "
                    f"Detected as non-numeric: {detected}."
                ),
            )
            cat_set = set(categorical_pick) & set(available_covariates)
            covariate_columns = sorted(set(numeric_pick) | cat_set)
            covariate_types = {
                c: ("categorical" if c in cat_set else "numeric")
                for c in covariate_columns
            }

            _keep_valid("fusion_moderator_columns", available_covariates, multi=True)
            moderator_columns = st.multiselect(
                "Effect modifiers (moderators)",
                options=available_covariates,
                key="fusion_moderator_columns",
                help=(
                    "Tested for **moderation**, not adjustment: each one gets "
                    "its own `greenery x modifier` model on the winning "
                    "composite, reporting the interaction test plus the "
                    "greenery effect at every level of the modifier. Adding a "
                    "column here does not adjust for it — list it as a "
                    "covariate too if you also want it controlled elsewhere. "
                    "Tag it above as categorical to contrast its levels; "
                    "otherwise it is centred and read at mean +/- 1 SD."
                ),
            )
            if moderator_columns:
                st.caption(
                    "Moderation reported for: "
                    + ", ".join(f"`{m}`" for m in moderator_columns)
                    + " → `moderation.csv`"
                )

            exposure_iqr_ui = st.number_input(
                "Exposure IQR for reporting (0 = from data)",
                min_value=0.0,
                max_value=10.0,
                step=0.001,
                format="%.3f",
                key="fusion_exposure_iqr",
                help=(
                    "The greenery increment the per-IQR effect in "
                    "`exposure_response.csv` is expressed in. Left at 0 it is "
                    "the interquartile range of the winning composite itself, "
                    "which makes runs on different exposures incomparable. Set "
                    "it to a published study's IQR to read your effect on that "
                    "study's scale. Reporting only — the search is unaffected."
                ),
            )
            if covariate_columns:
                st.caption(
                    "Fitting: "
                    + ", ".join(
                        f"`{c}`" + (" (categorical)" if c in cat_set else "")
                        for c in covariate_columns
                    )
                )
        else:
            covariate_columns = []
            covariate_types = {}
            moderator_columns = []
            exposure_iqr_ui = 0.0
            st.caption(
                "_No attribute columns available for covariates "
                "(raster target or no spare columns)._"
            )

    # ── Objective metric + test split + stratification bins ─────
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
            format_func=lambda m: _CROSS_METRIC_LABELS.get(m, m),
            key="fusion_objective_metric",
            help=(
                "Quantity each stability-selection trial scores on its "
                "out-of-bag rows (maximised; nrmse minimised). "
                "**Partial distance correlation** (default) captures linear and "
                "nonlinear association and conditions on covariates nonlinearly. "
                "The logistic and GEE-logistic options require a two-valued "
                "outcome and report a log-odds ratio."
            ),
        )
        if objective_metric in _BINARY_ONLY_METRICS:
            st.caption(
                ":orange[Requires a two-valued outcome column.] "
                "A continuous outcome scores zero on every trial."
            )
    with col_o2:
        _default("fusion_test_size", 0.25)
        test_size = st.slider(
            "Test set size",
            min_value=0.1,
            max_value=0.5,
            step=0.05,
            key="fusion_test_size",
            help=(
                "Held-out evaluation fraction; the remainder is the train+val "
                "pool that stability selection resamples."
            ),
        )
    with col_o3:
        _default("fusion_stratification_bins", 5)
        n_bins = st.number_input(
            "Stratification bins",
            min_value=3,
            max_value=10,
            key="fusion_stratification_bins",
            help="Quantile bins for the stratified train / test split.",
        )
    st.caption(
        f"_Test = {float(test_size)*100:.0f}% held out · "
        f"train+val pool = {(1.0 - float(test_size))*100:.0f}% "
        "(resampled by stability selection)._"
    )

    # ── Covariate residualization (cross-sectional metrics) ─────
    residualize_method = "linear"
    if not is_longitudinal:
        _res_ignored = objective_metric in _RESIDUALIZE_IGNORED_METRICS
        _res_rec = (
            "This metric conditions on covariates intrinsically, so the setting "
            "has no effect."
            if _res_ignored
            else "Spline removes nonlinear covariate effects (recommended when a "
            "covariate may relate to the outcome nonlinearly); linear is faster "
            "and assumes covariate effects are linear."
        )
        _default("fusion_residualize_method", "linear")
        residualize_method = st.selectbox(
            "Covariate residualization",
            options=["linear", "spline"],
            format_func=lambda m: {
                "linear": "Linear",
                "spline": "Spline (natural cubic)",
            }[m],
            key="fusion_residualize_method",
            disabled=_res_ignored,
            help=(
                "How continuous covariates are partialled out of the "
                f"greenery↔outcome association. {_res_rec}"
            ),
        )

    # ── Longitudinal-only mixed-effects toggles ─────────────────
    mixedlm_random_slope = True
    mixedlm_time_fixed = True
    association_target = "level"
    decline_average_exposure = False
    decline_exposure_change = False
    include_wave_fixed_effects = True
    area_id_col = None
    search_scoring_method = "mom_em3"
    if is_longitudinal:
        _target_keys = ["level", "decline_overall", "decline_average", "decline_change"]
        _target_labels = {
            "level": "Level — overall association",
            "decline_overall": "Decline — greenspace × time",
            "decline_average": "Decline — average exposure",
            "decline_change": "Decline — exposure change",
        }
        _default("fusion_lon_assoc_target", "level", _target_keys)
        association_target = st.selectbox(
            "Fit the greenspace formula to",
            options=_target_keys,
            format_func=lambda k: _target_labels[k],
            key="fusion_lon_assoc_target",
            help=(
                "What the search tunes the greenspace formula to detect. "
                "**Level** = association with the outcome itself. **Decline** = "
                "association with the outcome's rate of change over time — overall "
                "(greenspace × time), or the between-person (average exposure) / "
                "within-person (exposure change) part. The objective metric below "
                "is then computed on the chosen term."
            ),
        )
        _default("fusion_lon_random_slope", True)
        _default("fusion_lon_include_time_fixed", True)
        _default("fusion_lon_wave_fe", True)
        mc1, mc2 = st.columns(2)
        with mc1:
            mixedlm_random_slope = st.checkbox(
                "Random slope on time per entity",
                key="fusion_lon_random_slope",
                help="Switches RE structure to `(1 + years_since_baseline | entity)`.",
            )
        with mc2:
            mixedlm_time_fixed = st.checkbox(
                "Include `years_since_baseline` as fixed effect",
                key="fusion_lon_include_time_fixed",
                help="Adds `+ years_since_baseline` to the fixed-effect design.",
            )
        wc1, wc2 = st.columns(2)
        with wc1:
            include_wave_fixed_effects = st.checkbox(
                "Wave fixed effects",
                key="fusion_lon_wave_fe",
                help=(
                    "Adds one indicator per wave. Per-wave greenery layers differ "
                    "for reasons unrelated to anyone's neighbourhood — a different "
                    "satellite, a different compositing window — and that drift "
                    "tracks calendar time, so without these it lands on the "
                    "greenspace × time terms. Leave on unless you have a reason."
                ),
            )
        with wc2:
            _area_opts = ["(none)"] + list(available_covariates or [])
            _keep_valid("fusion_lon_area_col", _area_opts)
            area_id_col = st.selectbox(
                "Neighbourhood / site column",
                options=_area_opts,
                key="fusion_lon_area_col",
                help=(
                    "Groups people who share a neighbourhood (FSA, census "
                    "subdivision, site). Entered as fixed effects. Greenspace is "
                    "an area attribute, so neighbours share almost the same "
                    "exposure; without this the greenspace term's standard error "
                    "is far too small."
                ),
            )
            area_id_col = None if area_id_col == "(none)" else area_id_col
        st.caption(
            "Exposure–change over time (reported alongside the overall "
            "greenspace × time effect):"
        )
        _default("fusion_lon_decline_between", False)
        _default("fusion_lon_decline_within", False)
        dc1, dc2 = st.columns(2)
        with dc1:
            decline_average_exposure = st.checkbox(
                "Average exposure effect",
                key="fusion_lon_decline_between",
                help=(
                    "Adds a between-person term: does a higher *average* exposure "
                    "track a slower change in the outcome over time? "
                    "(person-mean greenspace × time)."
                ),
            )
        with dc2:
            decline_exposure_change = st.checkbox(
                "Exposure-change effect",
                key="fusion_lon_decline_within",
                help=(
                    "Adds a within-person term: does *increasing* exposure over "
                    "time track a slower change in the outcome? "
                    "(deviation from person-mean greenspace × time). Needs "
                    "per-wave greenery that varies over time."
                ),
            )
        _ssm_keys = ["mom_em3", "mom_em1", "mom", "exact"]
        _ssm_labels = {
            "mom_em3": "Method of Moments + 3 EM steps",
            "mom_em1": "Method of Moments + 1 EM step",
            "mom": "Method of Moments",
            "exact": "Exact Mixed Linear Model (per trial)",
        }
        _default("fusion_search_scoring_method", "mom_em3", _ssm_keys)
        search_scoring_method = st.selectbox(
            "Trial scoring method",
            options=_ssm_keys,
            format_func=lambda k: _ssm_labels[k],
            key="fusion_search_scoring_method",
            help=(
                "How each trial's mixed-model variance components are estimated "
                "while searching. Method of Moments is the fastest; adding "
                "Expectation-Maximization steps refines the estimate toward the "
                "full model at a small extra cost. More steps mean closer "
                "agreement with the exact fit and less trial-score inflation; "
                "fewer steps are faster. Exact Mixed Linear Model fits the full "
                "model on every trial — most faithful but far slower. The "
                "winning configuration is always re-fit exactly for the "
                "reported results."
            ),
        )

    # ── Resume + standalones ────────────────────────────────────
    _default("fusion_resume_study", True)
    _default("fusion_clear_preaggr", False)
    _default("fusion_run_standalones", False)
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        resume_existing_study = st.checkbox(
            "Resume previous study if exists",
            key="fusion_resume_study",
            help=(
                "Reuses the existing SQLite study and runs only remaining trials. "
                "A finished study has already met its budget, so re-running with "
                "this on reproduces the previous answer rather than searching "
                "again."
            ),
        )
        clear_preaggr_cache = st.checkbox(
            "Clear this target's pre-aggregation cache first",
            key="fusion_clear_preaggr",
            help=(
                "Rebuilds the per-(entity, radius) cache from scratch. Needed "
                "only when the underlying metric files changed at the same path."
            ),
        )
    with col_r2:
        run_standalones = st.checkbox(
            "Also optimize each metric on its own (NDVI / Vegetation / Terrain)",
            key="fusion_run_standalones",
            help="Adds three single-metric Optuna studies alongside the combined CGI run.",
        )

    # ── Channel collinearity check (iterative VIF) ──────────────
    # Runs after pre-aggregation, before optimization. Drops channels
    # whose pixel-level values are redundant with the others (high VIF).
    # Dropped channels get pinned to weight 0 in every subsequent trial.
    _default("fusion_check_collinearity", True)
    _default("fusion_vif_threshold", 10.0)
    col_c1, col_c2 = st.columns([2, 1])
    with col_c1:
        check_collinearity = st.checkbox(
            "Check channel collinearity (iterative VIF)",
            key="fusion_check_collinearity",
            help=(
                "Iteratively drop the highest-VIF channel until all remaining "
                "VIFs are below threshold, pinning dropped channels to weight 0."
            ),
        )
    with col_c2:
        vif_threshold_ui = st.number_input(
            "VIF threshold",
            min_value=2.0,
            max_value=100.0,
            step=1.0,
            key="fusion_vif_threshold",
            disabled=not check_collinearity,
            help=(
                "Rule of thumb: VIF > 5 moderate, > 10 severe, > 20 very "
                "severe multicollinearity. 10 is the textbook default."
            ),
        )

    # ── Stability selection ─────────────────────────────────────
    # Tuning is bootstrap stability selection: B random-sampler studies on
    # resamples of the train+val pool, scored on out-of-bag rows. The channel
    # mix is chosen by automated threshold calibration (Bodinier) — the
    # selection size and threshold are calibrated by maximizing the stability
    # score, so there are no manual selection knobs.
    st.markdown("**Stability selection**")
    st.caption(
        "The channel-mix winner is calibrated automatically (selection size "
        "K + threshold π maximize the stability score, with a reported PFER "
        "bound). Set the resampling effort and the PFER cap here."
    )
    _default(
        "fusion_weight_bin_pct",
        int(_cgi_formulas.WEIGHT_BIN_PCT),
        [5, 10, 20, 25, 50],
    )
    _default("fusion_n_bootstraps", 30)
    _default("fusion_n_trials_per_bootstrap", 150)
    col_s0, col_s1, col_s2 = st.columns(3)
    with col_s0:
        weight_bin_pct_ui = st.select_slider(
            "Weight cell size (%)",
            options=[5, 10, 20, 25, 50],
            key="fusion_weight_bin_pct",
            help=(
                "Bin width for the channel-mix weight cells the stability "
                "selection ranks. Wider cells → fewer, coarser cells (a good "
                "region fragments less and each cell collects more trials); "
                "narrower cells → finer resolution but many more cells to "
                "cover. **20 % is the default**: below that, neighbouring "
                "cells are near-identical composites and the vote splits "
                "across them so nothing is ever declared stable. The finer "
                "resolution is recovered by the refinement stage below."
            ),
        )
        _refine_options = [o for o in (5, 10, 20, 25) if o < int(weight_bin_pct_ui)]
        if _refine_options:
            _default(
                "fusion_weight_refine_bin_pct",
                min(_refine_options, key=lambda o: abs(o - _cgi_formulas.WEIGHT_REFINE_BIN_PCT)),
                _refine_options,
            )
            weight_refine_bin_pct_ui = st.select_slider(
                "Refinement cell size (%)",
                options=_refine_options,
                key="fusion_weight_refine_bin_pct",
                help=(
                    "Third selection stage. After the channel mix and the "
                    "spatial scale are settled, the surviving trials are "
                    "re-binned at this width and the best sub-cell by median "
                    "out-of-bag score is kept — so the final weights come from "
                    "trials that agree, not from an average across the whole "
                    "coarse cell."
                ),
            )
        else:
            weight_refine_bin_pct_ui = None
            st.caption("_Refinement needs a cell size above 5 %._")

    with col_s1:
        n_bootstraps_ui = st.number_input(
            "Bootstraps (B)",
            min_value=5,
            max_value=200,
            step=5,
            key="fusion_n_bootstraps",
            help=(
                "Number of ⌊n/2⌋ subsamples of the train+val pool (drawn as "
                "complementary pairs). 20-50 typical; raise for a tighter calibration."
            ),
        )
    with col_s2:
        n_trials_per_bootstrap_ui = st.number_input(
            "Trials per bootstrap",
            min_value=100,
            max_value=800,
            step=10,
            key="fusion_n_trials_per_bootstrap",
            help=(
                "Random-sampler trials inside each bootstrap. Coverage of the "
                "weight cells matters more than depth — see the per-cell density "
                "below."
            ),
        )
    _n_weight_cells = max(
        1, _cgi_formulas.weight_cell_count(cgi_formula, int(weight_bin_pct_ui))
    )
    _avg_trials_per_cell = int(n_trials_per_bootstrap_ui) / _n_weight_cells
    st.caption(
        f"≈ **{_avg_trials_per_cell:.1f} trials per weight cell** on average "
        f"({int(n_trials_per_bootstrap_ui)} trials ÷ {_n_weight_cells} cells at "
        f"{int(weight_bin_pct_ui)}% bins). Aim for ≥ 5 so each cell's OOB "
        "ranking is reproducible across resamples; raise the trial count or the "
        "cell size if this is low."
    )
    _default("fusion_max_pfer", 1.0)
    max_pfer_ui = st.number_input(
        "Max PFER (approx.)",
        min_value=0.0,
        max_value=50.0,
        step=0.5,
        key="fusion_max_pfer",
        help=(
            "Caps the calibrated selection size K and threshold π so the "
            "reported PFER bound K²/((2π−1)·N) stays at or below this value — "
            "tighter values keep the stable set small and the error control "
            "meaningful, looser values let more cells be called stable. "
            "0 = no cap."
        ),
    )
    # ── Internals ───────────────────────────────────────────────
    # Minimum trials per radius sub-cell, and the quantile q_worst reports at.
    min_cell_count_ui = 3
    worst_quantile_ui = 0.10

    # ── Per-pixel CGI scoring (vector targets) ──────────────────
    cgi_grid_spacing_m = 50
    whole_grid_scaling = True
    area_balanced_split = True
    normalize_channels = True
    spatial_split = False
    spatial_block_size_m: float | None = None
    n_spatial_blocks: int | None = None
    spatial_adjust_method = "none"
    spatial_adjust_max_df = 10
    spatial_adjust_eps_m: float | None = None
    if is_vector_target:
        st.markdown("**Per-pixel CGI scoring**")
        _default("fusion_cgi_grid_spacing_m", 50, list(range(10, 510, 10)))
        _default("fusion_whole_grid_scaling", True)
        _default("fusion_normalize_channels", True)
        _default("fusion_area_balanced_split", True)
        _default("fusion_spatial_split", True)
        _default("fusion_spatial_block_size_m", 0)
        cgi_grid_spacing_m = st.select_slider(
            "CGI grid pixel size (m)",
            options=list(range(10, 510, 10)),
            key="fusion_cgi_grid_spacing_m",
            help=(
                "Per-pixel CGI grid spacing. Smaller = higher fidelity and a bigger cache."
            ),
        )
        whole_grid_scaling = st.checkbox(
            "Scale composite map to [0, 1]",
            key="fusion_whole_grid_scaling",
            help="Min-max normalise the optimized greenery map to [0, 1] over the whole grid.",
        )
        normalize_channels = st.checkbox(
            "Normalize channels before fusion",
            key="fusion_normalize_channels",
            help=(
                "Scale each channel to [0, 1] before combining, so CGI weights "
                "are comparable across channels and the comparison with the "
                "standalone studies is on equal footing. Off = legacy raw mix."
            ),
        )
        if is_polygon_target:
            area_balanced_split = st.checkbox(
                "Area-balanced stratified split",
                key="fusion_area_balanced_split",
                help="Balance polygon area (not count) across train / val / test within each quartile.",
            )

        spatial_split = st.checkbox(
            "Spatial block validation",
            key="fusion_spatial_split",
            help=(
                "Hold out whole spatial blocks (and resample blocks during "
                "stability selection) so geographic autocorrelation can't "
                "inflate scores. Test blocks are striped across the full "
                "extent so every region is represented in the held-out test."
            ),
        )
        if spatial_split:
            block_size_ui = st.number_input(
                "Spatial block size (m, 0 = auto)",
                min_value=0,
                step=100,
                key="fusion_spatial_block_size_m",
                help=(
                    "Edge length of each block. 0 sizes blocks from the data "
                    "extent and the catchment radius (a block stays wider than "
                    "the greenery autocorrelation range)."
                ),
            )
            spatial_block_size_m = (
                float(block_size_ui) if block_size_ui and block_size_ui > 0 else None
            )

        # ── Spatial-confounding adjustment ──────────────────
        _spatial_adjust_labels = {
            "none": "Off (control listed covariates only)",
            "ks_aic": "KS-AIC (recommended)",
            "spatial_plus": "Spatial+ (df-Spatial+)",
        }
        _spatial_adjust_keys = ["none", "ks_aic", "spatial_plus"]
        _default("fusion_spatial_adjust_method", "none", _spatial_adjust_keys)
        spatial_adjust_method = st.selectbox(
            "Spatial-confounding adjustment",
            options=_spatial_adjust_keys,
            format_func=lambda k: _spatial_adjust_labels[k],
            key="fusion_spatial_adjust_method",
            help=(
                "Remove unmeasured smooth spatial confounding by adding a "
                "coordinate smooth (per spatial cluster, df chosen by AIC) to the "
                "objective. KS-AIC (Keller & Szpiro) is recommended; Spatial+ "
                "residualizes the greenery exposure on the smooth instead. The "
                "reported CGI association becomes fine-scale (within-cluster) "
                "contrast, so the optimal CGI parameters will shift versus an "
                "unadjusted run."
            ),
        )
        if spatial_adjust_method != "none":
            _default("fusion_spatial_adjust_max_df", 10)
            _default("fusion_spatial_adjust_eps_m", 0)
            with st.expander("Spatial adjustment — advanced", expanded=False):
                spatial_adjust_max_df = int(
                    st.number_input(
                        "Max spatial df per cluster",
                        min_value=1,
                        max_value=50,
                        step=1,
                        key="fusion_spatial_adjust_max_df",
                        help=(
                            "Upper bound on the radial smooth functions per spatial "
                            "cluster; AIC selects the actual count up to this."
                        ),
                    )
                )
                eps_ui = st.number_input(
                    "Cluster gap eps (m, 0 = auto)",
                    min_value=0,
                    step=100,
                    key="fusion_spatial_adjust_eps_m",
                    help=(
                        "Distance above which entities fall into different spatial "
                        "clusters (so the smooth never spans a void). 0 derives it "
                        "from the data via nearest-neighbour connectivity."
                    ),
                )
                spatial_adjust_eps_m = float(eps_ui) if eps_ui and eps_ui > 0 else None

    return {
        "cgi_formula": cgi_formula,
        "covariate_columns": list(covariate_columns or []),
        "covariate_types": dict(covariate_types or {}),
        "moderator_columns": list(moderator_columns or []),
        # ``None`` restores the default: the IQR of the composite itself.
        "exposure_iqr": (
            float(exposure_iqr_ui) if exposure_iqr_ui and exposure_iqr_ui > 0 else None
        ),
        "objective_metric": objective_metric,
        "residualize_method": str(residualize_method),
        "search_scoring_method": str(search_scoring_method),
        "test_size": float(test_size),
        "n_bins": int(n_bins),
        "mixedlm_random_slope": bool(mixedlm_random_slope),
        "mixedlm_time_fixed": bool(mixedlm_time_fixed),
        "association_target": str(association_target),
        "include_wave_fixed_effects": bool(include_wave_fixed_effects),
        "area_id_col": area_id_col,
        "decline_average_exposure": bool(decline_average_exposure),
        "decline_exposure_change": bool(decline_exposure_change),
        "resume_existing_study": bool(resume_existing_study),
        "run_standalones": bool(run_standalones),
        "cgi_grid_spacing_m": int(cgi_grid_spacing_m),
        "whole_grid_scaling": bool(whole_grid_scaling),
        "area_balanced_split": bool(area_balanced_split),
        "normalize_channels": bool(normalize_channels),
        "spatial_adjust_method": str(spatial_adjust_method),
        "spatial_adjust_max_df": int(spatial_adjust_max_df),
        "spatial_adjust_eps_m": spatial_adjust_eps_m,
        "spatial_split": bool(spatial_split),
        "spatial_block_size_m": spatial_block_size_m,
        "n_spatial_blocks": n_spatial_blocks,
        "n_bootstraps": int(n_bootstraps_ui),
        "n_trials_per_bootstrap": int(n_trials_per_bootstrap_ui),
        "weight_bin_pct": int(weight_bin_pct_ui),
        "weight_refine_bin_pct": (
            int(weight_refine_bin_pct_ui)
            if weight_refine_bin_pct_ui is not None
            else None
        ),
        "min_cell_count": int(min_cell_count_ui),
        "worst_quantile": float(worst_quantile_ui),
        "max_pfer": float(max_pfer_ui),
        "check_collinearity": bool(check_collinearity),
        "vif_threshold": float(vif_threshold_ui),
    }


# ────────────────────────────────────────────────────────────────────
# Tab render entry point
# ────────────────────────────────────────────────────────────────────


def _clear_fusion_results_cb() -> None:
    """Unload the loaded results (button callback — never aborts the run)."""
    st.session_state.fusion_engine = None
    st.session_state.fusion_results = None
    st.session_state.fusion_engines_by_target = {}


def _remove_fusion_outcome_cb(idx: int) -> None:
    """Drop one outcome column (button callback — never aborts the run)."""
    cols = st.session_state.get("fusion_outcome_columns") or []
    if 0 <= idx < len(cols):
        cols.pop(idx)


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


def _render_collinearity_report(results_view: dict) -> None:
    """Show the iterative-VIF channel reduction report, if one was produced.

    Reads ``results_view["collinearity_report"]`` and renders:
    * Pairwise Pearson r between channels (3×3 matrix as a heatmap-styled table)
    * VIF before / after each drop iteration
    * Channels kept vs dropped
    """
    report = results_view.get("collinearity_report") or None
    if not report:
        return
    try:
        import pandas as _pd
    except Exception:
        return

    st.divider()
    st.markdown("**Channel collinearity check (iterative VIF)**")

    channels = report.get("channels_in") or ["veg", "terrain", "ndvi"]
    initial = report.get("initial_vifs") or []
    final = report.get("final_vifs") or []
    kept = report.get("kept") or []
    dropped = report.get("dropped") or []
    threshold = float(report.get("vif_threshold", 10.0))
    n_sample = int(report.get("sample_size", 0))

    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("VIF threshold", f"{threshold:.1f}")
    with summary_cols[1]:
        st.metric("Pixels sampled", f"{n_sample:,}")
    with summary_cols[2]:
        st.metric(
            "Channels kept",
            f"{len(kept)}/{len(channels)}",
            delta=(f"dropped: {', '.join(dropped)}" if dropped else "all retained"),
            delta_color="off",
        )

    # Pairwise Pearson r heatmap.
    pearson = report.get("pearson_matrix") or []
    if pearson and len(pearson) == len(channels):
        st.markdown("**Pairwise Pearson r**")
        df_pearson = _pd.DataFrame(pearson, index=channels, columns=channels).round(3)
        st.dataframe(df_pearson, width="stretch")
        st.caption(
            "Pairwise correlation on the sampled CGI grid pixels. "
            "|r| > 0.85 is a common collinearity threshold; "
            "|r| > 0.95 is near-perfect redundancy."
        )

    # VIFs before / after, side by side.
    st.markdown("**VIF: before vs after reduction**")
    rows: list[dict] = []
    final_by_name = {kept[i]: final[i] for i in range(len(kept))} if final else {}
    for ch, vif_in in zip(channels, initial):
        rows.append(
            {
                "Channel": ch,
                "Initial VIF": (
                    round(float(vif_in), 3) if vif_in is not None else float("nan")
                ),
                "Status": "dropped" if ch in dropped else "kept",
                "Final VIF": (
                    round(float(final_by_name[ch]), 3) if ch in final_by_name else "—"
                ),
            }
        )
    st.dataframe(_pd.DataFrame(rows), width="stretch")

    # Per-iteration drop history (only if anything was actually dropped).
    iterations = report.get("iterations") or []
    if iterations:
        with st.expander(
            f"Drop history ({len(iterations)} iteration(s))", expanded=False
        ):
            for i, it in enumerate(iterations, start=1):
                st.write(f"**Iteration {i}** — removed `{it.get('removed', '?')}`")
                before = it.get("vifs_before") or []
                after = it.get("vifs_after") or []
                st.caption(
                    f"VIFs before: {[round(float(v), 2) for v in before]} → "
                    f"after: {[round(float(v), 2) for v in after]}"
                )


_TERM_COLUMN_CONFIG = {
    "coef": st.column_config.NumberColumn("coef", format="%.4f"),
    "std_err": st.column_config.NumberColumn("SE", format="%.4f"),
    "t_stat": st.column_config.NumberColumn("t", format="%.2f"),
    "partial_r2": st.column_config.NumberColumn("partial R²", format="%.4f"),
}

_DECLINE_TERM_LABELS = {
    "overall": "Greenspace × time (overall)",
    "between": "Average exposure × time (between-person)",
    "within": "Exposure change × time (within-person)",
}


def _render_decline_terms(results_view: dict) -> None:
    """Longitudinal exposure–decline terms: does greenspace track the outcome's
    rate of change (overall, and the average / change decomposition)."""
    dt = results_view.get("decline_terms")
    if not dt or not dt.get("terms"):
        return

    st.divider()
    st.markdown("**Greenspace and rate of change over time**")
    st.caption(
        "Greenspace × time slopes on the winning composite. A term ≠ 0 means "
        "greenspace tracks how fast the outcome changes; the sign follows the "
        "outcome's scale. Between-person = higher *average* exposure; "
        "within-person = *increasing* exposure over time."
    )
    rows = []
    for term in dt["terms"]:
        rows.append(
            {
                "term": _DECLINE_TERM_LABELS.get(term["key"], term["key"]),
                "coef": term.get("coef"),
                "std_err": term.get("std_err"),
                "t_stat": term.get("t_stat"),
                "p": term.get("p_display"),
                "direction": term.get("direction"),
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config=_TERM_COLUMN_CONFIG,
    )
    if dt.get("within_estimable") is False:
        st.caption(
            "_Within-person term not estimable — the greenspace exposure does "
            "not vary over time (use per-wave greenery files to enable it)._"
        )
    _render_period_confounding(dt)


def _render_period_confounding(dt: dict) -> None:
    """Whether the greenspace × time terms are measuring exposure or vintage.

    Per-wave greenery layers change between waves for reasons that have nothing
    to do with anyone's neighbourhood — a different satellite, a different
    compositing window. That drift tracks calendar time, so it lands on a
    greenspace × time slope. The placebo replaces the exposure with its per-wave
    mean, keeping the period structure and discarding every spatial difference:
    a real effect collapses, an artefact survives.
    """
    diag = dt.get("period_diagnostic") or {}
    placebo = dt.get("placebo_terms") or []
    r = diag.get("within_time_corr")
    if r is None and not placebo:
        return

    with st.expander("Is this exposure, or exposure vintage?", expanded=bool(diag.get("confounded"))):
        if r is not None:
            st.metric(
                "Within-person corr(exposure change, time)",
                f"{float(r):+.3f}",
                help=(
                    "Near zero means exposure change is unrelated to when a "
                    "person was measured. Far from zero means the exposure is "
                    "largely a function of measurement timing, and the "
                    "greenspace × time terms cannot separate the two."
                ),
            )
        if diag.get("confounded"):
            st.warning(
                "The within-person exposure change is largely explained by "
                "**when** each person was measured, not by where they live. "
                "Treat the within-person term as uninterpretable unless the "
                "per-wave layers are harmonised to a common sensor and season."
            )
        if placebo:
            st.markdown("**Placebo — exposure replaced by its per-wave mean**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "term": _DECLINE_TERM_LABELS.get(x["key"], x["key"]),
                            "coef": x.get("coef"),
                            "t_stat": x.get("t_stat"),
                            "p": x.get("p_display"),
                        }
                        for x in placebo
                    ]
                ),
                width="stretch",
                hide_index=True,
                column_config=_TERM_COLUMN_CONFIG,
            )
            st.caption(
                "This exposure carries no spatial information at all. Any term "
                "here that matches the real one above is measuring the wave, "
                "not the neighbourhood."
            )
        per_wave = diag.get("per_wave_mean_exposure") or {}
        if per_wave:
            st.caption(
                "Mean composite by wave: "
                + " · ".join(f"{k}: {v:.3f}" for k, v in per_wave.items())
            )


def _render_covariate_impact(results_view: dict, metric_name: str) -> None:
    """Show per-covariate effect direction + importance + lift over CGI-only."""
    impact = results_view.get("covariate_impact")
    if not impact:
        return

    st.divider()
    is_mixedlm = impact.get("model") == "mixedlm"
    st.markdown("**Covariate impact (CGI study)**")
    if is_mixedlm:
        st.caption(
            "Two mixed-effects models fit on the full dataset using the "
            "stability-selected params: **Full** = "
            "`target ~ CGI + covariates [+ time] + (RE | entity)`, **CGI-only** = "
            "`target ~ CGI [+ time] + (RE | entity)`. Standard errors and Wald "
            "p-values come from the mixed model, so they account for the "
            "within-entity correlation of the repeated measures. R² is the "
            "Nakagawa marginal R² (fixed-effects variance share); partial R² is "
            "the drop when a covariate is removed from Full."
        )
    else:
        st.caption(
            "Two OLS models fit on the full dataset using the stability-selected "
            "params: **Full** = `target ~ CGI + covariates`, **CGI-only** = "
            "`target ~ CGI`. Coefficients show each covariate's effect direction "
            "and magnitude in the full model; partial R² is the variance only "
            "that covariate explains (drop in R² when it's removed from Full)."
        )

    _r2_label = "Marginal R²" if is_mixedlm else "R²"
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric(f"Full model {_r2_label}", f"{impact.get('r2_full', 0):.4f}")
    with summary_cols[1]:
        st.metric(f"CGI-only {_r2_label}", f"{impact.get('r2_cgi_only', 0):.4f}")
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
    if "p_display" not in df.columns:
        df["p_display"] = df["pvalue"].map(_mes.format_pvalue)
    df = df.rename(columns={"p_display": "p"})
    df = df[
        ["covariate", "direction", "coef", "std_err", "t_stat", "p", "partial_r2"]
    ]
    df = df.sort_values("partial_r2", ascending=False).reset_index(drop=True)
    st.dataframe(
        df, width="stretch", hide_index=True, column_config=_TERM_COLUMN_CONFIG
    )

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


# Longest edge of a decimated preview read. The figure is 4.5 in at 140 dpi, so
# ~630 px on screen; anything beyond this is discarded by the renderer anyway.
_MAP_PREVIEW_MAX_EDGE = 2000
# Above this cell count even a decimated read walks too many blocks to be worth
# blocking the page for.
_MAP_PREVIEW_CELL_LIMIT = 4_000_000_000


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
    _labels = [label for label, _ in composite_options]
    _default("fusion_map_picks", [_labels[0]], _labels, multi=True)
    picks = st.multiselect(
        "Studies to plot",
        options=_labels,
        key="fusion_map_picks",
    )

    if not picks:
        return

    try:
        import rasterio
    except Exception:
        st.warning("rasterio not available; cannot render maps.")
        return

    # Read each composite TIFF + the target geometries once, decimated. A
    # catchment grid over distant clusters can be gigapixel-sized and almost
    # entirely nodata; a full-resolution read would exhaust memory for a preview
    # that is only a few hundred pixels wide on screen.
    composites: list[tuple[str, np.ndarray, Any, Any]] = []
    common_crs = None
    for label in picks:
        path = label_to_path[label]
        try:
            with rasterio.open(path) as src:
                if src.width * src.height > _MAP_PREVIEW_CELL_LIMIT:
                    st.info(
                        f"**{label}** is {src.width:,} × {src.height:,} cells — "
                        "too large to preview here. Download it below and open "
                        "it in QGIS."
                    )
                    continue
                scale = max(
                    1.0,
                    max(src.width, src.height) / _MAP_PREVIEW_MAX_EDGE,
                )
                out_h = max(1, int(src.height / scale))
                out_w = max(1, int(src.width / scale))
                arr = src.read(
                    1,
                    masked=True,
                    out_shape=(out_h, out_w),
                    resampling=rasterio.enums.Resampling.average,
                )
                # The transform has to describe the decimated grid, not the
                # native one, or the extent is wrong.
                composites.append(
                    (label, arr, src.transform * src.transform.scale(scale, scale), src.crs)
                )
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

    # Screen dpi — the composite viewer is an on-screen preview, not an export,
    # so a high-dpi canvas is rebuilt on every results rerun (study switch /
    # map pick) for no visible gain and steady memory growth.
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 4.5 * nrows),
        dpi=140,
        constrained_layout=True,
    )
    flat_axes = np.atleast_1d(axes).ravel().tolist()

    panel_idx = 0
    # ── Target panel ────────────────────────────────────────────
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

    # ── Composite panels (shared colorbar at [0, 1]) ────────────
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


def _direction_badge(direction: object) -> str:
    """Human label for a ``+1`` / ``-1`` greenery↔outcome direction sign."""
    if direction is None:
        return "—"
    try:
        return "↑ positive" if int(direction) > 0 else "↓ negative"
    except (TypeError, ValueError):
        return "—"


def _agg_label(stat: str | None, percentile: int | float | None) -> str:
    """Pretty aggregator label (Mean / Median / Nth percentile)."""
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


def _render_stability_diagnostics(summary: dict, metric_name: str) -> None:
    """Diagnostics panel for one study's bootstrap stability selection.

    Reads the cell-aggregation outputs ``bootstrap_stability_selection``
    records (surfaced on each study's ``stability_summary`` bundle): the
    ranked weight cells, the winning cell's empirical OOB-score
    distribution, and the per-bootstrap leaderboard. Returns silently when
    none are present.
    """
    cell_stats = summary.get("cell_stats") or []
    oob_scores = summary.get("winning_cell_oob_scores") or []
    per_bs = summary.get("per_bootstrap_summary") or []
    if not cell_stats and not oob_scores and not per_bs:
        return

    try:
        import pandas as _pd
        import plotly.express as _px
    except Exception:
        st.info("Plotly + pandas required for stability-selection diagnostics.")
        return

    higher_is_better = bool(summary.get("higher_is_better", True))
    worst_q = float(summary.get("worst_quantile") or 0.10)

    # ── Automated threshold calibration (Bodinier) ──────────────
    if summary.get("selection_threshold") is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "Threshold π*",
                f"{float(summary['selection_threshold']):.2f}",
                help="Calibrated selection-probability threshold (maximizes the stability score).",
            )
        with c2:
            st.metric(
                "Selection size K*",
                str(summary.get("selection_size_k", "—")),
                help="Calibrated number of top cells counted as selected per resample (the sparsity / λ analogue).",
            )
        with c3:
            nss = summary.get("n_stably_selected")
            ncc = summary.get("n_candidate_cells")
            st.metric(
                "Stable cells",
                f"{nss} / {ncc}" if nss is not None and ncc is not None else "—",
                help="Channel-mix cells with selection probability ≥ π* — the calibrated stable set.",
            )
        with c4:
            pfer = summary.get("pfer")
            st.metric(
                "PFER (approx.)",
                f"{float(pfer):.2f}" if pfer is not None else "—",
                help=(
                    "Expected number of falsely-stable cells (Meinshausen–Bühlmann "
                    "bound), rigorous under the ⌊n/2⌋ complementary-pairs "
                    "subsampling used here."
                ),
            )
        sscore = summary.get("stability_score")
        if sscore is not None:
            st.caption(
                "Channel-mix selection is calibrated automatically (Bodinier): K "
                f"and π maximize the stability score ({float(sscore):.1f}). The "
                "winner is the most consistently selected cell in the stable set, "
                "ties broken by the worst-quantile OOB score (`q_worst`)."
            )

        # Degenerate-regime flag: when the PFER bound exceeds the number of
        # stably-selected cells (or K covers most candidates), the "stable set"
        # carries no real error control — selection probability isn't
        # discriminating and q_worst is the effective decision. An empty stable
        # set means the PFER cap admitted no recurring cell at all.
        nss = summary.get("n_stably_selected")
        ncc = summary.get("n_candidate_cells")
        pfer = summary.get("pfer")
        ksel = summary.get("selection_size_k")
        no_stable_set = nss is not None and int(nss) == 0
        degenerate = (
            summary.get("pfer_controlled") is False
            or no_stable_set
            or (pfer is not None and nss and float(pfer) >= max(1.0, float(nss)))
            or (ksel and ncc and float(ksel) > 0.5 * float(ncc))
        )
        if no_stable_set:
            st.warning(
                "No stable channel-mix cell fits under your PFER cap — the cap "
                "was honoured (reported PFER stays within budget), but no cell "
                "recurs across resamples often enough to be called stable at "
                "this error budget. The `q_worst` winner is the trustworthy "
                "signal here; raise **Max PFER** to admit a (less error-"
                "controlled) stable set."
            )
        elif degenerate:
            detail = (
                f" (PFER ≈ {float(pfer):.0f} vs {nss} stable cells)"
                if pfer is not None and nss
                else ""
            )
            st.warning(
                f"Selection not error-controlled here{detail} — the channel "
                "mixes aren't separable, so the `q_worst` winner is the "
                "trustworthy signal, not the stable-set size. Lean on the "
                "held-out test / all-entity effect."
            )

    direction_msg = (
        f"Higher {metric_name} is better — `q_worst` is the {worst_q:.0%} "
        "*lower* quantile of OOB scores in the cell."
        if higher_is_better
        else f"Lower {metric_name} is better — `q_worst` is the "
        f"{1.0 - worst_q:.0%} *upper* quantile of OOB scores in the cell."
    )
    st.caption(direction_msg)

    # ── Top cells ranking ───────────────────────────────────────
    if cell_stats:
        st.markdown("**Top weight cells (ranked by selection probability)**")
        rows: list[dict] = []
        for rank, c in enumerate(cell_stats, start=1):
            row: dict = {"Rank": rank}
            for k, v in (c.get("weights") or {}).items():
                short = k.removeprefix("w_").removesuffix("_weight").upper()
                row[short] = int(v)
            row["Count"] = int(c.get("count", 0))
            row[f"q_worst {metric_name}"] = round(
                float(c.get("q_worst", float("nan"))), 4
            )
            row[f"Median {metric_name}"] = round(
                float(c.get("median", float("nan"))), 4
            )
            row["Selection prob."] = round(
                float(c.get("selection_probability", float("nan"))), 3
            )
            rows.append(row)
        st.dataframe(_pd.DataFrame(rows), width="stretch")
        st.caption(
            "Each row is a 10-percent weight bucket. **Count** = trials "
            "across all bootstraps that landed in this bucket. "
            "**Selection prob.** = fraction of resamples where the cell was in "
            "the calibrated top-K by OOB score; the winner is the cell with the "
            "highest selection probability in the stable set (≥ π*), ties broken "
            "by the worst-quantile OOB score (`q_worst`). When selection "
            "probability saturates (every stable cell at 1.0), `q_worst` is the "
            "effective decision — the most robust cell wins. A tight cluster of "
            "similar runners-up is more credible than an isolated winner."
        )

    # ── Stage-2 radius sub-cells (winning weight cell) ──────────
    radius_stats = summary.get("radius_cell_stats") or []
    if radius_stats:
        bin_m = summary.get("radius_bin_m")
        st.markdown(
            "**Top radius sub-cells** (within the winning weight cell, "
            "ranked by `q_worst`)"
        )
        rrows: list[dict] = []
        for rank, c in enumerate(radius_stats, start=1):
            row = {"Rank": rank}
            for rk, lo in (c.get("radii") or {}).items():
                label = rk.removesuffix("_radius").upper() + " radius (m)"
                try:
                    hi = int(lo) + int(bin_m) if bin_m else None
                    row[label] = f"{int(lo)}–{hi}" if hi else int(lo)
                except (TypeError, ValueError):
                    row[label] = lo
            row["Count"] = int(c.get("count", 0))
            row[f"q_worst {metric_name}"] = round(
                float(c.get("q_worst", float("nan"))), 4
            )
            row[f"Median {metric_name}"] = round(
                float(c.get("median", float("nan"))), 4
            )
            rrows.append(row)
        st.dataframe(_pd.DataFrame(rrows), width="stretch")
        st.caption(
            "Stage 2 of selection: with the channel mix fixed by the winning "
            "weight cell above, the radii are stability-selected the same way. "
            "Each row is a radius bucket"
            + (f" of width {int(bin_m)} m" if bin_m else "")
            + ". The final composite uses the params averaged within the "
            "top radius sub-cell, so the reported radii are a validated "
            "configuration rather than a mean across disagreeing trials."
        )

    # ── Winning cell OOB distribution ───────────────────────────
    if oob_scores and len(oob_scores) >= 3:
        st.markdown("**Winning-cell OOB score distribution**")
        df_oob = _pd.DataFrame({f"OOB {metric_name}": list(oob_scores)})
        fig = _px.histogram(
            df_oob,
            x=f"OOB {metric_name}",
            nbins=min(30, max(5, len(oob_scores) // 3)),
            opacity=0.85,
        )
        q_w = float(summary.get("q_worst", float("nan")) or float("nan"))
        med = float(summary.get("median", float("nan")) or float("nan"))
        if np.isfinite(q_w):
            fig.add_vline(
                x=q_w,
                line_dash="dash",
                line_color="red",
                annotation_text=f"q_worst={q_w:.3f}",
                annotation_position="top left",
            )
        if np.isfinite(med):
            fig.add_vline(
                x=med,
                line_dash="dot",
                line_color="green",
                annotation_text=f"median={med:.3f}",
                annotation_position="top right",
            )
        fig.update_layout(
            height=350,
            margin={"l": 20, "r": 20, "t": 30, "b": 20},
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Each bar counts trials in the winning cell with that OOB "
            f"{metric_name}. The closer `q_worst` sits to `median`, the "
            "tighter the cell's distribution — i.e. the more consistently "
            "the picked weights performed across bootstrap resamples."
        )

    # ── Per-bootstrap leaderboard ───────────────────────────────
    if per_bs:
        st.markdown("**Per-bootstrap leaderboard** (one row per resample)")
        rows = []
        for entry in per_bs:
            row = {
                "Bootstrap": int(entry.get("bootstrap", -1)),
                "Trials": int(entry.get("n_trials", 0)),
                f"Top OOB {metric_name}": round(
                    float(entry.get("top_oob", float("nan"))), 4
                ),
                f"Median OOB {metric_name}": round(
                    float(entry.get("median_oob", float("nan"))), 4
                ),
                "OOB range": (
                    f"[{float(entry.get('min_oob', float('nan'))):.3f}, "
                    f"{float(entry.get('max_oob', float('nan'))):.3f}]"
                ),
            }
            for k, v in (entry.get("top_params") or {}).items():
                short = k.removeprefix("w_").removesuffix("_weight").upper()
                row[short] = int(v) if v is not None else None
            rows.append(row)
        st.dataframe(_pd.DataFrame(rows), width="stretch")
        st.caption(
            "Each row is one bootstrap resample. **Top OOB** is the "
            "highest-scoring trial in that bootstrap; the right-hand columns "
            "show that trial's weights. If one set of weights wins across "
            "many bootstraps, that's strong evidence the winner is stable."
        )

    # ── Per-trial history (stability's Optuna trial log) ────────
    history = summary.get("trial_history") or []
    if history and len(history) >= 5:
        with st.expander(f"Trial history ({len(history)} trials across all resamples)"):
            hist_df = _pd.DataFrame(history)

            # OOB score per resample as a pair of boxplots: one for the
            # resample's calibrated top-K trials (winners), one for the rest
            # (losers). Boxes show each group's spread per resample instead of a
            # cloud of individual dots.
            if {"bootstrap", "oob_score"}.issubset(hist_df.columns):
                if "in_top_k" in hist_df.columns:
                    hist_df["Membership"] = np.where(
                        hist_df["in_top_k"].astype(bool), "Top-K", "Other"
                    )
                    color_arg: dict = {
                        "color": "Membership",
                        "color_discrete_map": {"Top-K": "#2ca02c", "Other": "#9aa0a6"},
                        "category_orders": {"Membership": ["Top-K", "Other"]},
                    }
                else:
                    color_arg = {}
                fig_h = _px.box(
                    hist_df,
                    x="bootstrap",
                    y="oob_score",
                    points=False,
                    labels={
                        "bootstrap": "Resample",
                        "oob_score": f"OOB {metric_name}",
                    },
                    **color_arg,
                )
                fig_h.update_layout(
                    height=360,
                    margin={"l": 20, "r": 20, "t": 30, "b": 20},
                    boxmode="group",
                )
                st.plotly_chart(fig_h, width="stretch")
                st.caption(
                    "Two boxplots per complementary-half resample: trials whose "
                    "channel-mix cell was in that resample's calibrated top-K "
                    "(green = winners) versus the rest (grey = losers). A winner "
                    "box can sit below a loser box in a resample — top-K "
                    "membership ranks cells by their *best* trial, so a winning "
                    "cell's other trials (different radii/aggregators) spread "
                    "lower. The chosen winner is the cell that recurs in the "
                    "top-K across resamples (selection probability), not the one "
                    "scoring highest in any single resample."
                )

            # Parallel coordinates over the channel weights, colored by OOB —
            # where the high-scoring weight combinations concentrate.
            weight_keys = (
                [
                    k
                    for k in (cell_stats[0].get("weights") or {}).keys()
                    if k in hist_df.columns and hist_df[k].notna().any()
                ]
                if cell_stats
                else []
            )
            if len(weight_keys) >= 2 and "oob_score" in hist_df.columns:
                pc_df = hist_df[weight_keys + ["oob_score"]].dropna()
                if len(pc_df) >= 5:
                    label_map = {
                        k: k.removeprefix("w_").removesuffix("_weight").upper()
                        for k in weight_keys
                    }
                    label_map["oob_score"] = f"OOB {metric_name}"
                    fig_pc = _px.parallel_coordinates(
                        pc_df,
                        dimensions=weight_keys + ["oob_score"],
                        color="oob_score",
                        labels=label_map,
                        color_continuous_scale=_px.colors.sequential.Viridis,
                    )
                    fig_pc.update_layout(
                        height=380, margin={"l": 60, "r": 40, "t": 40, "b": 30}
                    )
                    st.plotly_chart(fig_pc, width="stretch")
                    st.caption(
                        "Each line is one trial's channel-weight combination, "
                        "colored by its OOB score. Brighter lines converging on "
                        "the same weight region show where the high-scoring "
                        "mixes concentrate."
                    )

            st.dataframe(hist_df, width="stretch", height=240)
            st.caption(
                "The full per-trial record (resample, OOB score, snapped cell, "
                "and every parameter), kept so the run's exploration is "
                "reproducible and re-scorable without re-running the search."
            )


def _num(x) -> float | None:
    """``float(x)`` when it is a real number, else ``None``.

    Every reporting block can carry ``NaN`` where a model didn't fit; formatting
    one prints the literal "nan" into a metric tile, so non-finite values are
    funnelled to ``None`` and rendered as "—" like any other missing value.
    """
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return None
    return fx if math.isfinite(fx) else None


def _usable_effect(block: dict | None) -> dict | None:
    """An effects block, or ``None`` when it carries no usable numbers.

    A block whose model didn't fit still arrives fully shaped, with ``NaN`` in
    every slot and ``status="fit_failed"``. Treating it as absent lets the
    caller fall back to whichever estimate did work.
    """
    if not block:
        return None
    if str(block.get("status") or "").lower() == "fit_failed":
        return None
    keys = ("score", "lower", "upper", "p_value")
    return block if any(_num(block.get(k)) is not None for k in keys) else None


_P_KIND_LABEL = {"permutation": "permutation p", "wald": "Wald p"}
_P_KIND_HELP = {
    "permutation": (
        "Permutation p-value (Freedman–Lane when covariates are controlled)."
    ),
    "wald": (
        "Wald p-value for the greenery fixed effect from the mixed model fit on "
        "this split. Mixed-effects objectives have no permutation analogue — a "
        "surrogate would have to be resampled and refit per entity, which the "
        "cluster bootstrap already covers."
    ),
}


def _render_results_headline(results_view: dict, metric_name: str) -> None:
    """At-a-glance CGI bottom line: the held-out test greenery effect (headline,
    with the whole-data figure in its tooltip), the held-out significance check,
    direction, and the CGI-vs-standalone verdict as two whole-data (all)
    sub-answers — the paired objective difference and the AIC/BIC
    penalized-model comparison."""
    effects = results_view.get("cgi_effects") or {}
    # A block whose mixed model didn't fit is all-NaN; drop it here so the
    # tiles below fall through to the held-out cluster bootstrap, which is an
    # independent fit of the same effect and often succeeds where it didn't.
    all_eff = _usable_effect(effects.get("all")) or {}
    test_eff = _usable_effect(effects.get("test")) or {}
    test_ci = (results_view.get("test_results") or {}).get("test_ci") or {}
    direction = results_view.get("direction_sign")
    aic_bic = results_view.get("cgi_vs_standalone_aic_bic") or None
    paired = results_view.get("cgi_vs_standalone_paired") or None
    has_standalones = bool(results_view.get("standalones"))
    _f = _num

    # Rows of two so the metric labels have room to breathe — narrow columns
    # clip every label with "…" in a narrow window. The CGI-vs-standalone
    # verdict is two adjacent sub-answers (objective + AIC/BIC) on their own row.
    cols = st.columns(2)
    cols2 = st.columns(2)
    cols3 = st.columns(2)

    # 1) Held-out test effect — the headline (params never saw this split).
    #    The whole-data (all) figure is folded into the tooltip as descriptive,
    #    in-sample context so it isn't read as an independent result.
    fit_failed = test_ci.get("status") == "fit_failed"
    with cols[0]:
        t_score = _f(test_eff.get("score"))
        if t_score is None:
            t_score = _f(test_ci.get("observed"))
        t_lo, t_hi = _f(test_ci.get("lower")), _f(test_ci.get("upper"))
        if t_lo is None and t_hi is None:
            t_lo, t_hi = _f(test_eff.get("lower")), _f(test_eff.get("upper"))
        t_p = _f(test_eff.get("p_value"))
        a_score = _f(all_eff.get("score"))
        a_lo, a_hi = _f(all_eff.get("lower")), _f(all_eff.get("upper"))
        if a_score is not None and a_lo is not None and a_hi is not None:
            all_str = (
                f" Whole-data (all, in-sample): {a_score:.4f} "
                f"[{a_lo:.4f}, {a_hi:.4f}] — optimistic (params were tuned on "
                "this data)."
            )
        elif a_score is not None:
            all_str = f" Whole-data (all, in-sample): {a_score:.4f} — optimistic."
        else:
            all_str = ""
        ci_str = (
            f"[{t_lo:.4f}, {t_hi:.4f}]"
            if t_lo is not None and t_hi is not None
            else "—"
        )
        # A mixed-effects interval comes from a capped, modest replicate count
        # (each one is a full refit), so the count belongs beside the bounds.
        n_boot = test_ci.get("n_boot") or test_eff.get("n_boot")
        if n_boot:
            ci_str += f" from {int(n_boot):,} replicates"
        p_str = (
            f"; {_P_KIND_LABEL.get(test_eff.get('p_kind'), 'p')}={t_p:.3g}"
            if t_p is not None
            else ""
        )
        # |t| is folded at zero, so its interval is not a significance
        # statement. The signed coefficient's interval is the one that can
        # straddle zero.
        s_score = _f(test_eff.get("signed_score"))
        s_lo, s_hi = _f(test_eff.get("signed_lower")), _f(test_eff.get("signed_upper"))
        if s_score is not None and s_lo is not None and s_hi is not None:
            all_str += (
                f" Signed greenery coefficient: {s_score:.4f} "
                f"[{s_lo:.4f}, {s_hi:.4f}] — read this interval, not the |t| one, "
                "for direction and significance."
            )
        if fit_failed:
            st.metric(
                f"Held-out test {metric_name}",
                "fit failed",
                help=(
                    "The held-out mixed model did not converge on the test split, "
                    "so there is no honest effect estimate — this is a non-fit, not "
                    f"a zero effect.{all_str}"
                ),
            )
        elif t_score is not None:
            st.metric(
                f"Held-out test {metric_name}",
                f"{t_score:.4f}",
                help=(
                    "The headline greenery effect, on the untouched test split "
                    f"the params never saw. 95% bootstrap CI: {ci_str}{p_str}.{all_str}"
                ),
            )
        else:
            st.metric(f"Held-out test {metric_name}", "—", help=all_str or None)

    # 2) Held-out significance on the untouched test split.
    with cols[1]:
        p = _f(test_eff.get("p_value"))
        p_kind = test_eff.get("p_kind")
        if p is None:
            p, p_kind = _f(test_ci.get("pvalue")), "wald"
        t_score = _f(test_eff.get("score"))
        if t_score is None:
            t_score = _f(test_ci.get("observed"))
        if fit_failed:
            st.metric(
                "Held-out significance",
                "—",
                help="No significance — the held-out mixed model did not converge.",
            )
        elif p is not None:
            st.metric(
                f"Held-out significance ({_P_KIND_LABEL.get(p_kind, 'p-value')})",
                f"p = {p:.3g}",
                help=(
                    _P_KIND_HELP.get(p_kind, "")
                    + " The honest generalizability check — the params never saw "
                    "this split."
                ),
            )
        elif t_score is not None:
            lo, hi = _f(test_ci.get("lower")), _f(test_ci.get("upper"))
            ci_str = (
                f"[{lo:.4f}, {hi:.4f}]" if lo is not None and hi is not None else "—"
            )
            st.metric(
                "Held-out significance",
                "—",
                help=f"No permutation p-value; 95% bootstrap CI: {ci_str}.",
            )
        else:
            st.metric("Held-out significance", "—")

    # 3) CGI vs best standalone — sub-answer A: whole-data (all) paired
    #    objective difference.
    with cols2[0]:
        if paired:
            diff = _f(paired.get("observed_diff"))
            lo, hi = _f(paired.get("lower")), _f(paired.get("upper"))
            pp = _f(paired.get("p_value"))
            pp_holm = _f(paired.get("p_value_holm"))
            fam_n = paired.get("family_size")
            ch = paired.get("standalone_channel")
            verdict = "CGI better" if paired.get("favors_cgi") else "not better"
            if diff is not None:
                ci_str = (
                    f"[{lo:.4f}, {hi:.4f}]"
                    if lo is not None and hi is not None
                    else "—"
                )
                p_str = f"{pp:.3g}" if pp is not None else "—"
                holm_str = (
                    f" Holm-adjusted p={pp_holm:.3g} across {fam_n} standalone "
                    "comparison(s)."
                    if pp_holm is not None and fam_n
                    else ""
                )
                help_txt = (
                    f"Whole-data (all) paired bootstrap difference in "
                    f"{metric_name} (CGI − `{ch}`): Δ={diff:.4f} {ci_str}, "
                    f"one-sided p={p_str} (positive favours CGI).{holm_str}"
                )
            else:
                help_txt = "Comparison unavailable."
            st.metric("CGI vs standalone — objective (all)", verdict, help=help_txt)
        elif has_standalones:
            # Standalones ran but the paired bootstrap produced nothing — the
            # whole-data model didn't fit for one of the two composites. Say so
            # instead of implying the studies were never enabled.
            st.metric(
                "CGI vs standalone — objective (all)",
                "unavailable",
                help=(
                    "The paired whole-data comparison could not be computed — "
                    "the mixed model did not fit for CGI or for the standalone "
                    "composite on the full dataset. See the job log for which "
                    "channel failed; the AIC/BIC comparison beside this tile is "
                    "the independent second opinion."
                ),
            )
        else:
            st.metric(
                "CGI vs standalone — objective (all)",
                "—",
                help="Enable standalone studies to compare CGI against them.",
            )

    # 4) CGI vs best standalone — sub-answer B: whole-data (all) AIC/BIC
    #    penalized-model comparison.
    with cols2[1]:
        if aic_bic and aic_bic.get("ok"):
            d_bic = _f(aic_bic.get("delta_bic"))
            st.metric(
                "CGI vs standalone — AIC/BIC (all)",
                str(aic_bic.get("verdict", "—")),
                help=(
                    "AIC/BIC of CGI (all channels) vs the best single channel "
                    f"(`{aic_bic.get('best_channel')}`), both fit on the whole "
                    "dataset. "
                    + (
                        f"ΔBIC={d_bic:.1f} (positive favours CGI)."
                        if d_bic is not None
                        else ""
                    )
                ),
            )
        elif aic_bic is not None:
            st.metric(
                "CGI vs standalone — AIC/BIC (all)",
                "inconclusive",
                help=str(aic_bic.get("reason", "Comparison unavailable.")),
            )
        else:
            st.metric(
                "CGI vs standalone — AIC/BIC (all)",
                "—",
                help="Enable standalone studies to compare CGI against them.",
            )

    # 5) Direction of the greenery↔outcome relationship.
    with cols3[0]:
        st.metric(
            "Direction",
            _direction_badge(direction),
            help=(
                "Sign of the greenery↔outcome relationship; reported separately "
                "because distance correlation is unsigned. Positive means the "
                "composite rises with the outcome."
            ),
        )


def _render_study_detail(
    study_view: dict,
    engine,
    formula,
    metric_name: str,
    study_key: str,
    covariates_used: list[str],
) -> None:
    """Full detail for one study (CGI or a standalone channel).

    Renders: test score + CI / direction / n tiles · the winning params
    (weights + radii + aggregators, formula-aware; standalones show their
    single active channel) · per-subset scores (bootstraps / held-out test /
    all) · the final params JSON · and the stability-selection diagnostics.
    """
    averaged_raw = study_view.get("averaged_params") or {}
    best_params = study_view.get("best_params") or {}
    final_params = {
        k: v for k, v in averaged_raw.items() if not str(k).startswith("__")
    } or {k: v for k, v in best_params.items() if not str(k).startswith("__")}
    summary = study_view.get("stability_summary") or {}
    test_res = study_view.get("test_results") or {}
    test_ci = test_res.get("test_ci") or {}
    subset_scores = study_view.get("subset_scores") or {}
    has_covariates = bool(covariates_used)
    is_cgi = study_key == "cgi"

    # ── Tiles: test score + CI · direction · n ──────────────────
    tile_cols = st.columns(3)
    with tile_cols[0]:
        if test_ci.get("status") == "fit_failed":
            st.metric(
                f"Test {metric_name}",
                "fit failed",
                delta="model did not converge",
                delta_color="off",
                help="The held-out mixed model did not converge — a non-fit, "
                "not a zero effect.",
            )
        else:
            obs = _num(test_ci.get("observed"))
            if obs is None:
                obs = _num(test_res.get("test_score"))
            lo, hi = _num(test_ci.get("lower")), _num(test_ci.get("upper"))
            if obs is not None:
                ci_str = (
                    f"95% CI [{lo:.4f}, {hi:.4f}]"
                    if lo is not None and hi is not None
                    else "no CI"
                )
                st.metric(
                    f"Test {metric_name}",
                    f"{obs:.4f}",
                    delta=ci_str,
                    delta_color="off",
                )
            else:
                st.metric(f"Test {metric_name}", "—")
    with tile_cols[1]:
        st.metric("Direction", _direction_badge(study_view.get("direction_sign")))
    with tile_cols[2]:
        test_n = (subset_scores.get("test") or {}).get("n")
        st.metric("Test entities (n)", f"{int(test_n)}" if test_n else "—")

    # ── Winning params ──────────────────────────────────────────
    if is_cgi:
        if formula.name == _cgi_formulas.WEIGHTED_AVERAGE:
            st.markdown("**Weights (stability-selected)**")
            weight_cols = st.columns(len(formula.weight_keys))
            total_weight = sum(
                float(final_params.get(k, 0)) for k in formula.weight_keys
            )
            for col, key in zip(weight_cols, formula.weight_keys):
                ch = key.removesuffix("_weight")
                label = _CHANNEL_DISPLAY.get(ch, ch.upper())
                pct = (
                    100.0 * float(final_params.get(key, 0)) / total_weight
                    if total_weight > 0
                    else 0.0
                )
                with col:
                    st.metric(label, f"{pct:.1f}%")
        else:
            st.markdown("**Weights and powers (stability-selected)**")
            all_keys = list(formula.weight_keys) + list(formula.power_keys)
            groups = [all_keys[i : i + 3] for i in range(0, len(all_keys), 3)]
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

        st.markdown("**Radii (m)**")
        radii_cols = st.columns(3)
        for col, key in zip(
            radii_cols, ("veg_radius", "terrain_radius", "ndvi_radius")
        ):
            ch = key.removesuffix("_radius")
            label = _CHANNEL_DISPLAY.get(ch, ch.upper())
            with col:
                try:
                    v = int(round(float(final_params.get(key, 0))))
                    st.metric(label, f"{v} m")
                except (TypeError, ValueError):
                    st.metric(label, "—")

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
    else:
        # Standalone: a single channel at 100 % — weights are not
        # meaningful, so only the active channel's radius + aggregator are
        # surfaced.
        ch_label = _CHANNEL_DISPLAY.get(study_key, study_key.upper())
        st.caption(f"Single channel — **100 % {ch_label}**.")
        radius_key = f"{study_key}_radius"
        stat_key = "ndvi_stat" if study_key == "ndvi" else "streetview_stat"
        pct_key = "ndvi_percentile" if study_key == "ndvi" else "streetview_percentile"
        detail_cols = st.columns(2)
        with detail_cols[0]:
            try:
                v = int(round(float(final_params.get(radius_key, 0))))
                st.metric(f"{ch_label} radius", f"{v} m")
            except (TypeError, ValueError):
                st.metric(f"{ch_label} radius", "—")
        with detail_cols[1]:
            st.metric(
                f"{ch_label} aggregator",
                _agg_label(final_params.get(stat_key), final_params.get(pct_key)),
            )

    # ── Per-subset scores ───────────────────────────────────────
    if subset_scores:
        # Bootstrap CIs + held-out permutation p-value. The in-pool slice
        # carries the train+val CI only (no permutation p: the params were
        # selected on this pool, so a permutation test there is in-sample and
        # optimistic). Test + all carry both CI and the held-out p.
        effects = study_view.get("cgi_effects") or {}
        _tci = test_res.get("test_ci") or {}
        # A run without a usable effects block — none computed, or the model
        # didn't fit and left an all-NaN one — still has the held-out cluster
        # bootstrap, which carries the same interval and a Wald p.
        _test_fallback = (
            {
                "lower": _tci.get("lower"),
                "upper": _tci.get("upper"),
                "p_value": _tci.get("pvalue"),
                "p_kind": "wald",
            }
            if _tci.get("status") == "ok"
            else None
        )
        eff_for = {
            "train": _usable_effect(effects.get("train_val")),
            "test": _usable_effect(effects.get("test")) or _test_fallback,
            "all": _usable_effect(effects.get("all")),
        }
        p_col = _P_KIND_LABEL.get((eff_for["test"] or {}).get("p_kind"), "p-value")
        # Friendly labels — there is no train→fit→validate step. The displayed
        # slices are the cross-resample bootstrap signal, the untouched held-out
        # test, and the whole dataset.
        subset_label = {
            "val": "Bootstraps",
            "test": "Held-out test",
            "all": "All",
        }

        def _fmt_ci(block: dict | None) -> str | None:
            if not block:
                return None
            lo, hi = _num(block.get("lower")), _num(block.get("upper"))
            if lo is None or hi is None:
                return None
            return f"[{lo:.4f}, {hi:.4f}]"

        st.markdown("**Scores by data subset**")
        rows: list[dict] = []
        for subset in ("val", "test", "all"):
            block = subset_scores.get(subset) or {}
            if not block:
                continue
            score = _num(block.get("score"))
            raw = _num(block.get("score_raw"))
            row = {
                "Subset": subset_label.get(subset, subset),
                metric_name: round(score, 4) if score is not None else None,
            }
            if has_covariates:
                row[f"{metric_name} (raw)"] = round(raw, 4) if raw is not None else None
            eff = eff_for.get(subset)
            row["95% CI"] = _fmt_ci(eff)
            p_val = _num((eff or {}).get("p_value"))
            row[p_col] = f"{p_val:.3g}" if p_val is not None else None
            row["n"] = block.get("n")
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch")
            cap = (
                "Stability selection resamples the full train+val pool into "
                "complementary halves, so there is no train→fit→validate step. "
                "**Bootstraps** = winning-cell median out-of-bag score across "
                "the complementary-half resamples (the cross-resample signal) · "
                "**Held-out test** = untouched test split the params never saw "
                "(the headline) · **All** = every entity (in-sample, descriptive). "
                f"95% CIs are percentile bootstrap; `{p_col}` is reported on the "
                "held-out test only (the params were tuned on the rest, so an "
                "all-slice p-value would double-dip)."
            )
            if has_covariates:
                cap += (
                    " The metric column is covariate-adjusted (partial); "
                    "`(raw)` is the unadjusted correlation."
                )
            if any(r.get(metric_name) is None for r in rows):
                cap += (
                    " A blank score is a slice where the mixed model did not "
                    "fit — a non-fit, not a zero effect."
                )
            st.caption(cap)

    # ── Final params JSON ───────────────────────────────────────
    with st.expander("Final parameters (composite is built from these)"):
        st.json(final_params)

    # ── Stability-selection diagnostics ─────────────────────────
    _render_stability_diagnostics(summary, metric_name)


def _render_cross_study_comparison(results_view: dict, metric_name: str) -> None:
    """CGI vs each standalone: subset-score bars + the AIC/BIC verdict detail."""
    standalones = results_view.get("standalones") or {}
    if not standalones:
        return

    st.divider()
    st.markdown("**CGI vs standalone single-metric studies**")

    studies: list[tuple[str, str, dict]] = [("cgi", "CGI (combined)", results_view)]
    for ch in ("veg", "terrain", "ndvi"):
        b = standalones.get(ch)
        if b:
            studies.append((ch, f"{_CHANNEL_DISPLAY.get(ch, ch)} (standalone)", b))

    # ── Score bars (bootstraps + all) ───────────────────────────
    _subset_label = {
        "val": "Bootstraps",
        "test": "Held-out test",
        "all": "All",
    }
    _default("fusion_compare_subsets", ["val", "all"])
    subset_picks = st.multiselect(
        "Subsets to compare",
        options=["val", "test", "all"],
        format_func=lambda s: _subset_label.get(s, s),
        key="fusion_compare_subsets",
        help="Each study's stability-selected params, scored on each subset.",
    )
    if subset_picks:
        rows: list[dict] = []
        missing: list[str] = []
        for _key, disp, b in studies:
            subs = b.get("subset_scores") or {}
            for subset in subset_picks:
                block = subs.get(subset) or {}
                fval = _num(block.get("score"))
                label = _subset_label.get(subset, subset)
                if fval is None:
                    missing.append(f"{disp} · {label}")
                rows.append({"Study": disp, "Subset": label, metric_name: fval})
        df = pd.DataFrame(rows)
        try:
            import plotly.express as _px

            fig = _px.bar(
                df,
                x="Study",
                y=metric_name,
                color="Subset",
                barmode="group",
                title=f"{metric_name} by study and subset",
            )
            fig.update_layout(margin=dict(l=60, r=20, t=60, b=80))
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.bar_chart(
                df.pivot(index="Study", columns="Subset", values=metric_name),
                width="stretch",
            )
        if missing:
            # A study×subset with no bar is a model that didn't fit, which is
            # not the same statement as a bar sitting at zero.
            st.caption(
                "No bar for " + ", ".join(f"**{m}**" for m in missing) + " — the "
                "mixed model did not fit on that slice, so there is no score to "
                "plot (a non-fit, not a zero effect)."
            )
        with st.expander("Show exact values"):
            st.dataframe(df, width="stretch")

    # ── Paired objective difference (Holm-corrected) ────────────
    fam = results_view.get("cgi_vs_standalone_paired_family") or []
    if fam:
        st.markdown("**Paired objective difference (CGI − standalone)**")
        fam_rows = []
        for d in fam:
            diff = d.get("observed_diff")
            lo, hi = d.get("lower"), d.get("upper")
            pr, ph = d.get("p_value"), d.get("p_value_holm")
            fam_rows.append(
                {
                    "Standalone": _CHANNEL_DISPLAY.get(
                        d.get("standalone_channel"), str(d.get("standalone_channel"))
                    ),
                    f"Δ {metric_name} (CGI − ch)": (
                        round(float(diff), 4) if diff is not None else None
                    ),
                    "95% CI": (
                        f"[{float(lo):.4f}, {float(hi):.4f}]"
                        if lo is not None and hi is not None
                        else "—"
                    ),
                    "p (one-sided)": f"{float(pr):.3g}" if pr is not None else None,
                    "p (Holm)": f"{float(ph):.3g}" if ph is not None else None,
                }
            )
        st.dataframe(pd.DataFrame(fam_rows), width="stretch")
        st.caption(
            "Whole-data paired bootstrap difference of CGI against each "
            "standalone channel (positive Δ favours CGI). `p (Holm)` controls "
            "the family-wise error rate across these comparisons. When several "
            "outcomes are optimised, treat those as a further family and discount "
            "the p-values accordingly."
        )

    # ── AIC/BIC verdict detail ──────────────────────────────────
    aic_bic = results_view.get("cgi_vs_standalone_aic_bic") or None
    if aic_bic and aic_bic.get("ok"):
        st.markdown("**Penalized model comparison (AIC / BIC)**")
        best_ch = _CHANNEL_DISPLAY.get(
            aic_bic.get("best_channel"), str(aic_bic.get("best_channel"))
        )
        st.caption(
            f"Full model = `outcome ~ veg + terrain + ndvi (+ covariates)`; "
            f"reduced model = `outcome ~ {best_ch} (+ covariates)` — the best "
            f"single channel by whole-data (all) score. Both models are fit on "
            f"the whole dataset (all entities). Lower AIC/BIC is better; a "
            f"positive Δ favours CGI. **Verdict: {aic_bic.get('verdict')}** "
            f"(n = {aic_bic.get('n')})."
        )
        comp_rows = [
            {
                "Model": "CGI (full, 3 channels)",
                "AIC": round(float(aic_bic.get("aic_full", float("nan"))), 2),
                "BIC": round(float(aic_bic.get("bic_full", float("nan"))), 2),
            },
            {
                "Model": f"{best_ch} (reduced, 1 channel)",
                "AIC": round(float(aic_bic.get("aic_reduced", float("nan"))), 2),
                "BIC": round(float(aic_bic.get("bic_reduced", float("nan"))), 2),
            },
            {
                "Model": "Δ (reduced − full)",
                "AIC": round(float(aic_bic.get("delta_aic", float("nan"))), 2),
                "BIC": round(float(aic_bic.get("delta_bic", float("nan"))), 2),
            },
        ]
        st.dataframe(pd.DataFrame(comp_rows), width="stretch")
    elif aic_bic is not None:
        st.info(
            "AIC/BIC comparison inconclusive: "
            + str(aic_bic.get("reason", "unavailable."))
        )


def _render_artifact_index(results_view: dict) -> None:
    """List every file the job wrote to disk so the user can find them."""
    artifacts_dir = results_view.get("artifacts_dir")
    if not artifacts_dir or not os.path.isdir(artifacts_dir):
        return

    entries: list[tuple[str, int]] = []
    for root, _dirs, files in os.walk(artifacts_dir):
        for fn in files:
            full = os.path.join(root, fn)
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            rel = os.path.relpath(full, artifacts_dir)
            entries.append((rel, size))
    if not entries:
        return

    st.divider()
    with st.expander(f"📁 Output files on disk ({len(entries)})"):
        st.caption(f"Job folder: `{artifacts_dir}`")

        def _human(n: int) -> str:
            for unit in ("B", "KB", "MB", "GB"):
                if n < 1024 or unit == "GB":
                    return (
                        f"{n:.0f} {unit}" if unit == "B" else f"{n / 1024:.1f} {unit}"
                    )
                n /= 1024
            return f"{n:.0f} B"

        rows = [{"File": rel, "Size": _human(size)} for rel, size in sorted(entries)]
        st.dataframe(
            pd.DataFrame(rows), width="stretch", height=min(420, 60 + 28 * len(rows))
        )


def _render_fusion_results_body(output_dir: str) -> None:
    """Inner body of the results panel — kept separate so the styled
    container above stays readable.

    Layout: a CGI headline (test score + CI, direction, AIC/BIC verdict),
    a study selector (CGI + each standalone), the selected study's full
    detail, then cross-study comparison, covariate impact, collinearity,
    composite maps, mixed-effects metrics, and an on-disk artifact index.
    """
    head_col, clear_col = st.columns([4, 1])
    with head_col:
        st.subheader("Optimization Results")
    with clear_col:
        st.button(
            "Clear",
            key="fusion_results_clear",
            width="stretch",
            help="Unload these results from the panel (does not delete files).",
            on_click=_clear_fusion_results_cb,
        )

    results = st.session_state.fusion_results
    if results.get("mode") == "multi":
        st.selectbox(
            "Select outcome",
            options=results["ordered_labels"],
            key="fusion_results_outcome_pick",
        )

    results_view, engine = _fusion_resolve_active_bundle()
    if results_view is None:
        st.warning("Optimization details are not available for the selected outcome.")
        return
    # ``engine`` may be ``None`` when results were rehydrated from disk after a
    # restart; every panel below reads ``results_view`` and only the composite
    # map viewer touches the engine (which it guards).

    metric_name = results_view["objective_metric"].upper()

    # Read run details from the persisted bundle first so they survive a disk
    # reload (when the live ``engine`` is gone); fall back to the engine.
    formula_name = results_view.get("cgi_formula") or getattr(
        engine, "cgi_formula", "weighted_average"
    )
    try:
        formula = _cgi_formulas.get_formula(formula_name)
    except ValueError:
        formula = _cgi_formulas.get_formula("weighted_average")
    covariates_used = list(
        results_view.get("covariate_columns")
        or getattr(engine, "_covariate_columns_user", None)
        or getattr(engine, "covariate_columns", [])
        or []
    )
    cov_types = (
        results_view.get("covariate_types")
        or getattr(engine, "covariate_types", {})
        or {}
    )
    target_name = results_view.get("target_display_name") or getattr(
        engine, "target_file", None
    )
    outcome_name = results_view.get("outcome_label") or results_view.get(
        "target_feature"
    )
    _FORMULA_DISPLAY = {"weighted_average": "Weighted Average", "synergy": "Synergy"}

    def _cov_chip(c: str) -> str:
        kind = "categorical" if str(cov_types.get(c)).lower() == "categorical" else None
        return f"`{c}`" + (f" _({kind})_" if kind else "")

    # ── Headline ────────────────────────────────────────────────
    _render_results_headline(results_view, metric_name)
    if target_name or outcome_name:
        bits = []
        if target_name:
            bits.append(f"**Target:** `{os.path.basename(str(target_name))}`")
        if outcome_name:
            bits.append(f"**Outcome:** `{outcome_name}`")
        st.caption("  ·  ".join(bits))
    st.caption(
        "**Formula:** "
        f"{_FORMULA_DISPLAY.get(formula.name, formula.name.title())}  ·  "
        "**Covariates:** "
        + (
            ", ".join(_cov_chip(c) for c in covariates_used)
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
    st.divider()

    # ── Study selector + per-study detail ───────────────────────
    standalones = results_view.get("standalones") or {}
    study_options: list[tuple[str, str]] = [("cgi", "CGI (combined)")]
    for ch in ("veg", "terrain", "ndvi"):
        if standalones.get(ch):
            study_options.append((ch, f"{_CHANNEL_DISPLAY.get(ch, ch)} (standalone)"))

    if len(study_options) > 1:
        picked_key = st.radio(
            "Study detail",
            options=[k for k, _ in study_options],
            format_func=lambda k: dict(study_options)[k],
            horizontal=True,
            key="fusion_study_detail_pick",
        )
    else:
        picked_key = "cgi"

    study_view = results_view if picked_key == "cgi" else standalones.get(picked_key)
    if study_view is not None:
        _render_study_detail(
            study_view, engine, formula, metric_name, picked_key, covariates_used
        )

    st.divider()
    st.caption(
        "_The sections below are fixed at the **CGI study** and **cross-study** "
        "scope — they don't change with the study selector above (which only "
        "switches the per-study detail)._"
    )

    # ── Cross-study comparison (CGI vs standalones) ─────────────
    _render_cross_study_comparison(results_view, metric_name)

    # ── Channel collinearity report ─────────────────────────────
    _render_collinearity_report(results_view)

    # ── Covariate impact panel ──────────────────────────────────
    _render_covariate_impact(results_view, metric_name)

    # ── Longitudinal exposure–decline terms ─────────────────────
    _render_decline_terms(results_view)

    # ── Composite map viewer ────────────────────────────────────
    _render_composite_map_viewer(results_view, engine)

    # ── Mixed-effects post-hoc metrics CSV viewer ───────────────
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

        # Filenames are ``mixedlm_metrics[__<outcome>][__<channel>].csv``: the
        # bare name is the CGI study, a channel suffix is that standalone's, and
        # a multi-outcome run puts the outcome label first. Splitting on "__"
        # recovers which study a file belongs to instead of pattern-replacing
        # the prefix, which labelled every standalone "CGI <channel>".
        def _study_of(fname: str) -> tuple[str, str] | None:
            parts = fname[: -len(".csv")].split("__")
            if parts[0] != "mixedlm_metrics":
                return None
            rest = parts[1:]
            channel = rest[-1] if rest and rest[-1] in _CHANNEL_DISPLAY else None
            outcome = "__".join(rest[: -1 if channel else None]) or None
            if _active_label is not None and outcome != _active_label:
                return None
            if _active_label is None and outcome is not None:
                return None
            if channel is None:
                return "cgi", "CGI (combined)"
            return channel, f"{_CHANNEL_DISPLAY[channel]} (standalone)"

        _csv_entries: list[tuple[str, str, str]] = []
        if os.path.isdir(_study_root):
            for _fn in sorted(os.listdir(_study_root)):
                if not _fn.startswith("mixedlm_metrics") or not _fn.endswith(".csv"):
                    continue
                _study = _study_of(_fn)
                if _study is None:
                    continue
                _csv_entries.append(
                    (_study[0], _study[1], os.path.join(_study_root, _fn))
                )

        # CGI first, then the channels in their canonical order.
        _order = {"cgi": 0, "veg": 1, "terrain": 2, "ndvi": 3}
        _csv_entries.sort(key=lambda e: _order.get(e[0], 99))

        if _csv_entries:
            st.divider()
            st.markdown("**Mixed-effects: all metrics across trial pools**")
            st.caption(
                "One tab per study — the combined CGI composite and each "
                "standalone single-channel study. Each pool's per-trial rows are "
                "followed by `__mean__`, `__ci_lo__`, `__ci_hi__`, and `__n__` "
                "summary rows."
            )

            _tabs = st.tabs([disp for _, disp, _ in _csv_entries])
            for _tab, (_, _, _path) in zip(_tabs, _csv_entries):
                with _tab:
                    st.caption(f"Source: `{_path}`")
                    st.dataframe(pd.read_csv(_path), width="stretch")
    except Exception as _exc:  # pragma: no cover -- UI-only guard
        st.warning(f"Could not read mixedlm_metrics CSVs: {_exc}")

    # ── On-disk artifact index ──────────────────────────────────
    _render_artifact_index(results_view)


@st.fragment
def _render_target_preview(
    target_picked_path: str | None,
    tmp_target_path: str | None,
    is_vector_target: bool,
    is_raster_target: bool,
    preview_vector_gdf: "gpd.GeoDataFrame | None",
) -> None:
    """Target preview map, isolated in a fragment.

    Its own controls (value column / preview band) rerun only this fragment, and
    with the cached target read the whole preview stays cheap on unrelated edits.
    """
    if not (target_picked_path and tmp_target_path):
        st.info("Upload a target file to preview")
        return

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
            preview_feature = None if preview_pick == geom_only else preview_pick

            if preview_feature and preview_feature in preview_gdf.columns:
                vals = preview_gdf[preview_feature].dropna()
                if len(vals) > 0:
                    if not _add_outcome_geometry_preview(
                        m_fusion_preview, preview_gdf, preview_feature
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
            # Starts on the outcome band, then follows its own selection.
            # Clamped in case a later raster has fewer bands.
            _n_bands = max(1, n_bands_preview)
            _default(
                "fusion_preview_raster_band",
                min(int(st.session_state.get("fusion_target_band", 1)), _n_bands),
            )
            st.session_state["fusion_preview_raster_band"] = min(
                int(st.session_state["fusion_preview_raster_band"]), _n_bands
            )
            preview_band = st.number_input(
                "Preview band",
                min_value=1,
                max_value=_n_bands,
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

                bounds_4326 = transform_bounds(src_crs, "EPSG:4326", *bounds_native)

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


def render(output_dir: str) -> None:
    st.header("Metric Fusion & Optimization")

    MetricFusionEngine = _MetricFusionEngine

    if MetricFusionEngine is None:
        st.error("MetricFusionEngine module not found in geofuse/fusion.py")
        return

    # A re-run confirmed on a job card in the sidebar monitor lands here:
    # seed the form before any of its widgets instantiate.
    from services import get_job_store

    _consume_fusion_restart(get_job_store())

    if "fusion_engine" not in st.session_state:
        st.session_state.fusion_engine = None
    if "fusion_results" not in st.session_state:
        st.session_state.fusion_results = None
    if "fusion_outcome_columns" not in st.session_state:
        st.session_state.fusion_outcome_columns = []
    if "fusion_engines_by_target" not in st.session_state:
        st.session_state.fusion_engines_by_target = {}

    # ── Loaded-results overview ─────────────────────────────────
    # Rendered first so it survives the configuration early-returns below.
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
                            _keep_valid("fusion_target_gpkg_layer", layers)
                            target_layer_for_engine = st.selectbox(
                                "Target layer",
                                options=layers,
                                key="fusion_target_gpkg_layer",
                                help="GeoPackage or archive layer containing geometries and attributes.",
                            )
                        elif len(layers) == 1:
                            target_layer_for_engine = layers[0]

                        rv_layer = None
                        if target_layer_for_engine is not None and suf in (
                            ".gpkg",
                            ".zip",
                        ):
                            rv_layer = target_layer_for_engine
                        preview_vector_gdf = _read_vector_for_ui(
                            tmp_target_path, layer=rv_layer
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
                                    st.button(
                                        "❌",
                                        key=f"fusion_outcome_remove_{i}",
                                        help="Remove this outcome column",
                                        on_click=_remove_fusion_outcome_cb,
                                        args=(i,),
                                    )

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
                            _default("fusion_multi_objective_run", False)
                            multi_objective_requested = st.checkbox(
                                "Multi-objective optimization run",
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
                        _default("fusion_target_band", 1)
                        st.session_state["fusion_target_band"] = min(
                            int(st.session_state["fusion_target_band"]), n_bands
                        )
                        target_band = st.number_input(
                            "Outcome band",
                            min_value=1,
                            max_value=n_bands,
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
        _render_target_preview(
            target_picked_path,
            tmp_target_path,
            bool(is_vector_target),
            bool(is_raster_target),
            preview_vector_gdf if is_vector_target else None,
        )

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

    # ── Metric-download defaults ────────────────────────────────
    ndvi_auto_start = date(2023, 6, 1)
    ndvi_auto_end = date(2023, 9, 30)
    cache_metrics = False
    ndvi_resolution_m = None
    gvi_grid_spacing_m = None

    # =========================================================================
    # SECTION 4 — Study details (form)
    # =========================================================================
    st.divider()

    # Attribute columns the user can pick as covariates — numeric columns are
    # used as-is; categorical (object / category / bool, or numeric-coded ones
    # the user tags) are one-hot encoded by the engine. ``categorical_candidates``
    # pre-marks the genuinely non-numeric columns.
    available_covariates: list[str] = []
    categorical_candidates: list[str] = []
    if is_vector_target and preview_vector_gdf is not None:
        outcome_set = set(target_outcome_columns)
        try:
            geom_name = preview_vector_gdf.geometry.name
        except Exception:
            geom_name = "geometry"

        def _num_cat(frame) -> tuple[set[str], set[str]]:
            num = set(frame.select_dtypes(include=[np.number]).columns)
            cat = set(
                frame.select_dtypes(include=["object", "category", "bool"]).columns
            )
            return num, cat

        numeric_set, categorical_set = _num_cat(preview_vector_gdf)
        wide_files = opt_state.get("wide_files") or []
        if is_longitudinal and opt_state.get("intake_mode") == "wide" and wide_files:
            common_num: set[str] | None = None
            common_cat: set[str] | None = None
            for wf in wide_files:
                try:
                    _frame = _read_vector_head(wf["path"], _file_sig(wf["path"]))
                except Exception:
                    continue
                _n, _c = _num_cat(_frame)
                common_num = _n if common_num is None else common_num & _n
                common_cat = _c if common_cat is None else common_cat & _c
            if common_num is not None:
                numeric_set = (common_num & numeric_set) or common_num
            if common_cat is not None:
                categorical_set = (common_cat & categorical_set) or common_cat

        def _ok(c: str) -> bool:
            return c not in outcome_set and c != geom_name

        numeric_cols = sorted(c for c in numeric_set if _ok(c))
        categorical_cols = sorted(c for c in categorical_set if _ok(c))
        available_covariates = sorted(set(numeric_cols) | set(categorical_cols))
        categorical_candidates = categorical_cols

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
            is_vector_target=bool(is_vector_target),
            categorical_candidates=categorical_candidates,
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
    covariate_types = study_state.get("covariate_types") or {}
    moderator_columns_param = list(study_state.get("moderator_columns") or [])
    exposure_iqr_param = study_state.get("exposure_iqr")
    objective_metric = study_state["objective_metric"]
    test_size = study_state["test_size"]
    n_bins = study_state["n_bins"]
    resume_existing_study = study_state["resume_existing_study"]
    run_standalones = study_state["run_standalones"]
    lon_random_slope = study_state["mixedlm_random_slope"]
    lon_include_time_fixed = study_state["mixedlm_time_fixed"]
    lon_decline_between = study_state.get("decline_average_exposure", False)
    lon_decline_within = study_state.get("decline_exposure_change", False)
    lon_association_target = study_state.get("association_target", "level")
    lon_wave_fe = study_state.get("include_wave_fixed_effects", True)
    lon_area_col = study_state.get("area_id_col") or None
    cgi_grid_spacing_m_param = study_state.get("cgi_grid_spacing_m")
    whole_grid_scaling_param = bool(study_state.get("whole_grid_scaling", False))
    area_balanced_split_param = bool(study_state.get("area_balanced_split", False))
    normalize_channels_param = bool(study_state.get("normalize_channels", False))
    residualize_method_param = str(study_state.get("residualize_method", "linear"))
    search_scoring_method_param = str(
        study_state.get("search_scoring_method", "mom_em3")
    )
    spatial_adjust_method_param = str(study_state.get("spatial_adjust_method", "none"))
    spatial_adjust_max_df_param = int(study_state.get("spatial_adjust_max_df", 10))
    spatial_adjust_eps_m_param = study_state.get("spatial_adjust_eps_m")
    spatial_split_param = bool(study_state.get("spatial_split", False))
    spatial_block_size_m_param = study_state.get("spatial_block_size_m")
    n_spatial_blocks_param = study_state.get("n_spatial_blocks")
    n_bootstraps_param = int(study_state.get("n_bootstraps", 30))
    n_trials_per_bootstrap_param = int(study_state.get("n_trials_per_bootstrap", 150))
    weight_bin_pct_param = int(
        study_state.get("weight_bin_pct", _cgi_formulas.WEIGHT_BIN_PCT)
    )
    weight_refine_bin_pct_param = study_state.get("weight_refine_bin_pct")
    min_cell_count_param = int(study_state.get("min_cell_count", 3))
    worst_quantile_param = float(study_state.get("worst_quantile", 0.10))
    max_pfer_param = float(study_state.get("max_pfer", 1.0))
    check_collinearity_param = bool(study_state.get("check_collinearity", False))
    vif_threshold_param = float(study_state.get("vif_threshold", 10.0))

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
            st.button(
                "🔄 Reset",
                width="stretch",
                key="fusion_reset",
                on_click=_clear_fusion_results_cb,
            )

    if fusion_run_clicked:
        # ── Basic validation ────────────────────────────────
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
            # ── Build LongitudinalSpec (or None) ────────
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
                        # Intake uses the canonical pair; keeping each file's
                        # own choice lets the setup form be restored exactly.
                        target_columns_per_wave = {
                            wf["wave_label"]: {
                                "entity_col": wf["entity_col"],
                                "date_col": wf["date_col"],
                            }
                            for wf in wide_files
                        }
                        entity_id_col = canonical_entity
                        date_col_eff = canonical_date
                    else:
                        target_files_per_wave = {}
                        target_columns_per_wave = {}
                        entity_id_col = ""
                        date_col_eff = ""
                    wave_col_eff: str | None = None
                else:
                    target_files_per_wave = {}
                    entity_id_col = opt_state["entity_id_col"] or ""
                    date_col_eff = opt_state["date_col"] or ""
                    wave_col_eff = opt_state["wave_col"]

                # Year-keyed assignment: wave labels are the calendar years and
                # each row's wave comes from its own measurement date, so no
                # wave column is consulted.
                assign_by_year = bool(opt_state.get("assign_by_year"))
                if assign_by_year:
                    wave_col_eff = None

                spec = _LonSpec(
                    intake_mode=intake,  # type: ignore[arg-type]
                    entity_id_col=entity_id_col,
                    wave_labels=wave_labels,
                    wave_col=wave_col_eff,
                    date_col=date_col_eff or "measurement_date",
                    greenery_files=greenery_files,
                    target_files_per_wave=target_files_per_wave,
                    target_columns_per_wave=target_columns_per_wave,
                    scoring_metric=objective_metric,
                    include_time_fixed_effect=lon_include_time_fixed,
                    random_slope_time=lon_random_slope,
                    association_target=str(lon_association_target),
                    include_wave_fixed_effects=bool(lon_wave_fe),
                    area_id_col=lon_area_col,
                    decline_average_exposure=bool(lon_decline_between),
                    decline_exposure_change=bool(lon_decline_within),
                    derive_wave_from_date=assign_by_year,
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

            # ── Configuration summary ───────────────────
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
                    f"**Stability selection:** {n_bootstraps_param} bootstraps × "
                    f"{n_trials_per_bootstrap_param} trials, "
                    f"{weight_bin_pct_param}% weight cells, "
                    f"{test_size*100:.0f}% test set · max PFER "
                    f"{'off' if max_pfer_param <= 0 else f'{max_pfer_param:g}'}"
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

            # Canonical run configuration — every ``run_fusion`` setting that
            # isn't a file path or runtime object. Recorded verbatim and seeded
            # back into the form on re-run (see ``_FUSION_RUN_CONFIG_KEYS`` /
            # ``_seed_fusion_form``) so a re-run can never silently fall back
            # to a default for a forgotten setting.
            run_config = {
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
                "objective_metric": objective_metric,
                "residualize_method": residualize_method_param,
                "search_scoring_method": search_scoring_method_param,
                "ndvi_start_date": ndvi_auto_start.isoformat(),
                "ndvi_end_date": ndvi_auto_end.isoformat(),
                "ndvi_project_id": None,
                "multi_objective_requested": fusion_multi_objective,
                "target_display_name": target_display_name,
                "resume_existing_study": resume_existing_study,
                "cgi_formula": cgi_formula,
                "covariate_columns": list(covariate_columns or []),
                "covariate_types": dict(covariate_types or {}),
                "moderator_columns": list(moderator_columns_param),
                "exposure_iqr": (
                    float(exposure_iqr_param) if exposure_iqr_param else None
                ),
                "standalone_channels": (
                    ["veg", "terrain", "ndvi"] if run_standalones else []
                ),
                "longitudinal_spec_payload": longitudinal_spec_payload,
                "cgi_grid_spacing_m": (
                    int(cgi_grid_spacing_m_param)
                    if cgi_grid_spacing_m_param is not None and is_vector_target
                    else None
                ),
                # Composite [0, 1] scaling: user-chosen for vector targets;
                # raster targets keep the long-standing normalized output.
                "whole_grid_scaling": (
                    bool(whole_grid_scaling_param) if is_vector_target else True
                ),
                "area_balanced_split": (
                    area_balanced_split_param if is_polygon_target_ui else False
                ),
                # Per-channel normalization applies to vector targets only.
                "normalize_channels": (
                    bool(normalize_channels_param) if is_vector_target else False
                ),
                # Spatial-confounding adjustment applies to vector targets only.
                "spatial_adjust_method": (
                    spatial_adjust_method_param if is_vector_target else "none"
                ),
                "spatial_adjust_max_df": int(spatial_adjust_max_df_param),
                "spatial_adjust_eps_m": (
                    spatial_adjust_eps_m_param if is_vector_target else None
                ),
                # Spatial block validation applies to vector targets only;
                # raster targets keep the row-level stratified split.
                "spatial_split": (
                    bool(spatial_split_param) if is_vector_target else False
                ),
                "spatial_block_size_m": (
                    spatial_block_size_m_param if is_vector_target else None
                ),
                "n_spatial_blocks": (
                    n_spatial_blocks_param if is_vector_target else None
                ),
                "n_bootstraps": int(n_bootstraps_param),
                "n_trials_per_bootstrap": int(n_trials_per_bootstrap_param),
                "weight_bin_pct": int(weight_bin_pct_param),
                "weight_refine_bin_pct": weight_refine_bin_pct_param,
                "min_cell_count": int(min_cell_count_param),
                "worst_quantile": float(worst_quantile_param),
                "max_pfer": float(max_pfer_param),
                "check_collinearity": bool(check_collinearity_param),
                "vif_threshold": float(vif_threshold_param),
            }
            _missing = [k for k in _FUSION_RUN_CONFIG_KEYS if k not in run_config]
            if _missing:
                raise RuntimeError(
                    f"run_config is missing keys {_missing}; refusing to submit a "
                    "job whose settings wouldn't round-trip on restart."
                )

            if st.session_state.get("fusion_clear_preaggr"):
                stale = _fusion_preaggr_cache_files(
                    {"target_path": tmp_target_path, **run_config}, output_dir
                )
                if stale:
                    n, freed = _purge_fusion_preaggr_cache(stale)
                    st.info(f"Cleared {n} cache file(s), freed {freed / 1024**2:.0f} MB.")

            fusion_record = store.submit(
                type="fusion",
                name=os.path.splitext(target_display_name)[0],
                params={
                    # Run settings — replayed verbatim on restart.
                    **run_config,
                    # File references + UI metadata — re-resolved on restart.
                    "is_vector_target": is_vector_target,
                    "outcome_columns": list(target_outcome_columns),
                    "target_band": job_target_band,
                    "target_layer": target_layer_for_engine,
                    "geometry_sha256": target_geom_sha,
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
            executor.submit_fusion_subprocess(
                fusion_record,
                dict(
                    # File / runtime args (re-resolved each run); the rest of the
                    # settings ride in verbatim via ``**run_config``.
                    target_path=tmp_target_path,
                    target_features_geojson=(
                        tuple(target_outcome_columns) if is_vector_target else ()
                    ),
                    target_band=job_target_band if is_raster_target else 1,
                    target_layer=(
                        target_layer_for_engine if is_vector_target else None
                    ),
                    target_cleanup_dir=(
                        target_mat.cleanup_dir if target_mat else None
                    ),
                    target_cleanup_file=(
                        target_mat.cleanup_file if target_mat else None
                    ),
                    veg_path=veg_path,
                    ndvi_path=ndvi_path,
                    output_dir=output_dir,
                    **run_config,
                ),
            )

            st.success("✅ Fusion job started! Check sidebar for progress.")
    # Results load on demand, from the **Load results** button on a job card.
