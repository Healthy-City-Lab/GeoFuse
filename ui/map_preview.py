"""Folium preview helpers: pick GeoJson vs clustered markers by geometry type and size."""

from __future__ import annotations

import html
import json
from collections.abc import Callable, Mapping
from typing import Any

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.api.types import is_scalar
from folium.plugins import FastMarkerCluster
from shapely.geometry import mapping as shapely_mapping

from helpers import apply_buffer_m

# Above this many point features, use Leaflet marker clustering (browser performance).
MAP_CLUSTER_POINT_THRESHOLD = 500
# Hard cap on points drawn (sample); caller may surface a warning.
# Large layers use FastMarkerCluster (compact JSON); keep a ceiling for very heavy payloads.
MAP_POINT_DISPLAY_MAX = 75_000
# Include per-point tooltips in fast-cluster data only below this count (tooltip strings inflate JSON).
MAP_FAST_CLUSTER_TOOLTIP_MAX = 8_000

_STUDY_OUTLINE_STYLE: Callable[[dict], Mapping[str, Any]] = lambda x: {
    "color": "#1a73e8",
    "weight": 2,
    "fill": False,
}
_BUFFER_STYLE: Callable[[dict], Mapping[str, Any]] = lambda x: {
    "color": "#f4910c",
    "weight": 2,
    "fillOpacity": 0.07,
    "dashArray": "6 4",
}


def split_points_and_nonpoints(
    gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return (non-point features, point-only rows with MultiPoint exploded to Points)."""
    if gdf.empty:
        return gdf.copy(), gdf.copy()
    types = gdf.geometry.geom_type
    is_pt = types.isin(["Point", "MultiPoint"])
    nonpt = gdf.loc[~is_pt].copy()
    pt_src = gdf.loc[is_pt]
    if pt_src.empty:
        return nonpt, gdf.iloc[:0].copy()
    if not (pt_src.geometry.geom_type == "MultiPoint").any():
        return nonpt, pt_src.copy()
    exploded: list[pd.Series] = []
    for _, row in pt_src.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Point":
            exploded.append(row)
        elif geom.geom_type == "MultiPoint":
            for p in geom.geoms:
                nr = row.copy()
                nr.geometry = p
                exploded.append(nr)
    if not exploded:
        return nonpt, gdf.iloc[:0].copy()
    pt_exp = gpd.GeoDataFrame(exploded, crs=gdf.crs)
    return nonpt, pt_exp


def trim_point_gdf_for_display(
    gdf: gpd.GeoDataFrame, *, max_points: int = MAP_POINT_DISPLAY_MAX
) -> tuple[gpd.GeoDataFrame, str | None]:
    """Random-sample when over ``max_points``; return optional warning label."""
    if len(gdf) <= max_points:
        return gdf, None
    msg = f"Displaying a random sample of {max_points:,} points (of {len(gdf):,})."
    return gdf.sample(max_points, random_state=0), msg


def _fast_cluster_uniform_callback(
    *,
    radius: int,
    color: str,
    fill_color: str,
    fill_opacity: float,
    bind_tooltip_from_row: bool,
) -> str:
    """Javascript callback body for FastMarkerCluster (circle markers, optional row[2] tooltip)."""
    return (
        "function (row) {"
        "var marker = L.circleMarker(new L.LatLng(row[0], row[1]), {"
        f"radius: {int(radius)}, "
        f"color: {json.dumps(color)}, "
        f"fillColor: {json.dumps(fill_color)}, "
        f"fillOpacity: {float(fill_opacity)}, "
        "weight: 1});"
        + (
            "if (row.length > 2 && row[2] != null && row[2] !== '') { "
            "marker.bindTooltip(String(row[2]), {sticky: true, direction: 'top'}); }"
            if bind_tooltip_from_row
            else ""
        )
        + "return marker; }"
    )


def _point_coords_and_optional_tooltips(
    gdf_points: gpd.GeoDataFrame,
    tooltip_fields: list[str],
    tooltip_aliases: list[str],
    *,
    include_tooltips: bool,
) -> list[list[Any]]:
    """Build FastMarkerCluster ``data`` rows: [lat, lon] or [lat, lon, tooltip_html]."""
    ys = gdf_points.geometry.y.to_numpy(dtype=float, copy=False)
    xs = gdf_points.geometry.x.to_numpy(dtype=float, copy=False)
    mask = np.isfinite(ys) & np.isfinite(xs)
    base = gdf_points.iloc[mask].reset_index(drop=True)
    ys = ys[mask]
    xs = xs[mask]
    if len(base) == 0:
        return []
    if not include_tooltips or not tooltip_fields:
        return np.column_stack([ys, xs]).tolist()
    fields = [f for f in tooltip_fields if f in base.columns]
    if not fields:
        return np.column_stack([ys, xs]).tolist()
    alias_map = dict(zip(tooltip_fields, tooltip_aliases or tooltip_fields))
    tips: list[str] = []
    field_aliases = [alias_map.get(f, f) for f in fields]
    for _, row in base.iterrows():
        tips.append(_row_tooltip_html(row, fields, field_aliases))
    return [[float(y), float(x), t] for y, x, t in zip(ys, xs, tips)]


def _row_tooltip_html(row: pd.Series, fields: list[str], aliases: list[str]) -> str:
    parts: list[str] = []
    for f, alias in zip(fields, aliases):
        if f not in row.index:
            continue
        v = row[f]
        if is_scalar(v) and bool(pd.isna(v)):
            continue
        parts.append(
            f"<b>{html.escape(str(alias))}</b> {html.escape(str(v))}"
        )
    return "<br>".join(parts) if parts else ""


def add_nonpoint_geojson(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
    *,
    style_function: Callable[[dict], Mapping[str, Any]],
    name: str | None = None,
    tooltip: folium.GeoJsonTooltip | None = None,
) -> None:
    if gdf.empty:
        return
    gj = folium.GeoJson(
        gdf,
        name=name,
        style_function=style_function,
        tooltip=tooltip,
    )
    gj.add_to(m)


def add_uniform_point_layer(
    m: folium.Map,
    gdf_points: gpd.GeoDataFrame,
    *,
    tooltip_fields: list[str],
    tooltip_aliases: list[str],
    geojson_marker: folium.Circle,
    cluster_circle_radius: int = 5,
    cluster_threshold: int = MAP_CLUSTER_POINT_THRESHOLD,
    cluster_color: str = "blue",
    cluster_fill_color: str = "blue",
    cluster_fill_opacity: float = 0.85,
    layer_name: str = "Points",
) -> None:
    """Uniform style: GeoJson for small point sets, FastMarkerCluster for large.

    Set ``cluster_threshold=0`` to always use FastMarkerCluster (e.g. dense grids).
    """
    if gdf_points.empty:
        return
    n = len(gdf_points)
    if n <= cluster_threshold:
        gj_kw: dict[str, Any] = {"marker": geojson_marker}
        if tooltip_fields:
            gj_kw["tooltip"] = folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases or tooltip_fields,
            )
        folium.GeoJson(gdf_points, **gj_kw).add_to(m)
        return
    want_tt = bool(tooltip_fields) and n <= MAP_FAST_CLUSTER_TOOLTIP_MAX
    data = _point_coords_and_optional_tooltips(
        gdf_points,
        tooltip_fields,
        tooltip_aliases,
        include_tooltips=want_tt,
    )
    if not data:
        return
    FastMarkerCluster(
        data=data,
        name=layer_name,
        callback=_fast_cluster_uniform_callback(
            radius=cluster_circle_radius,
            color=cluster_color,
            fill_color=cluster_fill_color,
            fill_opacity=cluster_fill_opacity,
            bind_tooltip_from_row=want_tt,
        ),
    ).add_to(m)


def add_study_area_layers(
    m: folium.Map,
    raw_gdf: gpd.GeoDataFrame,
    *,
    study_name: str,
    buffer_m: float,
    buffer_name: str | None = None,
) -> None:
    """Study area (blue): polygons/lines as GeoJson; point clouds use FastMarkerCluster when large.

    A single ``folium.GeoJson`` on tens of thousands of points freezes the UI and bypasses
    clustering — never pass an all-point ``GeoDataFrame`` through raw GeoJson.
    """
    nonpt, pts = split_points_and_nonpoints(raw_gdf)
    if not nonpt.empty:
        folium.GeoJson(
            nonpt,
            name=study_name,
            style_function=_STUDY_OUTLINE_STYLE,
        ).add_to(m)
    if not pts.empty:
        add_uniform_point_layer(
            m,
            pts,
            tooltip_fields=[],
            tooltip_aliases=[],
            geojson_marker=folium.Circle(
                radius=3,
                color="#1a73e8",
                weight=2,
                fill=False,
            ),
            cluster_circle_radius=4,
            cluster_color="#1a73e8",
            cluster_fill_color="#1a73e8",
            cluster_fill_opacity=0.35,
            layer_name="Study points",
        )
    if buffer_m > 0:
        buf_gdf = apply_buffer_m(raw_gdf, buffer_m)
        bn = buffer_name if buffer_name is not None else f"{study_name} (buffer)"
        folium.GeoJson(
            buf_gdf,
            name=bn,
            style_function=_BUFFER_STYLE,
        ).add_to(m)


def add_mixed_geojson_preview(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
    *,
    nonpoint_style: Callable[[dict], Mapping[str, Any]] | None = None,
    point_geojson_marker: folium.Circle | None = None,
    point_tooltip_fields: list[str] | None = None,
    point_tooltip_aliases: list[str] | None = None,
    layer_name: str | None = None,
    cluster_circle_radius: int = 5,
    cluster_color: str = "blue",
    cluster_fill_color: str = "blue",
    cluster_fill_opacity: float = 0.85,
    point_layer_name: str = "Points",
) -> None:
    """Generic GeoJSON preview: polygons/lines as GeoJson; points via cluster when large."""
    if gdf.empty:
        return
    nonpt, pt = split_points_and_nonpoints(gdf)
    style = nonpoint_style or (lambda x: {"color": "#3388ff", "weight": 2, "fillOpacity": 0.2})
    if not nonpt.empty:
        folium.GeoJson(
            nonpt,
            name=layer_name,
            style_function=style,
        ).add_to(m)
    if not pt.empty:
        fields = point_tooltip_fields or []
        aliases = point_tooltip_aliases or fields
        marker = point_geojson_marker or folium.Circle(
            radius=6, color="blue", fill=True, fill_opacity=0.7
        )
        add_uniform_point_layer(
            m,
            pt,
            tooltip_fields=fields,
            tooltip_aliases=aliases,
            geojson_marker=marker,
            cluster_circle_radius=cluster_circle_radius,
            cluster_color=cluster_color,
            cluster_fill_color=cluster_fill_color,
            cluster_fill_opacity=cluster_fill_opacity,
            layer_name=point_layer_name,
        )


def add_outcome_colored_geometry_layer(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
    *,
    value_column: str,
    value_to_hex: Callable[[float], str],
    fill_opacity: float = 0.7,
    line_opacity: float = 0.7,
    cluster_threshold: int = MAP_CLUSTER_POINT_THRESHOLD,
) -> bool:
    """Fusion-style outcome coloring: GeoJson for areas/lines; clustered CircleMarkers for many points."""
    vals = gdf[value_column].dropna()
    if len(vals) == 0:
        return False

    poly_features: list[dict] = []
    point_specs: list[tuple[float, float, str, float]] = []

    gt = gdf.geometry.geom_type
    mask_other = ~gt.isin(["Point", "MultiPoint"])
    for _, row in gdf.loc[mask_other].iterrows():
        try:
            v = float(row[value_column])
        except (TypeError, ValueError):
            continue
        if not pd.notna(v) or not np.isfinite(v):
            continue
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        hc = value_to_hex(v)
        try:
            geom_d = shapely_mapping(geom)
        except Exception:
            continue
        poly_features.append(
            {
                "type": "Feature",
                "geometry": geom_d,
                "properties": {"_color": hc, "_v": v},
            }
        )

    sub_pt = gdf.loc[gt == "Point"]
    if len(sub_pt) > 0:
        vnp = np.asarray(
            pd.to_numeric(sub_pt[value_column], errors="coerce"),
            dtype=np.float64,
        )
        ok = np.isfinite(vnp)
        sub_pt = sub_pt.iloc[ok]
        vnp = vnp[ok]
        ys = sub_pt.geometry.y.to_numpy(dtype=float, copy=False)
        xs = sub_pt.geometry.x.to_numpy(dtype=float, copy=False)
        for i in range(len(sub_pt)):
            v = float(vnp[i])
            point_specs.append((float(ys[i]), float(xs[i]), value_to_hex(v), v))

    sub_mp = gdf.loc[gt == "MultiPoint"]
    for _, row in sub_mp.iterrows():
        try:
            v = float(row[value_column])
        except (TypeError, ValueError):
            continue
        if not pd.notna(v) or not np.isfinite(v):
            continue
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        hc = value_to_hex(v)
        for p in geom.geoms:
            point_specs.append((float(p.y), float(p.x), hc, v))

    if not poly_features and not point_specs:
        return False

    if poly_features:
        folium.GeoJson(
            {"type": "FeatureCollection", "features": poly_features},
            style_function=lambda f, _fo=fill_opacity, _lo=line_opacity: {
                "fillColor": f["properties"]["_color"],
                "color": f["properties"]["_color"],
                "weight": 2,
                "fillOpacity": _fo,
                "opacity": _lo,
                "radius": 6,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["_v"],
                labels=False,
                sticky=True,
                localize=True,
            ),
        ).add_to(m)

    if point_specs:
        if len(point_specs) > cluster_threshold:
            data = [
                [float(lat), float(lon), hc, float(v)]
                for lat, lon, hc, v in point_specs
            ]
            outcome_callback = (
                "function (row) {"
                "var marker = L.circleMarker(new L.LatLng(row[0], row[1]), {"
                "radius: 5, color: row[2], fillColor: row[2], weight: 1, "
                f"fillOpacity: {float(fill_opacity)}"
                "});"
                "marker.bindTooltip(String(row[3]), {sticky: true, direction: 'top'});"
                "return marker; }"
            )
            FastMarkerCluster(
                data=data,
                name="Outcome points",
                callback=outcome_callback,
            ).add_to(m)
        else:
            pf = [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat],
                    },
                    "properties": {"_color": hc, "_v": v},
                }
                for lat, lon, hc, v in point_specs
            ]
            folium.GeoJson(
                {"type": "FeatureCollection", "features": pf},
                style_function=lambda f, _fo=fill_opacity, _lo=line_opacity: {
                    "fillColor": f["properties"]["_color"],
                    "color": f["properties"]["_color"],
                    "weight": 2,
                    "fillOpacity": _fo,
                    "opacity": _lo,
                    "radius": 6,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["_v"],
                    labels=False,
                    sticky=True,
                    localize=True,
                ),
            ).add_to(m)
    return True
