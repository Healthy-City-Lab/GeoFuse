"""Canonical WGS84 (lon/lat) handling for GeoFuse pipelines and Folium previews.

GeoJSON may declare ``OGC:CRS84`` (lon, lat) or legacy ``EPSG:4326``. PyProj / GDAL can
interpret axis order differently across CRS objects. We normalize geographic inputs with
``always_xy=True`` so stored coordinates stay **x = longitude, y = latitude** in degrees,
which matches Leaflet/Folium and ``generate_raster_grid`` in ``geofuse.core``.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from pyproj import CRS as PyProjCRS
from pyproj import Transformer
from shapely.ops import transform as shapely_xy_transform

# Single canonical CRS for web maps, Earth Engine clip geometries, and GVI/NDVI download.
WGS84_EPSG = "EPSG:4326"


def _raise_if_geographic_coords_outside_degree_range(gdf: gpd.GeoDataFrame) -> None:
    """Detect invalid GeoJSON: geographic CRS metadata but projected metre coordinates."""
    if gdf.empty or gdf.crs is None or not gdf.crs.is_geographic:
        return
    try:
        w, s, e, n = gdf.total_bounds
    except Exception:
        return
    if not np.isfinite((w, s, e, n)).all():
        return
    tol = 1e-3
    if (
        abs(w) > 180.0 + tol
        or abs(e) > 180.0 + tol
        or abs(s) > 90.0 + tol
        or abs(n) > 90.0 + tol
    ):
        raise ValueError(
            "This file declares a geographic CRS (e.g. OGC:CRS84 / EPSG:4326), but "
            "its coordinates are outside valid degree ranges (longitude ±180°, "
            "latitude ±90°). The numbers look like **projected metres** (e.g. "
            "EPSG:3978) while the GeoJSON `crs` field was set to a lon/lat CRS "
            "without transforming the geometry — the file is internally inconsistent.\n\n"
            "Fix when exporting: either keep a CRS-aware format (GeoPackage), or "
            "reproject to WGS84 before GeoJSON:\n"
            '  gdf.to_crs("EPSG:4326").to_file(..., driver="GeoJSON")\n'
            "If coordinates are EPSG:3978 but CRS was lost, assign then reproject:\n"
            '  gdf.set_crs("EPSG:3978").to_crs("EPSG:4326").to_file(...)'
        )


def _geographic_to_epsg4326_always_xy(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject geographic CRS (incl. OGC:CRS84) to EPSG:4326 with lon/lat as x/y."""
    if gdf.empty:
        return gdf.copy()
    if gdf.crs is None:
        return gdf.set_crs(WGS84_EPSG)
    transformer = Transformer.from_crs(gdf.crs, WGS84_EPSG, always_xy=True)
    geom_col = gdf.geometry.name

    if len(gdf) > 0 and (gdf.geometry.geom_type == "Point").all():
        x = gdf.geometry.x.to_numpy(dtype=float, copy=False)
        y = gdf.geometry.y.to_numpy(dtype=float, copy=False)
        nx, ny = transformer.transform(x, y)
        return gpd.GeoDataFrame(
            gdf.drop(columns=geom_col),
            geometry=gpd.points_from_xy(nx, ny),
            crs=WGS84_EPSG,
        )

    def _tf(xx: float, yy: float, zz: float | None = None) -> tuple[float, float]:
        rx, ry = transformer.transform(xx, yy)
        return (rx, ry)

    new_geom = gdf.geometry.map(
        lambda g: shapely_xy_transform(_tf, g) if g is not None and not g.is_empty else g
    )
    return gpd.GeoDataFrame(
        gdf.drop(columns=geom_col),
        geometry=new_geom,
        crs=WGS84_EPSG,
    )


def normalize_geographic_gdf_to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalize **geographic** CRS to EPSG:4326; leave **projected** CRS unchanged.

    Use for metric GeoJSON that may be UTM (keep) or CRS84/4326 (normalize).
    """
    if gdf.empty:
        return gdf.copy()
    if gdf.crs is None:
        out = gdf.set_crs(WGS84_EPSG)
        _raise_if_geographic_coords_outside_degree_range(out)
        return out
    if not gdf.crs.is_geographic:
        return gdf
    _raise_if_geographic_coords_outside_degree_range(gdf)
    return _geographic_to_epsg4326_always_xy(gdf)


def crs_uses_metre_axes(crs) -> bool:
    """True if ``crs`` is projected and planar axes use the metre unit."""
    if crs is None:
        return False
    try:
        c = PyProjCRS.from_user_input(crs)
    except Exception:
        return False
    if not c.is_projected:
        return False
    axes = c.axis_info
    if not axes:
        return False
    for ax in axes:
        unit = getattr(ax, "unit_name", None) or ""
        if unit != "metre":
            return False
    return True


def _estimate_utm_epsg_from_wgs84_centroid(gdf_wgs84: gpd.GeoDataFrame) -> str:
    """Fallback UTM EPSG code from geometry centroid in lon/lat degrees."""
    u = gdf_wgs84.geometry.union_all()
    if u is None or u.is_empty:
        raise ValueError("Cannot estimate UTM for empty geometry.")
    lon, lat = float(u.centroid.x), float(u.centroid.y)
    zone = int((lon + 180.0) / 6.0) + 1
    zone = min(60, max(1, zone))
    epsg = 32600 + zone if lat >= 0.0 else 32700 + zone
    return f"EPSG:{epsg}"


def estimate_metre_projected_crs_for_gdf(gdf: gpd.GeoDataFrame) -> str | PyProjCRS:
    """Pick a projected CRS for metre-accurate planar ops (buffer, length).

    Prefer GeoPandas ``estimate_utm_crs`` from the extent; fall back to UTM from the
    WGS84 centroid if estimation fails.
    """
    try:
        est = gdf.estimate_utm_crs()
        if est is not None:
            return est
    except Exception:
        pass
    g_ll = gdf if gdf.crs and gdf.crs.is_geographic else gdf.to_crs(WGS84_EPSG)
    if g_ll.crs is None:
        g_ll = g_ll.set_crs(WGS84_EPSG)
    return _estimate_utm_epsg_from_wgs84_centroid(g_ll)


def buffer_gdf_union_metres(
    gdf: gpd.GeoDataFrame,
    buffer_m: float,
) -> gpd.GeoDataFrame:
    """Single-feature GeoDataFrame: ``union_all(gdf).buffer(buffer_m)`` in metres.

    Distance is never applied in geographic degrees: buffering uses either the
    dataset CRS when it is already projected with metre axes, or an estimated UTM
    CRS, then results are reprojected back to ``gdf.crs`` (or EPSG:4326 if unset).
    """
    if buffer_m <= 0:
        return gdf
    if gdf.empty:
        return gdf.copy()

    orig_crs = gdf.crs
    if orig_crs is not None and crs_uses_metre_axes(orig_crs):
        merged = gdf.geometry.union_all().buffer(buffer_m)
        return gpd.GeoDataFrame({"geometry": [merged]}, crs=orig_crs)

    gdf_work = gdf.set_crs(WGS84_EPSG) if orig_crs is None else gdf
    utm = estimate_metre_projected_crs_for_gdf(gdf_work)
    gdf_utm = gdf_work.to_crs(utm)
    merged = gdf_utm.geometry.union_all().buffer(buffer_m)
    out = gpd.GeoDataFrame({"geometry": [merged]}, crs=gdf_utm.crs)
    if orig_crs is None:
        return out.to_crs(WGS84_EPSG)
    return out.to_crs(orig_crs)


def reproject_geodataframe_to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Put every feature in EPSG:4326 (x=lon°, y=lat°) for engines and map previews."""
    if gdf.empty:
        return gdf.copy()
    if gdf.crs is None:
        out = gdf.set_crs(WGS84_EPSG)
        _raise_if_geographic_coords_outside_degree_range(out)
        return out
    if gdf.crs.is_geographic:
        _raise_if_geographic_coords_outside_degree_range(gdf)
        return _geographic_to_epsg4326_always_xy(gdf)
    return gdf.to_crs(WGS84_EPSG)

