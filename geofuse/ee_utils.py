"""Earth Engine glue helpers used by the GeoFuse engines.

Small, reusable utilities that sit between client-side Shapely / GeoPandas
geometry and Earth Engine's API.  Kept separate from ``ndvi.py`` so future
engines (GVI raster-side, fusion target prep, etc.) can pick them up without
pulling in NDVI-specific state.
"""

from __future__ import annotations

import ee
import geopandas as gpd
from shapely.geometry import mapping

from .logger import get_logger

_log = get_logger("EE")


def shapely_to_ee_geometry(geom):
    """Convert a Shapely geometry to ``ee.Geometry`` via GeoJSON.

    Going through ``shapely.geometry.mapping`` is stable across geemap
    versions and avoids relying on private rasterio/geemap conversion
    helpers.
    """
    return ee.Geometry(mapping(geom))


def crs_to_ee_string(crs) -> str:
    """Return an Earth Engine compatible CRS identifier.

    Prefers ``EPSG:<n>`` when the CRS has an EPSG code; otherwise emits an
    OGC WKT1 (GDAL flavour) string. WKT1_GDAL is the format Earth Engine
    accepts for the custom LCC / Polar Stereographic projections that
    :func:`geofuse.crs_utils.select_grid_crs` synthesises for wide or
    polar inputs.
    """
    epsg = crs.to_epsg()
    if epsg is not None:
        return f"EPSG:{epsg}"
    return crs.to_wkt("WKT1_GDAL")


# Earth Engine's /compute endpoint caps each request body at 10 MiB.  The
# full AOI geometry is embedded in every request the engine makes
# (filterBounds, clip, export region), so a vertex-heavy input — a country
# boundary at full resolution, or a several-hundred-thousand-point sample
# layer — blows past the limit on the first ``.size().getInfo()`` call,
# long before any download starts.  ``shrink_gdf_for_ee`` shrinks only the
# EE-side AOI; callers should retain the precise input geometry for
# client-side per-tile clipping so output precision is unaffected.
EE_REQUEST_BUDGET_BYTES = (
    5_000_000  # ~5 MiB — leave headroom for the rest of the EE compute graph
)


def shrink_gdf_for_ee(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame whose GeoJSON serialisation fits the EE budget.

    Pass-through when the input already fits.  Otherwise simplify the
    geometry with growing tolerance (5 m → ~160 km across 15 doublings),
    preserving topology.  If simplification cannot shrink the payload
    enough — typically a dense point cloud, where ``simplify`` is a no-op
    — fall back to the convex hull of the union, which keeps EE's
    ``filterBounds`` correct.

    The CRS of the returned GeoDataFrame matches the input; the input is
    not modified in place.
    """

    def _size_bytes(g: gpd.GeoDataFrame) -> int:
        return len(g.to_json().encode("utf-8"))

    n0 = _size_bytes(gdf)
    if n0 <= EE_REQUEST_BUDGET_BYTES:
        return gdf

    # 1 deg ≈ 111 km at most latitudes, so 5e-5 deg ≈ 5.5 m.
    tol_deg = 5e-5
    cur = gdf
    for _ in range(15):
        simp = cur.copy()
        simp.geometry = cur.geometry.simplify(tol_deg, preserve_topology=True)
        n = _size_bytes(simp)
        if n <= EE_REQUEST_BUDGET_BYTES:
            _log(
                "WARN",
                f"Input geometry was {n0 / 1e6:.1f} MB serialised, exceeding "
                "the Earth Engine 10 MB request limit; simplified to "
                f"{n / 1e6:.1f} MB at ~{tol_deg * 111_000:.0f} m tolerance. "
                "Original geometry is retained for per-tile clipping.",
            )
            return simp
        tol_deg *= 2

    # Simplification didn't shrink it enough (point clouds, very dense
    # boundaries).  Replace with the convex hull of the union — EE's
    # ``filterBounds`` only needs a region that intersects scenes, and the
    # precise per-tile geometry still drives the actual clip/download.
    hull = cur.geometry.union_all().convex_hull
    out = gpd.GeoDataFrame({"geometry": [hull]}, crs=cur.crs)
    _log(
        "WARN",
        f"Input geometry was {n0 / 1e6:.1f} MB serialised and could not be "
        "simplified under the Earth Engine 10 MB request limit; falling back "
        f"to convex hull ({_size_bytes(out) / 1e6:.1f} MB). Per-tile clipping "
        "still uses the original geometry.",
    )
    return out
