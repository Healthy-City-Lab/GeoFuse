"""Read vector datasets from disk (GeoJSON, GeoPackage, Shapefile, zip)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import geopandas as gpd

from geofuse.crs_utils import reproject_geodataframe_to_wgs84


def target_path_is_raster(path: str | Path) -> bool:
    """True if ``path`` should be read as a GeoTIFF (or raster inside a zip), not fiona."""
    suf = Path(path).suffix.lower()
    if suf in (".tif", ".tiff"):
        return True
    if suf in (".geojson", ".json", ".shp", ".gpkg"):
        return False
    if suf == ".zip":
        try:
            import rasterio

            with rasterio.open(os.fspath(path)) as src:
                _ = src.count
            return True
        except Exception:
            return False
    return False


def list_gpkg_layer_names(path: str | Path) -> list[str]:
    """Return Fiona layer names for a GeoPackage (or other multi-layer sources)."""
    import fiona

    return list(fiona.listlayers(os.fspath(path)))


def vector_format_from_path(path: str | Path) -> str:
    """Return a short format label for logging or UI (not exhaustive)."""
    ext = Path(path).suffix.lower()
    if ext == ".gpkg":
        return "geopackage"
    if ext in (".geojson", ".json"):
        return "geojson"
    if ext == ".shp":
        return "shapefile"
    if ext == ".zip":
        return "zip"
    return ext.lstrip(".") or "unknown"


def geometry_sha256(gdf: gpd.GeoDataFrame) -> str:
    """Stable SHA-256 over a GeoDataFrame's geometries + CRS string.

    Used to verify on job restart that the user re-uploaded the same input
    file. Stable across runs for an unchanged GDF: features are sorted by
    pandas index, each geometry's WKB is appended to the hash, then the CRS
    string is appended. Attribute columns are intentionally excluded so
    column reorderings or dtype roundtrips don't invalidate the hash.
    """
    h = hashlib.sha256()
    # str(crs) covers EPSG codes and full WKT; falls back to "None" when unset.
    h.update(f"crs:{gdf.crs}\n".encode("utf-8"))
    # Sort by index so row order can't change the hash.
    for _, geom in gdf.geometry.sort_index().items():
        if geom is None or geom.is_empty:
            h.update(b"\x00")
            continue
        h.update(geom.wkb)
        h.update(b"\x1e")  # record separator
    return h.hexdigest()


def read_vector_path(
    path: str | Path,
    *,
    layer: str | int | None = None,
) -> gpd.GeoDataFrame:
    """Load a vector file with fiona/GDAL; preserves attributes; returns EPSG:4326.

    ``path`` may be a ``.shp`` path (``.dbf`` / ``.shx`` / ``.prj`` in the same folder
    are read automatically), a ``.gpkg`` layer, GeoJSON, or a ``.zip`` archive GDAL can
    open (often a zipped shapefile or GeoPackage).
    """
    p = os.fspath(path)
    kwargs: dict[str, Any] = {}
    if layer is not None:
        kwargs["layer"] = layer
    gdf = gpd.read_file(p, **kwargs)
    return reproject_geodataframe_to_wgs84(gdf)
