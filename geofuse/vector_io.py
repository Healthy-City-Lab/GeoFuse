"""Read vector datasets from disk (GeoJSON, GeoPackage, Shapefile, zip)."""

from __future__ import annotations

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
