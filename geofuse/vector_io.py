"""Read vector datasets from disk (GeoJSON, GeoPackage, Shapefile, zip)."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd

from geofuse.crs_utils import reproject_geodataframe_to_wgs84


def vector_field_names(
    path: str | Path, layer: str | int | None = None
) -> list[str] | None:
    """Non-geometry field names of a vector file, read from its header.

    Lets a caller decide which columns to request before paying to read any
    rows. Returns ``None`` when the driver cannot report a schema, leaving
    the caller to fall back to reading the whole file.
    """
    try:
        import pyogrio

        kwargs: dict = {} if layer is None else {"layer": layer}
        return [str(f) for f in pyogrio.read_info(os.fspath(path), **kwargs)["fields"]]
    except Exception:
        return None


def match_column_alias(
    columns: Iterable[str], aliases: Iterable[str]
) -> str | None:
    """The column matching the earliest of *aliases*, compared case-insensitively.

    Returns the name as it is spelled in *columns*, or ``None`` if no alias
    is present.
    """
    lowered = {str(c).lower(): str(c) for c in columns}
    return next((lowered[a] for a in aliases if str(a).lower() in lowered), None)


def _layer_kwargs(layer: str | int | None) -> dict:
    return {} if layer is None else {"layer": layer}


def read_vector_subset(
    path: str | Path,
    columns: Iterable[str],
    *,
    layer: str | int | None = None,
) -> gpd.GeoDataFrame:
    """Read *path* carrying only *columns*, plus geometry.

    Requested names absent from the file are skipped rather than raising, so
    the caller's own validation reports them against the file's real schema
    instead of pruning turning a missing column into a silently empty one.
    Falls back to a full read when the driver cannot report a schema.
    """
    read_kwargs = _layer_kwargs(layer)
    fields = vector_field_names(path, layer=layer)
    if fields is not None:
        available = set(fields)
        read_kwargs["columns"] = [c for c in columns if c in available]
    return gpd.read_file(os.fspath(path), **read_kwargs)


def read_vector_aliased_column(
    path: str | Path,
    aliases: Sequence[str],
    *,
    layer: str | int | None = None,
    description: str = "value",
) -> tuple[gpd.GeoDataFrame, str]:
    """Read the single column named by the first matching alias, plus geometry.

    Returns the frame and the resolved column name. Raises ``ValueError``
    naming the file's real columns when no alias matches, whether or not the
    schema could be read up front.
    """

    def _missing(available: list[str]) -> ValueError:
        return ValueError(
            f"Cannot find {description} column in {path}. Expected one of "
            f"{list(aliases)}; got columns {available}."
        )

    fields = vector_field_names(path, layer=layer)
    if fields is not None:
        col = match_column_alias(fields, aliases)
        if col is None:
            raise _missing(fields)
        return read_vector_subset(path, [col], layer=layer), col

    gdf = gpd.read_file(os.fspath(path), **_layer_kwargs(layer))
    col = match_column_alias(gdf.columns, aliases)
    if col is None:
        raise _missing(list(gdf.columns))
    return gdf, col


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
    h.update(f"crs:{gdf.crs}\n".encode())
    # Sort by index so row order can't change the hash. WKB is produced for the
    # whole column in one pass; the per-feature framing below is unchanged, so
    # digests still match those written by earlier runs.
    geoms = gdf.geometry.sort_index()
    wkbs = geoms.to_wkb()
    empties = (geoms.is_empty | geoms.isna()).to_numpy()
    for wkb, is_empty in zip(wkbs, empties):
        if is_empty or wkb is None:
            h.update(b"\x00")
            continue
        h.update(wkb)
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
