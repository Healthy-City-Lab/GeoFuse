"""Read vector datasets from disk (GeoJSON, GeoPackage, Shapefile, zip)."""

from __future__ import annotations

import datetime
import hashlib
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

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


def match_column_alias(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
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


# ────────────────────────────────────────────────────────────────────
# Input files: fingerprints, loading, merging
# ────────────────────────────────────────────────────────────────────


def file_size_mtime_fingerprint(path: str | None) -> str:
    """Cheap deterministic fingerprint for a file: ``"<size>:<mtime_int>"``.

    Used by the restart flow to detect whether an input file has been
    edited / moved / replaced since the original job ran. An empty string
    means the path is missing or ``None`` (treated as drift on comparison).
    Size + integer-mtime is intentionally lightweight — a full SHA256 of a
    large NDVI raster would block the submit handler for several seconds
    and isn't needed to detect the cases the restart panel cares about
    (file gone, file rewritten in place, file truncated).
    """
    if not path:
        return ""
    try:
        st = os.stat(path)
    except OSError:
        return ""
    return f"{st.st_size}:{int(st.st_mtime)}"


def path_drift_status(path: str | None, expected_fingerprint: str) -> str:
    """Return one of ``"ok"`` / ``"no-path"`` / ``"missing"`` / ``"modified"``.

    Drift semantics shared across every tab's restart flow:
    - ``no-path``: the spec never had a value for this slot (empty string / None)
    - ``missing``: the recorded path is no longer present on disk
    - ``modified``: the file exists but its size+mtime fingerprint differs
      from the one recorded at submit time
    - ``ok``: file present and fingerprint matches
    """
    if not path:
        return "no-path"
    if not os.path.isfile(path):
        return "missing"
    if (
        expected_fingerprint
        and file_size_mtime_fingerprint(path) != expected_fingerprint
    ):
        return "modified"
    return "ok"


def fingerprint_paths(paths: dict[str, str | None]) -> dict[str, str]:
    """Fingerprint every path in ``paths``; missing files produce ``""``."""
    return {k: file_size_mtime_fingerprint(v) for k, v in paths.items()}


def sanitize_gdf_attributes_for_json(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Cast non-geometry attributes to JSON-friendly values (Folium, ``json.dumps``).

    GeoJSON sources with date/time properties are often read as ``datetime64`` or
    ``Timestamp`` in object columns; Folium's ``GeoJson(gdf)`` then fails with
    "Object of type Timestamp is not JSON serializable".
    """
    geom_col = gdf.geometry.name
    out = gdf.copy()
    for col in out.columns:
        if col == geom_col:
            continue
        s = out[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[col] = s.astype(str)
        elif isinstance(s.dtype, pd.CategoricalDtype):
            out[col] = s.astype(str)
        elif s.dtype == object:

            def _cell(v):
                if isinstance(v, (pd.Timestamp, datetime.datetime, datetime.date)):
                    return v.isoformat()
                if isinstance(v, datetime.time):
                    return str(v)
                return v

            out[col] = s.map(_cell)
    return out


def load_vector_paths(
    paths: Sequence[str],
) -> list[tuple[str, gpd.GeoDataFrame]]:
    """Read vector datasets from disk paths to ``(basename, gdf)`` pairs.

    Direct path-based sibling of :func:`load_vector_upload_sessions` for
    flows that capture absolute filesystem paths instead of HTTP-uploaded
    bytes (the file-picker workflow). Each path is read lazily through
    :func:`geofuse.vector_io.read_vector_path` and the attributes are
    sanitised the same way the uploads pathway does. Shapefile sidecars
    must already sit beside the ``.shp`` in the same folder — GDAL picks
    them up automatically.
    """
    results: list[tuple[str, gpd.GeoDataFrame]] = []
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        gdf = read_vector_path(path)
        results.append((os.path.basename(path), sanitize_gdf_attributes_for_json(gdf)))
    return results


def sanitize_name_for_file(name: str) -> str:
    """Make an arbitrary group name safe to use as an output filename stem."""
    cleaned = "".join(c if (c.isalnum() or c in "-_ .") else "_" for c in str(name))
    cleaned = cleaned.strip().replace(" ", "_").replace(".", "_")
    return cleaned or "group"


def merge_gdfs_wgs84(gdfs: Sequence[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Concatenate several vector layers into one GeoDataFrame in EPSG:4326.

    Each input is reprojected to WGS84 first so layers in different CRSs
    combine cleanly; the column set is the union across inputs (missing values
    become NaN). Used to merge several uploaded files into a single job so,
    e.g., every 2020 measurement across survey waves lands in one per-year
    output.
    """
    frames: list[gpd.GeoDataFrame] = []
    for g in gdfs:
        if g is None or len(g) == 0:
            continue
        gg = (
            reproject_geodataframe_to_wgs84(g)
            if g.crs is not None
            else g.set_crs("EPSG:4326")
        )
        frames.append(gg)
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
