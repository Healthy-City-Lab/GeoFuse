import datetime
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st

from geofuse.core import (  # noqa: F401  (re-exported for UI modules)
    generate_raster_grid,
)
from geofuse.crs_utils import buffer_gdf_union_metres
from geofuse.vector_io import read_vector_path, target_path_is_raster

FUSION_VECTOR_UPLOAD_TYPES: list[str] = [
    "geojson",
    "json",
    "gpkg",
    "zip",
    "shp",
    "dbf",
    "shx",
    "prj",
    "cpg",
    "qpj",
]
FUSION_RASTER_UPLOAD_TYPES: list[str] = ["tif", "tiff"]
FUSION_TARGET_UPLOAD_TYPES: list[str] = (
    FUSION_VECTOR_UPLOAD_TYPES + FUSION_RASTER_UPLOAD_TYPES
)


@dataclass(frozen=True)
class MaterializedDataset:
    """A single dataset written to disk for GDAL/rasterio (caller cleans up)."""

    path: str
    is_vector: bool
    display_name: str
    cleanup_dir: str | None
    cleanup_file: str | None

_SINGLE_VECTOR_SUFFIXES: frozenset[str] = frozenset(
    {".geojson", ".json", ".gpkg", ".zip"}
)
_RASTER_SUFFIXES: frozenset[str] = frozenset({".tif", ".tiff"})


def materialize_uploaded_dataset(uploaded_files: Sequence) -> MaterializedDataset:
    """Write one uploaded vector or raster target/metric to a temp path for GDAL I/O.

    Shapefile sidecars share a basename stem and are written into one folder.
    Exactly one logical dataset is allowed per invocation.

    The returned path stays on disk until ``cleanup_dir`` / ``cleanup_file`` is
    removed (e.g. after a background job finishes).
    """
    if not uploaded_files:
        raise ValueError("No files uploaded.")
    files = list(uploaded_files)
    by_stem: dict[str, list] = defaultdict(list)
    for f in files:
        by_stem[Path(f.name).stem].append(f)
    if len(by_stem) > 1:
        raise ValueError(
            "Multiple datasets in one upload. Select files for a single dataset only."
        )
    _stem, group = next(iter(by_stem.items()))
    suffixes = {Path(f.name).suffix.lower() for f in group}
    has_shp = ".shp" in suffixes
    has_tif = bool(suffixes & _RASTER_SUFFIXES)
    if has_shp and has_tif:
        raise ValueError(
            "Cannot mix shapefile components and GeoTIFF in the same upload batch."
        )

    if has_shp:
        tmpd = tempfile.mkdtemp(prefix="gf_mat_shp_")
        for f in group:
            out_p = os.path.join(tmpd, os.path.basename(f.name))
            with open(out_p, "wb") as w:
                w.write(f.getvalue())
        shp_upload = next(f for f in group if Path(f.name).suffix.lower() == ".shp")
        shp_disk = os.path.join(tmpd, os.path.basename(shp_upload.name))
        return MaterializedDataset(
            path=shp_disk,
            is_vector=True,
            display_name=os.path.basename(shp_upload.name),
            cleanup_dir=tmpd,
            cleanup_file=None,
        )

    if has_tif:
        tif_members = [f for f in group if Path(f.name).suffix.lower() in _RASTER_SUFFIXES]
        if len(tif_members) != 1:
            raise ValueError("Upload exactly one GeoTIFF file for a raster dataset.")
        f = tif_members[0]
        suf = Path(f.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
            tmp.write(f.getvalue())
            tpath = tmp.name
        return MaterializedDataset(
            path=tpath,
            is_vector=False,
            display_name=f.name,
            cleanup_dir=None,
            cleanup_file=tpath,
        )

    vector_files = [f for f in group if Path(f.name).suffix.lower() in _SINGLE_VECTOR_SUFFIXES]
    if len(vector_files) != 1:
        raise ValueError(
            "Upload one vector file (GeoJSON, GeoPackage, or zip), "
            "or all parts of one shapefile."
        )
    f = vector_files[0]
    suf = Path(f.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
        tmp.write(f.getvalue())
        tpath = tmp.name
    is_vector = not target_path_is_raster(tpath)
    return MaterializedDataset(
        path=tpath,
        is_vector=is_vector,
        display_name=f.name,
        cleanup_dir=None,
        cleanup_file=tpath,
    )


def cleanup_materialized_dataset(ds: MaterializedDataset) -> None:
    """Remove temp paths created by :func:`materialize_uploaded_dataset`."""
    if ds.cleanup_dir and os.path.isdir(ds.cleanup_dir):
        shutil.rmtree(ds.cleanup_dir, ignore_errors=True)
    if ds.cleanup_file and os.path.isfile(ds.cleanup_file):
        try:
            os.remove(ds.cleanup_file)
        except OSError:
            pass


def apply_buffer_m(gdf: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame whose geometry is the union of ``gdf`` buffered by
    ``buffer_m`` metres, re-projected back to the original CRS.

    Distances are applied in a **projected** CRS (metre-based dataset CRS or estimated
    UTM), never as geographic degrees.

    If ``buffer_m`` is 0 the original GeoDataFrame is returned unchanged.
    """
    return buffer_gdf_union_metres(gdf, buffer_m)


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


def load_vector_upload_sessions(
    uploaded_files: Sequence,
) -> list[tuple[str, gpd.GeoDataFrame]]:
    """Load vector uploads (GeoJSON, GeoPackage, Shapefile sidecars, zip) to EPSG:4326.

    Shapefile components are matched by **basename** (``roads.shp``, ``roads.dbf``,
    ``roads.shx``, ``roads.prj``, …) and written to a temporary folder so GDAL reads
    the full bundle including the attribute table (``.dbf``).

    Returns ``(dataset_key, gdf)`` where ``dataset_key`` is the main filename (e.g.
    ``study.shp`` or ``areas.geojson``) for session state and job names.
    """
    if not uploaded_files:
        return []
    files = list(uploaded_files)
    by_stem: dict[str, list] = defaultdict(list)
    for f in files:
        stem = Path(f.name).stem
        by_stem[stem].append(f)

    results: list[tuple[str, gpd.GeoDataFrame]] = []

    for _stem, group in by_stem.items():
        suffixes = {Path(f.name).suffix.lower() for f in group}
        if ".shp" in suffixes:
            tmpd = tempfile.mkdtemp(prefix="gf_shp_")
            try:
                for f in group:
                    out_p = os.path.join(tmpd, os.path.basename(f.name))
                    with open(out_p, "wb") as w:
                        w.write(f.getvalue())
                shp_upload = next(
                    f for f in group if Path(f.name).suffix.lower() == ".shp"
                )
                shp_disk = os.path.join(tmpd, os.path.basename(shp_upload.name))
                gdf = read_vector_path(shp_disk)
                results.append(
                    (
                        os.path.basename(shp_upload.name),
                        sanitize_gdf_attributes_for_json(gdf),
                    )
                )
            finally:
                shutil.rmtree(tmpd, ignore_errors=True)
            continue

        for f in group:
            suf = Path(f.name).suffix.lower()
            if suf not in _SINGLE_VECTOR_SUFFIXES:
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
                tmp.write(f.getvalue())
                tpath = tmp.name
            try:
                gdf = read_vector_path(tpath)
                results.append((f.name, sanitize_gdf_attributes_for_json(gdf)))
            finally:
                if os.path.exists(tpath):
                    os.remove(tpath)

    return results
