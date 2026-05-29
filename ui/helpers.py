import datetime
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from geofuse.core import (  # noqa: F401  (re-exported for UI modules)
    generate_clustered_grid,
    generate_raster_grid,
)
from geofuse.crs_utils import buffer_gdf_union_metres
from geofuse.vector_io import geometry_sha256, read_vector_path, target_path_is_raster

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
        tif_members = [
            f for f in group if Path(f.name).suffix.lower() in _RASTER_SUFFIXES
        ]
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

    vector_files = [
        f for f in group if Path(f.name).suffix.lower() in _SINGLE_VECTOR_SUFFIXES
    ]
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


def rasterize_points_for_preview(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    max_dim: int = 1200,
    bounds: tuple[float, float, float, float] | None = None,
):
    """Bin a point GeoDataFrame into a (h, w) array for a Folium overlay.

    When ``bounds`` is provided (left, bottom, right, top) the GDF is first
    clipped to that bbox and the output grid spans exactly the bbox; otherwise
    the GDF's ``total_bounds`` drive the grid. The longer axis becomes
    ``max_dim`` pixels. Cells with no point stay NaN. Returns
    ``(arr, (left, bottom, right, top), width, height)`` or ``None``.
    """
    import numpy as _np
    from rasterio.transform import from_bounds, rowcol

    if gdf is None or gdf.empty or value_col not in gdf.columns:
        return None
    res = gdf.dropna(subset=[value_col])
    if res.empty:
        return None
    if bounds is not None:
        left, bottom, right, top = bounds
        res = res.cx[left:right, bottom:top]
        if res.empty:
            return None
    else:
        left, bottom, right, top = res.total_bounds
    width_unit = right - left
    height_unit = top - bottom
    if width_unit <= 0 or height_unit <= 0:
        return None
    if width_unit >= height_unit:
        width = max_dim
        height = max(1, int(round(max_dim * height_unit / width_unit)))
    else:
        height = max_dim
        width = max(1, int(round(max_dim * width_unit / height_unit)))
    transform = from_bounds(left, bottom, right, top, width, height)
    arr = _np.full((height, width), _np.nan, dtype=_np.float32)
    rows, cols = rowcol(transform, res.geometry.x.values, res.geometry.y.values)
    rows = _np.asarray(rows, dtype=int)
    cols = _np.asarray(cols, dtype=int)
    in_range = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    arr[rows[in_range], cols[in_range]] = res[value_col].to_numpy()[in_range]
    return arr, (left, bottom, right, top), width, height


def file_size_mtime_fingerprint(path: str | None) -> str:
    """Cheap deterministic fingerprint for a file: ``"<size>:<mtime_int>"``.

    Used by the fusion restart flow to detect whether per-wave longitudinal
    files have been edited / moved / replaced since the original job ran. An
    empty string means the path is missing or ``None`` (treated as drift on
    comparison). Size + integer-mtime is intentionally lightweight — a full
    SHA256 of a large NDVI raster would block the submit handler for several
    seconds and isn't needed to detect the cases the restart panel cares
    about (file gone, file rewritten in place, file truncated).
    """
    if not path:
        return ""
    try:
        st = os.stat(path)
    except OSError:
        return ""
    return f"{st.st_size}:{int(st.st_mtime)}"


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


# ─── Job Restart Workflow ───────────────────────────────────────────

RESTART_SESSION_KEY = "_restart_job_id"


def render_job_restart_panel(
    rec,
    *,
    accept_types: Sequence[str],
    summary_lines: Sequence[str],
    extra_inputs_renderer: Callable[[], dict] | None = None,
    on_confirm: Callable[..., None],
) -> None:
    """Render the geometry-restart workflow for one terminal job.

    Flow inside an expander:
      1. Show the job's original params (read-only via ``summary_lines``).
      2. File uploader for the input geometry.
      3. Once uploaded, load via :func:`load_vector_upload_sessions` and hash.
         * If the record carries a ``geometry_sha256`` (recorded by the job at
           submit time) the hash must match — mismatch shows ``st.error``
           and stops.
         * If the hash is missing (pre-update job) we emit ``st.toast`` and
           proceed without verification.
      4. Optional tab-specific inputs (e.g. an API key for GVI in API mode)
         render after a successful match.
      5. Final "Verify & re-run" button calls ``on_confirm(gdf, extras)``.
         The caller's submit code re-uses the original ``rec.params`` for
         step / buffer / save_* flags etc.

    The caller is expected to set ``st.session_state[RESTART_SESSION_KEY] =
    rec.id`` on Restart click; this function clears that key on success or
    Cancel.
    """
    import streamlit as st  # local import to keep helpers.py importable headless

    expected_hash = (rec.params or {}).get("geometry_sha256")
    fname_orig = (rec.params or {}).get("fname", "(unknown)")
    expander_label = f"↻ Restart job: {rec.name or rec.id}"

    with st.expander(expander_label, expanded=True):
        st.caption(
            "Re-upload the original input file. The job will be re-submitted "
            "with the same parameters; cached panoramas / tiles will be reused "
            "where available."
        )
        for line in summary_lines:
            st.write(line)

        cancel_col, _ = st.columns([1, 4])
        with cancel_col:
            if st.button("Cancel restart", key=f"restart_cancel_{rec.id}"):
                st.session_state[RESTART_SESSION_KEY] = None
                st.rerun()

        uploaded = st.file_uploader(
            f"Re-upload input geometry (original: `{fname_orig}`)",
            accept_multiple_files=True,
            type=list(accept_types),
            key=f"restart_upload_{rec.id}",
        )
        if not uploaded:
            st.info("Select the original input file(s) to continue.")
            return

        try:
            loaded = load_vector_upload_sessions(uploaded)
        except Exception as e:
            st.error(f"Failed to load uploaded files: {e}")
            return
        if not loaded:
            st.error("Could not parse any vector dataset from the upload.")
            return
        if len(loaded) > 1:
            st.error(
                "Restart accepts a single dataset; please upload only one "
                "logical file."
            )
            return

        fname_new, gdf = loaded[0]
        actual_hash = geometry_sha256(gdf)

        if expected_hash:
            if actual_hash != expected_hash:
                st.error(
                    "Hash mismatch — the uploaded file is not the same as the "
                    "original input. Restart refused.\n\n"
                    f"  Expected: `{expected_hash[:16]}…`\n"
                    f"  Got:      `{actual_hash[:16]}…`"
                )
                return
            st.success("Input geometry verified against the original.")
        else:
            st.toast(
                "Original geometry hash was not recorded (pre-update job) — "
                "skipping verification.",
                icon="⚠️",
            )
            st.warning(
                "This job pre-dates geometry hashing. Verify the file name and "
                "feature count below match what you remember submitting."
            )
            st.write(f"  Uploaded as: `{fname_new}` · {len(gdf):,} features")

        extras: dict = {}
        if extra_inputs_renderer is not None:
            extras = extra_inputs_renderer() or {}

        if st.button(
            "Verify & re-run", type="primary", key=f"restart_confirm_{rec.id}"
        ):
            try:
                on_confirm(gdf, fname_new, extras)
            except Exception as e:
                st.error(f"Re-submission failed: {e}")
                return
            st.session_state[RESTART_SESSION_KEY] = None
            st.success("Restart submitted. Monitor progress in the sidebar.")
            st.rerun()
