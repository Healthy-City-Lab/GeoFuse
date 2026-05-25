from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import ee
import geemap
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import mapping

from .core import build_planar_tiles
from .crs_utils import (
    crs_to_ee_string,
    reproject_geodataframe_to_wgs84,
    reproject_raster_to_wgs84,
    select_grid_crs_with_warning,
    stream_mosaic_to_geotiff,
)
from .jobs import progress_interval_s, retry_with_backoff
from .logger import attach_external_logger, get_logger
from .persistence.ndvi_tile_cache import DEFAULT_MAX_BYTES, NdviTileCache
from .raster_sampling import sample_raster_at_features_to_file
from .vector_io import geometry_sha256

_log = get_logger("NDVI")

# TODO: NDVI_TEMPORAL - Add time-series analysis for seasonal greenery changes

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Persistent NDVI tile cache root. Cross-process caches live under one
# well-known directory. Tile bodies live under ``<root>/<resume_key>/``
# alongside a SQLite WAL index used for LRU eviction.
_TILE_CACHE_ROOT = os.path.join(_REPO_ROOT, "logs", "caches", "ndvi_tiles")

# Concurrent tile-download workers. Earth Engine allows several parallel
# ``ee_export_image`` calls per user well above this; the cap exists to avoid
# saturating local sockets and to keep the post-download rasterio decode +
# reproject step from contending too heavily for the GIL on a small machine.
# Mirrors the GVI engine's ``_MAX_CONCURRENT_POINTS = 4`` choice.
_MAX_CONCURRENT_TILES = 4


def _compute_resume_key(
    geom_wgs84_gdf: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    cloud_max: float,
    resolution: int,
    ee_collection: str,
) -> str:
    """Stable 16-hex-char key identifying a unique NDVI run.

    Two runs that hash to the same key produce identical output, so they can
    share an on-disk tile workspace — that's what lets resume work. The key
    includes everything that affects the per-tile contents: study area
    (geometry + CRS), date range, cloud threshold, export resolution, and
    Earth Engine collection ID. Changing any one of these spawns a fresh
    workspace and forces a full re-download (which is what you want — old
    tiles would be from a different question).
    """
    h = hashlib.sha256()
    # Schema tag — bump when the on-disk tile format changes. ``planar_v1``
    # marks the post-Step-13.5 layout (planar-CRS tiles snapped to a global
    # grid). Older WGS84-tile cache entries hash to different keys and are
    # naturally orphaned (LRU-evicted as space pressure builds).
    h.update(b"planar_v1|")
    h.update(geometry_sha256(geom_wgs84_gdf).encode())
    h.update(f"|{start_date}|{end_date}|{cloud_max}|{resolution}|".encode())
    h.update(ee_collection.encode())
    return h.hexdigest()[:16]


def _discover_resumable_tiles(
    work_dir: str, tiles: list[dict]
) -> tuple[list[str], list[dict]]:
    """Split ``tiles`` into (already-on-disk, still-pending).

    A tile counts as "already on disk" iff ``tile_<idx>.tif`` exists in
    ``work_dir`` and opens cleanly. Half-written / corrupt files are
    discarded so the next run re-downloads them.
    """
    existing_files: list[str] = []
    pending: list[dict] = []
    for ts in tiles:
        idx = int(ts["tile_idx"])
        path = os.path.join(work_dir, f"tile_{idx}.tif")
        if not os.path.isfile(path):
            pending.append(ts)
            continue
        try:
            with rasterio.open(path) as _src:
                _ = _src.bounds  # touch metadata to detect truncation
            existing_files.append(path)
        except Exception:
            try:
                os.remove(path)
            except OSError:
                pass
            pending.append(ts)
    return existing_files, pending


def _emit_ndvi_progress(
    ndvi_progress_callback: Callable[[Mapping[str, Any]], None] | None,
    *,
    sub_progress: float,
    phase: str | None = None,
    tiles: tuple[int, int] | None = None,
    clear_bracket: bool = False,
) -> None:
    if not ndvi_progress_callback:
        return
    payload: dict[str, Any] = {"sub_progress": float(sub_progress)}
    if phase is not None:
        payload["phase"] = phase
    if tiles is not None:
        payload["tiles"] = tiles
    if clear_bracket:
        payload["clear_bracket"] = True
    ndvi_progress_callback(payload)


def _shapely_to_ee_geometry(geom):
    """Convert Shapely geometry to ``ee.Geometry`` via GeoJSON (stable across geemap versions)."""
    return ee.Geometry(mapping(geom))


def _write_ndvi_sidecar(
    folder: str,
    output_name: str,
    *,
    grid_crs,
    export_crs: str,
    export_crs_name: str,
    export_distortion: float,
    n_clusters: int,
    tiles_total: int,
    tiles_succeeded: int,
    tiles_failed: int = 0,
    tiles_resumed: int = 0,
    failed_tile_refs: list[dict] | None = None,
    resume_key: str = "",
    start_date: str,
    end_date: str,
    used_start_date: str | None = None,
    used_end_date: str | None = None,
    coverage_widened: bool = False,
    cloud_max: float,
    resolution_m: int,
    max_tile_size_km: float,
    ee_collection: str,
    satellite: str = "sentinel2",
    bands: list[str] | None = None,
    n_cloud_filtered_images: int | None = None,
) -> str:
    """Write ``{output_name}_ndvi.json`` capturing the parameters and grid CRS.

    Mirrors GVI's ``_gvi.json`` so any downstream tool (fusion, custom
    notebooks) can introspect an NDVI raster after the fact: which planar CRS
    pixels were rasterised in, how much distortion that introduced, how many
    clusters and tiles the input decomposed into, the exact date range, and
    which Earth Engine ImageCollection was queried. ``failed_tile_refs``
    lists the cluster/tile IDs that exhausted their retry budget — those
    areas appear as NaN gaps in the mosaic.
    """
    sidecar_path = os.path.join(folder, f"{output_name}_ndvi.json")
    payload = {
        "export_crs": export_crs,
        "export_crs_name": export_crs_name,
        "export_crs_wkt": grid_crs.to_wkt() if grid_crs is not None else None,
        "distortion": float(export_distortion),
        "n_clusters": int(n_clusters),
        "tiles_total": int(tiles_total),
        "tiles_succeeded": int(tiles_succeeded),
        "tiles_failed": int(tiles_failed),
        "tiles_resumed": int(tiles_resumed),
        "resume_key": resume_key,
        "failed_tile_refs": list(failed_tile_refs or []),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "used_start_date": str(used_start_date or start_date),
        "used_end_date": str(used_end_date or end_date),
        "coverage_widened": bool(coverage_widened),
        "cloud_max": float(cloud_max),
        "resolution_m": int(resolution_m),
        "max_tile_size_km": float(max_tile_size_km),
        "ee_collection": ee_collection,
        "satellite": satellite,
        "bands": list(bands or ["NDVI", "valid_obs"]),
        "n_cloud_filtered_images": (
            int(n_cloud_filtered_images)
            if n_cloud_filtered_images is not None
            else None
        ),
    }
    with open(sidecar_path, "w") as f:
        json.dump(payload, f, indent=2)
    return sidecar_path


def _select_export_crs(geom_wgs84_gdf: gpd.GeoDataFrame):
    """Pick an EE export CRS for true-ground-metre pixels at any latitude.

    Thin NDVI-side adapter on top of :func:`select_grid_crs_with_warning`
    that also formats the CRS for Earth Engine. Returns
    ``(ee_string, crs_obj, distortion, choice_name)``; ``crs_obj`` is the
    pyproj CRS so the cluster-aware tile builder can reproject geometries
    without re-running the selector.
    """
    crs, distortion, choice_name = select_grid_crs_with_warning(
        geom_wgs84_gdf, _log, role="Export CRS"
    )
    return crs_to_ee_string(crs), crs, distortion, choice_name


class NDVIEngine:
    #: Default Earth Engine ImageCollection. ``download_and_process``
    #: substitutes a Landsat collection when ``satellite="landsat"`` or
    #: ``satellite="auto"`` resolves to Landsat (pre-Sentinel-2 dates).
    EE_COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"

    #: First date covered by ``COPERNICUS/S2_SR_HARMONIZED``. Anything
    #: earlier comes back empty; ``auto`` satellite mode picks Landsat for
    #: ranges that end before this. Diagnostics also reference it.
    S2_COLLECTION_START = "2017-03-28"

    #: Merged Landsat 8/9 Collection 2 Tier 1 Level-2 source for the
    #: pre-Sentinel-2 fallback. LC08 covers 2013-04+, LC09 covers 2021-10+;
    #: combined as a single ee.ImageCollection at query time.
    LANDSAT8_COLLECTION_ID = "LANDSAT/LC08/C02/T1_L2"
    LANDSAT9_COLLECTION_ID = "LANDSAT/LC09/C02/T1_L2"

    def __init__(
        self,
        project_id=None,
        *,
        tile_cache: NdviTileCache | None = None,
        tile_cache_dir: str | None = None,
        tile_cache_max_bytes: int = DEFAULT_MAX_BYTES,
    ):
        """Initialize Earth Engine and open the persistent tile cache.

        ``tile_cache`` lets the caller inject a shared :class:`NdviTileCache`
        (e.g. a singleton from ``ui/services.py``). When omitted, the engine
        opens its own at ``tile_cache_dir`` (default
        ``<repo>/logs/caches/ndvi_tiles/``) so direct API callers — fusion
        auto-download, CLI, notebooks — benefit from the same cross-run
        cache the UI uses. The cache is keyed by ``resume_key`` (Step 9) so
        repeat runs over the same area + date range are near-instant.

        Earth Engine's stdlib ``logging`` output is captured into the bound
        per-job log here (with propagation to the root logger disabled),
        keeping the Streamlit host terminal free of EE chatter.
        """
        # Route the Earth Engine stdlib logger (and its descendants) into
        # the per-job pipeline before EE itself starts talking. Idempotent
        # across engine instances and processes.
        attach_external_logger("ee", logging.INFO)

        try:
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()
        except ee.EEException as e:
            _log("WARN", f"Earth Engine init failed: {e}")
            try:
                ee.Authenticate()
                ee.Initialize()
            except Exception as final_e:
                raise final_e

        if tile_cache is not None:
            self.tile_cache = tile_cache
        else:
            root = tile_cache_dir if tile_cache_dir is not None else _TILE_CACHE_ROOT
            self.tile_cache = NdviTileCache(root, max_bytes=tile_cache_max_bytes)

    def prep_ndvi(self, img):
        scale = 0.0001
        red = img.select("B4").multiply(scale)
        nir = img.select("B8").multiply(scale)
        ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")

        # SCL masking (Standard in Harmonized collection)
        scl = img.select("SCL")
        mask_scl = (
            scl.neq(3)
            .And(scl.neq(7))
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )
        return ndvi.updateMask(mask_scl).copyProperties(img, img.propertyNames())

    def get_collection(self, aoi, start_date, end_date, cloud_max=10):
        return (
            ee.ImageCollection(self.EE_COLLECTION_ID)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
            .map(self.prep_ndvi)
        )

    def prep_ndvi_landsat(self, img):
        """NDVI from Landsat 8/9 Collection 2 Level-2 surface reflectance.

        Bands: ``SR_B4`` (red) / ``SR_B5`` (NIR). Apply the C2 L2 scale +
        offset (``DN * 0.0000275 - 0.2``) before computing the index, and
        mask cloud / cloud-shadow via the ``QA_PIXEL`` band so cloudy
        observations don't pollute the composite. Same output band name
        (``NDVI``) as :meth:`prep_ndvi` so downstream code is collection-
        agnostic.
        """
        scale = 0.0000275
        offset = -0.2
        red = img.select("SR_B4").multiply(scale).add(offset)
        nir = img.select("SR_B5").multiply(scale).add(offset)
        ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
        qa = img.select("QA_PIXEL")
        # Bit 3 = cloud, bit 4 = cloud shadow, bit 5 = snow.
        cloudy = (
            qa.bitwiseAnd(1 << 3).Or(qa.bitwiseAnd(1 << 4)).Or(qa.bitwiseAnd(1 << 5))
        )
        return ndvi.updateMask(cloudy.eq(0)).copyProperties(img, img.propertyNames())

    def get_collection_landsat(self, aoi, start_date, end_date, cloud_max=10):
        """Merged LC08 + LC09 collection with the same shape as
        :meth:`get_collection` for Sentinel-2."""
        lc8 = ee.ImageCollection(self.LANDSAT8_COLLECTION_ID).filterBounds(aoi)
        lc9 = ee.ImageCollection(self.LANDSAT9_COLLECTION_ID).filterBounds(aoi)
        merged = lc8.merge(lc9)
        return (
            merged.filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUD_COVER", cloud_max))
            .map(self.prep_ndvi_landsat)
        )

    def _pick_satellite(self, start_date: str, end_date: str, satellite: str) -> str:
        """Resolve ``satellite='auto'`` → ``'sentinel2'`` or ``'landsat'``.

        ``end_date`` falling before the Sentinel-2 collection start picks
        Landsat; everything else stays on Sentinel-2 (higher resolution +
        more frequent revisit). Explicit ``'sentinel2'`` / ``'landsat'`` are
        passed through untouched.
        """
        if satellite != "auto":
            return satellite
        return "landsat" if str(end_date) < self.S2_COLLECTION_START else "sentinel2"

    #: Block-iteration buffer cap. Points accumulate here across raster
    #: windows; once we cross this threshold we build a GeoDataFrame and
    #: append it to the GeoPackage, then reset. Keeps peak memory bounded
    #: while batching enough features that GPKG open/append overhead doesn't
    #: dominate
    _VECTOR_FLUSH_POINTS = 500_000

    def _raster_to_ndvi_points(
        self,
        final_tif: str,
        geometry,
        *,
        geojson_path: str | None,
        gpkg_path: str | None,
        ndvi_progress_callback: Callable[[Mapping[str, Any]], None] | None,
        cancel_callback: Callable[[], bool] | None,
        sub_start: float,
        sub_end: float,
        meta_extra: dict[str, Any] | None = None,
    ) -> dict:
        """Stream raster blocks → points, append to GeoPackage in chunks.

        Walks the raster's native block grid (typically 256×256 tiles for
        our outputs) instead of slurping the whole band into RAM. Each
        valid-pixel chunk gets buffered until the buffer hits
        ``_VECTOR_FLUSH_POINTS``, then a small GeoDataFrame is built,
        clipped to the study geometry, and appended to the GeoPackage.
        Keeps peak memory usage low.

        GeoJSON output (when requested) is materialised at the end from the
        GeoPackage — it has no streaming-append story of its own, but
        anyone asking for GeoJSON at this scale has already been warned
        in the UI that it's a compatibility-only path.
        """
        if cancel_callback and cancel_callback():
            return {"status": "cancelled", "message": "Cancelled by user"}

        _emit_ndvi_progress(
            ndvi_progress_callback,
            sub_progress=sub_start,
            phase="Writing vector samples",
        )

        # Prepare the clip union geometry once (in 4326) so per-chunk filter
        # is a single shapely predicate rather than a full ``gpd.clip``.
        clip_geom = None
        if isinstance(geometry, gpd.GeoDataFrame) and not geometry.empty:
            clip_gdf = (
                geometry
                if geometry.crs == "EPSG:4326"
                else geometry.to_crs("EPSG:4326")
            )
            try:
                clip_geom = clip_gdf.geometry.union_all()
                if clip_geom is not None and clip_geom.is_empty:
                    clip_geom = None
            except Exception:
                clip_geom = None

        layer_name = "ndvi_samples"
        # Always stream into the GeoPackage when one was requested. When the
        # user only wants GeoJSON, route through a temp GPKG next door (same
        # streaming benefit) and convert at the end.
        if gpkg_path:
            stream_target = gpkg_path
            stream_target_is_temp = False
        elif geojson_path:
            stream_target = geojson_path + ".tmp.gpkg"
            stream_target_is_temp = True
        else:
            # Neither vector output requested — the caller should not have
            # called us. Treat as a no-op success.
            meta = {"crs": "EPSG:4326"}
            if meta_extra:
                meta.update(meta_extra)
            return {
                "status": "success",
                "tif": final_tif,
                "geojson": None,
                "gpkg": None,
                "meta": meta,
            }

        # Working buffers (Python lists — appending is amortised O(1) and
        # numpy conversion happens once per flush).
        buf_x: list[float] = []
        buf_y: list[float] = []
        buf_v: list[float] = []
        n_written = 0
        first_chunk = True

        try:
            with rasterio.open(final_tif) as src:
                nodata = src.nodata if src.nodata is not None else -9999
                blocks = list(src.block_windows(1))
                n_blocks = max(1, len(blocks))

                def flush() -> None:
                    nonlocal first_chunk, n_written
                    if not buf_v:
                        return
                    gdf_chunk = gpd.GeoDataFrame(
                        {"NDVI": np.round(np.asarray(buf_v, dtype=np.float32), 4)},
                        geometry=gpd.points_from_xy(
                            np.round(np.asarray(buf_x), 5),
                            np.round(np.asarray(buf_y), 5),
                        ),
                        crs="EPSG:4326",
                    )
                    if clip_geom is not None:
                        gdf_chunk = gdf_chunk[gdf_chunk.geometry.intersects(clip_geom)]
                    if not gdf_chunk.empty:
                        mode = "w" if first_chunk else "a"
                        gdf_chunk.to_file(
                            stream_target,
                            driver="GPKG",
                            layer=layer_name,
                            mode=mode,
                        )
                        first_chunk = False
                        n_written += len(gdf_chunk)
                    buf_x.clear()
                    buf_y.clear()
                    buf_v.clear()

                span = max(sub_end - sub_start, 0.0)
                for bi, (_block_idx, window) in enumerate(blocks):
                    if cancel_callback and cancel_callback():
                        return {"status": "cancelled", "message": "Cancelled by user"}

                    arr = src.read(1, window=window)
                    # Treat both NaN and the EE nodata sentinel as missing.
                    if np.issubdtype(arr.dtype, np.floating):
                        valid = (arr != nodata) & ~np.isnan(arr)
                    else:
                        valid = arr != nodata
                    if not valid.any():
                        _emit_ndvi_progress(
                            ndvi_progress_callback,
                            sub_progress=sub_start + span * ((bi + 1) / n_blocks),
                            phase="Writing vector samples",
                        )
                        continue

                    rows_local, cols_local = np.nonzero(valid)
                    rows_global = rows_local + int(window.row_off)
                    cols_global = cols_local + int(window.col_off)
                    xs, ys = rasterio.transform.xy(
                        src.transform,
                        rows_global.tolist(),
                        cols_global.tolist(),
                        offset="center",
                    )
                    buf_x.extend(xs if isinstance(xs, list) else [xs])
                    buf_y.extend(ys if isinstance(ys, list) else [ys])
                    buf_v.extend(arr[valid].tolist())

                    if len(buf_v) >= self._VECTOR_FLUSH_POINTS:
                        flush()

                    _emit_ndvi_progress(
                        ndvi_progress_callback,
                        sub_progress=sub_start + span * ((bi + 1) / n_blocks),
                        phase="Writing vector samples",
                    )

                flush()

            if n_written == 0:
                # Clean up the empty temp GPKG before reporting empty raster.
                if stream_target_is_temp and os.path.exists(stream_target):
                    try:
                        os.remove(stream_target)
                    except OSError:
                        pass
                return {"status": "error", "message": "Raster is empty."}

            # GeoJSON convert path: read the streamed GPKG back and dump.
            # Acceptable for the UI's small / medium GeoJSON use case; the
            # checkbox is documented as compatibility-only.
            if geojson_path:
                try:
                    gdf_full = gpd.read_file(stream_target, layer=layer_name)
                    gdf_full.to_file(geojson_path, driver="GeoJSON")
                except Exception as e:
                    _log("WARN", f"GeoJSON conversion failed: {e}")
            if stream_target_is_temp and os.path.exists(stream_target):
                try:
                    os.remove(stream_target)
                except OSError:
                    pass

            _emit_ndvi_progress(
                ndvi_progress_callback,
                sub_progress=sub_end,
                phase="Writing vector samples",
            )

            meta: dict[str, Any] = {
                "crs": "EPSG:4326",
                "vector_points": n_written,
            }
            if meta_extra:
                meta.update(meta_extra)
            return {
                "status": "success",
                "tif": final_tif,
                "geojson": geojson_path,
                "gpkg": gpkg_path,
                "meta": meta,
            }

        except Exception as e:
            return {"status": "error", "message": f"Pixel Extraction Failed: {str(e)}"}

    def download_and_process(
        self,
        geometry,
        start_date,
        end_date,
        output_name,
        cloud_max=10,
        resolution=10,
        folder="output_results",
        max_tile_size_km=5,
        cancel_callback: Callable[[], bool] | None = None,
        ndvi_progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        write_geotiff: bool = True,
        write_geojson: bool = True,
        write_geopackage: bool = False,
        write_cluster_tiles: bool = False,
        satellite: str = "auto",
        coverage_rescue: bool = True,
        sample_at_features: gpd.GeoDataFrame | None = None,
        sample_radius_m: float = 0.0,
        sample_stat: str = "mean",
    ):
        """
        Download and process NDVI data with automatic tiling for large areas.

        Args:
            geometry: GeoDataFrame or Shapely geometry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_name: Base name for output files
            cloud_max: Maximum cloud percentage
            resolution: Pixel resolution in meters
            folder: Output directory
            max_tile_size_km: Maximum tile dimension in kilometers (prevents GEE size limit)
            cancel_callback: If provided, return True to stop between tiles / major steps.
            ndvi_progress_callback: Receives a mapping with optional keys:
                ``sub_progress`` (0–1 for this download), ``phase`` (status line),
                ``tiles`` ``(k, n)`` while downloading tiles (shows ``[k/n]`` on the bar),
                ``clear_bracket`` (drop tile bracket after tiles are done).
            write_geotiff: Persist ``{output_name}_ndvi.tif`` when True.
            write_geojson: Persist ``{output_name}_ndvi.geojson`` when True (needs raster).
            write_geopackage: Persist ``{output_name}_ndvi.gpkg`` when True (needs raster).
            write_cluster_tiles: For multi-cluster inputs, mosaic each cluster
                into its own ``{output_name}_ndvi_tiles/cluster_NNNN.tif`` plus
                a ``tiles_index.json``. Mirrors GVI's per-cluster GeoTIFF tile
                output and avoids the giant mostly-NaN mosaic that scattered
                national-scale inputs would otherwise produce.
            satellite: ``'auto'`` (default — Sentinel-2 for ≥2017, Landsat 8/9
                otherwise), ``'sentinel2'``, or ``'landsat'``. Auto picks
                Landsat for date ranges that end before
                :attr:`S2_COLLECTION_START`.
            coverage_rescue: When the cloud-filtered collection has fewer
                than 3 images, widen the date range by ±50 % once and retry.
                Documented in the sidecar so the user knows the composite
                spans a wider window than they originally asked for.
        """
        os.makedirs(folder, exist_ok=True)

        if not (
            write_geotiff or write_geojson or write_geopackage or write_cluster_tiles
        ):
            return {
                "status": "error",
                "message": "Enable at least one output format (GeoTIFF, GeoPackage, GeoJSON, or cluster tiles).",
            }

        # 1. Convert Geometry — reproject_geodataframe_to_wgs84 raises on
        # malformed inputs (geographic CRS metadata + metre-valued coords) so
        # direct API callers get the same guard the UI runner gets.
        if isinstance(geometry, gpd.GeoDataFrame):
            geom_wgs84 = reproject_geodataframe_to_wgs84(geometry)
            js = json.loads(geom_wgs84.to_json())
            js.pop("crs", None)
            aoi = ee.FeatureCollection(js["features"]).geometry()
            geom_for_crs = geom_wgs84
        else:
            aoi = _shapely_to_ee_geometry(geometry)
            geom_for_crs = gpd.GeoDataFrame({"geometry": [geometry]}, crs="EPSG:4326")

        # 2. Pick an EE export CRS so pixels are rasterised in true ground
        # metres regardless of latitude (vs. the legacy hardcoded Web Mercator,
        # which doubles cell area near 60°N).
        export_crs, grid_crs, export_distortion, export_crs_name = _select_export_crs(
            geom_for_crs
        )

        # Global snap grid for every EE export in this run. Anchoring the
        # transform at (0, 0) in the planar CRS means adjacent tiles share
        # one pixel grid — boundaries align to the pixel and the mosaic step
        # joins them with zero sub-pixel drift (the root cause of the
        # tile-gap / tile-shift artefacts the WGS84-per-tile path used to
        # produce). ``crs_transform`` is the standard GDAL affine
        # ``[a, b, c, d, e, f]`` form: ``[scale, 0, x0, 0, -scale, y0]``.
        crs_transform = [float(resolution), 0.0, 0.0, 0.0, -float(resolution), 0.0]

        # 3. Pick the satellite collection. ``auto`` falls back to Landsat
        # for ranges that end before Sentinel-2 SR_HARMONIZED started.
        chosen_satellite = self._pick_satellite(
            str(start_date), str(end_date), satellite
        )
        if chosen_satellite == "landsat":
            ee_collection_id = (
                f"{self.LANDSAT8_COLLECTION_ID}+{self.LANDSAT9_COLLECTION_ID}"
            )

            def _get_col(s, e, cmax):
                return self.get_collection_landsat(aoi, s, e, cmax)

            def _get_base(s, e):
                lc8 = ee.ImageCollection(self.LANDSAT8_COLLECTION_ID).filterBounds(aoi)
                lc9 = ee.ImageCollection(self.LANDSAT9_COLLECTION_ID).filterBounds(aoi)
                return lc8.merge(lc9).filterDate(s, e)

            cloud_field = "CLOUD_COVER"
        else:
            ee_collection_id = self.EE_COLLECTION_ID

            def _get_col(s, e, cmax):
                return self.get_collection(aoi, s, e, cmax)

            def _get_base(s, e):
                return (
                    ee.ImageCollection(self.EE_COLLECTION_ID)
                    .filterBounds(aoi)
                    .filterDate(s, e)
                )

            cloud_field = "CLOUDY_PIXEL_PERCENTAGE"

        _log(
            "INFO",
            f"Satellite: {chosen_satellite} (collection: {ee_collection_id})",
        )

        # 3a. Distinguish "no images in range" vs "all images filtered by
        # cloud threshold" so the user gets an actionable error instead of
        # the legacy "No images found." (which conflated the two).
        used_start, used_end = str(start_date), str(end_date)
        coverage_widened = False
        n_total = _get_base(used_start, used_end).size().getInfo()
        if n_total == 0:
            suggestion = ""
            if chosen_satellite == "sentinel2" and used_end < self.S2_COLLECTION_START:
                suggestion = (
                    f" Sentinel-2 SR_HARMONIZED starts {self.S2_COLLECTION_START}"
                    " — set satellite='landsat' (or 'auto') for earlier dates."
                )
            return {
                "status": "error",
                "message": (
                    f"No {chosen_satellite} images in collection for "
                    f"{used_start} → {used_end}.{suggestion}"
                ),
            }
        n_cloud_filtered = (
            _get_base(used_start, used_end)
            .filter(ee.Filter.lt(cloud_field, cloud_max))
            .size()
            .getInfo()
        )
        if n_cloud_filtered == 0:
            return {
                "status": "error",
                "message": (
                    f"All {n_total} {chosen_satellite} image(s) for "
                    f"{used_start} → {used_end} exceeded the {cloud_max}% cloud "
                    "threshold. Try raising the cloud max."
                ),
            }

        # 3b. Coverage rescue: if too few cloud-free images, widen the date
        # range by ±50 % once and re-query. Single attempt — anything more
        # aggressive belongs in user-controlled config.
        MIN_IMAGES_FOR_COMPOSITE = 3
        if (
            coverage_rescue
            and n_cloud_filtered < MIN_IMAGES_FOR_COMPOSITE
            and used_end >= used_start
        ):
            try:
                from datetime import date as _date
                from datetime import timedelta as _timedelta

                s_dt = _date.fromisoformat(used_start)
                e_dt = _date.fromisoformat(used_end)
                span = max((e_dt - s_dt).days, 1)
                pad = _timedelta(days=int(span * 0.5))
                w_start = (s_dt - pad).isoformat()
                w_end = (e_dt + pad).isoformat()
                n_widened = (
                    _get_base(w_start, w_end)
                    .filter(ee.Filter.lt(cloud_field, cloud_max))
                    .size()
                    .getInfo()
                )
                if n_widened > n_cloud_filtered:
                    _log(
                        "WARN",
                        f"Coverage rescue: only {n_cloud_filtered} cloud-free "
                        f"image(s) in {used_start}→{used_end}, widening to "
                        f"{w_start}→{w_end} ({n_widened} images).",
                    )
                    used_start, used_end = w_start, w_end
                    n_cloud_filtered = n_widened
                    coverage_widened = True
            except Exception as e:
                _log("WARN", f"Coverage rescue skipped: {e}")

        col = _get_col(used_start, used_end, cloud_max)

        if cancel_callback and cancel_callback():
            return {"status": "cancelled", "message": "Cancelled by user"}

        # 3c. Build a 2-band image: NDVI median + valid-obs count per pixel.
        # ``count()`` over the masked collection is the number of unmasked
        # (i.e. cloud-free, in-collection) observations at each pixel; The
        # right "did this pixel get enough samples?" signal for downstream
        # consumers. ``toUint16()`` keeps the band small. Each band gets its
        # own nodata sentinel (NDVI: −9999 in float32, valid_obs: 0 in
        # uint16) so a per-tile ``unmask(-9999)`` later wouldn't saturate the
        # integer band.
        ndvi_band = col.select("NDVI").median().rename("NDVI").unmask(-9999)
        obs_band = col.select("NDVI").count().rename("valid_obs").toUint16().unmask(0)
        ndvi_median = ndvi_band.addBands(obs_band).clip(aoi)

        # 4. Build cluster-aware tile list in true metres. Scattered national
        # inputs decompose into connected components, and tiles that fall over
        # empty bbox regions (ocean, gaps between provinces) are STRtree-culled
        # before they reach Earth Engine.
        tiles = build_planar_tiles(geom_for_crs, grid_crs, max_tile_size_km)
        n_tiles = len(tiles)
        if n_tiles == 0:
            return {
                "status": "error",
                "message": "No tiles cover the study area — check input geometry.",
            }

        n_clusters = len({t["cluster_id"] for t in tiles})

        # 5. Compute the resume key from the *effective* run parameters
        # (post coverage-rescue date range + post satellite-pick collection).
        # Anchoring the key on what was actually downloaded means a future
        # run that hits the same widened window / Landsat fallback reuses
        # the cache; runs that resolve differently get isolated workspaces.
        resume_key = _compute_resume_key(
            geom_for_crs,
            used_start,
            used_end,
            cloud_max,
            resolution,
            ee_collection_id,
        )

        if n_tiles == 1:
            _log(
                "INFO",
                f"Single-tile download ({n_clusters} cluster(s), planar metres).",
            )
            result = self._download_single(
                ndvi_median,
                aoi,
                geometry,
                output_name,
                resolution,
                folder,
                export_crs=export_crs,
                export_distortion=export_distortion,
                export_crs_name=export_crs_name,
                crs_transform=crs_transform,
                cancel_callback=cancel_callback,
                ndvi_progress_callback=ndvi_progress_callback,
                write_geotiff=write_geotiff,
                write_geojson=write_geojson,
                write_geopackage=write_geopackage,
            )
        else:
            _log(
                "INFO",
                f"Tiled download: {n_tiles} tiles across {n_clusters} cluster(s) "
                f"(max {max_tile_size_km} km/tile).",
            )
            result = self._download_with_tiling(
                ndvi_median,
                aoi,
                tiles,
                output_name,
                resolution,
                folder,
                max_tile_size_km,
                export_crs=export_crs,
                export_distortion=export_distortion,
                export_crs_name=export_crs_name,
                crs_transform=crs_transform,
                n_clusters=n_clusters,
                resume_key=resume_key,
                cancel_callback=cancel_callback,
                ndvi_progress_callback=ndvi_progress_callback,
                write_geotiff=write_geotiff,
                write_geojson=write_geojson,
                write_geopackage=write_geopackage,
                write_cluster_tiles=write_cluster_tiles,
            )

        # Sidecar: write on success so direct API callers (CLI / fusion auto-
        # download / notebooks) get the same parameter+CRS audit trail the GVI
        # ``_gvi.json`` provides. Failures and cancellations skip this — the
        # sidecar should only describe outputs that actually landed on disk.
        if result.get("status") == "success":
            raw_meta = result.get("meta")
            result_meta: dict = raw_meta if isinstance(raw_meta, dict) else {}
            tiles_succeeded = int(result_meta.get("tiles", n_tiles))
            tiles_failed = int(result_meta.get("tiles_failed", 0))
            tiles_resumed = int(result_meta.get("tiles_resumed", 0))
            failed_refs = result_meta.get("failed_tile_refs") or []
            try:
                sidecar_path = _write_ndvi_sidecar(
                    folder,
                    output_name,
                    grid_crs=grid_crs,
                    export_crs=export_crs,
                    export_crs_name=export_crs_name,
                    export_distortion=export_distortion,
                    n_clusters=n_clusters,
                    tiles_total=n_tiles,
                    tiles_succeeded=tiles_succeeded,
                    tiles_failed=tiles_failed,
                    tiles_resumed=tiles_resumed,
                    failed_tile_refs=failed_refs,
                    resume_key=resume_key,
                    start_date=str(start_date),
                    end_date=str(end_date),
                    used_start_date=used_start,
                    used_end_date=used_end,
                    coverage_widened=coverage_widened,
                    cloud_max=cloud_max,
                    resolution_m=resolution,
                    max_tile_size_km=max_tile_size_km,
                    ee_collection=ee_collection_id,
                    satellite=chosen_satellite,
                    bands=["NDVI", "valid_obs"],
                    n_cloud_filtered_images=n_cloud_filtered,
                )
                result["sidecar"] = sidecar_path
            except Exception as e:
                _log("WARN", f"Sidecar write failed (non-fatal): {e}")

            # Sample-at-features (Step 15): attach NDVI values to the
            # caller's features and write a side GeoPackage. Skipped when
            # the user didn't ask for it or the raster wasn't written. The
            # source raster is the final NDVI mosaic; Read once, sampled
            # per-feature via the shared helper.
            if sample_at_features is not None and not sample_at_features.empty:
                final_tif = os.path.join(folder, f"{output_name}_ndvi.tif")
                if os.path.exists(final_tif):
                    samples_path = os.path.join(
                        folder, f"{output_name}_ndvi_at_features.gpkg"
                    )
                    try:
                        n = sample_raster_at_features_to_file(
                            final_tif,
                            sample_at_features,
                            samples_path,
                            band=1,
                            radius_m=float(sample_radius_m or 0.0),
                            stat=sample_stat,
                            value_column="NDVI",
                            count_column="ndvi_obs",
                        )
                        result["features_samples"] = samples_path
                        _log(
                            "OK",
                            f"Sampled NDVI at {n} feature(s) → "
                            f"{samples_path} (radius={sample_radius_m} m, "
                            f"stat={sample_stat}).",
                        )
                    except Exception as e:
                        _log("WARN", f"Sample-at-features failed: {e}")
                else:
                    _log(
                        "WARN",
                        "Sample-at-features requested but the NDVI raster "
                        "wasn't kept on disk; enable write_geotiff or skip "
                        "this option.",
                    )
        return result

    def _download_single(
        self,
        ndvi_median,
        aoi,
        geometry,
        output_name,
        resolution,
        folder,
        *,
        export_crs: str = "EPSG:3857",
        export_distortion: float = 0.0,
        export_crs_name: str = "Web Mercator (legacy)",
        crs_transform: list[float] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
        ndvi_progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        write_geotiff: bool = True,
        write_geojson: bool = True,
        write_geopackage: bool = False,
    ):
        """Download NDVI as a single tile (for small areas)."""
        _emit_ndvi_progress(
            ndvi_progress_callback,
            sub_progress=0.05,
            phase="Downloading from Earth Engine",
            tiles=(0, 1),
        )
        if cancel_callback and cancel_callback():
            return {"status": "cancelled", "message": "Cancelled by user"}
        temp_tif = os.path.join(folder, f"temp_{output_name}.tif")
        final_tif = os.path.join(folder, f"{output_name}_ndvi.tif")

        try:

            def _do_export() -> None:
                # EE export goes straight into the planar temp file; the
                # single final reproject below turns it into the WGS84
                # user-visible output. ``crs_transform`` is passed for
                # symmetry with the tiled path even though there's only one
                # tile here — it keeps the EE call shape identical between paths.
                geemap.ee_export_image(
                    # Per-band nodata already applied in download_and_process
                    # so the uint16 valid_obs band isn't clobbered with -9999.
                    ndvi_median,
                    filename=temp_tif,
                    crs=export_crs,
                    crs_transform=crs_transform,
                    region=aoi,
                    file_per_band=False,
                )

            # Same retry policy as the tiled path so a flaky network doesn't
            # blow up the single small-area run on the first transient.
            retry_with_backoff(
                _do_export,
                attempts=3,
                base_delay=2.0,
                cancel_callback=cancel_callback,
                log_fn=_log,
                label="EE export",
            )

            if cancel_callback and cancel_callback():
                if os.path.exists(temp_tif):
                    os.remove(temp_tif)
                return {"status": "cancelled", "message": "Cancelled by user"}

            _emit_ndvi_progress(
                ndvi_progress_callback,
                sub_progress=0.28,
                phase="Downloading from Earth Engine",
                tiles=(1, 1),
            )
            _emit_ndvi_progress(
                ndvi_progress_callback,
                sub_progress=0.32,
                phase="Reprojecting GeoTIFF",
                clear_bracket=True,
            )

            # Final reproject: planar → WGS84 with explicit nodata so the
            # ``-9999`` fill doesn't bleed into edge pixels. Builds overviews
            # for fast Folium previews.
            reproject_raster_to_wgs84(
                temp_tif,
                final_tif,
                target_resolution_m=resolution,
                build_overviews=True,
                src_nodata=-9999,
                dst_nodata=-9999,
            )

            if os.path.exists(temp_tif):
                os.remove(temp_tif)

        except Exception as e:
            if os.path.exists(temp_tif):
                os.remove(temp_tif)
            return {"status": "error", "message": f"Export/Reproject Failed: {str(e)}"}

        if cancel_callback and cancel_callback():
            return {"status": "cancelled", "message": "Cancelled by user"}

        try:
            with rasterio.open(final_tif) as src:
                crs_meta = str(src.crs)
        except Exception:
            crs_meta = "EPSG:4326"
        meta: dict[str, Any] = {
            "crs": crs_meta,
            "tiles": 1,
            "tiles_total": 1,
            "tiles_failed": 0,
            "tiles_resumed": 0,
            "failed_tile_refs": [],
            "n_clusters": 1,
            "cluster_tiles_dir": None,
            "cluster_tiles_written": 0,
            "export_crs": export_crs,
            "export_crs_name": export_crs_name,
            "export_distortion": export_distortion,
        }

        if not (write_geojson or write_geopackage):
            _emit_ndvi_progress(
                ndvi_progress_callback,
                sub_progress=0.99,
                phase="Finalizing",
            )
            return {
                "status": "success",
                "tif": final_tif if write_geotiff else None,
                "geojson": None,
                "gpkg": None,
                "meta": meta,
            }

        geojson_path = (
            os.path.join(folder, f"{output_name}_ndvi.geojson")
            if write_geojson
            else None
        )
        gpkg_path = (
            os.path.join(folder, f"{output_name}_ndvi.gpkg")
            if write_geopackage
            else None
        )
        out = self._raster_to_ndvi_points(
            final_tif,
            geometry,
            geojson_path=geojson_path,
            gpkg_path=gpkg_path,
            ndvi_progress_callback=ndvi_progress_callback,
            cancel_callback=cancel_callback,
            sub_start=0.52,
            sub_end=0.99,
            meta_extra={
                "tiles": 1,
                "tiles_total": 1,
                "tiles_failed": 0,
                "tiles_resumed": 0,
                "failed_tile_refs": [],
                "n_clusters": 1,
                "cluster_tiles_dir": None,
                "cluster_tiles_written": 0,
                "export_crs": export_crs,
                "export_crs_name": export_crs_name,
                "export_distortion": export_distortion,
            },
        )
        if out.get("status") != "success":
            return out
        if not write_geotiff and os.path.isfile(final_tif):
            os.remove(final_tif)
            out["tif"] = None
        elif write_geotiff:
            out["tif"] = final_tif
        return out

    def _write_per_cluster_outputs(
        self,
        work_dir: str,
        tiles: list[dict],
        failed_tile_refs: list[dict],
        output_root: str,
        export_crs: str,
        export_crs_name: str,
        resume_key: str,
        resolution: int,
    ) -> list[dict]:
        """Mosaic cached tiles per cluster and write a ``tiles_index.json``.

        Same two-step shape as the main mosaic: per-cluster planar mosaic
        into a scratch file in ``work_dir``, then a single reproject to
        WGS84 with explicit nodata. Keeps boundaries aligned to the pixel
        and avoids the ``-9999``-bleed edge artefacts.

        Failed tiles in ``failed_tile_refs`` are skipped. Per-cluster
        failures log a WARN and continue — the rest of the clusters still
        land. Returns the per-cluster index entries (also written to disk).
        """
        from collections import defaultdict

        os.makedirs(output_root, exist_ok=True)

        cluster_to_paths: dict[int, list[str]] = defaultdict(list)
        failed_pairs = {
            (int(r["cluster_id"]), int(r["tile_idx"])) for r in failed_tile_refs
        }
        for ts in tiles:
            cid = int(ts["cluster_id"])
            tidx = int(ts["tile_idx"])
            if (cid, tidx) in failed_pairs:
                continue
            p = os.path.join(work_dir, f"tile_{tidx}.tif")
            if os.path.isfile(p):
                cluster_to_paths[cid].append(p)

        entries: list[dict] = []
        for cid in sorted(cluster_to_paths):
            paths = cluster_to_paths[cid]
            if not paths:
                continue
            cluster_path = os.path.join(output_root, f"cluster_{cid:04d}.tif")
            planar_tmp = os.path.join(work_dir, f"_cluster_{cid:04d}_planar.tif")
            try:
                stream_mosaic_to_geotiff(
                    paths, planar_tmp, nodata=-9999, build_overviews=False
                )
                reproject_raster_to_wgs84(
                    planar_tmp,
                    cluster_path,
                    target_resolution_m=resolution,
                    build_overviews=True,
                    src_nodata=-9999,
                    dst_nodata=-9999,
                )
            except Exception as e:
                _log("WARN", f"Cluster {cid} mosaic failed: {e}")
                continue
            finally:
                if os.path.exists(planar_tmp):
                    try:
                        os.remove(planar_tmp)
                    except OSError:
                        pass
            try:
                with rasterio.open(cluster_path) as src:
                    b = src.bounds
                    bounds_4326 = [
                        float(b.left),
                        float(b.bottom),
                        float(b.right),
                        float(b.top),
                    ]
            except Exception:
                bounds_4326 = None
            entries.append(
                {
                    "cluster_id": cid,
                    "path": os.path.basename(cluster_path),
                    "bounds_4326": bounds_4326,
                    "n_tiles": len(paths),
                }
            )

        index_path = os.path.join(output_root, "tiles_index.json")
        with open(index_path, "w") as f:
            json.dump(
                {
                    "crs": "EPSG:4326",
                    "export_crs": export_crs,
                    "export_crs_name": export_crs_name,
                    "resume_key": resume_key,
                    "n_clusters": len(entries),
                    "clusters": entries,
                },
                f,
                indent=2,
            )
        return entries

    def _download_one_tile(
        self,
        tile_spec: dict,
        ndvi_median,
        work_dir: str,
        resolution: int,
        export_crs: str,
        crs_transform: list[float],
        cancel_callback: Callable[[], bool] | None,
    ) -> dict:
        """Worker: one tile downloaded from EE straight into the planar cache.

        Runs on a ThreadPoolExecutor thread (see :meth:`_download_with_tiling`).
        Returns ``{success, tile_final, idx, error}``; failures surface as
        ``success=False`` rather than raising so a single bad tile can't sink
        the whole batch.

        Tiles stay in the engine's chosen planar CRS — *no* per-tile reproject
        to WGS84 — so all tiles share one snap grid (``crs_transform`` fixes
        the origin globally) and the mosaic step joins them with zero
        sub-pixel drift. A single final reproject converts the merged planar
        mosaic to WGS84 with explicit nodata.

        Cancel checks short-circuit at safe boundaries — Earth Engine's HTTP
        call is one blocking step that can't be killed mid-flight, so a
        freshly-cancelled run may still have a few in-flight downloads finish
        and be discarded by the caller.
        """
        idx = int(tile_spec["tile_idx"])
        tile_geom = tile_spec["tile_geom_4326"]
        result: dict = {"success": False, "tile_final": None, "idx": idx, "error": None}

        if cancel_callback and cancel_callback():
            return result

        tile_final = os.path.join(work_dir, f"tile_{idx}.tif")

        try:
            tile_aoi = _shapely_to_ee_geometry(tile_geom)
            tile_ndvi = ndvi_median.clip(tile_aoi)

            def _do_export() -> None:
                # Save the EE export *directly* as the final cache tile —
                # no intermediate temp file, no per-tile WGS84 reproject.
                # ``crs_transform`` snaps every tile to one global grid in
                # the planar CRS so adjacent tiles' boundaries align to the
                # pixel.
                geemap.ee_export_image(
                    # Per-band nodata already baked in (see download_and_process).
                    tile_ndvi,
                    filename=tile_final,
                    crs=export_crs,
                    crs_transform=crs_transform,
                    region=tile_aoi,
                    file_per_band=False,
                )

            # 3 attempts with 2 s base delay, doubling each time (≈ 2 s, 4 s
            # between retries before jitter). Flaky network = single failed
            # tile, not a swiss-cheese mosaic.
            retry_with_backoff(
                _do_export,
                attempts=3,
                base_delay=2.0,
                cancel_callback=cancel_callback,
                log_fn=_log,
                label=f"Tile {idx + 1} EE export",
            )

            if cancel_callback and cancel_callback():
                if os.path.exists(tile_final):
                    os.remove(tile_final)
                return result

            result["success"] = True
            result["tile_final"] = tile_final
            return result
        except Exception as e:
            if os.path.exists(tile_final):
                try:
                    os.remove(tile_final)
                except OSError:
                    pass
            result["error"] = f"{type(e).__name__}: {e}"
            return result

    def _download_with_tiling(
        self,
        ndvi_median,
        aoi,
        tiles: list[dict],
        output_name,
        resolution,
        folder,
        max_tile_size_km,
        *,
        export_crs: str = "EPSG:3857",
        export_distortion: float = 0.0,
        export_crs_name: str = "Web Mercator (legacy)",
        crs_transform: list[float] | None = None,
        n_clusters: int = 1,
        resume_key: str = "",
        cancel_callback: Callable[[], bool] | None = None,
        ndvi_progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        write_geotiff: bool = True,
        write_geojson: bool = True,
        write_geopackage: bool = False,
        write_cluster_tiles: bool = False,
    ):
        """Download NDVI for a precomputed cluster-aware tile list and mosaic.

        ``tiles`` is the output of :func:`geofuse.core.build_planar_tiles` — a list of
        ``{cluster_id, tile_idx, tile_geom_4326}`` dicts. Each tile is exported
        independently and mosaicked into one GeoTIFF; per-cluster GeoTIFF tile
        outputs are a separate downstream step (Phase 4).

        ``resume_key`` selects the tile workspace; tiles already present from
        a previous interrupted run with the same key are skipped.
        """
        n_tiles = len(tiles)
        _log(
            "INFO",
            f"Downloading {n_tiles} tile(s) across {n_clusters} cluster(s) "
            f"(max {max_tile_size_km} km/tile)...",
        )

        _emit_ndvi_progress(
            ndvi_progress_callback,
            sub_progress=0.02,
            phase="Downloading tiles",
            tiles=(0, n_tiles),
        )

        # The cache directory doubles as both the in-progress workspace
        # *and* the cross-run cache: successful runs leave their tiles in
        # place so a future run with the same ``resume_key`` (same area +
        # date range + cloud max + resolution + collection) reuses every
        # tile instantly. The cache's LRU + size cap reclaims old entries
        # when it grows past the configured ceiling.
        work_dir = self.tile_cache.workspace_dir(resume_key or "scratch")

        # Resume: any tile already present on disk (and openable as a
        # rasterio dataset) is reused as-is; only the remainder gets queued
        # for download. Same lookup whether we're resuming an interrupted
        # run or hitting a previously-cached one — only the user-facing log
        # message differs.
        existing_tile_files, pending_tiles = _discover_resumable_tiles(work_dir, tiles)
        n_resumed = len(existing_tile_files)
        if n_resumed:
            _log(
                "INFO",
                f"Resumed {n_resumed}/{n_tiles} tile(s) from previous run "
                f"({work_dir}).",
            )

        tile_files: list[str] = list(existing_tile_files)
        # Tiles that exhausted the retry budget. Recorded with their
        # ``cluster_id`` / ``tile_idx`` / final error so the sidecar can
        # surface exactly which areas are NaN gaps in the mosaic.
        failed_tile_refs: list[dict] = []
        mosaic_ok = False
        final_tif = os.path.join(folder, f"{output_name}_ndvi.tif")

        try:
            # Parallel tile downloads. Each worker handles one tile end-to-end
            # (Earth Engine export → reproject to WGS84). The ThreadPoolExecutor
            # caps concurrency at ``_MAX_CONCURRENT_TILES`` so we don't saturate
            # local sockets or oversubscribe EE per-user. Heartbeat emits are
            # throttled by the shared :func:`progress_interval_s` so the UI
            # bracket stays responsive without slamming the JobStore lock on
            # big runs.
            done_count = n_resumed
            last_emit_t = {"v": time.monotonic()}
            interval_s = progress_interval_s(n_tiles)

            def _emit_progress(k: int) -> None:
                now = time.monotonic()
                is_final = k >= n_tiles
                if not is_final and now - last_emit_t["v"] < interval_s:
                    return
                last_emit_t["v"] = now
                _emit_ndvi_progress(
                    ndvi_progress_callback,
                    sub_progress=0.70 * k / n_tiles if n_tiles else 0.0,
                    phase="Downloading tiles",
                    tiles=(k, n_tiles),
                )

            # Show the resumed count immediately so the bracket jumps to the
            # right starting point instead of flashing 0/N.
            _emit_progress(n_resumed)

            with ThreadPoolExecutor(
                max_workers=_MAX_CONCURRENT_TILES,
                thread_name_prefix="ndvi-tile",
            ) as pool:
                futures = {
                    pool.submit(
                        self._download_one_tile,
                        ts,
                        ndvi_median,
                        work_dir,
                        resolution,
                        export_crs,
                        crs_transform,
                        cancel_callback,
                    ): ts
                    for ts in pending_tiles
                }
                cancelled = False
                for fut in as_completed(futures):
                    res = fut.result()
                    done_count += 1
                    if res["success"] and res["tile_final"] is not None:
                        tile_files.append(res["tile_final"])
                    elif res.get("error"):
                        tile_spec = futures[fut]
                        failed_tile_refs.append(
                            {
                                "cluster_id": int(tile_spec["cluster_id"]),
                                "tile_idx": int(res["idx"]),
                                "error": str(res["error"]),
                            }
                        )
                        _log(
                            "WARN",
                            f"Tile {res['idx'] + 1} failed after retry: "
                            f"{res['error']}",
                        )
                    _emit_progress(done_count)
                    if cancel_callback and cancel_callback():
                        cancelled = True
                        # Cancel any not-yet-running futures; in-flight tiles
                        # finish on their own (EE export is one blocking call,
                        # we can't kill it mid-flight) and their results are
                        # discarded.
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break

            if cancelled or (cancel_callback and cancel_callback()):
                return {"status": "cancelled", "message": "Cancelled by user"}

            if not tile_files:
                return {"status": "error", "message": "All tiles failed to download"}

            if failed_tile_refs:
                _log(
                    "WARN",
                    f"{len(failed_tile_refs)}/{n_tiles} tile(s) failed after retry "
                    "— mosaic will have NaN gaps in those areas. "
                    "See sidecar JSON for the cluster/tile IDs.",
                )

            _log(
                "OK",
                f"Successfully downloaded {len(tile_files)}/{n_tiles} tiles. "
                "Mosaicking...",
            )

            if cancel_callback and cancel_callback():
                return {"status": "cancelled", "message": "Cancelled by user"}

            _emit_ndvi_progress(
                ndvi_progress_callback,
                sub_progress=0.72,
                phase="Mosaicking rasters",
                tiles=(0, len(tile_files)),
                clear_bracket=True,
            )

            # Two-step output: (1) stream-mosaic the per-tile rasters in
            # their shared **planar** CRS, then (2) reproject the merged
            # mosaic to WGS84 once. This keeps tile boundaries aligned to
            # the pixel.
            planar_mosaic_tif = os.path.join(work_dir, "_mosaic_planar.tif")
            try:

                def _mosaic_progress(k: int, n: int) -> None:
                    span = 0.76 - 0.72
                    _emit_ndvi_progress(
                        ndvi_progress_callback,
                        sub_progress=0.72 + (span * k / n if n else 0.0),
                        phase="Mosaicking rasters",
                        tiles=(k, n),
                    )

                # Intermediate planar mosaic — no overviews, they'd be
                # discarded by the WGS84 reproject below.
                stream_mosaic_to_geotiff(
                    tile_files,
                    planar_mosaic_tif,
                    nodata=-9999,
                    progress_cb=_mosaic_progress,
                    build_overviews=False,
                )

                _emit_ndvi_progress(
                    ndvi_progress_callback,
                    sub_progress=0.76,
                    phase="Reprojecting to WGS84",
                    clear_bracket=True,
                )

                # Single final reproject with explicit nodata so the
                # ``-9999`` fill never bleeds into edge pixels.
                reproject_raster_to_wgs84(
                    planar_mosaic_tif,
                    final_tif,
                    target_resolution_m=resolution,
                    build_overviews=True,
                    src_nodata=-9999,
                    dst_nodata=-9999,
                )

                _emit_ndvi_progress(
                    ndvi_progress_callback,
                    sub_progress=0.80,
                    phase="Reprojecting to WGS84",
                )

                _log("OK", f"Mosaic complete: {final_tif}")
                mosaic_ok = True

            except Exception as e:
                return {"status": "error", "message": f"Mosaic failed: {str(e)}"}
            finally:
                # The planar intermediate is not part of the cache (we only
                # cache the per-tile inputs); drop it whether or not the
                # mosaic step succeeded.
                if os.path.exists(planar_mosaic_tif):
                    try:
                        os.remove(planar_mosaic_tif)
                    except OSError:
                        pass

            # Per-cluster GeoTIFF tiles + tiles_index.json. Best-effort:
            # main mosaic already succeeded, so per-cluster failures log
            # a WARN rather than failing the whole run.
            if write_cluster_tiles:
                cluster_tiles_dir = os.path.join(folder, f"{output_name}_ndvi_tiles")
                _emit_ndvi_progress(
                    ndvi_progress_callback,
                    sub_progress=0.82,
                    phase="Writing per-cluster tiles",
                )
                try:
                    entries = self._write_per_cluster_outputs(
                        work_dir,
                        tiles,
                        failed_tile_refs,
                        cluster_tiles_dir,
                        export_crs,
                        export_crs_name,
                        resume_key,
                        resolution,
                    )
                    _log(
                        "OK",
                        f"Wrote {len(entries)} per-cluster tile(s) to "
                        f"{cluster_tiles_dir}",
                    )
                    cluster_tiles_written = len(entries)
                except Exception as e:
                    _log("WARN", f"Per-cluster tile output failed: {e}")
                    cluster_tiles_dir = None
                    cluster_tiles_written = 0
            else:
                cluster_tiles_dir = None
                cluster_tiles_written = 0

        finally:
            if work_dir and os.path.isdir(work_dir):
                if mosaic_ok and resume_key:
                    # Don't delete the workspace — it's the persistent cache
                    # now. Record the on-disk footprint + access time so the
                    # LRU eviction has correct numbers, then trim to the cap.
                    _emit_ndvi_progress(
                        ndvi_progress_callback,
                        sub_progress=0.86,
                        phase="Updating tile cache",
                    )
                    try:
                        self.tile_cache.touch(resume_key)
                        n_evicted, bytes_freed = self.tile_cache.evict_lru()
                        if n_evicted:
                            _log(
                                "INFO",
                                f"Tile cache: evicted {n_evicted} LRU entry/entries "
                                f"({bytes_freed / 1024**2:.0f} MB) to stay under cap.",
                            )
                    except Exception as e:
                        _log("WARN", f"Tile cache bookkeeping failed: {e}")
                elif not mosaic_ok:
                    _log(
                        "INFO",
                        f"Tile workspace preserved for resume: {work_dir}",
                    )

        if cancel_callback and cancel_callback():
            return {"status": "cancelled", "message": "Cancelled by user"}

        n_mosaic_tiles = len(tile_files)
        meta_base: dict[str, Any] = {
            "tiles": n_mosaic_tiles,
            "tiles_total": n_tiles,
            "tiles_failed": len(failed_tile_refs),
            "tiles_resumed": n_resumed,
            "failed_tile_refs": failed_tile_refs,
            "n_clusters": n_clusters,
            "resume_key": resume_key,
            "cluster_tiles_dir": cluster_tiles_dir,
            "cluster_tiles_written": cluster_tiles_written,
            "export_crs": export_crs,
            "export_crs_name": export_crs_name,
            "export_distortion": export_distortion,
        }
        try:
            with rasterio.open(final_tif) as src:
                meta_base["crs"] = str(src.crs)
        except Exception:
            meta_base["crs"] = "EPSG:4326"

        if not (write_geojson or write_geopackage):
            _emit_ndvi_progress(
                ndvi_progress_callback,
                sub_progress=0.99,
                phase="Finalizing",
            )
            return {
                "status": "success",
                "tif": final_tif if write_geotiff else None,
                "geojson": None,
                "gpkg": None,
                "meta": meta_base,
            }

        geojson_path = (
            os.path.join(folder, f"{output_name}_ndvi.geojson")
            if write_geojson
            else None
        )
        gpkg_path = (
            os.path.join(folder, f"{output_name}_ndvi.gpkg")
            if write_geopackage
            else None
        )
        out = self._raster_to_ndvi_points(
            final_tif,
            None,
            geojson_path=geojson_path,
            gpkg_path=gpkg_path,
            ndvi_progress_callback=ndvi_progress_callback,
            cancel_callback=cancel_callback,
            sub_start=0.88,
            sub_end=0.99,
            meta_extra={
                "tiles": n_mosaic_tiles,
                "tiles_total": n_tiles,
                "tiles_failed": len(failed_tile_refs),
                "tiles_resumed": n_resumed,
                "failed_tile_refs": failed_tile_refs,
                "n_clusters": n_clusters,
                "resume_key": resume_key,
                "cluster_tiles_dir": cluster_tiles_dir,
                "cluster_tiles_written": cluster_tiles_written,
                "export_crs": export_crs,
                "export_crs_name": export_crs_name,
                "export_distortion": export_distortion,
            },
        )
        if out.get("status") != "success":
            return out
        if not write_geotiff and os.path.isfile(final_tif):
            os.remove(final_tif)
            out["tif"] = None
        elif write_geotiff:
            out["tif"] = final_tif
        return out
