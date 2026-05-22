from __future__ import annotations

import json
import os
import shutil
import time
import uuid
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
from .jobs import progress_interval_s
from .logger import get_logger

_log = get_logger("NDVI")

# TODO: NDVI_CACHE - Implement local caching of Earth Engine tiles to reduce API calls
# TODO: NDVI_LANDSAT - Add Landsat 8/9 support alongside Sentinel-2
# TODO: NDVI_TEMPORAL - Add time-series analysis for seasonal greenery changes

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Concurrent tile-download workers. Earth Engine allows several parallel
# ``ee_export_image`` calls per user well above this; the cap exists to avoid
# saturating local sockets and to keep the post-download rasterio decode +
# reproject step from contending too heavily for the GIL on a small machine.
# Mirrors the GVI engine's ``_MAX_CONCURRENT_POINTS = 4`` choice.
_MAX_CONCURRENT_TILES = 4


def _ndvi_tile_workspace(output_name: str) -> str:
    """Per-run workspace under ``<repo>/temp/ndvi_tiles/`` (removed after mosaic)."""
    root = os.path.join(_REPO_ROOT, "temp", "ndvi_tiles")
    os.makedirs(root, exist_ok=True)
    unique = f"{output_name}_{uuid.uuid4().hex[:10]}"
    path = os.path.join(root, unique)
    os.makedirs(path, exist_ok=True)
    return path


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
    start_date: str,
    end_date: str,
    cloud_max: float,
    resolution_m: int,
    max_tile_size_km: float,
    ee_collection: str,
) -> str:
    """Write ``{output_name}_ndvi.json`` capturing the parameters and grid CRS.

    Mirrors GVI's ``_gvi.json`` so any downstream tool (fusion, custom
    notebooks) can introspect an NDVI raster after the fact: which planar CRS
    pixels were rasterised in, how much distortion that introduced, how many
    clusters and tiles the input decomposed into, the exact date range, and
    which Earth Engine ImageCollection was queried.
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
        "start_date": str(start_date),
        "end_date": str(end_date),
        "cloud_max": float(cloud_max),
        "resolution_m": int(resolution_m),
        "max_tile_size_km": float(max_tile_size_km),
        "ee_collection": ee_collection,
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
    #: Earth Engine ImageCollection ID used by :meth:`get_collection`. Exposed
    #: as a class attribute so the sidecar JSON can record exactly which
    #: dataset produced a given NDVI output (and so a subclass can override
    #: the collection without re-implementing the wrapper).
    EE_COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"

    def __init__(self, project_id=None):
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
        """Read NDVI raster, build a point GDF, write to GeoJSON and/or GeoPackage."""
        if cancel_callback and cancel_callback():
            return {"status": "cancelled", "message": "Cancelled by user"}

        _emit_ndvi_progress(
            ndvi_progress_callback,
            sub_progress=sub_start,
            phase="Writing vector samples",
        )

        try:
            with rasterio.open(final_tif) as src:
                band1 = src.read(1)
                band1 = np.where(band1 == -9999, np.nan, band1)
                height, width = band1.shape

                cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                xs, ys = rasterio.transform.xy(
                    src.transform, rows, cols, offset="center"
                )

                xs = np.array(xs).flatten()
                ys = np.array(ys).flatten()
                values = band1.flatten()

                valid_mask = ~np.isnan(values)

                if not np.any(valid_mask):
                    return {"status": "error", "message": "Raster is empty."}

                xs_r = np.round(xs[valid_mask], 5)
                ys_r = np.round(ys[valid_mask], 5)
                vals_r = np.round(values[valid_mask], 4)

                gdf_out = gpd.GeoDataFrame(
                    {"NDVI": vals_r.astype(np.float32)},
                    geometry=gpd.points_from_xy(xs_r, ys_r),
                    crs="EPSG:4326",
                )

                if isinstance(geometry, gpd.GeoDataFrame):
                    clip_geom = geometry
                    if clip_geom.crs != "EPSG:4326":
                        clip_geom = clip_geom.to_crs("EPSG:4326")
                    gdf_out = gpd.clip(gdf_out, clip_geom)

                if gpkg_path:
                    gdf_out.to_file(gpkg_path, driver="GPKG", layer="ndvi_samples")
                if geojson_path:
                    gdf_out.to_file(geojson_path, driver="GeoJSON")

                _emit_ndvi_progress(
                    ndvi_progress_callback,
                    sub_progress=sub_end,
                    phase="Writing vector samples",
                )

                meta = {"crs": str(src.crs)}
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
        """
        os.makedirs(folder, exist_ok=True)

        if not (write_geotiff or write_geojson or write_geopackage):
            return {
                "status": "error",
                "message": "Enable at least one output format (GeoTIFF, GeoPackage, or GeoJSON).",
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
            geom_for_crs = gpd.GeoDataFrame(
                {"geometry": [geometry]}, crs="EPSG:4326"
            )

        # 2. Pick an EE export CRS so pixels are rasterised in true ground
        # metres regardless of latitude (vs. the legacy hardcoded Web Mercator,
        # which doubles cell area near 60°N).
        export_crs, grid_crs, export_distortion, export_crs_name = (
            _select_export_crs(geom_for_crs)
        )

        # 3. Get Collection
        col = self.get_collection(aoi, str(start_date), str(end_date), cloud_max)
        if col.size().getInfo() == 0:
            return {
                "status": "error",
                "message": f"No images found for {start_date} to {end_date}.",
            }

        if cancel_callback and cancel_callback():
            return {"status": "cancelled", "message": "Cancelled by user"}

        ndvi_median = col.median().clip(aoi)

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
                n_clusters=n_clusters,
                cancel_callback=cancel_callback,
                ndvi_progress_callback=ndvi_progress_callback,
                write_geotiff=write_geotiff,
                write_geojson=write_geojson,
                write_geopackage=write_geopackage,
            )

        # Sidecar: write on success so direct API callers (CLI / fusion auto-
        # download / notebooks) get the same parameter+CRS audit trail the GVI
        # ``_gvi.json`` provides. Failures and cancellations skip this — the
        # sidecar should only describe outputs that actually landed on disk.
        if result.get("status") == "success":
            raw_meta = result.get("meta")
            result_meta: dict = raw_meta if isinstance(raw_meta, dict) else {}
            tiles_succeeded = int(result_meta.get("tiles", n_tiles))
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
                    start_date=str(start_date),
                    end_date=str(end_date),
                    cloud_max=cloud_max,
                    resolution_m=resolution,
                    max_tile_size_km=max_tile_size_km,
                    ee_collection=self.EE_COLLECTION_ID,
                )
                result["sidecar"] = sidecar_path
            except Exception as e:
                _log("WARN", f"Sidecar write failed (non-fatal): {e}")
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
            geemap.ee_export_image(
                ndvi_median.unmask(-9999),
                filename=temp_tif,
                scale=resolution,
                crs=export_crs,
                region=aoi,
                file_per_band=False,
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

            # 4. Reproject to EPSG:4326 with aspect-ratio correction
            reproject_raster_to_wgs84(
                temp_tif, final_tif, target_resolution_m=resolution
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
            "n_clusters": 1,
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

    def _download_one_tile(
        self,
        tile_spec: dict,
        ndvi_median,
        work_dir: str,
        resolution: int,
        export_crs: str,
        cancel_callback: Callable[[], bool] | None,
    ) -> dict:
        """Worker: one tile through Earth Engine + reproject to WGS84.

        Runs on a ThreadPoolExecutor thread (see :meth:`_download_with_tiling`).
        Returns ``{success, tile_final, idx, error}``; failures surface as
        ``success=False`` rather than raising so a single bad tile can't sink
        the whole batch. Cancel checks short-circuit at safe boundaries —
        Earth Engine's HTTP call is one blocking step that can't be killed
        mid-flight, so a freshly-cancelled run may still have a few in-flight
        downloads finish and be discarded by the caller.
        """
        idx = int(tile_spec["tile_idx"])
        tile_geom = tile_spec["tile_geom_4326"]
        result: dict = {"success": False, "tile_final": None, "idx": idx, "error": None}

        if cancel_callback and cancel_callback():
            return result

        tile_temp = os.path.join(work_dir, f"tile_{idx}_temp.tif")
        tile_final = os.path.join(work_dir, f"tile_{idx}.tif")

        try:
            tile_aoi = _shapely_to_ee_geometry(tile_geom)
            tile_ndvi = ndvi_median.clip(tile_aoi)
            geemap.ee_export_image(
                tile_ndvi.unmask(-9999),
                filename=tile_temp,
                scale=resolution,
                crs=export_crs,
                region=tile_aoi,
                file_per_band=False,
            )

            if cancel_callback and cancel_callback():
                if os.path.exists(tile_temp):
                    os.remove(tile_temp)
                return result

            reproject_raster_to_wgs84(
                tile_temp, tile_final, target_resolution_m=resolution
            )
            if os.path.exists(tile_temp):
                os.remove(tile_temp)

            result["success"] = True
            result["tile_final"] = tile_final
            return result
        except Exception as e:
            if os.path.exists(tile_temp):
                try:
                    os.remove(tile_temp)
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
        n_clusters: int = 1,
        cancel_callback: Callable[[], bool] | None = None,
        ndvi_progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        write_geotiff: bool = True,
        write_geojson: bool = True,
        write_geopackage: bool = False,
    ):
        """Download NDVI for a precomputed cluster-aware tile list and mosaic.

        ``tiles`` is the output of :func:`geofuse.core.build_planar_tiles` — a list of
        ``{cluster_id, tile_idx, tile_geom_4326}`` dicts. Each tile is exported
        independently and mosaicked into one GeoTIFF; per-cluster GeoTIFF tile
        outputs are a separate downstream step (Phase 4).
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

        work_dir = _ndvi_tile_workspace(output_name)
        tile_files: list[str] = []
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
            done_count = 0
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
                        cancel_callback,
                    ): ts
                    for ts in tiles
                }
                cancelled = False
                for fut in as_completed(futures):
                    res = fut.result()
                    done_count += 1
                    if res["success"] and res["tile_final"] is not None:
                        tile_files.append(res["tile_final"])
                    elif res.get("error"):
                        _log(
                            "WARN",
                            f"Tile {res['idx'] + 1} failed: {res['error']}",
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

            try:
                # Stream-mosaic the per-tile rasters into the final GeoTIFF.
                # Memory ceiling is one tile at a time (vs. ``rasterio.merge``,
                # which would materialise the entire mosaic up front and OOM
                # on national-scale outputs).
                def _mosaic_progress(k: int, n: int) -> None:
                    span = 0.78 - 0.72
                    _emit_ndvi_progress(
                        ndvi_progress_callback,
                        sub_progress=0.72 + (span * k / n if n else 0.0),
                        phase="Mosaicking rasters",
                        tiles=(k, n),
                    )

                stream_mosaic_to_geotiff(
                    tile_files,
                    final_tif,
                    nodata=-9999,
                    progress_cb=_mosaic_progress,
                )

                _emit_ndvi_progress(
                    ndvi_progress_callback,
                    sub_progress=0.78,
                    phase="Mosaicking rasters",
                    clear_bracket=True,
                )

                _log("OK", f"Mosaic complete: {final_tif}")
                mosaic_ok = True

            except Exception as e:
                return {"status": "error", "message": f"Mosaic failed: {str(e)}"}

        finally:
            if work_dir and os.path.isdir(work_dir):
                if mosaic_ok:
                    _emit_ndvi_progress(
                        ndvi_progress_callback,
                        sub_progress=0.86,
                        phase="Removing temporary tiles",
                    )
                shutil.rmtree(work_dir, ignore_errors=True)

        if cancel_callback and cancel_callback():
            return {"status": "cancelled", "message": "Cancelled by user"}

        n_mosaic_tiles = len(tile_files)
        meta_base: dict[str, Any] = {
            "tiles": n_mosaic_tiles,
            "tiles_total": n_tiles,
            "n_clusters": n_clusters,
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
                "n_clusters": n_clusters,
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
