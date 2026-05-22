from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable, Mapping
from typing import Any

import ee
import geemap
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import box, mapping

from .crs_utils import reproject_geodataframe_to_wgs84, select_grid_crs
from .logger import get_logger

_log = get_logger("NDVI")

# TODO: NDVI_CACHE - Implement local caching of Earth Engine tiles to reduce API calls
# TODO: NDVI_LANDSAT - Add Landsat 8/9 support alongside Sentinel-2
# TODO: NDVI_TEMPORAL - Add time-series analysis for seasonal greenery changes

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


class NDVIEngine:
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
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
            .map(self.prep_ndvi)
        )

    def _reproject_tile_to_4326(
        self, src_path: str, dst_path: str, resolution: int
    ) -> None:
        """Reproject a 3857 GeoTIFF to 4326 with per-latitude aspect-ratio correction.

        Downloads from Earth Engine arrive in EPSG:3857 (metres). A naive
        reprojection produces non-square pixels in geographic space. This method
        calculates the exact degree-per-metre scale at the tile's centroid
        latitude and forces an explicit square-metre output resolution.
        """
        from rasterio.warp import transform as warp_transform

        with rasterio.open(src_path) as src:
            left, bottom, right, top = array_bounds(
                src.height, src.width, src.transform
            )
            cx, cy = (left + right) / 2, (bottom + top) / 2
            lon_c, lat_c = warp_transform(src.crs, "EPSG:4326", [cx], [cy])
            avg_lat = lat_c[0]

            lat_rad = np.radians(avg_lat)
            m_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * lat_rad)
            m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)

            res_x_deg = resolution / m_per_deg_lon
            res_y_deg = resolution / m_per_deg_lat

            dst_transform, width, height = calculate_default_transform(
                src.crs,
                "EPSG:4326",
                src.width,
                src.height,
                *src.bounds,
                resolution=(res_x_deg, res_y_deg),
            )

            kwargs = src.meta.copy()
            kwargs.update(
                {
                    "crs": "EPSG:4326",
                    "transform": dst_transform,
                    "width": width,
                    "height": height,
                }
            )

            with rasterio.open(dst_path, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs="EPSG:4326",
                        resampling=Resampling.nearest,
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
            bounds = geom_wgs84.total_bounds
        else:
            aoi = _shapely_to_ee_geometry(geometry)
            bounds = geometry.bounds

        # 2. Get Collection
        col = self.get_collection(aoi, str(start_date), str(end_date), cloud_max)
        if col.size().getInfo() == 0:
            return {
                "status": "error",
                "message": f"No images found for {start_date} to {end_date}.",
            }

        if cancel_callback and cancel_callback():
            return {"status": "cancelled", "message": "Cancelled by user"}

        ndvi_median = col.median().clip(aoi)

        # 3. Check if tiling is needed based on area size
        minx, miny, maxx, maxy = bounds
        width_km = (maxx - minx) * 111  # Rough approximation at equator
        height_km = (maxy - miny) * 111

        needs_tiling = width_km > max_tile_size_km or height_km > max_tile_size_km

        if needs_tiling:
            _log(
                "INFO",
                f"Large area detected ({width_km:.1f}x{height_km:.1f} km). "
                "Using tiled download...",
            )
            return self._download_with_tiling(
                ndvi_median,
                aoi,
                bounds,
                output_name,
                resolution,
                folder,
                max_tile_size_km,
                cancel_callback=cancel_callback,
                ndvi_progress_callback=ndvi_progress_callback,
                write_geotiff=write_geotiff,
                write_geojson=write_geojson,
                write_geopackage=write_geopackage,
            )
        else:
            _log(
                "INFO",
                f"Area size: {width_km:.1f}x{height_km:.1f} km. Single download...",
            )
            return self._download_single(
                ndvi_median,
                aoi,
                geometry,
                output_name,
                resolution,
                folder,
                cancel_callback=cancel_callback,
                ndvi_progress_callback=ndvi_progress_callback,
                write_geotiff=write_geotiff,
                write_geojson=write_geojson,
                write_geopackage=write_geopackage,
            )

    def _download_single(
        self,
        ndvi_median,
        aoi,
        geometry,
        output_name,
        resolution,
        folder,
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
        # Export as EPSG:3857 (Meters) first
        temp_tif = os.path.join(folder, f"temp_{output_name}.tif")
        final_tif = os.path.join(folder, f"{output_name}_ndvi.tif")

        try:
            geemap.ee_export_image(
                ndvi_median.unmask(-9999),
                filename=temp_tif,
                scale=resolution,
                crs="EPSG:3857",
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
            self._reproject_tile_to_4326(temp_tif, final_tif, resolution)

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
        meta: dict[str, Any] = {"crs": crs_meta}

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
        )
        if out.get("status") != "success":
            return out
        if not write_geotiff and os.path.isfile(final_tif):
            os.remove(final_tif)
            out["tif"] = None
        elif write_geotiff:
            out["tif"] = final_tif
        return out

    def _download_with_tiling(
        self,
        ndvi_median,
        aoi,
        bounds,
        output_name,
        resolution,
        folder,
        max_tile_size_km,
        cancel_callback: Callable[[], bool] | None = None,
        ndvi_progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        write_geotiff: bool = True,
        write_geojson: bool = True,
        write_geopackage: bool = False,
    ):
        """Download NDVI in tiles and mosaic them together (for large areas)."""
        minx, miny, maxx, maxy = bounds

        tile_size_deg = max_tile_size_km / 111  # Approximate degrees

        tiles = []
        x = minx
        tile_idx = 0

        _log("INFO", f"Creating tile grid (max {max_tile_size_km} km per tile)...")

        while x < maxx:
            x_end = min(x + tile_size_deg, maxx)
            y = miny

            while y < maxy:
                y_end = min(y + tile_size_deg, maxy)
                tiles.append((tile_idx, box(x, y, x_end, y_end)))
                tile_idx += 1
                y = y_end

            x = x_end

        _log("INFO", f"Generated {len(tiles)} tiles. Downloading...")

        n_tiles = len(tiles)
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
            for idx, tile_geom in tiles:
                if cancel_callback and cancel_callback():
                    return {"status": "cancelled", "message": "Cancelled by user"}

                _log("INFO", f"Downloading tile {idx+1}/{len(tiles)}...")

                tile_aoi = _shapely_to_ee_geometry(tile_geom)
                tile_ndvi = ndvi_median.clip(tile_aoi)

                tile_temp = os.path.join(work_dir, f"tile_{idx}_temp.tif")
                tile_final = os.path.join(work_dir, f"tile_{idx}.tif")

                try:
                    geemap.ee_export_image(
                        tile_ndvi.unmask(-9999),
                        filename=tile_temp,
                        scale=resolution,
                        crs="EPSG:3857",
                        region=tile_aoi,
                        file_per_band=False,
                    )

                    if cancel_callback and cancel_callback():
                        if os.path.exists(tile_temp):
                            os.remove(tile_temp)
                        return {"status": "cancelled", "message": "Cancelled by user"}

                    self._reproject_tile_to_4326(tile_temp, tile_final, resolution)

                    tile_files.append(tile_final)

                    if os.path.exists(tile_temp):
                        os.remove(tile_temp)

                    k = len(tile_files)
                    _emit_ndvi_progress(
                        ndvi_progress_callback,
                        sub_progress=0.70 * k / n_tiles if n_tiles else 0.0,
                        phase="Downloading tiles",
                        tiles=(k, n_tiles),
                    )

                except Exception as e:
                    _log("WARN", f"Tile {idx+1} failed: {e}")
                    continue

            if cancel_callback and cancel_callback():
                return {"status": "cancelled", "message": "Cancelled by user"}

            if not tile_files:
                return {"status": "error", "message": "All tiles failed to download"}

            _log(
                "OK",
                f"Successfully downloaded {len(tile_files)}/{len(tiles)} tiles. "
                "Mosaicking...",
            )

            if cancel_callback and cancel_callback():
                return {"status": "cancelled", "message": "Cancelled by user"}

            _emit_ndvi_progress(
                ndvi_progress_callback,
                sub_progress=0.72,
                phase="Mosaicking rasters",
                clear_bracket=True,
            )

            try:
                src_files_to_mosaic = []
                for tile_file in tile_files:
                    src = rasterio.open(tile_file)
                    src_files_to_mosaic.append(src)

                mosaic, out_trans = merge(src_files_to_mosaic, nodata=-9999)

                out_meta = src_files_to_mosaic[0].meta.copy()
                out_meta.update(
                    {
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": out_trans,
                    }
                )

                _emit_ndvi_progress(
                    ndvi_progress_callback,
                    sub_progress=0.78,
                    phase="Mosaicking rasters",
                )

                with rasterio.open(final_tif, "w", **out_meta) as dest:
                    dest.write(mosaic)

                for src in src_files_to_mosaic:
                    src.close()

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
        meta_base: dict[str, Any] = {"tiles": n_mosaic_tiles}
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
            meta_extra={"tiles": n_mosaic_tiles},
        )
        if out.get("status") != "success":
            return out
        if not write_geotiff and os.path.isfile(final_tif):
            os.remove(final_tif)
            out["tif"] = None
        elif write_geotiff:
            out["tif"] = final_tif
        return out
