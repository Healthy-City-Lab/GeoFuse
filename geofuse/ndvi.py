import json
import os
import subprocess
import tempfile

import ee
import geemap
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import Point, box

# TODO: NDVI_CACHE - Implement local caching of Earth Engine tiles to reduce API calls
# TODO: NDVI_LANDSAT - Add Landsat 8/9 support alongside Sentinel-2
# TODO: NDVI_TEMPORAL - Add time-series analysis for seasonal greenery changes


class NDVIEngine:
    def __init__(self, project_id=None):
        try:
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()
        except ee.EEException as e:
            print(f"[WARN] Earth Engine Init Failed: {e}")
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

    def _reproject_tile_to_4326(self, src_path: str, dst_path: str, resolution: int) -> None:
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
        """
        os.makedirs(folder, exist_ok=True)

        # 1. Convert Geometry
        if isinstance(geometry, gpd.GeoDataFrame):
            geom_wgs84 = geometry.to_crs(epsg=4326)
            js = json.loads(geom_wgs84.to_json())
            features = geemap.geojson_to_ee(js)
            aoi = features.geometry()
            bounds = geom_wgs84.total_bounds
        else:
            aoi = geemap.shapely_to_ee(geometry)
            bounds = geometry.bounds

        # 2. Get Collection
        col = self.get_collection(aoi, str(start_date), str(end_date), cloud_max)
        if col.size().getInfo() == 0:
            return {
                "status": "error",
                "message": f"No images found for {start_date} to {end_date}.",
            }

        ndvi_median = col.median().clip(aoi)

        # 3. Check if tiling is needed based on area size
        minx, miny, maxx, maxy = bounds
        width_km = (maxx - minx) * 111  # Rough approximation at equator
        height_km = (maxy - miny) * 111

        needs_tiling = width_km > max_tile_size_km or height_km > max_tile_size_km

        if needs_tiling:
            print(
                f"[NDVI] Large area detected ({width_km:.1f}x{height_km:.1f} km). Using tiled download..."
            )
            return self._download_with_tiling(
                ndvi_median,
                aoi,
                bounds,
                output_name,
                resolution,
                folder,
                max_tile_size_km,
            )
        else:
            print(
                f"[NDVI] Area size: {width_km:.1f}x{height_km:.1f} km. Single download..."
            )
            return self._download_single(
                ndvi_median, aoi, geometry, output_name, resolution, folder
            )

    def _download_single(self, ndvi_median, aoi, geometry, output_name, resolution, folder):
        """Download NDVI as a single tile (for small areas)."""
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

            # 4. Reproject to EPSG:4326 with aspect-ratio correction
            self._reproject_tile_to_4326(temp_tif, final_tif, resolution)

            if os.path.exists(temp_tif):
                os.remove(temp_tif)

        except Exception as e:
            if os.path.exists(temp_tif):
                os.remove(temp_tif)
            return {"status": "error", "message": f"Export/Reproject Failed: {str(e)}"}

        # 5. Extract Points (Points will align because they use the same transform)
        geojson_path = os.path.join(folder, f"{output_name}_ndvi.geojson")

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

                df = pd.DataFrame(
                    {
                        "NDVI": values[valid_mask],
                        "x": xs[valid_mask],
                        "y": ys[valid_mask],
                    }
                )

                gdf_out = gpd.GeoDataFrame(
                    df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:4326"
                )

                if isinstance(geometry, gpd.GeoDataFrame):
                    if geometry.crs != "EPSG:4326":
                        geometry = geometry.to_crs("EPSG:4326")
                    gdf_out = gpd.clip(gdf_out, geometry)

                gdf_out.to_file(geojson_path, driver="GeoJSON")

                return {
                    "status": "success",
                    "tif": final_tif,
                    "geojson": geojson_path,
                    "meta": {"crs": str(src.crs)},
                }

        except Exception as e:
            return {"status": "error", "message": f"Pixel Extraction Failed: {str(e)}"}

    def _download_with_tiling(
        self,
        ndvi_median,
        aoi,
        bounds,
        output_name,
        resolution,
        folder,
        max_tile_size_km,
    ):
        """Download NDVI in tiles and mosaic them together (for large areas)."""
        minx, miny, maxx, maxy = bounds

        # Calculate tile size in degrees
        tile_size_deg = max_tile_size_km / 111  # Approximate degrees

        # Generate tile grid
        tiles = []
        x = minx
        tile_idx = 0

        print(f"[NDVI] Creating tile grid (max {max_tile_size_km} km per tile)...")

        while x < maxx:
            x_end = min(x + tile_size_deg, maxx)
            y = miny

            while y < maxy:
                y_end = min(y + tile_size_deg, maxy)
                tiles.append((tile_idx, box(x, y, x_end, y_end)))
                tile_idx += 1
                y = y_end

            x = x_end

        print(f"[NDVI] Generated {len(tiles)} tiles. Downloading...")

        # Download each tile
        with tempfile.TemporaryDirectory() as temp_dir:
            tile_files = []

            for idx, tile_geom in tiles:
                print(f"[NDVI] Downloading tile {idx+1}/{len(tiles)}...")

                tile_aoi = geemap.shapely_to_ee(tile_geom)
                tile_ndvi = ndvi_median.clip(tile_aoi)

                tile_temp = os.path.join(temp_dir, f"tile_{idx}_temp.tif")
                tile_final = os.path.join(temp_dir, f"tile_{idx}.tif")

                try:
                    # Export tile in EPSG:3857
                    geemap.ee_export_image(
                        tile_ndvi.unmask(-9999),
                        filename=tile_temp,
                        scale=resolution,
                        crs="EPSG:3857",
                        region=tile_aoi,
                        file_per_band=False,
                    )

                    # Reproject to EPSG:4326 with aspect ratio correction
                    self._reproject_tile_to_4326(tile_temp, tile_final, resolution)

                    tile_files.append(tile_final)

                    # Cleanup temp file
                    if os.path.exists(tile_temp):
                        os.remove(tile_temp)

                except Exception as e:
                    print(f"[NDVI] Warning: Tile {idx+1} failed: {e}")
                    continue

            if not tile_files:
                return {"status": "error", "message": "All tiles failed to download"}

            print(
                f"[NDVI] Successfully downloaded {len(tile_files)}/{len(tiles)} tiles. Mosaicking..."
            )

            # Mosaic tiles together
            final_tif = os.path.join(folder, f"{output_name}_ndvi.tif")

            try:
                # Read all tiles and mosaic
                src_files_to_mosaic = []
                for tile_file in tile_files:
                    src = rasterio.open(tile_file)
                    src_files_to_mosaic.append(src)

                mosaic, out_trans = merge(src_files_to_mosaic, nodata=-9999)

                # Get metadata from first tile
                out_meta = src_files_to_mosaic[0].meta.copy()
                out_meta.update(
                    {
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": out_trans,
                    }
                )

                # Write mosaic
                with rasterio.open(final_tif, "w", **out_meta) as dest:
                    dest.write(mosaic)

                # Close all source files
                for src in src_files_to_mosaic:
                    src.close()

                print(f"[NDVI] Mosaic complete: {final_tif}")

            except Exception as e:
                return {"status": "error", "message": f"Mosaic failed: {str(e)}"}

        # Extract points from final mosaic
        geojson_path = os.path.join(folder, f"{output_name}_ndvi.geojson")

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
                    return {"status": "error", "message": "Mosaic is empty."}

                df = pd.DataFrame(
                    {
                        "NDVI": values[valid_mask],
                        "x": xs[valid_mask],
                        "y": ys[valid_mask],
                    }
                )

                gdf_out = gpd.GeoDataFrame(
                    df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:4326"
                )

                gdf_out.to_file(geojson_path, driver="GeoJSON")

                return {
                    "status": "success",
                    "tif": final_tif,
                    "geojson": geojson_path,
                    "meta": {"crs": str(src.crs), "tiles": len(tile_files)},
                }

        except Exception as e:
            return {"status": "error", "message": f"Point extraction failed: {str(e)}"}
