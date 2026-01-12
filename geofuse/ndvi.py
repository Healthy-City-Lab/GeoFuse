import ee
import geemap
import os
import subprocess
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
from shapely.geometry import Point
import json


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

    def download_and_process(
        self,
        geometry,
        start_date,
        end_date,
        output_name,
        cloud_max=10,
        resolution=10,
        folder="output_results",
    ):
        os.makedirs(folder, exist_ok=True)

        # 1. Convert Geometry
        if isinstance(geometry, gpd.GeoDataFrame):
            geom_wgs84 = geometry.to_crs(epsg=4326)
            js = json.loads(geom_wgs84.to_json())
            features = geemap.geojson_to_ee(js)
            aoi = features.geometry()
        else:
            aoi = geemap.shapely_to_ee(geometry)

        # 2. Get Collection
        col = self.get_collection(aoi, str(start_date), str(end_date), cloud_max)
        if col.size().getInfo() == 0:
            return {
                "status": "error",
                "message": f"No images found for {start_date} to {end_date}.",
            }

        ndvi_median = col.median().clip(aoi)

        # 3. Export as EPSG:3857 (Meters) first
        # We start with meters to ensure the raw pixel data represents true shape
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

            # 4. Reproject to EPSG:4326 with ASPECT RATIO CORRECTION
            # This is the step that fixes the "irregular/rectangular" pixels.
            with rasterio.open(temp_tif) as src:
                # Calculate centroid latitude to determine degrees-per-meter
                left, bottom, right, top = array_bounds(
                    src.height, src.width, src.transform
                )
                # Convert 3857 centers to Lat/Lon
                from rasterio.warp import transform

                cx, cy = (left + right) / 2, (bottom + top) / 2
                lon_c, lat_c = transform(src.crs, "EPSG:4326", [cx], [cy])
                avg_lat = lat_c[0]

                # Calculate Metric-to-Degree factors (Same logic as GVI)
                lat_rad = np.radians(avg_lat)
                m_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * lat_rad)
                m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)

                # Calculate target resolution in Degrees
                res_x_deg = resolution / m_per_deg_lon
                res_y_deg = resolution / m_per_deg_lat

                # Create transform with EXPLICIT resolution
                dst_transform, width, height = calculate_default_transform(
                    src.crs,
                    "EPSG:4326",
                    src.width,
                    src.height,
                    *src.bounds,
                    resolution=(res_x_deg, res_y_deg),  # <--- Force Square Meters
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

                with rasterio.open(final_tif, "w", **kwargs) as dst:
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
