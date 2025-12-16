import ee
import geemap
import os
import subprocess
from datetime import timedelta
import pandas as pd
import geopandas as gpd


class NDVIEngine:
    def __init__(self, project_id=None):
        try:
            # 1. Try standard initialization (uses saved credentials)
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()

        except ee.EEException as e:
            error_msg = str(e)

            # 2. Handle Missing Project ID
            if "not registered" in error_msg or "project" in error_msg:
                print("\n[WARN] Google Cloud Project not set for Earth Engine.")

                # Interactive fix
                try:
                    user_project = input(
                        ">> Enter your Google Cloud Project ID (e.g., ee-my-project): "
                    ).strip()
                    if user_project:
                        print(f"[INFO] Setting default project to '{user_project}'...")

                        # PERSISTENCE: Save this setting globally
                        subprocess.run(
                            f"earthengine set_project {user_project}", shell=True
                        )

                        # Initialize with new project
                        ee.Initialize(project=user_project)
                        print(
                            "[SUCCESS] Project saved. You won't need to enter this again."
                        )
                        return
                except Exception:
                    pass

            # 3. Fallback to Auth
            print(f"[INFO] Triggering Authentication (Reason: {e})...")
            try:
                ee.Authenticate()
                ee.Initialize()
            except Exception as final_e:
                print(f"[FAIL] Could not initialize Earth Engine: {final_e}")
                raise final_e

    def get_collection(self, aoi, date, tolerance_days=15, cloud_max=10):
        start = (pd.to_datetime(date) - timedelta(days=tolerance_days)).strftime(
            "%Y-%m-%d"
        )
        end = (pd.to_datetime(date) + timedelta(days=tolerance_days)).strftime(
            "%Y-%m-%d"
        )

        # UPDATED: Use Harmonized Sentinel-2 to avoid Deprecation Warnings
        return (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
            .map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI"))
        )

    def export_geotiff(
        self, geometry_path, target_date, output_path, resolution=10, tolerance=15
    ):
        # Load Geometry & Convert to WGS84 for GEE
        gdf = gpd.read_file(geometry_path)
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        aoi = geemap.geopandas_to_ee(gdf_wgs84)

        col = self.get_collection(aoi, target_date, tolerance_days=tolerance)
        count = col.size().getInfo()

        if count == 0:
            print(f"No images found for {target_date} (+/- {tolerance} days).")
            return False

        print(f"Found {count} images. Creating median composite...")
        img = col.median().clip(aoi)

        print(f"Exporting NDVI to {output_path}...")
        geemap.ee_export_image(
            img,
            filename=output_path,
            scale=resolution,
            region=aoi.geometry(),
            file_per_band=False,
        )
        return True
