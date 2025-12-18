import unittest
import sys
import os
import shutil
import warnings

# -------------------------------------------------------------------------
# CRITICAL IMPORT ORDER FIX FOR WINDOWS
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import geofuse  # Must import before geopandas to load DLLs correctly

import pandas as pd
import geopandas as gpd
import numpy as np
from unittest.mock import MagicMock, patch

from geofuse.gvi import GVIEngine
from geofuse.vision import DeepLabSegmenter


class TestGeoFuse(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 1. Define specific output directory for this test
        cls.output_dir = os.path.join("tests", "output", "test1_logic")

        # ROBUST CLEANUP: Try to delete, but don't crash if Windows locks it.
        if os.path.exists(cls.output_dir):
            try:
                shutil.rmtree(cls.output_dir)
            except OSError as e:
                # This handles [WinError 5] Access is denied
                print(
                    f"\n[WARN] Could not delete old output folder (Windows lock?). Proceeding to overwrite files instead."
                )

        os.makedirs(cls.output_dir, exist_ok=True)

        # 2. Ensure sample data exists (Force overwrite to ensure correct size)
        sample_path = "data/samples/test_area.geojson"
        os.makedirs("data/samples", exist_ok=True)

        # Create a larger 5x5km box (~0.05 deg) for better visual verification
        from shapely.geometry import Polygon

        p = Polygon(
            [
                (-114.10, 51.00),
                (-114.10, 51.05),
                (-114.05, 51.05),
                (-114.05, 51.00),
                (-114.10, 51.00),
            ]
        )
        gdf = gpd.GeoDataFrame({"geometry": [p]}, crs="EPSG:4326")
        gdf.to_file(sample_path, driver="GeoJSON")
        print(f"[INFO] Created/Updated sample data at {sample_path} (Size: ~5km x 5km)")

    def test_gvi_pipeline_logic(self):
        """Tests the Grid Gen -> Loop -> Raster logic without needing a real GPU."""
        print("\n[TEST] Testing GVI Pipeline Logic (Mocked)...")

        # --- SUPPRESS PYTORCH 2.4+ WARNING ---
        # We place this INSIDE the test to ensure unittest doesn't override it.
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="Python 3.14 will, by default, filter extracted tar archives",
        )

        # Setup Mock Engine
        engine = GVIEngine(download_mode="package")
        engine.segmenter = MagicMock()
        mock_mask = np.zeros((100, 100), dtype=int)
        mock_mask[0:50, :] = 8  # Vegetation ID
        engine.segmenter.predict.return_value = mock_mask
        engine.segmenter.calculate_gvi_from_mask.return_value = {
            "GVI_Total": 0.5,
            "GVI_Terrain": 0.0,
        }
        engine._get_pano_img = MagicMock(return_value="ValidImageObject")

        # Prepare Data
        input_path = "data/samples/test_area.geojson"
        gdf = gpd.read_file(input_path)

        # Run Analysis
        # Step 0.005 ensures a ~10x10 grid (100 points) for good visual verification
        results_gdf = engine.run_analysis(
            gdf, step=0.005, folder=self.output_dir, save_panos=False, save_masks=False
        )

        # Verify Output
        expected_tif = os.path.join(self.output_dir, "gvi_distribution.tif")
        self.assertTrue(os.path.exists(expected_tif))
        print(f"   [PASS] GVI GeoTIFF Created at {expected_tif}")

    @patch("geofuse.ndvi.ee")
    @patch("geofuse.ndvi.geemap")
    def test_ndvi_logic(self, mock_geemap, mock_ee):
        """Tests that the GEE wrapper constructs the correct calls."""
        print("\n[TEST] Testing NDVI Logic (Mocked GEE)...")
        from geofuse.ndvi import NDVIEngine

        # Setup Mock
        mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filter.return_value.map.return_value.size.return_value.getInfo.return_value = (
            5
        )

        # Side effect to create dummy file
        def create_dummy_file(image, filename, **kwargs):
            with open(filename, "w") as f:
                f.write("Dummy GeoTIFF content for testing.")

        mock_geemap.ee_export_image.side_effect = create_dummy_file

        engine = NDVIEngine()

        # Run Export
        input_geo = "data/samples/test_area.geojson"
        output_tif = os.path.join(self.output_dir, "test_ndvi.tif")

        engine.export_geotiff(input_geo, "2024-06-01", output_tif)

        # Verify
        mock_geemap.ee_export_image.assert_called_once()
        self.assertTrue(os.path.exists(output_tif))
        print(f"   [PASS] NDVI Export Logic Verified (File created at {output_tif})")


if __name__ == "__main__":
    unittest.main()
