import unittest
import sys
import os
import shutil

# -------------------------------------------------------------------------
# CRITICAL IMPORT ORDER FIX FOR WINDOWS
# -------------------------------------------------------------------------
# We must import geofuse (and thus torch) BEFORE geopandas/gdal.
# If geopandas loads first, it locks incompatible DLLs, causing torch to crash.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import geofuse

# -------------------------------------------------------------------------

import pandas as pd
import geopandas as gpd
import numpy as np
from unittest.mock import MagicMock, patch

from geofuse.gvi import GVIEngine
from geofuse.fusion import FusionOptimizer
from geofuse.vision import DeepLabSegmenter


class TestGeoFuse(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create output dir
        os.makedirs("tests/output", exist_ok=True)
        # Ensure sample data exists
        if not os.path.exists("data/samples/test_area.geojson"):
            print("[WARN] Sample data missing. Run scripts/create_samples.py first.")
            sys.exit(1)

    # -------------------------------------------------------------------------
    # TEST 1: FUSION OPTIMIZATION
    # -------------------------------------------------------------------------
    def test_fusion_engine(self):
        print("\n[TEST] Testing Fusion Engine...")
        df = pd.read_csv("data/samples/test_fusion.csv")

        # Test Optimization
        opt = FusionOptimizer(df, "CognitiveScore", ["NDVI", "GVI_Tree"])
        best_params = opt.run_optimization(total_trials=10, random_trials=5)

        # Verify Logic
        self.assertTrue("NDVI" in best_params)
        self.assertTrue("GVI_Tree" in best_params)
        print(f"   [PASS] Optimization Params: {best_params}")

        # Test Application
        final_df = opt.apply_best_weights()
        self.assertTrue("CGI" in final_df.columns)
        print("   [PASS] Weights Applied Successfully")

    # -------------------------------------------------------------------------
    # TEST 2: GVI PIPELINE (Mocked Vision)
    # -------------------------------------------------------------------------
    def test_gvi_pipeline_logic(self):
        """
        Tests the Grid Gen -> Loop -> Raster logic without needing a real GPU.
        We mock the 'segmenter' and 'download' functions.
        """
        print("\n[TEST] Testing GVI Pipeline Logic (Mocked)...")

        # Mock the Engine's components
        engine = GVIEngine(download_mode="package")

        # 1. Mock the Segmenter to return a random mask
        engine.segmenter = MagicMock()
        # Mock a 100x100 mask with some "Vegetation" (ID 8)
        mock_mask = np.zeros((100, 100), dtype=int)
        mock_mask[0:50, :] = 8  # Half tree

        engine.segmenter.predict.return_value = mock_mask
        engine.segmenter.calculate_gvi_from_mask.return_value = {
            "GVI_Tree": 0.5,
            "GVI_Grass": 0.0,
            "GVI_Total": 0.5,
        }

        # 2. Mock Download to return a fake "Image" object (just True)
        engine._get_pano_img = MagicMock(return_value="ValidImageObject")

        # Run Pipeline
        input_shp = "data/samples/test_area.shp"
        output_tif = "tests/output/test_gvi.tif"

        # Run with large resolution to generate few points (fast test)
        engine.process_polygon(input_shp, output_tif, resolution=100, save_files=False)

        # Verify Output
        self.assertTrue(os.path.exists(output_tif))
        print("   [PASS] GVI GeoTIFF Created")

    # -------------------------------------------------------------------------
    # TEST 3: NDVI LOGIC (Mocked GEE)
    # -------------------------------------------------------------------------
    @patch("geofuse.ndvi.ee")
    @patch("geofuse.ndvi.geemap")
    def test_ndvi_logic(self, mock_geemap, mock_ee):
        """
        Tests that the GEE wrapper constructs the correct calls.
        """
        print("\n[TEST] Testing NDVI Logic (Mocked GEE)...")
        from geofuse.ndvi import NDVIEngine

        # Setup Mock
        mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filter.return_value.map.return_value.size.return_value.getInfo.return_value = (
            5
        )

        engine = NDVIEngine()

        # Run Export
        input_geo = "data/samples/test_area.geojson"
        output_tif = "tests/output/test_ndvi.tif"

        engine.export_geotiff(input_geo, "2024-06-01", output_tif)

        # Verify geemap export was called
        mock_geemap.ee_export_image.assert_called_once()
        print("   [PASS] NDVI Export Logic Verified")


if __name__ == "__main__":
    unittest.main()
