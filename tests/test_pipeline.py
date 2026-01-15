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

    @patch("geofuse.gvi.search_panoramas")
    def test_gvi_pipeline_logic(self, mock_search):
        """Tests the Grid Gen -> Loop -> Raster logic without needing a real GPU."""
        print("\n[TEST] Testing GVI Pipeline Logic (Mocked)...")

        # --- SUPPRESS PYTORCH 2.4+ WARNING ---
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="Python 3.14 will, by default, filter extracted tar archives",
        )

        # Mock the external streetview search to return a dummy result
        # This prevents hitting the real API or scraping
        mock_search.return_value = [{"panoid": "test_pano_id_123"}]

        # Setup Mock Engine
        engine = GVIEngine(download_mode="package")
        engine.segmenter = MagicMock()

        # Mock the Internal Async Wrapper (Critical for preventing ResourceWarnings)
        # The engine uses `_download_async_wrapper`, NOT `_get_pano_img`
        engine._download_async_wrapper = MagicMock(
            return_value=np.zeros((300, 600, 3), dtype=np.uint8)
        )

        mock_mask = np.zeros((100, 100), dtype=int)
        mock_mask[0:50, :] = 8  # Vegetation ID
        engine.segmenter.predict.return_value = mock_mask
        engine.segmenter.calculate_gvi_from_mask.return_value = {
            "GVI_Total": 0.5,
            "GVI_Terrain": 0.0,
        }

        # Prepare Data
        input_path = "data/samples/test_area.geojson"
        gdf = gpd.read_file(input_path)

        # Capture results since engine doesn't return them directly in this version
        results = []

        def result_callback(res):
            results.append(res)

        # Run Analysis
        # Step 0.005 ensures a ~10x10 grid (100 points) for good visual verification
        results_gdf = engine.run_analysis(
            gdf,
            step=0.005,
            folder=self.output_dir,
            save_panos=False,
            save_masks=False,
            result_callback=result_callback,
        )

        # Verify Output
        # Since the engine currently returns an empty DF and relies on callbacks/persistence,
        # we verify that we processed the expected number of points.
        self.assertEqual(len(results), 100, "Should have processed 100 points")
        print(f"   [PASS] GVI Pipeline processed {len(results)} points")

    @patch("geofuse.ndvi.array_bounds", return_value=(0, 0, 10, 10))
    @patch(
        "geofuse.ndvi.calculate_default_transform", return_value=(MagicMock(), 10, 10)
    )
    @patch("geofuse.ndvi.reproject")
    @patch("geofuse.ndvi.rasterio")
    @patch("geofuse.ndvi.ee")
    @patch("geofuse.ndvi.geemap")
    def test_ndvi_logic(
        self, mock_geemap, mock_ee, mock_rasterio, mock_reproject, mock_cdt, mock_ab
    ):
        """Tests that the GEE wrapper constructs the correct calls."""
        print("\n[TEST] Testing NDVI Logic (Mocked GEE)...")
        from geofuse.ndvi import NDVIEngine

        # Setup Mock
        mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filter.return_value.map.return_value.size.return_value.getInfo.return_value = (
            5
        )

        # Side effect to create dummy file (needed for os.path.exists checks logic)
        def create_dummy_file(image, filename, **kwargs):
            # Just create an empty file so os.path.exists returns True
            with open(filename, "w") as f:
                f.write("Dummy")

        mock_geemap.ee_export_image.side_effect = create_dummy_file

        # Setup Rasterio Mock
        mock_src = MagicMock()
        mock_src.read.return_value = np.zeros((10, 10))  # Dummy band data
        mock_src.transform = MagicMock()
        mock_src.meta = {
            "crs": "EPSG:3857",
            "width": 10,
            "height": 10,
            "transform": object(),
        }
        mock_src.bounds = (0, 0, 10, 10)
        mock_src.count = 1

        # Configure the context manager
        mock_rasterio.open.return_value.__enter__.return_value = mock_src
        # Allow rasterio.band to be called safely on mocks
        mock_rasterio.band = MagicMock()

        engine = NDVIEngine()

        # Run Export
        input_geo = "data/samples/test_area.geojson"
        gdf = gpd.read_file(input_geo)

        engine.download_and_process(
            geometry=gdf,
            start_date="2024-06-01",
            end_date="2024-06-30",
            output_name="test_ndvi",
            folder=self.output_dir,
        )

        # Verify
        mock_geemap.ee_export_image.assert_called_once()
        print(f"   [PASS] NDVI Export Logic Verified")


if __name__ == "__main__":
    unittest.main()
