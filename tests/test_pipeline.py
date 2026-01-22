import os
import shutil
import sys
import unittest
import warnings

# -------------------------------------------------------------------------
# CRITICAL IMPORT ORDER FIX FOR WINDOWS
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd

import geofuse  # Must import before geopandas to load DLLs correctly
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

        # Import the shared sample creator to ensure consistency
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from create_samples import create_sample_data

        # Create standardized test data (~4.4km x 4.4km to avoid NDVI tiling)
        create_sample_data()

        print(f"[INFO] Using standardized test data at {sample_path}")

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

        # Setup Mock Engine (auto-selects best device: CUDA > MPS > CPU)
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
        # Step 500 meters for a ~4.4km area should give ~9x9 = 81 grid points
        # Actual points may be less due to polygon clipping
        results_gdf = engine.run_analysis(
            gdf,
            step=500,  # 500 meters spacing
            folder=self.output_dir,
            save_panos=False,
            save_masks=False,
            result_callback=result_callback,
        )

        # Verify Output
        # The engine processes points that fall within the polygon
        # For a ~4.4km area with 500m step, expect ~60-85 points
        self.assertGreater(len(results), 50, "Should have processed at least 50 points")
        self.assertLess(len(results), 100, "Should not exceed 100 points")
        print(f"   [PASS] GVI Pipeline processed {len(results)} points")

    @patch("geofuse.ndvi.transform")
    @patch("geofuse.ndvi.array_bounds", return_value=(0, 0, 10, 10))
    @patch(
        "geofuse.ndvi.calculate_default_transform", return_value=(MagicMock(), 10, 10)
    )
    @patch("geofuse.ndvi.reproject")
    @patch("geofuse.ndvi.rasterio")
    @patch("geofuse.ndvi.ee")
    @patch("geofuse.ndvi.geemap")
    def test_ndvi_logic(
        self,
        mock_geemap,
        mock_ee,
        mock_rasterio,
        mock_reproject,
        mock_cdt,
        mock_ab,
        mock_warp_transform,
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

        # Setup Rasterio Mock for single-tile workflow
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
        mock_src.crs = "EPSG:3857"
        mock_src.width = 10
        mock_src.height = 10

        # Configure the context manager
        mock_rasterio.open.return_value.__enter__.return_value = mock_src
        mock_rasterio.open.return_value.__exit__.return_value = None

        # Allow rasterio.band to be called safely on mocks
        mock_rasterio.band = MagicMock()

        # Mock rasterio.warp.transform for coordinate conversion
        mock_warp_transform.return_value = ([5.0], [5.0])  # Center coordinates

        # Mock rasterio.transform.xy for point extraction
        mock_rasterio.transform.xy = MagicMock(
            return_value=(np.array([0, 1, 2]), np.array([0, 1, 2]))
        )

        engine = NDVIEngine()

        # Run Export (area is now ~4.4km, should trigger single download)
        input_geo = "data/samples/test_area.geojson"
        gdf = gpd.read_file(input_geo)

        engine.download_and_process(
            geometry=gdf,
            start_date="2024-06-01",
            end_date="2024-06-30",
            output_name="test_ndvi",
            folder=self.output_dir,
        )

        # Verify - should be single call for area <5km
        mock_geemap.ee_export_image.assert_called_once()
        print(f"   [PASS] NDVI Export Logic Verified")


# TODO: NDVI_UNIT_TESTS - Add comprehensive unit tests for NDVIEngine with mocked Earth Engine API
# TODO: FUSION_TESTS - Add test cases for FusionOptimizer when fully implemented

if __name__ == "__main__":
    unittest.main()
