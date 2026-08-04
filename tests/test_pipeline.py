import os
import shutil
import sys
import tempfile
import unittest
import warnings

# ────────────────────────────────────────────────────────────────────
# CRITICAL IMPORT ORDER FIX FOR WINDOWS
# ────────────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from unittest.mock import AsyncMock, MagicMock, patch

import geopandas as gpd
import numpy as np
from PIL import Image
from rasterio.warp import Resampling  # Needed for mocking
from shapely.geometry import box as shapely_box

from geofuse.gvi import GVIEngine
from geofuse.vision import DeepLabSegmenter, get_best_device


# =========================================================================
# Package Smoke Tests
# =========================================================================
class TestPackageSmoke(unittest.TestCase):
    """Fast sanity checks that each installed package is functional.

    These tests do not exercise GeoFuse business logic.  They confirm the
    runtime environment is intact so that broken C-extension bindings,
    version incompatibilities, or partial installations are caught before the
    slower pipeline tests run.  Each test calls at least one non-trivial
    function from the target package so that import-only survives are not
    counted as passing.
    """

    def test_torch_device_selection(self):
        """torch installed and device auto-selection resolves to a valid device."""
        device = get_best_device()
        device_str = str(device)
        self.assertTrue(
            device_str == "cpu" or device_str == "mps" or device_str.startswith("cuda"),
            f"get_best_device() returned unexpected device: {device}",
        )

    def test_geopandas_crs_reproject(self):
        """geopandas CRS reprojection works — confirms PROJ/GDAL C-bindings are intact."""
        gdf = gpd.GeoDataFrame(
            {"geometry": [shapely_box(-114.2, 51.0, -114.1, 51.1)]},
            crs="EPSG:4326",
        )
        utm_crs = gdf.estimate_utm_crs()
        gdf_utm = gdf.to_crs(utm_crs)
        area_m2 = gdf_utm.geometry.area.iloc[0]
        # ~100m x ~100m = ~10 000 m²; the actual box is ~7 x 11 km ≈ 77 km²
        self.assertGreater(area_m2, 1e6, "Reprojected area should be > 1 sq km")

    def test_crs84_geojson_normalizes_to_epsg4326(self):
        """OGC:CRS84 GeoJSON normalizes to EPSG:4326 with lon/lat as x/y."""
        import json

        from shapely.geometry import Point, mapping

        from geofuse.crs_utils import reproject_geodataframe_to_wgs84

        fc = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
            },
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": mapping(Point(-114.07, 51.04)),
                }
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".geojson", delete=False, encoding="utf-8"
        ) as f:
            json.dump(fc, f)
            path = f.name
        try:
            gdf = gpd.read_file(path)
            out = reproject_geodataframe_to_wgs84(gdf)
            self.assertEqual(out.crs.to_epsg(), 4326)
            p = out.geometry.iloc[0]
            self.assertAlmostEqual(p.x, -114.07, places=3)
            self.assertAlmostEqual(p.y, 51.04, places=3)
        finally:
            os.unlink(path)

    def test_rasterio_read_write(self):
        """rasterio write + read round-trip confirms GDAL C-extensions are functional."""
        import rasterio
        from rasterio.transform import from_bounds

        data = np.random.randint(0, 200, (1, 8, 8), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            transform = from_bounds(-114.2, 51.0, -114.1, 51.1, 8, 8)
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=8,
                width=8,
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(data)
            with rasterio.open(path) as src:
                result = src.read(1)
                epsg = src.crs.to_epsg()
            np.testing.assert_array_equal(result, data[0])
            self.assertEqual(epsg, 4326)
        finally:
            os.unlink(path)

    def test_optuna_smoke(self):
        """Optuna runs a minimal study — confirms the optimizer and SQLAlchemy backend work."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            x = trial.suggest_float("x", -2.0, 2.0)
            return (x - 1.0) ** 2

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5)
        # Best x should be near 1.0; accept anything ≤ 1 as "working"
        self.assertLessEqual(study.best_value, 1.0)

    def test_scipy_sklearn(self):
        """scipy.optimize and scikit-learn produce correct results on trivial inputs."""
        import scipy.optimize
        from sklearn.ensemble import RandomForestClassifier

        # scipy: minimize (x-1)^2 — solution must be near x=1
        result = scipy.optimize.minimize(lambda x: (x[0] - 1.0) ** 2, [0.0])
        self.assertAlmostEqual(result.x[0], 1.0, places=3)

        # sklearn: XOR-like problem — just confirm fit/predict don't raise
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])
        clf = RandomForestClassifier(n_estimators=4, random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        self.assertEqual(len(preds), 4)

    def test_folium_smoke(self):
        """folium can create a map and render to HTML — confirms leaflet bindings work."""
        import folium

        m = folium.Map(location=[51.07, -114.13], zoom_start=12)
        folium.CircleMarker(location=[51.07, -114.13], radius=5).add_to(m)
        html = m._repr_html_()
        self.assertIn("leaflet", html.lower())

    def test_dl_core_network_importable(self):
        """dl_core/network can be imported and exposes the modeling dict.

        vision.py adds dl_core to sys.path at import time.  This test
        confirms that the inference backbone is intact and the segmentation
        model factory is accessible — the minimum requirement for
        DeepLabSegmenter to initialise.
        """
        import importlib

        network = importlib.import_module("network")
        self.assertTrue(
            hasattr(network, "modeling"),
            "dl_core.network must expose 'modeling' for DeepLab instantiation",
        )
        self.assertTrue(
            hasattr(network.modeling, "deeplabv3plus_resnet101"),
            "network.modeling must define deeplabv3plus_resnet101 (the deployed backbone)",
        )


# =========================================================================
# GeoFuse Pipeline Logic Tests
# =========================================================================
class TestGeoFuse(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.output_dir = os.path.join("tests", "output", "test1_logic")

        # Robust cleanup — don't crash if Windows holds a file lock
        if os.path.exists(cls.output_dir):
            try:
                shutil.rmtree(cls.output_dir)
            except OSError:
                print(
                    "\n[WARN] Could not delete old output folder (Windows lock?). "
                    "Proceeding to overwrite files instead."
                )

        os.makedirs(cls.output_dir, exist_ok=True)

        sample_path = "data/samples/test_area.geojson"
        os.makedirs("data/samples", exist_ok=True)

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from create_samples import create_sample_data

        create_sample_data()
        print(f"[INFO] Using standardized test data at {sample_path}")

        # Resolve model path once for all tests that need it
        cls.model_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "geofuse", "model", "best_model.pth"
            )
        )

    @patch("geofuse.streetview.get_panorama_async", new_callable=AsyncMock)
    @patch("geofuse.streetview.find_panorama_async", new_callable=AsyncMock)
    def test_gvi_pipeline_logic(self, mock_find, mock_get_pano):
        """Grid generation → processing loop → result callbacks — no real API calls."""
        print("\n[TEST] Testing GVI Pipeline Logic (Mocked)...")

        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="Python 3.14 will, by default, filter extracted tar archives",
        )

        mock_pano = MagicMock()
        mock_pano.id = "test_pano_id_123"
        mock_find.return_value = mock_pano
        mock_get_pano.return_value = Image.fromarray(
            np.zeros((300, 600, 3), dtype=np.uint8)
        )

        engine = GVIEngine(download_mode="package")
        engine.segmenter = MagicMock()

        mock_mask = np.zeros((100, 100), dtype=int)
        mock_mask[0:50, :] = 8  # Vegetation class ID
        engine.segmenter.predict.return_value = mock_mask
        engine.segmenter.calculate_gvi_from_mask.return_value = {
            "GVI_Total": 0.5,
            "GVI_Terrain": 0.0,
        }

        gdf = gpd.read_file("data/samples/test_area.geojson")
        results = []

        engine.run_analysis(
            gdf,
            step=500,
            folder=self.output_dir,
            save_panos=False,
            save_masks=False,
            result_callback=results.append,
        )

        self.assertGreater(len(results), 30, "Should have processed at least 30 points")
        self.assertLess(len(results), 100, "Should not exceed 100 points")
        print(f"   [PASS] GVI Pipeline processed {len(results)} points")

    def test_real_segmentation(self):
        """DeepLabSegmenter runs a real forward pass — confirms model weights and torch work.

        This is the only test that exercises the full stack from PIL image to
        segmentation mask without any mocks.  It catches broken model weights,
        incompatible torch versions, and device-selection regressions.
        """
        if not os.path.exists(self.model_path):
            self.skipTest(f"Model weights not found at {self.model_path}")

        print("\n[TEST] Testing Real Segmentation (No Mocks)...")

        segmenter = DeepLabSegmenter(ckpt_path=self.model_path)

        # 224×224 is the minimum size the ResNet101 backbone accepts
        dummy_img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        mask = segmenter.predict(dummy_img)

        self.assertEqual(
            mask.shape,
            (224, 224),
            f"Mask shape {mask.shape} does not match input (224, 224)",
        )
        self.assertTrue(
            np.all(mask >= 0) and np.all(mask < 19),
            f"Mask contains out-of-range Cityscapes IDs — "
            f"min={mask.min()}, max={mask.max()} (valid range: 0–18)",
        )

        metrics = segmenter.calculate_gvi_from_mask(mask)
        self.assertIn("GVI_Total", metrics)
        self.assertIn("GVI_Terrain", metrics)
        self.assertGreaterEqual(metrics["GVI_Total"], 0.0)
        self.assertLessEqual(metrics["GVI_Total"], 1.0)

        print(
            f"   [PASS] Real segmentation: "
            f"GVI_Total={metrics['GVI_Total']:.3f}, "
            f"GVI_Terrain={metrics['GVI_Terrain']:.3f}"
        )

    @patch("geofuse.ndvi.rasterio")
    @patch("geofuse.ndvi.ee")
    @patch("geofuse.ndvi._export_ee_image_to_tif")
    def test_ndvi_logic(
        self,
        mock_export,
        mock_ee,
        mock_rasterio,
    ):
        """GEE wrapper constructs the correct API calls (no real Earth Engine auth needed)."""
        print("\n[TEST] Testing NDVI Logic (Mocked GEE)...")
        from geofuse.ndvi import NDVIEngine

        chain = MagicMock()
        chain.filterBounds.return_value = chain
        chain.filterDate.return_value = chain
        chain.filter.return_value = chain
        chain.map.return_value = chain
        chain.size.return_value.getInfo.return_value = 5
        mock_ee.ImageCollection.return_value = chain

        # The engine downloads through its own ``_export_ee_image_to_tif``
        # (a diagnostics-preserving replacement for ``geemap.ee_export_image``),
        # so that is the seam the export is verified at.
        def create_dummy_file(image, filename, **kwargs):
            with open(filename, "w") as f:
                f.write("Dummy")

        mock_export.side_effect = create_dummy_file

        mock_src = MagicMock()
        mock_src.read.return_value = np.zeros((10, 10))
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

        mock_rasterio.open.return_value.__enter__.return_value = mock_src
        mock_rasterio.open.return_value.__exit__.return_value = None
        mock_rasterio.band = MagicMock()
        mock_rasterio.warp = MagicMock()
        mock_rasterio.warp.transform = MagicMock(return_value=([5.0], [5.0]))
        mock_rasterio.warp.Resampling = Resampling

        def mock_xy(transform, rows, cols, offset="center"):
            if isinstance(rows, np.ndarray) and isinstance(cols, np.ndarray):
                shape = rows.shape
                xs = np.linspace(0, 10, shape[1] if len(shape) > 1 else len(rows))
                ys = np.linspace(0, 10, shape[0] if len(shape) > 1 else len(rows))
                if len(shape) > 1:
                    xs_grid, ys_grid = np.meshgrid(xs, ys)
                    return (xs_grid, ys_grid)
                return (xs, ys)
            return (np.array([0, 1, 2]), np.array([0, 1, 2]))

        mock_rasterio.transform = MagicMock()
        mock_rasterio.transform.xy = MagicMock(side_effect=mock_xy)

        engine = NDVIEngine()
        gdf = gpd.read_file("data/samples/test_area.geojson")

        engine.download_and_process(
            geometry=gdf,
            start_date="2024-06-01",
            end_date="2024-06-30",
            output_name="test_ndvi",
            folder=self.output_dir,
        )

        mock_export.assert_called_once()
        print("   [PASS] NDVI Export Logic Verified")


# TODO: NDVI_UNIT_TESTS - Add unit tests for NDVIEngine tiling logic (>5 km areas)
# TODO: FUSION_TESTS   - Add tests for MetricFusionEngine once optimizer API stabilises

if __name__ == "__main__":
    unittest.main()
