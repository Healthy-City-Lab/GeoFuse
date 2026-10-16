"""Tests for the fusion results and categorical-covariate fixes.

Covers the one-hot expansion of categorical covariates on the engine.
"""

import os
import sys
import tempfile
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


class TestCategoricalCovariateExpansion(unittest.TestCase):
    def _engine(self, cov_cols, cov_types):
        from geofuse.fusion import MetricFusionEngine

        tmp = tempfile.mkdtemp()
        return MetricFusionEngine(
            target_file="dummy.geojson",
            covariate_columns=cov_cols,
            covariate_types=cov_types,
            cache_dir=tmp,
        )

    def _gdf(self, n=12):
        rng = np.random.default_rng(0)
        return gpd.GeoDataFrame(
            {
                "income": rng.normal(50, 10, n),
                "land_use": (["res", "com", "ind"] * n)[:n],
                "ses_band": ([1, 2, 3, 4] * n)[:n],
                "geometry": [Point(i, i) for i in range(n)],
            },
            crs="EPSG:4326",
        )

    def test_categorical_replaced_by_dummies(self):
        eng = self._engine(
            ["income", "land_use"], {"income": "numeric", "land_use": "categorical"}
        )
        eng.target_gdf = self._gdf()
        eng._expand_categorical_covariates()
        # income stays; land_use becomes drop-first dummies
        self.assertIn("income", eng.covariate_columns)
        self.assertNotIn("land_use", eng.covariate_columns)
        dummies = [c for c in eng.covariate_columns if c.startswith("land_use=")]
        self.assertEqual(len(dummies), 2)  # 3 levels, drop-first
        for c in dummies:
            self.assertTrue(pd.api.types.is_numeric_dtype(eng.target_gdf[c]))
        self.assertEqual(eng._covariate_dummy_map["land_use"], dummies)

    def test_numeric_coded_categorical(self):
        # An int-coded band tagged categorical is one-hot encoded.
        eng = self._engine(["ses_band"], {"ses_band": "categorical"})
        eng.target_gdf = self._gdf()
        eng._expand_categorical_covariates()
        self.assertNotIn("ses_band", eng.covariate_columns)
        self.assertTrue(all(c.startswith("ses_band=") for c in eng.covariate_columns))

    def test_untagged_non_numeric_raises(self):
        eng = self._engine(["land_use"], {"land_use": "numeric"})
        eng.target_gdf = self._gdf()
        with self.assertRaises(ValueError):
            eng._expand_categorical_covariates()

    def test_all_numeric_unchanged(self):
        eng = self._engine(["income", "ses_band"], {})
        eng.target_gdf = self._gdf()
        before = list(eng.covariate_columns)
        eng._expand_categorical_covariates()
        self.assertEqual(eng.covariate_columns, before)
        self.assertEqual(eng._covariate_dummy_map, {})

    def test_idempotent(self):
        eng = self._engine(["land_use"], {"land_use": "categorical"})
        eng.target_gdf = self._gdf()
        eng._expand_categorical_covariates()
        cols = list(eng.covariate_columns)
        eng._expand_categorical_covariates()  # second call is a no-op
        self.assertEqual(eng.covariate_columns, cols)


class TestRasterTargetExtent(unittest.TestCase):
    def test_a_projected_raster_is_buffered_in_metres(self):
        import rasterio
        from rasterio.transform import from_origin

        from geofuse.fusion import MetricFusionEngine

        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "target.tif")
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=10,
            height=10,
            count=1,
            dtype="float32",
            crs="EPSG:32617",
            transform=from_origin(500000, 4800000, 100, 100),
        ) as dst:
            dst.write(np.ones((1, 10, 10), dtype="float32"))
        eng = MetricFusionEngine(target_file=path, cache_dir=tmp)
        eng.buffer_meters = 1500
        extent = eng.load_target()
        self.assertEqual(extent.crs.to_epsg(), 4326)
        bounds = extent.to_crs("EPSG:32617").total_bounds
        np.testing.assert_allclose(
            bounds, [498500, 4797500, 502500, 4801500], atol=1.0
        )


class TestDispersionCheck(unittest.TestCase):
    def test_rare_dummies_and_binary_outcomes_are_not_flagged(self):
        from geofuse.fusion import MetricFusionEngine

        eng = object.__new__(MetricFusionEngine)
        eng.target_feature = "event"
        eng.covariate_columns = ["area=b", "income"]
        rng = np.random.default_rng(1)
        n = 1000
        income = rng.normal(50, 10, n)
        income[0] = 5e6  # a sentinel the check exists to catch
        df = pd.DataFrame(
            {
                "event": (rng.random(n) < 0.05).astype(float),
                "area=b": (rng.random(n) < 0.03).astype(float),
                "income": income,
            }
        )
        report = eng.check_covariate_dispersion(data=df)
        self.assertEqual(report["flagged"], ["income"])


class TestCollinearityCheck(unittest.TestCase):
    @staticmethod
    def _engine(formula):
        from geofuse.fusion import MetricFusionEngine

        eng = object.__new__(MetricFusionEngine)
        eng.cgi_formula = formula
        eng._active_greenery_channel = "cgi"
        return eng

    def test_a_two_channel_study_checks_ndvi_against_gvi(self):
        rng = np.random.default_rng(0)
        veg, ter = rng.random(500), rng.random(500)
        ndvi = veg + ter + 0.01 * rng.normal(size=500)
        df = pd.DataFrame({"veg": veg, "terrain": ter, "ndvi": ndvi})
        eng = self._engine("weighted_average_gvi")
        report = eng.check_channel_collinearity(data=df)
        self.assertEqual(report["channels_in"], ["ndvi", "gvi"])
        self.assertEqual(len(eng._high_vif_channels), 1)

    def test_a_one_channel_study_has_nothing_to_check(self):
        eng = self._engine("weighted_average_gvi")
        eng._active_greenery_channel = "ndvi"
        df = pd.DataFrame({"veg": [0.1, 0.2], "terrain": [0.0, 0.1], "ndvi": [0.3, 0.6]})
        self.assertIsNone(eng.check_channel_collinearity(data=df))


if __name__ == "__main__":
    unittest.main()
