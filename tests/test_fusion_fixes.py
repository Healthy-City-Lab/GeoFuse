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

from geofuse import cgi_formulas as cf


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


if __name__ == "__main__":
    unittest.main()
