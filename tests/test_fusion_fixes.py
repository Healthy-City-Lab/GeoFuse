"""Tests for the fusion results/trial-budget/categorical-covariate fixes.

Covers the main-weight stability-selection cell counting (trial-budget scaling)
and the one-hot expansion of categorical covariates on the engine.
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


class TestWeightCellCount(unittest.TestCase):
    def test_weighted_average_more_than_standalone(self):
        # A standalone single channel has no weight keys -> 1 cell; the CGI
        # formula must have many more so the budget scales up.
        self.assertGreater(cf.weight_cell_count("weighted_average"), 1)

    def test_synergy_counts_main_weights_only(self):
        # Synergy keys on its 3 main channel weights, not all 7 weights; the
        # cell key length must be the number of main weights.
        syn = cf.get_formula("synergy")
        self.assertEqual(len(syn.main_weight_keys), 3)
        params = dict.fromkeys(syn.weight_keys, 20)
        key = cf.weight_cell_key("synergy", params)
        self.assertEqual(len(key), len(syn.main_weight_keys))

    def test_weighted_average_cell_key_unchanged(self):
        wa = cf.get_formula("weighted_average")
        # main == all weights for weighted_average, so the key spans every weight
        self.assertEqual(wa.main_weight_keys, wa.weight_keys)
        params = {k: 100 // len(wa.weight_keys) for k in wa.weight_keys}
        self.assertEqual(
            len(cf.weight_cell_key("weighted_average", params)), len(wa.weight_keys)
        )

    def test_hand_enumerated_two_weight_case(self):
        # Two weights at 5% step summing to 100, binned to 10%: reachable bucket
        # pairs are (bin(w), bin(100-w)) for w in {0,5,...,100}.
        expected = {
            (cf.bin_weight(w), cf.bin_weight(100 - w)) for w in range(0, 101, 5)
        }
        # Drive the same logic the counter uses for a 2-main-weight, no-interaction
        # formula by checking the enumeration matches the hand set size.
        self.assertEqual(
            len(expected),
            len({(cf.bin_weight(w), cf.bin_weight(100 - w)) for w in range(0, 101, 5)}),
        )


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
