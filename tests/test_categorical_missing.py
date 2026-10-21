"""A missing categorical covariate value is handled like a missing numeric one.

One-hot coding without a missing level turns a missing value into all-zero
dummies, which is the reference level, and the row then passes the NaN filter
that removes a row missing a numeric covariate. Pinned here:

- by default (``categorical_missing="drop"``) the row is dropped, leaving the
  same entities as a numeric NaN in the same row;
- ``"indicator"`` keeps it under a ``<column>=missing`` dummy, with every level
  dummy 0;
- rows missing each covariate are counted and logged;
- a target without missing values is prepared exactly as before.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from geofuse import fusion
from geofuse.fusion import MetricFusionEngine


def _engine(cov_types, policy="drop", cols=None):
    return MetricFusionEngine(
        target_file="dummy.geojson",
        covariate_columns=cols or list(cov_types),
        covariate_types=cov_types,
        categorical_missing=policy,
        cache_dir=tempfile.mkdtemp(),
    )


def _frame(n=12, missing_at=(3,)):
    land = (["res", "com", "ind"] * n)[:n]
    for i in missing_at:
        land[i] = None
    return gpd.GeoDataFrame(
        {
            "income": np.arange(n, dtype=float),
            "land_use": land,
            "geometry": [Point(i, i) for i in range(n)],
        },
        crs="EPSG:4326",
    )


# ────────────────────────────────────────────────────────────────────
# Expansion of the covariates
# ────────────────────────────────────────────────────────────────────


class TestExpansion(unittest.TestCase):
    def test_drop_marks_the_missing_rows_nan(self):
        eng = _engine({"land_use": "categorical"})
        eng.target_gdf = _frame()
        eng._expand_categorical_covariates()
        dummies = eng._covariate_dummy_map["land_use"]
        block = eng.target_gdf[dummies]
        self.assertTrue(block.iloc[3].isna().all())
        self.assertFalse(block.drop(index=3).isna().any().any())
        self.assertFalse(any(c.endswith("=missing") for c in dummies))

    def test_drop_leaves_the_other_rows_coded_as_before(self):
        gdf = _frame()
        legacy = pd.get_dummies(
            gdf["land_use"], prefix="land_use", prefix_sep="=", drop_first=True
        ).astype(np.float64)
        eng = _engine({"land_use": "categorical"})
        eng.target_gdf = gdf
        eng._expand_categorical_covariates()
        got = eng.target_gdf[list(legacy.columns)]
        pd.testing.assert_frame_equal(
            got.drop(index=3), legacy.drop(index=3), check_names=False
        )

    def test_indicator_adds_a_missing_level(self):
        eng = _engine({"land_use": "categorical"}, policy="indicator")
        eng.target_gdf = _frame()
        eng._expand_categorical_covariates()
        dummies = eng._covariate_dummy_map["land_use"]
        self.assertIn("land_use=missing", dummies)
        self.assertIn("land_use=missing", eng.covariate_columns)
        row = eng.target_gdf.loc[3, dummies]
        self.assertEqual(row["land_use=missing"], 1.0)
        # No level of its own: the missing row is not the reference level.
        self.assertEqual(float(row.drop("land_use=missing").sum()), 0.0)
        self.assertEqual(
            float(eng.target_gdf["land_use=missing"].drop(index=3).sum()), 0.0
        )
        self.assertFalse(eng.target_gdf[dummies].isna().any().any())

    def test_without_missing_values_both_policies_give_the_legacy_dummies(self):
        gdf = _frame(missing_at=())
        legacy = pd.get_dummies(
            gdf["land_use"], prefix="land_use", prefix_sep="=", drop_first=True
        ).astype(np.float64)
        for policy in fusion.CATEGORICAL_MISSING:
            eng = _engine({"income": "numeric", "land_use": "categorical"}, policy)
            eng.target_gdf = gdf.copy()
            eng._expand_categorical_covariates()
            self.assertEqual(eng.covariate_columns, ["income", *legacy.columns])
            pd.testing.assert_frame_equal(
                eng.target_gdf[list(legacy.columns)], legacy, check_names=False
            )

    def test_one_level_and_missing_values(self):
        gdf = _frame()
        gdf["land_use"] = ["res" if v is not None else None for v in gdf["land_use"]]
        # Dropping: a single observed level carries nothing; the covariate goes.
        eng = _engine({"income": "numeric", "land_use": "categorical"})
        eng.target_gdf = gdf.copy()
        eng._expand_categorical_covariates()
        self.assertEqual(eng.covariate_columns, ["income"])
        # With a missing level there are two levels: observed and missing.
        eng = _engine({"income": "numeric", "land_use": "categorical"}, "indicator")
        eng.target_gdf = gdf.copy()
        eng._expand_categorical_covariates()
        self.assertEqual(eng.covariate_columns, ["income", "land_use=missing"])

    def test_a_level_named_missing_keeps_its_own_dummy(self):
        gdf = _frame()
        gdf["land_use"] = gdf["land_use"].replace({"res": "missing"})
        eng = _engine({"land_use": "categorical"}, policy="indicator")
        eng.target_gdf = gdf
        eng._expand_categorical_covariates()
        dummies = eng._covariate_dummy_map["land_use"]
        self.assertIn("land_use=missing", dummies)
        self.assertIn("land_use=<missing>", dummies)
        self.assertEqual(eng.target_gdf.loc[3, "land_use=<missing>"], 1.0)
        self.assertEqual(eng.target_gdf.loc[3, "land_use=missing"], 0.0)

    def test_an_unknown_policy_is_refused(self):
        with self.assertRaises(ValueError):
            _engine({"land_use": "categorical"}, policy="impute")


class TestCounts(unittest.TestCase):
    def test_each_covariate_is_counted(self):
        gdf = _frame(missing_at=(3, 5))
        gdf.loc[7, "income"] = np.nan
        eng = _engine({"income": "numeric", "land_use": "categorical"})
        eng.target_gdf = gdf
        eng._expand_categorical_covariates()
        report = eng._covariate_missing_report
        self.assertEqual(report["categorical_missing"], "drop")
        self.assertEqual(report["n_rows"], 12)
        by_col = {r["column"]: r for r in report["covariates"]}
        self.assertEqual(by_col["land_use"]["n_missing"], 2)
        self.assertEqual(by_col["land_use"]["type"], "categorical")
        self.assertEqual(by_col["income"]["n_missing"], 1)
        self.assertEqual(by_col["income"]["handling"], "drop")

    def test_the_counts_are_logged(self):
        for policy, words in (
            ("drop", "dropped"),
            ("indicator", "coded as level 'land_use=missing'"),
        ):
            eng = _engine({"land_use": "categorical"}, policy)
            eng.target_gdf = _frame(missing_at=(3, 5))
            with mock.patch.object(fusion, "_log") as log:
                eng._expand_categorical_covariates()
            lines = [c.args[1] for c in log.call_args_list]
            self.assertTrue(
                any("'land_use': 2 row(s) missing" in s and words in s for s in lines),
                lines,
            )

    def test_no_covariates_no_report(self):
        eng = MetricFusionEngine(
            target_file="dummy.geojson", cache_dir=tempfile.mkdtemp()
        )
        eng.target_gdf = _frame()
        eng._expand_categorical_covariates()
        self.assertIsNone(eng._covariate_missing_report)


# ────────────────────────────────────────────────────────────────────
# The vector prep on a small synthetic study
# ────────────────────────────────────────────────────────────────────


class TestPrep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.tmp = tempfile.mkdtemp()
        x0, y0 = -114.10, 51.00
        n = 20
        pts = [Point(x0 + 0.004 * (i % 5), y0 + 0.004 * (i // 5)) for i in range(n)]
        base = gpd.GeoDataFrame(
            {
                "outcome": rng.normal(size=n),
                "age": rng.normal(60, 8, n),
                "sex": (["f", "m"] * n)[:n],
            },
            geometry=pts,
            crs="EPSG:4326",
        )
        cls.paths = {"base": os.path.join(cls.tmp, "base.gpkg")}
        base.to_file(cls.paths["base"], driver="GPKG")
        cat = base.copy()
        cat["sex"] = cat["sex"].astype(object)
        cat.loc[3, "sex"] = None
        cls.paths["cat"] = os.path.join(cls.tmp, "cat.gpkg")
        cat.to_file(cls.paths["cat"], driver="GPKG")
        num = base.copy()
        num.loc[3, "age"] = np.nan
        cls.paths["num"] = os.path.join(cls.tmp, "num.gpkg")
        num.to_file(cls.paths["num"], driver="GPKG")

        cls.npath = os.path.join(cls.tmp, "n.tif")
        with rasterio.open(
            cls.npath,
            "w",
            driver="GTiff",
            height=160,
            width=160,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(x0 - 0.01, y0 + 0.04, 0.0004, 0.0004),
        ) as d:
            d.write(rng.random((160, 160)).astype("float32"), 1)
        gpts = [
            Point(x0 - 0.005 + rng.random() * 0.04, y0 - 0.005 + rng.random() * 0.03)
            for _ in range(600)
        ]
        gvi = gpd.GeoDataFrame(
            {"gvi_veg": rng.random(600), "gvi_ter": rng.random(600)},
            geometry=gpts,
            crs="EPSG:4326",
        )
        cls.gpath = os.path.join(cls.tmp, "g.gpkg")
        gvi.to_file(cls.gpath, driver="GPKG")
        cls._runs = 0

    def _prep(self, which, policy="drop"):
        type(self)._runs += 1
        eng = MetricFusionEngine(
            target_file=self.paths[which],
            target_feature="outcome",
            buffer_meters=400,
            gvi_buffer_min_m=200,
            gvi_buffer_max_m=400,
            gvi_buffer_step_m=200,
            ndvi_buffer_min_m=200,
            ndvi_buffer_max_m=400,
            ndvi_buffer_step_m=200,
            cache_dir=os.path.join(self.tmp, f"cache{self._runs}"),
            cgi_grid_spacing_m=100,
            covariate_columns=["age", "sex"],
            covariate_types={"age": "numeric", "sex": "categorical"},
            categorical_missing=policy,
        )
        eng.load_target()
        eng.load_metrics(veg_file=self.gpath, ndvi_file=self.npath, cache_metrics=False)
        df = eng.prepare_fusion_data()
        return eng, df

    @staticmethod
    def _entities(df):
        return set(df["polygon_id"].astype(int).unique())

    def test_a_missing_category_drops_the_row_like_a_missing_number(self):
        _, base = self._prep("base")
        eng, cat = self._prep("cat")
        _, num = self._prep("num")
        self.assertIn(3, self._entities(base))
        self.assertEqual(self._entities(cat), self._entities(base) - {3})
        self.assertEqual(self._entities(cat), self._entities(num))
        self.assertEqual(eng._covariate_missing_report["rows_dropped"], 1)

    def test_the_indicator_keeps_the_row_under_its_own_level(self):
        _, base = self._prep("base")
        eng, cat = self._prep("cat", policy="indicator")
        self.assertEqual(self._entities(cat), self._entities(base))
        self.assertIn("sex=missing", eng.covariate_columns)
        rows = cat[cat["polygon_id"].astype(int) == 3]
        self.assertTrue((rows["sex=missing"] == 1.0).all())
        self.assertTrue((rows["sex=m"] == 0.0).all())
        self.assertEqual(float(cat["sex=missing"].sum()), float(len(rows)))
        self.assertEqual(eng._covariate_missing_report["rows_dropped"], 0)

    def test_a_complete_target_is_prepared_alike_under_both_policies(self):
        eng_d, drop = self._prep("base", policy="drop")
        eng_i, ind = self._prep("base", policy="indicator")
        self.assertEqual(eng_d.covariate_columns, eng_i.covariate_columns)
        self.assertEqual(eng_d.covariate_columns, ["age", "sex=m"])
        pd.testing.assert_frame_equal(drop, ind)


if __name__ == "__main__":
    unittest.main()
