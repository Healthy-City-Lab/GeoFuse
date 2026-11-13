"""Keller & Szpiro (2020) scale selection and the effective bandwidth.

Pinned here:

- the criteria are statsmodels' OLS AIC and BIC;
- a confounder that is a coarse surface, an exposure that adds a finer field:
  AIC-NE picks a scale at or finer than the confounder's in most replicates
  and removes most of the bias, BIC-NE picks a coarser or equal scale;
- the selection reads the train+val rows only: permuting the test rows'
  outcomes leaves every choice unchanged;
- ``k̂`` (both rules) falls as the basis scale falls;
- the summary counts each scale's wins per criterion;
- a ``scale_ic`` run reports the chosen scale's rows as ``selected`` beside
  the ladder at every candidate, and writes ``scale_selection.csv`` and
  ``effective_bandwidth.csv``.
"""

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas  # noqa: F401  (pyproj before torchvision, see vision.py)
import numpy as np
import pandas as pd

from geofuse import spatial_scale_selection as ss

SCALES = (40000, 20000, 10000, 5000, 2500)


def _field(seed, n=1500):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 40_000
    z = np.sin(2 * np.pi * xy[:, 0] / 40_000) + np.cos(2 * np.pi * xy[:, 1] / 40_000)
    fine = np.sin(2 * np.pi * xy[:, 0] / 3_000) * np.cos(2 * np.pi * xy[:, 1] / 3_000)
    x = z + fine + 0.3 * rng.normal(size=n)
    y = 0.3 * x + z + rng.normal(size=n)
    return xy, x, y, rng


class CriteriaTests(unittest.TestCase):
    def test_statsmodels_aic_and_bic(self):
        import statsmodels.api as sm

        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4))
        X[:, 3] = X[:, 0] + X[:, 1]
        y = X[:, 0] + rng.normal(size=200)
        ours = ss.information_criteria(y, X)
        fit = sm.OLS(y, sm.add_constant(X)).fit()
        self.assertAlmostEqual(ours["aic"], fit.aic, delta=1e-8)
        self.assertAlmostEqual(ours["bic"], fit.bic, delta=1e-8)


class SelectionTests(unittest.TestCase):
    def test_aic_ne_finds_the_confounder_and_bic_ne_is_coarser(self):
        finer, removed = 0, 0
        reps = 10
        for rep in range(reps):
            xy, x, y, _rng = _field(rep)
            rows = ss.profile(xy, {"y": y}, None, SCALES, eps_m=1e6, min_component=1)
            aic = ss.chosen(rows, "y", "aic")
            bic = ss.chosen(rows, "y", "bic")
            self.assertIsNotNone(aic)
            if aic <= 20_000:
                finer += 1
            self.assertGreaterEqual(bic or np.inf, aic)
            bases = ss.candidate_bases(xy, [aic], eps_m=1e6, min_component=1)
            S = ss._columns(bases[aic])

            def slope(design):
                parts = [np.ones(len(x)), x] + ([] if design is None else [design])
                D = np.column_stack(parts)
                return np.linalg.lstsq(D, y, rcond=None)[0][1]

            if abs(slope(S) - 0.3) < 0.5 * abs(slope(None) - 0.3):
                removed += 1
        self.assertGreaterEqual(finer, 8)
        self.assertGreaterEqual(removed, 8)

    def test_the_summary_counts_the_wins(self):
        rows = []
        picks = ((5000.0, 20000.0), (5000.0, None), (2500.0, None))
        for split, (aic, bic) in enumerate(picks):
            for scale in (None, 20000.0, 5000.0, 2500.0):
                rows.append(
                    {
                        "split": split,
                        "stage": "reporting",
                        "outcome": "target",
                        "scale_m": scale,
                        "chosen_aic": scale == aic,
                        "chosen_bic": scale == bic,
                    }
                )
        (entry,) = ss.summary(rows, "bic")
        self.assertEqual(entry["aic_wins"], {"5000": 2, "2500": 1})
        self.assertEqual(entry["bic_wins"], {"20000": 1, "none": 2})
        self.assertEqual(entry["bic_median_m"], 0.0)


class EngineTests(unittest.TestCase):
    def _engine(self):
        from geofuse.fusion import MetricFusionEngine

        eng = MetricFusionEngine(
            target_file="dummy.geojson",
            cache_dir=tempfile.mkdtemp(),
            spatial_adjust_method="basis",
            spatial_adjust_select="scale_ic",
            spatial_select_scales_m=list(SCALES),
            spatial_adjust_eps_m=1e6,
            spatial_adjust_min_component=1,
        )
        xy, x, y, rng = _field(3, n=1200)
        frame = pd.DataFrame(
            {
                "polygon_id": np.arange(len(y)),
                "_cx": xy[:, 0],
                "_cy": xy[:, 1],
                "target": y,
            }
        )
        test = rng.random(len(y)) < 0.25
        eng.train_val_data = frame[~test].copy()
        eng.test_data = frame[test].copy()
        return eng

    def test_the_test_rows_never_enter(self):
        eng = self._engine()
        eng._select_scales()
        before = (eng._spatial_scale, list(eng._scale_profile))
        eng.test_data["target"] = np.random.default_rng(9).permutation(
            eng.test_data["target"].to_numpy()
        )
        eng._select_scales()
        self.assertEqual(before[0], eng._spatial_scale)
        self.assertEqual(before[1], eng._scale_profile)
        self.assertIn(eng._spatial_scale, SCALES)

    def test_the_tuning_scale_is_chosen_per_split(self):
        from geofuse.fusion import MetricFusionEngine

        eng = MetricFusionEngine(
            target_file="dummy.geojson",
            cache_dir=tempfile.mkdtemp(),
            spatial_adjust_method="basis",
            spatial_adjust_select="fixed_scale",
            spatial_adjust_scale_m=[5000.0],
            spatial_tune_select="scale_ic",
            spatial_select_scales_m=list(SCALES),
            spatial_adjust_eps_m=1e6,
            spatial_adjust_min_component=1,
        )
        ref = self._engine()
        eng.train_val_data, eng.test_data = ref.train_val_data, ref.test_data
        eng._select_scales()
        ref._select_scales()
        self.assertEqual(eng.spatial_tune_scale_m, ref._spatial_scale)
        self.assertEqual(eng._spatial_scale, 5000.0)
        stages = {r["stage"] for r in eng._scale_profile}
        self.assertEqual(stages, {"reporting", "tuning"})

    def test_the_bandwidth_falls_with_the_scale(self):
        eng = self._engine()
        rows = eng.effective_bandwidths(seed=1)
        for rule in ("k_ks", "k_rk"):
            ks = [r[rule] for r in rows]
            self.assertTrue(all(k is not None for k in ks), rule)
            falling = all(a > b for a, b in zip(ks, ks[1:], strict=False))
            self.assertTrue(falling, (rule, ks))


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_fusion_run import _write_study, run_study

        cls.folder = tempfile.mkdtemp(prefix="geofuse-scale-ic-")
        _write_study(cls.folder)
        cls.ctx, cls.bundle = run_study(
            cls.folder,
            """
            mode = "cross-sectional"
            ndvi = "ndvi_2015.tif"
            gvi = "gvi.gpkg"
            """,
            spatial_adjust_method="basis",
            spatial_adjust_select="scale_ic",
            spatial_select_scales_m=[2000, 1000, 500],
            spatial_adjust_eps_m=50_000.0,
            outer_splits=2,
            report_paired_bootstrap=20,
        )

    def test_selected_rows_and_the_ladder(self):
        rows = self.bundle["outer_splits"]["splits"]
        beta = [r for r in rows if r["quantity"] == "held_out_beta"]
        scales = {r.get("spatial_scale_m") for r in beta}
        self.assertIn("selected", scales)
        self.assertTrue({"2000", "1000", "500", "none"} <= scales)
        selected = [r for r in beta if r.get("spatial_scale_m") == "selected"]
        self.assertEqual({r["split"] for r in selected}, {0, 1})
        self.assertTrue(all("spatial_selected_scale_m" in r for r in selected))

    def test_the_tables(self):
        folder = self.bundle["report_dir"]
        for name in ("scale_selection.csv", "effective_bandwidth.csv"):
            self.assertTrue(os.path.exists(os.path.join(folder, name)), name)
        table = pd.read_csv(os.path.join(folder, "scale_selection.csv"))
        self.assertEqual(set(table["split"]), {0, 1})
        self.assertEqual(int(table["chosen_bic"].sum()), 2 * table["outcome"].nunique())


if __name__ == "__main__":
    unittest.main()
