"""Sampling weights in the reported estimates.

Pinned here:

- constant weights reproduce every unweighted point estimate (slope in SD
  and raw units, incremental R², D, ΔD, the search's residualising);
- integer weights give the estimates of the data with each row repeated that
  many times;
- in a population whose effect differs between two groups, sampled with
  unequal probabilities, the weighted slope recovers the population slope with
  its intervals covering it, while the unweighted one recovers the sample's;
- rows without a usable weight are left out and counted, and Kish's n_eff is
  reported;
- a run adds weighted rows next to the unweighted ones on every split, the
  unweighted rows untouched; constant weights reproduce the unweighted
  estimates; a weighted search with constant weights is the unweighted search;
  without a weight column nothing is added.
"""

import csv
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas as gpd
import numpy as np

from geofuse import bayesian_index as bi
from geofuse import negative_controls as nc
from geofuse import objective_scoring as osc
from geofuse import settings as S
from geofuse import weighting as wt


def _data(seed=1, n=600):
    rng = np.random.default_rng(seed)
    cov = rng.normal(size=(n, 2))
    x = rng.normal(size=n) + 0.3 * cov[:, 0]
    other = 0.6 * x + rng.normal(size=n)
    y = 0.4 * x + cov @ [0.5, -0.2] + rng.normal(size=n)
    control = 0.1 * x + rng.normal(size=n)
    return rng, cov, x, other, y, control


class TestConstantWeights(unittest.TestCase):
    def setUp(self):
        _, self.cov, self.x, self.other, self.y, self.c = _data()
        self.w = np.full(len(self.x), 3.0)

    def test_slopes(self):
        s = wt.slope(self.x, self.y, self.cov, self.w, scale=2.0)
        self.assertAlmostEqual(
            s["beta"], nc.partial_slope(self.x, self.y, self.cov)["beta"], places=12
        )
        raw = nc.raw_slope(self.x, self.y, self.cov, scale=2.0)
        self.assertAlmostEqual(s["estimate"], raw["estimate"], places=12)
        self.assertAlmostEqual(s["n_eff"], len(self.x))

    def test_scores_and_contrasts(self):
        r2 = wt.r2(self.y, self.x, self.cov, None, self.w)["estimate"]
        self.assertAlmostEqual(r2, osc.score("r2", self.y, self.x, self.cov), 12)
        d = wt.d(self.x, self.y, self.c, self.cov, None, None, self.w)["estimate"]
        ref = nc.paired_contrast(self.x, self.y, self.c, self.cov, n_boot=0)
        self.assertAlmostEqual(d, ref["delta"], places=12)
        dd = wt.delta_d(self.x, self.other, self.y, self.c, self.cov, None, self.w)
        ref = nc.specificity_contrast(
            self.x, self.other, self.y, self.c, self.cov, n_boot=0
        )
        self.assertAlmostEqual(dd["estimate"], ref["delta"], places=12)

    def test_the_search_residualising(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(len(self.x), 2, 3, 2))
        a = bi.prep(X, self.y, self.cov)
        b = bi.prep(X, self.y, self.cov, weights=self.w)
        np.testing.assert_allclose(a[0], b[0], atol=1e-12)
        np.testing.assert_allclose(a[1], b[1], atol=1e-12)


class TestIntegerWeights(unittest.TestCase):
    def setUp(self):
        rng, self.cov, self.x, self.other, self.y, self.c = _data(2)
        self.k = rng.integers(1, 4, len(self.x))
        self.rep = np.repeat(np.arange(len(self.x)), self.k)

    def test_point_estimates_equal_repeated_rows(self):
        r = self.rep
        s = wt.slope(self.x, self.y, self.cov, self.k)
        self.assertAlmostEqual(
            s["beta"], nc.partial_slope(self.x[r], self.y[r], self.cov[r])["beta"], 12
        )
        r2 = wt.r2(self.y, self.x, self.cov, None, self.k)["estimate"]
        self.assertAlmostEqual(
            r2, osc.score("r2", self.y[r], self.x[r], self.cov[r]), places=12
        )
        d = wt.d(self.x, self.y, self.c, self.cov, None, None, self.k)["estimate"]
        ref = nc.paired_contrast(self.x[r], self.y[r], self.c[r], self.cov[r], n_boot=0)
        self.assertAlmostEqual(d, ref["delta"], places=12)

    def test_the_search_gram_equals_repeated_rows(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(len(self.x), 2, 2, 2))
        r = self.rep
        a = bi.prep(X[r], self.y[r], self.cov[r])
        b = bi.prep(X, self.y, self.cov, weights=self.k)
        ga = (a[0][:, None] * a[1].reshape(len(r), -1)).mean(0)
        gb = (b[0][:, None] * b[1].reshape(len(self.x), -1)).mean(0)
        np.testing.assert_allclose(ga, gb, atol=1e-12)


class TestInformativeDesign(unittest.TestCase):
    """A population where the slope is 1.0 in a group of 30 % and 0.2 in the
    rest; the samples take the first group ten times as often."""

    def test_the_weighted_slope_recovers_the_population(self):
        rng = np.random.default_rng(0)
        N = 200_000
        group = rng.random(N) < 0.3
        x = rng.normal(size=N)
        y = np.where(group, 1.0, 0.2) * x + rng.normal(size=N)
        population = np.polyfit(x, y, 1)[0]
        p = np.where(group, 0.02, 0.002)
        weighted, unweighted, covered = [], [], 0
        for r in range(200):
            take = np.random.default_rng(100 + r).random(N) < p
            s = wt.slope(x[take], y[take], None, 1.0 / p[take])
            weighted.append(s["estimate"])
            unweighted.append(nc.raw_slope(x[take], y[take])["estimate"])
            covered += s["estimate_ci_low"] <= population <= s["estimate_ci_high"]
        self.assertAlmostEqual(np.mean(weighted), population, delta=0.01)
        sample = (0.3 * 0.02 * 1.0 + 0.7 * 0.002 * 0.2) / (0.3 * 0.02 + 0.7 * 0.002)
        self.assertAlmostEqual(np.mean(unweighted), sample, delta=0.02)
        self.assertGreaterEqual(covered / 200, 0.88)


class TestWeights(unittest.TestCase):
    def test_unusable_weights_are_dropped_and_counted(self):
        _, cov, x, _, y, _ = _data(3)
        w = np.ones(len(x))
        w[:10], w[10:15] = np.nan, 0.0
        s = wt.slope(x, y, cov, w)
        self.assertEqual(s["n_dropped"], 15)
        self.assertEqual(s["n"], len(x) - 15)
        ref = nc.partial_slope(x[15:], y[15:], cov[15:])
        self.assertAlmostEqual(s["beta"], ref["beta"], places=12)

    def test_kish(self):
        self.assertAlmostEqual(wt.kish(np.ones(50)), 50.0)
        self.assertAlmostEqual(wt.kish(np.array([1.0, 1.0, 2.0])), 16 / 6)

    def test_a_weighted_search_needs_weights_and_the_linear_form(self):
        with self.assertRaises(S.SettingError):
            S.check("fusion", {"weight_search": True, "weight_column": ""})
        with self.assertRaises(S.SettingError):
            S.check(
                "fusion",
                {"weight_search": True, "weight_column": "w", "index_form": "sweep"},
            )
        out = S.check(
            "fusion",
            {"weight_search": True, "weight_column": "w", "index_form": "linear"},
        )
        self.assertTrue(out["weight_search"])


INPUTS = """
    mode = "cross-sectional"
    ndvi = "ndvi_2015.tif"
    gvi = "gvi.gpkg"
    """

COMPARATORS = {
    "comparators": ["ndvi_here", "noise"],
    "comparators_primary": ["ndvi_here"],
}


def _study() -> str:
    """The repeated-split study with a weight column and a constant one."""
    from test_outer_splits import _write_smooth_study

    folder = tempfile.mkdtemp(prefix="geofuse-weights-")
    _write_smooth_study(folder)
    path = os.path.join(folder, "cohort.gpkg")
    cohort = gpd.read_file(path)
    rng = np.random.default_rng(11)
    cohort["wt"] = rng.uniform(0.5, 2.0, len(cohort))
    cohort["unit"] = 3.0
    cohort.to_file(path, driver="GPKG")
    return folder


def _by(rows, weighting=None):
    return {
        (r["split"], r["quantity"], r["exposure"], r.get("control")): r
        for r in rows
        if r.get("weighting") == weighting
    }


class TestWeightedRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_fusion_run import run_study

        cls.folder = _study()
        kw = {"outer_splits": 2, "specificity_contrast": True, **COMPARATORS}
        cls.plain = run_study(cls.folder, INPUTS, **kw)[1]
        cls.unit = run_study(cls.folder, INPUTS, weight_column="unit", **kw)[1]
        cls.weighted = run_study(cls.folder, INPUTS, weight_column="wt", **kw)[1]
        linear = {"index_form": "linear", "outer_splits": 2}
        cls.search_plain = run_study(cls.folder, INPUTS, **linear)[1]
        cls.search_unit = run_study(
            cls.folder, INPUTS, weight_column="unit", weight_search=True, **linear
        )[1]

    def test_unweighted_rows_are_untouched(self):
        plain = _by(self.plain["outer_splits"]["splits"])
        mine = _by(self.weighted["outer_splits"]["splits"])
        self.assertEqual(set(plain), set(mine))
        for key, row in plain.items():
            self.assertEqual(mine[key]["estimate"], row["estimate"], key)

    def test_constant_weights_reproduce_the_unweighted_estimates(self):
        plain = _by(self.unit["outer_splits"]["splits"])
        weighted = _by(self.unit["outer_splits"]["splits"], "weighted")
        compared = 0
        for key, row in weighted.items():
            if key[1] in ("negative_control_d",) or key not in plain:
                continue
            ref = plain[key]["estimate"]
            if ref is None or row["estimate"] is None:
                continue
            self.assertAlmostEqual(row["estimate"], ref, places=8, msg=key)
            compared += 1
        self.assertGreater(compared, 20)
        for row in weighted.values():
            self.assertAlmostEqual(row["n_eff"], row["n"], places=6)

    def test_every_split_has_weighted_rows(self):
        rows = [
            r for r in self.weighted["outer_splits"]["splits"] if r.get("weighting")
        ]
        self.assertEqual({r["split"] for r in rows}, {0, 1})
        quantities = {r["quantity"] for r in rows}
        for q in (
            "held_out_score",
            "held_out_beta",
            "negative_control_beta",
            "negative_control_d",
            "cgi_minus_channel",
            "cgi_minus_comparator",
            "specificity_difference",
            "held_out_per_iqr",
            "exposure_per_iqr",
        ):
            self.assertIn(q, quantities)
        for r in rows:
            self.assertLess(r["n_eff"], r["n"])
        weighted = [
            e for e in self.weighted["outer_splits"]["summary"] if e.get("weighting")
        ]
        self.assertTrue(weighted)
        beta = next(
            e
            for e in weighted
            if (e["quantity"], e["exposure"]) == ("held_out_beta", "cgi")
        )
        self.assertIsNotNone(beta["corrected_p"])

    def test_the_tables_and_record(self):
        report = self.weighted["report_dir"]
        with open(os.path.join(report, "weighted.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(r["weighting"] == "weighted" for r in rows))
        with open(os.path.join(report, "run_config.json"), encoding="utf-8") as f:
            record = json.load(f)
        self.assertEqual(record["weight_column"], "wt")
        self.assertFalse(record["weight_search"])
        with open(os.path.join(report, "results_summary.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["weighted"]["weight_column"], "wt")
        plain = self.plain["report_dir"]
        self.assertFalse(os.path.exists(os.path.join(plain, "weighted.csv")))
        with open(os.path.join(plain, "run_config.json"), encoding="utf-8") as f:
            self.assertNotIn("weight_column", json.load(f))

    def test_a_constant_weight_search_is_the_unweighted_search(self):
        a = _by(self.search_plain["outer_splits"]["splits"])
        b = _by(self.search_unit["outer_splits"]["splits"])
        for key in a:
            if key[1] in ("held_out_beta", "posterior_beta", "posterior_weight"):
                self.assertAlmostEqual(
                    b[key]["estimate"], a[key]["estimate"], places=6, msg=key
                )
        with open(
            os.path.join(self.search_unit["report_dir"], "run_config.json"),
            encoding="utf-8",
        ) as f:
            self.assertTrue(json.load(f)["weight_search"])

    def test_the_command_line_prints_both(self):
        from pathlib import Path

        from geofuse.cli import results

        lines = results.summary("outcome", self.weighted, Path(self.folder))
        text = "\n".join(lines)
        self.assertIn("Weighted         by 'wt'", text)
        self.assertIn("(w)", text)


if __name__ == "__main__":
    unittest.main()
