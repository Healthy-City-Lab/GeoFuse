"""Repeated held-out evaluation: the tuning redone on other test splits.

Pinned here:

- under a null outcome the corrected resampled t over ten splits rejects at
  about its level, while a naive t over the same splits rejects several times
  as often;
- the summary gives each result's spread, the share of splits where it holds
  and the corrected p-value, Holm-corrected over the primary comparators;
- on the repeats only the primary comparators are scored unless all are asked;
- each split's posterior (β, every term's weight, R50 / R90 per channel, the
  form) becomes rows that are summarised but not tested, with the share of
  splits in which each term has the largest weight;
- a run with three splits reports every quantity per split, a planted effect
  holds in most of them, the first split's results are the single-split run's
  (its posterior rows are ``discovery.csv``'s), and the tables are written.
"""

import csv
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas as gpd  # noqa: F401
import numpy as np
from scipy import stats

from geofuse.jobs import outer_splits
from geofuse.statistical_testing import corrected_resampled_t


def _null_slopes(rng, n=1000, k=10, p=5, test_fraction=0.25):
    """Held-out slopes of the column picked on each split's training rows, for
    an outcome unrelated to every column."""
    cov = 0.6 * np.ones((p, p)) + 0.4 * np.eye(p)
    x = rng.normal(size=(n, p)) @ np.linalg.cholesky(cov).T
    y = rng.normal(size=n)
    m = int(test_fraction * n)
    slopes = []
    for _ in range(k):
        idx = rng.permutation(n)
        test, train = idx[:m], idx[m:]
        r = [abs(np.corrcoef(x[train, j], y[train])[0, 1]) for j in range(p)]
        j = int(np.argmax(r))
        slopes.append(np.polyfit(x[test, j], y[test], 1)[0])
    return np.array(slopes)


class TestCorrectedResampledT(unittest.TestCase):
    def test_the_null_rejection_rate_is_near_its_level(self):
        rng = np.random.default_rng(1)
        corrected = naive = 0
        runs = 400
        for _ in range(runs):
            b = _null_slopes(rng)
            corrected += corrected_resampled_t(b, 0.25)["p_value"] < 0.05
            t = b.mean() / (b.std(ddof=1) / np.sqrt(b.size))
            naive += 2 * stats.t.sf(abs(t), b.size - 1) < 0.05
        self.assertLessEqual(corrected / runs, 0.08)
        self.assertGreater(naive / runs, 2 * corrected / runs)

    def test_a_planted_effect_is_found(self):
        b = np.array([0.031, 0.028, 0.035, 0.026, 0.030, 0.033, 0.029, 0.027])
        test = corrected_resampled_t(b, 0.25, alternative="greater")
        self.assertLess(test["p_value"], 0.01)
        self.assertEqual(test["k"], 8)

    def test_too_few_values_give_no_test(self):
        self.assertIsNone(corrected_resampled_t([0.1], 0.25)["p_value"])
        self.assertIsNone(corrected_resampled_t([0.1, None, 0.1], 0.25)["t"])
        with self.assertRaises(ValueError):
            corrected_resampled_t([0.1, 0.2], 0.25, alternative="less")


def _rows(quantity, exposure, estimates, p=None, **extra):
    return [
        {
            "quantity": quantity,
            "exposure": exposure,
            "estimate": e,
            "ci_low": e - 0.01,
            "ci_high": e + 0.01,
            "p_value": p,
            "split": k,
            "test_share": 0.25,
            **extra,
        }
        for k, e in enumerate(estimates)
    ]


class TestSummary(unittest.TestCase):
    def test_spread_share_and_corrected_p(self):
        rows = _rows("held_out_beta", "cgi", [0.03, 0.02, 0.04, -0.005, 0.03])
        (entry,) = outer_splits.summarise(rows)
        self.assertEqual(entry["n_splits"], 5)
        self.assertAlmostEqual(entry["median"], 0.03)
        self.assertEqual((entry["min"], entry["max"]), (-0.005, 0.04))
        # The fourth interval holds zero.
        self.assertAlmostEqual(entry["share_holds"], 0.8)
        expected = corrected_resampled_t([0.03, 0.02, 0.04, -0.005, 0.03], 0.25)
        self.assertAlmostEqual(entry["corrected_p"], expected["p_value"])

    def test_the_score_is_summarised_but_not_tested(self):
        (entry,) = outer_splits.summarise(
            _rows("held_out_score", "cgi", [0.004, 0.001], p=0.01)
        )
        self.assertEqual(entry["share_holds"], 1.0)
        self.assertIsNone(entry["corrected_p"])

    def test_holm_runs_over_the_primary_comparators(self):
        rows = (
            _rows("cgi_minus_comparator", "a", [0.02, 0.03, 0.025], primary=True)
            + _rows("cgi_minus_comparator", "b", [0.0, 0.01, -0.01], primary=True)
            + _rows("cgi_minus_comparator", "c", [0.02, 0.02, 0.03], primary=False)
            + _rows("cgi_minus_channel", "ndvi", [0.01, 0.0, 0.02])
        )
        summary = outer_splits.summarise(rows)
        by = {(e["quantity"], e["exposure"]): e for e in summary}
        a, b = by[("cgi_minus_comparator", "a")], by[("cgi_minus_comparator", "b")]
        holm = sorted([a["corrected_p"], b["corrected_p"]])
        smallest = min(a["corrected_p_holm"], b["corrected_p_holm"])
        self.assertAlmostEqual(smallest, min(1.0, 2 * holm[0]))
        self.assertIsNone(by[("cgi_minus_comparator", "c")]["corrected_p_holm"])
        self.assertIsNone(by[("cgi_minus_channel", "ndvi")]["corrected_p_holm"])
        first = outer_splits.summarise(rows)[0]
        self.assertEqual(first["quantity"], "cgi_minus_channel")

    def test_the_repeats_score_the_primary_comparators(self):
        pick = outer_splits.scored_comparators
        self.assertEqual(pick(["a", "b", "c"], ["b"], "primary"), ["b"])
        self.assertEqual(pick(["a", "b", "c"], ["b"], "all"), ["a", "b", "c"])
        self.assertEqual(pick(["a", "b"], [], "primary"), ["a", "b"])

    def test_split_rows_read_the_engine_reports(self):
        rows = outer_splits.split_rows(
            effects_test={
                "score": 0.01,
                "lower": 0.0,
                "upper": 0.02,
                "p_value": 0.03,
            },
            slope_test={
                "beta": 0.1,
                "ci_low": 0.02,
                "ci_high": 0.18,
                "t": 2.5,
                "df": 90,
            },
            channel_paired={
                "standalone_channel": "ndvi",
                "observed_diff": 0.002,
                "lower": -0.001,
                "upper": 0.005,
                "p_value": 0.2,
            },
            comparator_paired=[{"comparator": "g13", "observed_diff": 0.001}],
            negative_controls={
                "splits": {
                    "test": {
                        "controls": {
                            "grip": {"beta": 0.03, "ci_low": -0.05, "ci_high": 0.11}
                        }
                    }
                }
            },
        )
        got = {(r["quantity"], r["exposure"]): r for r in rows}
        self.assertEqual(
            list(got),
            [
                ("held_out_score", "cgi"),
                ("held_out_beta", "cgi"),
                ("cgi_minus_channel", "ndvi"),
                ("cgi_minus_comparator", "g13"),
                ("negative_control_beta", "grip"),
            ],
        )
        self.assertAlmostEqual(
            got[("held_out_beta", "cgi")]["p_value"], 2 * stats.t.sf(2.5, 90)
        )
        self.assertIsNone(got[("negative_control_beta", "grip")]["p_value"])


def _posterior(weights, r50, form="linear", beta=0.04):
    return {
        "form": form,
        "channels": ["ndvi", "gvi"],
        "weight_labels": ["ndvi", "gvi", "ndvi×gvi"][: len(weights)],
        "weight_mean": list(weights),
        "weight_ci_low": [w - 0.1 for w in weights],
        "weight_ci_high": [w + 0.1 for w in weights],
        "radius_profile": [[0.5, 0.5], [0.5, 0.5]],
        "r50_mean": list(r50),
        "r50_ci_low": [r - 50 for r in r50],
        "r50_ci_high": [r + 50 for r in r50],
        "r90_mean": [r * 2 for r in r50],
        "r90_ci_low": [r * 2 - 50 for r in r50],
        "r90_ci_high": [r * 2 + 50 for r in r50],
        "beta_mean": beta,
        "beta_ci_low": beta - 0.02,
        "beta_ci_high": beta + 0.02,
    }


class TestPosteriorRows(unittest.TestCase):
    def test_one_row_per_term_and_channel_carrying_the_form(self):
        rows = outer_splits.posterior_rows(
            _posterior([0.5, 0.3, 0.2], [400, 200], form="synergy")
        )
        got = [(r["quantity"], r["exposure"]) for r in rows]
        self.assertEqual(
            got,
            [
                ("posterior_beta", "cgi"),
                ("posterior_weight", "ndvi"),
                ("posterior_weight", "gvi"),
                ("posterior_weight", "ndvi×gvi"),
                ("posterior_r50", "ndvi"),
                ("posterior_r50", "gvi"),
                ("posterior_r90", "ndvi"),
                ("posterior_r90", "gvi"),
            ],
        )
        self.assertTrue(all(r["form"] == "synergy" for r in rows))
        r50 = rows[4]
        interval = (r50["estimate"], r50["ci_low"], r50["ci_high"])
        self.assertEqual(interval, (400, 350, 450))
        self.assertEqual(outer_splits.posterior_rows(None), [])

    def test_the_summary_describes_and_counts_the_largest_weight(self):
        splits = [
            _posterior([0.7, 0.3], [400, 200]),
            _posterior([0.6, 0.4], [500, 250]),
            _posterior([0.2, 0.8], [300, 150], form="synergy"),
        ]
        rows = []
        for k, post in enumerate(splits):
            for r in outer_splits.posterior_rows(post):
                r.update(split=k, test_share=0.25)
                rows.append(r)
        by = {(e["quantity"], e["exposure"]): e for e in outer_splits.summarise(rows)}
        ndvi = by[("posterior_weight", "ndvi")]
        self.assertAlmostEqual(ndvi["share_largest"], 2 / 3)
        self.assertAlmostEqual(by[("posterior_weight", "gvi")]["share_largest"], 1 / 3)
        self.assertAlmostEqual(by[("posterior_r50", "ndvi")]["median"], 400)
        forms = by[("posterior_beta", "cgi")]["forms"]
        self.assertEqual(forms, "linear 2/3, synergy 1/3")
        for e in by.values():
            self.assertIsNone(e["share_holds"])
            self.assertIsNone(e["corrected_p"])
        line = outer_splits.describe_posterior(list(by.values()))
        self.assertIn("R50 ndvi 400 m [300, 500]", line)
        self.assertIn("weight ndvi 0.600 [0.200, 0.700], largest in 67%", line)


def _write_smooth_study(folder: str) -> None:
    """``test_fusion_run``'s study with a smooth NDVI field, so that the
    neighbourhood means the composite reads carry the outcome's signal."""
    import geopandas as gpd
    import rasterio
    from test_fusion_run import _PX, _X0, _Y0, _write_study

    _write_study(folder)
    rows, cols = np.mgrid[0:160, 0:160]
    field = (
        0.5 + 0.25 * np.sin(rows / 9.0) + 0.25 * np.cos(cols / 11.0)
    ).astype("float32")
    path = os.path.join(folder, "ndvi_2015.tif")
    with rasterio.open(path, "r+") as dst:
        dst.write(field, 1)
    cohort = gpd.read_file(os.path.join(folder, "cohort.gpkg"))
    col = ((cohort.geometry.x - (_X0 - 0.01)) / _PX).astype(int)
    row = (((_Y0 + 0.04) - cohort.geometry.y) / _PX).astype(int)
    rng = np.random.default_rng(5)
    cohort["outcome"] = 0.8 * field[row, col] + rng.normal(0, 0.15, len(cohort))
    cohort.to_file(os.path.join(folder, "cohort.gpkg"), driver="GPKG")


class TestRepeatedRun(unittest.TestCase):
    """A small study whose composite carries a planted effect, run on one
    held-out split and on three."""

    INPUTS = """
        mode = "cross-sectional"
        ndvi = "ndvi_2015.tif"
        gvi = "gvi.gpkg"
        """

    @classmethod
    def setUpClass(cls):
        from test_fusion_run import run_study

        folder = tempfile.mkdtemp(prefix="geofuse-outer-")
        _write_smooth_study(folder)
        comparators = {
            "comparators": ["ndvi_here", "noise", "patchy"],
            "comparators_primary": ["ndvi_here", "patchy"],
        }
        cls.single = run_study(folder, cls.INPUTS, **comparators)[1]
        cls.ctx, cls.bundle = run_study(
            folder, cls.INPUTS, outer_splits=3, **comparators
        )
        cls.outer = cls.bundle["outer_splits"]

    def test_every_split_reports_every_quantity(self):
        self.assertEqual(self.outer["k"], 3)
        self.assertEqual(self.outer["seeds"], [42, 43, 44])
        by_split: dict = {}
        for r in self.outer["splits"]:
            by_split.setdefault(r["split"], set()).add((r["quantity"], r["exposure"]))
        self.assertEqual(sorted(by_split), [0, 1, 2])
        for quantities in by_split.values():
            names = {q for q, _ in quantities}
            self.assertTrue(
                {
                    "held_out_score",
                    "held_out_beta",
                    "cgi_minus_channel",
                    "cgi_minus_comparator",
                    "negative_control_beta",
                }
                <= names,
                names,
            )
            comparators = {e for q, e in quantities if q == "cgi_minus_comparator"}
            self.assertEqual(comparators, {"ndvi_here", "patchy"})
        test_sets = {r["n_test"] for r in self.outer["splits"]}
        self.assertTrue(all(n > 0 for n in test_sets))

    def test_a_planted_effect_holds_in_most_splits(self):
        beta = next(
            e
            for e in self.outer["summary"]
            if (e["quantity"], e["exposure"]) == ("held_out_beta", "cgi")
        )
        self.assertGreater(beta["median"], 0)
        self.assertGreaterEqual(beta["share_holds"], 2 / 3)

    def test_the_first_split_is_the_single_split_run(self):
        self.assertAlmostEqual(
            self.bundle["cgi_effects"]["test"]["score"],
            self.single["cgi_effects"]["test"]["score"],
        )
        first = {
            (r["quantity"], r["exposure"]): r["estimate"]
            for r in self.outer["splits"]
            if r["split"] == 0
        }
        self.assertAlmostEqual(
            first[("held_out_score", "cgi")],
            self.single["cgi_effects"]["test"]["score"],
        )

    def test_the_tables_are_written(self):
        report = self.bundle["report_dir"]
        with open(os.path.join(report, "outer_splits.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual({r["split"] for r in rows}, {"0", "1", "2"})
        with open(os.path.join(report, "outer_summary.csv"), encoding="utf-8") as f:
            self.assertIn("corrected_p", next(csv.reader(f)))

    def test_the_first_split_posterior_is_discovery_csv(self):
        report = self.bundle["report_dir"]
        with open(os.path.join(report, "discovery.csv"), encoding="utf-8") as f:
            disc = [r for r in csv.DictReader(f) if r["study"] == "cgi"]
        first = {
            (r["quantity"], r["exposure"]): r
            for r in self.outer["splits"]
            if r["split"] == 0 and r["quantity"].startswith("posterior_")
        }
        for d in disc:
            w = first[("posterior_weight", d["channel"])]
            self.assertAlmostEqual(w["estimate"], float(d["weight"]))
            self.assertAlmostEqual(w["ci_low"], float(d["weight_ci_low"]))
            self.assertEqual(w["form"], d["form"])
            if d["r50_m"]:
                r50 = first[("posterior_r50", d["channel"])]
                self.assertAlmostEqual(r50["estimate"], float(d["r50_m"]))
                r90 = first[("posterior_r90", d["channel"])]
                self.assertAlmostEqual(r90["estimate"], float(d["r90_m"]))
        beta = first[("posterior_beta", "cgi")]
        self.assertAlmostEqual(beta["estimate"], float(disc[0]["beta_mean"]))

    def test_every_split_reports_its_posterior(self):
        terms: dict = {}
        for r in self.outer["splits"]:
            if r["quantity"] == "posterior_weight":
                terms.setdefault(r["split"], []).append(r)
        self.assertEqual(sorted(terms), [0, 1, 2])
        for group in terms.values():
            names = [r["exposure"] for r in group]
            self.assertEqual(len(names), len(set(names)))
        largest = [
            max(g, key=lambda r: r["estimate"])["exposure"] for g in terms.values()
        ]
        for e in self.outer["summary"]:
            if e["quantity"] == "posterior_weight":
                self.assertAlmostEqual(
                    e["share_largest"], largest.count(e["exposure"]) / 3
                )
                values = [
                    r["estimate"]
                    for g in terms.values()
                    for r in g
                    if r["exposure"] == e["exposure"]
                ]
                self.assertAlmostEqual(e["median"], float(np.median(values)))


if __name__ == "__main__":
    unittest.main()
