"""Plasmode recovery: a CGI planted in a correlated exposure tensor.

The fast tests check the scaffolding — scenario construction, the planted
index, the true R50, seeding, the tidy table and the report files. The
acceptance test runs the real pipeline many times on a tensor whose columns are
as correlated as real greenery layers (AR(1) across rungs, rho 0.9; 0.95
across statistics); it takes minutes, so it runs only with
``GEOFUSE_SLOW_TESTS=1``.
"""

import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from geofuse import bayesian_index as bi
from geofuse import validation as v

RADII = [100.0, 200.0, 300.0, 500.0, 800.0]
STATS = ["mean", "p25", "p75"]


def correlated_tensor(n=5000, seed=0, rho_rung=0.9, rho_stat=0.95):
    """``(n, 2, rungs, stats)`` with AR(1) rungs and near-duplicate statistics.

    Two covariates lean on the first channel, as age and income lean on
    greenness, so the residualisation has something to remove.
    """
    rng = np.random.default_rng(seed)
    nr, ns = len(RADII), len(STATS)
    base = np.empty((n, 2, nr))
    base[:, :, 0] = rng.normal(size=(n, 2))
    for r in range(1, nr):
        base[:, :, r] = (rho_rung * base[:, :, r - 1]
                         + np.sqrt(1 - rho_rung ** 2) * rng.normal(size=(n, 2)))
    X = (np.sqrt(rho_stat) * base[..., None]
         + np.sqrt(1 - rho_stat) * rng.normal(size=(n, 2, nr, ns)))
    cov = np.column_stack([0.4 * base[:, 0, 2] + rng.normal(size=n),
                           rng.normal(size=n)])
    return X, cov


class TestScenarios(unittest.TestCase):
    def test_all_five_for_two_channels(self):
        names = [t.name for t in v.default_scenarios(2, RADII, STATS)]
        self.assertEqual(names, ["S0", "S1", "S2", "S3", "S4"])

    def test_one_channel_skips_the_two_channel_truths(self):
        names = [t.name for t in v.default_scenarios(1, RADII, STATS)]
        self.assertEqual(names, ["S0", "S1", "S4"])

    def test_every_kernel_and_blend_is_a_distribution(self):
        for t in v.default_scenarios(2, RADII, STATS):
            np.testing.assert_allclose(np.asarray(t.kernel).sum(1), 1.0)
            np.testing.assert_allclose(np.asarray(t.aggregator).sum(1), 1.0)

    def test_a_ladder_restricted_channel_only_gets_its_own_rungs(self):
        ladder = [[0, 1, 2, 3, 4], [2, 3, 4]]
        for t in v.default_scenarios(2, RADII, STATS, radius_idx=ladder):
            self.assertEqual(float(np.asarray(t.kernel)[1, :2].sum()), 0.0)


class TestPlantedTruth(unittest.TestCase):
    def setUp(self):
        X, cov = correlated_tensor(n=600)
        self.cov = cov
        _, self.Xr, self.info = bi.prep(X, np.zeros(len(X)), cov, return_info=True)

    def test_a_single_cell_truth_is_that_column(self):
        s1 = v.default_scenarios(2, RADII, STATS)[1]
        z = v.planted_index(self.Xr, s1, [0, 1])
        col = self.Xr[:, 0, 1, 0]
        self.assertAlmostEqual(abs(float(np.corrcoef(z, col)[0, 1])), 1.0, places=10)

    def test_the_planted_index_is_free_of_the_covariates(self):
        s2 = v.default_scenarios(2, RADII, STATS)[2]
        z = v.planted_index(self.Xr, s2, [0, 1])
        self.assertLess(abs(float(np.corrcoef(z, self.cov[:, 0])[0, 1])), 1e-8)

    def test_a_between_rung_truth_has_an_r50_between_its_rungs(self):
        truths = {t.name: t for t in v.default_scenarios(2, RADII, STATS)}
        sd = self.info["column_sd"]
        r50_s4 = v.true_r50(truths["S4"], RADII, sd, [0, 1], STATS)[0]
        self.assertGreater(r50_s4, RADII[1] / np.sqrt(2) * 0.9)
        self.assertLess(r50_s4, RADII[2] / np.sqrt(2) * 1.1)

    def test_seeds_are_stable_across_processes(self):
        # Python's ``hash`` is salted per process; the seed must not be.
        self.assertEqual(v._seed(0, "S1", 0.002, 3), v._seed(0, "S1", 0.002, 3))
        self.assertNotEqual(v._seed(0, "S1", 0.002, 3), v._seed(0, "S2", 0.002, 3))
        self.assertEqual(v._seed(7, "S0", 0.0, 0), 1842464698)


class TestPlasmodeRun(unittest.TestCase):
    """End to end with tiny settings, in-process: the table and the files."""

    @classmethod
    def setUpClass(cls):
        X, cov = correlated_tensor(n=500, seed=1)
        truths = [t for t in v.default_scenarios(2, RADII, STATS)
                  if t.name in ("S0", "S2")]
        cls.out = v.plasmode_recovery(
            X, cov, channels=("ndvi", "gvi"), radii=RADII, stats=STATS,
            scenarios=truths, reps=2, partial_r2=(0.03,), forms=("linear",),
            pipeline_kwargs=dict(draws=60, warmup=60, chains=1, sweep_splits=2),
            seed=3, workers=1,
        )

    def test_the_settings_declare_the_light_sampler(self):
        s = self.out["settings"]
        self.assertEqual((s["draws"], s["warmup"], s["chains"], s["sweep_splits"]),
                         (60, 60, 1, 2))
        self.assertEqual(s["reps"], 2)
        self.assertEqual([t["name"] for t in s["scenarios"]], ["S0", "S2"])

    def test_the_null_runs_once_at_no_effect(self):
        s0 = [r for r in self.out["replicates"] if r["scenario"] == "S0"]
        self.assertEqual(len(s0), 2)
        self.assertTrue(all(r["partial_r2"] == 0.0 for r in s0))

    def test_the_table_has_every_metric(self):
        metrics = {(r["scenario"], r["metric"]) for r in self.out["table"]}
        self.assertIn(("S0", "false_positive_rate"), metrics)
        for m in ("power", "beta_bias", "beta_coverage", "r50_abs_log_error",
                  "r50_coverage", "aggregator_tv_error", "weight_coverage",
                  "weight_abs_error", "form_recovery"):
            self.assertIn(("S2", m), metrics)

    def test_weight_coverage_reads_only_interior_true_weights(self):
        # S2's 0.7 / 0.3 are interior; both count, once per replicate.
        row = next(r for r in self.out["table"]
                   if (r["scenario"], r["metric"]) == ("S2", "weight_coverage"))
        self.assertEqual(row["n"], 4)

    def test_the_rate_carries_an_exact_interval(self):
        row = next(r for r in self.out["table"] if r["metric"] == "false_positive_rate")
        k = round(row["value"] * row["n"])
        self.assertEqual((row["ci_low"], row["ci_high"]), bi.clopper_pearson(k, row["n"]))

    def test_the_report_files_are_written(self):
        out_dir = tempfile.mkdtemp(prefix="geofuse-plasmode-")
        paths = v.write_plasmode_report(self.out, out_dir)
        self.assertEqual(sorted(os.path.basename(p) for p in paths),
                         ["plasmode_results.json", "plasmode_summary.csv"])


class TestEngineEntryPoint(unittest.TestCase):
    """``MetricFusionEngine.plasmode_recovery`` hands over the study's tensor."""

    def test_partial_coverage_rows_are_left_out_and_the_ladders_passed(self):
        import types

        from geofuse.fusion import MetricFusionEngine as E

        X, cov = correlated_tensor(n=400, seed=2)
        X[:7, 1, 0, :] = np.nan            # a few entities lack small-radius gvi
        eng = types.SimpleNamespace(cgi_formula="weighted_average_gvi",
                                    _active_greenery_channel="cgi")
        eng.build_index_tensor = lambda subset="train_val": (
            X, np.asarray(RADII), list(STATS), ["ndvi", "gvi"], {"cov": cov})
        eng._preaggr_radii = lambda: (list(RADII), list(RADII[:4]))
        for name in ("_index_layout", "plasmode_recovery"):
            setattr(eng, name, types.MethodType(getattr(E, name), eng))
        truth = v.default_scenarios(2, RADII, STATS)[1]
        out_dir = tempfile.mkdtemp(prefix="geofuse-plasmode-engine-")
        out = eng.plasmode_recovery(
            scenarios=[truth], reps=1, partial_r2=(0.05,), forms=("linear",),
            pipeline_kwargs=dict(draws=40, warmup=40, chains=1, sweep_splits=2),
            workers=1, out_dir=out_dir)
        self.assertEqual(out["settings"]["n"], 393)
        self.assertEqual(out["settings"]["radius_idx"], [[0, 1, 2, 3], [0, 1, 2, 3, 4]])
        self.assertTrue(os.path.exists(os.path.join(out_dir, "plasmode_summary.csv")))


@unittest.skipUnless(os.environ.get("GEOFUSE_SLOW_TESTS") == "1",
                     "minutes of NUTS fits; set GEOFUSE_SLOW_TESTS=1")
class TestPlasmodeAcceptance(unittest.TestCase):
    """The tracker's G4 acceptance thresholds, calibrated once and frozen."""

    @classmethod
    def setUpClass(cls):
        X, cov = correlated_tensor(n=5000, seed=0)
        truths = [t for t in v.default_scenarios(2, RADII, STATS)
                  if t.name in ("S0", "S1")]
        cls.out = v.plasmode_recovery(
            X, cov, channels=("ndvi", "gvi"), radii=RADII, stats=STATS,
            scenarios=truths, reps=16, partial_r2=(0.002,), seed=0)
        cls.rows = {(r["scenario"], r["metric"]): r for r in cls.out["table"]}

    def test_the_null_false_positive_interval_reaches_ten_percent(self):
        self.assertLessEqual(self.rows[("S0", "false_positive_rate")]["ci_low"], 0.10)

    def test_a_single_cell_truth_has_its_r50_covered(self):
        self.assertGreaterEqual(self.rows[("S1", "r50_coverage")]["value"], 0.8)


if __name__ == "__main__":
    unittest.main()
