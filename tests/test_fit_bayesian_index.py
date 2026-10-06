"""`MetricFusionEngine.fit_bayesian_index` end to end, on a stub engine.

The engine method only orchestrates: it residualises, sweeps, runs the honesty
loops, fits the posterior and assembles the bundle. Each stage has its own tests
in ``test_bayesian_index.py``; what is checked here is that the stages are wired
to each other, which no stage-level test can see. The stub supplies the tensor
the real engine would build from its greenery cache, and every pool is held to
one worker so the run stays in-process.
"""

import os
import sys
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from geofuse import parallel
from geofuse.fusion import MetricFusionEngine

RADII = (100.0, 300.0, 900.0)
STATS = ["mean", "p50"]


def _stub_engine(n=300, seed=0):
    rng = np.random.default_rng(seed)
    # Channel 0 is ndvi, channel 1 gvi; the outcome follows gvi at 300 m.
    X = rng.normal(size=(n, 2, len(RADII), len(STATS)))
    X[:, :, :, 0] *= np.array([1.0, 2.0, 4.0])[None, None, :]
    y = 0.5 * X[:, 1, 1, 0] / 2.0 + rng.normal(size=n)
    cov = np.column_stack([rng.normal(size=n), rng.integers(0, 2, size=n)])
    static = {"target": y, "cov": cov}

    eng = types.SimpleNamespace(
        cgi_formula="weighted_average_gvi",
        _active_greenery_channel="cgi",
        ndvi_buffer_max_m=900.0,
        gvi_buffer_max_m=900.0,
    )
    eng.build_index_tensor = lambda subset="train_val": (
        X, np.asarray(RADII), list(STATS), ["ndvi", "gvi"], static)
    eng._preaggr_radii = lambda: (list(RADII), list(RADII))
    eng._sweep_objective = lambda metric: None
    eng._params_from_sweep = types.MethodType(
        MetricFusionEngine._params_from_sweep, eng)
    return eng


class TestFitBayesianIndexWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._real_workers = parallel.process_worker_count
        parallel.process_worker_count = lambda *a, **k: 1
        cls.params = MetricFusionEngine.fit_bayesian_index(
            _stub_engine(), "r2", sweep_splits=3, reps=1, shuffles=2,
            gain_splits=2, gain_perm=2, null_runs=2, null_reselect_form=True,
            draws=120, warmup=120, chains=1, radius_kernel="dirichlet",
        )

    @classmethod
    def tearDownClass(cls):
        parallel.process_worker_count = cls._real_workers

    def test_the_bundle_carries_every_stage(self):
        for key in ("__sweep__", "__posterior__", "__discovery__",
                    "__holdout_gain__", "__null_calibration__"):
            self.assertIn(key, self.params)
        self.assertIn("selection_score", self.params["__sweep__"])
        self.assertIn("form_counts", self.params["__holdout_gain__"])

    def test_distance_decay_is_on_the_raw_scale(self):
        post = self.params["__posterior__"]
        self.assertEqual(post["distance_scale"], "raw")
        self.assertEqual(len(post["r50_mean"]), 2)
        for lo, mid, hi in zip(post["r50_ci_low"], post["r50_mean"],
                               post["r50_ci_high"]):
            self.assertLessEqual(lo, mid + 1e-9)
            self.assertLessEqual(mid, hi + 1e-9)
            self.assertLessEqual(hi, max(RADII))

    def test_null_calibration_reselects_the_form_and_states_its_precision(self):
        null = self.params["__null_calibration__"]
        self.assertTrue(null["reselect_form"])
        self.assertEqual(sum(null["form_counts"].values()), 2)
        self.assertTrue(null["imprecise"])
        self.assertIn("rate_ci_high", null)

    def test_weights_land_on_the_formula_keys(self):
        from geofuse import cgi_formulas

        name = cgi_formulas.formula_for(("ndvi", "gvi"),
                                        self.params["__sweep__"]["form"])
        keys = cgi_formulas.get_formula(name).weight_keys
        self.assertEqual(sum(int(self.params[k]) for k in keys), 100)


if __name__ == "__main__":
    unittest.main()
