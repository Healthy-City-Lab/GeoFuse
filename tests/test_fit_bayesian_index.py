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
    for name in ("_params_from_sweep", "_index_layout", "_synergy_scaling"):
        setattr(eng, name, types.MethodType(getattr(MetricFusionEngine, name), eng))
    return eng


class TestFitBayesianIndexWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(setattr, parallel, "process_worker_count",
                            parallel.process_worker_count)
        parallel.process_worker_count = lambda *a, **k: 1
        cls.params = MetricFusionEngine.fit_bayesian_index(
            _stub_engine(), "r2", sweep_splits=3, reps=1, shuffles=2,
            gain_splits=2, gain_perm=2, null_runs=2, null_reselect_form=True,
            draws=120, warmup=120, chains=1, radius_kernel="dirichlet",
        )

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


class TestSynergyRunRecordsItsScaling(unittest.TestCase):
    """A synergy study's params carry the curve the composite path needs."""

    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(setattr, parallel, "process_worker_count",
                            parallel.process_worker_count)
        parallel.process_worker_count = lambda *a, **k: 1
        cls.params = MetricFusionEngine.fit_bayesian_index(
            _stub_engine(), "r2", forms=("synergy",), sweep_splits=2, reps=0,
            gain_splits=0, null_runs=0, draws=100, warmup=100, chains=1,
            radius_kernel="dirichlet",
        )

    def test_every_channel_gets_a_center_and_a_scale(self):
        self.assertEqual(self.params["__sweep__"]["form"], "synergy")
        for ch in ("ndvi", "gvi"):
            self.assertIn(f"{ch}_center", self.params)
            self.assertGreater(self.params[f"{ch}_scale"], 0.0)

    def test_the_composite_path_evaluates_with_them(self):
        from geofuse import cgi_formulas

        rng = np.random.default_rng(0)
        comps = {"ndvi": rng.normal(size=50), "gvi": rng.normal(size=50)}
        out = cgi_formulas.compute_cgi("synergy_gvi", self.params, comps)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(((out > 0) & (out < 1)).all())


def _retune_stub(n=600, seed=3):
    """U rides on ndvi at 100 m and drives both outcomes; the target alone
    also follows gvi at 900 m. Re-tuning on the control should find the 100 m
    ndvi cell the confounding lives in."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2, len(RADII), 1))
    u = rng.normal(size=n)
    X[:, 0, 0, 0] = u + 0.3 * rng.normal(size=n)
    target = 0.3 * u + 1.0 * X[:, 1, 2, 0] + rng.normal(size=n)
    control = 1.0 * u + rng.normal(size=n)
    static = {"target": target, "cov": None, "controls": {"grip": control}}
    eng = types.SimpleNamespace(
        cgi_formula="weighted_average_gvi", _active_greenery_channel="cgi",
        ndvi_buffer_max_m=900.0, gvi_buffer_max_m=900.0,
        negative_control_columns=["grip"], is_longitudinal=False,
    )
    eng.build_index_tensor = lambda subset="train_val": (
        X, np.asarray(RADII), ["mean"], ["ndvi", "gvi"], static)
    eng._preaggr_radii = lambda: (list(RADII), list(RADII))
    eng._sweep_objective = lambda metric: None
    for name in ("_params_from_sweep", "_index_layout", "_synergy_scaling",
                 "fit_bayesian_index", "retune_concordance"):
        setattr(eng, name, types.MethodType(getattr(MetricFusionEngine, name), eng))
    return eng


class TestRetuneOnAControl(unittest.TestCase):
    """The tracker's G1 acceptance 2: the control's kernel lands on U's rung."""

    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(setattr, parallel, "process_worker_count",
                            parallel.process_worker_count)
        parallel.process_worker_count = lambda *a, **k: 1
        cls.eng = _retune_stub()
        kw = dict(forms=("linear",), sweep_splits=3, draws=150, warmup=150,
                  chains=1, radius_kernel="dirichlet")
        cls.target = cls.eng.fit_bayesian_index(
            "r2", reps=0, shuffles=0, gain_splits=0, gain_perm=0, null_runs=0,
            **kw)
        kw.pop("radius_kernel")
        cls.formula_before = cls.eng.cgi_formula
        cls.retune = cls.eng.retune_concordance(
            cls.target, "r2", radius_kernel="dirichlet", **kw)

    def test_the_control_concentrates_ndvi_near_the_confounded_rung(self):
        ndvi = self.retune["grip"]["channels"]["ndvi"]
        self.assertLess(ndvi["r50_control"], 150.0)

    def test_the_two_tunings_put_the_weight_on_different_channels(self):
        ndvi = self.retune["grip"]["channels"]["ndvi"]
        self.assertGreater(ndvi["weight_control"], 0.5)
        self.assertLess(ndvi["weight_target"], 0.5)

    def test_the_target_study_state_is_left_alone(self):
        self.assertEqual(self.eng.cgi_formula, self.formula_before)

    def test_an_unknown_outcome_is_refused(self):
        with self.assertRaises(ValueError):
            self.eng.fit_bayesian_index("r2", outcome="height", sweep_splits=2,
                                        reps=0, gain_splits=0, null_runs=0)


if __name__ == "__main__":
    unittest.main()
