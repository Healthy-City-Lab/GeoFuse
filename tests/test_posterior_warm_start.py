"""Warm-started posteriors: same target, fewer warm-up iterations.

A held-out split after the first starts each posterior from the engine's
latest fit of the same channel and form, and a synergy posterior with no such
fit starts from a short linear pre-fit mapped onto the synergy model. Only the
Monte Carlo draws change: without a warm start the draws are today's, a
warm-started fit that mixes worse than its source is redone cold, and every
fit records where it started.
"""

import os
import sys
import tempfile
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas  # noqa: F401  (pyproj before torchvision, see vision.py)
import numpy as np

from geofuse import bayesian_index as bi
from geofuse import settings
from geofuse.fusion import WARM_ESS_SHARE, WARM_RHAT_MAX, MetricFusionEngine
from geofuse.jobs import outer_splits

RADII = np.array([200.0, 400.0, 600.0])
STATS = ["mean", "p50"]


def _grid(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2, len(RADII), len(STATS)))
    y = 0.3 * X[:, 0, 1, 0] + rng.normal(size=n)
    yr, Xr = bi.prep(X, y, None)
    return X, Xr, yr


KW = dict(radii=RADII, stats=STATS, chains=1, draws=60)


class SamplerTests(unittest.TestCase):
    def test_without_a_warm_start_the_draws_are_unchanged(self):
        _, Xr, yr = _grid()
        a = bi.fit(Xr, yr, warmup=60, seed=4, **KW)
        b = bi.fit(Xr, yr, warmup=60, seed=4, warm_start=None, **KW)
        np.testing.assert_array_equal(
            np.asarray(a.get_samples()["beta"]), np.asarray(b.get_samples()["beta"])
        )

    def test_a_linear_fit_maps_onto_the_synergy_model(self):
        X, Xr, yr = _grid()
        lin = bi.linear_prefit(
            Xr, yr, chains=1, seed=1, draws=60, warmup=60, radii=RADII, stats=STATS
        )
        start = bi.synergy_warm_start(lin, n_terms=1, warmup=60)
        w = start["init"]["w"]
        self.assertEqual(len(w), 3)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(w[-1]), bi.SYNERGY_START_SHARE, places=6)
        m = bi.fit(
            Xr,
            yr,
            form="synergy",
            warmup=60,
            seed=2,
            warm_start=start,
            design=bi.SynergyDesign.build(X),
            design_channels=(0, 1),
            **KW,
        )
        beta = np.asarray(m.get_samples()["beta"])
        self.assertGreater(float(beta.std()), 0.0)
        self.assertGreater(float(beta.mean()), 0.15)


class FallbackTests(unittest.TestCase):
    def check(self, rhat, ess, reference=None, start=True):
        post = types.SimpleNamespace(rhat_max=rhat, ess_min=ess)
        return MetricFusionEngine._warm_fit_fallback(
            post, {} if start else None, reference, 1000
        )

    def test_rules(self):
        self.assertIsNone(self.check(1.002, 900, {"ess_per_draw": 1.0}))
        self.assertIn("R-hat", self.check(WARM_RHAT_MAX + 0.01, 900))
        self.assertIn("R-hat", self.check(float("nan"), 900))
        low = WARM_ESS_SHARE * 1.0 * 1000 - 1
        message = self.check(1.0, low, {"ess_per_draw": 1.0})
        self.assertIn("effective sample size", message)
        self.assertIsNone(self.check(5.0, 1, None, start=False))


class EngineStartTests(unittest.TestCase):
    def setUp(self):
        self.eng = MetricFusionEngine(
            target_file="t.geojson", cache_dir=tempfile.mkdtemp()
        )

    def start(self, form, **flags):
        def prefit():
            raise RuntimeError("no data")

        flags = {"warm_start": False, "synergy_warm_start": False, **flags}
        return self.eng._posterior_start(form, prefit=prefit, n_terms=1, **flags)

    def test_the_latest_fit_of_the_same_kind(self):
        self.eng._posterior_warm[("cgi", "linear")] = {
            "warm": {"w": 1},
            "label": "split 3",
        }
        started = self.start("linear", warm_start=True)[:2]
        self.assertEqual(started, ({"w": 1, "adapt_mass_matrix": False}, "split 3"))
        self.assertEqual(self.start("linear")[0], None)
        self.eng._active_greenery_channel = "ndvi"
        self.assertEqual(self.start("linear", warm_start=True)[0], None)

    def test_a_failed_prefit_starts_cold(self):
        self.assertEqual(self.start("synergy", synergy_warm_start=True)[0], None)


class GuardTests(unittest.TestCase):
    def test_a_failed_warm_fit_is_refitted_cold_and_recorded(self):
        import test_fit_bayesian_index as fixture
        from unittest import mock

        eng = fixture._stub_engine()
        kw = dict(
            sweep_splits=2,
            reps=0,
            shuffles=0,
            gain_splits=0,
            null_runs=0,
            forms=("linear",),
            draws=60,
            warmup=80,
            chains=1,
        )
        first = MetricFusionEngine.fit_bayesian_index(eng, "r2", **kw)
        self.assertEqual(first["__posterior__"]["warm_start"]["source"], "none")
        eng._fit_label = "split 1"
        eng._warm_fit_fallback = lambda post, start, ref, n: (
            "forced" if start is not None else None
        )
        with mock.patch("geofuse.fusion._log") as log:
            second = MetricFusionEngine.fit_bayesian_index(
                eng, "r2", warm_start=True, **kw
            )
        record = second["__posterior__"]["warm_start"]
        want = {"source": "split 0", "warmup": 80, "fallback": "forced"}
        self.assertEqual(record, want)
        warns = [c.args[1] for c in log.call_args_list if c.args[0] == "WARN"]
        self.assertTrue(any("refitted from a cold start: forced" in w for w in warns))
        self.assertEqual(eng._posterior_warm[("cgi", "linear")]["label"], "split 1")


class RecordTests(unittest.TestCase):
    def test_split_rows_say_how_the_posterior_started(self):
        rows = outer_splits.posterior_rows(
            {
                "beta_mean": 0.1,
                "beta_ci_low": 0.0,
                "beta_ci_high": 0.2,
                "form": "synergy",
                "warm_start": {"source": "split 0", "warmup": 150, "fallback": None},
            }
        )
        self.assertEqual(rows[0]["warm_start"], "split 0")
        self.assertEqual(rows[0]["warmup"], 150)
        self.assertEqual(rows[0]["warm_fallback"], "")

    def test_settings_default_on_and_the_budget_off(self):
        d = settings.defaults("fusion")
        self.assertTrue(d["outer_warm_start"])
        self.assertTrue(d["synergy_warm_start"])
        for key in (
            "outer_posterior_draws",
            "outer_posterior_warmup",
            "outer_posterior_chains",
        ):
            self.assertIsNone(d[key])


if __name__ == "__main__":
    unittest.main()
