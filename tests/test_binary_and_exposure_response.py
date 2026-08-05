"""Validation for the binary-outcome scorers and the exposure-response layer.

Three things are asserted, and each is the thing that would actually break:

* the exact GEE logistic fit reproduces ``statsmodels.GEE`` with the same
  working structure, so the implementation is not merely self-consistent;
* the fast one-step scorer ranks a trial pool the same way an exact refit does
  and holds its Type I error, which is the whole basis for using it in the
  search;
* the spline non-linearity test does not fire on a genuinely linear
  exposure–response, which a raw (non-orthogonalised) spline basis does.
"""

import os
import sys
import unittest
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.stats import spearmanr

from geofuse import binary_longitudinal as bl
from geofuse import exposure_response as er
from geofuse import objective_scoring as osc


def make_panel(n_entities=1200, n_waves=3, beta_g=0.30, seed=0, re_sd=1.0):
    """Binary panel with a person random intercept and a known greenery effect."""
    rng = np.random.default_rng(seed)
    eid = np.repeat(np.arange(n_entities), n_waves)
    n = len(eid)
    u = np.repeat(rng.normal(0, re_sd, n_entities), n_waves)
    cov = rng.normal(size=(n, 4))
    g = 0.5 * np.repeat(rng.normal(size=n_entities), n_waves) + 0.5 * rng.normal(size=n)
    eta = -0.3 + beta_g * g + cov @ np.array([0.4, -0.25, 0.15, 0.0]) + u
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return y, g, eid, cov


class TestExactGEE(unittest.TestCase):
    """The exact fit must match an independent implementation."""

    def test_matches_statsmodels(self):
        try:
            import statsmodels.api as sm
        except ImportError:  # pragma: no cover - statsmodels is a hard dep
            self.skipTest("statsmodels unavailable")
        y, g, eid, cov = make_panel(seed=1)
        X = np.column_stack([np.ones(len(y)), g, cov])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = sm.GEE(
                y, X, groups=eid,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable(),
            ).fit()
        coef = bl.score_gee_logit("gee_logit_coef", y, g, eid, cov)
        z = bl.score_gee_logit("gee_logit_tstat", y, g, eid, cov)
        self.assertAlmostEqual(coef, float(ref.params[1]), places=3)
        self.assertAlmostEqual(
            z, abs(float(ref.params[1]) / float(ref.bse[1])), places=2
        )

    def test_degenerate_inputs_fail_soft(self):
        y, g, eid, cov = make_panel(n_entities=100, seed=3)
        # A constant exposure carries no information and must score zero
        # rather than raising — a bad trial should rank last, not abort a study.
        self.assertEqual(
            bl.score_gee_logit("gee_logit_tstat", y, np.ones_like(g), eid, cov), 0.0
        )
        # A continuous outcome is not a binary outcome.
        self.assertEqual(
            bl.score_gee_logit("gee_logit_tstat", g, g, eid, cov), 0.0
        )


class TestFastOneStep(unittest.TestCase):
    """The search path must rank like the exact path and stay calibrated."""

    def test_ranks_a_trial_pool_like_exact_refit(self):
        y, g0, eid, cov = make_panel(n_entities=1500, seed=2, beta_g=0.25)
        rng = np.random.default_rng(7)
        trials = [
            (0.2 + 0.8 * rng.uniform()) * g0
            + rng.uniform(0, 0.9) * rng.normal(size=len(g0))
            for _ in range(40)
        ]
        baseline = bl.estimate_fold_baseline(y, eid, cov)
        self.assertIsNotNone(baseline)
        fast = np.array(
            [bl.score_gee_logit_fast("gee_logit_tstat", t, baseline) for t in trials]
        )
        exact = np.array(
            [bl.score_gee_logit("gee_logit_tstat", y, t, eid, cov) for t in trials]
        )
        self.assertGreater(spearmanr(fast, exact).statistic, 0.99)
        self.assertLess(float(np.max(np.abs(fast - exact))), 0.25)
        self.assertEqual(int(np.argmax(fast)), int(np.argmax(exact)))

    def test_coefficient_tracks_exact_refit(self):
        y, g, eid, cov = make_panel(n_entities=1500, seed=11, beta_g=0.30)
        baseline = bl.estimate_fold_baseline(y, eid, cov)
        fast = bl.score_gee_logit_fast("gee_logit_coef", g, baseline)
        exact = bl.score_gee_logit("gee_logit_coef", y, g, eid, cov)
        self.assertLess(abs(fast - exact), 0.05)

    def test_type_one_error_under_the_null(self):
        rejections = 0
        reps = 200
        for r in range(reps):
            y, g, eid, cov = make_panel(n_entities=400, seed=1000 + r, beta_g=0.0)
            baseline = bl.estimate_fold_baseline(y, eid, cov)
            if baseline is None:
                continue
            if bl.score_gee_logit_fast("gee_logit_tstat", g, baseline) > 1.959964:
                rejections += 1
        rate = rejections / reps
        # Monte-Carlo band for 200 draws at a nominal 0.05.
        self.assertLess(rate, 0.11, f"Type I error {rate:.3f} is inflated")

    def test_clustering_correction_does_work(self):
        """A pooled logit on the same panel must reject more often than GEE.

        If it does not, the cluster-robust variance is not being applied and
        the panel structure is being ignored.
        """
        pooled = gee = 0
        reps = 150
        for r in range(reps):
            y, g, eid, cov = make_panel(
                n_entities=300, seed=4000 + r, beta_g=0.0, re_sd=1.5
            )
            baseline = bl.estimate_fold_baseline(y, eid, cov)
            if baseline is None:
                continue
            gee += bl.score_gee_logit_fast("gee_logit_tstat", g, baseline) > 1.959964
            pooled += osc.score("logit_tstat", y, g, cov) > 1.959964
        self.assertGreaterEqual(pooled, gee)


class TestCrossSectionalLogit(unittest.TestCase):
    def test_registered_and_recovers_the_effect(self):
        self.assertIn("logit_tstat", osc.SUPPORTED_METRICS)
        self.assertIn("logit_coef", osc.SUPPORTED_METRICS)
        self.assertIn("logit_tstat", osc.HIGHER_IS_BETTER)
        rng = np.random.default_rng(5)
        n = 6000
        cov = rng.normal(size=(n, 3))
        g = rng.normal(size=n)
        eta = -0.5 + 0.45 * g + cov @ np.array([0.3, -0.2, 0.1])
        y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
        self.assertAlmostEqual(osc.score("logit_coef", y, g, cov), 0.45, delta=0.08)
        self.assertGreater(osc.score("logit_tstat", y, g, cov), 5.0)

    def test_continuous_outcome_is_degenerate_not_an_error(self):
        rng = np.random.default_rng(6)
        g = rng.normal(size=500)
        self.assertEqual(osc.score("logit_tstat", g + rng.normal(size=500), g), 0.0)


class TestExposureResponse(unittest.TestCase):
    def test_iqr_scaling_moves_units_only(self):
        out = er.iqr_scaled_effect(0.5, 0.1, iqr=0.06, logistic=True)
        self.assertAlmostEqual(out["estimate"], 0.03)
        self.assertAlmostEqual(out["std_error"], 0.006)
        self.assertAlmostEqual(out["odds_ratio"], float(np.exp(0.03)))
        # The z-statistic is invariant, which is the point of a unit change.
        self.assertAlmostEqual(out["estimate"] / out["std_error"], 0.5 / 0.1)

    def test_quartile_contrasts_are_monotone_for_a_monotone_effect(self):
        rng = np.random.default_rng(8)
        n = 8000
        g = rng.normal(size=n)
        y = 0.8 * g + rng.normal(size=n)
        out = er.quartile_terms(g, er.make_ols_fitter(y), None)
        self.assertIsNotNone(out)
        coefs = [c["coef"] for c in out["contrasts"]]
        self.assertEqual(coefs, sorted(coefs))
        self.assertLess(out["trend_p"], 1e-10)

    def test_spline_does_not_fire_on_a_linear_response(self):
        """The calibration failure an unorthogonalised spline basis produces."""
        rng = np.random.default_rng(9)
        n = 4000
        g = rng.normal(size=n)
        y = 0.6 * g + rng.normal(size=n)
        out = er.spline_nonlinearity_test(g, er.make_ols_fitter(y), None)
        self.assertIsNotNone(out)
        self.assertTrue(out["exact_wald"])
        self.assertGreater(out["nonlinearity_p"], 0.01)

    def test_spline_detects_a_curved_response(self):
        rng = np.random.default_rng(10)
        n = 4000
        g = rng.normal(size=n)
        y = 0.9 * g**2 + rng.normal(size=n)
        out = er.spline_nonlinearity_test(g, er.make_ols_fitter(y), None)
        self.assertLess(out["nonlinearity_p"], 1e-6)
        self.assertEqual(len(out["curve"]["exposure"]), 50)

    def test_spline_type_one_error_is_calibrated(self):
        fired = 0
        reps = 200
        for r in range(reps):
            rng = np.random.default_rng(20000 + r)
            g = rng.normal(size=800)
            y = 0.4 * g + rng.normal(size=800)
            out = er.spline_nonlinearity_test(g, er.make_ols_fitter(y), None)
            if out and out["nonlinearity_p"] < 0.05:
                fired += 1
        rate = fired / reps
        self.assertLess(rate, 0.12, f"non-linearity test fires at {rate:.3f} on a line")


class TestEngineRouting(unittest.TestCase):
    """The scoring seam must send a binary panel metric to GEE, not MixedLM."""

    def _engine(self, method="mom_em3"):
        from geofuse.fusion import MetricFusionEngine

        engine = MetricFusionEngine.__new__(MetricFusionEngine)
        engine.longitudinal_spec = object()  # ``is_longitudinal`` reads this
        engine.spatial_adjust_method = "none"
        engine.search_scoring_method = method
        return engine

    def test_score_greenery_routes_to_gee_and_matches_the_scorer(self):
        y, g, eid, cov = make_panel(n_entities=800, seed=21, beta_g=0.3)
        engine = self._engine()
        routed = engine._score_greenery(
            "gee_logit_tstat", y, g, covariates=cov, entity_id=eid
        )
        direct = bl.score_gee_logit("gee_logit_tstat", y, g, eid, cov)
        self.assertAlmostEqual(routed, direct, places=6)

    def test_fold_baseline_is_built_and_cached(self):
        y, g, eid, cov = make_panel(n_entities=600, seed=22, beta_g=0.3)
        engine = self._engine()
        static = {"target": y, "entity_id": eid, "cov": cov}
        first = engine._fold_gee_baseline(static)
        self.assertIsInstance(first, bl.GEEFoldBaseline)
        # Second call must reuse it — rebuilding per trial would defeat the
        # entire point of the fast path.
        self.assertIs(engine._fold_gee_baseline(static), first)

        fast = engine._score_greenery(
            "gee_logit_tstat", y, g, covariates=cov, entity_id=eid,
            fast_components=first,
        )
        exact = bl.score_gee_logit("gee_logit_tstat", y, g, eid, cov)
        self.assertLess(abs(fast - exact), 0.3)

    def test_exact_method_declines_the_fold_baseline(self):
        y, _g, eid, cov = make_panel(n_entities=400, seed=23)
        engine = self._engine(method="exact")
        static = {"target": y, "entity_id": eid, "cov": cov}
        self.assertIsNone(engine._fold_gee_baseline(static))


if __name__ == "__main__":
    unittest.main(verbosity=2)
