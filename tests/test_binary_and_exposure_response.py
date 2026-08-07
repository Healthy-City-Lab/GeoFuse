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

    def test_fitter_survives_a_rank_deficient_design(self):
        """A tied exposure makes columns collinear; the fit must still return."""
        rng = np.random.default_rng(11)
        n = 2000
        g = rng.normal(size=n)
        y = 0.5 * g + rng.normal(size=n)
        # A duplicated column is exactly singular, the limit of what a heavily
        # tied quantile or spline block produces.
        design = np.column_stack([np.ones(n), g, g])
        out = er.make_ols_fitter(y)(design, ["const", "g", "g_dup"])
        self.assertIsNotNone(out)
        self.assertTrue(np.isfinite(out["g"][0]))
        self.assertTrue(np.isfinite(out["g"][1]))

    def test_a_nan_row_makes_lapack_reject_the_whole_design(self):
        """Why the reporting fits mask: one NaN row poisons the matrix norm."""
        from geofuse.fusion import MetricFusionEngine

        rng = np.random.default_rng(13)
        n = 500
        g = rng.normal(size=n)
        y = 0.5 * g + rng.normal(size=n)
        cov = rng.normal(size=(n, 2))
        cov[7, 0] = np.nan  # what reindex on an unmatched polygon_id produces

        design = np.column_stack([np.ones(n), g, cov])
        self.assertIsNone(er.make_ols_fitter(y)(design, ["c", "g", "x1", "x2"]))

        keep = MetricFusionEngine._finite_rows(y, g, cov)
        self.assertEqual(int((~keep).sum()), 1)
        out = er.make_ols_fitter(y[keep])(design[keep], ["c", "g", "x1", "x2"])
        self.assertIsNotNone(out)
        self.assertAlmostEqual(out["g"][0], 0.5, delta=0.15)

    def test_a_tied_exposure_still_reports_quartiles(self):
        rng = np.random.default_rng(12)
        n = 3000
        # Three quarters of the mass on one value — the shape a terrain channel
        # takes where most pixels carry no terrain vegetation at all.
        g = np.where(rng.random(n) < 0.75, 0.0, rng.random(n))
        y = 0.3 * g + rng.normal(size=n)
        fitter = er.make_ols_fitter(y)
        self.assertIsNotNone(er.quartile_terms(g, fitter, None))
        self.assertIsNotNone(er.spline_nonlinearity_test(g, fitter, None))


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


class TestSelectionGranularity(unittest.TestCase):
    """The three-stage weight/radius/weight ladder and its defaults."""

    def test_default_bins_are_coarse_enough_to_be_selectable(self):
        from geofuse import cgi_formulas as cf

        self.assertEqual(cf.WEIGHT_BIN_PCT, 20)
        self.assertLess(cf.WEIGHT_REFINE_BIN_PCT, cf.WEIGHT_BIN_PCT)
        # The audit's measured threshold: at 66 cells a strong planted signal
        # is recovered 0 % of the time, at 21 cells it is recovered always.
        self.assertLessEqual(cf.weight_cell_count("weighted_average"), 21)
        self.assertLessEqual(cf.weight_cell_count("synergy"), 56)

    def test_refinement_selects_a_subset_of_the_coarse_cell(self):
        """The invariant that makes stage 3 safe.

        Cell *labels* do not nest across bin widths — ``_snap_weight_buckets``
        rounds to nearest, so a 10 % and a 20 % grid have unaligned boundaries.
        What has to hold is the membership property: re-binning the trials that
        are already inside the winning coarse cell partitions exactly those
        trials, so the refined winner is a subset of them and the final weights
        stay a convex combination of trials the coarse cell contained.
        """
        from geofuse import cgi_formulas as cf

        rng = np.random.default_rng(0)
        trials = []
        for _ in range(4000):
            w = rng.integers(0, 21, 3) * 5
            trials.append(
                {
                    "veg_weight": int(w[0]),
                    "terrain_weight": int(w[1]),
                    "ndvi_weight": int(w[2]),
                }
            )
        by_coarse: dict[tuple, list] = {}
        for t in trials:
            by_coarse.setdefault(
                cf.weight_cell_key("weighted_average", t, 20), []
            ).append(t)

        # Take the busiest coarse cell and re-bin only its members.
        members = max(by_coarse.values(), key=len)
        fine: dict[tuple, list] = {}
        for t in members:
            fine.setdefault(cf.weight_cell_key("weighted_average", t, 10), []).append(t)

        self.assertEqual(sum(len(v) for v in fine.values()), len(members))
        for group in fine.values():
            for t in group:
                self.assertIn(t, members)

    def test_a_coarse_cell_really_does_contain_several_fine_cells(self):
        """Otherwise stage 3 would be a no-op and the coarsening a pure loss."""
        from collections import Counter

        from geofuse import cgi_formulas as cf

        rng = np.random.default_rng(1)
        by_coarse: dict[tuple, set] = {}
        for _ in range(2000):
            w = rng.integers(0, 21, 3) * 5
            params = {
                "veg_weight": int(w[0]),
                "terrain_weight": int(w[1]),
                "ndvi_weight": int(w[2]),
            }
            by_coarse.setdefault(
                cf.weight_cell_key("weighted_average", params, 20), set()
            ).add(cf.weight_cell_key("weighted_average", params, 10))
        counts = Counter(len(v) for v in by_coarse.values())
        self.assertGreater(max(counts), 1)
        self.assertGreaterEqual(max(len(v) for v in by_coarse.values()), 4)


class TestModeration(unittest.TestCase):
    """Effect modification: the interaction test and the simple slopes."""

    @staticmethod
    def _categorical(seed=0, slopes=(-0.2, -0.5, -0.9), n=9000):
        rng = np.random.default_rng(seed)
        cov = rng.normal(size=(n, 3))
        g = rng.normal(size=n)
        m = rng.integers(0, len(slopes), n).astype(float)
        truth = np.array([slopes[int(v)] for v in m])
        y = truth * g + 0.3 * m + cov @ np.array([0.4, -0.2, 0.1]) + rng.normal(size=n)
        return y, g, m, cov

    def test_simple_slopes_match_separate_stratified_fits(self):
        """The claim that makes this worth having over running three jobs."""
        y, g, m, cov = self._categorical()
        out = er.moderation_terms(
            g, m, er.make_ols_fitter(y), cov, categorical=True, moderator_name="M"
        )
        self.assertIsNotNone(out)
        by_level = {r["moderator_value"]: r["slope"] for r in out["simple_slopes"]}
        for level in (0.0, 1.0, 2.0):
            mask = m == level
            fit = er.make_ols_fitter(y[mask])(
                np.column_stack([np.ones(int(mask.sum())), g[mask], cov[mask]]),
                ["intercept", "greenery", "c0", "c1", "c2"],
            )
            self.assertAlmostEqual(by_level[level], fit["greenery"][0], places=2)

    def test_recovers_known_slopes(self):
        slopes = (-0.2, -0.5, -0.9)
        y, g, m, cov = self._categorical(slopes=slopes)
        out = er.moderation_terms(
            g, m, er.make_ols_fitter(y), cov, categorical=True, moderator_name="M"
        )
        for row in out["simple_slopes"]:
            self.assertAlmostEqual(
                row["slope"], slopes[int(row["moderator_value"])], delta=0.05
            )
        self.assertLess(out["interaction_p"], 1e-10)

    def test_does_not_fire_when_the_slope_is_constant(self):
        rng = np.random.default_rng(3)
        n = 9000
        cov = rng.normal(size=(n, 3))
        g = rng.normal(size=n)
        m = rng.integers(0, 3, n).astype(float)
        y = -0.5 * g + 0.3 * m + cov @ np.array([0.4, -0.2, 0.1]) + rng.normal(size=n)
        out = er.moderation_terms(
            g, m, er.make_ols_fitter(y), cov, categorical=True, moderator_name="M"
        )
        self.assertGreater(out["interaction_p"], 0.01)

    def test_continuous_moderator_is_centred_and_read_at_one_sd(self):
        rng = np.random.default_rng(4)
        n = 9000
        g = rng.normal(size=n)
        m = rng.normal(size=n)
        y = (-0.5 - 0.4 * m) * g + 0.2 * m + rng.normal(size=n)
        out = er.moderation_terms(
            g, m, er.make_ols_fitter(y), None, categorical=False, moderator_name="Mc"
        )
        self.assertFalse(out["categorical"])
        self.assertAlmostEqual(out["centred_at"], float(np.mean(m)), places=6)
        self.assertEqual(len(out["simple_slopes"]), 3)
        for row in out["simple_slopes"]:
            expected = -0.5 - 0.4 * row["moderator_value"]
            self.assertAlmostEqual(row["slope"], expected, delta=0.05)

    def test_binary_outcome_reports_odds_ratios(self):
        rng = np.random.default_rng(5)
        n = 12000
        g = rng.normal(size=n)
        m = rng.integers(0, 2, n).astype(float)
        eta = (-0.3 - 0.6 * m) * g + 0.2 * m
        y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
        out = er.moderation_terms(
            g, m, bl.make_logit_fitter(y), None,
            categorical=True, logistic=True, moderator_name="sex",
        )
        self.assertIsNotNone(out)
        for row in out["simple_slopes"]:
            self.assertIn("odds_ratio", row)
            self.assertAlmostEqual(
                row["odds_ratio"], float(np.exp(row["slope"])), places=6
            )
        self.assertLess(out["interaction_p"], 0.01)

    def test_single_level_moderator_is_declined(self):
        rng = np.random.default_rng(6)
        n = 500
        g = rng.normal(size=n)
        self.assertIsNone(
            er.moderation_terms(
                g, np.ones(n), er.make_ols_fitter(g + rng.normal(size=n)),
                None, categorical=True,
            )
        )


class TestIncrementalR2(unittest.TestCase):
    """``r2`` must measure association, not agreement of scales."""

    def test_uncontrolled_r2_is_a_fitted_regression_not_an_identity_score(self):
        """The defect that made an unadjusted score read as -1.06.

        A composite on [0, 1] against a CES-D score on [0, 30] scored as a
        direct prediction returns a large negative number describing the scale
        gap. Scored as a fitted regression it returns the association.
        """
        rng = np.random.default_rng(0)
        n = 5000
        outcome = rng.gamma(2, 2.5, n)              # 0-30-ish, like CES-D-10
        composite = rng.uniform(0, 1, n) + 0.02 * outcome   # 0-1 greenery index
        s = osc.score("r2", outcome, composite)
        self.assertGreaterEqual(s, 0.0)
        self.assertLess(s, 1.0)
        # It must track the genuine explained variance, not the scale offset.
        X = np.column_stack([np.ones(n), composite])
        beta, *_ = np.linalg.lstsq(X, outcome, rcond=None)
        resid = outcome - X @ beta
        expected = 1.0 - resid @ resid / float(((outcome - outcome.mean()) ** 2).sum())
        self.assertAlmostEqual(s, expected, places=10)

    def test_shifting_the_composite_scale_leaves_the_score_alone(self):
        """R² of a fitted regression is invariant to affine rescaling."""
        rng = np.random.default_rng(1)
        n = 3000
        outcome = rng.normal(10, 4, n)
        composite = rng.uniform(0, 1, n) + 0.05 * outcome
        base = osc.score("r2", outcome, composite)
        for scale, shift in ((30.0, 0.0), (1.0, 100.0), (0.01, -5.0)):
            self.assertAlmostEqual(
                osc.score("r2", outcome, composite * scale + shift), base, places=8
            )

    def test_incremental_over_covariates_is_still_incremental(self):
        rng = np.random.default_rng(2)
        n = 4000
        cov = rng.normal(size=(n, 3))
        composite = rng.normal(size=n)
        outcome = cov @ np.array([1.0, -0.5, 0.25]) + 0.3 * composite + rng.normal(size=n)
        full = osc.score("r2", outcome, composite, cov)
        alone = osc.score("r2", outcome, composite)
        # Adjusting for covariates that carry most of the variance leaves the
        # greenery term a smaller share than it claims on its own.
        self.assertGreater(alone, full)
        self.assertGreater(full, 0.0)


class TestScaleInvariance(unittest.TestCase):
    """A covariate's measurement unit must not change any answer.

    Not a style preference: the estimating equations go through normal
    equations, whose condition number is the square of the design's. Before the
    designs were column-scaled, a covariate a billion times another's returned
    ``|z| = 2e20`` — confidently wrong rather than failing.
    """

    def test_gee_is_invariant_to_covariate_units(self):
        y, g, eid, cov = make_panel(n_entities=800, seed=0, beta_g=0.30)
        base_z = bl.score_gee_logit("gee_logit_tstat", y, g, eid, cov)
        base_coef = bl.score_gee_logit("gee_logit_coef", y, g, eid, cov)
        for scale in (1e3, 1e6, 1e9, 1e12):
            rescaled = cov.copy()
            rescaled[:, 0] *= scale
            self.assertAlmostEqual(
                bl.score_gee_logit("gee_logit_tstat", y, g, eid, rescaled),
                base_z, places=6, msg=f"|z| moved at scale {scale:g}",
            )
            self.assertAlmostEqual(
                bl.score_gee_logit("gee_logit_coef", y, g, eid, rescaled),
                base_coef, places=6, msg=f"coef moved at scale {scale:g}",
            )

    def test_fast_scorer_is_invariant_to_covariate_units(self):
        y, g, eid, cov = make_panel(n_entities=800, seed=1, beta_g=0.30)
        baseline = bl.estimate_fold_baseline(y, eid, cov)
        base_z = bl.score_gee_logit_fast("gee_logit_tstat", g, baseline)
        for scale in (1e3, 1e9):
            rescaled = cov.copy()
            rescaled[:, 0] *= scale
            shifted = bl.estimate_fold_baseline(y, eid, rescaled)
            self.assertAlmostEqual(
                bl.score_gee_logit_fast("gee_logit_tstat", g, shifted),
                base_z, places=6,
            )

    def test_cross_sectional_metrics_are_invariant(self):
        """Why normalising the inputs is not required for the OLS-based path."""
        rng = np.random.default_rng(2)
        n = 3000
        cov = rng.normal(size=(n, 3))
        g = rng.uniform(0, 1, n)
        y = 5.0 + 3.0 * g + cov @ np.array([1.0, -0.5, 0.25]) + rng.normal(size=n)

        def minmax(a):
            a = np.asarray(a, dtype=float)
            lo, hi = a.min(axis=0), a.max(axis=0)
            return (a - lo) / np.where(hi - lo == 0, 1, hi - lo)

        for metric in ("r2", "distance_corr", "nrmse", "mutual_info"):
            self.assertAlmostEqual(
                osc.score(metric, y, g, cov),
                osc.score(metric, minmax(y), minmax(g), minmax(cov)),
                places=8,
                msg=f"{metric} is not scale-invariant",
            )
