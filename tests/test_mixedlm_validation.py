"""Ground-truth validation of the mixed-effects scoring engine.

Synthetic panels with a known data-generating process, so every assertion has a
right answer rather than a golden value. Monte-Carlo counts are kept modest and
seeded; the bands are wide enough that a correct implementation passes but a
broken one does not.

Data-generating process::

    y_it = 1 + b_level*g_it + X_it.b_cov + b_time*t_it
         + b_between*(gbar_i * t_it) + b_within*((g_it - gbar_i) * t_it)
         + u0_i + u1_i*t_it + e_it
"""

import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geofuse import mixed_effects_scoring as mes  # noqa: E402

warnings.simplefilter("ignore")

WAVES = (0.0, 3.0, 6.0)


def make_panel(
    n_entities=600,
    *,
    b_level=0.0,
    b_time=-0.10,
    b_between=0.0,
    b_within=0.0,
    b_cov=(0.0, 0.0, 0.0),
    cov_confounds_greenery=0.0,
    sd_u0=1.0,
    sd_u1=0.15,
    sd_e=1.0,
    sd_g_within=0.4,
    freeze_exposure=False,
    seed=0,
):
    """Balanced panel of ``n_entities`` x 3 waves. Returns a dict of arrays."""
    rng = np.random.default_rng(seed)
    T = len(WAVES)
    n = n_entities * T
    eid = np.repeat(np.arange(n_entities), T)
    t = np.tile(np.asarray(WAVES, dtype=float), n_entities)

    g_between_i = rng.normal(0.0, 1.0, n_entities)
    g_between = np.repeat(g_between_i, T)
    if freeze_exposure:
        g = g_between.copy()
    else:
        g_within = rng.normal(0.0, sd_g_within, n) * (t / WAVES[-1])
        g_within -= np.repeat(g_within.reshape(n_entities, T).mean(axis=1), T)
        g = g_between + g_within

    # cov0 optionally shares variance with the exposure; cov1 drifts with time;
    # cov2 is a fixed binary trait.
    c0_i = cov_confounds_greenery * g_between_i + rng.normal(
        0.0, np.sqrt(max(1.0 - cov_confounds_greenery**2, 1e-9)), n_entities
    )
    X = np.column_stack(
        [
            np.repeat(c0_i, T),
            np.repeat(rng.normal(0.0, 1.0, n_entities), T) + t / 10.0,
            np.repeat(rng.integers(0, 2, n_entities).astype(float), T),
        ]
    )

    gbar = np.repeat(g.reshape(n_entities, T).mean(axis=1), T)
    y = (
        1.0
        + b_level * g
        + X @ np.asarray(b_cov, dtype=float)
        + b_time * t
        + b_between * (gbar * t)
        + b_within * ((g - gbar) * t)
        + np.repeat(rng.normal(0.0, sd_u0, n_entities), T)
        + np.repeat(rng.normal(0.0, sd_u1, n_entities), T) * t
        + rng.normal(0.0, sd_e, n)
    )
    return {"y": y, "g": g, "eid": eid, "t": t, "X": X}


class TestPointEstimates(unittest.TestCase):
    """Does the scorer recover effects it was given?"""

    def test_level_coefficient_recovered(self):
        d = make_panel(n_entities=1200, b_level=0.40, b_cov=(0.8, -0.5, 1.2), seed=7)
        coef = mes.score_mixedlm(
            "mixedlm_coef", d["y"], d["g"], d["eid"], d["t"], d["X"]
        )
        self.assertAlmostEqual(float(coef), 0.40, delta=0.08)

    def test_association_targets_score_their_own_term(self):
        d = make_panel(
            n_entities=1500, b_level=0.40, b_between=-0.10, b_within=0.25,
            b_cov=(0.8, -0.5, 1.2), seed=7,
        )
        got = {
            tgt: float(
                mes.score_mixedlm(
                    "mixedlm_coef", d["y"], d["g"], d["eid"], d["t"], d["X"], target=tgt
                )
            )
            for tgt in ("level", "decline_average", "decline_change")
        }
        self.assertAlmostEqual(got["level"], 0.40, delta=0.10)
        self.assertAlmostEqual(got["decline_average"], -0.10, delta=0.05)
        self.assertAlmostEqual(got["decline_change"], 0.25, delta=0.10)

    def test_report_path_matches_scoring_path(self):
        d = make_panel(
            n_entities=800, b_level=0.3, b_between=-0.1, b_within=0.25, seed=11
        )
        report = mes.decline_terms_mixedlm(
            d["y"], d["g"], d["eid"], d["t"], d["X"],
            want_between=True, want_within=True,
        )
        by_key = {r["key"]: r["coef"] for r in report["terms"]}
        for key, tgt in (("between", "decline_average"), ("within", "decline_change")):
            scored = float(
                mes.score_mixedlm(
                    "mixedlm_coef", d["y"], d["g"], d["eid"], d["t"], d["X"], target=tgt
                )
            )
            self.assertAlmostEqual(by_key[key], scored, places=6)

    def test_covariate_adjustment_removes_a_planted_confound(self):
        d = make_panel(
            n_entities=1200, b_level=0.0, b_cov=(1.0, -0.5, 1.2),
            cov_confounds_greenery=0.7, seed=5,
        )
        unadjusted = float(
            mes.score_mixedlm("mixedlm_coef", d["y"], d["g"], d["eid"], d["t"], None)
        )
        adjusted = float(
            mes.score_mixedlm("mixedlm_coef", d["y"], d["g"], d["eid"], d["t"], d["X"])
        )
        self.assertGreater(abs(unadjusted), 0.25)
        self.assertLess(abs(adjusted), 0.12)

    def test_covariate_impact_recovers_coefficients(self):
        d = make_panel(n_entities=1500, b_level=0.3, b_cov=(1.0, -0.5, 0.0), seed=17)
        out = mes.covariate_impact_mixedlm(
            d["y"], d["g"], d["eid"], d["t"], d["X"], ["strong", "mid", "null"]
        )
        by_name = {r["covariate"]: r for r in out["per_covariate"]}
        self.assertAlmostEqual(by_name["strong"]["coef"], 1.0, delta=0.08)
        self.assertAlmostEqual(by_name["mid"]["coef"], -0.5, delta=0.08)
        self.assertGreater(by_name["null"]["pvalue"], 0.01)
        self.assertGreater(
            by_name["strong"]["partial_r2"], by_name["null"]["partial_r2"]
        )


class TestCalibration(unittest.TestCase):
    """Under a true null, are the reported p-values honest?"""

    N_REPS = 80

    def _rejection_rate(self, key, **panel_kw):
        hits = 0
        for rep in range(self.N_REPS):
            d = make_panel(n_entities=250, seed=4000 + rep, **panel_kw)
            if key == "level":
                _c, p = mes.score_mixedlm(
                    "mixedlm_coef", d["y"], d["g"], d["eid"], d["t"], d["X"],
                    return_pvalue=True,
                )
            else:
                terms = mes.decline_terms_mixedlm(
                    d["y"], d["g"], d["eid"], d["t"], d["X"],
                    want_between=True, want_within=True,
                )
                p = {r["key"]: r["pvalue"] for r in terms["terms"]}[key]
            hits += int(np.isfinite(p) and float(p) < 0.05)
        return hits / self.N_REPS

    def test_null_rejection_rates_are_nominal(self):
        # Binomial 99% band at n=120 around 0.05 is roughly [0.0, 0.12].
        for key in ("level", "overall", "between", "within"):
            with self.subTest(term=key):
                rate = self._rejection_rate(key, b_cov=(0.8, -0.5, 1.2))
                self.assertLess(rate, 0.15, f"{key} rejects a true null too often")

    def test_level_effect_does_not_leak_into_time_terms(self):
        rate = self._rejection_rate("within", b_level=0.5, b_cov=(0.8, -0.5, 1.2))
        self.assertLess(rate, 0.15)

    def test_within_effect_is_detected(self):
        hits = 0
        for rep in range(40):
            d = make_panel(n_entities=400, b_within=0.30, seed=6000 + rep)
            terms = mes.decline_terms_mixedlm(
                d["y"], d["g"], d["eid"], d["t"], d["X"],
                want_between=True, want_within=True,
            )
            p = {r["key"]: r["pvalue"] for r in terms["terms"]}["within"]
            hits += int(float(p) < 0.05)
        self.assertGreater(hits / 40, 0.6)


class TestMetricCorrectness(unittest.TestCase):
    """The two metrics that were invalid as shipped."""

    def test_likelihood_ratio_tracks_the_wald_statistic(self):
        # LR and t^2 test the same restriction, so they agree closely at n large.
        for b in (0.0, 0.10, 0.25):
            with self.subTest(b=b):
                d = make_panel(n_entities=1000, b_level=b, b_cov=(0.8, -0.5, 1.2), seed=5)
                lr = float(
                    mes.score_mixedlm("mixedlm_lr", d["y"], d["g"], d["eid"], d["t"], d["X"])
                )
                tstat = float(
                    mes.score_mixedlm(
                        "mixedlm_tstat", d["y"], d["g"], d["eid"], d["t"], d["X"]
                    )
                )
                self.assertGreaterEqual(lr, 0.0)
                self.assertAlmostEqual(lr, tstat**2, delta=0.15 * max(tstat**2, 1.0))

    def test_likelihood_ratio_is_not_stuck_at_zero(self):
        d = make_panel(n_entities=1000, b_level=0.25, b_cov=(0.8, -0.5, 1.2), seed=5)
        lr = float(mes.score_mixedlm("mixedlm_lr", d["y"], d["g"], d["eid"], d["t"], d["X"]))
        self.assertGreater(lr, 10.0)

    def test_marginal_r2_survives_a_confounded_exposure(self):
        # Same |effect|, opposite signs: both explain variance, so neither may
        # come back as an exact zero.
        got = {}
        for b in (-0.3, 0.3):
            d = make_panel(
                n_entities=1000, b_level=b, b_cov=(1.0, -0.5, 1.2),
                cov_confounds_greenery=0.8, seed=9,
            )
            got[b] = float(
                mes.score_mixedlm(
                    "mixedlm_marginal_r2", d["y"], d["g"], d["eid"], d["t"], d["X"]
                )
            )
        for b, val in got.items():
            self.assertGreater(val, 0.0, f"marginal_r2 clamped to zero at b={b}")

    def test_fast_scorer_ranks_like_the_exact_refit(self):
        from scipy.stats import spearmanr

        d = make_panel(n_entities=500, b_level=0.25, b_cov=(0.8, -0.5, 1.2), seed=3)
        comp = mes.estimate_fold_components(
            d["y"], d["X"], d["t"], d["eid"],
            method="mom_em3", include_time_fixed=True, random_slope=True,
        )
        self.assertIsNotNone(comp)
        rng = np.random.default_rng(0)
        fast, exact = [], []
        for k in range(12):
            alpha = k / 11.0
            gk = alpha * d["g"] + (1 - alpha) * rng.normal(0, 1, len(d["y"]))
            fast.append(
                float(
                    mes.score_mixedlm_fast(
                        "mixedlm_tstat", d["y"], gk, d["eid"], d["t"], d["X"],
                        components=comp,
                    )
                )
            )
            exact.append(
                float(mes.score_mixedlm("mixedlm_tstat", d["y"], gk, d["eid"], d["t"], d["X"]))
            )
        self.assertGreater(float(spearmanr(fast, exact).statistic), 0.95)
        self.assertLess(float(np.max(np.abs(np.array(fast) - np.array(exact)))), 0.5)


class TestEstimability(unittest.TestCase):
    def test_time_invariant_exposure_is_not_estimable(self):
        d = make_panel(n_entities=300, freeze_exposure=True, seed=3)
        _gm, gdev, ok = mes._within_between(d["g"], d["eid"])
        self.assertFalse(ok, "float dust must not count as within-person variation")
        self.assertLess(float(np.var(gdev)), 1e-20)

    def test_within_term_reports_not_estimable(self):
        d = make_panel(n_entities=300, freeze_exposure=True, seed=3)
        out = mes.decline_terms_mixedlm(
            d["y"], d["g"], d["eid"], d["t"], d["X"],
            want_between=True, want_within=True,
        )
        self.assertFalse(out["within_estimable"])
        within = {r["key"]: r for r in out["terms"]}["within"]
        self.assertTrue(np.isnan(within["coef"]))


class TestPeriodConfounding(unittest.TestCase):
    """Wave fixed effects and the placebo, on a panel where the exposure only
    changes because the layer's vintage changed."""

    def _vintage_panel(self, seed=0, n_entities=600):
        rng = np.random.default_rng(seed)
        T = 3
        eid = np.repeat(np.arange(n_entities), T)
        wave = np.tile(np.arange(T), n_entities)
        # Waves are collected over a couple of years each, as in a real cohort,
        # so time is not a deterministic function of the wave.
        t = np.tile(np.asarray(WAVES), n_entities) + np.repeat(
            rng.uniform(0.0, 2.0, n_entities), T
        ) * np.tile(np.arange(T), n_entities)
        # Nobody moves; the layer drifts down each wave, with a little
        # per-location noise so the drift is not perfectly collinear with wave.
        g = (
            np.repeat(rng.normal(0, 1, n_entities), T)
            - 0.15 * wave
            + rng.normal(0, 0.03, n_entities * T)
        )
        # The outcome jumps between waves, then plateaus (a retest effect).
        practice = np.tile(np.asarray([0.0, 0.62, 0.68]), n_entities)
        y = (
            4.0
            + practice
            + np.repeat(rng.normal(0, 1, n_entities), T)
            + rng.normal(0, 1, n_entities * T)
        )
        return {"y": y, "g": g, "eid": eid, "t": t, "wave": wave}

    def test_diagnostic_flags_the_confound(self):
        d = self._vintage_panel(seed=1)
        rep = mes.period_confounding_report(
            d["g"], d["eid"], d["t"], wave_index=d["wave"]
        )
        self.assertTrue(rep["confounded"])
        self.assertGreater(abs(rep["within_time_corr"]), 0.5)
        self.assertEqual(len(rep["per_wave_mean_exposure"]), 3)

    def test_wave_fixed_effects_neutralise_the_confound(self):
        d = self._vintage_panel(seed=2)
        without = mes.decline_terms_mixedlm(
            d["y"], d["g"], d["eid"], d["t"], None, want_within=True
        )
        with_fe = mes.decline_terms_mixedlm(
            d["y"], d["g"], d["eid"], d["t"], None,
            want_within=True, wave_index=d["wave"],
        )
        p_without = {r["key"]: r["pvalue"] for r in without["terms"]}["within"]
        p_with = {r["key"]: r["pvalue"] for r in with_fe["terms"]}["within"]
        self.assertLess(p_without, 0.01, "the artefact should be present to begin with")
        self.assertGreater(p_with, 0.05, "wave fixed effects should absorb it")

    def test_placebo_reproduces_the_artefact(self):
        # An exposure holding only the per-wave mean carries no spatial
        # information, so a term it reproduces was never about greenery.
        d = self._vintage_panel(seed=3)
        real = mes.decline_terms_mixedlm(
            d["y"], d["g"], d["eid"], d["t"], None, want_within=True
        )
        placebo = mes.decline_terms_mixedlm(
            d["y"],
            mes.placebo_exposure(d["g"], d["wave"]),
            d["eid"], d["t"], None,
            want_within=True,
        )
        real_overall = {r["key"]: r for r in real["terms"]}["overall"]
        placebo_overall = {r["key"]: r for r in placebo["terms"]}["overall"]
        self.assertLess(placebo_overall["pvalue"], 0.05)
        self.assertGreater(
            abs(placebo_overall["coef"]), 0.5 * abs(real_overall["coef"])
        )


class TestAreaClustering(unittest.TestCase):
    """An exposure shared across a neighbourhood needs the neighbourhood in the
    model; a person-level random effect alone is not enough."""

    def _area_panel(self, seed, n_areas=30, per_area=12):
        rng = np.random.default_rng(seed)
        T = 3
        n_ent = n_areas * per_area
        area_ent = np.repeat(np.arange(n_areas), per_area)
        eid = np.repeat(np.arange(n_ent), T)
        t = np.tile(np.asarray(WAVES), n_ent)
        # Exposure is an area attribute; so is part of the outcome.
        g = np.repeat(
            rng.normal(0, 1, n_areas)[area_ent] + rng.normal(0, 0.15, n_ent), T
        )
        y = (
            4.0
            + np.repeat(rng.normal(0, 0.6, n_areas)[area_ent], T)
            + np.repeat(rng.normal(0, 1, n_ent), T)
            + rng.normal(0, 1, n_ent * T)
        )
        return {"y": y, "g": g, "eid": eid, "t": t, "area": np.repeat(area_ent, T)}

    def test_area_fixed_effects_restore_calibration(self):
        reps = 60
        hits_without = hits_with = 0
        for rep in range(reps):
            d = self._area_panel(seed=7000 + rep)
            _c, p0 = mes.score_mixedlm(
                "mixedlm_coef", d["y"], d["g"], d["eid"], d["t"], None,
                return_pvalue=True,
            )
            _c, p1 = mes.score_mixedlm(
                "mixedlm_coef", d["y"], d["g"], d["eid"], d["t"], None,
                return_pvalue=True, area_id=d["area"],
            )
            hits_without += int(float(p0) < 0.05)
            hits_with += int(float(p1) < 0.05)
        self.assertGreater(hits_without / reps, 0.12, "the inflation should be visible")
        self.assertLess(hits_with / reps, 0.13, "area effects should fix it")


class TestFormatting(unittest.TestCase):
    def test_pvalue_formatting(self):
        self.assertEqual(mes.format_pvalue(0.0), "< 1e-300")
        self.assertEqual(mes.format_pvalue(0.4603), "0.4603")
        self.assertEqual(mes.format_pvalue(1.95e-232), "1.9e-232")
        self.assertEqual(mes.format_pvalue(None), "—")
        self.assertEqual(mes.format_pvalue(float("nan")), "—")


if __name__ == "__main__":
    unittest.main(verbosity=2)
