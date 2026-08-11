"""The quartile-contrast objective, and the sweep honouring the job's metric.

The metric answers the question the greenspace literature asks of a greenness
gradient: how far apart are the outcomes of the greenest and least green
quarter, adjusted for covariates (Villeneuve et al. 2022; Irvin et al. 2024).
It is a magnitude, so a protective and a harmful gradient of the same size
score the same, and the search maximises it either way.

The second half of this file guards something easy to lose: the objective the
job selects has to decide which configuration wins, not merely be reported
after a different criterion has already chosen one.
"""

import math
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from geofuse import bayesian_index as bi
from geofuse import objective_scoring as scoring

RADII = np.array([200.0, 400.0, 600.0, 800.0])
STATS = ["mean", "p50", "p90"]


def _quartile_objective(exposure, target):
    return scoring.score("quartile_contrast", target, exposure, None)


class TestQuartileContrastMetric(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n = 4000
        self.cov = rng.normal(size=(self.n, 2))
        self.g = rng.normal(size=self.n)
        self.noise = rng.normal(size=self.n)

    def _score(self, y, g=None, cov=None):
        return scoring.score(
            "quartile_contrast", y, self.g if g is None else g,
            self.cov if cov is None else cov,
        )

    def test_it_is_registered_as_a_maximised_metric(self):
        self.assertIn("quartile_contrast", scoring.SUPPORTED_METRICS)
        self.assertIn("quartile_contrast", scoring.HIGHER_IS_BETTER)
        self.assertNotIn("quartile_contrast", scoring.BINARY_ONLY_METRICS)
        # Covariates enter the fit, so the residualization basis does not apply.
        self.assertIn("quartile_contrast", scoring.RESIDUALIZE_IGNORED)

    def test_a_steeper_gradient_scores_higher(self):
        weak = self._score(1.0 * self.g + self.noise)
        strong = self._score(6.0 * self.g + self.noise)
        self.assertGreater(strong, weak)

    def test_no_association_scores_near_zero(self):
        rng = np.random.default_rng(11)
        self.assertLess(self._score(rng.normal(size=self.n)), 0.2)

    def test_the_magnitude_ignores_direction(self):
        up = self._score(3.0 * self.g + self.noise)
        down = self._score(-3.0 * self.g + self.noise)
        self.assertGreaterEqual(up, 0.0)
        self.assertGreaterEqual(down, 0.0)
        self.assertLess(abs(up - down), 0.35 * max(up, down))

    def test_a_constant_composite_has_no_contrast(self):
        self.assertEqual(self._score(3.0 * self.g + self.noise,
                                     g=np.ones(self.n)), 0.0)

    def test_covariates_are_adjusted_for(self):
        # An outcome driven only by a covariate that is correlated with the
        # composite must not score as a greenery gradient.
        conf = self.cov[:, 0]
        g = conf + 0.25 * np.random.default_rng(5).normal(size=self.n)
        y = 4.0 * conf + self.noise
        adjusted = self._score(y, g=g)
        unadjusted = scoring.score("quartile_contrast", y, g, None)
        self.assertLess(adjusted, unadjusted)

    def test_a_degenerate_input_scores_zero_not_nan(self):
        s = scoring.score("quartile_contrast", np.ones(50), np.ones(50), None)
        self.assertEqual(s, 0.0)


class TestSweepHonoursTheObjective(unittest.TestCase):
    """The selected metric must decide the winner, not just be reported."""

    def _planted(self, n=1500, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 2, len(RADII), len(STATS)))
        y = 0.7 * X[:, 1, 2, 1] + rng.normal(size=n)
        yr, Xr = bi.prep(X, y, None)
        return Xr, yr

    def _sweep(self, Xr, yr, **kw):
        # One worker keeps this in-process: a spawned pool cannot re-import
        # __main__ once another test in the same interpreter has replaced it.
        return bi.sweep(Xr, RADII, STATS, yr, channels=("a", "b"),
                        channel_index=(0, 1), splits=6, workers=1, **kw)

    def test_the_objective_changes_the_reported_score_scale(self):
        Xr, yr = self._planted()
        default = self._sweep(Xr, yr)
        quart = self._sweep(Xr, yr, objective=_quartile_objective,
                            rescore_top=20)
        # A correlation t and a contrast magnitude are different quantities;
        # if they matched, the objective was being ignored.
        self.assertNotAlmostEqual(default.score, quart.score, places=6)

    def test_both_objectives_still_recover_a_planted_signal(self):
        Xr, yr = self._planted()
        for kw in ({}, {"objective": _quartile_objective, "rescore_top": 20}):
            res = self._sweep(Xr, yr, **kw)
            self.assertEqual(res.picked[1], (600, "p50"), kw)

    def test_the_winner_comes_from_the_rescored_shortlist(self):
        Xr, yr = self._planted()
        res = self._sweep(Xr, yr, objective=_quartile_objective, rescore_top=5)
        # Every per-split winner must be a real candidate, not an index into a
        # shortlist misread as an index into the full grid.
        for combo in res.winner_counts:
            self.assertIn(combo, [tuple(c) for c in res.combos])

    def test_a_tiny_shortlist_still_returns_a_usable_pick(self):
        Xr, yr = self._planted()
        res = self._sweep(Xr, yr, objective=_quartile_objective, rescore_top=1)
        self.assertEqual(len(res.picked), 2)
        self.assertTrue(math.isfinite(res.score))


if __name__ == "__main__":
    unittest.main()
