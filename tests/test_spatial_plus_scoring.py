"""Scores under ``spatial_plus`` residualise both sides on covariates + smooth.

Pinned here:

- on a fixed smooth, ``spearman``, ``distance_corr``, ``nrmse`` and the
  direction under ``spatial_plus`` equal ``ks_aic``;
- the partial correlation behind ``spearman`` is the Frisch–Waugh one (the
  exposure residualised on covariates and smooth), so an exposure that tracks
  the covariates is not attenuated;
- ``r2`` is unchanged;
- with the exposure independent of the covariates the score moves only by
  sampling noise from the smooth-only exposure side.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from scipy.stats import rankdata

from geofuse import objective_scoring as osc


def _data(seed=0, n=800, tie=0.8):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1, size=(n, 2))
    smooth = np.column_stack(
        [np.sin(3 * xy[:, 0]), np.cos(2 * xy[:, 1]), xy[:, 0] * xy[:, 1]]
    )
    cov = rng.normal(size=(n, 2))
    x = tie * cov[:, 0] + 0.5 * smooth[:, 0] + rng.normal(size=n)
    y = 0.1 * x + cov @ [0.6, -0.3] + smooth @ [0.4, 0.2, 0.1] + rng.normal(size=n)
    return y, x, cov, smooth


def _resid(v, X):
    Xc = np.column_stack([np.ones(len(v)), X])
    return v - Xc @ np.linalg.lstsq(Xc, v, rcond=None)[0]


class TestSpatialPlusMatchesKsAic(unittest.TestCase):
    def setUp(self):
        self.y, self.x, self.cov, self.s = _data()

    def _score(self, metric, method):
        return osc.score(
            metric,
            self.y,
            self.x,
            self.cov,
            spatial_basis=self.s,
            spatial_method=method,
        )

    def test_the_residualising_metrics(self):
        for metric in ("spearman", "distance_corr", "nrmse", "r2"):
            self.assertAlmostEqual(
                self._score(metric, "spatial_plus"),
                self._score(metric, "ks_aic"),
                places=12,
                msg=metric,
            )

    def test_the_direction(self):
        for method in ("spatial_plus", "ks_aic"):
            self.assertEqual(
                osc.relationship_sign(
                    self.y,
                    self.x,
                    self.cov,
                    spatial_basis=self.s,
                    spatial_method=method,
                ),
                1,
            )

    def test_spearman_is_the_frisch_waugh_partial_correlation(self):
        ranks = np.column_stack([rankdata(c) for c in self.cov.T])
        design = np.column_stack([ranks, self.s])
        ry = _resid(rankdata(self.y), design)
        rx = _resid(rankdata(self.x), design)
        ref = abs(np.corrcoef(ry, rx)[0, 1])
        self.assertAlmostEqual(self._score("spearman", "spatial_plus"), ref, 12)
        smooth_only = abs(np.corrcoef(ry, _resid(rankdata(self.x), self.s))[0, 1])
        self.assertLess(smooth_only, ref)

    def test_r2_is_the_incremental_r2_over_covariates_and_smooth(self):
        design = np.column_stack([self.cov, self.s])
        full = np.column_stack([design, self.x])

        def r2(X):
            return 1 - np.var(_resid(self.y, X)) / np.var(self.y)

        self.assertAlmostEqual(
            self._score("r2", "spatial_plus"), r2(full) - r2(design), places=12
        )


class TestIndependentExposure(unittest.TestCase):
    def test_the_score_moves_only_by_noise(self):
        y, x, cov, s = _data(seed=3, n=4000, tie=0.0)
        new = osc.score(
            "spearman", y, x, cov, spatial_basis=s, spatial_method="spatial_plus"
        )
        ranks = np.column_stack([rankdata(c) for c in cov.T])
        ry = _resid(rankdata(y), np.column_stack([ranks, s]))
        old = abs(np.corrcoef(ry, _resid(rankdata(x), s))[0, 1])
        self.assertLess(abs(new - old), 2.0 / np.sqrt(len(y)) * 0.25)


if __name__ == "__main__":
    unittest.main()
