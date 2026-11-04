"""Raw-unit effect sizes: the slope per IQR and per 5th–95th percentile.

Pinned here: ``raw_slope`` is the OLS coefficient of the outcome on
``exposure / scale`` with the covariates (and a smooth's columns): a planted
0.5 per unit gives 0.5 × IQR per IQR, a comparator twice the exposure gives
the same per-IQR contrast and half the per-unit slope, and its t is the
per-SD slope's.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from geofuse import negative_controls as nc


class TestRawSlope(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.n = 4000
        self.x = rng.gamma(2.0, 0.1, self.n)
        self.cov = rng.normal(size=(self.n, 2))
        self.y = 0.5 * self.x + self.cov @ [0.3, -0.2] + rng.normal(0, 0.05, self.n)
        q25, q75 = np.quantile(self.x, [0.25, 0.75])
        self.iqr = q75 - q25

    def test_a_planted_slope_is_recovered_per_iqr(self):
        r = nc.raw_slope(self.x, self.y, self.cov, scale=self.iqr)
        self.assertAlmostEqual(r["estimate"], 0.5 * self.iqr, delta=3 * r["se"])
        self.assertLess(r["ci_low"], 0.5 * self.iqr)
        self.assertGreater(r["ci_high"], 0.5 * self.iqr)
        self.assertEqual(r["scale"], self.iqr)

    def test_a_rescaled_exposure_gives_the_same_contrast(self):
        a = nc.raw_slope(self.x, self.y, self.cov, scale=self.iqr)
        b = nc.raw_slope(2 * self.x, self.y, self.cov, scale=2 * self.iqr)
        self.assertAlmostEqual(a["estimate"], b["estimate"], places=12)
        self.assertAlmostEqual(a["ci_low"], b["ci_low"], places=12)
        unit_a = nc.raw_slope(self.x, self.y, self.cov)
        unit_b = nc.raw_slope(2 * self.x, self.y, self.cov)
        self.assertAlmostEqual(unit_b["estimate"], unit_a["estimate"] / 2, places=12)

    def test_it_is_ols_on_the_scaled_exposure_with_a_smooth(self):
        from geofuse import spatial_basis

        rng = np.random.default_rng(9)
        xy = rng.uniform(0, 20_000, (self.n, 2))
        basis = spatial_basis.build_block_basis(xy, max_df=8)
        smooth = basis.design(8)
        controls = np.column_stack([self.cov, smooth])
        r = nc.raw_slope(self.x, self.y, controls, scale=self.iqr)
        design = np.column_stack([np.ones(self.n), self.x / self.iqr, controls])
        coef, *_ = np.linalg.lstsq(design, self.y, rcond=None)
        self.assertAlmostEqual(r["estimate"], coef[1], delta=1e-8)
        resid = self.y - design @ coef
        dof = self.n - np.linalg.matrix_rank(design)
        cov_b = (resid @ resid / dof) * np.linalg.pinv(design.T @ design)
        self.assertAlmostEqual(r["se"], np.sqrt(cov_b[1, 1]), delta=1e-8)
        self.assertEqual(r["df"], dof)

    def test_its_t_is_the_per_sd_slopes(self):
        r = nc.raw_slope(self.x, self.y, self.cov, scale=self.iqr)
        sd = nc.partial_slope(self.x, self.y, self.cov)
        self.assertAlmostEqual(r["t"], sd["t"], places=8)

    def test_a_missing_scale_gives_no_estimate(self):
        r = nc.raw_slope(self.x, self.y, self.cov, scale=float("nan"))
        self.assertTrue(np.isnan(r["estimate"]))


if __name__ == "__main__":
    unittest.main()
