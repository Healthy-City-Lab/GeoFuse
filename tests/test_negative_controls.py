"""Negative-control transfer test: specificity of a tuned exposure.

A confounder U that moves with the exposure and drives both the target and the
control makes the target association partly non-specific. These tests plant
exactly that and check that the control picks up the confounded part, that the
paired contrast separates a real effect from a shared one, and that the flag
fires when there is nothing specific to find.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from geofuse import bayesian_index as bi
from geofuse import negative_controls as nc


class TestPartialSlope(unittest.TestCase):
    def test_matches_ols_with_covariates(self):
        import statsmodels.api as sm

        rng = np.random.default_rng(1)
        n = 500
        cov = np.column_stack([rng.normal(size=n), rng.integers(0, 3, size=n)])
        x = 0.5 * cov[:, 0] + rng.normal(size=n)
        y = 0.3 * x + 0.8 * cov[:, 0] + rng.normal(size=n)
        got = nc.partial_slope(x, y, cov)
        fit = sm.OLS(y, sm.add_constant(np.column_stack([x, cov]))).fit()
        self.assertAlmostEqual(got["t"], float(fit.tvalues[1]), places=8)
        self.assertEqual(got["df"], int(fit.df_resid))
        # In SD units of the residuals the slope is the partial correlation.
        q, _ = bi.covariate_basis(cov, n)
        xr, yr = x - q @ (q.T @ x), y - q @ (q.T @ y)
        self.assertAlmostEqual(got["beta"], float(np.corrcoef(xr, yr)[0, 1]),
                               places=10)
        self.assertLess(got["ci_low"], got["beta"])
        self.assertGreater(got["ci_high"], got["beta"])

    def test_missing_rows_are_dropped_not_propagated(self):
        rng = np.random.default_rng(2)
        x, y = rng.normal(size=200), rng.normal(size=200)
        y[:5] = np.nan
        got = nc.partial_slope(x, y)
        self.assertEqual(got["n"], 195)
        self.assertTrue(np.isfinite(got["beta"]))


class TestPairedContrast(unittest.TestCase):
    @staticmethod
    def _data(n=800, seed=3, clusters=None):
        """With ``clusters``, the target's slope varies by area.

        A shock shared by every variable moves both slopes together and cancels
        in Δ; an area-level slope is the clustering Δ is sensitive to.
        """
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        slope = 0.4
        if clusters is not None:
            slope = 0.4 + rng.normal(0.0, 0.4, size=clusters)[np.arange(n) % clusters]
        a = slope * x + rng.normal(size=n)
        b = 0.1 * x + rng.normal(size=n)
        cov = rng.normal(size=(n, 2))
        return x, a, b, cov

    def test_unit_weights_reproduce_the_two_partial_slopes(self):
        x, a, b, cov = self._data()
        got = nc.paired_contrast(x, a, b, cov, n_boot=0)
        ba = nc.partial_slope(x, a, cov)["beta"]
        bb = nc.partial_slope(x, b, cov)["beta"]
        self.assertAlmostEqual(got["delta"], abs(ba) - abs(bb), places=10)
        self.assertAlmostEqual(got["ratio"], bb / ba, places=10)

    def test_the_interval_covers_the_estimate_and_excludes_zero_here(self):
        x, a, b, cov = self._data()
        got = nc.paired_contrast(x, a, b, cov, n_boot=1000, seed=4)
        self.assertLess(got["ci_low"], got["delta"])
        self.assertGreater(got["ci_high"], got["delta"])
        self.assertGreater(got["ci_low"], 0.0)

    def test_resampling_whole_clusters_widens_a_clustered_interval(self):
        n, k = 800, 40
        x, a, b, cov = self._data(n=n, clusters=k)
        labels = np.arange(n) % k
        ent = nc.paired_contrast(x, a, b, cov, n_boot=600, seed=5)
        clu = nc.paired_contrast(x, a, b, cov, n_boot=600, seed=5, clusters=labels)
        self.assertEqual(clu["clusters"], k)
        self.assertGreater(clu["ci_high"] - clu["ci_low"],
                           ent["ci_high"] - ent["ci_low"])

    def test_the_contrast_uses_rows_where_both_outcomes_exist(self):
        x, a, b, cov = self._data()
        b = b.copy()
        b[:30] = np.nan
        got = nc.paired_contrast(x, a, b, cov, n_boot=0)
        self.assertEqual(got["n"], len(x) - 30)


def _confounded(n=3000, effect=0.25, seed=0):
    """Two-channel tensor; U rides on channel 0 at rung 0 and drives both outcomes.

    The target additionally responds to channel 1 at rung 2 when ``effect`` is
    non-zero. The control responds to U only.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2, 3, 1))
    u = rng.normal(size=n)
    X[:, 0, 0, 0] = 0.7 * u + 0.7 * rng.normal(size=n)
    target = 0.3 * u + effect * X[:, 1, 2, 0] + rng.normal(size=n)
    control = 0.3 * u + rng.normal(size=n)
    return X, target, control


def _tune_and_transfer(X, target, control, seed=0, n_boot=400):
    """Tune on 75 % of rows exactly as the sweep does, transfer on the rest."""
    n = len(target)
    te = np.random.default_rng(seed + 100).random(n) < 0.25
    tr = ~te
    yr, Xr = bi.prep(X[tr], target[tr], None)
    res = bi.sweep(Xr, np.array([100.0, 300.0, 900.0]), ["mean"], yr,
                   channels=("a", "b"), channel_index=(0, 1), splits=6,
                   workers=1, forms=("linear",))
    cols = list(res.columns)
    flat = X.reshape(n, -1)
    mu, sd = flat[tr][:, cols].mean(0), flat[tr][:, cols].std(0)
    apply_fn, _ = bi.build_index((flat[tr][:, cols] - mu) / sd, yr, "linear")
    exposure = apply_fn((flat[te][:, cols] - mu) / sd)
    return nc.transfer_test(exposure, target[te], {"grip": control[te]},
                            n_boot=n_boot, seed=seed), res


class TestTransferAcceptance(unittest.TestCase):
    """The tracker's G1 acceptance scenarios, tuned through the real sweep."""

    def test_a_real_effect_beats_the_shared_confounding(self):
        out, res = _tune_and_transfer(*_confounded(effect=0.25))
        grip = out["controls"]["grip"]
        self.assertGreater(grip["delta"], 0.0)
        self.assertGreater(grip["delta_ci_low"], 0.0)
        self.assertFalse(grip["nonspecific"])
        # The control's slope is the confounded part: positive, and well below
        # the target's.
        self.assertGreater(grip["beta"], 0.0)
        self.assertLess(grip["beta"], out["target"]["beta"])

    def test_no_true_effect_is_flagged_nonspecific_in_most_replicates(self):
        flags = []
        for s in range(6):
            out, _ = _tune_and_transfer(*_confounded(effect=0.0, seed=s), seed=s)
            flags.append(out["controls"]["grip"]["nonspecific"])
        self.assertGreaterEqual(sum(flags), 4, flags)

    def test_no_controls_means_no_entries(self):
        x = np.random.default_rng(0).normal(size=100)
        out = nc.transfer_test(x, x + 1.0, {})
        self.assertEqual(out["controls"], {})


if __name__ == "__main__":
    unittest.main()
