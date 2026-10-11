"""The radius prior can be tied to the range a ladder spans, not to its rungs.

Under ``radius_prior="rungs"`` the kernel-location prior is the mean and SD of
the listed log radii and every rung counts alike, so a linear ladder (dense at
its top in log distance) pulls the prior and the kernel's mass towards its top.
``"range"`` takes the prior from the range the rungs span and weighs each rung
by the log distance it covers. These tests draw from the model's own prior
(the fitted NumPyro model run through ``Predictive``), so they check the model
that is sampled, not a copy of it.
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

LINEAR = np.arange(50, 2001, 50, dtype=float)
LOG15 = np.unique(np.round(np.geomspace(50, 2000, 15)))


def _prior_radius_w(radii, radius_prior, kernel="lognormal", mask=None, n=6000):
    """``radius_w`` draws from the prior of the model ``fit`` builds."""
    import jax
    import jax.numpy as jnp
    from numpyro.infer import Predictive

    rng = np.random.default_rng(0)
    nc = 1 if mask is None else len(mask)
    E = rng.normal(size=(30, nc, len(radii), 1))
    y = rng.normal(size=30)
    mcmc = bi.fit(
        E,
        y,
        radii=radii,
        stats=("mean",),
        radius_mask=mask,
        radius_kernel=kernel,
        aggregator="fixed",
        draws=2,
        warmup=2,
        chains=1,
        radius_prior=radius_prior,
    )
    pred = Predictive(mcmc.sampler.model, num_samples=n, return_sites=["radius_w"])
    return np.asarray(pred(jax.random.PRNGKey(1), jnp.asarray(E), None)["radius_w"])


def _prior_r50(radii, radius_prior):
    k = _prior_radius_w(radii, radius_prior)
    d = bi.distance_quantiles(k, radii, q=(0.5,))[0.5][:, 0]
    return np.percentile(d, [2.5, 50, 97.5])


class PriorMomentsTests(unittest.TestCase):
    def test_range_prior_is_the_log_uniform_moments(self):
        c, s = bi._radius_prior(LINEAR, "range")
        a, b = math.log(50), math.log(2000)
        self.assertAlmostEqual(c, (a + b) / 2)
        self.assertAlmostEqual(s, (b - a) / math.sqrt(12))
        self.assertAlmostEqual(math.exp(c), math.sqrt(50 * 2000))
        self.assertEqual(bi._radius_prior(LINEAR, "range"), bi._radius_prior(LOG15, "range"))

    def test_rung_prior_is_unchanged(self):
        lr = np.log(LINEAR)
        self.assertEqual(bi._radius_prior(LINEAR), (float(lr.mean()), float(lr.std())))
        self.assertEqual(bi._radius_prior(LINEAR), bi._radius_prior(LINEAR, "rungs"))

    def test_a_point_rung_is_outside_the_range(self):
        r = np.array([0.0, 100.0, 250.0, 500.0, 1000.0])
        c, _ = bi._radius_prior(r, "range")
        self.assertAlmostEqual(math.exp(c), math.sqrt(100 * 1000))


class LogWidthTests(unittest.TestCase):
    def test_trapezoid_widths_in_log_radius(self):
        r = np.array([100.0, 200.0, 400.0, 1000.0])
        w = bi.log_widths(r)[0]
        g = np.diff(np.log(r))
        np.testing.assert_allclose(w, [g[0] / 2, (g[0] + g[1]) / 2, (g[1] + g[2]) / 2, g[2] / 2])
        self.assertAlmostEqual(w.sum(), math.log(10))

    def test_widths_follow_each_channels_own_rungs(self):
        r = np.array([50.0, 100.0, 200.0, 400.0])
        mask = np.array([[True, True, True, True], [False, True, False, True]])
        w = bi.log_widths(r, mask)
        self.assertEqual(w[1, 0], 0.0)
        self.assertEqual(w[1, 2], 0.0)
        np.testing.assert_allclose(w[1, [1, 3]], [math.log(4) / 2] * 2)

    def test_point_and_single_rungs(self):
        r = np.array([0.0, 100.0, 250.0, 500.0])
        w = bi.log_widths(r)[0]
        self.assertAlmostEqual(w[0], w[1:].min())
        self.assertEqual(bi.log_widths(np.array([300.0]))[0, 0], 1.0)


class SpacingInvarianceTests(unittest.TestCase):
    """G10 acceptance (1): one range, two spacings, the same prior R50."""

    @classmethod
    def setUpClass(cls):
        cls.range_lin = _prior_r50(LINEAR, "range")
        cls.range_log = _prior_r50(LOG15, "range")
        cls.rungs_lin = _prior_r50(LINEAR, "rungs")
        cls.rungs_log = _prior_r50(LOG15, "rungs")

    def test_range_prior_r50_agrees_across_spacings(self):
        rel = np.abs(self.range_lin - self.range_log) / self.range_log
        # Median and upper tail agree to well within 5 %. The lower tail is
        # bounded by the linear ladder's resolution: it has no rung between
        # 50 and 100 m, so it cannot express a disc blend in that band.
        self.assertLess(rel[1], 0.05)
        self.assertLess(rel[2], 0.05)
        self.assertLess(rel[0], 0.12)

    def test_rung_prior_does_not(self):
        rel = np.abs(self.rungs_lin - self.rungs_log) / self.rungs_log
        self.assertGreater(rel[1], 1.0)


class KernelWeightingTests(unittest.TestCase):
    def test_fixed_kernel_is_uniform_in_log_radius(self):
        k = _prior_radius_w(LINEAR, "range", kernel="fixed", n=3)
        w = bi.log_widths(LINEAR)[0]
        np.testing.assert_allclose(k[0, 0], w / w.sum(), rtol=1e-5)

    def test_dirichlet_prior_mean_follows_the_widths(self):
        k = _prior_radius_w(LOG15[:6], "range", kernel="dirichlet", n=20000)
        w = bi.log_widths(LOG15[:6])[0]
        np.testing.assert_allclose(k[:, 0].mean(0), w / w.sum(), atol=0.01)

    def test_masked_rungs_take_no_weight(self):
        r = np.array([50.0, 100.0, 200.0, 400.0])
        mask = np.array([[True, True, True, True], [False, True, False, True]])
        k = _prior_radius_w(r, "range", mask=mask, n=500)
        self.assertEqual(float(np.abs(k[:, 1, [0, 2]]).max()), 0.0)
        np.testing.assert_allclose(k[:, 1].sum(-1), 1.0, rtol=1e-6)


class RecordedModeTests(unittest.TestCase):
    def test_summary_and_refit_kwargs_carry_the_mode(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(60, 1, 4, 1))
        y = X[:, 0, 2, 0] + rng.normal(size=60)
        radii = np.array([50.0, 100.0, 200.0, 400.0])
        post, _grid, kwargs = bi.grid_posterior(
            X,
            y,
            channels=("ndvi",),
            channel_index=[0],
            radii=radii,
            stats=("mean",),
            radius_idx=[[0, 1, 2, 3]],
            form="linear",
            picked=((200, "mean"),),
            draws=50,
            warmup=50,
            chains=1,
            radius_prior="range",
        )
        self.assertEqual(kwargs["radius_prior"], "range")
        s = post.summary()
        self.assertEqual(s["radius_prior"], "range")
        c, sd = bi._radius_prior(radii, "range")
        np.testing.assert_allclose(
            s["peak_radius_prior_ci"], [math.exp(c - 1.96 * sd), math.exp(c + 1.96 * sd)]
        )

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            bi.fit(
                np.zeros((10, 1, 2, 1)),
                np.zeros(10),
                radii=[100, 200],
                draws=1,
                warmup=1,
                chains=1,
                radius_prior="ladder",
            )


if __name__ == "__main__":
    unittest.main()
