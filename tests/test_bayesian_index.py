"""Validation gates for the sweep + posterior discovery engine.

These are the checks that decide whether a reported discovery means anything:
a planted signal has to be recovered, the parsimony tiebreak must not quietly
swap which channels are active, a radius picked at the edge of the search range
has to be flagged, and the whole procedure re-run on a permuted outcome must
not keep firing. A run that passes the smoke test but fails these is producing
confident noise.

The sweeps here run with a single worker on purpose. A spawned process pool
cannot re-import ``__main__`` once another test in the same interpreter has
replaced it, which made these results depend on test ordering rather than on
the code.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from geofuse import bayesian_index as bi

RADII = np.array([200.0, 400.0, 600.0, 800.0])
STATS = ["mean", "p50", "p90"]


def _planted(n=1500, channel=1, radius=2, stat=1, strength=0.6, seed=0):
    """Synthetic cube whose outcome depends on exactly one (channel, radius, stat)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2, len(RADII), len(STATS)))
    y = strength * X[:, channel, radius, stat] + rng.normal(size=n)
    return bi.prep(X, y, None)[::-1]  # (Xr, yr)


class TestRecoversAPlantedSignal(unittest.TestCase):
    def test_the_sweep_finds_the_column_the_outcome_was_built_from(self):
        Xr, yr = _planted()
        res = bi.sweep(
            Xr, RADII, STATS, yr,
            channels=("a", "b"), channel_index=(0, 1), splits=6, workers=1,
        )
        self.assertEqual(res.picked[1], (600, "p50"))

    def test_the_weight_lands_on_the_channel_carrying_the_signal(self):
        Xr, yr = _planted()
        res = bi.sweep(
            Xr, RADII, STATS, yr,
            channels=("a", "b"), channel_index=(0, 1), splits=6, workers=1,
        )
        Z = Xr.reshape(len(Xr), -1)[:, list(res.columns)]
        kept, w = bi.simplex_fit(Z.T @ Z, Z.T @ yr)
        self.assertIn(1, kept)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)
        self.assertTrue(np.all(w >= 0.0))
        # The noise channel cannot carry the majority of a simplex weight.
        share = dict(zip(kept, w))
        self.assertGreater(share.get(1, 0.0), share.get(0, 0.0))


class TestParsimonyTiebreak(unittest.TestCase):
    """One-SE ranking must not change which channels are active.

    Ranking across channel sets let the surviving channel flip, which reverses
    the sign of the reported effect while claiming to be the same model.
    """

    def test_the_one_se_pick_keeps_the_same_active_channels(self):
        Xr, yr = _planted()
        res = bi.sweep(
            Xr, RADII, STATS, yr,
            channels=("a", "b"), channel_index=(0, 1), splits=6, workers=1,
        )
        self.assertEqual(len(res.one_se_columns), len(res.columns))
        n_radii, n_stats = Xr.shape[2], Xr.shape[3]
        picked_ch = [c // (n_radii * n_stats) for c in res.columns]
        one_se_ch = [c // (n_radii * n_stats) for c in res.one_se_columns]
        self.assertEqual(picked_ch, one_se_ch)

    def test_the_one_se_pick_is_no_larger_than_the_winner(self):
        Xr, yr = _planted()
        res = bi.sweep(
            Xr, RADII, STATS, yr,
            channels=("a", "b"), channel_index=(0, 1), splits=6, workers=1,
        )
        n_radii, n_stats = Xr.shape[2], Xr.shape[3]
        one_se = bi._decode(res.one_se_columns, n_radii, n_stats, RADII, STATS)
        self.assertLessEqual(
            sum(r for r, _ in one_se), sum(r for r, _ in res.picked)
        )


class TestBoundaryIsFlagged(unittest.TestCase):
    """A radius at the edge of the range means the optimum may be outside it."""

    def test_a_signal_at_the_largest_radius_raises_the_flag(self):
        Xr, yr = _planted(radius=len(RADII) - 1, strength=1.2)
        res = bi.sweep(
            Xr, RADII, STATS, yr,
            channels=("a", "b"), channel_index=(0, 1), splits=6, workers=1,
        )
        self.assertEqual(res.picked[1][0], 800)
        self.assertIn("b", res.boundary_hit)

    def test_an_interior_signal_does_not(self):
        Xr, yr = _planted(radius=2)
        res = bi.sweep(
            Xr, RADII, STATS, yr,
            channels=("a", "b"), channel_index=(0, 1), splits=6, workers=1,
        )
        self.assertNotIn("b", res.boundary_hit)


class TestNullCalibration(unittest.TestCase):
    """On a permuted outcome the posterior interval must mostly cover zero."""

    def test_a_shuffled_outcome_rarely_produces_an_interval_excluding_zero(self):
        Xr, yr = _planted(n=600)
        res = bi.sweep(
            Xr, RADII, STATS, yr,
            channels=("a", "b"), channel_index=(0, 1), splits=4, workers=1,
        )
        E = Xr.reshape(len(Xr), -1)[:, list(res.columns)]
        out = bi.null_calibration(E, yr, form=res.form, n=8, workers=1)
        self.assertEqual(out["runs"], 8)
        # 8 runs is too few to pin 5 %, but a procedure firing on most of them
        # is broken, not merely noisy.
        self.assertLessEqual(out["rate"], 0.5)


class TestSimplexFit(unittest.TestCase):
    def test_weights_stay_on_the_simplex_for_a_correlated_design(self):
        rng = np.random.default_rng(3)
        base = rng.normal(size=(400, 1))
        Z = np.hstack([base + 0.1 * rng.normal(size=(400, 1)) for _ in range(3)])
        y = (Z @ np.array([0.5, 0.3, 0.2])) + rng.normal(size=400) * 0.1
        kept, w = bi.simplex_fit(Z.T @ Z, Z.T @ y)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)
        self.assertTrue(np.all(w >= -1e-12))
        self.assertEqual(len(kept), len(w))


if __name__ == "__main__":
    unittest.main()
