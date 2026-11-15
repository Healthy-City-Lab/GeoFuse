"""Null outcomes for calibrating a decision rule.

Pinned here, with a held-out slope rule (ten random splits, corrected
resampled t) standing in for the search:

- a spatial surrogate's variogram brackets the outcome's in every distance
  bin, and the surrogate never reads the exposure;
- for a well-specified model (the group means adjusted) the free and the
  spatial null both reject at about 5 %;
- when the outcome's group means line up with the exposure's, the unadjusted
  rule rejects far more often under the spatial null than under the free one;
- free nulls permute within strata, each null's draw depends only on its seed,
  kind and index, and the sequential rule stops at its evaluation point only.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from geofuse import null_outcomes as no
from geofuse.negative_controls import partial_slope
from geofuse.statistical_testing import corrected_resampled_t


def _exponential_field(xy, groups, rng, range_m):
    """A Gaussian field with exponential covariance, drawn per group."""
    out = np.zeros(len(xy))
    for g in np.unique(groups):
        rows = groups == g
        d = np.hypot(*(xy[rows][:, None, :] - xy[rows][None, :, :]).transpose(2, 0, 1))
        chol = np.linalg.cholesky(np.exp(-d / range_m) + 1e-9 * np.eye(rows.sum()))
        out[rows] = chol @ rng.standard_normal(rows.sum())
    return out


def _study(seed=1, n_groups=8, per=150):
    """Entities in ``n_groups`` areas far apart, each area ~3 km across."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, 200_000, (n_groups, 2))
    groups = np.repeat(np.arange(n_groups), per)
    xy = centres[groups] + rng.normal(0, 3000, (n_groups * per, 2))
    return rng, xy, groups


def _rule_p(y, x, cov, k=10, share=0.25):
    """Corrected resampled t of the held-out slope over ``k`` random splits."""
    n = len(y)
    slopes = []
    for s in range(k):
        rng = np.random.default_rng(100 + s)
        test = np.zeros(n, dtype=bool)
        test[rng.choice(n, int(share * n), replace=False)] = True
        part = None if cov is None else cov[test]
        slopes.append(partial_slope(x[test], y[test], part)["beta"])
    return corrected_resampled_t(np.asarray(slopes), share)["p_value"]


class TestSpatialSurrogates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng, cls.xy, cls.groups = _study()
        means = rng.normal(0, 1, 8)
        cls.y = (
            means[cls.groups]
            + _exponential_field(cls.xy, cls.groups, rng, 1500)
            + rng.normal(0, 0.5, len(cls.groups))
        )
        cls.null = no.SpatialNull(cls.y, cls.xy, cls.groups, seed=3)
        cls.draws = [cls.null.draw(no.null_seed(1, "spatial", r)) for r in range(100)]

    def test_the_variogram_is_matched_in_every_bin(self):
        table = self.null.match_table([d["variogram"] for d in self.draws])
        self.assertEqual(len(table), len(no.VARIOGRAM_BINS_M) - 1)
        for row in table:
            self.assertGreater(row["n_pairs"], 0)
            self.assertLessEqual(row["surrogate_p05"], row["outcome"], row)
            self.assertGreaterEqual(row["surrogate_p95"], row["outcome"], row)

    def test_group_means_are_permuted_and_spread_kept(self):
        y = self.draws[0]["y"]
        means = sorted(y[self.groups == g].mean() for g in range(8))
        real = sorted(self.y[self.groups == g].mean() for g in range(8))
        np.testing.assert_allclose(means, real, atol=1e-9)
        dev = y - np.array([y[self.groups == g].mean() for g in self.groups])
        self.assertAlmostEqual(dev.std(), self.null.deviations.std(), places=9)

    def test_a_null_is_the_same_on_a_rerun(self):
        again = self.null.draw(no.null_seed(1, "spatial", 4))["y"]
        np.testing.assert_array_equal(again, self.draws[4]["y"])
        other = self.null.draw(no.null_seed(1, "free", 4))["y"]
        self.assertFalse(np.allclose(other, self.draws[4]["y"]))


class TestDecisionRuleRates(unittest.TestCase):
    """200 nulls of each kind through the ten-split held-out slope rule."""

    N = 200

    @classmethod
    def setUpClass(cls):
        rng, xy, cls.groups = _study()
        means = rng.normal(0, 1, 8)
        cls.x = 0.8 * means[cls.groups] + _exponential_field(xy, cls.groups, rng, 2000)
        cls.y = means[cls.groups] + rng.normal(0, 1, len(cls.groups))
        cls.dummies = (cls.groups[:, None] == np.arange(1, 8)).astype(float)
        cls.spatial = no.SpatialNull(cls.y, xy, cls.groups, seed=3)

    def _rate(self, kind, cov):
        hits = 0
        for r in range(self.N):
            rng = no.null_seed(7, kind, r)
            y = (
                no.free_null(self.y, rng)
                if kind == "free"
                else self.spatial.draw(rng)["y"]
            )
            hits += _rule_p(y, self.x, cov) < no.NULL_ALPHA
        return hits, no.clopper_pearson(hits, self.N)

    def test_a_well_specified_rule_rejects_at_its_level(self):
        for kind in ("free", "spatial"):
            hits, (lo, hi) = self._rate(kind, self.dummies)
            self.assertLessEqual(lo, 0.05, (kind, hits))
            self.assertGreaterEqual(hi, 0.05, (kind, hits))

    def test_aligned_group_means_inflate_the_spatial_null(self):
        free, (_, free_hi) = self._rate("free", None)
        spatial, (spatial_lo, _) = self._rate("spatial", None)
        self.assertGreater(spatial_lo, free_hi, (free, spatial))


class TestHelpers(unittest.TestCase):
    def test_free_nulls_permute_within_strata(self):
        y = np.arange(20.0)
        strata = np.repeat([0, 1], 10)
        z = no.free_null(y, np.random.default_rng(0), strata)
        self.assertEqual(sorted(z[:10]), list(range(10)))
        self.assertEqual(sorted(z[10:]), list(range(10, 20)))
        self.assertFalse(np.array_equal(z, y))

    def test_clopper_pearson(self):
        lo, hi = no.clopper_pearson(0, 50)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.0711, places=4)
        lo, hi = no.clopper_pearson(10, 200)
        self.assertAlmostEqual(lo, 0.0242, places=4)
        self.assertAlmostEqual(hi, 0.0900, places=4)

    def test_the_sequential_rule_decides_at_its_point_only(self):
        rule = {"after": 50, "low": 1, "high": 7}
        self.assertEqual(no.stop_decision(1, 50, **rule), "not inflated")
        self.assertEqual(no.stop_decision(7, 50, **rule), "inflated")
        self.assertIsNone(no.stop_decision(3, 50, **rule))
        self.assertIsNone(no.stop_decision(0, 49, **rule))
        self.assertIsNone(no.stop_decision(0, 51, **rule))
        self.assertIsNone(no.stop_decision(0, 50, after=None, low=1, high=7))

    def test_kinds(self):
        self.assertEqual(no.kinds("both"), ["spatial", "free"])
        self.assertEqual(no.kinds("free"), ["free"])
        with self.assertRaises(ValueError):
            no.kinds("other")


if __name__ == "__main__":
    unittest.main()
