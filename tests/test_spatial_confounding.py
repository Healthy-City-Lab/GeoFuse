"""Tests for the spatial-confounding adjustment (KS-AIC / Spatial+).

Covers the block-diagonal thin-plate basis builder, the AIC df-selection, and
the end-to-end behaviour through the objective and mixed-effects scorers: a
smooth spatial confounder is removed while a genuine fine-scale effect survives.
"""

import os
import sys
import unittest
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from geofuse import objective_scoring as osc
from geofuse import spatial_basis as sb


def _resid(y, X):
    if X is None or X.shape[1] == 0:
        return y - y.mean()
    Xc = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return y - Xc @ beta


class TestConnectedComponents(unittest.TestCase):
    def test_two_clusters_separated_by_void(self):
        rng = np.random.default_rng(0)
        a = rng.normal([0, 0], 30, size=(120, 2))
        b = rng.normal([5000, 5000], 30, size=(120, 2))
        xy = np.vstack([a, b])
        labels, _ = sb.connected_components(xy)
        self.assertEqual(labels.max() + 1, 2)
        # k-NN connectivity never leaves singletons for n > 1.
        counts = np.bincount(labels)
        self.assertTrue((counts > 1).all())

    def test_single_blob_is_one_component(self):
        rng = np.random.default_rng(1)
        xy = rng.normal([0, 0], 100, size=(200, 2))
        labels, _ = sb.connected_components(xy)
        self.assertEqual(labels.max() + 1, 1)

    def test_explicit_eps_override(self):
        rng = np.random.default_rng(2)
        xy = rng.uniform(0, 100, size=(50, 2))
        labels, eps = sb.connected_components(xy, eps=1000.0)
        self.assertEqual(eps, 1000.0)
        self.assertEqual(labels.max() + 1, 1)  # huge eps merges everything


class TestBlockBasis(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        a = rng.normal([0, 0], 30, size=(120, 2))
        b = rng.normal([5000, 5000], 30, size=(120, 2))
        self.xy = np.vstack([a, b])
        self.labels, _ = sb.connected_components(self.xy)
        self.basis = sb.build_block_basis(self.xy, max_df=8)

    def test_block_diagonal_no_cross_cluster_columns(self):
        B = self.basis.design(self.basis.max_df)
        c0 = self.labels == 0
        active0 = (np.abs(B[c0]) > 1e-9).any(axis=0)
        active1 = (np.abs(B[~c0]) > 1e-9).any(axis=0)
        # No basis column is active in both clusters.
        self.assertEqual(int((active0 & active1).sum()), 0)

    def test_knots_only_at_observed_points(self):
        # A wide void: no basis row exists in empty space because the design is
        # only evaluated at the observed coordinates (n rows, never the void).
        self.assertEqual(self.basis.n, len(self.xy))
        self.assertEqual(self.basis.design(0).shape[0], len(self.xy))

    def test_df_ladder_monotonic(self):
        for d in range(0, self.basis.max_df + 1):
            cols = self.basis.design(d)
            self.assertEqual(cols.shape[0], len(self.xy))
        # Higher df ⇒ at least as many columns.
        self.assertGreaterEqual(
            self.basis.design(self.basis.max_df).shape[1], self.basis.design(0).shape[1]
        )

    def test_degenerate_geometry(self):
        xy = np.zeros((10, 2))  # all coincident
        basis = sb.build_block_basis(xy, max_df=5)
        self.assertFalse(basis.has_spatial)


class TestSelectDfAic(unittest.TestCase):
    def test_recovers_spatial_signal(self):
        rng = np.random.default_rng(3)
        xy = rng.uniform(0, 10000, size=(400, 2))
        y = np.sin(xy[:, 0] / 3000.0) + np.cos(xy[:, 1] / 3000.0)
        y = y + rng.normal(0, 0.2, len(xy))
        basis = sb.build_block_basis(xy, max_df=12)
        df, cols = sb.select_df_aic(y, basis, None)
        self.assertIsNotNone(df)
        self.assertIsNotNone(cols)
        self.assertGreater(cols.shape[1], 0)

    def test_pure_noise_outcome_prefers_no_or_little_spatial(self):
        rng = np.random.default_rng(4)
        xy = rng.uniform(0, 10000, size=(300, 2))
        y = rng.normal(0, 1, len(xy))  # no spatial structure
        basis = sb.build_block_basis(xy, max_df=12)
        df, _ = sb.select_df_aic(y, basis, None)
        # AIC should not load up on basis functions for pure noise.
        self.assertTrue(df is None or df <= 1)

    def test_nan_in_outcome_is_tolerated(self):
        rng = np.random.default_rng(5)
        xy = rng.uniform(0, 10000, size=(200, 2))
        y = np.sin(xy[:, 0] / 3000.0) + rng.normal(0, 0.2, len(xy))
        y[::20] = np.nan
        basis = sb.build_block_basis(xy, max_df=10)
        df, cols = sb.select_df_aic(y, basis, None)
        # Returns full-length columns aligned with the original rows.
        if cols is not None:
            self.assertEqual(cols.shape[0], len(xy))


class TestScoreSpatialAdjustment(unittest.TestCase):
    def test_between_cluster_confounder_removed(self):
        rng = np.random.default_rng(0)
        a = rng.normal([0, 0], 30, size=(120, 2))
        b = rng.normal([5000, 5000], 30, size=(120, 2))
        xy = np.vstack([a, b])
        u = np.sin(xy[:, 0] / 1500.0) + np.cos(xy[:, 1] / 1500.0)
        cgi = u + rng.normal(0, 0.3, len(xy))
        target = 2.0 * u + rng.normal(0, 0.3, len(xy))  # true CGI effect = 0
        basis = sb.build_block_basis(xy, max_df=8)
        _df, cols = sb.select_df_aic(target, basis, None)
        naive = osc.score("distance_corr", target, cgi)
        adj = osc.score(
            "distance_corr", target, cgi, spatial_basis=cols, spatial_method="ks_aic"
        )
        self.assertGreater(naive, 0.6)
        self.assertLess(adj, 0.2)

    def test_real_fine_scale_effect_survives(self):
        rng = np.random.default_rng(7)
        xy = rng.uniform(0, 10000, size=(400, 2))
        broad = np.sin(xy[:, 0] / 3000.0)
        fine = rng.normal(0, 1, len(xy))
        cgi = broad + fine
        target = 1.5 * fine + 2.0 * broad + rng.normal(0, 0.3, len(xy))
        basis = sb.build_block_basis(xy, max_df=12)
        _df, cols = sb.select_df_aic(target, basis, None)
        adj = osc.score(
            "distance_corr", target, cgi, spatial_basis=cols, spatial_method="ks_aic"
        )
        self.assertGreater(adj, 0.3)  # genuine effect kept

    def test_mutual_info_ignores_spatial_basis(self):
        rng = np.random.default_rng(8)
        xy = rng.uniform(0, 10000, size=(300, 2))
        u = np.sin(xy[:, 0] / 3000.0)
        cgi = u + rng.normal(0, 0.3, len(xy))
        target = 2.0 * u + rng.normal(0, 0.3, len(xy))
        basis = sb.build_block_basis(xy, max_df=10)
        _df, cols = sb.select_df_aic(target, basis, None)
        mi0 = osc.score("mutual_info", target, cgi)
        mi1 = osc.score(
            "mutual_info", target, cgi, spatial_basis=cols, spatial_method="ks_aic"
        )
        self.assertAlmostEqual(mi0, mi1, places=10)

    def test_none_method_matches_no_basis(self):
        rng = np.random.default_rng(9)
        xy = rng.uniform(0, 10000, size=(200, 2))
        cgi = rng.normal(0, 1, len(xy))
        target = cgi + rng.normal(0, 0.5, len(xy))
        basis = sb.build_block_basis(xy, max_df=8)
        _df, cols = sb.select_df_aic(target, basis, None)
        plain = osc.score("distance_corr", target, cgi)
        off = osc.score(
            "distance_corr", target, cgi, spatial_basis=cols, spatial_method="none"
        )
        self.assertAlmostEqual(plain, off, places=10)

    def test_invalid_method_raises(self):
        t = np.arange(10.0)
        c = np.arange(10.0)
        with self.assertRaises(ValueError):
            osc.score("distance_corr", t, c, spatial_method="bogus")


class TestMixedlmSpatialAdjustment(unittest.TestCase):
    def test_confounder_removed_in_mixed_model(self):
        from geofuse import mixed_effects_scoring as mx

        warnings.filterwarnings("ignore")
        rng = np.random.default_rng(3)
        n_ent = 120
        ent_xy = rng.uniform(0, 10000, size=(n_ent, 2))
        u = np.sin(ent_xy[:, 0] / 3000.0) + np.cos(ent_xy[:, 1] / 3000.0)
        eid, t, cx, cy, g, y = [], [], [], [], [], []
        for i in range(n_ent):
            for w in range(3):
                eid.append(i)
                t.append(float(w))
                cx.append(ent_xy[i, 0])
                cy.append(ent_xy[i, 1])
                g.append(u[i] + rng.normal(0, 0.3))
                y.append(2.0 * u[i] + 0.5 * w + rng.normal(0, 0.3))
        eid = np.array(eid)
        t = np.array(t)
        g = np.array(g)
        y = np.array(y)
        coords = np.column_stack([cx, cy])
        basis = sb.build_block_basis(coords, max_df=10)
        _df, cols = sb.select_df_aic(y, basis, None)
        none = mx.score_mixedlm("mixedlm_tstat", y, g, eid, t)
        ks = mx.score_mixedlm(
            "mixedlm_tstat", y, g, eid, t, spatial_basis=cols, spatial_method="ks_aic"
        )
        self.assertGreater(none, 5.0)
        self.assertLess(ks, none / 2.0)


if __name__ == "__main__":
    unittest.main()
