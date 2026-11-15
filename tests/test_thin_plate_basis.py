"""The thin-plate basis carries the null-space constraint; df selection is
rank-safe.

Pinned here:

- the radial functions' knot coefficients are orthogonal to ``[1, x, y]`` at
  the knots, ``K − 3`` of them for ``K`` knots, none below 4 knots;
- a cluster's design ``[1, linear, radial]`` is full rank;
- a constant-plus-trend exposure is absorbed by the linear terms alone (its
  radial coefficients are zero);
- on a rank-deficient df design the fast criterion equals the ``lstsq`` one;
  on a full-rank design the nested basis is the economy QR unchanged.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from geofuse import spatial_basis as sb


def _cluster(seed=0, n=400, side=10_000.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, side, size=(n, 2))


class TestConstraint(unittest.TestCase):
    def test_coefficients_are_orthogonal_to_the_linear_terms(self):
        knots = np.random.default_rng(1).normal(size=(12, 2))
        delta = sb.radial_coefficients(knots)
        self.assertEqual(delta.shape, (12, 9))
        T = np.column_stack([np.ones(12), knots])
        np.testing.assert_allclose(T.T @ delta, 0.0, atol=1e-12)

    def test_fewer_than_four_knots_give_linear_terms_only(self):
        for k in (1, 2, 3):
            self.assertEqual(sb.radial_coefficients(np.eye(k, 2)).shape[1], 0)

    def test_a_cluster_design_is_full_rank(self):
        for kw in ({"scale_m": 3000.0}, {"max_df": 8}):
            basis = sb.build_block_basis(_cluster(), **kw)
            X = np.column_stack([np.ones(basis.n), basis.linear, basis.radial])
            self.assertEqual(np.linalg.matrix_rank(X), X.shape[1], kw)

    def test_scale_mode_has_k_minus_three_radial_functions(self):
        xy = _cluster(2)
        basis = sb.build_block_basis(xy, scale_m=3000.0)
        z = xy - xy.mean(axis=0)
        span = float(np.sqrt((z**2).sum(axis=1)).std())
        knots = sb.covering_knots(z / span, 0.5 * 3000.0 / span)
        self.assertEqual(basis.radial.shape[1], len(knots) - 3)

    def test_a_trend_is_absorbed_by_the_linear_terms(self):
        xy = _cluster(3)
        trend = 2.0 + 0.3 * xy[:, 0] / 1000 - 0.1 * xy[:, 1] / 1000
        basis = sb.build_block_basis(xy, scale_m=3000.0)
        n_lin = basis.linear.shape[1]
        X = np.column_stack([np.ones(basis.n), basis.linear, basis.radial])
        coef = np.linalg.lstsq(X, trend, rcond=None)[0]
        np.testing.assert_allclose(coef[1 + n_lin :], 0.0, atol=1e-8)
        resid = trend - np.column_stack([np.ones(basis.n), basis.linear]) @ (
            np.linalg.lstsq(
                np.column_stack([np.ones(basis.n), basis.linear]), trend, rcond=None
            )[0]
        )
        self.assertLess(float(np.abs(resid).max()), 1e-9)


class TestRankSafeSelection(unittest.TestCase):
    def _deficient(self):
        basis = sb.build_block_basis(_cluster(4), max_df=6)
        dup = basis.radial[:, basis.radial_rank == 0]
        radial = np.column_stack([basis.radial, dup])
        rank = np.concatenate([basis.radial_rank, np.full(dup.shape[1], 2)])
        return sb.SpatialBasis(
            linear=basis.linear, radial=radial, radial_rank=rank, n=basis.n
        )

    def test_the_fast_criterion_equals_lstsq_on_a_deficient_design(self):
        basis = self._deficient()
        pre = sb.precompute_df_selection(basis)
        self.assertLess(int(pre.spans[-1]), int(pre.k_of_df[-1]))
        y = np.random.default_rng(5).normal(size=basis.n)
        y += basis.radial[:, 0]
        for criterion in ("aic", "bic"):
            slow = sb.select_df_aic(y, basis, criterion=criterion, return_profile=True)
            fast = sb.select_df_aic_fast(
                y, basis, pre, criterion=criterion, return_profile=True
            )
            self.assertEqual(set(slow[2]), set(fast[2]))
            for key, ic in slow[2].items():
                self.assertAlmostEqual(fast[2][key], ic, delta=1e-8, msg=key)
            self.assertEqual(slow[0], fast[0])

    def test_a_full_rank_design_keeps_its_qr(self):
        M = np.random.default_rng(6).normal(size=(200, 12))
        Q, kept = sb.nested_basis(M)
        np.testing.assert_array_equal(Q, np.linalg.qr(M, mode="reduced")[0])
        np.testing.assert_array_equal(kept, np.arange(1, 13))

    def test_a_dependent_column_adds_no_direction(self):
        M = np.random.default_rng(7).normal(size=(100, 4))
        M = np.column_stack([M[:, :2], M[:, 0] + 2 * M[:, 1], M[:, 2:]])
        Q, kept = sb.nested_basis(M)
        np.testing.assert_array_equal(kept, [1, 2, 2, 3, 4])
        np.testing.assert_allclose(Q.T @ Q, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(Q @ (Q.T @ M) - M, 0.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
