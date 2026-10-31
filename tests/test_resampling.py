"""Exact fast paths of a fusion study's resampling.

Every rewrite is checked against the computation it replaces, on the same
inputs and the same resamples:

- the block engine (:mod:`geofuse.resampling`) against per-replicate ``lstsq``
  over ``[1, covariates, smooth]``, including replicates that leave a spatial
  cluster out and tiny or rank-deficient clusters;
- the bootstrap, BCa jackknife and Freedman–Lane permutation fast paths of
  :mod:`geofuse.statistical_testing` against their per-replicate loops.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas  # noqa: F401  (pyproj before torchvision, see vision.py)
import numpy as np

from geofuse import objective_scoring as osc
from geofuse import resampling as rs
from geofuse import spatial_basis as sb
from geofuse import statistical_testing as st


def _study(n=900, seed=0, tiny=(5, 6)):
    """Six clusters plus tiny ones; a smooth field shared by every variable."""
    rng = np.random.default_rng(seed)
    sizes = list(rng.multinomial(n - sum(tiny), np.ones(6) / 6)) + list(tiny)
    xy = np.vstack(
        [rng.random(2) * 1e6 + rng.normal(size=(m, 2)) * 3_000 for m in sizes]
    )
    nn = len(xy)
    u = np.sin(2 * np.pi * xy[:, 0] / 40_000) + np.cos(2 * np.pi * xy[:, 1] / 50_000)
    cov = np.column_stack(
        [rng.normal(size=nn), np.eye(3)[rng.integers(0, 3, nn)][:, 1:]]
    )
    g = rng.normal(size=nn)
    x = 0.7 * u + g
    x2 = 0.7 * u + 0.6 * g + 0.5 * rng.normal(size=nn)
    y = 0.15 * g + 0.5 * u + 0.3 * cov[:, 0] + rng.normal(size=nn)
    c = 0.4 * u + rng.normal(size=nn)
    basis = sb.build_block_basis(xy, max_df=6)
    return dict(xy=xy, cov=cov, x=x, x2=x2, y=y, c=c, S=basis.design(6), n=nn)


def _r2_ref(y, x, cov, S):
    return osc.score("r2", y, x, cov, spatial_basis=S, spatial_method="spatial_plus")


class BlockEngineTests(unittest.TestCase):
    def setUp(self):
        self.d = _study()

    def test_blocks_are_the_clusters(self):
        blocks = rs.smooth_blocks(self.d["S"])
        cols = np.concatenate([c for _, c in blocks])
        self.assertEqual(sorted(cols.tolist()), list(range(self.d["S"].shape[1])))
        rows = [set(r.tolist()) for r, _ in blocks]
        for i, a in enumerate(rows):
            for b in rows[i + 1 :]:
                self.assertFalse(a & b)

    def test_replicates_equal_lstsq_fits(self):
        d = self.d
        rng = np.random.default_rng(1)
        idx = rng.integers(0, d["n"], size=(12, d["n"]))
        idx[0] = rng.integers(0, d["n"] - 11, size=d["n"])  # no tiny clusters
        idx[1] = rng.integers(0, 150, size=d["n"])  # mostly one cluster
        design = rs.BlockDesign(d["n"], d["cov"], d["S"])
        W = rs.row_weights(idx, d["n"])
        M = design.cross(np.column_stack([d["y"], d["x"]]), W)
        got = rs.incremental_r2(
            M, rs.weighted_sst(d["y"], W), rs.weighted_sst(d["x"], W)
        )
        want = [_r2_ref(d["y"][i], d["x"][i], d["cov"][i], d["S"][i]) for i in idx]
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-10)

    def test_rank_deficient_covariates(self):
        d = self.d
        cov = np.column_stack([d["cov"], d["cov"][:, 0] * 2.0, np.ones(d["n"])])
        design = rs.BlockDesign(d["n"], cov, d["S"])
        W = np.ones((1, d["n"]))
        M = design.cross(np.column_stack([d["y"], d["x"]]), W)
        got = rs.incremental_r2(
            M, rs.weighted_sst(d["y"], W), rs.weighted_sst(d["x"], W)
        )
        self.assertAlmostEqual(got[0], _r2_ref(d["y"], d["x"], cov, d["S"]), places=10)

    def test_weighted_residuals_equal_lstsq(self):
        d = self.d
        draw = np.random.default_rng(2).integers(0, d["n"], d["n"])
        w = rs.row_weights(draw, d["n"])[0]
        X = np.column_stack([np.ones(d["n"]), d["cov"], d["S"]])
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], d["y"] * sw, rcond=None)
        got = rs.BlockDesign(d["n"], d["cov"], d["S"]).residuals(d["y"], w)
        np.testing.assert_allclose(got[w > 0], (d["y"] - X @ beta)[w > 0], atol=1e-9)

    def test_the_score_takes_one_projection(self):
        d = self.d
        X = np.column_stack([d["cov"], d["S"]])
        full = np.column_stack([np.ones(d["n"]), X, d["x"]])
        red = np.column_stack([np.ones(d["n"]), X])

        def r2(A):
            b, *_ = np.linalg.lstsq(A, d["y"], rcond=None)
            r = d["y"] - A @ b
            return 1 - r @ r / ((d["y"] - d["y"].mean()) ** 2).sum()

        self.assertAlmostEqual(
            _r2_ref(d["y"], d["x"], d["cov"], d["S"]), r2(full) - r2(red), places=12
        )


class ResamplingLoopTests(unittest.TestCase):
    def setUp(self):
        d = _study(n=700)
        self.d = d
        self.ctrl = np.column_stack([d["cov"], d["S"]])
        self.k = d["cov"].shape[1]
        k = self.k

        def score_fn(t, p, c):
            return _r2_ref(t, p, c[:, :k], c[:, k:])

        self.score_fn = score_fn

    def _boot(self, **kw):
        d = self.d
        args = dict(
            score_fn=self.score_fn, n_bootstrap=60, seed=3, covariates=self.ctrl
        )
        args.update(kw)
        slow = st.bootstrap_score_ci(d["y"], d["x"], **args)
        fast = st.bootstrap_score_ci(
            d["y"], d["x"], **args, batch_scorer_factory=rs.r2_bootstrap_scorer(self.k)
        )
        for key in ("observed", "mean", "lower", "upper"):
            self.assertAlmostEqual(slow[key], fast[key], places=10, msg=key)

    def test_row_bootstrap(self):
        self._boot(method="percentile")

    def test_group_bootstrap(self):
        groups = np.random.default_rng(4).integers(0, 80, self.d["n"])
        self._boot(method="percentile", groups=groups)

    def test_bca_jackknife(self):
        self._boot(method="BCa")

    def test_freedman_lane_permutations(self):
        d = self.d
        args = dict(score_fn=self.score_fn, n_perm=60, seed=4, covariates=self.ctrl)
        slow = st.permutation_pvalue(d["y"], d["x"], **args)
        fast = st.permutation_pvalue(
            d["y"],
            d["x"],
            **args,
            permutation_scorer_factory=rs.r2_permutation_scorer(self.k),
        )
        self.assertEqual(slow["p_value"], fast["p_value"])
        self.assertAlmostEqual(slow["null_mean"], fast["null_mean"], places=12)

    def test_residual_metrics(self):
        d, k = self.d, self.k
        for method in ("spatial_plus", "ks_aic"):
            for metric in ("distance_corr", "nrmse"):

                def score_fn(t, p, c, metric=metric, method=method):
                    return osc.score(
                        metric, t, p, c[:, :k], spatial_basis=c[:, k:],
                        spatial_method=method,
                    )

                args = dict(
                    score_fn=score_fn, n_bootstrap=30, seed=3, covariates=self.ctrl
                )
                slow = st.bootstrap_score_ci(d["y"], d["x"], **args)
                factory = rs.residual_bootstrap_scorer(metric, k, method)
                fast = st.bootstrap_score_ci(
                    d["y"], d["x"], **args, batch_scorer_factory=factory
                )
                for key in ("observed", "mean", "lower", "upper"):
                    self.assertAlmostEqual(slow[key], fast[key], places=9)
                pargs = dict(score_fn=score_fn, n_perm=30, seed=4, covariates=self.ctrl)
                slow = st.permutation_pvalue(d["y"], d["x"], **pargs)
                fast = st.permutation_pvalue(
                    d["y"],
                    d["x"],
                    **pargs,
                    permutation_scorer_factory=rs.residual_permutation_scorer(
                        metric, k, method
                    ),
                )
                self.assertEqual(slow["p_value"], fast["p_value"])
                self.assertAlmostEqual(slow["null_mean"], fast["null_mean"], places=9)

    def test_no_smooth_keeps_the_direct_fits(self):
        factory = rs.r2_bootstrap_scorer(self.k)
        self.assertIsNone(factory(self.d["y"], self.d["x"], self.d["cov"]))


if __name__ == "__main__":
    unittest.main()
