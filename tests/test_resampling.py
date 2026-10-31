"""Exact fast paths of a fusion study's resampling, sweep and worker set-up.

Every rewrite is checked against the computation it replaces, on the same
inputs and the same resamples:

- the block engine (:mod:`geofuse.resampling`) against per-replicate ``lstsq``
  over ``[1, covariates, smooth]``, including replicates that leave a spatial
  cluster out and tiny or rank-deficient clusters;
- the bootstrap, BCa jackknife and Freedman–Lane permutation fast paths of
  :mod:`geofuse.statistical_testing` against their per-replicate loops;
- the negative-control and specificity statistics with a smooth against the
  dense per-cluster Gram path they had;
- the batched sweep (:func:`bayesian_index.simplex_fit_batch`, the sweep split,
  the training-|t| pick) against :func:`bayesian_index.simplex_fit` in a loop;
- the BLAS thread counts of command-line, job and pool processes, and the
  memory sampler of a job's process tree.
"""

import os
import subprocess
import sys
import textwrap
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas  # noqa: F401  (pyproj before torchvision, see vision.py)
import numpy as np

from geofuse import bayesian_index as bi
from geofuse import negative_controls as nc
from geofuse import objective_scoring as osc
from geofuse import parallel
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


class NegativeControlTests(unittest.TestCase):
    def setUp(self):
        d = _study(n=600, seed=3)
        self.d = d
        self.sx = sb.select_df_aic(d["x"], sb.build_block_basis(d["xy"], max_df=6))[1]
        if self.sx is None:
            self.sx = d["S"]
        self.sx2 = d["S"][:, ::2]

    def _same(self, got, want):
        for key, value in want.items():
            if isinstance(value, float) and np.isfinite(value):
                self.assertAlmostEqual(got[key], value, places=10, msg=key)

    def test_paired_contrast_with_a_smooth(self):
        d = self.d
        clusters = np.random.default_rng(5).integers(0, 60, d["n"])
        for kw in ({}, {"clusters": clusters}):
            got = nc.paired_contrast(
                d["x"], d["y"], d["c"], d["cov"], extra={"exposure": self.sx},
                n_boot=80, seed=2, **kw,
            )
            want = nc.paired_contrast(
                d["x"], d["y"], d["c"], np.column_stack([d["cov"], self.sx]),
                n_boot=80, seed=2, **kw,
            )
            self._same(got, want)

    def test_specificity_contrast_with_smooths(self):
        d = self.d
        got = nc.specificity_contrast(
            d["x"], d["x2"], d["y"], d["c"], d["cov"],
            extra={"exposure": self.sx, "other": self.sx2}, n_boot=60, seed=4,
        )
        def beta(x, y, smooth):
            controls = np.column_stack([d["cov"], smooth])
            return nc.partial_slope(x, y, controls)["beta"]

        b = [
            beta(d["x"], d["y"], self.sx),
            beta(d["x"], d["c"], self.sx),
            beta(d["x2"], d["y"], self.sx2),
            beta(d["x2"], d["c"], self.sx2),
        ]
        for key, value in zip(
            ("beta_target", "beta_control", "beta_target_other", "beta_control_other"),
            b,
            strict=True,
        ):
            self.assertAlmostEqual(got[key], value, places=10, msg=key)


class SweepTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(6)
        self.n, self.p = 400, 18
        Z = rng.normal(size=(self.n, self.p)) + rng.normal(size=(self.n, 1))
        Z[:, 5] = Z[:, 4]  # a duplicated column: singular active sets
        self.Z = (Z - Z.mean(0)) / Z.std(0)
        self.y = 0.2 * Z[:, 2] - 0.1 * Z[:, 9] + rng.normal(size=self.n)
        self.combos = np.array(
            [(a, b, c) for a in range(6) for b in range(6, 12) for c in range(12, 18)]
            + [(4, 5, 12)]
        )

    def test_batch_equals_the_loop(self):
        g, c = self.Z.T @ self.Z, self.Z.T @ self.y
        found, w = bi.simplex_fit_batch(g, c, self.combos)
        for k, combo in enumerate(self.combos):
            fit = bi.simplex_fit(g[np.ix_(combo, combo)], c[combo])
            self.assertEqual(found[k], fit is not None)
            if fit is None:
                continue
            want = np.zeros(3)
            want[fit[0]] = fit[1]
            np.testing.assert_allclose(w[k], want, atol=1e-12)

    def test_sweep_split_equals_the_loop(self):
        import math
        import tempfile

        fd, path = tempfile.mkstemp()
        os.close(fd)
        try:
            mm = np.memmap(path, dtype=np.float64, mode="w+", shape=self.Z.shape)
            mm[:] = self.Z
            mm.flush()
            del mm
            task = (path, self.Z.shape, self.combos.tolist(), self.y, 7, 0.25, None)
            got = bi._sweep_split(task)
        finally:
            os.unlink(path)
        te = bi._holdout(self.n, 7, 0.25)
        tr, ztr, zte = ~te, self.Z[~te], self.Z[te]
        g_tr, c_tr = ztr.T @ ztr, ztr.T @ self.y[tr]
        g_te, c_te = zte.T @ zte, zte.T @ self.y[te]
        yn = float(np.linalg.norm(self.y[te]))
        for k, combo in enumerate(self.combos):
            fit = bi.simplex_fit(g_tr[np.ix_(combo, combo)], c_tr[combo])
            if fit is None:
                self.assertEqual(got[k], 0.0)
                continue
            j = combo[fit[0]]
            w = fit[1]
            den = math.sqrt(max(float(w @ g_te[np.ix_(j, j)] @ w), 1e-15))
            r = float(np.clip(w @ c_te[j] / den / yn, -0.999999, 0.999999))
            want = abs(r) * math.sqrt(te.sum() - 2) / math.sqrt(1 - r * r)
            self.assertAlmostEqual(got[k], want, places=12)

    def test_training_pick_equals_the_loop(self):
        best, best_v = None, -np.inf
        g, c = self.Z.T @ self.Z, self.Z.T @ self.y
        for combo in self.combos:
            fit = bi.simplex_fit(g[np.ix_(combo, combo)], c[combo])
            if fit is None:
                continue
            v = bi._tstat(self.Z[:, combo[fit[0]]] @ fit[1], self.y)
            if v > best_v:
                best_v, best = v, tuple(int(i) for i in combo)
        self.assertEqual(bi._best_combo(self.Z, self.y, self.combos), best)


class ProcessSetupTests(unittest.TestCase):
    def _env_after_import(self, argv0: str) -> str:
        code = textwrap.dedent(
            f"""
            import os, sys
            sys.argv = [{argv0!r}]
            os.environ.pop("OPENBLAS_NUM_THREADS", None)
            import geofuse
            print(os.environ.get("OPENBLAS_NUM_THREADS"))
            """
        )
        env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_NUM_THREADS"}
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()

    def test_command_line_processes_pin_blas(self):
        self.assertEqual(self._env_after_import("-m"), "1")
        self.assertEqual(self._env_after_import("geofuse.exe"), "1")
        self.assertEqual(self._env_after_import("streamlit"), "None")

    def test_a_job_gets_its_count_and_hands_none_to_its_pools(self):
        before = {v: os.environ.get(v) for v in parallel._PINNED_THREAD_VARS}
        with parallel.job_thread_env(4):
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "4")
            parallel.release_job_threads()
            self.assertNotIn("OPENBLAS_NUM_THREADS", os.environ)
            with parallel._pinned_child_env():
                self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
                self.assertEqual(os.environ["GDAL_CACHEMAX"], "128")
        after = {v: os.environ.get(v) for v in parallel._PINNED_THREAD_VARS}
        self.assertEqual(before, after)

    def test_the_job_thread_count_is_a_limit(self):
        from geofuse import userconfig

        self.assertEqual(userconfig.LIMIT_DEFAULTS["job_threads"], 4)
        self.assertGreaterEqual(parallel.job_threads(), 1)

    def test_the_memory_sampler_reads_the_tree(self):
        import time

        sampler = parallel.MemorySampler(interval_s=0.05).start()
        time.sleep(0.3)
        sampler.stop()
        self.assertGreater(sampler.peak(), 0)
        self.assertEqual(sampler.peak(start=time.time() + 60), 0)

    def test_stage_rows_carry_the_peak(self):
        from geofuse.jobs.fusion_outputs import stage_timing_rows
        from geofuse.jobs.stage_ledger import StageLedger

        led = StageLedger.from_steps([("a", "A"), ("b", "B")])
        led.set_status("a", "running")
        sampler = parallel.MemorySampler(interval_s=0.05).start()
        import time

        time.sleep(0.2)
        led.set_status("a", "done", progress=1.0)
        sampler.stop()
        rows = stage_timing_rows(led, sampler)
        self.assertEqual(rows[0]["key"], "a")
        self.assertGreater(rows[0]["peak_gb"], 0)


if __name__ == "__main__":
    unittest.main()
