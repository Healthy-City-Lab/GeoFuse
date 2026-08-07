"""Trials run in parallel for every metric, so scoring must be thread-safe.

Two engine-owned scoring caches used to make that unsafe: the partial-distance
side cache and the spline-basis cache both evicted by walking their own
iterator, which cannot survive a concurrent insert. The tests here hammer both
from many threads and require the concurrent answers to equal the serial ones
exactly — a race in this layer would not raise, it would quietly return another
subset's basis and corrupt a score.
"""

import os
import sys
import unittest
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from geofuse import mixed_effects_scoring, objective_scoring, pdcor  # noqa: E402
from geofuse.fusion import MetricFusionEngine  # noqa: E402


def _subsets(n_subsets, n_rows, n_cov, seed=0):
    """Distinct scored subsets, as consecutive resamples produce."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_subsets):
        cov = rng.normal(size=(n_rows, n_cov))
        greenery = rng.normal(size=n_rows)
        target = 0.7 * greenery + cov @ rng.normal(size=n_cov) + rng.normal(size=n_rows)
        out.append((target, greenery, cov))
    return out


class TestSplineBasisCacheUnderThreads(unittest.TestCase):
    def test_concurrent_expansion_matches_serial(self):
        # More distinct subsets than the cache holds, so eviction runs
        # constantly while other threads insert.
        subsets = _subsets(objective_scoring._SPLINE_CACHE_MAX * 4, 240, 3, seed=1)
        mats = [cov for _, _, cov in subsets]

        serial_cache: dict = {}
        expected = [
            objective_scoring._expand_covariate_basis(m, "spline", serial_cache)
            for m in mats
        ]

        shared: dict = {}
        errors: list = []

        def work(i):
            try:
                # Each matrix expanded repeatedly, so hits and misses interleave.
                for _ in range(6):
                    got = objective_scoring._expand_covariate_basis(
                        mats[i], "spline", shared
                    )
                    np.testing.assert_array_equal(expected[i], got)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(work, list(range(len(mats))) * 4))
        self.assertEqual(errors, [])
        self.assertLessEqual(len(shared), objective_scoring._SPLINE_CACHE_MAX)

    def test_cache_is_still_bounded(self):
        mats = [cov for _, _, cov in _subsets(40, 120, 2, seed=2)]
        shared: dict = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(
                ex.map(
                    lambda m: objective_scoring._expand_covariate_basis(
                        m, "spline", shared
                    ),
                    mats,
                )
            )
        self.assertLessEqual(len(shared), objective_scoring._SPLINE_CACHE_MAX)


class TestPdcorSideCacheUnderThreads(unittest.TestCase):
    def test_concurrent_sides_match_serial(self):
        subsets = _subsets(pdcor._MAX_SIDE_ENTRIES * 3, 160, 2, seed=3)
        serial_cache: OrderedDict = OrderedDict()
        expected = [pdcor._get_side(serial_cache, t, c).cc for t, _, c in subsets]

        shared: OrderedDict = OrderedDict()
        errors: list = []

        def work(i):
            try:
                t, _, c = subsets[i]
                for _ in range(4):
                    self.assertAlmostEqual(
                        expected[i], pdcor._get_side(shared, t, c).cc, places=10
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(work, list(range(len(subsets))) * 3))
        self.assertEqual(errors, [])
        self.assertLessEqual(len(shared), pdcor._MAX_SIDE_ENTRIES)


class TestScoreIsConcurrencyInvariant(unittest.TestCase):
    """End to end: the same inputs must score the same, serial or threaded."""

    def _run(self, metric, workers, subsets, caches):
        def score_one(args):
            target, greenery, cov = args
            return objective_scoring.score(
                metric,
                target,
                greenery,
                covariates=cov,
                residualize_method="spline",
                pdcor_cache=caches[0],
                spline_cache=caches[1],
            )

        if workers == 1:
            return [score_one(s) for s in subsets]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(score_one, subsets))

    def test_metrics_agree_between_serial_and_threaded(self):
        subsets = _subsets(24, 200, 3, seed=4)
        for metric in ("r2", "spearman", "nrmse", "partial_distance_corr"):
            with self.subTest(metric=metric):
                serial = self._run(metric, 1, subsets, (OrderedDict(), {}))
                threaded = self._run(metric, 8, subsets, (OrderedDict(), {}))
                np.testing.assert_allclose(
                    np.asarray(serial, dtype=float),
                    np.asarray(threaded, dtype=float),
                    rtol=0,
                    atol=0,
                    err_msg=metric,
                )


class TestSearchParallelismGate(unittest.TestCase):
    """Every metric parallelises except the one where threading measured slower."""

    def _engine(self, longitudinal, metric_workers=8):
        engine = MetricFusionEngine.__new__(MetricFusionEngine)
        # ``is_longitudinal`` is derived from the spec, so set the spec.
        engine.longitudinal_spec = object() if longitudinal else None
        engine._search_workers = metric_workers
        engine.search_scoring_method = "mom_em3"
        engine.spatial_adjust_method = "none"
        return engine

    def test_cross_sectional_metrics_all_parallelise(self):
        engine = self._engine(False)
        for metric in ("r2", "partial_distance_corr", "spearman", "nrmse"):
            self.assertEqual(engine._search_n_jobs(metric), 8, metric)

    def test_longitudinal_non_mixedlm_metrics_parallelise(self):
        engine = self._engine(True)
        for metric in ("r2", "partial_distance_corr", "spearman", "nrmse"):
            self.assertEqual(engine._search_n_jobs(metric), 8, metric)

    def test_fast_mixedlm_metrics_parallelise(self):
        engine = self._engine(True)
        for metric in sorted(mixed_effects_scoring.FAST_SEARCH_METRICS):
            self.assertEqual(engine._search_n_jobs(metric), 8, metric)

    def test_exact_mixedlm_refits_stay_sequential(self):
        engine = self._engine(True)
        exact = set(mixed_effects_scoring.MIXEDLM_METRICS) - set(
            mixed_effects_scoring.FAST_SEARCH_METRICS
        )
        self.assertTrue(exact)
        for metric in sorted(exact):
            self.assertEqual(engine._search_n_jobs(metric), 1, metric)

    def test_a_spatial_adjustment_no_longer_forces_sequential(self):
        engine = self._engine(True)
        engine.spatial_adjust_method = "ks_aic"
        self.assertEqual(engine._search_n_jobs("r2"), 8)


if __name__ == "__main__":
    unittest.main()
