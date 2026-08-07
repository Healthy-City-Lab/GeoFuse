"""Equivalence tests for the fusion engine's cached scoring fast paths.

Every fast path here is a pure acceleration of an existing computation, so
each test pins the fast result against the stock implementation it replaces:
the cached partial distance correlation vs ``dcor``, the bootstrap /
permutation replicate scorers vs the generic ``score_fn`` loops, the
codes-based entity collapse vs the factorizing collapse, the nested-QR
spatial-df selection vs the least-squares search, and the single-pass ring
binning vs per-ring masking.
"""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from geofuse import objective_scoring, pdcor, spatial_basis, statistical_testing


def _stock_pdcor(x, y, z):
    import dcor

    return float(
        dcor.partial_distance_correlation(
            np.asarray(x, dtype=np.float64).reshape(len(x), -1),
            np.asarray(y, dtype=np.float64).reshape(len(y), -1),
            np.asarray(z, dtype=np.float64).reshape(len(x), -1),
        )
    )


class TestCachedPdcor(unittest.TestCase):
    def test_matches_dcor_across_sizes_and_z_widths(self):
        rng = np.random.default_rng(7)
        from collections import OrderedDict

        for n in (5, 50, 500):
            for z_cols in (1, 6):
                x = rng.normal(size=n)
                y = 0.4 * x + rng.normal(size=n)
                z = rng.normal(size=(n, z_cols))
                cache = OrderedDict()
                expected = _stock_pdcor(x, y, z)
                got = pdcor.partial_distance_correlation_cached(
                    x.reshape(-1, 1), y.reshape(-1, 1), z, cache
                )
                self.assertLess(abs(got - expected), 1e-5, f"n={n} z_cols={z_cols}")
                # Second call hits the cached side and must agree with itself.
                again = pdcor.partial_distance_correlation_cached(
                    x.reshape(-1, 1), y.reshape(-1, 1), z, cache
                )
                self.assertEqual(got, again)
                self.assertEqual(len(cache), 1)

    def test_side_cache_reused_across_varying_y(self):
        from collections import OrderedDict

        rng = np.random.default_rng(3)
        n = 200
        x = rng.normal(size=n)
        z = rng.normal(size=(n, 2))
        cache = OrderedDict()
        for _trial in range(5):
            y = rng.normal(size=n)
            got = pdcor.partial_distance_correlation_cached(
                x.reshape(-1, 1), y.reshape(-1, 1), z, cache
            )
            self.assertLess(abs(got - _stock_pdcor(x, y, z)), 1e-5)
        self.assertEqual(len(cache), 1)  # one fixed side across all trials

    def test_constant_input_returns_zero(self):
        from collections import OrderedDict

        rng = np.random.default_rng(0)
        n = 40
        y = np.ones(n)
        x = rng.normal(size=n)
        z = rng.normal(size=(n, 1))
        got = pdcor.partial_distance_correlation_cached(
            x.reshape(-1, 1), y.reshape(-1, 1), z, OrderedDict()
        )
        self.assertEqual(got, 0.0)

    def test_score_with_cache_matches_without(self):
        from collections import OrderedDict

        rng = np.random.default_rng(11)
        n = 300
        t = rng.normal(size=n)
        c = 0.5 * t + rng.normal(size=n)
        cov = rng.normal(size=(n, 3))
        base = objective_scoring.score("partial_distance_corr", t, c, cov)
        fast = objective_scoring.score(
            "partial_distance_corr", t, c, cov, pdcor_cache=OrderedDict()
        )
        self.assertLess(abs(float(base) - float(fast)), 1e-5)


class TestReplicateScorers(unittest.TestCase):
    def test_bootstrap_replicate_scorer_matches_direct(self):
        rng = np.random.default_rng(5)
        n = 150
        t = rng.normal(size=n)
        p = 0.6 * t + rng.normal(size=n)
        z = rng.normal(size=(n, 2))
        scorer = pdcor.pdcor_replicate_scorer_factory(t, p, z)
        self.assertIsNotNone(scorer)
        for _ in range(5):
            idx = rng.integers(0, n, size=n)
            got = scorer(idx)
            expected = _stock_pdcor(t[idx], p[idx], z[idx])
            self.assertLess(abs(got - expected), 1e-4)

    def test_surrogate_scorer_matches_direct(self):
        rng = np.random.default_rng(9)
        n = 120
        p = rng.normal(size=n)
        z = rng.normal(size=(n, 3))
        scorer = pdcor.pdcor_surrogate_scorer_factory(p, z)
        self.assertIsNotNone(scorer)
        for _ in range(5):
            t_star = rng.normal(size=n)
            got = scorer(t_star)
            expected = _stock_pdcor(t_star, p, z)
            self.assertLess(abs(got - expected), 1e-4)

    def test_paired_scorers_share_resample(self):
        rng = np.random.default_rng(13)
        n = 100
        t = rng.normal(size=n)
        p1 = 0.5 * t + rng.normal(size=n)
        p2 = rng.normal(size=n)
        z = rng.normal(size=(n, 2))
        pair = pdcor.pdcor_paired_replicate_scorers(t, p1, p2, z)
        self.assertIsNotNone(pair)
        idx = rng.integers(0, n, size=n)
        self.assertLess(abs(pair[0](idx) - _stock_pdcor(t[idx], p1[idx], z[idx])), 1e-4)
        self.assertLess(abs(pair[1](idx) - _stock_pdcor(t[idx], p2[idx], z[idx])), 1e-4)

    def test_factory_declines_without_conditioning(self):
        rng = np.random.default_rng(1)
        t = rng.normal(size=50)
        self.assertIsNone(pdcor.pdcor_replicate_scorer_factory(t, t, None))
        self.assertIsNone(pdcor.pdcor_surrogate_scorer_factory(t, None))


class TestStatisticalFastPaths(unittest.TestCase):
    def _score_fn(self, t, p, cov=None):
        return objective_scoring.score("partial_distance_corr", t, p, cov)

    def test_bootstrap_ci_fast_vs_generic(self):
        rng = np.random.default_rng(21)
        n = 80
        t = rng.normal(size=n)
        p = 0.7 * t + rng.normal(size=n)
        cov = rng.normal(size=(n, 2))
        kwargs = dict(
            score_fn=self._score_fn,
            n_bootstrap=200,
            ci_level=0.95,
            method="percentile",
            seed=42,
            covariates=cov,
        )
        generic = statistical_testing.bootstrap_score_ci(t, p, **kwargs)
        fast = statistical_testing.bootstrap_score_ci(
            t,
            p,
            replicate_scorer_factory=pdcor.pdcor_replicate_scorer_factory,
            **kwargs,
        )
        self.assertAlmostEqual(generic["observed"], fast["observed"], places=10)
        self.assertAlmostEqual(generic["lower"], fast["lower"], delta=1e-3)
        self.assertAlmostEqual(generic["upper"], fast["upper"], delta=1e-3)
        self.assertAlmostEqual(generic["mean"], fast["mean"], delta=1e-3)

    def test_permutation_fast_vs_generic(self):
        rng = np.random.default_rng(23)
        n = 70
        t = rng.normal(size=n)
        p = 0.8 * t + rng.normal(size=n)
        cov = rng.normal(size=(n, 2))
        kwargs = dict(
            score_fn=self._score_fn,
            higher_is_better=True,
            n_perm=200,
            seed=42,
            covariates=cov,
        )
        generic = statistical_testing.permutation_pvalue(t, p, **kwargs)
        fast = statistical_testing.permutation_pvalue(
            t,
            p,
            surrogate_scorer_factory=pdcor.pdcor_surrogate_scorer_factory,
            **kwargs,
        )
        self.assertAlmostEqual(generic["observed"], fast["observed"], places=10)
        self.assertAlmostEqual(generic["p_value"], fast["p_value"], delta=0.02)
        self.assertAlmostEqual(generic["null_mean"], fast["null_mean"], delta=1e-3)


class TestCollapseStatics(unittest.TestCase):
    def _synthetic(self, rng, n_entities=25, pixels_per=8, string_keys=False):
        pid = np.repeat(np.arange(n_entities), pixels_per)
        rng.shuffle(pid)
        if string_keys:
            pid = np.array([f"e{p:03d}|w1" for p in pid])
        values = rng.normal(size=len(pid))
        return pid, values

    def test_mean_collapse_matches_legacy(self):
        from geofuse.fusion import MetricFusionEngine

        rng = np.random.default_rng(31)
        for string_keys in (False, True):
            pid, values = self._synthetic(rng, string_keys=string_keys)
            codes, uniq = pd.factorize(pid, sort=True)
            legacy = MetricFusionEngine._collapse_to_entities(values, pid, None, "mean")
            fast = MetricFusionEngine._collapse_mean_from_codes(
                values, codes.astype(np.int64), len(uniq), None
            )
            np.testing.assert_allclose(fast, legacy, rtol=1e-12)

    def test_mean_collapse_with_mask_matches_legacy(self):
        from geofuse.fusion import MetricFusionEngine

        rng = np.random.default_rng(37)
        pid, values = self._synthetic(rng)
        codes, uniq = pd.factorize(pid, sort=True)
        # Mask that keeps at least one row per entity (the nearest-pixel
        # guarantee the engine maintains for point/line targets).
        mask = rng.random(len(pid)) < 0.5
        first_of = pd.Series(np.arange(len(pid))).groupby(pid).first().to_numpy()
        mask[first_of] = True
        legacy = MetricFusionEngine._collapse_to_entities(values, pid, mask, "mean")
        fast = MetricFusionEngine._collapse_mean_from_codes(
            values, codes.astype(np.int64), len(uniq), mask
        )
        np.testing.assert_allclose(fast, legacy, rtol=1e-12)

    def test_first_take_matches_groupby_first(self):
        rng = np.random.default_rng(41)
        pid, _ = self._synthetic(rng, string_keys=True)
        # Per-entity constants, like target / covariates on pixel rows.
        entity_value = {p: rng.normal() for p in np.unique(pid)}
        values = np.array([entity_value[p] for p in pid])
        codes, uniq = pd.factorize(pid, sort=True)
        codes = codes.astype(np.int64)
        order = np.argsort(codes, kind="stable")
        first_idx = order[np.searchsorted(codes[order], np.arange(len(uniq)))]
        legacy = pd.Series(values).groupby(pid).first().to_numpy()
        np.testing.assert_allclose(values[first_idx], legacy, rtol=0)


class TestSpatialDfFastPath(unittest.TestCase):
    def test_fast_selection_matches_legacy(self):
        rng = np.random.default_rng(51)
        n = 160
        xy = rng.uniform(0, 1000, size=(n, 2))
        basis = spatial_basis.build_block_basis(xy, max_df=6)
        self.assertTrue(basis.has_spatial)
        pre = spatial_basis.precompute_df_selection(basis)
        for i in range(5):
            y = rng.normal(size=n) + 0.002 * xy[:, 0] * (i % 2)
            df_legacy, cols_legacy = spatial_basis.select_df_aic(y, basis, None)
            df_fast, cols_fast = spatial_basis.select_df_aic_fast(y, basis, pre)
            self.assertEqual(df_legacy, df_fast)
            if cols_legacy is None:
                self.assertIsNone(cols_fast)
            else:
                # Same column span (order may differ): identical projections.
                res_legacy = self._resid(y, cols_legacy)
                res_fast = self._resid(y, cols_fast)
                np.testing.assert_allclose(res_fast, res_legacy, atol=1e-8)

    @staticmethod
    def _resid(y, X):
        Xc = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        return y - Xc @ beta

    def test_fast_path_falls_back_on_nonfinite(self):
        rng = np.random.default_rng(53)
        n = 60
        xy = rng.uniform(0, 100, size=(n, 2))
        basis = spatial_basis.build_block_basis(xy, max_df=4)
        pre = spatial_basis.precompute_df_selection(basis)
        y = rng.normal(size=n)
        y[3] = np.nan
        df_fast, _ = spatial_basis.select_df_aic_fast(y, basis, pre)
        df_legacy, _ = spatial_basis.select_df_aic(y, basis, None)
        self.assertEqual(df_fast, df_legacy)


class TestRingBinning(unittest.TestCase):
    def test_raster_ring_values_match_buffered_disc_masks(self):
        """Annuli partition the same buffered discs the cache reduces over."""
        import geopandas as gpd
        from rasterio.features import geometry_mask
        from rasterio.transform import from_origin
        from shapely.geometry import Point

        from geofuse import metric_sampling, preaggregation

        rng = np.random.default_rng(61)
        h = w = 60
        data = rng.random((h, w))
        transform = from_origin(0, h, 1.0, 1.0)  # 1 m pixels
        metric = {
            "data": data,
            "transform": transform,
            "crs": gpd.GeoSeries([Point(0, 0)], crs="EPSG:32611").crs,
        }
        pts = gpd.GeoDataFrame(
            geometry=[Point(30.5, 30.5), Point(10.2, 45.7)], crs="EPSG:32611"
        )
        radii = np.array([5, 10, 15], dtype=np.int64)
        rows = metric_sampling.precompute_raster_ring_values(metric, pts, radii)

        for pos, pt in enumerate(pts.geometry):
            for k, outer in enumerate(radii):
                # Reference disc: the entity buffered at this radius, masked
                # with the raster semantic used throughout (all_touched).
                disc = preaggregation.buffer_at(pt, float(outer))
                mask = geometry_mask(
                    [disc],
                    out_shape=(h, w),
                    transform=transform,
                    invert=True,
                    all_touched=True,
                )
                expected = np.sort(data[mask])
                # Rings 0..k concatenated must reproduce disc k exactly.
                got = np.sort(np.concatenate(rows[pos][: k + 1]))
                np.testing.assert_allclose(got, expected)

    def test_vector_point_ring_values_match_distance_reference(self):
        import geopandas as gpd
        from shapely.geometry import Point

        from geofuse import metric_sampling

        rng = np.random.default_rng(67)
        n_metric = 300
        # Coordinates near the zone-11 central meridian so the function's
        # WGS84-derived UTM zone matches the input CRS (no reprojection skew
        # between the implementation and this reference).
        xy = np.column_stack(
            [
                500_000 + rng.uniform(0, 200, size=n_metric),
                5_650_000 + rng.uniform(0, 200, size=n_metric),
            ]
        )
        vals = rng.random(n_metric)
        metric = gpd.GeoDataFrame(
            {"gvi": vals},
            geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]),
            crs="EPSG:32611",
        )
        pts = gpd.GeoDataFrame(
            geometry=[Point(500_100, 5_650_100), Point(500_050, 5_650_150)],
            crs="EPSG:32611",
        )
        radii = np.array([20, 40, 60], dtype=np.int64)
        rows = metric_sampling.precompute_vector_ring_values(metric, pts, radii, "gvi")
        for pos, pt in enumerate(pts.geometry):
            d = np.hypot(xy[:, 0] - pt.x, xy[:, 1] - pt.y)
            for k, outer in enumerate(radii):
                inner = 0.0 if k == 0 else float(radii[k - 1])
                sel = (d <= outer) if k == 0 else ((d <= outer) & (d > inner))
                np.testing.assert_allclose(
                    np.sort(rows[pos][k]), np.sort(vals[sel]), atol=1e-9
                )

    def test_prefix_mean_matches_per_point_reduce(self):
        from geofuse import metric_sampling

        rng = np.random.default_rng(71)
        radii = np.array([10, 20, 30], dtype=np.int64)
        rows = []
        for _ in range(15):
            rows.append(
                [
                    rng.random(rng.integers(0, 6)).astype(np.float64)
                    for _ in range(len(radii))
                ]
            )
        prefix = metric_sampling.ring_prefix_stats(rows)
        for r in radii:
            legacy = metric_sampling.aggregate_from_ring_cache(
                radii, rows, float(r), "mean", 50
            )
            fast = metric_sampling.aggregate_from_ring_cache(
                radii, rows, float(r), "mean", 50, prefix=prefix
            )
            np.testing.assert_allclose(fast, legacy, rtol=1e-12, equal_nan=True)


class TestSplineBasisCache(unittest.TestCase):
    def test_cached_expansion_matches_uncached(self):
        rng = np.random.default_rng(81)
        X = rng.normal(size=(120, 3))
        cache: dict = {}
        a = objective_scoring._expand_covariate_basis(X, "spline")
        b = objective_scoring._expand_covariate_basis(X, "spline", cache)
        c = objective_scoring._expand_covariate_basis(X, "spline", cache)
        if a is not None and b is not None:
            np.testing.assert_allclose(a, b)
            self.assertIs(b, c)  # second cached call returns the memo


if __name__ == "__main__":
    unittest.main()
