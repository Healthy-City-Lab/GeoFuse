"""Tuning within areas: the distance-defined smooth joins the search.

With ``spatial_tune_scale_m`` the smooth at that scale, built on the rows a
search uses, is appended to the covariates of every search stage, so the
layer, radius, statistic and weights are chosen on variation within areas.
Unset, the search sees the covariates alone. A confounder smooth at the scale
drives one channel and the outcome; a second channel carries a within-area
effect: an unadjusted search picks the first, a tuned search the second.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas  # noqa: F401  (pyproj before torchvision, see vision.py)
import numpy as np

import test_fit_bayesian_index as fixture
from geofuse import bayesian_index, cgi_formulas, parallel
from geofuse import spatial_basis as sb
from geofuse.fusion import MetricFusionEngine

SCALE = 5_000.0
SIDE = 40_000.0


def _field(xy):
    """A smooth surface of wavelength 20 km, four times the scale."""
    k = 2 * np.pi / 20_000.0
    return np.sin(k * xy[:, 0]) + np.cos(k * xy[:, 1])


def _confounded(seed, n=1_500, effect=0.35):
    """Channel ndvi follows the smooth confounder, which drives the outcome;
    channel gvi carries a within-area effect of ``effect``."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, SIDE, size=(n, 2))
    u = _field(xy)
    b = rng.normal(size=n)
    shape = (n, 2, len(fixture.RADII), len(fixture.STATS))
    X = np.empty(shape)
    X[:, 0] = (u + 0.3 * rng.normal(size=n))[:, None, None]
    X[:, 1] = b[:, None, None]
    X += 0.1 * rng.normal(size=shape)
    cov = rng.normal(size=(n, 1))
    y = u + effect * b + 0.5 * cov[:, 0] + rng.normal(size=n)
    return xy, X, y, cov


def _engine(xy, X, y, cov, scale=SCALE):
    eng = fixture._stub_engine()
    static = {"target": y, "cov": cov, "coords": xy}
    eng.build_index_tensor = lambda subset="train_val": (
        X,
        np.asarray(fixture.RADII),
        list(fixture.STATS),
        ["ndvi", "gvi"],
        static,
    )
    eng._synergy_design = fixture._entity_cells(X, static)
    eng.spatial_tune_scale_m = scale
    eng.spatial_adjust_eps_m = None
    eng.spatial_adjust_neighbours = sb.DEFAULT_NEIGHBOURS
    eng.spatial_adjust_min_component = sb.DEFAULT_MIN_COMPONENT
    eng._spatial_basis_cache = {}
    eng._tune_eps = None if scale is None else 2.0 * scale
    for name in ("_tuning_record", "_warn_tuning_size"):
        setattr(eng, name, getattr(MetricFusionEngine, name).__get__(eng))
    return eng, static


FIT = dict(
    sweep_splits=3,
    reps=0,
    shuffles=0,
    gain_splits=0,
    null_runs=0,
    forms=("linear",),
    draws=150,
    warmup=150,
    chains=1,
)


def _ols_residual(a, design):
    D = np.column_stack([np.ones(len(a)), design])
    coef, *_ = np.linalg.lstsq(D, a, rcond=None)
    return a - D @ coef


def _z(a):
    return (a - a.mean(0)) / a.std(0)


class SinglePool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(
            setattr, parallel, "process_worker_count", parallel.process_worker_count
        )
        parallel.process_worker_count = lambda *a, **k: 1


class SearchCovariateTests(SinglePool):
    def test_unset_returns_the_covariates_themselves(self):
        xy, X, y, cov = _confounded(0, n=300)
        eng, static = _engine(xy, X, y, cov, scale=None)
        out, record = MetricFusionEngine._search_covariates(eng, static)
        self.assertIs(out, cov)
        self.assertIsNone(record)
        params = MetricFusionEngine.fit_bayesian_index(eng, "r2", **FIT)
        self.assertNotIn("__spatial_tuning__", params)

    def test_prep_equals_ols_residuals_on_covariates_and_smooth(self):
        xy, X, y, cov = _confounded(1, n=600)
        eng, static = _engine(xy, X, y, cov)
        aug, record = eng._search_covariates(static)
        smooth = aug[:, cov.shape[1] :]
        basis = sb.build_block_basis(
            xy,
            eps=2 * SCALE,
            scale_m=SCALE,
            min_component=sb.DEFAULT_MIN_COMPONENT,
        )
        np.testing.assert_array_equal(smooth, basis.design(basis.max_df))
        self.assertEqual(record["columns"], smooth.shape[1])
        yr, Xr = bayesian_index.prep(X, y, aug)
        design = np.column_stack([cov, smooth])
        np.testing.assert_allclose(yr, _z(_ols_residual(y, design)), atol=1e-10)
        flat = X.reshape(len(X), -1)
        want = _z(np.column_stack([_ols_residual(c, design) for c in flat.T]))
        np.testing.assert_allclose(Xr.reshape(len(X), -1), want, atol=1e-10)

    def test_one_smooth_per_row_set(self):
        xy, X, y, cov = _confounded(2, n=400)
        eng, static = _engine(xy, X, y, cov)
        with mock.patch.object(
            sb, "build_block_basis", wraps=sb.build_block_basis
        ) as build:
            eng._search_covariates(static)
            eng._search_covariates(static)
            self.assertEqual(build.call_count, 1)
            other = dict(static, coords=xy[::-1].copy(), cov=cov[::-1].copy())
            eng._search_covariates(other)
            self.assertEqual(build.call_count, 2)

    def test_synergy_terms_are_residualised_on_the_smooth(self):
        xy, X, y, cov = _confounded(3, n=500)
        X = np.abs(X)
        eng, static = _engine(xy, X, y, cov)
        aug, _ = eng._search_covariates(static)
        seen = {}
        design_of = eng._synergy_design

        def spy(static_, kept, channel_index, *rest, cov=None):
            seen["cov"] = cov
            return design_of(static_, kept, channel_index, *rest, cov=cov)

        eng._synergy_design = spy
        MetricFusionEngine.fit_bayesian_index(
            eng, "r2", **dict(FIT, forms=("linear", "synergy"))
        )
        np.testing.assert_array_equal(seen["cov"], aug)
        design = bayesian_index.SynergyDesign.build(X, aug, channels=[0, 1])
        cols = [0, len(fixture.RADII) * len(fixture.STATS)]
        got, _ = design.columns(cols)
        flat = X.reshape(len(X), -1)
        raw = np.column_stack(cgi_formulas.synergy_terms([flat[:, c] for c in cols]))
        want = _z(np.column_stack([_ols_residual(t, aug) for t in raw.T]))
        np.testing.assert_allclose(got, want, atol=2e-5)

    def test_the_guard_warns_on_a_crowded_smooth(self):
        xy, X, y, cov = _confounded(4, n=300)
        eng, static = _engine(xy, X, y, cov, scale=1_000.0)
        with mock.patch("geofuse.fusion._log") as log:
            eng._search_covariates(static)
        warns = [c.args[1] for c in log.call_args_list if c.args[0] == "WARN"]
        self.assertTrue(any("1 km" in w and "fit noise" in w for w in warns))


class RecoveryTests(SinglePool):
    def _largest(self, params):
        post = params["__posterior__"]
        weights = dict(zip(post["weight_labels"], post["weight_mean"], strict=True))
        return max(weights, key=weights.get)

    def test_the_tuned_search_picks_the_within_area_channel(self):
        untuned, tuned = [], []
        for seed in range(5):
            xy, X, y, cov = _confounded(10 + seed)
            eng, _ = _engine(xy, X, y, cov, scale=None)
            untuned.append(
                self._largest(MetricFusionEngine.fit_bayesian_index(eng, "r2", **FIT))
            )
            eng, _ = _engine(xy, X, y, cov)
            params = MetricFusionEngine.fit_bayesian_index(eng, "r2", **FIT)
            tuned.append(self._largest(params))
            record = params["__spatial_tuning__"]
            self.assertEqual(record["scale_m"], SCALE)
            self.assertGreater(record["absorbed"]["ndvi"], 0.5)
            self.assertLess(record["absorbed"]["gvi"], 0.2)
        self.assertGreaterEqual(sum(w.startswith("ndvi") for w in untuned), 4)
        self.assertGreaterEqual(sum(w.startswith("gvi") for w in tuned), 4)

        # The within-area slope of gvi on fresh rows, under the same smooth.
        xy, X, y, cov = _confounded(99, n=3_000)
        basis = sb.build_block_basis(xy, eps=2 * SCALE, scale_m=SCALE)
        design = np.column_stack([cov, basis.design(basis.max_df)])
        b = _ols_residual(X[:, 1, 0, 0], design)
        slope = float(b @ _ols_residual(y, design) / (b @ b))
        self.assertGreater(slope, 0.0)

    def test_null_calibration_holds_with_the_smooth(self):
        xy, X, y, cov = _confounded(20, n=800, effect=0.0)
        eng, _ = _engine(xy, X, y, cov)
        params = MetricFusionEngine.fit_bayesian_index(
            eng, "r2", **dict(FIT, null_runs=20)
        )
        null = params["__null_calibration__"]
        self.assertEqual(null["runs"], 20)
        self.assertLessEqual(null["rate_ci_low"], 0.10)


class EngineSettingTests(unittest.TestCase):
    def _make(self, **kw):
        return MetricFusionEngine(
            target_file="dummy.geojson", cache_dir=tempfile.mkdtemp(), **kw
        )

    def test_the_reported_scale_defaults_to_the_tuning_scale(self):
        eng = self._make(spatial_adjust_method="spatial_plus", spatial_tune_scale_m=5000)
        self.assertEqual(eng.spatial_adjust_scales, (5000.0,))
        self.assertEqual(eng._tune_eps, 10_000.0)

    def test_a_different_primary_scale_warns(self):
        with mock.patch("geofuse.fusion._log") as log:
            self._make(
                spatial_adjust_method="spatial_plus",
                spatial_tune_scale_m=5000,
                spatial_adjust_scale_m=[20_000, 5000],
            )
        warns = [c.args[1] for c in log.call_args_list if c.args[0] == "WARN"]
        self.assertTrue(any("Tuned within areas of 5 km" in w for w in warns))

    def test_it_needs_a_spatial_method(self):
        with self.assertRaises(ValueError):
            self._make(spatial_tune_scale_m=5000)
        with self.assertRaises(ValueError):
            self._make(spatial_adjust_method="spatial_plus", spatial_tune_scale_m=-1)

    def test_unset_changes_nothing(self):
        eng = self._make(spatial_adjust_method="spatial_plus")
        self.assertIsNone(eng.spatial_tune_scale_m)
        self.assertEqual(eng.spatial_adjust_scales, ())


if __name__ == "__main__":
    unittest.main()
