"""The composite path rebuilds the model the discovery fitted.

The sweep and the posterior fit weights and forms on each channel divided by its
covariate-adjusted SD; the composite path builds maps and test scores from raw
channel values and the saved parameters (``cgi_formulas.compute_cgi``). With the
recorded ``<channel>_center`` / ``<channel>_scale`` it standardises each channel
first — directly for the weighted average, through ``Φ`` for synergy — so the
composite is the fitted index rather than a reweighted cousin of it.
"""

import os
import sys
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from scipy.stats import norm

from geofuse import bayesian_index as bi
from geofuse import cgi_formulas


def _raw(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0.1, 0.8, n), rng.gamma(2.0, 0.1, n)])


def _params(R, **extra):
    p = {
        "ndvi_center": float(R[:, 0].mean()),
        "ndvi_scale": float(R[:, 0].std()),
        "gvi_center": float(R[:, 1].mean()),
        "gvi_scale": float(R[:, 1].std()),
    }
    p.update(extra)
    return p


class TestWeightedAverage(unittest.TestCase):
    def test_weights_apply_to_standardised_channels(self):
        R = _raw()
        p = _params(R, ndvi_weight=70, gvi_weight=30)
        z = (R - R.mean(0)) / R.std(0)
        got = cgi_formulas.compute_cgi(
            "weighted_average_gvi", p, {"ndvi": R[:, 0], "gvi": R[:, 1]}
        )
        np.testing.assert_allclose(got, 0.7 * z[:, 0] + 0.3 * z[:, 1], atol=1e-6)

    def test_the_three_channel_form_too(self):
        rng = np.random.default_rng(2)
        comps = {
            c: rng.normal(0.5, s, 60)
            for c, s in (("ndvi", 0.1), ("veg", 0.3), ("terrain", 0.05))
        }
        p = {"ndvi_weight": 50, "veg_weight": 25, "terrain_weight": 25}
        for c, a in comps.items():
            p[f"{c}_center"], p[f"{c}_scale"] = float(a.mean()), float(a.std())
        z = {c: (a - a.mean()) / a.std() for c, a in comps.items()}
        got = cgi_formulas.compute_cgi("weighted_average", p, comps)
        np.testing.assert_allclose(
            got, 0.5 * z["ndvi"] + 0.25 * z["veg"] + 0.25 * z["terrain"], atol=1e-6
        )

    def test_params_recorded_without_a_center_replay_on_raw_values(self):
        got = cgi_formulas.compute_cgi(
            "weighted_average_gvi",
            {"ndvi_weight": 50, "gvi_weight": 50},
            {"ndvi": np.array([0.2, 0.6]), "gvi": np.array([0.4, 0.0])},
        )
        np.testing.assert_allclose(got, [0.3, 0.3], atol=1e-7)

    def test_the_composite_reproduces_the_models_index_given_covariates(self):
        # Covariates explain more of one channel than the other, so dividing by
        # the raw SD would distort the shares; the covariate-adjusted SD that
        # ``prep`` divided by reproduces the fitted index exactly.
        rng = np.random.default_rng(5)
        n = 2000
        cov = rng.normal(size=(n, 2))
        X = rng.normal(size=(n, 2, 2, 1))
        X[:, 0, :, 0] += 3.0 * cov[:, :1]
        X[:, 0] *= 0.1
        yr, Xr, info = bi.prep(X, rng.normal(size=n), cov, return_info=True)
        w = np.array([0.6, 0.4])
        model_index = Xr[:, 0, 1, 0] * w[0] + Xr[:, 1, 0, 0] * w[1]
        scaling = bi.composite_scaling(
            X,
            info["column_sd"],
            channels=["ndvi", "gvi"],
            channel_index=[0, 1],
            radii=[100.0, 200.0],
            stats=["mean"],
            picked=((200, "mean"), (100, "mean")),
        )
        p = dict(scaling, ndvi_weight=60, gvi_weight=40)
        comp = cgi_formulas.compute_cgi(
            "weighted_average_gvi", p, {"ndvi": X[:, 0, 1, 0], "gvi": X[:, 1, 0, 0]}
        )
        q, _ = bi.covariate_basis(cov, n)
        comp_r = comp - q @ (q.T @ comp)
        self.assertAlmostEqual(
            float(np.corrcoef(comp_r, model_index)[0, 1]), 1.0, places=10
        )
        raw_sd = dict(
            p,
            ndvi_scale=float(X[:, 0, 1, 0].std()),
            gvi_scale=float(X[:, 1, 0, 0].std()),
        )
        naive = cgi_formulas.compute_cgi(
            "weighted_average_gvi",
            raw_sd,
            {"ndvi": X[:, 0, 1, 0], "gvi": X[:, 1, 0, 0]},
        )
        naive_r = naive - q @ (q.T @ naive)
        self.assertLess(float(np.corrcoef(naive_r, model_index)[0, 1]), 0.999)


class TestSynergy(unittest.TestCase):
    def test_compute_cgi_reproduces_the_sweeps_synergy_index(self):
        R = _raw()
        sf = bi.SynergyFit(
            np.array([0.3, 0.2, 0.5]), np.array([0.6, 0.9]), R.mean(0), R.std(0)
        )
        p = _params(
            R, w_ndvi=30, w_gvi=20, w_ndvi_gvi=50, ndvi_power=0.6, gvi_power=0.9
        )
        got = cgi_formulas.compute_cgi(
            "synergy_gvi", p, {"ndvi": R[:, 0], "gvi": R[:, 1]}
        )
        np.testing.assert_allclose(got, sf.apply(R), rtol=1e-6, atol=1e-9)

    def test_the_three_channel_form_uses_the_same_curve(self):
        rng = np.random.default_rng(1)
        comps = {c: rng.normal(0.5, 0.1, 50) for c in ("ndvi", "veg", "terrain")}
        params = {
            "w_ndvi": 50,
            "w_veg": 0,
            "w_ter": 0,
            "w_ndvi_veg": 50,
            "w_ndvi_ter": 0,
            "w_ter_veg": 0,
            "w_ndvi_veg_ter": 0,
            "ndvi_power": 0.5,
            "veg_power": 1.0,
            "terrain_power": 1.0,
        }
        for c in comps:
            params[f"{c}_center"], params[f"{c}_scale"] = 0.5, 0.1
        zn = norm.cdf((comps["ndvi"] - 0.5) / 0.1)
        zv = norm.cdf((comps["veg"] - 0.5) / 0.1)
        got = cgi_formulas.compute_cgi("synergy", params, comps)
        np.testing.assert_allclose(got, 0.5 * zn**0.5 + 0.5 * zn * zv, rtol=1e-6)

    def test_values_outside_the_training_range_are_not_clipped(self):
        params = {
            "w_ndvi": 100,
            "w_gvi": 0,
            "w_ndvi_gvi": 0,
            "ndvi_power": 1.0,
            "gvi_power": 1.0,
            "ndvi_center": 0.4,
            "ndvi_scale": 0.1,
            "gvi_center": 0.2,
            "gvi_scale": 0.1,
        }
        got = cgi_formulas.compute_cgi(
            "synergy_gvi", params, {"ndvi": np.array([0.9, 1.2]), "gvi": np.zeros(2)}
        )
        self.assertLess(got[0], got[1])
        self.assertLess(got[1], 1.0)

    def test_params_recorded_without_a_center_replay_as_fitted(self):
        params = {
            "w_ndvi": 50,
            "w_gvi": 50,
            "w_ndvi_gvi": 0,
            "ndvi_power": 1.0,
            "gvi_power": 1.0,
        }
        got = cgi_formulas.compute_cgi(
            "synergy_gvi",
            params,
            {"ndvi": np.array([-0.2, 0.4, 1.5]), "gvi": np.array([0.2, 0.2, 0.2])},
        )
        np.testing.assert_allclose(got, [0.1, 0.3, 0.6], rtol=1e-6)


class TestScalingIsRecorded(unittest.TestCase):
    @staticmethod
    def _tensor():
        rng = np.random.default_rng(3)
        X = rng.normal(0.5, 0.2, size=(300, 2, 3, 2))
        X[:5, 1, 2, 1] = np.nan
        sd = np.full((2, 3, 2), 0.15)
        return X, sd

    def test_center_is_the_cell_mean_and_scale_the_adjusted_sd(self):
        X, sd = self._tensor()
        sd[1, 2, 1] = 0.07
        out = bi.composite_scaling(
            X,
            sd,
            channels=["ndvi", "gvi"],
            channel_index=[0, 1],
            radii=[100.0, 200.0, 400.0],
            stats=["mean", "p50"],
            picked=((200, "mean"), (400, "p50")),
        )
        self.assertAlmostEqual(out["ndvi_center"], float(X[:, 0, 1, 0].mean()))
        self.assertAlmostEqual(out["gvi_center"], float(X[5:, 1, 2, 1].mean()))
        self.assertEqual((out["ndvi_scale"], out["gvi_scale"]), (0.15, 0.07))

    def test_the_engine_carries_it_onto_normalised_channels(self):
        from geofuse.fusion import MetricFusionEngine as E

        X, sd = self._tensor()
        eng = types.SimpleNamespace(
            normalize_channels=True,
            _channel_minmax={"ndvi": (0.0, 1.0), "gvi": (0.0, 2.0)},
        )
        eng._channel_scaling = types.MethodType(E._channel_scaling, eng)
        out = eng._channel_scaling(
            X,
            np.array([100.0, 200.0, 400.0]),
            ["mean", "p50"],
            ["ndvi", "gvi"],
            ["ndvi", "gvi"],
            ((200, "mean"), (400, "p50")),
            sd,
        )
        self.assertAlmostEqual(out["gvi_center"], float(X[5:, 1, 2, 1].mean()) / 2.0)
        self.assertAlmostEqual(out["gvi_scale"], 0.075)
        self.assertAlmostEqual(out["ndvi_scale"], 0.15)


if __name__ == "__main__":
    unittest.main()
