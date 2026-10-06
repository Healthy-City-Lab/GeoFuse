"""The synergy form is one curve wherever it is evaluated.

The sweep fits it (``bayesian_index.SynergyFit``), the posterior samples it, and
the composite path builds maps and test scores from the saved parameters
(``cgi_formulas.compute_cgi``). All three read each channel as its approximate
percentile, ``Φ((x − center) / scale)``. If the composite path scaled the
channels any other way, the map would not be the model that was fitted.
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


class TestCompositeMatchesTheFittedForm(unittest.TestCase):
    def test_compute_cgi_reproduces_the_sweeps_synergy_index(self):
        R = _raw()
        sf = bi.SynergyFit(np.array([0.3, 0.2, 0.5]), np.array([0.6, 0.9]),
                           R.mean(0), R.std(0))
        params = {
            "w_ndvi": 30, "w_gvi": 20, "w_ndvi_gvi": 50,
            "ndvi_power": 0.6, "gvi_power": 0.9,
            "ndvi_center": float(R[:, 0].mean()), "ndvi_scale": float(R[:, 0].std()),
            "gvi_center": float(R[:, 1].mean()), "gvi_scale": float(R[:, 1].std()),
        }
        got = cgi_formulas.compute_cgi("synergy_gvi", params,
                                       {"ndvi": R[:, 0], "gvi": R[:, 1]})
        np.testing.assert_allclose(got, sf.apply(R), rtol=1e-6, atol=1e-9)

    def test_the_three_channel_form_uses_the_same_curve(self):
        rng = np.random.default_rng(1)
        comps = {c: rng.normal(0.5, 0.1, 50) for c in ("ndvi", "veg", "terrain")}
        params = {"w_ndvi": 50, "w_veg": 0, "w_ter": 0, "w_ndvi_veg": 50,
                  "w_ndvi_ter": 0, "w_ter_veg": 0, "w_ndvi_veg_ter": 0,
                  "ndvi_power": 0.5, "veg_power": 1.0, "terrain_power": 1.0}
        for c in comps:
            params[f"{c}_center"], params[f"{c}_scale"] = 0.5, 0.1
        zn = norm.cdf((comps["ndvi"] - 0.5) / 0.1)
        zv = norm.cdf((comps["veg"] - 0.5) / 0.1)
        got = cgi_formulas.compute_cgi("synergy", params, comps)
        np.testing.assert_allclose(got, 0.5 * zn ** 0.5 + 0.5 * zn * zv, rtol=1e-6)

    def test_values_outside_the_training_range_are_not_clipped(self):
        params = {"w_ndvi": 100, "w_gvi": 0, "w_ndvi_gvi": 0, "ndvi_power": 1.0,
                  "gvi_power": 1.0, "ndvi_center": 0.4, "ndvi_scale": 0.1,
                  "gvi_center": 0.2, "gvi_scale": 0.1}
        got = cgi_formulas.compute_cgi(
            "synergy_gvi", params,
            {"ndvi": np.array([0.9, 1.2]), "gvi": np.zeros(2)})
        self.assertLess(got[0], got[1])
        self.assertLess(got[1], 1.0)

    def test_params_recorded_without_a_center_replay_as_fitted(self):
        params = {"w_ndvi": 50, "w_gvi": 50, "w_ndvi_gvi": 0,
                  "ndvi_power": 1.0, "gvi_power": 1.0}
        got = cgi_formulas.compute_cgi(
            "synergy_gvi", params,
            {"ndvi": np.array([-0.2, 0.4, 1.5]), "gvi": np.array([0.2, 0.2, 0.2])})
        np.testing.assert_allclose(got, [0.1, 0.3, 0.6], rtol=1e-6)


class TestEngineRecordsTheScaling(unittest.TestCase):
    @staticmethod
    def _engine(normalize=False):
        from geofuse.fusion import MetricFusionEngine as E

        eng = types.SimpleNamespace(normalize_channels=normalize,
                                    _channel_minmax={"ndvi": (0.0, 1.0),
                                                     "gvi": (0.0, 2.0)})
        for name in ("_synergy_scaling", "_normalize_channels"):
            setattr(eng, name, types.MethodType(getattr(E, name), eng))
        return eng

    def _tensor(self):
        rng = np.random.default_rng(3)
        X = rng.normal(0.5, 0.2, size=(300, 2, 3, 2))
        X[:5, 1, 2, 1] = np.nan
        return X

    def test_mean_and_sd_at_the_cell_the_composite_is_built_from(self):
        X = self._tensor()
        out = self._engine()._synergy_scaling(
            X, np.array([100.0, 200.0, 400.0]), ["mean", "p50"], ["ndvi", "gvi"],
            ["ndvi", "gvi"], ((200, "mean"), (400, "p50")))
        self.assertAlmostEqual(out["ndvi_center"], float(X[:, 0, 1, 0].mean()))
        gvi = X[5:, 1, 2, 1]
        self.assertAlmostEqual(out["gvi_center"], float(gvi.mean()))
        self.assertAlmostEqual(out["gvi_scale"], float(gvi.std()))

    def test_normalised_channels_are_scaled_where_they_arrive(self):
        X = self._tensor()
        out = self._engine(normalize=True)._synergy_scaling(
            X, np.array([100.0, 200.0, 400.0]), ["mean", "p50"], ["ndvi", "gvi"],
            ["ndvi", "gvi"], ((200, "mean"), (400, "p50")))
        gvi = np.clip(X[5:, 1, 2, 1] / 2.0, 0.0, 1.0)
        self.assertAlmostEqual(out["gvi_center"], float(gvi.mean()))


if __name__ == "__main__":
    unittest.main()
