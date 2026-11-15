"""Spatial+ step 1 on the search inputs (a geofuse extension).

Every search column is residualized on its own penalized smooth per group,
and the outcome and the residuals are multiplied by ``(I − H)^{1/2}``, ``H``
the outcome smoother fitted without any exposure. Pinned here:

- one input, one group: the transformed OLS slope equals Spatial+'s β̂⁺ with
  the outcome smoother's penalty fixed at its outcome-only value, and stays
  within 1 % of β̂⁺ with the penalty chosen jointly;
- the covariates map to zero under the transform, and ``ỹᵀỹ = yᵀ(I − H)y``;
- a coarse confounder drives channel ndvi and the outcome while channel gvi
  carries a within-area effect: the transformed search gives gvi the largest
  weight in at least 4 of 5 seeds;
- the synergy form is refused; the default tuning is unchanged.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas  # noqa: F401  (pyproj before torchvision, see vision.py)
import numpy as np
import test_spatial_tuning as tuning

from geofuse import settings
from geofuse import spatial_plus as sp
from geofuse.fusion import MetricFusionEngine


def _one_group(seed=0, n=500, covariates=True):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 10_000
    z = np.sin(xy[:, 0] / 2500) + np.cos(xy[:, 1] / 3000)
    cov = rng.normal(size=(n, 2)) if covariates else None
    x = 0.5 * z + 0.3 * rng.normal(size=n)
    y = 0.4 * x + z + rng.normal(size=n)
    if covariates:
        y = y + 0.3 * cov[:, 0]
    return xy, x, y, cov


class TransformTests(unittest.TestCase):
    def test_the_slope_is_spatial_plus_at_the_outcome_only_penalty(self):
        xy, x, y, cov = _one_group()
        labels = np.zeros(len(y), int)
        out = sp.search_transform(xy, labels, x[:, None], y, cov, k=60)
        xt, yt = out["X"][:, 0], out["y"]
        slope = float(xt @ yt / (xt @ xt))
        sm = out["smoother"]
        r = sp.exposure_residual(sm, x, "gcv")["r"]
        design = sp.Design(sm, cov)
        fixed = design.quad(r, y, out["lam_y"]) / design.quad(r, r, out["lam_y"])
        self.assertAlmostEqual(slope, fixed, delta=1e-8 * abs(fixed))
        joint = sp.outcome_fit(sm, y, r, cov, "gcv")["beta"]
        self.assertLess(abs(slope - joint), 0.01 * abs(joint))

    def test_covariates_vanish_and_the_outcome_norm(self):
        xy, x, y, cov = _one_group(seed=1)
        out = sp.search_transform(xy, np.zeros(len(y), int), x[:, None], y, cov, k=60)
        design = out["design"]
        W = sp.half_residual(design, out["lam_y"])
        self.assertLess(np.abs(W(cov)).max(), 1e-10)
        self.assertLess(np.abs(W(np.ones(len(y)))).max(), 1e-10)
        self.assertAlmostEqual(
            float(out["y"] @ out["y"]),
            design.quad(y, y, out["lam_y"]),
            delta=1e-8 * float(y @ y),
        )


def _engine(xy, X, y, cov):
    eng, static = tuning._engine(xy, X, y, cov, scale=None)
    eng.spatial_tune_method = "spatial_plus_inputs"
    eng.spatial_plus_criterion = "gcv"
    eng.spatial_plus_k = 60
    eng.spatial_plus_groups_column = None
    eng.is_panel = False
    eng._plus_inputs_cache = {}
    eng._spatial_plus_inputs = MetricFusionEngine._spatial_plus_inputs.__get__(eng)
    return eng, static


class SearchTests(tuning.SinglePool):
    def test_the_within_area_channel_wins(self):
        picks = []
        for seed in range(5):
            xy, X, y, cov = tuning._confounded(20 + seed)
            eng, _ = _engine(xy, X, y, cov)
            params = MetricFusionEngine.fit_bayesian_index(eng, "r2", **tuning.FIT)
            picks.append(tuning.RecoveryTests._largest(None, params))
            record = params["__spatial_tuning__"]
            self.assertEqual(record["method"], "spatial_plus_inputs")
            self.assertGreater(record["absorbed"]["ndvi"], record["absorbed"]["gvi"])
        self.assertGreaterEqual(picks.count("gvi"), 4, picks)

    def test_the_penalties_are_reused_for_the_same_rows(self):
        xy, X, y, cov = tuning._confounded(3, n=400)
        eng, _ = _engine(xy, X, y, cov)
        MetricFusionEngine.fit_bayesian_index(eng, "r2", **tuning.FIT)
        (entry,) = eng._plus_inputs_cache.values()
        MetricFusionEngine.fit_bayesian_index(eng, "r2", **tuning.FIT)
        (again,) = eng._plus_inputs_cache.values()
        np.testing.assert_array_equal(entry["lam_y"], again["lam_y"])

    def test_synergy_is_refused(self):
        xy, X, y, cov = tuning._confounded(4, n=300)
        eng, _ = _engine(xy, X, y, cov)
        with self.assertRaises(ValueError):
            MetricFusionEngine.fit_bayesian_index(
                eng, "r2", **dict(tuning.FIT, forms=("linear", "synergy"))
            )
        with self.assertRaises(settings.SettingError):
            settings.check(
                "fusion",
                {
                    "spatial_adjust_method": "spatial_plus",
                    "spatial_tune_method": "spatial_plus_inputs",
                    "index_form": "sweep",
                },
            )

    def test_the_command_line_names_the_tuning(self):
        from geofuse.cli.results import _spatial_lines

        lines = _spatial_lines(
            {
                "spatial_tuning": {
                    "method": "spatial_plus_inputs",
                    "columns": 120,
                    "clusters": 3,
                    "outcome_edf": 12.3,
                    "absorbed": {"ndvi": 0.6},
                }
            }
        )
        self.assertEqual(len(lines), 1)
        self.assertIn("Spatial+ residuals of 120 search columns", lines[0])

    def test_the_default_tunes_as_before(self):
        self.assertEqual(settings.defaults("fusion")["spatial_tune_method"], "basis")


if __name__ == "__main__":
    unittest.main()
