"""Scale-ladder slopes in one exposure unit, and the smooth's absorbed shares.

Pinned here:

- with a constant per-unit effect and no confounding, the slope per SD0 stays
  constant along the ladder (within sampling error) while the per-SD slope
  falls with the share of the exposure the smooth absorbs;
- the per-SD0 rows' t and p equal the per-SD slope's at every scale;
- ``absorbed_given_cov`` is ``1 − x'M_[Z,S]x / x'M_Z x`` by OLS, and the
  record carries ``chance_share``;
- an unadjusted engine reports no per-SD0 contrast and no "not applied" label;
  an adjusted one labels the covariates-only reports.
"""

import os
import sys
import tempfile
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas  # noqa: F401  (pyproj before torchvision, see vision.py)
import numpy as np
import pandas as pd

import test_spatial_reporting as reporting_fixture
from geofuse import negative_controls as nc
from geofuse.fusion import MetricFusionEngine, _absorbed_given_cov
from geofuse.jobs import outer_splits
from test_spatial_scale import _scaled

_EXTRA = ("compute_raw_effects",)


def _engine(n_ent=2400, n_test=1200, seed=3, effect=0.1):
    """A stand-in engine (the reporting fixture's methods) over entities on a
    40 km square: the exposure is a smooth field with 8 km waves plus its own
    noise, the target loads on the exposure only (no confounding)."""
    rng = np.random.default_rng(seed)
    xy = rng.random((n_ent, 2)) * 40_000.0
    wave = 2 * np.pi / 8_000.0
    u = np.sin(wave * xy[:, 0]) * np.cos(wave * xy[:, 1])
    u = (u - u.mean()) / u.std()
    x = u + 0.8 * rng.normal(size=n_ent)
    y = effect * x + rng.normal(size=n_ent)
    control = rng.normal(size=n_ent)
    age = rng.normal(size=n_ent)
    ids = np.arange(n_ent)
    pid = np.repeat(ids, 2)
    full = pd.DataFrame(
        {
            "polygon_id": pid,
            "target": y[pid],
            "age": age[pid],
            "grip": control[pid],
            "_cx": xy[pid, 0],
            "_cy": xy[pid, 1],
        }
    )
    frame = pd.DataFrame({"polygon_id": ids, "target": y, "composite": x})
    eng = types.SimpleNamespace(
        negative_control_columns=["grip"],
        is_panel=False,
        covariate_columns=["age"],
        _covariate_dummy_map={},
        test_data=full[full["polygon_id"] < n_test],
        _active_greenery_channel="cgi",
        spatial_adjust_method="spatial_plus",
        spatial_adjust_max_df=10,
        spatial_adjust_df=tuple(range(11)),
        spatial_adjust_criterion="aic",
        spatial_adjust_neighbours=10,
        spatial_adjust_knots_min=35,
        spatial_adjust_knots_max=150,
        spatial_adjust_eps_m=None,
        spatial_adjust_min_component=None,
        _spatial_scale=None,
        _spatial_basis_cache={},
        _spatial_adjust_summary=None,
        _spatial_run_summary=None,
        _spatial_selections={"n": 0, "at_cap": 0},
    )
    eng.apply_fusion = lambda weights=None: frame
    eng._full_data_frame = lambda: full
    eng._composite_channels = lambda: ()
    for name in reporting_fixture._METHODS + _EXTRA:
        setattr(eng, name, types.MethodType(getattr(MetricFusionEngine, name), eng))
    eng._finite_rows = MetricFusionEngine._finite_rows
    eng._with_smooth = MetricFusionEngine._with_smooth
    return _scaled(eng, (5_000.0, 10_000.0, 20_000.0), eps=1e6)


def _rows(eng):
    rows = outer_splits.scale_rows(
        eng, {}, [5_000.0, 10_000.0, 20_000.0, None], n_boot=0, raw_exposures={}
    )
    by = {}
    for r in rows:
        if r.get("control") is None and r["exposure"] == "cgi":
            by[(r["quantity"], r["spatial_scale_m"])] = r
    return by


class TestFixedUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = _engine()
        cls.rows = _rows(cls.eng)

    def test_the_slope_per_sd0_is_constant_while_beta_falls(self):
        scales = ("5000", "10000", "20000", "none")
        per = [self.rows[("held_out_per_sd0", s)] for s in scales]
        beta = [self.rows[("held_out_beta", s)]["estimate"] for s in scales]
        sd0 = {round(r["sd0"], 12) for r in per}
        self.assertEqual(len(sd0), 1)
        ref = per[-1]["estimate"]
        for r in per:
            se = (r["ci_high"] - r["ci_low"]) / (2 * 1.96)
            self.assertLess(abs(r["estimate"] - ref), 2.5 * se, r["spatial_scale_m"])
        # The finest scale absorbs most of the exposure; its per-SD slope is
        # the smallest, by about √(1 − absorbed given the covariates).
        self.assertLess(beta[0], beta[1])
        self.assertLess(beta[1], beta[-1])
        with self.eng.spatial_scale(5_000.0):
            slope = self.eng.compute_partial_slopes({})["test"]
        absorbed = slope["spatial_adjustment"]["absorbed_given_cov"]
        self.assertGreater(absorbed, 0.25)
        self.assertAlmostEqual(beta[0] / beta[-1], np.sqrt(1 - absorbed), delta=0.1)

    def test_t_and_p_equal_the_per_sd_slope(self):
        for s in ("5000", "10000", "20000", "none"):
            per = self.rows[("held_out_per_sd0", s)]
            beta = self.rows[("held_out_beta", s)]
            self.assertAlmostEqual(per["p_value"], beta["p_value"], delta=1e-10, msg=s)

    def test_the_record_carries_both_shares(self):
        with self.eng.spatial_scale(5_000.0):
            record = self.eng.compute_partial_slopes({})["test"]["spatial_adjustment"]
        for key in ("absorbed", "absorbed_given_cov", "chance_share"):
            self.assertIn(key, record)
        self.assertGreater(record["chance_share"], 0.0)
        self.assertLess(record["chance_share"], record["absorbed_given_cov"])
        r = self.rows[("held_out_beta", "5000")]
        self.assertIn("spatial_absorbed_given_cov", r)
        self.assertIn("spatial_chance_share", r)


class TestAbsorbedGivenCovariates(unittest.TestCase):
    def test_it_is_the_ols_share(self):
        rng = np.random.default_rng(0)
        n = 500
        cov = rng.normal(size=(n, 3))
        smooth = rng.normal(size=(n, 12))
        x = cov @ [0.5, -0.2, 0.1] + smooth[:, :3] @ [0.4, 0.3, 0.2] + rng.normal(size=n)

        def resid(X):
            Xc = np.column_stack([np.ones(n), X])
            P = Xc @ np.linalg.pinv(Xc)
            r = x - P @ x
            return float(r @ r)

        want = 1 - resid(np.column_stack([cov, smooth])) / resid(cov)
        self.assertAlmostEqual(_absorbed_given_cov(x, cov, smooth), want, delta=1e-10)

    def test_without_covariates_it_is_the_absorbed_share(self):
        from geofuse.fusion import _absorbed_share

        rng = np.random.default_rng(1)
        smooth = rng.normal(size=(300, 5))
        x = smooth[:, 0] + rng.normal(size=300)
        self.assertAlmostEqual(
            _absorbed_given_cov(x, None, smooth), _absorbed_share(x, smooth), 6
        )


class TestUnadjusted(unittest.TestCase):
    def test_no_per_sd0_without_a_smooth(self):
        eng = reporting_fixture._engine("none")
        eng.compute_raw_effects = types.MethodType(
            MetricFusionEngine.compute_raw_effects, eng
        )
        eng._composite_channels = lambda: ()
        eng.spatial_adjust_scales = ()
        raw = eng.compute_raw_effects({})
        self.assertNotIn("per_sd0", raw["exposures"]["cgi"]["target"])
        self.assertFalse(
            [r for r in outer_splits.raw_rows(raw) if r["quantity"].endswith("sd0")]
        )

    def test_the_label_follows_the_method(self):
        def engine(**kw):
            return MetricFusionEngine(
                target_file="dummy.geojson", cache_dir=tempfile.mkdtemp(), **kw
            )

        self.assertIsNone(engine()._smooth_not_applied)
        self.assertEqual(
            engine(spatial_adjust_method="ks_aic")._smooth_not_applied, "not applied"
        )

    def test_the_per_sd0_slope_is_the_raw_slope(self):
        eng = _engine(n_ent=800, n_test=400)
        raw = eng.compute_raw_effects({})
        per = raw["exposures"]["cgi"]["target"]["per_sd0"]
        frame = eng.apply_fusion()
        test = frame["polygon_id"].to_numpy() < 400
        x = frame["composite"].to_numpy()[test]
        age = eng._full_data_frame().groupby("polygon_id")["age"].first().to_numpy()
        xr, _, _, _ = nc._residualised(x, frame["target"].to_numpy()[test], age[test])
        self.assertAlmostEqual(per["sd0"], float(xr.std()), delta=1e-12)


if __name__ == "__main__":
    unittest.main()
