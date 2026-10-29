"""Spatial adjustment reaches the slopes, negative controls and specificity.

With ``spatial_adjust_method`` other than ``none`` the reported slopes adjust
for the same kind of coordinate smooth the scores do. Pinned here:

- ``extra`` columns in the paired statistics: an exposure's join both of its
  slopes, an outcome's only the slopes on it, and unit weights reproduce
  :func:`negative_controls.partial_slope` on each slope's own design.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas as gpd  # noqa: F401
import numpy as np

from geofuse import negative_controls as nc

# ────────────────────────────────────────────────────────────────────
# The statistics' extra columns
# ────────────────────────────────────────────────────────────────────


def _data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x, other = rng.normal(size=n), rng.normal(size=n)
    target = 0.3 * x + rng.normal(size=n)
    control = 0.1 * x + rng.normal(size=n)
    cov = rng.normal(size=(n, 2))
    blocks = {k: rng.normal(size=(n, 3)) for k in ("a", "b", "t", "c")}
    return x, other, target, control, cov, blocks


def _beta(x, y, *design):
    return nc.partial_slope(x, y, np.column_stack(design))["beta"]


class TestExtraColumns(unittest.TestCase):
    def test_an_exposures_columns_are_the_covariates_of_both_slopes(self):
        x, _, target, control, cov, b = _data()
        got = nc.paired_contrast(
            x, target, control, cov, extra={"exposure": b["a"]}, n_boot=50, seed=3
        )
        want = nc.paired_contrast(
            x, target, control, np.column_stack([cov, b["a"]]), n_boot=50, seed=3
        )
        self.assertEqual(got, want)

    def test_an_outcomes_columns_join_only_its_own_slope(self):
        x, _, target, control, cov, b = _data()
        got = nc.paired_contrast(
            x, target, control, cov, extra={"target": b["t"], "control": b["c"]}
        )
        d = abs(_beta(x, target, cov, b["t"])) - abs(_beta(x, control, cov, b["c"]))
        self.assertAlmostEqual(got["delta"], d, places=10)

    def test_specificity_slopes_use_each_slopes_own_design(self):
        x, other, target, control, cov, b = _data()
        cases = {
            "by exposure": (
                {"exposure": b["a"], "other": b["b"]},
                [(x, target, b["a"]), (x, control, b["a"])]
                + [(other, target, b["b"]), (other, control, b["b"])],
            ),
            "by outcome": (
                {"target": b["t"], "control": b["c"]},
                [(x, target, b["t"]), (x, control, b["c"])]
                + [(other, target, b["t"]), (other, control, b["c"])],
            ),
        }
        keys = ("beta_target", "beta_control", "beta_target_other")
        keys += ("beta_control_other",)
        for name, (extra, slopes) in cases.items():
            with self.subTest(name):
                got = nc.specificity_contrast(
                    x, other, target, control, cov, extra=extra, n_boot=0
                )
                for key, (e, y, block) in zip(keys, slopes, strict=True):
                    self.assertAlmostEqual(got[key], _beta(e, y, cov, block), places=10)

    def test_the_same_exposure_and_columns_differ_by_zero_on_every_resample(self):
        x, _, target, control, cov, b = _data(n=300)
        got = nc.specificity_contrast(
            x,
            x.copy(),
            target,
            control,
            cov,
            extra={"exposure": b["a"], "other": b["a"].copy()},
            n_boot=200,
            seed=1,
        )
        self.assertEqual((got["delta"], got["ci_low"], got["ci_high"]), (0, 0, 0))

    def test_transfer_test_hands_each_slope_its_columns(self):
        x, _, target, control, cov, b = _data()
        out = nc.transfer_test(
            x,
            target,
            {"grip": control},
            cov,
            extra={"target": b["t"], "controls": {"grip": b["c"]}},
            n_boot=0,
        )
        self.assertAlmostEqual(
            out["target"]["beta"], _beta(x, target, cov, b["t"]), places=12
        )
        self.assertAlmostEqual(
            out["controls"]["grip"]["beta"], _beta(x, control, cov, b["c"]), places=12
        )

    def test_a_misspelt_role_raises(self):
        x, _, target, control, cov, b = _data(n=50)
        with self.assertRaises(ValueError):
            nc.paired_contrast(x, target, control, cov, extra={"outcome": b["t"]})
        with self.assertRaises(ValueError):
            nc.transfer_test(x, target, {"g": control}, cov, extra={"other": b["t"]})


if __name__ == "__main__":
    unittest.main()
