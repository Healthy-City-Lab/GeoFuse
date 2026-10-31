"""Which spatial smooth each exposure of a paired CGI-vs-exposure test gets.

``spatial_paired_smooth`` decides it under Spatial+: ``own`` scores each
exposure with the smooth selected on itself, as when it is scored alone, so the
paired difference is the difference of the two printed scores; ``cgi`` gives
both the CGI's smooth (the earlier behaviour); ``outcome`` both one selected on
the outcome given the covariates (the KS-AIC rule); ``max`` both the smooth at
the larger of the two exposures' dfs. Under KS-AIC every rule is ``outcome``.
Every paired row records the rule and each side's df. Unadjusted runs carry no
record.
"""

import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas  # noqa: F401  (pyproj before torchvision, see vision.py)
import numpy as np
import pandas as pd

from geofuse import objective_scoring
from geofuse.fusion import MetricFusionEngine as E

_METHODS = (
    "_test_entity_mask",
    "_whole_data_covariates",
    "_whole_data_coords",
    "_whole_data_longitudinal_keys",
    "_paired_smooths",
    "_smooth_at",
    "_spatial_selection",
    "_spatial_smooth",
    "_spatial_basis_columns",
    "_spatial_candidates",
    "_selection_record",
    "_count_selection",
    "_block_resampling",
    "_paired_difference_on",
    "paired_objective_difference",
)


def _engine(method="spatial_plus", rule="own", n=500, seed=5):
    """The paired-difference methods over a CGI whose spatial part is smooth
    and a standalone exposure whose spatial part is rough, so Spatial+ picks
    different dfs for the two."""
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 10_000.0
    smooth = np.sin(2 * np.pi * xy[:, 0] / 20_000.0) + np.cos(
        2 * np.pi * xy[:, 1] / 25_000.0
    )
    rough = np.sin(2 * np.pi * xy[:, 0] / 4_000.0) * np.cos(
        2 * np.pi * xy[:, 1] / 5_000.0
    )
    g = rng.normal(size=n)
    cgi = smooth + 0.6 * g + 0.3 * rng.normal(size=n)
    other = rough + 0.5 * g + 0.4 * rng.normal(size=n)
    age = rng.normal(size=n)
    target = 0.3 * g + 0.5 * smooth + 0.2 * age + rng.normal(size=n)
    ids = np.arange(n)
    full = pd.DataFrame(
        {
            "polygon_id": ids,
            "target": target,
            "age": age,
            "_cx": xy[:, 0],
            "_cy": xy[:, 1],
        }
    )
    frames = {
        "cgi": pd.DataFrame({"polygon_id": ids, "target": target, "composite": cgi}),
        "veg": pd.DataFrame({"polygon_id": ids, "target": target, "composite": other}),
    }
    eng = types.SimpleNamespace(
        is_panel=False,
        is_longitudinal=False,
        covariate_columns=["age"],
        test_data=full[full["polygon_id"] < n // 2],
        _active_greenery_channel="cgi",
        spatial_adjust_method=method,
        spatial_paired_smooth=rule,
        spatial_adjust_df=tuple(range(11)),
        spatial_adjust_max_df=10,
        spatial_adjust_criterion="aic",
        spatial_adjust_neighbours=10,
        spatial_adjust_knots_min=35,
        spatial_adjust_knots_max=150,
        spatial_adjust_eps_m=None,
        _spatial_basis_cache={},
        _spatial_adjust_summary=None,
        _spatial_selections={"n": 0, "at_cap": 0},
        residualize_method="linear",
        _residualize_options={"residualize_method": "linear"},
        _pdcor_cache={},
        _spline_basis_cache={},
        _cancel_callback=None,
    )
    eng.apply_fusion = lambda weights=None: frames[eng._active_greenery_channel]
    eng._full_data_frame = lambda: full
    for name in _METHODS:
        setattr(eng, name, types.MethodType(getattr(E, name), eng))
    eng._with_smooth = E._with_smooth
    eng._paired_engine = E._paired_engine
    eng.data = dict(xy=xy, target=target, cgi=cgi, other=other, age=age)
    return eng


def _paired(eng, n_bootstrap=200):
    return eng.paired_objective_difference(
        {}, {}, "veg", "r2", n_bootstrap=n_bootstrap, seed=3
    )


def _held_out_score(eng, exposure):
    """The exposure's held-out score as it is printed: its own smooth."""
    d = eng.data
    test = np.arange(len(d["target"])) < len(d["target"]) // 2
    cov = d["age"][test][:, None]
    sb = eng._spatial_basis_columns(
        d["xy"][test], d["target"][test], cov, exposure[test]
    )
    return objective_scoring.score(
        "r2",
        d["target"][test],
        exposure[test],
        cov,
        spatial_basis=sb,
        spatial_method=eng.spatial_adjust_method,
    )


class PairedSmoothTests(unittest.TestCase):
    def test_own_differences_the_printed_scores(self):
        eng = _engine(rule="own")
        out = _paired(eng)
        cgi = _held_out_score(eng, eng.data["cgi"])
        other = _held_out_score(eng, eng.data["other"])
        self.assertAlmostEqual(out["cgi_score"], cgi, places=12)
        self.assertAlmostEqual(out["standalone_score"], other, places=12)
        self.assertAlmostEqual(out["observed_diff"], cgi - other, places=10)
        rec = out["spatial_adjustment"]
        self.assertEqual(rec["rule"], "own")
        self.assertNotEqual(rec["df_cgi"], rec["df_exposure"])
        self.assertLessEqual(out["lower"], out["observed_diff"])
        self.assertGreaterEqual(out["upper"], out["observed_diff"])

    def test_cgi_is_the_earlier_behaviour(self):
        fast = _paired(_engine(rule="cgi"))
        # The per-replicate fits the block engine replaces.
        eng = _engine(rule="cgi")
        with mock.patch.object(eng, "_block_resampling", lambda metric: False):
            slow = _paired(eng)
        for key in ("observed_diff", "cgi_score", "standalone_score"):
            self.assertAlmostEqual(fast[key], slow[key], places=12, msg=key)
        for key in ("lower", "upper", "p_value"):
            self.assertAlmostEqual(fast[key], slow[key], places=10, msg=key)
        rec = fast["spatial_adjustment"]
        self.assertEqual(rec["df_cgi"], rec["df_exposure"])

    def test_outcome_equals_a_ks_aic_run(self):
        plus = _paired(_engine(rule="outcome"))
        ks = _paired(_engine(method="ks_aic"))
        for key in ("observed_diff", "cgi_score", "standalone_score", "lower", "upper"):
            self.assertAlmostEqual(plus[key], ks[key], places=12, msg=key)
        self.assertEqual(plus["spatial_adjustment"]["rule"], "outcome")

    def test_max_takes_the_larger_df(self):
        own = _paired(_engine(rule="own"))["spatial_adjustment"]
        rec = _paired(_engine(rule="max"))["spatial_adjustment"]
        larger = max(d for d in (own["df_cgi"], own["df_exposure"]) if d is not None)
        self.assertEqual((rec["df_cgi"], rec["df_exposure"]), (larger, larger))

    def test_every_rule_is_outcome_under_ks_aic(self):
        rules = ("own", "cgi", "max")
        rows = [_paired(_engine(method="ks_aic", rule=r)) for r in rules]
        for row in rows[1:]:
            for key in ("observed_diff", "lower", "upper", "p_value"):
                self.assertEqual(row[key], rows[0][key], key)
        self.assertEqual({r["spatial_adjustment"]["rule"] for r in rows}, {"outcome"})

    def test_unadjusted_runs_have_no_record(self):
        eng = _engine(method="none")
        out = _paired(eng)
        self.assertNotIn("spatial_adjustment", out)
        d = eng.data
        test = np.arange(len(d["target"])) < len(d["target"]) // 2
        want = objective_scoring.score(
            "r2", d["target"][test], d["cgi"][test], d["age"][test][:, None]
        )
        self.assertEqual(out["cgi_score"], want)

    def test_the_record_reaches_the_tables(self):
        from geofuse.jobs.outer_splits import spatial_columns, spatial_label

        rec = _paired(_engine(rule="own"))["spatial_adjustment"]
        cols = spatial_columns(rec)
        self.assertEqual(cols["spatial_rule"], "own")
        self.assertIn("spatial_df_cgi", cols)
        self.assertIn("spatial_df_exposure", cols)
        self.assertIn("cgi", spatial_label(rec))


if __name__ == "__main__":
    unittest.main()
