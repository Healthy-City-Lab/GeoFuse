"""Specificity contrast: is the CGI less non-specific than another exposure.

With ``D = |β_target| − |β_control|`` per exposure, ``ΔD = D_cgi − D_e > 0``
says the CGI is the more specific of the two. Pinned here:

- unit weights reproduce the four partial slopes on the common rows;
- a benchmark loading on a confounder that drives target and control gives
  ``ΔD > 0`` with an interval above zero, and swapping the roles negates it;
- an exposure equal to the CGI gives ``ΔD = 0`` on every resample;
- the engine compares on the held-out rows, aligns each exposure by entity,
  reports each exposure's own slopes, and skips panels.
"""

import os
import sys
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas as gpd  # noqa: F401
import numpy as np
import pandas as pd

from geofuse import negative_controls as nc


def _confounded(n=4000, seed=0):
    """A confounder U drives target and control; the benchmark loads on U,
    the CGI carries the pathway G with little U."""
    rng = np.random.default_rng(seed)
    u, g = rng.normal(size=n), rng.normal(size=n)
    target = 0.3 * g + 0.3 * u + rng.normal(size=n)
    control = 0.3 * u + rng.normal(size=n)
    cgi = g + 0.1 * u + 0.3 * rng.normal(size=n)
    bench = 0.5 * g + u + 0.3 * rng.normal(size=n)
    cov = rng.normal(size=(n, 2))
    return cgi, bench, target, control, cov


class TestContrast(unittest.TestCase):
    def test_unit_weights_reproduce_the_four_partial_slopes(self):
        cgi, bench, target, control, cov = _confounded(n=600)
        control[:40] = np.nan
        got = nc.specificity_contrast(cgi, bench, target, control, cov, n_boot=0)
        rows = np.isfinite(control)
        self.assertEqual(got["n"], int(rows.sum()))
        for key, x, y in (
            ("beta_target", cgi, target),
            ("beta_control", cgi, control),
            ("beta_target_other", bench, target),
            ("beta_control_other", bench, control),
        ):
            want = nc.partial_slope(x[rows], y[rows], cov[rows])["beta"]
            self.assertAlmostEqual(got[key], want, places=10)
        self.assertAlmostEqual(
            got["delta"], got["d"] - got["d_other"], places=12
        )

    def test_a_confounded_benchmark_is_less_specific(self):
        cgi, bench, target, control, cov = _confounded()
        got = nc.specificity_contrast(
            cgi, bench, target, control, cov, n_boot=600, seed=1
        )
        self.assertGreater(got["delta"], 0.0)
        self.assertGreater(got["ci_low"], 0.0)
        self.assertLess(got["p_value"], 0.01)
        self.assertLess(got["d_other"], got["d"])
        swapped = nc.specificity_contrast(
            bench, cgi, target, control, cov, n_boot=600, seed=1
        )
        self.assertAlmostEqual(swapped["delta"], -got["delta"], places=12)
        self.assertLess(swapped["ci_high"], 0.0)
        self.assertGreater(swapped["p_value"], 0.99)

    def test_the_same_exposure_differs_by_zero_on_every_resample(self):
        cgi, _, target, control, cov = _confounded(n=500)
        got = nc.specificity_contrast(
            cgi, cgi.copy(), target, control, cov, n_boot=300, seed=2
        )
        self.assertEqual(got["delta"], 0.0)
        self.assertEqual((got["ci_low"], got["ci_high"]), (0.0, 0.0))
        self.assertEqual(got["p_value"], 1.0)

    def test_whole_clusters_are_resampled_on_request(self):
        cgi, bench, target, control, cov = _confounded(n=800)
        labels = np.arange(800) % 40
        got = nc.specificity_contrast(
            cgi, bench, target, control, cov, n_boot=100, clusters=labels
        )
        self.assertEqual(got["clusters"], 40)

    def test_too_few_rows_give_empty_values(self):
        got = nc.specificity_contrast([1.0, 2.0], [2.0, 1.0], [1.0, 3.0], [0.0, 1.0])
        self.assertTrue(np.isnan(got["delta"]))
        self.assertEqual(got["n"], 2)


def _engine_stub(n_ent=400, n_test=120, seed=7, controls=("grip", "height")):
    """The engine's reporting helpers over a frame the test controls: the CGI,
    a confounded comparator ``bench`` (rows in another order) and a channel."""
    from geofuse.fusion import MetricFusionEngine as E
    from geofuse.fusion import comparator_mode

    rng = np.random.default_rng(seed)
    u, g = rng.normal(size=n_ent), rng.normal(size=n_ent)
    target = 0.3 * g + 0.3 * u + rng.normal(size=n_ent)
    grip = 0.3 * u + rng.normal(size=n_ent)
    grip[:12] = np.nan
    height = rng.normal(size=n_ent)
    age = rng.normal(size=n_ent)
    pid = np.repeat(np.arange(n_ent), 2)
    full = pd.DataFrame(
        {
            "polygon_id": pid,
            "target": target[pid],
            "age": age[pid],
            "grip": grip[pid],
            "height": height[pid],
            "fsa": (pid // 10).astype(str),
        }
    )
    ids = np.arange(n_ent)

    def frame(values, order=ids):
        return pd.DataFrame(
            {
                "polygon_id": ids[order],
                "target": target[order],
                "composite": values[order],
            }
        )

    bench = 0.5 * g + u + 0.3 * rng.normal(size=n_ent)
    frames = {
        "cgi": frame(g + 0.1 * u + 0.3 * rng.normal(size=n_ent)),
        comparator_mode("bench"): frame(bench, rng.permutation(n_ent)),
        "ndvi": frame(g + 0.5 * rng.normal(size=n_ent)),
    }
    eng = types.SimpleNamespace(
        negative_control_columns=list(controls),
        is_panel=False,
        covariate_columns=["age"],
        _covariate_dummy_map={},
        test_data=full[full["polygon_id"] < n_test],
        _active_greenery_channel="cgi",
    )
    eng.apply_fusion = lambda weights=None: frames[eng._active_greenery_channel]
    eng._full_data_frame = lambda: full
    for name in (
        "_reporting_covariate_matrix",
        "_reporting_raw_column",
        "_reporting_clusters",
        "_test_entity_mask",
        "compute_specificity_contrast",
    ):
        setattr(eng, name, types.MethodType(getattr(E, name), eng))
    eng.frames = frames
    eng.entity = {"target": target, "age": age, "bench": bench}
    return eng


class TestEngineContrast(unittest.TestCase):
    def test_every_exposure_and_control_on_the_held_out_rows(self):
        eng = _engine_stub()
        exposures = {"bench": ("comparator:bench", {}), "ndvi": ("ndvi", {})}
        out = eng.compute_specificity_contrast({}, exposures, n_boot=200)
        self.assertEqual(out["split"], "test")
        self.assertEqual(out["n_test"], 120)
        self.assertEqual(
            [(r["exposure"], r["control"]) for r in out["rows"]],
            [
                ("bench", "grip"),
                ("bench", "height"),
                ("ndvi", "grip"),
                ("ndvi", "height"),
            ],
        )
        grip = next(r for r in out["rows"] if r["control"] == "grip")
        self.assertEqual(grip["n"], 120 - 12)
        self.assertEqual(eng._active_greenery_channel, "cgi")
        slopes = out["slopes"]["bench"]
        self.assertEqual(slopes["target"]["n"], 120)
        self.assertEqual(slopes["controls"]["grip"]["n"], 108)

    def test_each_exposure_is_aligned_by_entity(self):
        eng = _engine_stub()
        out = eng.compute_specificity_contrast(
            {}, {"bench": ("comparator:bench", {})}, n_boot=0
        )
        e = eng.entity
        want = nc.partial_slope(e["bench"][:120], e["target"][:120], e["age"][:120])
        got = out["slopes"]["bench"]["target"]
        self.assertAlmostEqual(got["beta"], want["beta"], places=10)
        row = next(r for r in out["rows"] if r["control"] == "height")
        self.assertAlmostEqual(row["beta_target_exposure"], want["beta"], places=10)

    def test_the_cgi_against_itself_differs_by_zero(self):
        out = _engine_stub().compute_specificity_contrast(
            {}, {"same": ("cgi", {})}, n_boot=200
        )
        for r in out["rows"]:
            self.assertEqual(r["delta_d"], 0.0)
            self.assertEqual((r["ci_low"], r["ci_high"]), (0.0, 0.0))

    def test_without_controls_or_exposures_there_is_nothing(self):
        self.assertIsNone(
            _engine_stub(controls=()).compute_specificity_contrast(
                {}, {"same": ("cgi", {})}
            )
        )
        self.assertIsNone(_engine_stub().compute_specificity_contrast({}, {}))

    def test_a_panel_is_skipped(self):
        eng = _engine_stub()
        eng.is_panel = True
        out = eng.compute_specificity_contrast({}, {"same": ("cgi", {})})
        self.assertEqual(out["skipped"], "longitudinal")


if __name__ == "__main__":
    unittest.main()
