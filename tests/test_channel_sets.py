"""The composite path has to speak whichever channel set the job selected.

Two channel sets are selectable, and ``ndvi + gvi`` is the default: a
two-channel study built on the combined green-view channel, and a three-channel
one built on ``veg`` and ``terrain`` separately. The search reads the set off
the formula; scoring, apply and the map stages used to assume the three-channel
names, so a default-configuration run reached the held-out score and died there.

These tests pin the vocabulary-dependent seams — which channels a job has, where
each one's radius and statistic come from, which of them has a source layer, and
how the composite is assembled — for both sets and for every standalone mode.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from geofuse import cgi_formulas
from geofuse import fusion_helpers as _helpers
from geofuse.fusion import MetricFusionEngine

GVI_FORMULAS = ("weighted_average_gvi", "synergy_gvi")
THREE_FORMULAS = ("weighted_average", "synergy")

PARAMS = {
    "gvi_radius": 300,
    "veg_radius": 250,
    "terrain_radius": 250,
    "ndvi_radius": 800,
    "streetview_stat": "percentile",
    "streetview_percentile": 75,
    "ndvi_stat": "mean",
    "ndvi_percentile": 50,
}


def _engine(formula: str, mode: str = "cgi") -> MetricFusionEngine:
    """An engine carrying only the attributes the channel seams read.

    Built without ``__init__`` on purpose: these are pure methods over the
    formula and the params, and standing up a real engine would need a study
    area, metric layers and a populated cache.
    """
    e = object.__new__(MetricFusionEngine)
    e.cgi_formula = formula
    e._active_greenery_channel = mode
    e.gvi_buffer_max_m = 500.0
    e.ndvi_buffer_max_m = 1000.0
    e.normalize_channels = False
    e._channel_minmax = None
    e.veg_data = "VEG-LAYER"
    e.terrain_data = "TERRAIN-LAYER"
    e.ndvi_data = "NDVI-LAYER"
    return e


def _weighted(formula: str) -> dict:
    params = dict(PARAMS)
    for key in cgi_formulas.get_formula(formula).weight_keys:
        params[key] = 50
    return params


class TestAJobKnowsItsOwnChannels(unittest.TestCase):
    def test_the_two_channel_set_is_ndvi_and_gvi(self):
        for formula in GVI_FORMULAS:
            with self.subTest(formula=formula):
                self.assertEqual(
                    _engine(formula)._composite_channels(), ("ndvi", "gvi"))

    def test_the_three_channel_set_keeps_veg_and_terrain_apart(self):
        for formula in THREE_FORMULAS:
            with self.subTest(formula=formula):
                self.assertEqual(
                    set(_engine(formula)._composite_channels()),
                    {"ndvi", "veg", "terrain"},
                )

    def test_a_standalone_job_has_exactly_its_own_channel(self):
        for mode in ("veg", "terrain", "ndvi", "gvi"):
            with self.subTest(mode=mode):
                e = _engine("weighted_average_gvi", mode=mode)
                self.assertEqual(e._composite_channels(), (mode,))
                self.assertEqual(e._channel_active({}), {mode: True})


class TestEveryChannelResolvesItsAggregation(unittest.TestCase):
    def test_ndvi_carries_its_own_statistic(self):
        e = _engine("weighted_average_gvi")
        self.assertEqual(e._channel_spec(PARAMS, "ndvi"), (800, "mean", 50))

    def test_the_street_view_channels_share_one_statistic(self):
        e = _engine("weighted_average_gvi")
        for ch, radius in (("gvi", 300), ("veg", 250), ("terrain", 250)):
            with self.subTest(channel=ch):
                self.assertEqual(
                    e._channel_spec(PARAMS, ch), (radius, "percentile", 75))

    def test_a_missing_radius_falls_back_to_the_ladder_maximum(self):
        e = _engine("weighted_average_gvi")
        self.assertEqual(e._channel_spec({}, "gvi")[0], 500)
        self.assertEqual(e._channel_spec({}, "ndvi")[0], 1000)

    def test_gvi_has_no_source_layer(self):
        # It is written during pre-aggregation and only ever read back, so
        # there is nothing to fall back to on a cache miss.
        e = _engine("weighted_average_gvi")
        self.assertIsNone(e._channel_source("gvi"))
        self.assertEqual(e._channel_source("veg"), "VEG-LAYER")
        self.assertEqual(e._channel_source("ndvi"), "NDVI-LAYER")

    def test_a_cache_only_channel_reports_a_miss_instead_of_aggregating(self):
        e = _engine("weighted_average_gvi")
        e._preaggregation_done = False
        with self.assertRaises(RuntimeError) as caught:
            e._aggregate_with_ring_cache(
                None, None, 300, "mean", 50,
                channel="gvi", fold_idx=-1, subset="test",
            )
        self.assertIn("gvi", str(caught.exception))


class TestTheCompositeIsBuiltFromThatSet(unittest.TestCase):
    def test_every_formula_builds_from_its_own_channels(self):
        for formula in GVI_FORMULAS + THREE_FORMULAS:
            with self.subTest(formula=formula):
                e = _engine(formula)
                params = _weighted(formula)
                blocks = {
                    c: np.linspace(0.1, 0.9, 5) for c in e._composite_channels()
                }
                out = e._composite_from(params, blocks)
                self.assertEqual(len(out), 5)
                self.assertTrue(np.all(np.isfinite(out)))

    def test_a_standalone_composite_is_its_channel_untouched(self):
        for mode in ("veg", "terrain", "ndvi", "gvi"):
            with self.subTest(mode=mode):
                e = _engine("weighted_average_gvi", mode=mode)
                values = np.arange(4.0)
                self.assertTrue(
                    np.array_equal(e._composite_from({}, {mode: values}), values))

    def test_normalization_is_keyed_by_channel_name(self):
        e = _engine("weighted_average_gvi")
        e.normalize_channels = True
        e._channel_minmax = {"ndvi": (0.0, 2.0), "gvi": (0.0, 4.0)}
        out = e._normalize_channels(
            {"ndvi": np.array([1.0]), "gvi": np.array([1.0])})
        self.assertAlmostEqual(float(out["ndvi"][0]), 0.5)
        self.assertAlmostEqual(float(out["gvi"][0]), 0.25)


class TestGviGetsRealNormalizationBounds(unittest.TestCase):
    """``gvi`` is not a column of the prepared frame — it is the components' sum.

    Reading it straight off the frame finds nothing and falls back to (0, 1),
    which silently clips the channel instead of scaling it.
    """

    FRAME = {"veg": [0.1, 0.2, 0.3],
             "terrain": [0.05, 0.05, 0.1],
             "ndvi": [0.4, 0.5, 0.6]}

    def _frame(self):
        import pandas as pd
        return pd.DataFrame(self.FRAME)

    def test_gvi_is_read_as_the_component_sum(self):
        col = MetricFusionEngine._channel_column(self._frame(), "gvi")
        self.assertEqual(
            [round(float(v), 3) for v in col], [0.15, 0.25, 0.4])

    def test_a_channel_with_no_column_and_no_rule_is_absent(self):
        import pandas as pd
        self.assertIsNone(
            MetricFusionEngine._channel_column(pd.DataFrame({"veg": [1.0]}), "gvi"))

    def test_the_bounds_are_derived_rather_than_defaulted(self):
        e = object.__new__(MetricFusionEngine)
        e.normalize_channels = True
        e._compute_channel_scale(self._frame())
        self.assertNotEqual(e._channel_minmax["gvi"], (0.0, 1.0))
        lo, hi = e._channel_minmax["gvi"]
        self.assertGreater(hi, lo)

    def test_the_toggle_still_switches_it_all_off(self):
        e = object.__new__(MetricFusionEngine)
        e.normalize_channels = False
        e._compute_channel_scale(self._frame())
        self.assertIsNone(e._channel_minmax)


class TestTheCatchmentComesFromTheJobsRadii(unittest.TestCase):
    def test_a_standalone_job_uses_its_own_channels_radius(self):
        self.assertEqual(
            _helpers.catchment_radius({"ndvi": 800.0, "gvi": 300.0}, "gvi"), 300.0)

    def test_a_combined_job_uses_the_largest(self):
        self.assertEqual(
            _helpers.catchment_radius({"ndvi": 800.0, "gvi": 300.0}, "cgi"), 800.0)

    def test_the_radii_map_covers_the_two_channel_set(self):
        e = _engine("weighted_average_gvi")
        self.assertEqual(
            e._channel_radii(PARAMS), {"ndvi": 800.0, "gvi": 300.0})

    def test_no_radii_is_not_a_crash(self):
        self.assertEqual(_helpers.catchment_radius({}, "cgi"), 0.0)


if __name__ == "__main__":
    unittest.main()
