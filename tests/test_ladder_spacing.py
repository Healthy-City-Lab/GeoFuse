"""Buffer ladders can be log-spaced as well as linear.

A log-spaced ladder places a chosen number of rungs geometrically between the
bounds, so every rung covers the same relative distance. Linear ladders, the
default, keep their radii and their study fingerprints exactly.
"""

import math
import os
import sys
import tempfile
import textwrap
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.join(ROOT, "ui"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from geofuse import bayesian_index as bi
from geofuse import metric_sampling as ms
from geofuse.fusion import MetricFusionEngine
from geofuse.jobs import runners


class LadderRadiiTests(unittest.TestCase):
    def test_linear_is_the_old_range(self):
        lo, hi, st = ms.radius_int_bounds(50, 2000, 50)
        self.assertEqual(ms.ladder_radii(50, 2000, 50), tuple(range(lo, hi + 1, st)))
        self.assertEqual(ms.ladder_radii(100, 1000, 300), (100, 400, 700, 1000))

    def test_log_ladder_is_geometric(self):
        r = np.asarray(ms.ladder_radii(50, 2000, 50, spacing="log", rungs=15), float)
        self.assertEqual(len(r), 15)
        self.assertEqual((r[0], r[-1]), (50, 2000))
        ratios = r[1:] / r[:-1]
        np.testing.assert_allclose(ratios, (2000 / 50) ** (1 / 14), rtol=0.03)
        # G10 acceptance (2): the rung-based prior centre is the geometric mean.
        centre = math.exp(bi._radius_prior(r)[0])
        self.assertAlmostEqual(centre, math.sqrt(50 * 2000), delta=0.01 * 316)

    def test_rounding_duplicates_are_dropped(self):
        r = ms.ladder_radii(1, 6, 10, spacing="log", rungs=30)
        self.assertEqual(r, tuple(sorted(set(r))))
        self.assertLess(len(r), 30)

    def test_unknown_spacing_is_refused(self):
        with self.assertRaises(ValueError):
            ms.ladder_radii(50, 500, 50, spacing="quadratic")


class EngineLadderTests(unittest.TestCase):
    def _engine(self, **over):
        eng = MetricFusionEngine.__new__(MetricFusionEngine)
        cfg = dict(
            gvi_buffer_min_m=20.0,
            gvi_buffer_max_m=500.0,
            gvi_buffer_step_m=20.0,
            ndvi_buffer_min_m=50.0,
            ndvi_buffer_max_m=2000.0,
            ndvi_buffer_step_m=50.0,
            gvi_ladder_spacing="linear",
            ndvi_ladder_spacing="linear",
            gvi_ladder_rungs=15,
            ndvi_ladder_rungs=15,
        )
        cfg.update(over)
        eng.__dict__.update(cfg)
        return eng

    def test_default_ladders_are_unchanged(self):
        gvi, ndvi = self._engine()._preaggr_radii()
        self.assertEqual(gvi, tuple(range(20, 501, 20)))
        self.assertEqual(ndvi, tuple(range(50, 2001, 50)))

    def test_each_family_takes_its_own_spacing(self):
        eng = self._engine(ndvi_ladder_spacing="log", ndvi_ladder_rungs=12)
        gvi, ndvi = eng._preaggr_radii()
        self.assertEqual(gvi, tuple(range(20, 501, 20)))
        self.assertEqual(ndvi, ms.ladder_radii(50, 2000, 50, spacing="log", rungs=12))
        np.testing.assert_array_equal(eng._outer_radii_metres(gvi=False), ndvi)
        self.assertEqual(eng._ladder_description("ndvi"), "log (12 rungs)")
        self.assertEqual(eng._ladder_description("gvi"), "linear (step 20 m)")


class FingerprintTests(unittest.TestCase):
    BASE = dict(
        buffer_meters=2000.0,
        gvi_buffer_min_m=20.0,
        gvi_buffer_max_m=500.0,
        gvi_buffer_step_m=20.0,
        ndvi_buffer_min_m=50.0,
        ndvi_buffer_max_m=2000.0,
        ndvi_buffer_step_m=50.0,
        cgi_formula="weighted_average_gvi",
        covariate_columns=["age"],
        whole_grid_scaling=True,
        cgi_grid_spacing_m=50,
        area_balanced_split=True,
        test_size=0.2,
    )

    def test_linear_ladders_keep_their_hash(self):
        a = runners._fusion_config_fingerprint(**self.BASE)
        b = runners._fusion_config_fingerprint(
            **self.BASE, gvi_ladder_spacing="linear", ndvi_ladder_rungs=30
        )
        self.assertEqual(a, b)

    def test_a_log_ladder_splits_the_cache(self):
        a = runners._fusion_config_fingerprint(**self.BASE)
        b = runners._fusion_config_fingerprint(**self.BASE, ndvi_ladder_spacing="log")
        c = runners._fusion_config_fingerprint(
            **self.BASE, ndvi_ladder_spacing="log", ndvi_ladder_rungs=20
        )
        self.assertEqual(len({a, b, c}), 3)


class LadderWidgetTests(unittest.TestCase):
    """The ladder row: spacing switches which of step / rungs is live."""

    APP = textwrap.dedent(
        """
        import os, sys
        ROOT = {root!r}
        for p in (os.path.join(ROOT, "ui"), ROOT):
            if p not in sys.path:
                sys.path.insert(0, p)
        import streamlit as st
        from tabs import fusion
        st.session_state["_ladder_state"] = fusion._render_metric_assignment_panel(
            [], year_aware=False
        )
        """
    )

    def test_spacing_toggles_step_and_rungs(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "app.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.APP.format(root=ROOT))
            at = AppTest.from_file(path, default_timeout=60)
            at.run()
            self.assertFalse(at.exception)
            spacing = at.selectbox(key="fusion_ndvi_ladder_spacing")
            self.assertEqual(spacing.label, "NDVI ladder spacing")
            self.assertEqual(spacing.value, "linear")
            self.assertFalse(at.number_input(key="fusion_ndvi_buffer_step").disabled)
            self.assertTrue(at.number_input(key="fusion_ndvi_ladder_rungs").disabled)
            self.assertIn("10 radii: 100, 200, 300", " ".join(c.value for c in at.caption))

            spacing.set_value("log")
            at.number_input(key="fusion_ndvi_ladder_rungs").set_value(5)
            at.run()
            self.assertTrue(at.number_input(key="fusion_ndvi_buffer_step").disabled)
            self.assertFalse(at.number_input(key="fusion_ndvi_ladder_rungs").disabled)
            self.assertIn("5 radii: 100, 178, 316, 562, 1000 m", [c.value for c in at.caption])
            # The GVI row keeps its own settings.
            self.assertEqual(at.selectbox(key="fusion_gvi_ladder_spacing").value, "linear")


class RadiusPriorWidgetTests(unittest.TestCase):
    """The study-details panel offers the radius prior and returns it."""

    APP = textwrap.dedent(
        """
        import os, sys
        ROOT = {root!r}
        for p in (os.path.join(ROOT, "ui"), ROOT):
            if p not in sys.path:
                sys.path.insert(0, p)
        import streamlit as st
        from tabs import fusion
        state = fusion._render_study_details_panel(
            False, ["age"], is_polygon_target=False, is_vector_target=True
        )
        st.session_state["_study_state"] = state
        """
    )

    def test_default_and_switch(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "app.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.APP.format(root=ROOT))
            at = AppTest.from_file(path, default_timeout=60)
            at.run()
            self.assertFalse(at.exception)
            prior = at.selectbox(key="fusion_radius_prior")
            self.assertEqual(prior.label, "Radius prior")
            self.assertEqual(prior.value, "rungs")
            self.assertEqual(
                prior.options, ["Rung-based", "Range-based (spacing-invariant)"]
            )
            self.assertEqual(at.session_state["_study_state"]["radius_prior"], "rungs")
            prior.set_value("range")
            at.run()
            self.assertEqual(at.session_state["_study_state"]["radius_prior"], "range")


if __name__ == "__main__":
    unittest.main()
