"""Tests for the fusion setup form's re-run seeding and state retention.

Covers the seeder's coverage of the recorded params, the stale-option guard
that keeps a seeded pick from outliving its column, and the pinning that stops
an aborted rerun from emptying a half-filled form.
"""

import os
import sys
import tempfile
import textwrap
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.join(ROOT, "ui"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st


class _StubState(dict):
    """Stand-in for ``st.session_state`` outside a Streamlit runtime."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc

    def __setattr__(self, k, v):
        self[k] = v


def _params(**over) -> dict:
    """A recorded fusion job, wide longitudinal intake keyed by year."""
    p = {
        "target_display_name": "wave1.gpkg",
        "target_path": "/data/wave1.gpkg",
        "outcome_columns": ["SCORE"],
        "objective_metric": "mixedlm_tstat",
        "cgi_formula": "weighted_average",
        "standalone_channels": ["veg", "terrain", "ndvi"],
        "gvi_buffer_min_m": 100.0,
        "gvi_buffer_max_m": 1000.0,
        "gvi_buffer_step_m": 100.0,
        "ndvi_buffer_min_m": 150.0,
        "ndvi_buffer_max_m": 900.0,
        "ndvi_buffer_step_m": 50.0,
        "covariate_columns": ["age", "sex"],
        "covariate_types": {"age": "numeric", "sex": "categorical"},
        "moderator_columns": ["income"],
        "longitudinal_spec_payload": {
            "scoring_metric": "mixedlm_tstat",
            "intake_mode": "wide",
            "derive_wave_from_date": True,
            "entity_id_col": "entity_id",
            "date_col": "measurement_date",
            "wave_labels": ["2015", "2018"],
            "target_files_per_wave": {
                "base": "/data/wave1.gpkg",
                "fup": "/data/wave2.gpkg",
            },
            "greenery_files": {
                "veg": {"2015": "/g/gvi_2015.tif", "2018": "/g/gvi_2018.tif"},
                "terrain": {"2015": "/g/gvi_2015.tif", "2018": "/g/gvi_2018.tif"},
                "ndvi": {"2015": "/g/ndvi_2015.tif", "2018": "/g/ndvi_2018.tif"},
            },
        },
    }
    p.update(over)
    return p


class TestSeeder(unittest.TestCase):
    def setUp(self):
        self.state = _StubState()
        st.session_state = self.state
        from tabs import fusion

        fusion.st.session_state = self.state
        self.fusion = fusion

    def test_run_mode_follows_the_scoring_metric(self):
        self.fusion._seed_fusion_form(_params())
        self.assertEqual(
            self.state["fusion_run_mode"], self.fusion._FUSION_RUN_MODE_LON
        )

    def test_cross_sectional_job_seeds_cross_mode(self):
        self.fusion._seed_fusion_form(_params(longitudinal_spec_payload=None))
        self.assertEqual(
            self.state["fusion_run_mode"], self.fusion._FUSION_RUN_MODE_CROSS
        )

    def test_standalone_studies_round_trip(self):
        self.fusion._seed_fusion_form(_params())
        self.assertTrue(self.state["fusion_run_standalones"])

        self.state.clear()
        self.fusion._seed_fusion_form(_params(standalone_channels=[]))
        self.assertFalse(self.state["fusion_run_standalones"])

    def test_buffer_ladders_seed_as_ints(self):
        self.fusion._seed_fusion_form(_params())
        for key, want in (
            ("fusion_gvi_buffer_min", 100),
            ("fusion_gvi_buffer_max", 1000),
            ("fusion_gvi_buffer_step", 100),
            ("fusion_ndvi_buffer_min", 150),
            ("fusion_ndvi_buffer_max", 900),
            ("fusion_ndvi_buffer_step", 50),
        ):
            self.assertEqual(self.state[key], want, key)
            # A float on an int-typed number_input is rejected by Streamlit.
            self.assertIsInstance(self.state[key], int, key)

    def test_wave_labels_reach_the_picker_label_widgets(self):
        from file_picker import path_to_widget_id

        self.fusion._seed_fusion_form(_params())
        self.assertEqual(
            self.state["fusion_target_paths"],
            ["/data/wave1.gpkg", "/data/wave2.gpkg"],
        )
        for wave, path in (("base", "/data/wave1.gpkg"), ("fup", "/data/wave2.gpkg")):
            key = f"fusion_target_paths__label__{path_to_widget_id(path)}"
            self.assertEqual(self.state[key], wave)

    def test_greenery_files_and_year_assignment(self):
        self.fusion._seed_fusion_form(_params())
        self.assertEqual(
            self.state["fusion_gvi_paths"], ["/g/gvi_2015.tif", "/g/gvi_2018.tif"]
        )
        self.assertEqual(
            self.state["fusion_gvi_year_assign"],
            {"/g/gvi_2015.tif": ["2015"], "/g/gvi_2018.tif": ["2018"]},
        )
        self.assertEqual(
            self.state["fusion_ndvi_year_assign"],
            {"/g/ndvi_2015.tif": ["2015"], "/g/ndvi_2018.tif": ["2018"]},
        )

    def test_assign_mode_and_categorical_split(self):
        self.fusion._seed_fusion_form(_params())
        self.assertEqual(
            self.state["fusion_lon_assign_mode"], self.fusion._FUSION_ASSIGN_BY_YEAR
        )
        # Categorical membership implies covariate membership.
        self.assertEqual(self.state["fusion_covariate_categorical"], ["sex"])
        self.assertEqual(self.state["fusion_covariate_columns"], ["age"])

    def test_a_record_missing_every_new_key_still_seeds(self):
        self.fusion._seed_fusion_form({"outcome_columns": ["SCORE"]})
        self.assertEqual(
            self.state["fusion_run_mode"], self.fusion._FUSION_RUN_MODE_CROSS
        )
        self.assertFalse(self.state["fusion_run_standalones"])

    def test_upload_signature_matches_the_picker_s_first_file(self):
        # The target picker wipes the outcome list when the first file's
        # signature differs from the one it holds, so the seeded signature has
        # to describe that file and not the recorded baseline path.
        with tempfile.TemporaryDirectory() as d:
            first = os.path.join(d, "wave1.gpkg")
            second = os.path.join(d, "wave2.gpkg")
            for path, body in ((first, b"aaaa"), (second, b"bb")):
                with open(path, "wb") as fh:
                    fh.write(body)
            p = _params(target_path=second)
            p["longitudinal_spec_payload"]["target_files_per_wave"] = {
                "base": first,
                "fup": second,
            }
            self.fusion._seed_fusion_form(p)

        self.assertEqual(self.state["fusion_target_paths"][0], first)
        self.assertEqual(
            self.state["fusion_target_upload_sig"], (first, 4)
        )
        self.assertEqual(self.state["fusion_outcome_columns"], ["SCORE"])


class TestEveryRecordedSettingIsRestored(unittest.TestCase):
    """A setting that is recorded but never seeded back is silently dropped.

    That is not a visible failure — the re-run form simply comes up without it
    and the job runs with a default. Effect modifiers hit exactly this: they
    reached ``_FUSION_RUN_CONFIG_KEYS`` and the submit payload, but nothing
    wrote them back into their widget.
    """

    def setUp(self):
        self.state = _StubState()
        from tabs import fusion

        # Restored in tearDown: replacing the module-level session_state
        # without putting it back leaks into the AppTest-based cases, which
        # need the real Streamlit one.
        self._saved = (st.session_state, fusion.st.session_state)
        st.session_state = self.state
        fusion.st.session_state = self.state
        self.fusion = fusion

    def tearDown(self):
        st.session_state, self.fusion.st.session_state = self._saved

    def test_moderator_columns_survive_a_re_run(self):
        self.fusion._seed_fusion_form(_params())
        self.assertEqual(self.state["fusion_moderator_columns"], ["income"])

    def test_moderators_are_independent_of_the_covariate_lists(self):
        """A moderator need not be a covariate, so it cannot be derived."""
        self.fusion._seed_fusion_form(_params())
        self.assertNotIn("income", self.state["fusion_covariate_columns"])
        self.assertNotIn("income", self.state["fusion_covariate_categorical"])
        self.assertIn("income", self.state["fusion_moderator_columns"])

    def test_absent_moderators_seed_an_empty_list_not_a_stale_one(self):
        self.state["fusion_moderator_columns"] = ["left", "over"]
        self.fusion._seed_fusion_form(_params(moderator_columns=[]))
        self.assertEqual(self.state["fusion_moderator_columns"], [])

    def test_recorded_config_keys_reach_a_widget_or_a_named_handler(self):
        """Every key in the recorded config must have a restore route.

        Keys handled by bespoke code in ``_seed_fusion_form`` are listed here
        explicitly; anything else has to be in the widget map, or it round-trips
        into the job record and out of the form.
        """
        handled = {
            "covariate_columns", "covariate_types", "moderator_columns",
            "standalone_channels", "longitudinal_spec_payload",
            "target_display_name", "ndvi_start_date", "ndvi_end_date",
            "ndvi_project_id", "cache_metrics", "buffer_meters",
            "ndvi_resolution_m", "gvi_grid_spacing_m", "n_spatial_blocks",
            "min_cell_count", "worst_quantile",
            # ``None`` carries meaning here (use the composite's own IQR) and
            # has to seed the widget as 0.0, which the generic map — which
            # skips ``None`` outright — cannot express.
            "exposure_iqr",
        }
        missing = [
            k for k in self.fusion._FUSION_RUN_CONFIG_KEYS
            if k not in self.fusion._FUSION_PARAM_TO_WIDGET and k not in handled
        ]
        self.assertEqual(missing, [], f"recorded but never restored: {missing}")


class TestStaleOptionGuard(unittest.TestCase):
    def setUp(self):
        self.state = _StubState()
        st.session_state = self.state
        from tabs import fusion

        fusion.st.session_state = self.state
        self.keep_valid = fusion._keep_valid
        self.default = fusion._default

    def test_stale_pick_is_replaced_by_the_default_not_the_first_option(self):
        # Pruning has to happen before the default fills in, or a dropped pick
        # leaves the widget on options[0] instead of its intended default.
        self.state["fusion_weight_bin_pct"] = 7
        self.assertEqual(
            self.default("fusion_weight_bin_pct", 10, [5, 10, 20, 25, 50]), 10
        )

    def test_a_valid_pick_survives_the_default(self):
        self.state["fusion_weight_bin_pct"] = 25
        self.assertEqual(
            self.default("fusion_weight_bin_pct", 10, [5, 10, 20, 25, 50]), 25
        )

    def test_an_emptied_multiselect_is_left_empty(self):
        # Deselecting everything is a choice; the default must not refill it.
        self.state["fusion_map_picks"] = []
        self.assertEqual(
            self.default("fusion_map_picks", ["CGI"], ["CGI", "NDVI"], multi=True), []
        )

    def test_single_pick_outside_options_is_dropped(self):
        self.state["fusion_lon_date_col"] = "visit_date"
        self.keep_valid("fusion_lon_date_col", ["other_date"])
        self.assertNotIn("fusion_lon_date_col", self.state)

    def test_single_pick_inside_options_is_kept(self):
        self.state["fusion_lon_date_col"] = "visit_date"
        self.keep_valid("fusion_lon_date_col", ["visit_date", "other_date"])
        self.assertEqual(self.state["fusion_lon_date_col"], "visit_date")

    def test_multi_pick_is_pruned_not_cleared(self):
        self.state["fusion_covariate_columns"] = ["age", "sex", "gone"]
        self.keep_valid("fusion_covariate_columns", ["age", "sex"], multi=True)
        self.assertEqual(self.state["fusion_covariate_columns"], ["age", "sex"])

    def test_absent_key_is_left_alone(self):
        self.keep_valid("fusion_lon_date_col", ["a"])
        self.assertNotIn("fusion_lon_date_col", self.state)


class TestPinSkips(unittest.TestCase):
    def test_widget_state_is_pinned_but_buttons_are_not(self):
        from helpers import pin_skips_key

        for key in (
            "fusion_run_mode",
            "fusion_gvi_paths",
            "fusion_gvi_year_assign",
            "fusion_target_paths__label__abc123",
        ):
            self.assertFalse(pin_skips_key(key), key)
        # Streamlit refuses API assignment to button and component keys.
        for key in (
            "fusion_form_run_submit",
            "fusion_results_clear",
            "fusion_gvi_paths__btn",
            "fusion_target_paths__rm__abc123",
            "fusion_gvi_sort_9912",
        ):
            self.assertTrue(pin_skips_key(key), key)


class TestFormSurvivesAbortedRerun(unittest.TestCase):
    """The form must not empty itself when a run aborts before it renders."""

    APP = textwrap.dedent(
        """
        import os, sys
        ROOT = {root!r}
        for p in (os.path.join(ROOT, "ui"), ROOT):
            if p not in sys.path:
                sys.path.insert(0, p)
        import streamlit as st
        from helpers import pin_fusion_form_state

        if os.environ.get("GF_PIN") == "1":
            pin_fusion_form_state()

        st.checkbox("break an earlier tab", key="boom")
        if st.session_state.get("boom"):
            raise RuntimeError("earlier tab failed while loading data")

        st.radio("Run mode", ("Cross-sectional", "Mixed-effects (longitudinal)"),
                 index=None, key="fusion_run_mode")
        st.selectbox("Objective", ["mixedlm_tstat", "mixedlm_coef"],
                     key="fusion_objective_metric")
        """
    )

    def _run(self, pin: str):
        from streamlit.testing.v1 import AppTest

        os.environ["GF_PIN"] = pin
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "app.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.APP.format(root=ROOT))
            at = AppTest.from_file(path, default_timeout=60)
            at.run()
            at.radio(key="fusion_run_mode").set_value(
                "Mixed-effects (longitudinal)"
            )
            at.run()
            at.selectbox(key="fusion_objective_metric").set_value("mixedlm_coef")
            at.run()
            at.checkbox(key="boom").check()
            at.run()  # aborts before the form
            at.session_state["boom"] = False
            at.run()  # recover
            return at

    def test_pinned_state_survives(self):
        at = self._run("1")
        self.assertEqual(
            at.session_state["fusion_run_mode"], "Mixed-effects (longitudinal)"
        )
        self.assertEqual(
            at.session_state["fusion_objective_metric"], "mixedlm_coef"
        )

    def test_without_the_pin_the_form_is_lost(self):
        at = self._run("0")
        self.assertIsNone(at.session_state["fusion_run_mode"])


if __name__ == "__main__":
    unittest.main()
