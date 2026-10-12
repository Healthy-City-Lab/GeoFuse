"""One schema per engine for every user-facing setting (``geofuse.settings``).

The schemas are the single record of each setting's type, bounds, choices and
default. These tests hold their choice lists equal to the engine constants
they mirror, their NDVI / GVI defaults equal to the built-in "Default"
presets, the fusion schema complete against the run options the fusion tab
records, and the value checks strict, with the setting named.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.join(ROOT, "ui"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from geofuse import presets, settings
from geofuse.settings import SettingError

# Run options the fusion tab records that are inputs or derived, not settings.
_FUSION_INPUTS = {
    "buffer_meters",
    "ndvi_resolution_m",
    "gvi_grid_spacing_m",
    "cache_metrics",
    "ndvi_start_date",
    "ndvi_end_date",
    "ndvi_project_id",
    "target_display_name",
    "cgi_formula",
    "covariate_columns",
    "covariate_types",
    "moderator_columns",
    "negative_controls",
    "standalone_channels",
    "longitudinal_spec_payload",
    "n_spatial_blocks",
}


class EngineConstantTests(unittest.TestCase):
    def test_choice_lists_equal_the_engines(self):
        from geofuse import bayesian_index, metric_sampling, ndvi
        from geofuse.jobs import runners
        from geofuse.longitudinal import MIXEDLM_METRICS

        self.assertEqual(settings.TEMPORAL_REDUCERS, ndvi.TEMPORAL_REDUCERS)
        self.assertEqual(settings.SATELLITES, ndvi.SATELLITES)
        self.assertEqual(settings.REFLECTANCES, ndvi.REFLECTANCES)
        self.assertEqual(settings.CLOUD_HANDLINGS, ndvi.CLOUD_HANDLINGS)
        self.assertEqual(settings.LADDER_SPACINGS, metric_sampling.LADDER_SPACINGS)
        self.assertEqual(settings.RADIUS_PRIORS, bayesian_index.RADIUS_PRIORS)
        self.assertEqual(settings.INDEX_FORMS[1:], bayesian_index.FORMS)
        self.assertEqual(settings.CHANNEL_SETS, runners._CHANNEL_SETS)
        n = len(MIXEDLM_METRICS)
        self.assertEqual(settings.LONGITUDINAL_METRICS[:n], MIXEDLM_METRICS)

    def test_the_fusion_tab_offers_the_same_metrics(self):
        from tabs import fusion

        self.assertEqual(settings.CROSS_SECTIONAL_METRICS, fusion._CROSS_METRICS)
        self.assertEqual(settings.LONGITUDINAL_METRICS, fusion._MIXEDLM_METRICS)
        self.assertEqual(settings.BINARY_ONLY_METRICS, fusion._BINARY_ONLY_METRICS)


class DefaultTests(unittest.TestCase):
    def test_ndvi_and_gvi_defaults_are_the_default_presets(self):
        for engine in presets.ENGINES:
            preset = presets.load_settings(engine, "Default")
            defaults = settings.defaults(engine)
            self.assertEqual({k: defaults[k] for k in preset}, preset, engine)
            missing = set(defaults) - set(preset) - {"target_year"}
            self.assertEqual(missing, set(), engine)

    def test_every_recorded_fusion_option_is_a_setting_or_an_input(self):
        from tabs import fusion

        recorded = set(fusion._FUSION_RUN_CONFIG_KEYS)
        self.assertEqual(recorded - set(settings.FUSION) - _FUSION_INPUTS, set())

    def test_defaults_are_valid_and_lists_are_copied(self):
        for engine, schema in settings.SCHEMAS.items():
            values = settings.defaults(engine)
            self.assertEqual(settings.check(engine, values), values, engine)
        a = settings.defaults("ndvi")["statistics_ranges"]
        a.append("max")
        self.assertEqual(settings.defaults("ndvi")["statistics_ranges"], ["median"])

    def test_presets_validate_against_the_same_schema(self):
        self.assertIs(presets.SCHEMAS["ndvi"], settings.NDVI)
        self.assertIs(presets.SCHEMAS["gvi"], settings.GVI)
        self.assertTrue(issubclass(presets.PresetError, SettingError))


class CheckTests(unittest.TestCase):
    def check(self, engine, **values):
        return settings.check(engine, values)

    def test_values_are_stored_as_their_type(self):
        out = self.check("fusion", test_size=0.3, vif_threshold=5, sweep_splits=40)
        self.assertEqual(
            out, {"test_size": 0.3, "vif_threshold": 5.0, "sweep_splits": 40}
        )
        self.assertIsInstance(out["vif_threshold"], float)

    def test_bad_values_name_the_setting(self):
        for engine, values, key in (
            ("ndvi", {"cloud_max": 120}, "cloud_max"),
            ("ndvi", {"cloud_max": 1.5}, "cloud_max"),
            ("ndvi", {"water_mask": 1}, "water_mask"),
            ("ndvi", {"statistics_column": ["p90"]}, "statistics_column"),
            ("gvi", {"month_policy": "keep"}, "month_policy"),
            ("fusion", {"test_size": 0.9}, "test_size"),
            ("fusion", {"test_size": "0.2"}, "test_size"),
            ("fusion", {"cgi_grid_spacing_m": 55}, "cgi_grid_spacing_m"),
            ("fusion", {"objective_metric": "pearson"}, "objective_metric"),
            ("fusion", {"check_collinearity": None}, "check_collinearity"),
        ):
            with self.assertRaisesRegex(SettingError, f"'{key}'"):
                settings.check(engine, values)

    def test_unknown_settings_are_refused(self):
        msg = "Unknown FUSION setting 'sweep_split'"
        with self.assertRaisesRegex(SettingError, msg):
            self.check("fusion", sweep_split=20)

    def test_zero_means_automatic_where_the_tabs_say_so(self):
        out = self.check(
            "fusion", spatial_block_size_m=0, exposure_iqr=0, spatial_adjust_eps_m=500
        )
        self.assertEqual(
            out,
            {
                "spatial_block_size_m": None,
                "exposure_iqr": None,
                "spatial_adjust_eps_m": 500.0,
            },
        )

    def test_optional_settings_accept_none(self):
        self.assertEqual(self.check("gvi", target_year=None), {"target_year": None})
        self.assertEqual(
            self.check("fusion", posterior_warmup=None), {"posterior_warmup": None}
        )

    def test_lists_follow_the_schema_order_and_seasons_run_forwards(self):
        self.assertEqual(
            self.check("ndvi", statistics_ranges=["max", "median"]),
            {"statistics_ranges": ["median", "max"]},
        )
        with self.assertRaisesRegex(SettingError, "season_start_month"):
            self.check("ndvi", season_start_month=10, season_end_month=4)


if __name__ == "__main__":
    unittest.main()
