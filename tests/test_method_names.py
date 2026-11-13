"""Spatial adjustment methods named by what they compute.

Pinned here:

- the older spellings map to what they computed: ``ks_aic`` is the basis whose
  df the outcome model picks, a ``spatial_plus`` recorded before the select
  setting existed the basis whose df the exposure model picks, and either one
  with a pattern scale the fixed-scale basis; a ``basis`` without a select
  gets ``fixed_scale`` with a scale and ``df_outcome_ic`` without;
- under the older and the new spelling the engine reports the same numbers;
- a spec recorded before the change builds the run it described;
- a new run file that sets ``spatial_plus`` with a basis-only setting is
  refused, naming the spelling to use;
- records and labels say ``basis`` and how its size was set, never
  ``spatial_plus`` for a basis run;
- the search fingerprint of a basis run keeps its earlier hash.
"""

import os
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas  # noqa: F401  (pyproj before torchvision, see vision.py)
import numpy as np

from geofuse import runfile, settings
from geofuse.fusion import SPATIAL_LABELS, MetricFusionEngine
from geofuse.jobs import outer_splits, plans, runners

import test_spatial_reporting as reporting_fixture  # noqa: E402


class MappingTests(unittest.TestCase):
    def test_the_older_spellings(self):
        pair = settings.spatial_method_pair
        self.assertEqual(pair("ks_aic", None), ("basis", "df_outcome_ic"))
        self.assertEqual(pair("spatial_plus", None), ("basis", "df_exposure_ic"))
        self.assertEqual(
            pair("spatial_plus", None, scales=[5000]), ("basis", "fixed_scale")
        )
        self.assertEqual(
            pair("ks_aic", None, tune_scale=5000), ("basis", "fixed_scale")
        )
        self.assertEqual(pair("basis", None), ("basis", "df_outcome_ic"))
        self.assertEqual(pair("basis", None, scales=[5000]), ("basis", "fixed_scale"))

    def test_the_new_spellings_pass_through(self):
        pair = settings.spatial_method_pair
        self.assertEqual(pair("spatial_plus", "df_outcome_ic"), ("spatial_plus", None))
        self.assertEqual(pair("basis", "scale_ic"), ("basis", "scale_ic"))
        self.assertEqual(pair("none", None), ("none", None))

    def test_a_settings_table_maps_ks_aic(self):
        out = settings.check("fusion", {"spatial_adjust_method": "ks_aic"})
        self.assertEqual(out["spatial_adjust_method"], "basis")
        self.assertEqual(out["spatial_adjust_select"], "df_outcome_ic")

    def test_the_engine_maps_ks_aic(self):
        eng = MetricFusionEngine(
            target_file="dummy.geojson",
            cache_dir=tempfile.mkdtemp(),
            spatial_adjust_method="ks_aic",
        )
        self.assertEqual(
            (eng.spatial_adjust_method, eng.spatial_adjust_select, eng._basis),
            ("basis", "df_outcome_ic", "outcome"),
        )

    def test_a_df_rule_with_a_scale_is_refused(self):
        with self.assertRaises(ValueError):
            MetricFusionEngine(
                target_file="dummy.geojson",
                cache_dir=tempfile.mkdtemp(),
                spatial_adjust_method="basis",
                spatial_adjust_select="df_exposure_ic",
                spatial_adjust_scale_m=[5000.0],
            )
        with self.assertRaises(settings.SettingError):
            settings.check(
                "fusion",
                {
                    "spatial_adjust_method": "basis",
                    "spatial_adjust_select": "df_outcome_ic",
                    "spatial_adjust_scale_m": [5000],
                },
            )


class SameNumbersTests(unittest.TestCase):
    """The engine's reports under the older spelling (through the mapping) and
    the new one are the same numbers."""

    def _report(self, eng):
        slopes = eng.compute_partial_slopes({})["test"]
        controls = eng.compute_negative_controls({}, n_boot=40)["splits"]["test"]
        return slopes, controls

    def test_every_df_rule(self):
        for old, rule in (("ks_aic", "outcome"), ("spatial_plus", "exposure")):
            with self.subTest(old):
                method, select = settings.spatial_method_pair(old, None)
                mapped = reporting_fixture._engine(rule)
                new = reporting_fixture._engine(
                    {"df_outcome_ic": "outcome", "df_exposure_ic": "exposure"}[select]
                )
                a, b = self._report(mapped), self._report(new)
                for key in ("beta", "ci_low", "ci_high", "t"):
                    self.assertEqual(a[0][key], b[0][key], key)
                ga, gb = a[1]["controls"]["grip"], b[1]["controls"]["grip"]
                for key in ("beta", "ci_low", "ci_high"):
                    self.assertEqual(ga[key], gb[key], key)
                self.assertEqual(method, "basis")


def _spec(settings_table: dict) -> plans.JobSpec:
    return plans.JobSpec(
        step="study",
        kind="fusion",
        name="study",
        output_dir="out",
        sources=[],
        merged=False,
        group=None,
        options={
            "vector": True,
            "polygon": False,
            "mode": "cross-sectional",
            "target": "cohort.gpkg",
            "covariates": [],
            "categorical": [],
            "negative_controls": [],
            "outcomes": ["y"],
            "moderators": [],
            "comparators": [],
            "comparators_primary": [],
        },
        settings=settings_table,
        preset=None,
    )


class RecordedSpecTests(unittest.TestCase):
    """A spec recorded before the select setting existed lacks every new key;
    its method maps to what it computed."""

    def _recorded(self, **changes):
        table = settings.defaults("fusion")
        for key in (
            "spatial_adjust_select",
            "spatial_tune_method",
            "spatial_tune_select",
            "spatial_plus_groups_column",
            "spatial_plus_k",
            "spatial_plus_criterion",
            "spatial_plus_checks",
            "spatial_plus_step1_covariates",
            "spatial_select_scales_m",
            "spatial_select_criterion",
            "spatial_select_controls",
        ):
            table.pop(key)
        table.update(changes)
        return plans.fusion_run_config(_spec(table))

    def test_df_mode_spatial_plus(self):
        config = self._recorded(spatial_adjust_method="spatial_plus")
        self.assertEqual(config["spatial_adjust_method"], "basis")
        self.assertEqual(config["spatial_adjust_select"], "df_exposure_ic")
        self.assertEqual(
            config["spatial_adjust_df"], list(settings.SPATIAL_DF_CANDIDATES)
        )

    def test_scale_mode_spatial_plus(self):
        config = self._recorded(
            spatial_adjust_method="spatial_plus", spatial_adjust_scale_m=[5000, 20000]
        )
        self.assertEqual(config["spatial_adjust_select"], "fixed_scale")
        self.assertEqual(config["spatial_adjust_scale_m"], [5000, 20000])

    def test_ks_aic(self):
        config = self._recorded(spatial_adjust_method="ks_aic")
        self.assertEqual(
            (config["spatial_adjust_method"], config["spatial_adjust_select"]),
            ("basis", "df_outcome_ic"),
        )

    def test_the_runner_maps_an_older_job(self):
        method, select = settings.spatial_method_pair("spatial_plus", None)
        self.assertEqual(
            runners._fingerprint_spelling(method, select, "basis"), "spatial_plus"
        )
        self.assertEqual(
            runners._fingerprint_spelling("basis", "df_outcome_ic", "basis"), "ks_aic"
        )
        # Spatial+ itself reaches only the reports, so the search keeps the
        # hash of an unadjusted run.
        self.assertEqual(
            runners._fingerprint_spelling("spatial_plus", None, "basis"), "none"
        )


class RefusalTests(unittest.TestCase):
    def _load(self, body: str):
        with tempfile.TemporaryDirectory() as d:
            for name in ("cohort.gpkg", "ndvi.tif", "gvi.gpkg"):
                open(os.path.join(d, name), "w").close()
            path = os.path.join(d, "run.toml")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(textwrap.dedent(body))
            return runfile.load(path)

    def test_spatial_plus_with_a_scale_is_refused(self):
        with self.assertRaises(runfile.RunFileError) as caught:
            self._load(
                """
                [[fusion]]
                name = "study"
                target = "cohort.gpkg"
                outcomes = ["y"]
                ndvi = "ndvi.tif"
                gvi = "gvi.gpkg"
                spatial_adjust_method = "spatial_plus"
                spatial_adjust_scale_m = [5000]
                """
            )
        text = str(caught.exception)
        self.assertIn("spatial_adjust_method = 'basis'", text)
        self.assertIn("spatial_adjust_select = 'fixed_scale'", text)

    def test_spatial_plus_with_a_df_list_is_refused(self):
        with self.assertRaises(runfile.RunFileError) as caught:
            self._load(
                """
                [[fusion]]
                name = "study"
                target = "cohort.gpkg"
                outcomes = ["y"]
                ndvi = "ndvi.tif"
                gvi = "gvi.gpkg"
                spatial_adjust_method = "spatial_plus"
                spatial_adjust_df = [0, 2, 4]
                """
            )
        self.assertIn("spatial_adjust_select = 'df_exposure_ic'", str(caught.exception))


class LabelTests(unittest.TestCase):
    def test_records_name_the_basis_and_its_size_rule(self):
        eng = reporting_fixture._engine("exposure")
        record = eng.compute_partial_slopes({})["test"]["spatial_adjustment"]
        self.assertEqual(record["method"], "basis")
        self.assertEqual(record["select"], "df_exposure_ic")
        columns = outer_splits.spatial_columns(record)
        self.assertEqual(columns["spatial_method"], "basis")
        self.assertNotIn("spatial_plus", " ".join(map(str, columns.values())))

    def test_the_labels_say_what_was_computed(self):
        for select in settings.SPATIAL_ADJUST_SELECTS:
            label = outer_splits.method_label({"method": "basis", "select": select})
            self.assertEqual(label, SPATIAL_LABELS[select])
            ref = "Rainey" if select == "df_exposure_ic" else "Keller & Szpiro"
            self.assertIn(ref, label)
        self.assertIn(
            "Dupont", outer_splits.method_label({"method": "spatial_plus"})
        )
        # A record from before the rename keeps its own name.
        self.assertEqual(outer_splits.method_label({"method": "ks_aic"}), "ks_aic")


if __name__ == "__main__":
    unittest.main()
