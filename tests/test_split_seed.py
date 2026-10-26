"""The held-out split's seed is a setting.

The test split used to be drawn with a fixed seed. Pinned here:

- the default seed is that fixed seed, so a default run holds out the same
  rows and keeps its cache fingerprint;
- a run file's seed reaches the run, and another seed changes the fingerprint
  and the held-out rows.
"""

import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas as gpd  # noqa: F401

from geofuse import runfile, settings
from geofuse.fusion import MetricFusionEngine
from geofuse.jobs import plans, runners


def _fingerprint(**extra) -> str:
    return runners._fusion_config_fingerprint(
        buffer_meters=300.0,
        gvi_buffer_min_m=100.0,
        gvi_buffer_max_m=300.0,
        gvi_buffer_step_m=100.0,
        ndvi_buffer_min_m=100.0,
        ndvi_buffer_max_m=300.0,
        ndvi_buffer_step_m=100.0,
        cgi_formula="weighted_average",
        covariate_columns=["age"],
        whole_grid_scaling=True,
        cgi_grid_spacing_m=50,
        area_balanced_split=False,
        test_size=0.25,
        **extra,
    )


class TestSplitSeedSetting(unittest.TestCase):
    def test_the_default_is_the_previous_fixed_seed(self):
        self.assertEqual(settings.defaults("fusion")["split_seed"], 42)

    def test_only_another_seed_changes_the_fingerprint(self):
        self.assertEqual(_fingerprint(), _fingerprint(split_seed=42))
        self.assertNotEqual(_fingerprint(), _fingerprint(split_seed=7))

    def test_a_run_file_seed_reaches_the_run(self):
        from test_fusion_run import _write_study

        folder = tempfile.mkdtemp(prefix="geofuse-seed-")
        _write_study(folder)
        path = os.path.join(folder, "run.toml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                f"output_dir = {os.path.join(folder, 'out')!r}\n"
                "[[fusion]]\n"
                'name = "t"\n'
                'target = "cohort.gpkg"\n'
                'outcomes = ["outcome"]\n'
                'mode = "cross-sectional"\n'
                'ndvi = "ndvi_2015.tif"\n'
                'gvi = "gvi.gpkg"\n'
                "split_seed = 7\n"
            )
        (spec,) = plans.expand(runfile.load(path, ()))
        self.assertEqual(plans.build(spec).kwargs["run_kwargs"]["split_seed"], 7)


class TestEngineSplitSeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_fusion_run import _write_study

        folder = tempfile.mkdtemp(prefix="geofuse-seed-")
        _write_study(folder)
        eng = MetricFusionEngine(
            target_file=os.path.join(folder, "cohort.gpkg"),
            target_feature="outcome",
            buffer_meters=300,
            gvi_buffer_min_m=100,
            gvi_buffer_max_m=300,
            gvi_buffer_step_m=100,
            ndvi_buffer_min_m=100,
            ndvi_buffer_max_m=300,
            ndvi_buffer_step_m=100,
            cache_dir=os.path.join(folder, "cache"),
            cgi_grid_spacing_m=100,
        )
        eng.load_target()
        eng.load_metrics(
            veg_file=os.path.join(folder, "gvi.gpkg"),
            ndvi_file=os.path.join(folder, "ndvi_2015.tif"),
            cache_metrics=False,
        )
        cls.df = eng.prepare_fusion_data()
        eng.precompute_aggregations()
        cls.eng = eng

    def _test_ids(self, **kwargs) -> set:
        self.eng.split_data(test_size=0.25, fusion_df=self.df, **kwargs)
        return set(self.eng.test_data["polygon_id"])

    def test_the_default_seed_holds_out_the_same_rows(self):
        self.assertEqual(self._test_ids(), self._test_ids(random_state=42))

    def test_another_seed_holds_out_other_rows(self):
        self.assertNotEqual(self._test_ids(), self._test_ids(random_state=7))


if __name__ == "__main__":
    unittest.main()
