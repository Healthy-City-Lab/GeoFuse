"""The automatic spatial block size follows where the entities are.

Dividing the bounding box gives a handful of huge blocks when the entities sit
in a few clusters across a large extent — a national sample of cities — and the
held-out set then becomes a few whole regions. The automatic edge now comes
from the occupied extent. Pinned here:

- ten tight clusters across a continent-sized box get close to the target
  number of occupied blocks, and a striped test set close to the asked share;
- one uniform region keeps exactly the blocks the bounding-box rule gave;
- an explicit block size keeps its meaning;
- a split whose blocks miss the asked test share, or are too few, says so and
  records the blocking it used.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas as gpd
import numpy as np

from geofuse import fusion, fusion_helpers
from geofuse.crs_utils import assign_spatial_blocks


def _points(x, y):
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs="EPSG:3978")


def _clusters(n_per=300, seed=0):
    """Ten towns of uniform density, 10 km across, in a 4,000 km box."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, 4_000_000, size=(10, 2))
    radius = 5_000 * np.sqrt(rng.uniform(size=(10, n_per)))
    angle = rng.uniform(0, 2 * np.pi, size=(10, n_per))
    x = (centres[:, :1] + radius * np.cos(angle)).ravel()
    y = (centres[:, 1:] + radius * np.sin(angle)).ravel()
    return _points(x, y)


def _bounding_box_rule(gdf, target):
    """The previous rule: the box's edge, binned the same way."""
    x, y = gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()
    s = np.sqrt(max((x.max() - x.min()) * (y.max() - y.min()), 1.0) / target)
    ix = np.floor((x - x.min()) / s).astype(int)
    iy = np.floor((y - y.min()) / s).astype(int)
    return np.unique(iy * (ix.max() + 1) + ix, return_inverse=True)[1]


class TestAutomaticBlockSize(unittest.TestCase):
    def test_clustered_entities_get_about_the_target_number_of_blocks(self):
        gdf = _clusters()
        target = 200
        old = _bounding_box_rule(gdf, target)
        self.assertLess(np.unique(old).size, 0.3 * target)
        blocks, info = assign_spatial_blocks(gdf, target_blocks=target)
        self.assertEqual(info["rule"], "occupied extent")
        self.assertGreaterEqual(info["n_blocks"], 0.8 * target)

        test_size = 0.25
        test = fusion_helpers.stripe_to_test_blocks(
            np.unique(blocks), test_size, np.random.default_rng(42)
        )
        share = float(np.isin(blocks, list(test)).mean())
        self.assertLess(abs(share - test_size), 0.05)

    def test_a_uniform_region_keeps_the_bounding_box_blocks(self):
        rng = np.random.default_rng(1)
        gdf = _points(rng.uniform(0, 30_000, 3000), rng.uniform(0, 20_000, 3000))
        blocks, info = assign_spatial_blocks(gdf, target_blocks=200)
        self.assertEqual(info["rule"], "bounding box")
        np.testing.assert_array_equal(blocks, _bounding_box_rule(gdf, 200))

    def test_a_given_size_is_used_as_given(self):
        _blocks, info = assign_spatial_blocks(_clusters(), block_size_m=5_000)
        self.assertEqual((info["block_size_m"], info["rule"]), (5_000.0, "given"))


class TestSplitReportsItsBlocks(unittest.TestCase):
    def test_too_few_blocks_are_reported_and_recorded(self):
        from geofuse.fusion import MetricFusionEngine
        from test_fusion_run import _write_study

        folder = tempfile.mkdtemp(prefix="geofuse-blocks-")
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
        df = eng.prepare_fusion_data()
        eng.precompute_aggregations()
        with mock.patch.object(fusion, "_log") as log:
            eng.split_data(
                test_size=0.25,
                random_state=42,
                fusion_df=df,
                spatial_split=True,
                spatial_block_size_m=1_500,
            )
        report = eng._spatial_split_report
        self.assertEqual(report["block_size_m"], 1_500.0)
        self.assertLess(report["occupied_blocks"], 20)
        self.assertEqual(report["test_size"], 0.25)
        n_test = eng.test_data["polygon_id"].nunique()
        n_all = n_test + eng.train_val_data["polygon_id"].nunique()
        self.assertAlmostEqual(report["test_share"], n_test / n_all)
        warnings = [c.args[1] for c in log.call_args_list if c.args[0] == "WARN"]
        self.assertTrue(any("occupied blocks" in m for m in warnings), warnings)


if __name__ == "__main__":
    unittest.main()
