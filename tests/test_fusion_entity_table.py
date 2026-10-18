"""A fusion run writes its per-entity CGI table to disk.

The table — sampled channels and the composite, one row per target entity —
used to exist only in the live engine, which stays in the job's own process.
``write_entity_table`` joins it to the target's geometry: polygons through
``polygon_id``, a target with one row per composite row column-wise, and
without geometry otherwise (CSV).
"""

import os
import sys
import tempfile
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from geofuse.jobs.fusion_outputs import write_entity_table


def _engine(polygons=None, target=None):
    return types.SimpleNamespace(target_polygons_gdf=polygons, target_gdf=target)


class EntityTableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stem = os.path.join(self.tmp.name, "composite_entities")

    def tearDown(self):
        self.tmp.cleanup()

    def test_polygons_join_on_their_id(self):
        polys = gpd.GeoDataFrame(
            {"name": ["a", "b"]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
            crs="EPSG:4326",
        )
        composite = pd.DataFrame(
            {"polygon_id": [1, 0], "target": [3.0, 4.0], "composite": [0.2, 0.8]}
        )
        path = write_entity_table(_engine(polygons=polys), composite, self.stem)
        self.assertTrue(path.endswith(".gpkg"))
        out = gpd.read_file(path, layer="cgi").sort_values("polygon_id")
        self.assertEqual(list(out["composite"]), [0.8, 0.2])
        self.assertEqual(out.geometry.iloc[1].bounds, (1.0, 0.0, 2.0, 1.0))

    def test_row_aligned_targets_take_the_columns(self):
        target = gpd.GeoDataFrame(
            {"id": [7, 8]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326"
        )
        composite = pd.DataFrame({"target": [1.0, 2.0], "composite": [0.1, 0.9]})
        path = write_entity_table(_engine(target=target), composite, self.stem)
        out = gpd.read_file(path, layer="cgi")
        self.assertEqual(list(out["id"]), [7, 8])
        self.assertEqual(list(out["composite"]), [0.1, 0.9])

    def test_without_geometry_it_is_a_csv(self):
        composite = pd.DataFrame({"target": [1.0], "composite": [0.5]})
        path = write_entity_table(_engine(), composite, self.stem)
        self.assertTrue(path.endswith(".csv"))
        self.assertEqual(list(pd.read_csv(path)["composite"]), [0.5])

    def test_each_entity_names_its_split(self):
        # The engine's held-out rows, so outside checks use the same entities.
        composite = pd.DataFrame(
            {"polygon_id": [0, 1, 2, 3], "target": [1.0] * 4, "composite": [0.5] * 4}
        )
        engine = _engine()
        engine.test_data = pd.DataFrame({"polygon_id": [1, 1, 3]})
        engine.train_val_data = pd.DataFrame({"polygon_id": [0, 0]})
        path = write_entity_table(engine, composite, self.stem)
        out = pd.read_csv(path).sort_values("polygon_id")
        self.assertEqual(list(out["split"]), ["train_val", "test", "dropped", "test"])


if __name__ == "__main__":
    unittest.main()
