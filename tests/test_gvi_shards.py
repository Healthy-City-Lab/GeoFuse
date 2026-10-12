"""A GVI run split into shards and merged (``geofuse.jobs.gvi_shards``).

Shards partition the sampling points in spatial blocks, the same way every
time; merging the shards of a run writes the GeoPackage, sidecar and tiles an
unsharded run writes (per year in column mode) and removes the shard files;
a merge with a shard missing names it.
"""

import json
import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geofuse  # noqa: F401  (import order: torch before GeoPandas)
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point, box

from geofuse.core import generate_clustered_grid, shard_mask
from geofuse.jobs import gvi_shards, runners

SHARDS = 3
AREA = box(-114.10, 51.02, -114.05, 51.06)


class _Ctx:
    def progress(self, *_a, **_kw):
        pass

    def heartbeat(self):
        pass

    def is_cancelled(self):
        return False

    def is_paused(self):
        return False


class _Engine:
    """Samples every point from its grid position; every seventh has none."""

    def run_analysis(self, gdf, *, result_callback, target_year=None, **_kw):
        for idx, pt in gdf.iterrows():
            row, col = int(pt.get("row", 0)), int(pt.get("col", 0))
            sampled = (row + col) % 7 != 0
            month = f"{target_year or 2021}-0{1 + col % 9}"
            result_callback(
                {
                    "orig_index": int(idx),
                    "geometry": pt.geometry,
                    "gvi_veg": round(0.01 * ((row * 31 + col) % 90), 4),
                    "gvi_ter": 0.05,
                    "pano_id": f"p{row}_{col}",
                    "pano_date": month,
                    "lat": pt.geometry.y,
                    "lon": pt.geometry.x,
                    "row": row,
                    "col": col,
                    "cluster_id": int(pt.get("cluster_id", 0)),
                }
                if sampled
                else {
                    "orig_index": int(idx),
                    "geometry": pt.geometry,
                    "gvi_veg": None,
                    "gvi_ter": None,
                    "pano_id": None,
                    "pano_date": None,
                    "lat": pt.geometry.y,
                    "lon": pt.geometry.x,
                    "row": row,
                    "col": col,
                    "cluster_id": int(pt.get("cluster_id", 0)),
                }
            )


def _gvi(runner, raw, out, *, shard=None, **kw):
    run_args = {"step": 50, "buffer": kw.pop("buffer", 0)}
    run_args.update(save_panos=False, save_masks=False)
    if shard is not None:
        run_args["shard"] = shard
    data = {"raw": raw, "type": kw.pop("gtype", "poly"), "processed": None}
    data.update(meta=None, accumulated=[], cache_ref={})
    kw.update(init_args={"model_path": "m"}, gpu_lock=threading.Lock())
    return runner(
        _Ctx(),
        fname="site.gpkg",
        dataset_data=data,
        run_args=run_args,
        output_dir=out,
        save_geotiff=True,
        save_geojson=False,
        **kw,
    )


def _merge(raw, out, **kw):
    return gvi_shards.run_gvi_merge(
        _Ctx(),
        fname="site.gpkg",
        dataset_data={"raw": raw, "type": kw.pop("gtype", "poly")},
        run_args={"step": 50, "buffer": kw.pop("buffer", 0)},
        output_dir=out,
        shards=SHARDS,
        save_geotiff=True,
        **kw,
    )


def _table(path):
    frame = gpd.read_file(path, layer="gvi_samples")
    frame = frame.sort_values(["cluster_id", "row", "col"]).reset_index(drop=True)
    return frame[[c for c in frame.columns if c != "geometry"]], frame.geometry


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.whole = os.path.join(self.tmp.name, "whole")
        self.split = os.path.join(self.tmp.name, "split")
        os.makedirs(self.whole)
        os.makedirs(self.split)
        patch = mock.patch.object(runners, "_get_gvi_engine", return_value=_Engine())
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def assert_same_outputs(self, a_dir, b_dir, name):
        a_tab, a_geo = _table(os.path.join(a_dir, f"{name}_gvi.gpkg"))
        b_tab, b_geo = _table(os.path.join(b_dir, f"{name}_gvi.gpkg"))
        self.assertEqual(len(a_tab), len(b_tab))
        self.assertTrue(a_tab.equals(b_tab))
        self.assertTrue(a_geo.geom_equals_exact(b_geo, 1e-9).all())
        with open(os.path.join(a_dir, f"{name}_gvi.json")) as f:
            a_side = json.load(f)
        with open(os.path.join(b_dir, f"{name}_gvi.json")) as f:
            b_side = json.load(f)
        self.assertEqual(a_side, b_side)
        a_tiles = os.path.join(a_dir, f"{name}_gvi_tiles")
        b_tiles = os.path.join(b_dir, f"{name}_gvi_tiles")
        tiles = sorted(os.listdir(a_tiles))
        self.assertEqual(tiles, sorted(os.listdir(b_tiles)))
        for tile in (t for t in tiles if t.endswith(".tif")):
            with rasterio.open(os.path.join(a_tiles, tile)) as a:
                with rasterio.open(os.path.join(b_tiles, tile)) as b:
                    np.testing.assert_array_equal(a.read(), b.read())


class ShardMaskTests(unittest.TestCase):
    def test_shards_partition_the_points_in_blocks(self):
        raw = gpd.GeoDataFrame(geometry=[AREA], crs="EPSG:4326")
        points, _meta = generate_clustered_grid(raw, buffer_m=0.0, step_m=50.0)
        masks = [shard_mask(points, i, SHARDS) for i in range(SHARDS)]
        self.assertTrue((np.sum(masks, axis=0) == 1).all())
        self.assertTrue(all(m.any() for m in masks))
        np.testing.assert_array_equal(masks[1], shard_mask(points, 1, SHARDS))
        rows, cols = points["row"] // 20, points["col"] // 20
        block = rows.astype(str) + ":" + cols.astype(str)
        for mask in masks:
            self.assertFalse(set(block[mask]) & set(block[~mask]))
        with self.assertRaises(ValueError):
            shard_mask(points, SHARDS, SHARDS)


class MergeTests(_Case):
    def test_a_sharded_run_writes_what_a_whole_run_writes(self):
        raw = gpd.GeoDataFrame(geometry=[AREA], crs="EPSG:4326")
        _gvi(runners.run_gvi, raw, self.whole)
        for i in range(SHARDS):
            out = _gvi(runners.run_gvi, raw, self.split, shard=[i, SHARDS])
            self.assertTrue(any(p.endswith(".json") for p in out["output_paths"]))
        merged = _merge(raw, self.split)
        self.assertIn(os.path.join(self.split, "site_gvi.gpkg"), merged["output_paths"])
        self.assert_same_outputs(self.whole, self.split, "site")
        self.assertFalse([n for n in os.listdir(self.split) if ".shard" in n])

    def test_column_mode_merges_each_year(self):
        points = [Point(-114.08 + 0.01 * i, 51.03 + 0.005 * i) for i in range(6)]
        years = {"year": [2019, 2019, 2020, 2020, 2021, 2021]}
        raw = gpd.GeoDataFrame(years, geometry=points, crs="EPSG:4326")
        kw = {"gtype": "point", "buffer": 300, "date_column": "year"}
        _gvi(runners.run_gvi_column, raw, self.whole, **kw)
        for i in range(SHARDS):
            _gvi(runners.run_gvi_column, raw, self.split, shard=[i, SHARDS], **kw)
        _merge(raw, self.split, **kw)
        whole = os.path.join(self.whole, "site_temporal_gvi")
        split = os.path.join(self.split, "site_temporal_gvi")
        for year in (2019, 2020, 2021):
            self.assert_same_outputs(whole, split, f"site_{year}")
        self.assertFalse([n for n in os.listdir(split) if ".shard" in n])

    def test_a_missing_shard_is_named(self):
        raw = gpd.GeoDataFrame(geometry=[AREA], crs="EPSG:4326")
        _gvi(runners.run_gvi, raw, self.split, shard=[0, SHARDS])
        with self.assertRaisesRegex(gvi_shards.ShardsMissing, "shard2of3, shard3of3"):
            _merge(raw, self.split)


if __name__ == "__main__":
    unittest.main()
