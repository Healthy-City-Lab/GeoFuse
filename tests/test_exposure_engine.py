"""Vector targets take their exposure from the CGI cells their geometry touches.

One small study, built once: overlapping polygon targets on a synthetic NDVI
raster and street-view points. Pinned here:

- every entity keeps every cell it shares area with, also where two entities
  overlap, and each (entity, cell) row carries its exposure weight;
- the applied composite is the weighted mean of per-cell composites, and for a
  linear composite it equals the same combination of the index tensor — the
  object the discovery engine fits — entity for entity.
"""

import os
import sys
import tempfile
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from geofuse.fusion import MetricFusionEngine

PARAMS = {
    "veg_weight": 30,
    "terrain_weight": 30,
    "ndvi_weight": 40,
    "veg_radius": 400,
    "terrain_radius": 400,
    "ndvi_radius": 400,
    "veg_stat": "mean",
    "veg_percentile": 50,
    "terrain_stat": "mean",
    "terrain_percentile": 50,
    "ndvi_stat": "mean",
    "ndvi_percentile": 50,
}


class TestVectorExposure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2)
        tmp = tempfile.mkdtemp()
        x0, y0 = -114.10, 51.00
        step = 0.004
        polys, vals = [], []
        for i in range(6):
            for j in range(5):
                x, y = x0 + i * step, y0 + j * step
                # Neighbours overlap by a quarter of a cell's width.
                polys.append(box(x, y, x + 1.25 * step, y + step))
                vals.append(float(rng.normal()))
        tgt = gpd.GeoDataFrame({"outcome": vals}, geometry=polys, crs="EPSG:4326")
        tpath = os.path.join(tmp, "t.gpkg")
        tgt.to_file(tpath, driver="GPKG")

        npath = os.path.join(tmp, "n.tif")
        with rasterio.open(
            npath,
            "w",
            driver="GTiff",
            height=160,
            width=160,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(x0 - 0.01, y0 + 0.04, 0.0004, 0.0004),
        ) as d:
            d.write(rng.random((160, 160)).astype("float32"), 1)
        gpts = [
            Point(x0 - 0.005 + rng.random() * 0.04, y0 - 0.005 + rng.random() * 0.03)
            for _ in range(600)
        ]
        gvi = gpd.GeoDataFrame(
            {"gvi_veg": rng.random(600), "gvi_ter": rng.random(600)},
            geometry=gpts,
            crs="EPSG:4326",
        )
        gpath = os.path.join(tmp, "g.gpkg")
        gvi.to_file(gpath, driver="GPKG")

        eng = MetricFusionEngine(
            target_file=tpath,
            target_feature="outcome",
            buffer_meters=400,
            gvi_buffer_min_m=200,
            gvi_buffer_max_m=400,
            gvi_buffer_step_m=200,
            ndvi_buffer_min_m=200,
            ndvi_buffer_max_m=400,
            ndvi_buffer_step_m=200,
            cache_dir=os.path.join(tmp, "cache"),
            cgi_grid_spacing_m=100,
        )
        eng.load_target()
        eng.load_metrics(veg_file=gpath, ndvi_file=npath, cache_metrics=False)
        cls.df = eng.prepare_fusion_data()
        eng.precompute_aggregations()
        eng.split_data(test_size=0.3, random_state=5, fusion_df=cls.df)
        cls.eng = eng

    def test_overlapping_entities_each_keep_their_shared_cells(self):
        tg = self.eng.target_gdf
        per_cell = tg.groupby("_global_pid")["polygon_id"].nunique()
        self.assertGreater(int((per_cell > 1).sum()), 0)
        self.assertIn("_weight", self.df.columns)
        self.assertTrue((self.df["_weight"] > 0).all())

    def test_a_linear_composite_is_the_same_combination_of_the_tensor(self):
        eng = self.eng
        X, radii, stats, channels, static = eng.build_index_tensor("train_val")
        ri = list(np.asarray(radii, dtype=int)).index(400)
        si = list(stats).index("mean")
        total = 100.0
        expected = sum(
            PARAMS[f"{ch}_weight"] / total * X[:, channels.index(ch), ri, si]
            for ch in ("veg", "terrain", "ndvi")
        )
        applied = eng.apply_fusion(weights=dict(PARAMS)).set_index("polygon_id")
        got = applied.loc[static["pid"], "composite"].to_numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
