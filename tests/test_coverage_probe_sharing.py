"""Veg and terrain are two columns of one street-view layer, so the coverage
probe's nearest-feature join should run once for both.

The join is the expensive half and depends only on geometry, so sharing it is
free — but only if it returns exactly what two separate joins returned, ties
included. These tests pin that, and pin the cases where the layers must *not*
be shared.
"""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from geofuse import metric_sampling  # noqa: E402
from geofuse.fusion import MetricFusionEngine  # noqa: E402

CRS = "EPSG:32617"


def gvi_layers(n=400, seed=0, same_geometry=True):
    """Two frames read from one file: identical geometry, different columns."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 4000, (n, 2))
    veg = gpd.GeoDataFrame(
        {"gvi_veg": rng.random(n)},
        geometry=[Point(x, y) for x, y in xy],
        crs=CRS,
    )
    veg.attrs["metric_column"] = "gvi_veg"
    ter_xy = xy if same_geometry else rng.uniform(0, 4000, (n, 2))
    terrain = gpd.GeoDataFrame(
        {"gvi_ter": rng.random(n)},
        geometry=[Point(x, y) for x, y in ter_xy],
        crs=CRS,
    )
    terrain.attrs["metric_column"] = "gvi_ter"
    return veg, terrain


def probe_points(n=250, seed=1):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 4000, (n, 2))
    return gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in xy], crs=CRS, index=range(100, 100 + n)
    )


class TestSharedJoinMatchesSeparateJoins(unittest.TestCase):
    def test_values_are_identical(self):
        veg, terrain = gvi_layers()
        points = probe_points()
        separate_veg = metric_sampling.nearest_metric_join(
            points, veg, "gvi_veg", 1000.0
        )
        separate_ter = metric_sampling.nearest_metric_join(
            points, terrain, "gvi_ter", 1000.0
        )

        shared = MetricFusionEngine._shared_gvi_layer(veg, terrain)
        self.assertIsNotNone(shared)
        layer, cols = shared
        joined = metric_sampling.nearest_metric_join_multi(
            points, layer, [cols["veg"], cols["terrain"]], 1000.0
        )
        np.testing.assert_array_equal(
            separate_veg.to_numpy(), joined["gvi_veg"].to_numpy()
        )
        np.testing.assert_array_equal(
            separate_ter.to_numpy(), joined["gvi_ter"].to_numpy()
        )

    def test_identical_on_an_exact_tie_grid(self):
        # A regular grid with query points equidistant from several sources is
        # where a tie-break difference would show up.
        gx, gy = np.meshgrid(np.arange(0, 500, 50.0), np.arange(0, 500, 50.0))
        pts = [Point(x, y) for x, y in zip(gx.ravel(), gy.ravel())]
        rng = np.random.default_rng(7)
        veg = gpd.GeoDataFrame({"gvi_veg": rng.random(len(pts))}, geometry=pts, crs=CRS)
        veg.attrs["metric_column"] = "gvi_veg"
        terrain = gpd.GeoDataFrame(
            {"gvi_ter": rng.random(len(pts))}, geometry=pts, crs=CRS
        )
        terrain.attrs["metric_column"] = "gvi_ter"
        # Query exactly at cell centres — equidistant from four sources.
        qx, qy = np.meshgrid(np.arange(25, 475, 50.0), np.arange(25, 475, 50.0))
        query = gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in zip(qx.ravel(), qy.ravel())], crs=CRS
        )
        sep_v = metric_sampling.nearest_metric_join(query, veg, "gvi_veg", 1000.0)
        sep_t = metric_sampling.nearest_metric_join(query, terrain, "gvi_ter", 1000.0)
        layer, cols = MetricFusionEngine._shared_gvi_layer(veg, terrain)
        joined = metric_sampling.nearest_metric_join_multi(
            query, layer, [cols["veg"], cols["terrain"]], 1000.0
        )
        np.testing.assert_array_equal(sep_v.to_numpy(), joined["gvi_veg"].to_numpy())
        np.testing.assert_array_equal(sep_t.to_numpy(), joined["gvi_ter"].to_numpy())

    def test_out_of_range_points_stay_nan(self):
        veg, terrain = gvi_layers()
        far = gpd.GeoDataFrame(geometry=[Point(9e6, 9e6)], crs=CRS)
        layer, cols = MetricFusionEngine._shared_gvi_layer(veg, terrain)
        joined = metric_sampling.nearest_metric_join_multi(
            far, layer, [cols["veg"], cols["terrain"]], 10.0
        )
        self.assertTrue(np.isnan(joined["gvi_veg"].to_numpy()).all())
        self.assertTrue(np.isnan(joined["gvi_ter"].to_numpy()).all())


class TestSampledFrameIsUnchanged(unittest.TestCase):
    """End to end: sharing the join must not change a single sampled column.

    The per-channel loop blanks each column as it reaches it, so a value written
    during another channel's turn is silently wiped — every channel has to end
    up populated, not just the one the join was keyed on.
    """

    def _engine(self, veg, terrain):
        engine = MetricFusionEngine.__new__(MetricFusionEngine)
        engine.veg_data = veg
        engine.terrain_data = terrain
        engine.ndvi_data = None
        engine.gvi_buffer_max_m = 1000.0
        engine.ndvi_buffer_max_m = 1000.0
        return engine

    def test_shared_and_unshared_agree_on_every_channel(self):
        veg, terrain = gvi_layers()
        points = probe_points()

        shared = self._engine(veg, terrain)._sample_metrics_at_points(
            points.copy(), quiet=True
        )

        # Force the unshared path by perturbing nothing but the shared check.
        engine = self._engine(veg, terrain)
        engine._shared_gvi_layer = staticmethod(lambda *_a, **_k: None)
        unshared = engine._sample_metrics_at_points(points.copy(), quiet=True)

        for channel in ("veg", "terrain"):
            got = shared[channel].to_numpy()
            self.assertTrue(np.isfinite(got).any(), f"{channel} came back entirely NaN")
            np.testing.assert_array_equal(
                unshared[channel].to_numpy(), got, err_msg=channel
            )

    def test_terrain_survives_the_channel_loop(self):
        veg, terrain = gvi_layers()
        out = self._engine(veg, terrain)._sample_metrics_at_points(
            probe_points().copy(), quiet=True
        )
        self.assertTrue(np.isfinite(out["terrain"].to_numpy()).any())
        self.assertTrue(np.isfinite(out["veg"].to_numpy()).any())


class TestSharingIsRefusedWhenUnsafe(unittest.TestCase):
    def test_different_geometry_is_not_shared(self):
        veg, terrain = gvi_layers(same_geometry=False)
        self.assertIsNone(MetricFusionEngine._shared_gvi_layer(veg, terrain))

    def test_a_raster_channel_is_not_shared(self):
        veg, _ = gvi_layers()
        raster = {"data": np.zeros((4, 4)), "transform": None, "crs": CRS}
        self.assertIsNone(MetricFusionEngine._shared_gvi_layer(veg, raster))
        self.assertIsNone(MetricFusionEngine._shared_gvi_layer(raster, veg))

    def test_mismatched_lengths_are_not_shared(self):
        veg, terrain = gvi_layers()
        self.assertIsNone(
            MetricFusionEngine._shared_gvi_layer(veg, terrain.iloc[:-1].copy())
        )

    def test_missing_source_is_not_shared(self):
        veg, _ = gvi_layers()
        self.assertIsNone(MetricFusionEngine._shared_gvi_layer(veg, None))
        self.assertIsNone(MetricFusionEngine._shared_gvi_layer(None, None))


if __name__ == "__main__":
    unittest.main()
