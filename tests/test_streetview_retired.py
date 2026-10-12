"""A retired Street View search fails a GVI run instead of reading as no coverage.

Google answers a retired search with a status and a message; that answer
raises ``ServiceRetired`` (a plain "no images here" stays ``None``), and a GVI
run whose searches all meet it stops with the reason rather than finishing
with no outputs.
"""

import asyncio
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geofuse  # noqa: F401  (import order: torch before GeoPandas)
import geopandas as gpd
from shapely.geometry import Point

from geofuse import streetview as sv
from geofuse.gvi import GVIEngine

RETIRED = [
    [
        [
            5,
            "generic",
            "GeoPhotoService.SingleImageSearch is decommissioned and turned down.",
        ]
    ]
]
NOTHING_HERE = [[[5, "generic", "Search returned no images."]]]


class RetiredSearchTests(unittest.TestCase):
    def test_a_retired_search_is_not_no_coverage(self):
        with self.assertRaisesRegex(sv.ServiceRetired, "decommissioned"):
            sv._parse_radius_response(RETIRED)
        self.assertIsNone(sv._parse_radius_response(NOTHING_HERE))
        self.assertIsNone(sv._parse_radius_response([]))

    def test_a_gvi_run_stops_with_the_reason(self):
        engine = GVIEngine.__new__(GVIEngine)
        gdf = gpd.GeoDataFrame(
            geometry=[Point(-114.0, 51.0), Point(-114.001, 51.0)], crs="EPSG:4326"
        )
        points = [
            {"orig_index": i, "lat": 51.0, "lon": geom.x, "geometry": geom,
             "row": 0, "col": i}
            for i, geom in enumerate(gdf.geometry)
        ]
        retired = mock.AsyncMock(side_effect=sv.ServiceRetired("turned down"))
        with mock.patch.object(sv, "find_panorama_async", retired):
            with self.assertRaisesRegex(sv.ServiceRetired, "turned down"):
                asyncio.run(
                    engine._run_analysis_async(
                        points, 2, 0, gdf, "out", False, False, {}, None, None,
                        None, None, None,
                    )
                )


if __name__ == "__main__":
    unittest.main()
