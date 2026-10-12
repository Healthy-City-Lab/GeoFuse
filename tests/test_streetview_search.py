"""Finding the panorama nearest a point without an API key (``streetview``).

The search reads Google's coverage tiles (every panorama in a map tile) and
asks for the nearest one's metadata: the nearest panorama of the wanted type
within the radius is chosen, tiles around the circle's edge are read too,
concurrent searches share each tile request, an unknown id reads as no
panorama, and an answer in an unexpected form raises ``ServiceRetired`` so
a GVI run stops with the reason instead of finishing empty.
"""

import asyncio
import json
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

LAT, LON = 51.0453, -114.0631
GUARD = ")]}'\n"


def _tile(*panos):
    """A coverage tile answer holding ``(type, id, lat, lon)`` panoramas."""
    entries = [
        [[[kind, pid], None, [[None, None, lat, lon], [1000.0], [0.0, 90.0, 0.0]]], []]
        for kind, pid, lat, lon in panos
    ]
    return GUARD + json.dumps([None, [None, entries]])


def _metadata(pid, status=1):
    if status != 1:
        return GUARD + json.dumps([[], [[[status], [2, pid]]]])
    sizes = [[[256, 512]], [[8192, 16384]]]
    message = [
        [1],
        [2, pid],
        [None, None, None, [sizes, [512, 512]]],
        None,
        None,
        [[None, [[None, None, LAT, LON]]]],
        [None, None, None, None, None, None, None, [2024, 6]],
    ]
    return GUARD + json.dumps([[], [message]])


class _Response:
    def __init__(self, text, status=200):
        self.text_body, self.status = text, status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        await asyncio.sleep(0.01)
        return self.text_body


class _Session:
    """Answers coverage tiles and metadata; counts the requests per kind."""

    def __init__(self, tile_text, metadata=None):
        self.tile_text, self.metadata = tile_text, metadata or _metadata
        self.calls = {"tile": 0, "metadata": 0}

    def get(self, url, headers=None):
        if "/ac/v1" in url:
            self.calls["tile"] += 1
            return _Response(self.tile_text)
        self.calls["metadata"] += 1
        return _Response(self.metadata(url.rsplit("2s", 1)[-1].split("!")[0]))


class _Case(unittest.TestCase):
    def setUp(self):
        sv._TILES.clear()
        self.addCleanup(sv._TILES.clear)

    def find(self, session, *, lat=LAT, lon=LON, radius=50.0, third_party=False):
        return asyncio.run(
            sv.find_panorama_async(
                lat, lon, session, radius=radius, search_third_party=third_party
            )
        )


class SearchTests(_Case):
    def test_the_nearest_panorama_within_the_radius(self):
        session = _Session(
            _tile(
                (2, "far", LAT + 0.0003, LON),  # about 33 m
                (2, "near", LAT + 0.0001, LON),  # about 11 m
                (10, "sphere", LAT, LON),  # a user photosphere on the spot
            )
        )
        pano = self.find(session)
        self.assertEqual((pano.id, pano.date), ("near", (2024, 6)))
        self.assertEqual(pano.image_sizes[-1], sv.Size(16384, 8192))
        self.assertEqual(self.find(session, third_party=True).id, "sphere")
        self.assertIsNone(self.find(session, radius=5.0))

    def test_nothing_near_or_an_unknown_id_is_no_panorama(self):
        self.assertIsNone(self.find(_Session(_tile((2, "x", LAT + 0.01, LON)))))
        ocean = _Session(GUARD + json.dumps([None, None, []]))
        self.assertIsNone(self.find(ocean))
        gone = _Session(_tile((2, "x", LAT, LON)), lambda pid: _metadata(pid, 2))
        self.assertIsNone(self.find(gone))

    def test_concurrent_searches_share_each_tile(self):
        session = _Session(_tile((2, "p", LAT, LON)))

        async def many():
            return await asyncio.gather(
                *(
                    sv.find_panorama_async(LAT, LON + 0.00001 * i, session)
                    for i in range(8)
                )
            )

        found = asyncio.run(many())
        self.assertEqual({p.id for p in found}, {"p"})
        self.assertEqual(session.calls["tile"], len(sv._tiles_around(LAT, LON, 50)))
        self.assertEqual(session.calls["metadata"], 8)
        self.find(session)
        self.assertEqual(session.calls["tile"], len(sv._tiles_around(LAT, LON, 50)))

    def test_a_circle_over_a_tile_edge_reads_both_tiles(self):
        x, y = sv._tile_xy(LAT, LON)
        tiles = sv._tiles_around(LAT, LON, 2000.0)
        self.assertIn((x, y), tiles)
        self.assertGreater(len(tiles), 1)

    def test_throttling_is_not_no_coverage(self):
        session = _Session(_tile((2, "p", LAT, LON)))
        session.get = lambda url, headers=None: _Response("", status=429)
        with self.assertRaises(sv.RateLimitedError):
            self.find(session)


class RetiredSearchTests(_Case):
    def test_an_unexpected_answer_is_not_no_coverage(self):
        for body in ("<html>moved</html>", GUARD + "{not json", GUARD + '"text"'):
            sv._TILES.clear()
            with self.assertRaises(sv.ServiceRetired):
                self.find(_Session(body))

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
