"""The keyed Street View tile API path (``streetview`` / ``gvi``).

Against a local stand-in for Google's tile API: one session token serves
concurrent searches; metadata becomes a panorama with its tiling, date and one
capture; misses, indoor imagery and user photospheres are no panorama; tiles
stitch at the requested zoom; throttling is retryable, a refused key is not,
and the key never appears in an error or a log line. The engine runs on the API
when given a key, stops when the key is refused, and refuses API mode without
one.
"""

import asyncio
import io
import os
import sys
import time
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geofuse  # noqa: F401  (import order: torch before GeoPandas)
import aiohttp
import geopandas as gpd
import numpy as np
import torch
from aiohttp import web
from PIL import Image
from shapely.geometry import Point

from geofuse import gvi as gvi_mod
from geofuse import logger
from geofuse import streetview as sv

KEY = "AIzaTESTKEY-0123456789abcdefghijklm"
PANO_ID = "AbCdEfGhIjKlMnOpQrStUv"


def _jpeg(shade: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (512, 512), (shade, shade, shade)).save(buf, "JPEG")
    return buf.getvalue()


class _Google:
    """The tile API's three endpoints, with behaviour chosen by latitude."""

    def __init__(self):
        self.sessions = 0
        self.tiles: list[str] = []
        self.refuse = False

    def check(self, request):
        if self.refuse or request.query.get("key") != KEY:
            reason = "API key not valid. Please pass a valid API key."
            error = _json({"error": {"message": reason}})
            raise web.HTTPForbidden(text=error, content_type="application/json")

    async def create_session(self, request):
        self.check(request)
        self.sessions += 1
        await asyncio.sleep(0.05)
        body = await request.json()
        assert body["mapType"] == "streetview"
        expiry = str(int(time.time()) + 14 * 86400)
        return web.json_response({"session": "tok", "expiry": expiry})

    async def metadata(self, request):
        self.check(request)
        assert request.query["session"] == "tok"
        lat, lng = float(request.query["lat"]), float(request.query["lng"])
        kind = round(lat)
        if kind == 0:
            return web.json_response({"error": {"message": "not found"}}, status=404)
        if kind == 2:
            return web.json_response({"error": {"message": "slow down"}}, status=429)
        answer = {
            "panoId": PANO_ID if kind != 3 else "CIHM0ogUSERPHOTOSPHERE123",
            "lat": lat + 1e-4,
            "lng": lng,
            "imageHeight": 8192,
            "imageWidth": 16384,
            "tileHeight": 512,
            "tileWidth": 512,
            "imageryType": "indoor" if kind == 1 else "outdoor",
            "date": "2023-05",
        }
        return web.json_response(answer)

    async def tile(self, request):
        self.check(request)
        assert request.query["session"] == "tok"
        m = request.match_info
        self.tiles.append(f"{m['z']}/{m['x']}/{m['y']}:{request.query['panoId']}")
        return web.Response(body=_jpeg(200 if m["x"] == "0" else 60))


def _json(obj) -> str:
    import json

    return json.dumps(obj)


class _ApiCase(unittest.TestCase):
    def setUp(self):
        sv._API_TOKENS.clear()
        sv._API_TOKEN_TASKS.clear()
        self.google = _Google()

    def serve(self, body):
        """Run ``body(session)`` against the stand-in server."""

        async def main():
            app = web.Application()
            app.router.add_post("/v1/createSession", self.google.create_session)
            app.router.add_get("/v1/streetview/metadata", self.google.metadata)
            app.router.add_get(
                "/v1/streetview/tiles/{z}/{x}/{y}", self.google.tile
            )
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            try:
                with mock.patch.object(sv, "_API_ROOT", f"http://127.0.0.1:{port}/v1"):
                    async with aiohttp.ClientSession() as session:
                        return await body(session)
            finally:
                await runner.cleanup()

        return asyncio.run(main())


class SearchTests(_ApiCase):
    def test_metadata_becomes_a_panorama(self):
        pano = self.serve(lambda s: sv.find_panorama_api_async(51.0, -114.0, s, KEY))
        self.assertEqual(pano.id, PANO_ID)
        self.assertAlmostEqual(pano.lat, 51.0001)
        self.assertEqual(pano.tile_size, sv.Size(512, 512))
        self.assertEqual(pano.image_sizes[1], sv.Size(1024, 512))
        self.assertEqual(pano.image_sizes[-1], sv.Size(16384, 8192))
        self.assertEqual(pano.date, (2023, 5))
        self.assertEqual(pano.captures, [sv.PanoCapture(PANO_ID, 2023, 5)])

    def test_concurrent_searches_share_one_session(self):
        async def many(session):
            return await asyncio.gather(
                *(sv.find_panorama_api_async(50 + i / 10, -114, session, KEY)
                  for i in range(8))
            )

        found = self.serve(many)
        self.assertEqual(len(found), 8)
        self.assertEqual(self.google.sessions, 1)

    def test_misses_indoor_and_photospheres_are_no_panorama(self):
        for lat in (0.0, 1.0, 3.0):
            pano = self.serve(
                lambda s, lat=lat: sv.find_panorama_api_async(lat, 10.0, s, KEY)
            )
            self.assertIsNone(pano, lat)

    def test_a_year_only_date_keeps_the_year(self):
        answer = {
            "panoId": PANO_ID, "lat": 1, "lng": 2, "imageWidth": 13312,
            "imageHeight": 6656, "tileWidth": 512, "tileHeight": 512, "date": "2019",
        }
        pano = sv._parse_api_metadata(answer)
        self.assertIsNone(pano.date)
        self.assertEqual(pano.captures, [sv.PanoCapture(PANO_ID, 2019, None)])
        self.assertEqual(pano.image_sizes[1], sv.Size(832, 416))


class DownloadTests(_ApiCase):
    def test_tiles_stitch_at_the_zoom(self):
        async def fetch(session):
            pano = await sv.find_panorama_api_async(51.0, -114.0, session, KEY)
            return await sv.get_panorama_api_async(pano, session, KEY, zoom=1)

        image = self.serve(fetch)
        self.assertEqual(image.size, (1024, 512))
        self.assertGreater(image.getpixel((100, 100))[0], 150)
        self.assertLess(image.getpixel((900, 100))[0], 100)
        self.assertEqual(
            sorted(self.google.tiles), [f"1/0/0:{PANO_ID}", f"1/1/0:{PANO_ID}"]
        )


class RefusalTests(_ApiCase):
    def test_throttling_is_retryable_and_names_no_key(self):
        with self.assertRaises(sv.RateLimitedError) as caught:
            self.serve(lambda s: sv.find_panorama_api_async(2.0, 0.0, s, KEY))
        self.assertEqual(caught.exception.status, 429)
        self.assertNotIn(KEY, f"{caught.exception} {caught.exception.url}")

    def test_a_refused_key_says_why_and_names_no_key(self):
        self.google.refuse = True
        with self.assertRaises(sv.ApiRequestRefused) as caught:
            self.serve(lambda s: sv.find_panorama_api_async(51.0, -114.0, s, KEY))
        message = str(caught.exception)
        self.assertIn("API key not valid", message)
        self.assertIn("Map Tiles API", message)
        self.assertNotIn(KEY, message)

    def test_a_connection_error_names_no_key(self):
        async def unreachable(session):
            with mock.patch.object(sv, "_API_ROOT", "http://127.0.0.1:9/v1"):
                return await sv.find_panorama_api_async(51.0, -114.0, session, KEY)

        with self.assertRaises(ConnectionError) as caught:
            self.serve(unreachable)
        self.assertNotIn(KEY, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)


# ────────────────────────────────────────────────────────────────────
# The engine on the API
# ────────────────────────────────────────────────────────────────────


class _Model:
    VEGETATION_CLASS, TERRAIN_CLASS = 8, 9
    effective_precision, graphs_in_use, precision = "fp32", False, "fp32"

    @staticmethod
    def preprocess_to_tensor(img):
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0)

    def segment(self, tensor, masks=False):
        found = np.where(tensor[:, 0].numpy() > 0.5, 8, 0)
        fractions = [
            {"GVI_Vegetation": float(np.mean(m == 8)), "GVI_Terrain": 0.0}
            for m in found
        ]
        return fractions, None


def _engine():
    engine = gvi_mod.GVIEngine.__new__(gvi_mod.GVIEngine)
    engine.device = torch.device("cpu")
    engine.segmenter = _Model()
    engine.download_mode, engine.api_key = "api", KEY
    return engine


def _points(n: int) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        geometry=[Point(-114.0 + 0.001 * i, 51.0) for i in range(n)], crs="EPSG:4326"
    )
    gdf["row"], gdf["col"], gdf["cluster_id"] = 0, list(range(n)), 0
    return gdf


class EngineTests(_ApiCase):
    def run_engine(self, n=4):
        """The engine's results; its log lines land in ``self.log``."""
        results = []
        job = f"api-test-{id(self)}"
        self.addCleanup(logger.drop_job_log_buffer, job)

        def run():
            logger.bind_job_log_buffer(job)
            try:
                _engine().run_analysis(
                    _points(n),
                    folder=os.path.join(ROOT, "tests", "output", "gvi_api"),
                    external_cache={},
                    result_callback=results.append,
                    progress_callback=lambda c, t: None,
                )
            finally:
                time.sleep(0.3)
                self.log = "\n".join(logger.get_job_log_lines(job))
                logger.unbind_job_log_buffer()

        async def body(_session):
            await asyncio.get_running_loop().run_in_executor(None, run)

        self.serve(body)
        return results

    def test_a_keyed_engine_samples_through_the_api(self):
        results = self.run_engine()
        self.assertEqual(len(results), 4)
        self.assertEqual({r["pano_id"] for r in results}, {PANO_ID})
        self.assertAlmostEqual(results[0]["gvi_veg"], 0.5, places=2)
        self.assertEqual(results[0]["pano_date"], "2023-05")
        self.assertTrue(self.google.tiles)
        self.assertIn("Throughput", self.log)
        self.assertNotIn(KEY, self.log)

    def test_a_refused_key_stops_the_run(self):
        self.google.refuse = True
        with self.assertRaises(sv.ApiRequestRefused):
            self.run_engine()
        self.assertNotIn(KEY, self.log)

    def test_api_mode_needs_a_key_and_a_key_means_api_mode(self):
        with self.assertRaises(ValueError):
            gvi_mod.GVIEngine(download_mode="api")
        with self.assertRaises(ValueError):
            gvi_mod.GVIEngine(download_mode="bing")
        with mock.patch.object(gvi_mod, "DeepLabSegmenter", lambda **kw: object()):
            engine = gvi_mod.GVIEngine(device="cpu", api_key=KEY)
            self.assertEqual(engine.download_mode, "api")
            engine = gvi_mod.GVIEngine(device="cpu")
            self.assertEqual(engine.download_mode, "package")


if __name__ == "__main__":
    unittest.main()
