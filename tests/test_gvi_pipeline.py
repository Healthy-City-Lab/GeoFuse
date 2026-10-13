"""The GVI pipeline: search, a download pool, and one GPU consumer (``gvi``).

With a stub downloader and a deterministic stub model: every point gets the
value its panorama's image implies, whatever the batch size and pool size; a
forward never receives more than ``batch_size`` images; a panorama shared by
several points is downloaded and segmented once; the download -> GPU queue
stays bounded; cancel ends the run promptly and pause holds the GPU stage;
the run records each stage's counts.
"""

import asyncio
import os
import sys
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas as gpd
import numpy as np
import torch
from PIL import Image
from shapely.geometry import Point

import geofuse  # noqa: F401  (import order: torch before GeoPandas)
from geofuse import streetview as sv
from geofuse.gvi import GVIEngine


def _image(pid: str) -> Image.Image:
    """A 256 x 128 panorama whose left share of bright pixels encodes ``pid``."""
    share = (sum(map(ord, pid)) % 9 + 1) / 10
    arr = np.full((128, 256, 3), 60, dtype=np.uint8)
    arr[:, : int(256 * share)] = 200
    return Image.fromarray(arr)


def _expected(pid: str) -> float:
    return int(256 * (sum(map(ord, pid)) % 9 + 1) / 10) / 256


class _Model:
    """Class 8 where the first channel is bright; records each forward."""

    VEGETATION_CLASS, TERRAIN_CLASS = 8, 9
    effective_precision, graphs_in_use = "fp32", False

    def __init__(self, delay=0.0):
        self.batches: list[int] = []
        self.delay = delay
        self.precision = "fp16"

    @staticmethod
    def preprocess_to_tensor(img):
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0)

    def _masks(self, tensor):
        time.sleep(self.delay)
        self.batches.append(int(tensor.shape[0]))
        return np.where(tensor[:, 0].numpy() > 0.5, 8, 0)

    def segment(self, tensor, masks=False):
        found = self._masks(tensor)
        fractions = [self.calculate_gvi_from_mask(m) for m in found]
        return fractions, found.astype(np.uint8) if masks else None

    def calculate_gvi_from_mask(self, mask):
        return {
            "GVI_Vegetation": float(np.mean(mask == 8)),
            "GVI_Terrain": float(np.mean(mask == 9)),
        }


def _engine(model):
    engine = GVIEngine.__new__(GVIEngine)
    engine.device = torch.device("cpu")
    engine.segmenter = model
    # The stub images are small: keep them at their own width.
    engine._preprocess_image = lambda img, target_width=256: img.resize((256, 128))
    return engine


def _points(n: int) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        geometry=[Point(-114.0 + 0.001 * i, 51.0) for i in range(n)], crs="EPSG:4326"
    )
    gdf["row"], gdf["col"], gdf["cluster_id"] = 0, list(range(n)), 0
    return gdf


class _Case(unittest.TestCase):
    def setUp(self):
        self.downloads: list[str] = []
        self.shared_every = 1  # point i -> panorama i // shared_every

        async def find(lat, lon, session, radius=50.0, **_):
            await asyncio.sleep(0.001)
            i = round((lon + 114.0) / 0.001)
            return sv.StreetViewPanorama(
                id=f"pano{i // self.shared_every:04d}",
                lat=lat,
                lon=lon,
                tile_size=sv.Size(256, 128),
                image_sizes=[sv.Size(256, 128), sv.Size(256, 128)],
                date=(2024, 6),
            )

        async def download(pano, session, zoom=1):
            self.downloads.append(pano.id)
            await asyncio.sleep(0.005)
            return _image(pano.id)

        fakes = {"find_panorama_async": find, "get_panorama_async": download}
        for name, value in fakes.items():
            patch = mock.patch.object(sv, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def run_engine(self, model, n=40, **kw):
        engine, results = _engine(model), []
        engine.run_analysis(
            _points(n),
            folder=os.path.join(ROOT, "tests", "output", "gvi_pipeline"),
            external_cache={},
            result_callback=results.append,
            progress_callback=lambda c, t: None,
            **kw,
        )
        return engine, {r["orig_index"]: r for r in results}


class PipelineTests(_Case):
    def test_values_do_not_depend_on_batch_or_pool_size(self):
        runs = [
            self.run_engine(_Model(), batch_size=b, download_concurrency=c)[1]
            for b, c in ((1, 1), (1, 16), (4, 16), (8, 3))
        ]
        for result in runs[0].values():
            self.assertAlmostEqual(result["gvi_veg"], _expected(result["pano_id"]))
        for other in runs[1:]:
            self.assertEqual(
                {k: v["gvi_veg"] for k, v in other.items()},
                {k: v["gvi_veg"] for k, v in runs[0].items()},
            )

    def test_a_forward_takes_at_most_batch_size_images(self):
        model = _Model(delay=0.01)
        self.run_engine(model, n=60, batch_size=4, batch_wait_ms=50)
        self.assertLessEqual(max(model.batches), 4)
        self.assertGreater(max(model.batches), 1)
        self.assertEqual(sum(model.batches), 60)
        model = _Model()
        self.run_engine(model, batch_size=1)
        self.assertEqual(set(model.batches), {1})

    def test_a_shared_panorama_is_downloaded_and_segmented_once(self):
        self.shared_every = 5
        model = _Model()
        engine, results = self.run_engine(model, n=40)
        self.assertEqual(len(results), 40)
        self.assertEqual(sorted(self.downloads), sorted(set(self.downloads)))
        self.assertEqual(len(self.downloads), 8)
        stats = engine.last_run_stats
        self.assertEqual((stats["searched"], stats["downloads"]), (40, 8))
        self.assertEqual(stats["segmented"], 8)
        self.assertEqual(stats["batch_size"], 1)
        self.assertEqual((stats["precision"], stats["cuda_graph"]), ("fp32", False))

    def test_precision_reaches_the_segmenter(self):
        model = _Model()
        self.run_engine(model, n=4, precision="fp32")
        self.assertEqual(model.precision, "fp32")
        model = _Model()
        self.run_engine(model, n=4)
        self.assertEqual(model.precision, "fp16")

    def test_the_gpu_queue_is_bounded(self):
        model = _Model(delay=0.02)
        waiting = []
        real_put = asyncio.Queue.put

        async def put(queue, item):
            if queue.maxsize:
                waiting.append(queue.qsize())
            return await real_put(queue, item)

        with mock.patch.object(asyncio.Queue, "put", put):
            self.run_engine(model, n=40, batch_size=2, download_concurrency=16)
        self.assertTrue(waiting)
        self.assertLessEqual(max(waiting), 4)


class ControlTests(_Case):
    def test_cancel_ends_the_run_promptly(self):
        model = _Model(delay=0.05)
        cancelled = threading.Event()
        threading.Timer(0.5, cancelled.set).start()
        started = time.perf_counter()
        self.run_engine(model, n=200, cancel_callback=cancelled.is_set)
        self.assertLess(time.perf_counter() - started, 3.0)
        self.assertLess(sum(model.batches), 200)

    def test_pause_holds_the_gpu_stage(self):
        model = _Model()
        paused = threading.Event()
        paused.set()
        threading.Timer(0.6, paused.clear).start()
        started = time.perf_counter()
        _engine_, results = self.run_engine(model, n=10, pause_callback=paused.is_set)
        self.assertGreaterEqual(time.perf_counter() - started, 0.5)
        self.assertEqual(len(results), 10)


if __name__ == "__main__":
    unittest.main()
