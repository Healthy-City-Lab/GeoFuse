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
from geofuse import gvi as gvi_mod
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


class _OutcomeCase(_Case):
    """Points 0, 4, 8, ... have no panorama, 1, 5, ... only a January
    capture, and 2, 6, ... a panorama whose download fails."""

    def setUp(self):
        super().setUp()

        async def find(lat, lon, session, radius=50.0, **_):
            i = round((lon + 114.0) / 0.001)
            if i % 4 == 0:
                return None
            return sv.StreetViewPanorama(
                id=f"pano{i:04d}",
                lat=lat,
                lon=lon,
                tile_size=sv.Size(256, 128),
                image_sizes=[sv.Size(256, 128), sv.Size(256, 128)],
                date=(2024, 1 if i % 4 == 1 else 6),
            )

        async def download(pano, session, zoom=1):
            if int(pano.id[4:]) % 4 == 2:
                raise RuntimeError("tile server error")
            return _image(pano.id)

        fakes = {"find_panorama_async": find, "get_panorama_async": download}
        for name, value in fakes.items():
            patch = mock.patch.object(sv, name, value)
            patch.start()
            self.addCleanup(patch.stop)


class OutcomeTests(_OutcomeCase):
    """Every point is counted once: sampled, no panorama, no capture in the
    window, or failed."""

    def counts(self, stats):
        """``(sampled, no panorama, no capture in the window, failed)``."""
        keys = ("sampled", "no_panorama", "no_capture_in_window", "failed")
        return tuple(stats[key] for key in keys)

    def test_a_month_window_moves_points_from_sampled_to_window_misses(self):
        engine, results = self.run_engine(_Model(), n=40)
        self.assertEqual(self.counts(engine.last_run_stats), (20, 10, 0, 10))
        self.assertEqual(len(results), 20)

        engine, results = self.run_engine(_Model(), n=40, month_window=(5, 10))
        stats = engine.last_run_stats
        self.assertEqual(self.counts(stats), (10, 10, 10, 10))
        self.assertEqual(stats["no_capture_in_month_window"], 10)
        self.assertEqual(sum(self.counts(stats)), stats["points"])
        self.assertEqual(len(results), 10)

    def test_a_year_window_miss_is_counted_as_such(self):
        engine, _results = self.run_engine(
            _Model(), n=40, target_year=2015, max_year_diff=1
        )
        stats = engine.last_run_stats
        self.assertEqual(stats["no_capture_in_window"], 30)
        self.assertEqual(stats["no_capture_in_year_window"], 30)
        self.assertEqual(stats["no_capture_in_month_window"], 0)

    def test_the_sidecar_holds_the_engine_counts(self):
        import json
        import tempfile

        from geofuse.jobs import runners

        engine, results = self.run_engine(_Model(), n=40, month_window=(5, 10))
        stats = engine.last_run_stats
        with tempfile.TemporaryDirectory() as out:
            runners._write_gvi_outputs(
                accumulated=list(results.values()),
                processed_crs="EPSG:4326",
                meta={"step_m": 50},
                out_name="site",
                output_dir=out,
                save_gpkg=True,
                save_geotiff=False,
                save_geojson=False,
                counts=runners.point_counts(stats),
            )
            with open(os.path.join(out, "site_gvi.json")) as f:
                side = json.load(f)
        self.assertEqual(side["n_grid_points"], 40)
        found = [side[k] for k in ("n_sampled", "n_no_panorama", "n_failed")]
        self.assertEqual(found, [10, 10, 10])
        self.assertEqual(side["n_no_capture_in_window"], 10)
        self.assertEqual(side["n_no_capture_in_month_window"], 10)
        self.assertEqual(sum(side["capture_years"].values()), side["n_sampled"])


class LogTests(_OutcomeCase):
    """The log does not grow with the points: a summary at most every
    ``_SUMMARY_EVERY_S`` and the warnings and errors; ``log_points`` brings
    back a DEBUG line for every point."""

    def run_logged(self, model=None, **kw):
        lines: list[tuple[str, str]] = []
        record = lambda level, msg: lines.append((level, msg))  # noqa: E731
        with mock.patch.object(gvi_mod, "_log", record):
            engine, _results = self.run_engine(model or _Model(), **kw)
        return engine, lines

    def test_the_log_does_not_grow_with_the_points(self):
        async def find(lat, lon, session, radius=50.0, **_):
            i = round((lon + 114.0) / 0.001)
            return sv.StreetViewPanorama(
                id=f"pano{i // 10:04d}",
                lat=lat,
                lon=lon,
                tile_size=sv.Size(256, 128),
                image_sizes=[sv.Size(256, 128), sv.Size(256, 128)],
                date=(2024, 6),
            )

        async def download(pano, session, zoom=1):
            return _image(pano.id)

        with (
            mock.patch.object(sv, "find_panorama_async", find),
            mock.patch.object(sv, "get_panorama_async", download),
            mock.patch.object(gvi_mod, "_SUMMARY_EVERY_S", 3600.0),
        ):
            _engine, small = self.run_logged(n=100)
            engine, large = self.run_logged(n=10_000)
        self.assertEqual(engine.last_run_stats["sampled"], 10_000)
        self.assertEqual([lvl for lvl, _ in large], [lvl for lvl, _ in small])
        self.assertLess(sum(len(m) for _, m in large), 2_000)

    def test_summaries_carry_the_run_counts(self):
        with mock.patch.object(gvi_mod, "_SUMMARY_EVERY_S", 0.02):
            engine, lines = self.run_logged(
                _Model(delay=0.01), n=80, month_window=(5, 10)
            )
        summaries = [m for lvl, m in lines if lvl == "INFO" and m.startswith("Points")]
        self.assertGreaterEqual(len(summaries), 2)
        stats = engine.last_run_stats
        self.assertTrue(
            summaries[-1].startswith(
                f"Points 80/80: {stats['sampled']} sampled, "
                f"{stats['no_panorama']} no panorama, "
                f"{stats['no_capture_in_window']} no capture in the window "
                f"(year 0, month {stats['no_capture_in_month_window']}), "
                f"{stats['failed']} failed; "
            ),
            summaries[-1],
        )
        self.assertEqual(stats["no_capture_in_window"], 20)

    def test_failures_and_throttling_are_still_logged(self):
        real_find = sv.find_panorama_async
        throttled: list[int] = []

        async def find(lat, lon, session, **kw):
            if not throttled:
                throttled.append(1)
                raise sv.RateLimitedError(429)
            return await real_find(lat, lon, session, **kw)

        with (
            mock.patch.object(sv, "find_panorama_async", find),
            mock.patch.object(gvi_mod, "_RL_BACKOFF_START_S", 0.05),
        ):
            _engine, lines = self.run_logged(n=20)
        errors = [m for lvl, m in lines if lvl == "ERROR"]
        self.assertEqual(len([m for m in errors if "Download error" in m]), 5)
        self.assertTrue(any("backing off" in m for lvl, m in lines if lvl == "WARN"))
        self.assertFalse([m for lvl, m in lines if lvl == "DEBUG"])

    def test_log_points_brings_back_a_line_per_point(self):
        _engine, lines = self.run_logged(n=20, log_points=True)
        debug = [m for lvl, m in lines if lvl == "DEBUG"]
        for i in range(20):
            self.assertTrue(any(m.startswith(f"Point {i} @") for m in debug), i)
        self.assertEqual(len([m for m in debug if "no panoramas within" in m]), 5)


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
