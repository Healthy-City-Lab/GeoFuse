"""Equivalence tests for the pre-aggregation build's fast paths.

Both accelerations here are pure: the stat kernel replaces ``np.percentile``
with a direct order-statistic selection, and the process pool replaces a thread
pool with worker interpreters. Neither may change a stored value, so each is
pinned against the implementation it replaces — the kernel against
``np.percentile`` itself, the pool against the in-process executor running the
identical batches.
"""

import contextlib
import gc
import os
import shutil
import sys
import tempfile
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from geofuse import parallel, preaggregation  # noqa: E402
from geofuse.fusion import MetricFusionEngine  # noqa: E402
from geofuse.raster_sampling import LazyRasterArray  # noqa: E402

CRS = "EPSG:32617"
RADII = (100, 200, 300)


def stock_stats(values):
    """What the kernel replaced: mean plus ``np.percentile`` on the grid."""
    if values.size == 0:
        return np.full(len(preaggregation.STAT_COLUMNS), np.nan, dtype=np.float32)
    out = np.empty(len(preaggregation.STAT_COLUMNS), dtype=np.float32)
    out[0] = float(values.mean())
    out[1:] = np.percentile(values, preaggregation.PERCENTILES)
    return np.round(out, 4)


class TestStatKernel(unittest.TestCase):
    def test_compute_all_stats_matches_numpy_percentile(self):
        rng = np.random.default_rng(0)
        for n in (1, 2, 3, 5, 8, 17, 129, 2048):
            for dtype in (np.float32, np.float64):
                values = (rng.random(n) * 2 - 1).astype(dtype)
                np.testing.assert_array_equal(
                    stock_stats(values), preaggregation.compute_all_stats(values)
                )

    def test_empty_sample_is_all_nan(self):
        out = preaggregation.compute_all_stats(np.empty(0, np.float64))
        self.assertTrue(np.isnan(out).all())

    def test_a_nan_in_the_sample_still_poisons_every_stat(self):
        values = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
        np.testing.assert_array_equal(
            stock_stats(values), preaggregation.compute_all_stats(values)
        )

    def test_prefix_stats_matches_reducing_each_prefix_separately(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            n = int(rng.integers(1, 900))
            ends = np.sort(rng.integers(0, n + 1, int(rng.integers(1, 8))))
            values = (rng.random(n) * 2 - 1).astype(np.float32)
            expected = np.full(
                (len(ends), len(preaggregation.STAT_COLUMNS)), np.nan, np.float32
            )
            for i, end in enumerate(ends):
                if end > 0:
                    expected[i] = stock_stats(values[:end])
            np.testing.assert_array_equal(
                expected, preaggregation.prefix_stats(values, ends)
            )


@contextlib.contextmanager
def scenario_dir():
    """A scratch directory that survives the handles a scenario leaves open.

    ``TemporaryDirectory`` raises on Windows if anything still holds a raster
    or a mapping inside it, which is a property of the platform rather than of
    the code under test.
    """
    tmpdir = tempfile.mkdtemp(prefix="preaggr-test-")
    try:
        yield tmpdir
    finally:
        preaggregation.worker_release()
        gc.collect()
        shutil.rmtree(tmpdir, ignore_errors=True)


def build_scenario(tmpdir, *, lazy_raster):
    """Two point channels sharing one geometry, plus an NDVI raster."""
    rng = np.random.default_rng(4)
    grid = np.arange(0, 2400, 40, dtype=np.float64)
    xx, yy = np.meshgrid(grid, grid)
    pts = [Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())]
    n_pts = len(pts)
    veg = gpd.GeoDataFrame({"gvi_veg": rng.random(n_pts)}, geometry=pts, crs=CRS)
    veg.attrs["metric_column"] = "gvi_veg"
    terrain = gpd.GeoDataFrame({"gvi_ter": rng.random(n_pts)}, geometry=pts, crs=CRS)
    terrain.attrs["metric_column"] = "gvi_ter"

    band = rng.random((240, 240)).astype(np.float64)
    band[:12, :12] = -9999.0
    transform = from_origin(0.0, 2400.0, 10.0, 10.0)
    if lazy_raster:
        path = os.path.join(tmpdir, "ndvi.tif")
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=240,
            width=240,
            count=1,
            dtype="float64",
            crs=CRS,
            transform=transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(band, 1)
        data = LazyRasterArray(path, band=1)
    else:
        data = np.ma.masked_equal(band, -9999.0)
    ndvi = {"data": data, "transform": transform, "crs": rasterio.crs.CRS.from_string(CRS)}

    entity_xy = rng.uniform(400.0, 2000.0, (192, 2))
    return veg, terrain, ndvi, entity_xy


class TestProcessPoolEquivalence(unittest.TestCase):
    """The pool must reproduce the in-process result cell for cell."""

    def _run_both(self, lazy_raster):
        with scenario_dir() as tmpdir:
            veg, terrain, ndvi, xy = build_scenario(tmpdir, lazy_raster=lazy_raster)
            geoms = [Point(x, y) for x, y in xy]
            positions = np.arange(len(xy), dtype=np.int64)
            engine = MetricFusionEngine.__new__(MetricFusionEngine)
            engine.cache_dir = tmpdir

            ticks = []
            stage_secs = {"vector_agg": 0.0, "raster_agg": 0.0}
            pooled = engine._aggregate_pixel_channels(
                positions,
                veg_src=veg,
                terrain_src=terrain,
                ndvi_src=ndvi,
                shared_gvi=True,
                gvi_radii=RADII,
                ndvi_radii=RADII,
                utm_crs=veg.crs,
                all_points=True,
                point_xy_utm=xy,
                entity_geoms_utm=geoms,
                max_workers=3,
                cancel_callback=None,
                stage_secs=stage_secs,
                progress_callback=ticks.append,
            )

            preps = {
                "veg": engine._prep_channel_source(veg, "veg", veg.crs, True),
                "terrain": engine._prep_channel_source(
                    terrain, "terrain", veg.crs, True
                ),
                "ndvi": engine._prep_channel_source(ndvi, "ndvi", veg.crs, True),
            }
            shared_xy = engine._shared_gvi_geometry(preps["veg"], preps["terrain"])
            self.assertIsNotNone(shared_xy)
            in_process = [
                np.full((len(xy), len(RADII), 6), np.nan, np.float32) for _ in range(3)
            ]

            def store(res):
                lo, hi = res[0], res[1]
                for slot, arr in enumerate(res[2:5]):
                    in_process[slot][lo:hi] = arr

            engine._aggregate_in_threads(
                positions,
                preps=preps,
                shared_xy=shared_xy,
                gvi_radii=RADII,
                ndvi_radii=RADII,
                utm_crs=veg.crs,
                point_xy_utm=xy,
                entity_geoms_utm=geoms,
                batches=[(0, 96), (96, len(xy))],
                workers=2,
                cancel_callback=None,
                on_result=store,
            )
            close_all = getattr(ndvi["data"], "close_all", None)
            if close_all is not None:
                close_all()
            return pooled, in_process, ticks

    def _assert_matches(self, lazy_raster):
        pooled, in_process, ticks = self._run_both(lazy_raster)
        self.assertIsNotNone(pooled)
        for name, got, expected in zip(("veg", "terrain", "ndvi"), pooled, in_process):
            self.assertTrue(
                np.isfinite(expected).any(), f"{name} reference is entirely NaN"
            )
            np.testing.assert_array_equal(expected, got, err_msg=name)
        self.assertEqual(ticks[-1], 192)

    def test_matches_with_an_in_memory_raster(self):
        self._assert_matches(lazy_raster=False)

    def test_matches_with_a_windowed_raster(self):
        self._assert_matches(lazy_raster=True)

    def test_cancel_abandons_the_build(self):
        with scenario_dir() as tmpdir:
            veg, terrain, ndvi, xy = build_scenario(tmpdir, lazy_raster=False)
            engine = MetricFusionEngine.__new__(MetricFusionEngine)
            engine.cache_dir = tmpdir
            out = engine._aggregate_pixel_channels(
                np.arange(len(xy), dtype=np.int64),
                veg_src=veg,
                terrain_src=terrain,
                ndvi_src=ndvi,
                shared_gvi=True,
                gvi_radii=RADII,
                ndvi_radii=RADII,
                utm_crs=veg.crs,
                all_points=True,
                point_xy_utm=xy,
                entity_geoms_utm=[Point(x, y) for x, y in xy],
                max_workers=2,
                cancel_callback=lambda: True,
                stage_secs={"vector_agg": 0.0, "raster_agg": 0.0},
            )
            self.assertIsNone(out)

    def test_scratch_directory_is_removed(self):
        with scenario_dir() as tmpdir:
            veg, terrain, ndvi, xy = build_scenario(tmpdir, lazy_raster=False)
            engine = MetricFusionEngine.__new__(MetricFusionEngine)
            engine.cache_dir = tmpdir
            engine._aggregate_pixel_channels(
                np.arange(len(xy), dtype=np.int64),
                veg_src=veg,
                terrain_src=terrain,
                ndvi_src=ndvi,
                shared_gvi=True,
                gvi_radii=RADII,
                ndvi_radii=RADII,
                utm_crs=veg.crs,
                all_points=True,
                point_xy_utm=xy,
                entity_geoms_utm=[Point(x, y) for x, y in xy],
                max_workers=2,
                cancel_callback=None,
                stage_secs={"vector_agg": 0.0, "raster_agg": 0.0},
            )
            leftovers = [d for d in os.listdir(tmpdir) if d.startswith("preaggr-")]
            self.assertEqual(leftovers, [])


class TestSharedArrayHandoff(unittest.TestCase):
    def test_published_array_reattaches_without_copying(self):
        tmpdir = tempfile.mkdtemp()
        try:
            values = np.arange(24, dtype=np.float32).reshape(12, 2)
            path = parallel.publish_array(tmpdir, "values", values)
            attached = parallel.attach_array(path)
            np.testing.assert_array_equal(values, attached)
            self.assertIsInstance(attached, np.memmap)
            del attached
        finally:
            # A live mapping keeps the file open on Windows.
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestSpawnSurvivesAStaleMainModule(unittest.TestCase):
    """A spawn worker re-executes the parent's main script before anything else.

    Streamlit's script runner repoints ``__main__`` at a scratch file, so once
    that file is cleaned up every worker would die on import. The pool has to
    keep running, since the engine is driven from exactly that environment.
    """

    @contextlib.contextmanager
    def _dangling_main(self):
        main = sys.modules["__main__"]
        had_file = hasattr(main, "__file__")
        previous_file = getattr(main, "__file__", None)
        previous_spec = getattr(main, "__spec__", None)
        scratch = tempfile.mkdtemp()
        ghost = os.path.join(scratch, "app.py")
        shutil.rmtree(scratch, ignore_errors=True)
        main.__file__ = ghost
        main.__spec__ = None
        try:
            yield
        finally:
            if had_file:
                main.__file__ = previous_file
            else:
                del main.__file__
            main.__spec__ = previous_spec

    def test_pool_still_runs(self):
        from geofuse import metric_columns

        with self._dangling_main():
            results = []
            completed = parallel.map_batches(
                metric_columns.channel_columns,
                [("veg",), ("ndvi",)],
                workers=2,
                on_result=results.append,
            )
        self.assertTrue(completed)
        self.assertEqual(len(results), 2)

    def test_main_module_is_left_as_it_was_found(self):
        with self._dangling_main():
            ghost = sys.modules["__main__"].__file__
            with parallel._spawnable_main():
                self.assertFalse(hasattr(sys.modules["__main__"], "__file__"))
            self.assertEqual(sys.modules["__main__"].__file__, ghost)

    def test_a_real_main_script_is_left_alone(self):
        main = sys.modules["__main__"]
        previous = getattr(main, "__file__", None)
        main.__file__ = __file__
        try:
            with parallel._spawnable_main():
                self.assertEqual(main.__file__, __file__)
        finally:
            if previous is None:
                del main.__file__
            else:
                main.__file__ = previous


def report_blas_env(_ignored):
    """Worker-side probe: what the thread limits look like inside a pool."""
    return {var: os.environ.get(var) for var in parallel._PINNED_THREAD_VARS}


class TestWorkersAreThreadPinned(unittest.TestCase):
    """Unpinned, numpy and OpenBLAS reserve ~3 GB of commit *per worker*.

    They size their reservation for the host's core count the moment they are
    imported, and a worker imports numpy while unpickling its own initializer —
    before any of our code runs. So the limits have to be in the environment the
    worker is *created* with, not set from inside it.
    """

    def test_a_worker_sees_the_limits(self):
        seen = []
        completed = parallel.map_batches(
            report_blas_env, [(0,)], workers=2, on_result=seen.append
        )
        self.assertTrue(completed)
        for var in parallel._PINNED_THREAD_VARS:
            self.assertEqual(seen[0].get(var), "1", f"{var} not pinned in the worker")

    def test_the_parent_environment_is_left_as_it_was_found(self):
        before = {v: os.environ.get(v) for v in parallel._PINNED_THREAD_VARS}
        with parallel._pinned_child_env():
            for var in parallel._PINNED_THREAD_VARS:
                self.assertEqual(os.environ.get(var), "1")
        self.assertEqual(
            before, {v: os.environ.get(v) for v in parallel._PINNED_THREAD_VARS}
        )

    def test_an_explicit_setting_is_not_overridden(self):
        previous = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "4"
        try:
            with parallel._pinned_child_env():
                self.assertEqual(os.environ["OMP_NUM_THREADS"], "4")
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "4")
        finally:
            if previous is None:
                del os.environ["OMP_NUM_THREADS"]
            else:
                os.environ["OMP_NUM_THREADS"] = previous


class TestPoolIsBoundedByMemory(unittest.TestCase):
    """A pool spends memory per worker, so cores alone cannot size it."""

    @contextlib.contextmanager
    def _pretend_free(self, free_bytes):
        original = parallel.available_memory_bytes
        parallel.available_memory_bytes = lambda: free_bytes
        try:
            yield
        finally:
            parallel.available_memory_bytes = original

    def test_a_tight_host_narrows_the_pool(self):
        with self._pretend_free(1 * 1024**3):
            self.assertLess(parallel.process_worker_count(10_000), 4)

    def test_a_roomy_host_lets_the_cores_decide(self):
        with self._pretend_free(512 * 1024**3):
            self.assertEqual(
                parallel.process_worker_count(10_000),
                max(2, (parallel.cpu_budget() * 2) // 3),
            )

    def test_the_override_is_bounded_too(self):
        previous = os.environ.get("GEOFUSE_WORKERS")
        os.environ["GEOFUSE_WORKERS"] = "64"
        try:
            with self._pretend_free(1 * 1024**3):
                self.assertLess(parallel.process_worker_count(10_000), 8)
        finally:
            if previous is None:
                del os.environ["GEOFUSE_WORKERS"]
            else:
                os.environ["GEOFUSE_WORKERS"] = previous

    def test_an_unreadable_figure_falls_back_to_cores(self):
        with self._pretend_free(0):
            self.assertEqual(
                parallel.process_worker_count(10_000),
                max(2, (parallel.cpu_budget() * 2) // 3),
            )

    def test_never_returns_less_than_one(self):
        with self._pretend_free(1):
            self.assertEqual(parallel.process_worker_count(10_000), 1)


class TestProcessWorkerCount(unittest.TestCase):
    def test_never_exceeds_the_work_available(self):
        self.assertEqual(parallel.process_worker_count(3), 3)

    def test_claims_more_of_the_host_than_the_thread_pool(self):
        self.assertGreaterEqual(
            parallel.process_worker_count(10_000), parallel.worker_count()
        )

    def test_cap_bounds_the_pool(self):
        self.assertLessEqual(parallel.process_worker_count(10_000, cap=2), 2)

    def test_env_override_wins(self):
        previous = os.environ.get("GEOFUSE_WORKERS")
        os.environ["GEOFUSE_WORKERS"] = "5"
        try:
            self.assertEqual(parallel.process_worker_count(10_000), 5)
        finally:
            if previous is None:
                del os.environ["GEOFUSE_WORKERS"]
            else:
                os.environ["GEOFUSE_WORKERS"] = previous


if __name__ == "__main__":
    unittest.main()
