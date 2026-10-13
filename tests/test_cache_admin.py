"""Measuring and purging the toolbox's caches (``geofuse.persistence.caches``).

The GUI's cache panel and ``geofuse cache`` share these: each cache reports
what it holds (a cache that was never created reads empty, and is not created
by looking), and a purge empties it and says what it freed. The panel shows
the same lines.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geofuse.persistence import caches


class CacheAdminTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        root = os.path.join(self.dir, "caches")
        patch = mock.patch.object(caches, "cache_root", lambda: root)
        patch.start()
        self.addCleanup(patch.stop)
        self.out = os.path.join(self.dir, "out")

    def tearDown(self):
        self.tmp.cleanup()

    def fill(self):
        os.makedirs(caches.cache_root(), exist_ok=True)
        pano = caches.PanoCache(caches.pano_db())
        pano["a"] = {"veg": 0.1, "ter": 0.2}
        pano["b"] = {"veg": 0.3, "ter": 0.4}
        pano.close()
        tiles = caches.NdviTileCache(caches.ndvi_tiles())
        with open(os.path.join(tiles.workspace_dir("k1"), "t.tif"), "wb") as f:
            f.write(b"x" * 2048)
        tiles.touch("k1")
        tiles.close()
        folder = os.path.join(caches.fusion_cache_dir(self.out), "greenery")
        os.makedirs(folder)
        for name in ("greenery-1.npz", "greenery-2.npz"):
            with open(os.path.join(folder, name), "wb") as f:
                f.write(b"y" * 1024)

    def test_absent_caches_read_empty_without_being_created(self):
        for key in caches.CACHE_LABELS:
            self.assertEqual(caches.describe_cache(key, output_dir=self.out), "empty")
        self.assertFalse(os.path.exists(caches.pano_db()))
        self.assertFalse(os.path.exists(caches.ndvi_tiles()))
        self.assertIn("already empty", caches.purge_cache("pano", output_dir=self.out))

    def test_each_cache_reports_and_purges(self):
        self.fill()

        def describe(key):
            return caches.describe_cache(key, output_dir=self.out)

        self.assertEqual(describe("pano"), "2 panoramas")
        self.assertEqual(describe("ndvi"), "2.0 KB · 1 workspaces")
        self.assertEqual(describe("fusion"), "2.0 KB · 2 files")
        self.assertEqual(len(caches.greenery_cache_files(self.out)), 2)

        self.assertIn("Cleared 2", caches.purge_cache("pano", output_dir=self.out))
        self.assertIn("freed 2.0 KB", caches.purge_cache("ndvi", output_dir=self.out))
        self.assertIn(
            "Removed 2 files", caches.purge_cache("fusion", output_dir=self.out)
        )
        for key in ("pano", "ndvi"):
            self.assertIn(describe(key), ("0 panoramas", "0 B · 0 workspaces"))
        self.assertEqual(describe("fusion"), "empty")

    def test_open_caches_are_measured_and_cleared_in_place(self):
        self.fill()
        pano = caches.PanoCache(caches.pano_db())
        self.assertIn("a", pano)
        caches.purge_cache("pano", pano=pano)
        self.assertNotIn("a", pano)
        pano.close()

    def test_unknown_caches_are_refused(self):
        with self.assertRaisesRegex(ValueError, "pano, ndvi, fusion"):
            caches.describe_cache("tiles")


class PanelTests(unittest.TestCase):
    def test_the_sidebar_panel_shows_each_cache(self):
        from streamlit.testing.v1 import AppTest

        # The app runs in this process: put back what its script replaces.
        self.addCleanup(setattr, caches, "describe_cache", caches.describe_cache)
        script = (
            "import sys\n"
            f"sys.path.insert(0, {os.path.join(ROOT, 'ui')!r})\n"
            "from geofuse.persistence import caches\n"
            "import services\n"
            "services.get_pano_cache = lambda: {}\n"
            "services.get_ndvi_tile_cache = lambda: None\n"
            "caches.describe_cache = lambda key, **kw: f'{key} line'\n"
            "from cache_manager import render_cache_manager\n"
            "render_cache_manager('out')\n"
        )
        at = AppTest.from_string(script, default_timeout=60)
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        captions = [c.value for c in at.sidebar.caption]
        for key in ("pano", "ndvi", "fusion"):
            self.assertIn(f"{key} line", captions)


if __name__ == "__main__":
    unittest.main()
