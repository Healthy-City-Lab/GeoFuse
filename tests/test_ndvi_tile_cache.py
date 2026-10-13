"""Emptying the NDVI tile cache (``NdviTileCache.clear``).

A purge removes every workspace, including ones a run never indexed, and
empties the index; eviction to a size cap is a different thing, and a cap of
zero leaves the cache alone.
"""

import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geofuse.persistence.caches import NdviTileCache


class ClearTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = NdviTileCache(self.tmp.name)
        for key in ("k1", "k2"):
            with open(os.path.join(self.cache.workspace_dir(key), "t.tif"), "wb") as f:
                f.write(b"x" * 1000)
        self.cache.touch("k1")

    def tearDown(self):
        self.cache.close()
        self.tmp.cleanup()

    def test_clear_removes_every_workspace(self):
        self.assertEqual(self.cache.evict_lru(0), (0, 0))
        self.assertEqual(self.cache.clear(), (2, 2000))
        self.assertEqual(self.cache.stats(), {"entries": 0, "bytes": 0})
        left = [p for p in os.listdir(self.tmp.name) if not p.startswith("_index")]
        self.assertEqual(left, [])


if __name__ == "__main__":
    unittest.main()
