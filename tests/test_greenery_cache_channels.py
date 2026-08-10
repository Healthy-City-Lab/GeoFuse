"""A cache unit must serve whichever channel set a job asks for.

The street-view components are stored apart from the merged green-view channel
because a percentile of ``veg + terrain`` is not the sum of the two components'
percentiles — only the mean is. So a unit built by a two-channel job and later
read by a three-channel one has to grow, and a unit that already holds more
than a job needs has to be reused untouched.
"""

import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from geofuse.preaggregation import GreeneryCache

RADII = (200, 400)
IDS = np.array([10, 20, 30], dtype=np.int64)


def _blocks(cache, channels, n_ids, fill=1.0):
    n_stats = len(cache.stats)
    return {
        ch: np.full((n_ids, len(RADII), n_stats), fill, np.float32)
        for ch in channels
    }


class TestChannelSets(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geofuse-cache-")
        self.cache = GreeneryCache(self.dir, spacing_m=40.0, crs_key="EPSG:3347")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _open(self, channels, ids=IDS):
        return self.cache.open_unit(
            "cfg", gvi_radii=RADII, ndvi_radii=RADII,
            required_ids=ids, channels=channels,
        )

    def test_a_two_channel_unit_round_trips(self):
        chans = ("ndvi", "gvi")
        _g, _n, missing, absent = self._open(chans)
        self.assertEqual(sorted(missing), [10, 20, 30])
        self.assertEqual(absent, ())
        self.cache.commit_unit("cfg", missing, _blocks(self.cache, chans, 3))

        # A second job wanting the same channels finds nothing missing.
        _g, _n, missing2, absent2 = self._open(chans)
        self.assertEqual(missing2.size, 0)
        self.assertEqual(absent2, ())

    def test_asking_for_a_channel_the_unit_never_held_rebuilds_every_row(self):
        self.cache.commit_unit(
            "cfg", self._open(("ndvi", "gvi"))[2],
            _blocks(self.cache, ("ndvi", "gvi"), 3),
        )
        _g, _n, missing, absent = self._open(("ndvi", "veg", "terrain"))
        # veg and terrain are absent, so all three ids must be recomputed —
        # a partial block stitched onto the existing id order would misalign.
        self.assertEqual(set(absent), {"veg", "terrain"})
        self.assertEqual(sorted(missing), [10, 20, 30])

    def test_the_rebuilt_unit_keeps_both_channel_sets(self):
        self.cache.commit_unit(
            "cfg", self._open(("ndvi", "gvi"))[2],
            _blocks(self.cache, ("ndvi", "gvi"), 3),
        )
        missing = self._open(("ndvi", "veg", "terrain"))[2]
        unit = self.cache._units["cfg"]
        self.cache.commit_unit(
            "cfg", missing, _blocks(self.cache, unit["channels"], len(missing))
        )
        self.assertEqual(set(unit["channels"]), {"veg", "terrain", "ndvi", "gvi"})
        for ch in unit["channels"]:
            self.assertEqual(len(unit[ch]), len(unit["ids"]))

    def test_a_wider_unit_serves_a_narrower_job_without_rebuilding(self):
        chans = ("veg", "terrain", "ndvi", "gvi")
        self.cache.commit_unit(
            "cfg", self._open(chans)[2], _blocks(self.cache, chans, 3)
        )
        _g, _n, missing, absent = self._open(("ndvi", "gvi"))
        self.assertEqual(missing.size, 0)
        self.assertEqual(absent, ())

    def test_the_stored_channel_set_survives_a_reload(self):
        chans = ("ndvi", "gvi")
        self.cache.commit_unit(
            "cfg", self._open(chans)[2], _blocks(self.cache, chans, 3)
        )
        fresh = GreeneryCache(self.dir, spacing_m=40.0, crs_key="EPSG:3347")
        _g, _n, missing, absent = fresh.open_unit(
            "cfg", gvi_radii=RADII, ndvi_radii=RADII,
            required_ids=IDS, channels=chans,
        )
        self.assertEqual(missing.size, 0)
        self.assertEqual(absent, ())
        self.assertEqual(set(fresh._units["cfg"]["channels"]), set(chans))

    def test_an_unknown_channel_is_refused(self):
        with self.assertRaises(ValueError):
            self._open(("ndvi", "canopy"))


if __name__ == "__main__":
    unittest.main()
