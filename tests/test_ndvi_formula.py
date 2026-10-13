"""NDVI from negative reflectance is masked, never outside [−1, 1].

Every prep computes ``(NIR − red) / (NIR + red)`` and masks the pixel where
either band is negative, as Earth Engine's ``normalizedDifference`` does. The
engine runs end to end on the numeric fake of Earth Engine; each scene puts a
negative red band in the west third, a negative near-infrared band in the
middle third and positive bands in the east third.
"""

import hashlib
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.dirname(__file__)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from test_ndvi_acquisition import (
    AOI,
    RES,
    THIRD_1,
    THIRD_2,
    E,
    _Case,
    _ndvi,
    _tm_harmonised,
    landsat_sr,
    landsat_toa,
    s2,
)

from geofuse import ndvi
from geofuse.vector_io import geometry_sha256


class Field:
    """A band that varies across the area, with the arithmetic the scene
    builders apply to a reflectance (offset, then scale)."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, X, Y):
        return self.fn(X, Y)

    def __add__(self, k):
        return Field(lambda X, Y: self.fn(X, Y) + k)

    def __truediv__(self, k):
        return Field(lambda X, Y: self.fn(X, Y) / k)


def _thirds(west, middle, east):
    return Field(
        lambda X, Y: np.select([X < THIRD_1, X < THIRD_2], [west, middle], east)
    )


RED = _thirds(-0.01, 0.05, 0.05)
NIR = _thirds(0.40, -0.02, 0.40)


class NegativeInputTests(_Case):
    """Acceptance 1: a negative red or NIR pixel is masked; the rest unchanged."""

    def check(self, collections, expected_east, **kw):
        res = self.run_engine(collections, **kw)
        self.assertEqual(res["status"], "success", res.get("message"))
        arr, X, _ = self.raster()
        negative = X < THIRD_2
        self.assertTrue((arr[negative] == -9999).all())
        np.testing.assert_allclose(arr[~negative], expected_east, rtol=1e-6)
        valid = arr[arr != -9999]
        self.assertTrue(((valid >= -1) & (valid <= 1)).all())
        self.assertEqual(self.sidecar()["ndvi_formula"], ndvi.NDVI_FORMULA)

    def test_landsat_surface_reflectance(self):
        col = {E.LANDSAT8_COLLECTION_ID: [landsat_sr("2015-07-01", 5, RED, NIR)]}
        self.check(
            col,
            _ndvi(0.05, 0.40),
            start="2015-06-01",
            end="2015-09-30",
            satellite="landsat",
        )

    def test_landsat_tm_masks_after_harmonisation(self):
        # 0.001 red is positive as read and negative on the OLI scale.
        red = _thirds(0.001, 0.05, 0.05)
        col = {
            E.LANDSAT5_COLLECTION_ID: [
                landsat_sr("2009-07-01", 5, red, NIR, tm=True),
            ]
        }
        self.check(
            col,
            _tm_harmonised(0.05, 0.40),
            start="2009-06-01",
            end="2009-09-30",
            satellite="landsat",
        )

    def test_landsat_toa(self):
        for tm, cid, year in ((True, 0, "2009"), (False, 2, "2015")):
            scene = landsat_toa(f"{year}-07-01", 5, RED, NIR, tm=tm)
            col = {E.LANDSAT_TOA_COLLECTION_IDS[cid]: [scene]}
            self.check(
                col,
                _ndvi(0.05, 0.40),
                start=f"{year}-06-01",
                end=f"{year}-09-30",
                satellite="landsat",
                reflectance="toa",
            )

    def test_sentinel2_both_products(self):
        for reflectance, cid, masks in (
            ("surface", E.EE_COLLECTION_ID, {"SCL": 4}),
            ("toa", E.S2_TOA_COLLECTION_ID, {"QA60": 0}),
        ):
            col = {cid: [s2("2020-07-01", 5, RED, NIR, **masks)]}
            self.check(
                col,
                _ndvi(0.05, 0.40),
                start="2020-06-01",
                end="2020-09-30",
                reflectance=reflectance,
            )

    def test_values_from_the_live_check_are_masked(self):
        """The ratio gave 1.07, 3 and 19 for these bands."""
        for red, nir in ((-0.01, 0.30), (0.01, -0.02), (-0.0009, 0.001)):
            col = {E.LANDSAT8_COLLECTION_ID: [landsat_sr("2015-07-01", 5, red, nir)]}
            res = self.run_engine(
                col, start="2015-06-01", end="2015-09-30", satellite="landsat"
            )
            self.assertEqual(res["status"], "success", res.get("message"))
            arr, _, _ = self.raster()
            self.assertTrue((arr == -9999).all())

    def test_a_negative_scene_no_longer_sets_the_maximum(self):
        col = {
            E.LANDSAT8_COLLECTION_ID: [
                landsat_sr("2015-07-01", 5, 0.05, 0.30),
                landsat_sr("2015-07-17", 5, -0.0009, 0.001),
            ]
        }
        res = self.run_engine(
            col,
            start="2015-06-01",
            end="2015-09-30",
            satellite="landsat",
            temporal_reducers="max",
        )
        self.assertEqual(res["status"], "success", res.get("message"))
        arr, _, _ = self.raster("site_max")
        np.testing.assert_allclose(arr, _ndvi(0.05, 0.30), rtol=1e-6)


class ResumeKeyTests(unittest.TestCase):
    """Acceptance 2: no key matches one computed with the bare ratio."""

    def test_keys_differ_from_the_unmasked_formula(self):
        args = (AOI, "2015-06-01", "2015-09-30", 10, RES, E.LANDSAT_MERGED_LABEL)
        for reducer, variant in (("median", ()), ("max", ("water50", "pixel_only"))):
            h = hashlib.sha256()
            h.update(b"planar_v1|")
            h.update(geometry_sha256(AOI).encode())
            h.update(f"|2015-06-01|2015-09-30|10|{RES}|".encode())
            h.update(E.LANDSAT_MERGED_LABEL.encode())
            if reducer != "median":
                h.update(f"|{reducer}".encode())
            for option in variant:
                h.update(f"|{option}".encode())
            before = h.hexdigest()[:16]
            after = ndvi._compute_resume_key(
                *args, temporal_reducer=reducer, variant=variant
            )
            self.assertNotEqual(after, before)


if __name__ == "__main__":
    unittest.main()
