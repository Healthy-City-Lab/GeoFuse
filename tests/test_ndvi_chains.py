"""NDVI aggregation chains: their grammar, and what the engine makes of them.

A chain orders per-scene neighbourhood statistics, one temporal statistic and
water masks. The engine runs end to end on the numeric fake of Earth Engine,
and every output is checked against numpy: a neighbourhood statistic over the
grid pixels within the radius, masked pixels skipped and a masked centre kept
masked, then the statistic across scenes.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.dirname(__file__)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from _fake_earth_engine import Image, focal
from test_ndvi_acquisition import (
    CLOUD_BIT,
    RES,
    THIRD_1,
    THIRD_2,
    E,
    _Case,
    _ndvi,
    landsat_sr,
)
from test_ndvi_formula import Field

from geofuse import ndvi
from geofuse.ndvi_chains import (
    ChainError,
    as_statistic,
    check_chains,
    format_lines,
    output_stems,
    parse_lines,
)

RADIUS = 60
WINDOW = {"start": "2015-06-01", "end": "2015-09-30", "satellite": "landsat"}


# ────────────────────────────────────────────────────────────────────
# Grammar
# ────────────────────────────────────────────────────────────────────


class GrammarTests(unittest.TestCase):
    def test_tables_and_lines_check_to_one_spelling(self):
        tables = [
            {
                "name": "a",
                "steps": [" Focal:MEAN:500.0 ", "temporal:mean", "water_mask"],
            },
            {"name": "b_2", "steps": ["temporal:max"]},
        ]
        lines = ["a: focal:mean:500, temporal:mean, water_mask", "b_2: temporal:max"]
        expected = [
            {"name": "a", "steps": ["focal:mean:500", "temporal:mean", "water_mask"]},
            {"name": "b_2", "steps": ["temporal:max"]},
        ]
        self.assertEqual(check_chains(tables), expected)
        self.assertEqual(check_chains(lines), expected)
        self.assertEqual(parse_lines("# reference\n\n" + "\n".join(lines)), expected)
        self.assertEqual(parse_lines(format_lines(expected)), expected)
        self.assertEqual(check_chains(None), [])
        self.assertEqual(check_chains([]), [])
        self.assertEqual(
            check_chains(["c: focal:median:2.5, temporal:median"])[0]["steps"][0],
            "focal:median:2.5",
        )

    def test_errors_name_the_chain(self):
        """Acceptance 4."""
        for chains, words in (
            (["g1: focal:sum:500, temporal:mean"], ("'g1'", "focal:sum:500")),
            (["g2: smooth, temporal:mean"], ("'g2'", "unknown operation 'smooth'")),
            (["g3: focal:mean:500"], ("'g3'", "exactly one temporal step", "none")),
            (["g4: temporal:mean, temporal:max"], ("'g4'", "it has 2")),
            (["g5: focal:mean:0, temporal:mean"], ("'g5'", "above 0")),
            (["g6: focal:mean:5001, temporal:mean"], ("'g6'", "5,000")),
            (["g7: focal:mean:far, temporal:mean"], ("'g7'", "number of metres")),
            (["g8: temporal:p90"], ("'g8'", "temporal statistic")),
            (
                ["dup: temporal:mean", "DUP: temporal:max"],
                ("'DUP'", "'dup'", "ignoring case"),
            ),
            (["bad-name: temporal:mean"], ("'bad-name'", "letters, digits")),
            ([{"name": "t", "steps": []}], ("'t'", "list of operations")),
            ([{"name": "u", "steps": ["temporal:mean"], "x": 1}], ("unknown key",)),
            (["temporal mean"], ("Chain 1", "name: operation")),
        ):
            with self.subTest(chains=chains):
                with self.assertRaises(ChainError) as ctx:
                    check_chains(chains)
                for word in words:
                    self.assertIn(word, str(ctx.exception))
        with self.assertRaises(ChainError):
            check_chains("a: temporal:mean")

    def test_statistic_chains_and_names(self):
        self.assertEqual(as_statistic(["temporal:max"]), ("max", False))
        self.assertEqual(as_statistic(["water_mask", "temporal:mean"]), ("mean", True))
        self.assertIsNone(as_statistic(["temporal:mean", "water_mask"]))
        self.assertIsNone(as_statistic(["focal:mean:500", "temporal:mean"]))
        self.assertEqual(
            output_stems("a_2015", ["median", "max"]),
            {"median": "a_2015", "max": "a_2015_max"},
        )
        chains = check_chains(["median: temporal:median", "g01: temporal:mean"])
        self.assertEqual(
            output_stems("a_2015", ["median"], chains),
            {"median": "a_2015_median", "g01": "a_2015_g01"},
        )


# ────────────────────────────────────────────────────────────────────
# Engine
# ────────────────────────────────────────────────────────────────────

_X0 = THIRD_1 - (THIRD_2 - THIRD_1)
_SPAN = 3 * (THIRD_2 - THIRD_1)

# Two scenes whose NDVI varies across the area, cloudy in different thirds.
NIR_A = Field(lambda X, Y: 0.15 + 0.25 * (X - _X0) / _SPAN + 0.05 * np.sin(Y / 50))
NIR_B = Field(lambda X, Y: 0.30 + 0.10 * np.cos(X / 70))
EAST_CLOUD = lambda X, Y: np.where(X >= THIRD_2, 1 << 3, 0)  # noqa: E731
SCENES = {
    E.LANDSAT8_COLLECTION_ID: [
        landsat_sr("2015-07-01", 5, 0.05, NIR_A, qa=CLOUD_BIT),
        landsat_sr("2015-07-17", 5, 0.05, NIR_B, qa=EAST_CLOUD),
    ]
}


def _scene_ndvi(nir, cloudy):
    def fn(X, Y):
        return np.where(cloudy(X, Y) != 0, np.nan, _ndvi(0.05, nir(X, Y)))

    return fn


A = _scene_ndvi(NIR_A, CLOUD_BIT)
B = _scene_ndvi(NIR_B, EAST_CLOUD)


def _mean(*arrays):
    out = np.nanmean(np.stack(arrays), axis=0)
    return np.where(np.isnan(out), -9999.0, out)


class EngineChainTests(_Case):
    def run_chains(self, chains, collections=SCENES, images=None, **kw):
        res = self.run_engine(collections, images, chains=chains, **WINDOW, **kw)
        self.assertEqual(res["status"], "success", res.get("message"))
        return res

    def test_a_statistic_chain_equals_the_statistic(self):
        """Acceptance 1: same pixels, same cache key."""
        self.run_engine(SCENES, temporal_reducers="mean", **WINDOW)
        stat_side = self.sidecar("site_mean")
        self.assertEqual(stat_side["chain"], ["temporal:mean"])
        self.assertIsNone(stat_side["chain_name"])
        stat, _, _ = self.raster("site_mean")

        res = self.run_chains([{"name": "m", "steps": ["temporal:mean"]}])
        self.assertEqual(set(res["reducers"]), {"m"})
        chain, _, _ = self.raster("site_m")
        np.testing.assert_array_equal(chain, stat)
        side = self.sidecar("site_m")
        self.assertEqual(side["resume_key"], stat_side["resume_key"])
        self.assertEqual(
            (side["chain"], side["chain_name"], side["temporal_reducer"]),
            (["temporal:mean"], "m", "mean"),
        )

    def test_the_water_mask_option_precedes_every_chain(self):
        water = {ndvi.WATER_DATASET: Image({"occurrence": 0.0})}
        self.run_engine(
            SCENES, water, temporal_reducers="max", water_mask=True, **WINDOW
        )
        key = self.sidecar("site_max")["resume_key"]
        for chain, water_mask in (
            ("w: temporal:max", True),
            ("w: water_mask, temporal:max", False),
            ("w: water_mask, temporal:max", True),
        ):
            self.run_chains([chain], images=water, water_mask=water_mask)
            side = self.sidecar("site_w")
            self.assertEqual(side["chain"], ["water_mask", "temporal:max"])
            self.assertEqual(side["resume_key"], key)

    def test_neighbourhoods_before_and_after_the_temporal_step(self):
        """Acceptance 2: per-scene focal then temporal equals numpy; the
        reverse order differs where the cloud masks differ."""
        res = self.run_chains(
            [
                f"scenes: focal:mean:{RADIUS}, temporal:mean",
                f"composite: temporal:mean, focal:mean:{RADIUS}",
            ]
        )
        scenes, X, Y = self.raster("site_scenes")
        expected = _mean(focal(A, X, Y, RADIUS, RES), focal(B, X, Y, RADIUS, RES))
        np.testing.assert_allclose(scenes, expected, rtol=1e-6)

        composite, _, _ = self.raster("site_composite")

        def temporal(X, Y):
            return np.nanmean(np.stack([A(X, Y), B(X, Y)]), axis=0)

        expected = focal(temporal, X, Y, RADIUS, RES)
        np.testing.assert_allclose(
            composite, np.where(np.isnan(expected), -9999.0, expected), rtol=1e-6
        )
        both = (scenes != -9999) & (composite != -9999)
        self.assertGreater(np.abs(scenes - composite)[both].max(), 0.01)

        sides = [self.sidecar(f"site_{n}") for n in ("scenes", "composite")]
        self.assertNotEqual(sides[0]["resume_key"], sides[1]["resume_key"])
        self.assertEqual(self.fake.exports, [(("scenes", "composite"), "float64")])
        self.assertEqual(res["reducers"]["scenes"]["status"], "success")

    def test_water_mask_on_the_composite_keeps_water_in_neighbourhoods(self):
        """Acceptance 3."""
        middle = lambda X, Y: (X >= THIRD_1) & (X < THIRD_2)  # noqa: E731
        nir = Field(lambda X, Y: np.where(middle(X, Y), 0.06, 0.40))
        scenes = {E.LANDSAT8_COLLECTION_ID: [landsat_sr("2015-07-01", 5, 0.05, nir)]}
        water = {
            ndvi.WATER_DATASET: Image(
                {"occurrence": lambda X, Y: np.where(middle(X, Y), 80.0, 0.0)}
            )
        }
        self.run_chains(
            [
                f"after: focal:mean:{RADIUS}, temporal:mean, water_mask",
                f"before: water_mask, focal:mean:{RADIUS}, temporal:mean",
            ],
            collections=scenes,
            images=water,
        )
        after, X, Y = self.raster("site_after")
        before, _, _ = self.raster("site_before")
        is_water = middle(X, Y)
        land = _ndvi(0.05, 0.40)
        for arr in (after, before):
            self.assertTrue((arr[is_water] == -9999).all())

        def scene(X, Y):
            return _ndvi(0.05, nir(X, Y))

        expected = focal(scene, X, Y, RADIUS, RES)
        np.testing.assert_allclose(after[~is_water], expected[~is_water], rtol=1e-6)
        np.testing.assert_allclose(before[~is_water], land, rtol=1e-6)
        shore = ~is_water & (np.abs(after - land) > 1e-3)
        self.assertTrue(shore.any(), "land beside water reads the water")


if __name__ == "__main__":
    unittest.main()
