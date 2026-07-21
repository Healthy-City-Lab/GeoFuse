"""Tests for year-keyed longitudinal intake and the unified raster disc path.

Covers the calendar-year wave labelling (greenery routed by each row's own
measurement date), and the raster aggregation invariants: buffered discs
reprojected into the raster CRS, nested-ring rasterization equivalence, and
uniform handling of every entity geometry type.
"""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import geopandas as gpd
import numpy as np
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from geofuse import preaggregation as pa
from geofuse.longitudinal import (
    LongitudinalSpec,
    build_long_format,
    channel_files_per_entity,
    validate_spec,
)


def _frame(dates, ids):
    return gpd.GeoDataFrame(
        {
            "pid": ids,
            "dt": dates,
            "y": np.arange(len(ids), dtype=float),
        },
        geometry=[Point(i, 0) for i in range(len(ids))],
        crs="EPSG:4326",
    )


def _spec(years, files_per_wave, greenery, **kw):
    return LongitudinalSpec(
        intake_mode=kw.pop("intake_mode", "wide"),
        entity_id_col="pid",
        wave_labels=tuple(years),
        date_col="dt",
        derive_wave_from_date=True,
        target_files_per_wave=files_per_wave,
        greenery_files={ch: dict(greenery) for ch in ("veg", "terrain", "ndvi")},
        **kw,
    )


class TestYearKeyedIntake(unittest.TestCase):
    def test_rows_take_their_own_measurement_year(self):
        """A file spanning a year boundary routes each row to its true year."""
        f1 = _frame(["2010-03-01", "2010-07-02", "2010-11-05"], ["a", "b", "c"])
        f2 = _frame(["2015-02-01", "2015-06-01", "2016-01-20"], ["a", "b", "c"])
        spec = _spec(
            ("2010", "2015", "2016"),
            {"fileA": "/f1.gpkg", "fileB": "/f2.gpkg"},
            {"2010": "g2010.tif", "2015": "g2015.tif", "2016": "g2015.tif"},
        )
        self.assertEqual(validate_spec(spec), [])
        out = build_long_format(spec, [("fileA", f1), ("fileB", f2)], outcome_col="y")
        waves = dict(zip(zip(out["entity_id"], out["wave"]), out["wave"]))
        # The third row of file B was measured in 2016, not file B's label.
        self.assertIn(("c", "2016"), waves)
        self.assertEqual(sorted(set(out["wave"])), ["2010", "2015", "2016"])

    def test_greenery_routes_per_year(self):
        f1 = _frame(["2010-03-01"], ["a"])
        f2 = _frame(["2016-01-20"], ["a"])
        spec = _spec(
            ("2010", "2016"),
            {"fileA": "/f1.gpkg", "fileB": "/f2.gpkg"},
            {"2010": "g2010.tif", "2016": "g2016.tif"},
        )
        out = build_long_format(spec, [("fileA", f1), ("fileB", f2)], outcome_col="y")
        routed = list(channel_files_per_entity(spec, out, "ndvi"))
        self.assertEqual(routed, ["g2010.tif", "g2016.tif"])

    def test_two_measurements_in_one_year_raise(self):
        f = _frame(["2010-01-01", "2010-09-09"], ["a", "a"])
        spec = _spec(("2010",), {"w": "/f.gpkg"}, {"2010": "g.tif"})
        with self.assertRaises(ValueError) as ctx:
            build_long_format(spec, [("w", f)], outcome_col="y")
        self.assertIn("same calendar year", str(ctx.exception))

    def test_unparseable_dates_are_dropped_and_counted(self):
        f = _frame(["2010-01-01", "not-a-date"], ["a", "b"])
        spec = _spec(("2010",), {"w": "/f.gpkg"}, {"2010": "g.tif"})
        out = build_long_format(spec, [("w", f)], outcome_col="y")
        self.assertEqual(len(out), 1)
        self.assertEqual(out.attrs["dropped_rows"]["unknown_wave"], 1)

    def test_wide_spec_no_longer_needs_a_file_per_wave_label(self):
        # Two files but three years: valid under year-keyed assignment.
        spec = _spec(
            ("2010", "2015", "2016"),
            {"fileA": "/f1.gpkg", "fileB": "/f2.gpkg"},
            {"2010": "g.tif", "2015": "g.tif", "2016": "g.tif"},
        )
        self.assertEqual(validate_spec(spec), [])

    def test_long_mode_without_wave_column(self):
        spec = _spec(("2010", "2012"), {}, {"2010": "g.tif", "2012": "g.tif"},
                     intake_mode="long")
        self.assertEqual(validate_spec(spec), [])
        f = _frame(["2010-05-01", "2012-05-01"], ["a", "a"])
        out = build_long_format(spec, f, outcome_col="y")
        self.assertEqual(list(out["wave"]), ["2010", "2012"])
        self.assertAlmostEqual(float(out["years_since_baseline"].iloc[0]), 0.0)

    def test_payload_round_trip(self):
        spec = _spec(
            ("2010", "2016"),
            {"fileA": "/f1.gpkg"},
            {"2010": "g.tif", "2016": "g.tif"},
        )
        back = LongitudinalSpec.from_payload(spec.to_payload())
        self.assertTrue(back.derive_wave_from_date)
        self.assertEqual(back.wave_labels, spec.wave_labels)
        self.assertEqual(
            dict(back.target_files_per_wave), dict(spec.target_files_per_wave)
        )


class TestRasterDiscAggregation(unittest.TestCase):
    """The raster path samples buffered discs reprojected into the raster CRS."""

    def setUp(self):
        from pyproj import Transformer
        from rasterio.transform import from_origin

        # Geographic raster at a high latitude, where an equatorial
        # degrees-to-metres factor would make a disc badly elliptical.
        self.transform = from_origin(-114.10, 51.10, 0.0001, 0.0001)
        rng = np.random.default_rng(0)
        self.arr = np.ma.masked_array(
            rng.random((400, 400)).astype(np.float64),
            mask=np.zeros((400, 400), dtype=bool),
        )
        self.utm = "EPSG:32611"
        self.to_raster = Transformer.from_crs(self.utm, "EPSG:4326", always_xy=True)
        fwd = Transformer.from_crs("EPSG:4326", self.utm, always_xy=True)
        self.x0, self.y0 = fwd.transform(-114.08, 51.08)
        self.radii = (100, 300, 600)

    def _reference(self, entity):
        """Per-radius geometry_mask over the reprojected buffer, one at a time."""
        import shapely
        from rasterio.features import geometry_mask

        out = np.full((len(self.radii), len(pa.STAT_COLUMNS)), np.nan, dtype=np.float32)
        for i, r in enumerate(self.radii):
            buf = entity.buffer(float(r), quad_segs=pa.BUFFER_QUAD_SEGS)
            buf_r = shapely.ops.transform(self.to_raster.transform, buf)
            m = geometry_mask(
                [buf_r],
                out_shape=self.arr.shape,
                transform=self.transform,
                invert=True,
                all_touched=True,
            )
            vals = np.ma.getdata(self.arr)[m & ~np.ma.getmaskarray(self.arr)]
            out[i, :] = pa.compute_all_stats(vals)
        return out

    def test_matches_per_radius_masks(self):
        ent = Point(self.x0, self.y0)
        got = pa.raster_batch_geometry_stats(
            self.arr, self.transform, [ent], self.radii, to_raster_crs=self.to_raster
        )[0]
        np.testing.assert_allclose(got, self._reference(ent), rtol=1e-6, equal_nan=True)

    def test_every_geometry_type_flows_through_one_path(self):
        x0, y0 = self.x0, self.y0
        geoms = [
            Point(x0, y0),
            MultiPoint([(x0, y0), (x0 + 50, y0 + 50)]),
            LineString([(x0, y0), (x0 + 200, y0)]),
            MultiLineString([[(x0, y0), (x0 + 100, y0)], [(x0, y0 + 60), (x0 + 90, y0 + 60)]]),
            Polygon([(x0, y0), (x0 + 150, y0), (x0 + 150, y0 + 150), (x0, y0 + 150)]),
            MultiPolygon(
                [
                    Polygon([(x0, y0), (x0 + 80, y0), (x0 + 80, y0 + 80), (x0, y0 + 80)]),
                    Polygon(
                        [
                            (x0 + 300, y0 + 300),
                            (x0 + 380, y0 + 300),
                            (x0 + 380, y0 + 380),
                            (x0 + 300, y0 + 380),
                        ]
                    ),
                ]
            ),
        ]
        out = pa.raster_batch_geometry_stats(
            self.arr, self.transform, geoms, self.radii, to_raster_crs=self.to_raster
        )
        self.assertEqual(out.shape, (len(geoms), len(self.radii), len(pa.STAT_COLUMNS)))
        self.assertTrue(np.isfinite(out[:, :, 0]).all())

    def test_disc_is_not_elliptical_at_latitude(self):
        """North-south and east-west reach match, which the old factor broke."""
        import shapely
        from rasterio.transform import rowcol

        r = 600.0
        buf = Point(self.x0, self.y0).buffer(r, quad_segs=pa.BUFFER_QUAD_SEGS)
        buf_r = shapely.ops.transform(self.to_raster.transform, buf)
        minx, miny, maxx, maxy = buf_r.bounds
        r_top, c_left = rowcol(self.transform, minx, maxy)
        r_bot, c_right = rowcol(self.transform, maxx, miny)
        # The disc spans the same number of metres on both axes; in a
        # geographic grid that means noticeably more columns than rows.
        self.assertGreater(abs(c_right - c_left), abs(r_bot - r_top))

    def test_nested_ring_grid_matches_individual_masks(self):
        from rasterio.features import geometry_mask
        from rasterio.transform import from_origin

        tr = from_origin(0, 500, 10, 10)
        shape = (50, 50)
        discs = [Point(250, 250).buffer(float(r), quad_segs=32) for r in (50, 120, 200)]
        ring = pa.ring_index_grid(discs, shape, tr)
        for i, g in enumerate(discs):
            ref = geometry_mask(
                [g], out_shape=shape, transform=tr, invert=True, all_touched=True
            )
            np.testing.assert_array_equal(ring <= i, ref)

    def test_point_template_matches_direct_buffer(self):
        templates = pa.origin_circle_templates([250.0])
        a = pa.buffer_at(Point(self.x0, self.y0), 250.0, templates)
        b = Point(self.x0, self.y0).buffer(250.0, quad_segs=pa.BUFFER_QUAD_SEGS)
        self.assertTrue(a.equals_exact(b, 1e-9))


class TestReportingScoreConsistency(unittest.TestCase):
    """Reporting CIs must score with the engine's own residualization basis."""

    def test_ci_observed_matches_held_out_score_under_spline(self):
        import tempfile
        import warnings

        import rasterio
        from rasterio.transform import from_origin

        from geofuse.fusion import MetricFusionEngine

        warnings.filterwarnings("ignore")
        tmp = tempfile.mkdtemp()
        rng = np.random.default_rng(4)
        x0, y0, s = -114.10, 51.00, 0.01

        cells, vals, cov = [], [], []
        for i in range(4):
            for j in range(3):
                cells.append(
                    Polygon(
                        [
                            (x0 + i * s, y0 + j * s),
                            (x0 + (i + 1) * s, y0 + j * s),
                            (x0 + (i + 1) * s, y0 + (j + 1) * s),
                            (x0 + i * s, y0 + (j + 1) * s),
                        ]
                    )
                )
                c = rng.normal()
                cov.append(c)
                # Nonlinear covariate effect, so a spline basis and a linear one
                # genuinely disagree.
                vals.append(float(c**2 + rng.normal(0, 0.2)))
        tgt = gpd.GeoDataFrame(
            {"outcome": vals, "cov1": cov}, geometry=cells, crs="EPSG:4326"
        )
        tpath = os.path.join(tmp, "t.gpkg")
        tgt.to_file(tpath, driver="GPKG")

        tr = from_origin(x0 - 0.005, y0 + 0.04, 0.0005, 0.0005)
        npath = os.path.join(tmp, "n.tif")
        with rasterio.open(
            npath,
            "w",
            driver="GTiff",
            height=140,
            width=140,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=tr,
        ) as d:
            d.write(rng.random((140, 140)).astype("float32"), 1)

        gpts = [
            Point(x0 + rng.random() * 0.04, y0 + rng.random() * 0.03)
            for _ in range(400)
        ]
        gvi = gpd.GeoDataFrame(
            {"gvi_veg": rng.random(400), "gvi_ter": rng.random(400)},
            geometry=gpts,
            crs="EPSG:4326",
        )
        gpath = os.path.join(tmp, "g.gpkg")
        gvi.to_file(gpath, driver="GPKG")

        eng = MetricFusionEngine(
            target_file=tpath,
            target_feature="outcome",
            buffer_meters=400,
            gvi_buffer_min_m=200,
            gvi_buffer_max_m=400,
            gvi_buffer_step_m=200,
            ndvi_buffer_min_m=200,
            ndvi_buffer_max_m=400,
            ndvi_buffer_step_m=200,
            cache_dir=os.path.join(tmp, "cache"),
            cgi_grid_spacing_m=250,
            covariate_columns=["cov1"],
            residualize_method="spline",
            spatial_adjust_method="ks_aic",
        )
        eng.load_target()
        eng.load_metrics(veg_file=gpath, ndvi_file=npath, cache_metrics=False)
        df = eng.prepare_fusion_data()
        eng.precompute_aggregations()
        eng.split_data(test_size=0.3, random_state=5, fusion_df=df)

        params = {
            "veg_weight": 30,
            "terrain_weight": 30,
            "ndvi_weight": 40,
            "streetview_stat": "mean",
            "streetview_percentile": 50,
            "veg_radius": 400,
            "terrain_radius": 400,
            "ndvi_stat": "mean",
            "ndvi_percentile": 50,
            "ndvi_radius": 400,
        }
        held_out = eng.evaluate_on_test(params=params, metric="distance_corr")
        ci = eng.bootstrap_test_score_ci(
            params, "distance_corr", n_bootstrap=20, method="percentile", seed=1
        )
        self.assertAlmostEqual(
            float(held_out["test_score"]), float(ci["observed"]), places=10
        )


if __name__ == "__main__":
    unittest.main()
