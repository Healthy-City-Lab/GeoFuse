"""Exposure weights on the CGI grid: which cells an entity reaches, and how often.

Cells are 10 m squares here; cell ``(row, col)`` is
``[10·col, 10·(col + 1)] × [−10·(row + 1), −10·row]``.
"""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

from geofuse import exposure

S = 10.0


def _cells(geoms, **kw):
    ent, row, col, w = exposure.cell_weights(geoms, S, **kw)
    return {(int(e), int(r), int(c)): float(x) for e, r, c, x in zip(ent, row, col, w)}


class TestPoints(unittest.TestCase):
    def test_a_point_inside_a_cell_counts_once_there(self):
        self.assertEqual(_cells([Point(5, -5)]), {(0, 0, 0): 1.0})

    def test_a_point_on_an_edge_counts_in_both_cells(self):
        self.assertEqual(_cells([Point(10, -5)]), {(0, 0, 0): 1.0, (0, 0, 1): 1.0})

    def test_a_point_on_a_corner_counts_in_all_four(self):
        self.assertEqual(
            set(_cells([Point(10, -10)])),
            {(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)},
        )

    def test_points_of_one_entity_in_one_cell_add_up(self):
        self.assertEqual(
            _cells([MultiPoint([(2, -2), (7, -7), (3, -8)])]), {(0, 0, 0): 3.0}
        )


class TestLines(unittest.TestCase):
    def test_a_line_counts_once_per_pass_through_a_cell(self):
        # Out of cell (0, 0) to the east, back in, and out again.
        u_turn = LineString([(2, -5), (15, -5), (15, -7), (3, -7), (3, -15)])
        got = _cells([u_turn])
        self.assertEqual(got[(0, 0, 0)], 2.0)
        self.assertEqual(got[(0, 0, 1)], 1.0)
        self.assertEqual(got[(0, 1, 0)], 1.0)

    def test_a_line_along_an_edge_counts_in_both_cells(self):
        got = _cells([LineString([(2, -10), (8, -10)])])
        self.assertEqual(got, {(0, 0, 0): 1.0, (0, 1, 0): 1.0})

    def test_each_part_of_a_multiline_counts(self):
        two = MultiLineString([[(1, -5), (9, -5)], [(1, -6), (9, -6)]])
        self.assertEqual(_cells([two]), {(0, 0, 0): 2.0})


class TestPolygons(unittest.TestCase):
    def test_a_polygon_smaller_than_a_cell_reaches_every_cell_it_overlaps(self):
        tiny = box(8, -6, 12, -4)
        self.assertEqual(_cells([tiny]), {(0, 0, 0): 1.0, (0, 0, 1): 1.0})

    def test_an_edge_on_a_grid_line_does_not_reach_the_neighbour(self):
        self.assertEqual(_cells([box(0, -10, 10, 0)]), {(0, 0, 0): 1.0})

    def test_a_concave_part_overlapping_a_cell_twice_counts_once(self):
        u = Polygon(
            [(1, -1), (9, -1), (9, -9), (7, -9), (7, -3), (3, -3), (3, -9), (1, -9)]
        )
        self.assertEqual(_cells([u]), {(0, 0, 0): 1.0})

    def test_overlapping_parts_count_per_part(self):
        parts = MultiPolygon([box(0, -10, 10, 0), box(5, -10, 15, 0)])
        self.assertEqual(_cells([parts]), {(0, 0, 0): 2.0, (0, 0, 1): 1.0})


class TestEntitiesAreIndependent(unittest.TestCase):
    def test_a_shared_cell_counts_in_full_for_each_entity(self):
        got = _cells([box(0, -10, 10, 0), box(2, -8, 8, -2), Point(5, -5)])
        self.assertEqual(got, {(0, 0, 0): 1.0, (1, 0, 0): 1.0, (2, 0, 0): 1.0})

    def test_an_empty_geometry_has_no_cells(self):
        got = _cells([Point(5, -5), Polygon()])
        self.assertEqual(set(got), {(0, 0, 0)})


class TestBuffers(unittest.TestCase):
    def test_uniform_reaches_every_cell_within_the_radius(self):
        got = _cells([Point(5, -5)], kernel="uniform", radius_m=6.0)
        # The four edge neighbours are 5 m away; the diagonal ones 7.07 m.
        self.assertEqual(
            set(got), {(0, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)}
        )
        self.assertTrue(all(w == 1.0 for w in got.values()))

    def test_linear_falls_from_the_plateau_to_the_radius(self):
        got = _cells([Point(5, -5)], kernel="linear", radius_m=10.0, plateau_m=0.0)
        self.assertAlmostEqual(got[(0, 0, 0)], 1.0)
        self.assertAlmostEqual(got[(0, 0, 1)], 0.5)
        self.assertAlmostEqual(got[(0, 1, 1)], 1.0 - np.hypot(5, 5) / 10.0)

    def test_gaussian_uses_sigma_and_stops_at_the_radius(self):
        got = _cells([Point(5, -5)], kernel="gaussian", radius_m=6.0, sigma_m=5.0)
        self.assertAlmostEqual(got[(0, 0, 1)], float(np.exp(-0.5)))
        self.assertNotIn((0, 1, 1), got)

    def test_buffered_passes_of_a_line_each_contribute(self):
        # Two parallel passes 4 m south of row 0's lower edge.
        two = MultiLineString([[(0, -14), (10, -14)], [(0, -16), (10, -16)]])
        got = _cells([two], kernel="uniform", radius_m=8.0)
        self.assertEqual(got[(0, 0, 0)], 2.0)

    def test_a_bad_buffer_definition_is_refused(self):
        with self.assertRaises(ValueError):
            exposure.cell_weights([Point(0, 0)], S, kernel="cone")
        with self.assertRaises(ValueError):
            exposure.cell_weights([Point(0, 0)], S, kernel="linear", radius_m=0.0)


if __name__ == "__main__":
    unittest.main()
