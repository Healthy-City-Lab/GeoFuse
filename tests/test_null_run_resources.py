"""Time and memory of the null calibration's workers.

Pinned here:

- in-process nulls hold ``GEOFUSE_WORKERS=1`` while they run, so their sweeps
  stay in the job's process, and the previous value comes back after;
- a null frees JAX's compiled programs only once the process has grown by
  ``FREE_COMPILED_BYTES`` since it last did;
- the pilot null runs in the job's process, hands its result on and reports
  how much the process tree grew.

That the nulls' rows are identical with every pool size, in-process or not,
is pinned in ``test_outer_null.py``.
"""

import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas  # noqa: F401  (pyproj before torchvision, see vision.py)

from geofuse import parallel
from geofuse.jobs import outer_null


class TestSearchesInProcess(unittest.TestCase):
    def test_the_variable_is_held_and_restored(self):
        with mock.patch.dict(os.environ, {"GEOFUSE_WORKERS": "6"}):
            with outer_null._searches_in_process(True):
                self.assertEqual(os.environ["GEOFUSE_WORKERS"], "1")
            self.assertEqual(os.environ["GEOFUSE_WORKERS"], "6")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEOFUSE_WORKERS", None)
            with outer_null._searches_in_process(True):
                self.assertEqual(os.environ["GEOFUSE_WORKERS"], "1")
            self.assertNotIn("GEOFUSE_WORKERS", os.environ)

    def test_a_pool_leaves_it_alone(self):
        with mock.patch.dict(os.environ, {"GEOFUSE_WORKERS": "6"}):
            with outer_null._searches_in_process(False):
                self.assertEqual(os.environ["GEOFUSE_WORKERS"], "6")


class TestFreeCompiled(unittest.TestCase):
    def setUp(self):
        outer_null._FREED.clear()
        self.addCleanup(outer_null._FREED.clear)
        self.jax = types.SimpleNamespace(clear_caches=mock.Mock())
        patcher = mock.patch.dict(sys.modules, {"jax": self.jax})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _free_at(self, gb):
        with mock.patch.object(
            parallel, "process_tree_bytes", return_value=int(gb * 2**30)
        ):
            outer_null._free_compiled()

    def test_it_frees_only_after_the_process_grew(self):
        self._free_at(3.0)
        self._free_at(3.5)
        self.jax.clear_caches.assert_not_called()
        self._free_at(4.1)
        self.assertEqual(self.jax.clear_caches.call_count, 1)
        # The next threshold counts from the size after freeing.
        self._free_at(4.5)
        self.assertEqual(self.jax.clear_caches.call_count, 1)
        self._free_at(5.2)
        self.assertEqual(self.jax.clear_caches.call_count, 2)

    def test_nothing_happens_without_jax(self):
        with mock.patch.dict(sys.modules, {"jax": None}):
            outer_null._free_compiled()
        self.jax.clear_caches.assert_not_called()


class TestPilot(unittest.TestCase):
    def test_it_runs_one_null_here_and_measures_the_growth(self):
        seen = []

        def fake_null(kind, index, values):
            self.assertEqual(os.environ["GEOFUSE_WORKERS"], "1")
            self.assertIs(outer_null._WORKER["engine"], state["engine"])
            return {"kind": kind, "index": index}

        state = {"engine": object()}
        sizes = iter([1 * 2**30] + [3 * 2**30] * 1000)
        run = mock.patch.object(outer_null, "_run_null", side_effect=fake_null)
        size = mock.patch.object(
            parallel, "process_tree_bytes", side_effect=lambda: next(sizes)
        )
        with run, size:
            grown = outer_null._pilot(("free", 0, None), state, seen.append)
        self.assertEqual(seen, [{"kind": "free", "index": 0}])
        self.assertEqual(grown, 2 * 2**30)


if __name__ == "__main__":
    unittest.main()
