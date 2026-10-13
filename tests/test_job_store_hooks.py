"""Two hooks the command line's supervisor needs from the job machinery.

``JobStore.submit`` takes the id a job already has, so a supervisor's jobs keep
the ids of its state folder; ``logger.set_log_dir`` moves per-job log files
out of the working directory's ``logs/jobs``.
"""

import os
import sys
import tempfile
import time
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geofuse import logger
from geofuse.persistence.job_store import JobStore


class JobStoreIdTests(unittest.TestCase):
    def test_a_given_id_is_kept_and_a_missing_one_drawn(self):
        store = JobStore(":memory:")
        try:
            self.assertEqual(store.submit("ndvi", {}, job_id="abc12345").id, "abc12345")
            self.assertEqual(store.get("abc12345").status, "queued")
            self.assertEqual(len(store.submit("gvi", {}).id), 8)
        finally:
            store.shutdown()


class LogDirTests(unittest.TestCase):
    def test_job_logs_follow_the_log_dir(self):
        before = logger._LOG_DIR
        with tempfile.TemporaryDirectory() as d:
            try:
                logger.set_log_dir(d)
                self.assertEqual(
                    logger.get_job_log_path("j1"), os.path.join(d, "j1.log")
                )
                logger.bind_job_log_buffer("j1")
                logger.get_logger("TEST")("INFO", "hello")
                for _ in range(100):
                    path = os.path.join(d, "j1.log")
                    if os.path.isfile(path) and os.path.getsize(path):
                        break
                    time.sleep(0.02)
                with open(os.path.join(d, "j1.log"), encoding="utf-8") as f:
                    self.assertIn("[TEST INFO] hello", f.read())
            finally:
                logger.unbind_job_log_buffer()
                logger.drop_job_log_buffer("j1")
                logger.set_log_dir(before)


if __name__ == "__main__":
    unittest.main()
