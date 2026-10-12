"""The user's configuration file (``geofuse.userconfig``).

Missing keys take their defaults, unknown keys and bad values are refused with
the key named, and ``set_value`` writes back a file that reads the same.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geofuse import userconfig
from geofuse.userconfig import ConfigError


class UserConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "config.toml"

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_missing_file_gives_the_defaults(self):
        cfg = userconfig.load(self.path)
        self.assertIsNone(cfg.state_dir)
        self.assertIsNone(cfg.ee_project)
        self.assertEqual(cfg.limits, userconfig.LIMIT_DEFAULTS)

    def test_set_values_round_trip(self):
        userconfig.set_value("earthengine.project", "my-proj", self.path)
        userconfig.set_value("limits.fusion", 2, self.path)
        userconfig.set_value("state_dir", "D:\\geo state", self.path)
        userconfig.set_value("remotes.cluster.host", "login.example.org", self.path)
        cfg = userconfig.load(self.path)
        self.assertEqual(cfg.ee_project, "my-proj")
        self.assertEqual(cfg.limits["fusion"], 2)
        self.assertEqual(cfg.limits["gpu"], 1)
        self.assertEqual(cfg.state_dir, "D:\\geo state")
        self.assertEqual(cfg.remotes, {"cluster": {"host": "login.example.org"}})
        userconfig.set_value("earthengine.project", None, self.path)
        self.assertIsNone(userconfig.load(self.path).ee_project)

    def test_bad_keys_and_values_are_named(self):
        for text, key in (
            ("colour = 1\n", "colour"),
            ("[limits]\ngpus = 2\n", "limits.gpus"),
            ("[limits]\ngpu = 0\n", "limits.gpu"),
            ("[earthengine]\nproject = 3\n", "earthengine.project"),
            ('[credentials]\nstore = "vault"\n', "credentials.store"),
        ):
            self.path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, key):
                userconfig.load(self.path)
        self.path.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "limits.gpu"):
            userconfig.set_value("limits.gpu", "two", self.path)


if __name__ == "__main__":
    unittest.main()
