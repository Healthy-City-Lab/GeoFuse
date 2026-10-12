"""``geofuse credentials`` and ``geofuse doctor``.

Credentials are read from a file or standard input and never printed back;
bad values exit 2 with the reason; ``doctor`` reports every check, skips the
network with ``--offline``, and recognises network filesystems.
"""

import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _credential_store import SERVICE_KEY, StoreCase

from geofuse import credentials as cr
from geofuse import ee_auth
from geofuse.cli import account
from geofuse.cli import commands as cli


class CommandTests(StoreCase):
    def geofuse(self, *argv, stdin=""):
        out, err = io.StringIO(), io.StringIO()
        with (
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
            mock.patch.object(sys, "stdin", io.StringIO(stdin)),
        ):
            code = cli.main(list(argv))
        return code, out.getvalue() + err.getvalue()

    def test_set_show_and_remove_never_print_the_value(self):
        code, out = self.geofuse(
            "credentials", "set", "streetview-key", stdin="AIzaSECRET\n"
        )
        self.assertEqual(code, 0, out)
        key_file = Path(self.tmp.name) / "sa.json"
        key_file.write_text(SERVICE_KEY, encoding="utf-8")
        code, out2 = self.geofuse(
            "credentials", "set", "earthengine-service-account", str(key_file)
        )
        self.assertEqual(code, 0, out2)
        self.assertIn("can be deleted", out2)
        code, shown = self.geofuse("credentials", "show")
        self.assertIn("runner@project.iam.gserviceaccount.com", shown)
        for text in (out, out2, shown):
            self.assertNotIn("AIzaSECRET", text)
            self.assertNotIn("PRIVATE KEY", text)
        self.assertEqual(self.geofuse("credentials", "remove", "streetview-key")[0], 0)
        self.assertIsNone(cr.get("streetview-key"))

    def test_a_bad_value_exits_2(self):
        code, out = self.geofuse("credentials", "set", "streetview-key", stdin="a b")
        self.assertEqual(code, 2)
        self.assertIn("no spaces", out)

    def test_doctor_reports_each_check(self):
        state = os.path.join(self.tmp.name, "state")
        config = os.path.join(self.tmp.name, "config.toml")
        argv = ["--state-dir", state, "--config", config, "doctor", "--json"]
        code, out = self.geofuse(*argv, "--offline")
        rows = {r["check"]: r for r in json.loads(out[out.index("[") :])}
        for check in ("Configuration", "State folder", "Resource locks", "Caches"):
            self.assertIn(check, rows)
        self.assertIn("not contacted", rows["Earth Engine"]["detail"])
        self.assertIn("use the scraper", rows["Street View"]["detail"])
        self.assertIn(code, (0, 3))
        with (
            mock.patch.object(account, "_reachable", lambda url: "no route"),
            mock.patch.object(ee_auth, "initialize", lambda *a, **k: "stub"),
        ):
            code, out = self.geofuse(*argv)
        rows = {r["check"]: r for r in json.loads(out[out.index("[") :])}
        self.assertEqual(rows["Street View"]["status"], "FAIL")
        self.assertIn("no route", rows["Street View"]["detail"])
        self.assertEqual(code, 3)


class NetworkFilesystemTests(unittest.TestCase):
    MOUNTS = [
        ("/", "ext4"),
        ("/home", "nfs4"),
        ("/home/me/scratch", "xfs"),
        ("/work", "lustre"),
    ]

    @unittest.skipIf(os.name == "nt", "mount tables are POSIX")
    def test_the_longest_mount_point_decides(self):
        detect = account.network_filesystem
        self.assertEqual(detect("/home/me/geofuse", self.MOUNTS), "nfs4")
        self.assertIsNone(detect("/home/me/scratch/geofuse", self.MOUNTS))
        self.assertEqual(detect("/work/project", self.MOUNTS), "lustre")
        self.assertIsNone(detect("/tmp/x", self.MOUNTS))
        self.assertIsNone(detect("/homework", [("/home", "nfs")]))

    @unittest.skipIf(os.name != "nt", "Windows drive types")
    def test_shares_are_network_and_local_drives_are_not(self):
        share = "\\\\server\\share\\geofuse"
        self.assertEqual(account.network_filesystem(share), "network share")
        self.assertIsNone(account.network_filesystem(os.getcwd()))


if __name__ == "__main__":
    unittest.main()
