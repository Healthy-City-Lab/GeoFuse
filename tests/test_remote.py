"""Remote links (``geofuse.jobs.remote``, ``geofuse remote``).

A stand-in ``ssh`` runs the remote command on this machine, so a link is a real
agent process behind a real pipe. ``attach`` mirrors the remote jobs and their
logs, forwards control requests and fetches outputs; a dropped link reads as
stale and reopens; requests without a link are not sent. The configuration
checks remote tables, and the commands add, list, remove and fetch.
"""

import contextlib
import io
import os
import shlex
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from test_agent import make_state

from geofuse import userconfig
from geofuse.cli import commands as cli
from geofuse.jobs.remote import Mirror, MirrorSource, Remote, attach
from geofuse.jobs.state import StateDir

FAKE_SSH = """
import shlex, subprocess, sys
sys.exit(subprocess.run(shlex.split(sys.argv[-1])).returncode)
"""


def _remote(tmp: Path, remote_state: StateDir, name: str = "lab") -> Remote:
    fake = tmp / "fake_ssh.py"
    fake.write_text(FAKE_SSH, encoding="utf-8")
    return Remote(
        name=name,
        host="lab.example.org",
        user="me",
        ssh=[sys.executable, str(fake)],
        command=f"{shlex.quote(sys.executable)} -m geofuse",
        state_dir=str(remote_state.root),
    )


def wait_for(condition, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError("timed out")


class _LinkCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ, {"PYTHONPATH": ROOT})
        env.start()
        self.addCleanup(env.stop)
        tmp = Path(self.tmp.name)
        self.remote_state, self.done, self.queued, self.out = make_state(tmp)
        self.local = StateDir(tmp / "local")
        self.remote = _remote(tmp, self.remote_state)
        self.source = MirrorSource(self.local, "lab", stale_after_s=5.0)

    def start_link(self):
        self.stop = threading.Event()
        self.clients = []
        self.link = threading.Thread(
            target=attach,
            args=(self.remote, self.local),
            kwargs={
                "stop": self.stop,
                "interval_s": 0.2,
                "retry_s": (0.2, 0.5),
                "on_connect": self.clients.append,
            },
            daemon=True,
        )
        self.link.start()
        self.addCleanup(self.close_link)
        wait_for(lambda: len(self.source.jobs()) == 2)

    def close_link(self):
        self.stop.set()
        self.link.join(20)


class AttachTests(_LinkCase):
    def test_jobs_and_logs_are_mirrored(self):
        self.remote_state.log_path(self.queued).write_bytes(b"first\n")
        self.start_link()
        link = self.source.link()
        self.assertEqual((link["state"], link["stale"]), ("connected", False))
        self.assertEqual(link["remote_state"], str(self.remote_state.root))
        states = {j.id: j.state for j in self.source.jobs()}
        self.assertEqual(states, {self.done: "completed", self.queued: "queued"})
        self.assertIn("lab · me@lab.example.org · connected", self.source.label)
        text, _ = self.source.read_log(self.done, None)
        self.assertTrue(text.endswith("line 19999\n"))
        with open(self.remote_state.log_path(self.queued), "ab") as f:
            f.write(b"second\n")
        wait_for(lambda: "second" in self.source.read_log(self.queued, None)[0])
        self.assertEqual(self.source.read_log(self.queued, None)[0], "first\nsecond\n")

    def test_requests_travel_over_the_link(self):
        self.start_link()
        message = self.source.control(self.queued, pause=True)
        self.assertEqual(message, "pause requested")
        self.assertTrue(self.remote_state.read_control(self.queued)["pause"])
        listed = sorted(Path(f["path"]).name for f in self.source.outputs(self.done))
        self.assertEqual(listed, ["a.tif", "b.tif", "city_ndvi.tif"])
        self.assertIn("fetching into", self.source.fetch(self.done))
        wait_for(lambda: len(self.source.fetched(self.done)) == 3)
        folder = Mirror(self.local, "lab").outputs_dir(self.done)
        self.assertEqual(
            (folder / "city_ndvi.tif").read_bytes(),
            (self.out / "city_ndvi.tif").read_bytes(),
        )
        self.assertEqual((folder / "tiles" / "b.tif").read_bytes(), b"B" * 20)
        self.assertIn("failed", self.source.control("zzzz", cancel=True))

    def test_a_dropped_link_reads_stale_then_reopens(self):
        import psutil

        self.start_link()
        ssh = psutil.Process(self.clients[0].proc.pid)
        for process in [*ssh.children(recursive=True), ssh]:
            process.kill()
        wait_for(lambda: self.source.link()["state"] == "reconnecting")
        self.assertTrue(self.source.link()["stale"])
        self.assertIn("stale", self.source.label)
        self.assertEqual(len(self.source.jobs()), 2)
        wait_for(lambda: len(self.clients) == 2)
        wait_for(lambda: self.source.link()["state"] == "connected")
        self.assertEqual(self.source.link()["connections"], 2)
        self.close_link()
        self.assertEqual(self.source.link()["state"], "stopped")

    def test_nothing_is_sent_without_a_link(self):
        message = self.source.control(self.queued, cancel=True)
        self.assertIn("not attached", message)
        self.assertEqual(self.source.outputs(self.done), [])
        self.assertEqual(self.remote_state.job(self.queued).state, "queued")
        self.assertFalse(Mirror(self.local, "lab").requests())


class RemoteSettingsTests(unittest.TestCase):
    def test_ssh_and_scp_commands(self):
        remote = Remote(
            name="hpc",
            host="login.example.org",
            user="me",
            port=2222,
            ssh_options=["-i", "/home/me/.ssh/key"],
            command="conda run -n geofuse geofuse",
            state_dir="/scratch/me/geofuse state",
        )
        self.assertEqual(
            remote.ssh_argv(),
            [
                "ssh", "-T", "-p", "2222", "-i", "/home/me/.ssh/key",
                "me@login.example.org",
                "conda run -n geofuse geofuse --state-dir "
                "'/scratch/me/geofuse state' agent --stdio",
            ],
        )
        self.assertEqual(
            remote.scp_argv("/scratch/out/a b.tif", "a.tif"),
            ["scp", "-q", "-P", "2222", "-i", "/home/me/.ssh/key",
             "me@login.example.org:'/scratch/out/a b.tif'", "a.tif"],
        )

    def test_remote_tables_are_checked(self):
        for table, word in (
            ({"user": "me"}, "host"),
            ({"host": "h", "colour": "red"}, "Unknown key"),
            ({"host": "h", "port": "22"}, "whole number"),
            ({"host": "h", "ssh_options": ["-v", 3]}, "text only"),
        ):
            with self.assertRaises(userconfig.ConfigError) as caught:
                userconfig._check({"remotes": {"x": table}})
            self.assertIn(word, str(caught.exception))
        userconfig._check({"remotes": {"x": {"host": "h", "ssh": ["a", "b"]}}})


class CommandTests(_LinkCase):
    def geofuse(self, *argv):
        out = io.StringIO()
        base = ["--state-dir", str(self.local.root), "--config", self.config]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = cli.main([*base, *argv])
        return code, out.getvalue()

    def setUp(self):
        super().setUp()
        self.config = str(Path(self.tmp.name) / "config.toml")

    def test_add_list_and_remove(self):
        code, out = self.geofuse(
            "remote", "add", "hpc", "login.example.org", "--user", "me",
            "--port", "2222", "--ssh-option", "-i /home/me/key",
            "--command", "conda run -n geofuse geofuse",
        )
        self.assertEqual(code, 0, out)
        table = userconfig.load(self.config).remotes["hpc"]
        self.assertEqual(table["ssh_options"], ["-i", "/home/me/key"])
        self.assertEqual(table["command"], "conda run -n geofuse geofuse")
        code, out = self.geofuse("remote", "list")
        self.assertIn("hpc: me@login.example.org", out)
        self.assertIn("link: not attached", out)
        self.assertEqual(self.geofuse("remote", "remove", "hpc")[0], 0)
        self.assertEqual(userconfig.load(self.config).remotes, {})
        code, out = self.geofuse("remote", "attach", "hpc")
        self.assertEqual(code, 2)
        self.assertIn("No remote 'hpc'", out)

    def test_fetch_copies_a_jobs_outputs(self):
        table = {
            k: v
            for k, v in vars(self.remote).items()
            if k != "name" and v not in (None, [])
        }
        userconfig.set_value("remotes.lab", table, self.config)
        dest = Path(self.tmp.name) / "fetched"
        job = self.done[:5]
        code, out = self.geofuse("remote", "fetch", "lab", job, "-o", str(dest))
        self.assertEqual(code, 0, out)
        self.assertIn("3 file(s)", out)
        self.assertEqual((dest / "tiles" / "a.tif").read_bytes(), b"A" * 10)
        code, out = self.geofuse("remote", "fetch", "lab", "zzzz")
        self.assertEqual(code, 2)
        self.assertIn("No job", out)


if __name__ == "__main__":
    unittest.main()
