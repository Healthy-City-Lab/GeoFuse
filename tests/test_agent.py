"""``geofuse agent --stdio`` and the local job source (``geofuse.jobs``).

A real agent process answers every operation over its standard streams, one
JSON line per request, and prints nothing else there; it reads only files a job
lists as outputs; bad requests get an error answer and the agent carries on.
The local source lists output folders file by file and re-runs a job from its
spec.
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geofuse.jobs import agent
from geofuse.jobs.sources import LocalSource, list_outputs, read_log_file
from geofuse.jobs.state import StateDir

SPEC = {
    "step": "ndvi.city",
    "kind": "ndvi",
    "name": "city",
    "output_dir": "out",
    "sources": [],
    "merged": False,
    "group": None,
    "options": {},
    "settings": {},
    "preset": None,
}


def make_state(root: Path) -> tuple[StateDir, str, str, Path]:
    """A state folder with one completed job (log, a file and a folder of
    outputs) and one queued job."""
    state = StateDir(root / "state")
    out = root / "out"
    (out / "tiles").mkdir(parents=True)
    (out / "city_ndvi.tif").write_bytes(bytes(range(256)) * 40)
    (out / "tiles" / "a.tif").write_bytes(b"A" * 10)
    (out / "tiles" / "b.tif").write_bytes(b"B" * 20)
    done = state.create_job(SPEC, run_id="r1", depends_on=[])
    state.write_status(
        done,
        {
            "status": "completed",
            "progress": 1.0,
            "outputs": [str(out / "city_ndvi.tif"), str(out / "tiles")],
        },
    )
    state.logs_dir.mkdir(parents=True, exist_ok=True)
    lines = "".join(f"line {i}\n" for i in range(20000))
    state.log_path(done).write_bytes(lines.encode())
    queued = state.create_job({**SPEC, "name": "town"}, run_id="r1", depends_on=[])
    return state, done, queued, out


class _AgentProcess:
    """``python -m geofuse agent --stdio`` on a state folder."""

    def __init__(self, state: StateDir):
        env = {**os.environ, "PYTHONPATH": ROOT}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "geofuse", "--state-dir", str(state.root),
             "agent", "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        self.next_id = 0

    def raw(self, line: bytes) -> dict:
        self.proc.stdin.write(line)
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def call(self, op: str, **args) -> dict:
        self.next_id += 1
        request = {"id": self.next_id, "op": op, **args}
        answer = self.raw((json.dumps(request) + "\n").encode())
        assert answer["id"] == self.next_id, answer
        return answer

    def close(self) -> bytes:
        self.proc.stdin.close()
        rest = self.proc.stdout.read()
        self.proc.wait(10)
        return rest


class AgentProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state, self.done, self.queued, self.out = make_state(Path(self.tmp.name))
        self.agent = _AgentProcess(self.state)
        self.addCleanup(lambda: self.agent.proc.poll() is None and self.agent.close())

    def test_a_round_trip_over_standard_streams(self):
        hello = self.agent.call("hello")["result"]
        self.assertEqual(hello["protocol"], agent.PROTOCOL)
        self.assertEqual(hello["state_dir"], str(self.state.root))

        snap = self.agent.call("snapshot")["result"]
        states = {j["id"]: j["status"]["status"] for j in snap["jobs"]}
        self.assertEqual(states, {self.done: "completed", self.queued: "queued"})

        tail = self.agent.call("log", job=self.done, offset=None)["result"]
        self.assertTrue(tail["text"].endswith("line 19999\n"))
        self.assertLessEqual(len(tail["text"]), 64 * 1024)
        with open(self.state.log_path(self.done), "ab") as f:
            f.write(b"one more\n")
        more = self.agent.call("log", job=self.done[:4], offset=tail["offset"])
        self.assertEqual(more["result"]["text"], "one more\n")

        files = self.agent.call("outputs", job=self.done)["result"]
        names = sorted(Path(f["path"]).name for f in files)
        self.assertEqual(names, ["a.tif", "b.tif", "city_ndvi.tif"])

        tif = str(self.out / "city_ndvi.tif")
        data, offset = b"", 0
        while True:
            chunk = self.agent.call(
                "read", job=self.done, path=tif, offset=offset, length=4000
            )["result"]
            piece = base64.b64decode(chunk["data"])
            data, offset = data + piece, offset + len(piece)
            if chunk["eof"]:
                break
        self.assertEqual(data, (self.out / "city_ndvi.tif").read_bytes())

        paused = self.agent.call("control", job=self.queued, flags={"pause": True})
        self.assertIn("requested", paused["result"]["message"])
        self.assertTrue(self.state.read_control(self.queued)["pause"])
        cancelled = self.agent.call("control", job=self.queued, flags={"cancel": True})
        self.assertEqual(cancelled["result"]["message"], "cancelled")
        self.assertEqual(self.state.job(self.queued).state, "cancelled")
        self.assertEqual(self.agent.close(), b"")

    def test_only_listed_outputs_can_be_read(self):
        secret = Path(self.tmp.name) / "secret.txt"
        secret.write_text("no", encoding="utf-8")
        for path in (str(secret), str(self.state.root / "jobs")):
            answer = self.agent.call("read", job=self.done, path=path)
            self.assertFalse(answer["ok"])
            self.assertIn("is not an output", answer["error"])

    def test_bad_requests_get_an_answer_and_the_agent_carries_on(self):
        self.assertFalse(self.agent.call("explode")["ok"])
        self.assertIn("No job", self.agent.call("log", job="zzzz")["error"])
        answer = self.agent.call("control", job=self.done, flags={"delete": True})
        self.assertIn("pause and cancel", answer["error"])
        broken = self.agent.raw(b"{not json\n")
        self.assertEqual((broken["id"], broken["ok"]), (None, False))
        self.assertTrue(self.agent.call("hello")["ok"])


class LocalSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state, self.done, self.queued, self.out = make_state(Path(self.tmp.name))
        self.source = LocalSource(self.state)

    def test_outputs_expand_folders_and_skip_missing_paths(self):
        found = list_outputs([str(self.out / "tiles"), str(self.out / "gone.tif")])
        names = sorted(Path(f["path"]).name for f in found)
        self.assertEqual(names, ["a.tif", "b.tif"])
        self.assertEqual({f["size"] for f in found}, {10, 20})
        self.assertEqual(self.source.outputs(self.queued), [])

    def test_a_log_reads_from_its_tail_then_onwards(self):
        text, offset = read_log_file(self.state.log_path(self.done), None)
        self.assertFalse(text.startswith("line 0\n"))
        self.assertEqual(read_log_file(self.state.log_path(self.done), offset)[0], "")
        self.assertEqual(read_log_file(self.state.log_path("none"), None), ("", 0))

    def test_rerun_queues_the_spec_again(self):
        with mock.patch("geofuse.jobs.supervisor.start_daemon") as started:
            message = self.source.rerun(self.done)
        started.assert_called_once()
        new_id = message.rsplit(" ", 1)[-1]
        spec = self.state.read_spec(new_id)
        self.assertEqual((spec["name"], spec["restart_of"]), ("city", self.done))
        self.assertEqual(self.source.rerun("nope"), "gone")


if __name__ == "__main__":
    unittest.main()
