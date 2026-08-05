"""Regressions for the native file picker's process isolation.

The bug this guards against: Tk wants to own the main thread of its
interpreter, and Streamlit runs the script body and every ``on_click``
callback on a per-rerun ScriptRunner thread. Calling ``tkinter`` there raises
``RuntimeError: main thread is not in main loop`` as soon as any Tk state
outlives the thread that made it. The dialog therefore runs in a child
process, and these tests pin the parts of that arrangement that can silently
regress: the child's wire protocol, its behaviour when there is no Tk at all,
and the parent's refusal to raise into the Streamlit page.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHILD = os.path.join(REPO, "ui", "file_picker_child.py")


def _request(**overrides) -> str:
    payload = {
        "title": "probe",
        "file_types": [["All files", "*.*"]],
        "initial_dir": REPO,
        "multi": False,
    }
    payload.update(overrides)
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(payload, handle)
    handle.close()
    return handle.name


class TestChildProtocol(unittest.TestCase):
    def test_missing_arguments_produce_an_empty_selection(self):
        proc = subprocess.run(
            [sys.executable, CHILD], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(json.loads(proc.stdout.strip()), [])

    def test_unreadable_request_reports_an_error_object(self):
        proc = subprocess.run(
            [sys.executable, CHILD, os.path.join(REPO, "does-not-exist.json")],
            capture_output=True, text=True, timeout=60,
        )
        self.assertIn("error", json.loads(proc.stdout.strip()))

    def test_absent_tkinter_reports_an_error_rather_than_a_traceback(self):
        stub = (
            "import sys; sys.modules['tkinter'] = None; "
            f"exec(open(r'{CHILD}').read())"
        )
        path = _request()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", stub, path],
                capture_output=True, text=True, timeout=60,
            )
        finally:
            os.unlink(path)
        payload = json.loads(proc.stdout.strip())
        self.assertIn("tkinter unavailable", payload.get("error", ""))

    def test_dialog_result_round_trips_as_json(self):
        """The real Tk path, with the dialog auto-answered.

        Proves Tk initialises inside the child and the selection comes back
        over the wire — the whole point of the subprocess.
        """
        stub = (
            "import sys, tkinter.filedialog as fd\n"
            "fd.askopenfilename = lambda **kw: r'C:/picked/file.gpkg'\n"
            "fd.askopenfilenames = lambda **kw: (r'C:/picked/a.gpkg',)\n"
            f"exec(open(r'{CHILD}').read())"
        )
        path = _request()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", stub, path],
                capture_output=True, text=True, timeout=120,
            )
        finally:
            os.unlink(path)
        if proc.returncode == 3:
            self.skipTest("no Tk build available on this machine")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout.strip()), ["C:/picked/file.gpkg"])

    def test_child_runs_from_a_non_main_thread(self):
        """The exact condition Streamlit creates."""
        result: dict = {}

        def worker() -> None:
            stub = (
                "import sys, tkinter.filedialog as fd\n"
                "fd.askopenfilename = lambda **kw: r'C:/picked/thread.gpkg'\n"
                f"exec(open(r'{CHILD}').read())"
            )
            path = _request()
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", stub, path],
                    capture_output=True, text=True, timeout=120,
                )
                result["rc"] = proc.returncode
                result["out"] = proc.stdout.strip()
            finally:
                os.unlink(path)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(180)
        self.assertFalse(thread.is_alive(), "picker child hung")
        if result.get("rc") == 3:
            self.skipTest("no Tk build available on this machine")
        self.assertEqual(result.get("rc"), 0)
        self.assertEqual(json.loads(result["out"]), ["C:/picked/thread.gpkg"])


class TestParentDegradesGracefully(unittest.TestCase):
    """A dialog that cannot open must not take the Streamlit page down."""

    def _picker(self):
        sys.path.insert(0, os.path.join(REPO, "ui"))
        import file_picker

        return file_picker

    def test_missing_helper_raises_picker_unavailable_not_oserror(self):
        fp = self._picker()
        real = os.path.isfile
        os.path.isfile = lambda p: False
        try:
            with self.assertRaises(fp.PickerUnavailable):
                fp._open_tk_picker(
                    "t", file_types=[("All", "*.*")], initial_dir=REPO, multi=False
                )
        finally:
            os.path.isfile = real

    def test_error_object_from_child_becomes_picker_unavailable(self):
        fp = self._picker()
        real_run = subprocess.run

        class _Result:
            returncode = 3
            stdout = json.dumps({"error": "tkinter unavailable: no display"})
            stderr = ""

        subprocess.run = lambda *a, **k: _Result()
        try:
            with self.assertRaises(fp.PickerUnavailable) as ctx:
                fp._open_tk_picker(
                    "t", file_types=[("All", "*.*")], initial_dir=REPO, multi=False
                )
            self.assertIn("no display", str(ctx.exception))
        finally:
            subprocess.run = real_run


if __name__ == "__main__":
    unittest.main(verbosity=2)
