"""Standalone native file dialog, run as its own process.

Tk requires every call into an interpreter to come from the thread that owns
it, and in practice it wants that to be the process's **main** thread. Streamlit
gives us neither: the script body and every ``on_click`` callback run on a
ScriptRunner thread, and a fresh one is created per rerun. Opening Tk there
raises ``RuntimeError: main thread is not in main loop`` as soon as any Tk
state outlives the thread that made it.

Running the dialog in a separate process sidesteps the whole problem — the
child has its own main thread, its own interpreter, and is torn down when the
user picks a file, so nothing can leak into the next rerun.

Protocol: a JSON request file path as ``argv[1]``; the selected paths are
written to stdout as a JSON list. Anything else on stdout is a bug, so the
parent parses only the last non-empty line.

    {"title": str, "file_types": [[label, pattern], ...],
     "initial_dir": str | null, "multi": bool}

This file is executed directly (``python file_picker_child.py request.json``),
not imported, so it must not depend on the ``ui`` package being importable.
"""

from __future__ import annotations

import json
import os
import sys


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("[]")
        return 2
    try:
        with open(argv[1], encoding="utf-8") as fh:
            request = json.load(fh)
    except Exception as exc:  # noqa: BLE001 - reported to the parent, not raised
        print(json.dumps({"error": f"could not read request: {exc}"}))
        return 2

    title = str(request.get("title") or "Select a file")
    file_types = [tuple(ft) for ft in (request.get("file_types") or [])]
    initial_dir = request.get("initial_dir") or os.path.expanduser("~")
    multi = bool(request.get("multi"))

    try:
        from tkinter import Tk, filedialog
    except Exception as exc:  # noqa: BLE001 - headless / no Tk build
        print(json.dumps({"error": f"tkinter unavailable: {exc}"}))
        return 3

    root = Tk()
    try:
        root.withdraw()
        # Without these the dialog can open behind the browser window, which
        # looks exactly like the picker having silently failed.
        root.wm_attributes("-topmost", True)
        root.update()
        if multi:
            picked = filedialog.askopenfilenames(
                title=title, initialdir=initial_dir, filetypes=file_types
            )
            paths = list(picked) if picked else []
        else:
            single = filedialog.askopenfilename(
                title=title, initialdir=initial_dir, filetypes=file_types
            )
            paths = [single] if single else []
    except Exception as exc:  # noqa: BLE001 - reported to the parent
        print(json.dumps({"error": f"dialog failed: {exc}"}))
        return 4
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    print(json.dumps([p for p in paths if p]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
