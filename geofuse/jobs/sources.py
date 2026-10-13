"""Where a client reads jobs from.

A source answers the same questions wherever its jobs run: the jobs, the
supervisor, a job's log and its output files; and takes the same requests:
pause / resume / cancel and re-run. :class:`LocalSource` reads a state folder
on this machine; ``geofuse.jobs.remote.MirrorSource`` reads the copy of a
remote one that ``geofuse remote attach`` keeps. ``geofuse watch``, the GUI's
Jobs tab and the agent (``geofuse.jobs.agent``) all read through one.

The module imports nothing heavy; clients read through it every second.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from geofuse.jobs.state import Job, StateDir

#: How much of a log a client shows when a job is first selected.
LOG_TAIL_BYTES = 64 * 1024
#: Files listed per output folder; a folder holding more is cut there.
MAX_LISTED_FILES = 2000


class JobSource(Protocol):
    label: str

    def jobs(self) -> list[Job]: ...

    def supervisor(self) -> dict | None: ...

    def control(self, job_id: str, **flags: bool) -> str: ...

    def rerun(self, job_id: str) -> str: ...

    def read_log(self, job_id: str, offset: int | None) -> tuple[str, int]: ...

    def outputs(self, job_id: str) -> list[dict]: ...

    def link(self) -> dict: ...


def read_log_file(path: Path, offset: int | None) -> tuple[str, int]:
    """The whole lines of a log from ``offset`` on (its last
    :data:`LOG_TAIL_BYTES` when ``offset`` is ``None`` or past the end), and
    the offset to read from next."""
    try:
        size = path.stat().st_size
    except OSError:
        return "", 0
    tail = offset is None or offset > size
    if tail:
        offset = max(0, size - LOG_TAIL_BYTES)
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    data = data[: data.rfind(b"\n") + 1]
    text = data.decode("utf-8", errors="replace")
    if tail and offset:
        text = text.split("\n", 1)[-1]
    return text, offset + len(data)


def list_outputs(paths: list) -> list[dict]:
    """Every file a job's recorded outputs name, a folder expanded to its
    files: ``{"path", "size", "mtime"}``, missing paths left out."""
    found: list[dict] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path)
        if key in seen:
            return
        try:
            info = path.stat()
        except OSError:
            return
        seen.add(key)
        found.append({"path": key, "size": info.st_size, "mtime": info.st_mtime})

    for raw in paths or []:
        path = Path(str(raw))
        if path.is_dir():
            listed = 0
            for root, _dirs, files in os.walk(path):
                for name in sorted(files):
                    if listed >= MAX_LISTED_FILES:
                        break
                    add(Path(root) / name)
                    listed += 1
        elif path.exists():
            add(path)
    return found


class LocalSource:
    """The jobs of a state folder on this machine."""

    kind = "local"

    def __init__(self, state: StateDir):
        self.state = state
        self.label = str(state.root)

    def jobs(self) -> list[Job]:
        return self.state.jobs()

    def supervisor(self) -> dict | None:
        return self.state.daemon()

    def control(self, job_id: str, **flags: bool) -> str:
        return self.state.control(job_id, **flags)

    def rerun(self, job_id: str) -> str:
        """Queue the job again from its spec on the shared supervisor."""
        from geofuse.jobs.supervisor import resubmit, start_daemon

        if self.state.read_spec(job_id) is None:
            return "gone"
        (new_id,) = resubmit(self.state, [job_id])
        start_daemon(self.state)
        return f"queued again as {new_id}"

    def read_log(self, job_id: str, offset: int | None) -> tuple[str, int]:
        return read_log_file(self.state.log_path(job_id), offset)

    def outputs(self, job_id: str) -> list[dict]:
        job = self.state.job(job_id)
        return list_outputs(job.status.get("outputs") or []) if job else []

    def link(self) -> dict:
        return {"state": "local"}
