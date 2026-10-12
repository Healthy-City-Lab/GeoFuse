"""The command line's job state folder.

Every job the command line runs lives in one folder::

    jobs/<id>/spec.json      what it runs — written once, at submission
    jobs/<id>/status.json    where it is — written only by the supervisor
                             that owns it
    jobs/<id>/control.json   pause / resume / cancel requests from any client
    jobs/<id>/claim          created by the supervisor that takes the job
    runs/<run>.json          a submitted run file and the jobs it became
    supervisors/<id>.json    each running supervisor and its heartbeat
    logs/<id>.log            each job's log

Each file is JSON replaced in one step (a temporary file, then a rename), so
a reader never sees half a file, on network filesystems too, where SQLite is
unsafe. A job that says it is running while its supervisor's heartbeat has
stopped reads as ``interrupted``.

The module imports nothing heavy: ``geofuse status`` and the monitors read
through it on every refresh.
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Statuses a job is still moving through.
ACTIVE = frozenset({"queued", "waiting", "running"})
#: Statuses a job ends in.
TERMINAL = frozenset({"completed", "error", "cancelled", "skipped", "interrupted"})
#: A supervisor heartbeat older than this means the supervisor is gone.
STALE_AFTER_S = 45.0


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_iso(text: str | None) -> float | None:
    """Seconds since the epoch of a :func:`now_iso` stamp, or ``None``."""
    if not text:
        return None
    try:
        stamp = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return None
    return stamp.replace(tzinfo=UTC).timestamp()


def default_state_dir() -> Path:
    """The per-user state folder (``%LOCALAPPDATA%\\geofuse`` on Windows,
    ``~/.local/state/geofuse`` on Linux, the Application Support folder on
    macOS)."""
    try:
        import platformdirs

        return Path(platformdirs.user_state_dir("geofuse", appauthor=False))
    except ImportError:
        return Path.home() / ".geofuse"


def write_json(path: str | os.PathLike, obj: Any) -> None:
    """Replace ``path`` with ``obj`` as JSON in one step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows refuses to replace a file another process has open for
            # reading; the reader holds it for microseconds.
            time.sleep(0.01 * (attempt + 1))
    os.replace(tmp, path)


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    """``path``'s JSON, or ``default`` when it is missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


@dataclass
class Job:
    """One job as clients see it: its spec, its status, its folder."""

    id: str
    spec: dict
    status: dict
    folder: Path

    @property
    def state(self) -> str:
        return str(self.status.get("status", "queued"))

    @property
    def name(self) -> str:
        return str(self.spec.get("name") or self.id)


class StateDir:
    """One state folder (see the module docstring)."""

    def __init__(self, root: str | os.PathLike | None = None):
        # Absolute, so a supervisor started in another folder finds the same one.
        root = Path(root).expanduser() if root is not None else default_state_dir()
        self.root = root.resolve()

    # ── Paths ───────────────────────────────────────────────────

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def supervisors_dir(self) -> Path:
        return self.root / "supervisors"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def log_path(self, job_id: str) -> Path:
        return self.logs_dir / f"{job_id}.log"

    # ── Jobs ────────────────────────────────────────────────────

    def new_id(self) -> str:
        while True:
            job_id = uuid.uuid4().hex[:8]
            if not self.job_dir(job_id).exists():
                return job_id

    def create_job(
        self,
        spec: dict,
        *,
        run_id: str,
        depends_on: list[str],
        owner: str | None = None,
        restart_of: str | None = None,
    ) -> str:
        """Write a new job's spec and its first status; return its id.

        ``spec`` is a :class:`~geofuse.jobs.plans.JobSpec` as a dict;
        ``depends_on`` names the jobs it waits for; ``owner`` reserves it for
        one supervisor (a foreground run) instead of the shared one.
        """
        job_id = self.new_id()
        folder = self.job_dir(job_id)
        folder.mkdir(parents=True)
        stamp = now_iso()
        write_json(
            folder / "spec.json",
            {
                **spec,
                "depends_on_steps": list(spec.get("depends_on") or []),
                "id": job_id,
                "run_id": run_id,
                "depends_on": list(depends_on),
                "owner": owner,
                "restart_of": restart_of,
                "submitted_at": stamp,
            },
        )
        write_json(
            folder / "status.json",
            {
                "id": job_id,
                "status": "queued",
                "progress": 0.0,
                "text": "",
                "owner": None,
                "submitted_at": stamp,
                "updated_at": stamp,
            },
        )
        return job_id

    def read_spec(self, job_id: str) -> dict | None:
        return read_json(self.job_dir(job_id) / "spec.json")

    def read_status(self, job_id: str) -> dict:
        return read_json(self.job_dir(job_id) / "status.json", {}) or {}

    def write_status(self, job_id: str, status: dict) -> None:
        status = {**status, "id": job_id, "updated_at": now_iso()}
        write_json(self.job_dir(job_id) / "status.json", status)

    def claim(self, job_id: str, supervisor_id: str) -> bool:
        """Take ``job_id`` for ``supervisor_id``; false if another has it."""
        path = self.job_dir(job_id) / "claim"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return read_json(path, {}).get("supervisor") == supervisor_id
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"supervisor": supervisor_id, "claimed_at": now_iso()}, f)
        return True

    def owner(self, job_id: str) -> str | None:
        return (read_json(self.job_dir(job_id) / "claim", {}) or {}).get("supervisor")

    def request(self, job_id: str, **flags: bool) -> dict:
        """Ask the owning supervisor to ``pause`` / ``cancel`` (true or false)."""
        path = self.job_dir(job_id) / "control.json"
        control = read_json(path, {}) or {}
        control.update({k: bool(v) for k, v in flags.items()})
        control["updated_at"] = now_iso()
        write_json(path, control)
        return control

    def read_control(self, job_id: str) -> dict:
        return read_json(self.job_dir(job_id) / "control.json", {}) or {}

    def job_ids(self) -> list[str]:
        if not self.jobs_dir.is_dir():
            return []
        return [p.name for p in self.jobs_dir.iterdir() if p.is_dir()]

    def jobs(self) -> list[Job]:
        """Every job, oldest submission first, with interruptions resolved."""
        alive = self.live_supervisors()
        out = []
        for job_id in self.job_ids():
            spec = self.read_spec(job_id)
            if spec is None:
                continue
            status = self.effective_status(job_id, self.read_status(job_id), alive)
            out.append(Job(job_id, spec, status, self.job_dir(job_id)))
        return sorted(out, key=lambda j: (j.spec.get("submitted_at") or "", j.id))

    def job(self, job_id: str) -> Job | None:
        spec = self.read_spec(job_id)
        if spec is None:
            return None
        status = self.effective_status(
            job_id, self.read_status(job_id), self.live_supervisors()
        )
        return Job(job_id, spec, status, self.job_dir(job_id))

    def find(self, prefix: str) -> list[str]:
        """Job ids starting with ``prefix`` (ids can be shortened)."""
        return sorted(i for i in self.job_ids() if i.startswith(prefix))

    def effective_status(self, job_id: str, status: dict, alive: set[str]) -> dict:
        """``status`` with a running or claimed job whose supervisor is gone
        marked ``interrupted``."""
        if status.get("status") not in ACTIVE:
            return status
        owner = self.owner(job_id)
        if owner is None or owner in alive:
            return status
        return {
            **status,
            "status": "interrupted",
            "error": status.get("error") or "Its supervisor stopped while it ran.",
        }

    def remove_job(self, job_id: str) -> None:
        import shutil

        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        try:
            self.log_path(job_id).unlink()
        except OSError:
            pass

    # ── Runs ────────────────────────────────────────────────────

    def write_run(self, run_id: str, record: dict) -> None:
        write_json(self.runs_dir / f"{run_id}.json", record)

    def read_run(self, run_id: str) -> dict | None:
        return read_json(self.runs_dir / f"{run_id}.json")

    # ── Supervisors ─────────────────────────────────────────────

    def write_supervisor(self, supervisor_id: str, record: dict) -> None:
        write_json(
            self.supervisors_dir / f"{supervisor_id}.json",
            {**record, "id": supervisor_id, "heartbeat": now_iso()},
        )

    def request_stop(self, supervisor_id: str) -> None:
        """Ask a supervisor to cancel its jobs and exit."""
        self.supervisors_dir.mkdir(parents=True, exist_ok=True)
        (self.supervisors_dir / f"{supervisor_id}.stop").touch()

    def remove_supervisor(self, supervisor_id: str) -> None:
        try:
            (self.supervisors_dir / f"{supervisor_id}.json").unlink()
        except OSError:
            pass

    def supervisor_records(self) -> Iterator[dict]:
        if not self.supervisors_dir.is_dir():
            return
        for path in self.supervisors_dir.glob("*.json"):
            record = read_json(path)
            if isinstance(record, dict):
                yield record

    def live_supervisors(self, stale_after_s: float = STALE_AFTER_S) -> set[str]:
        """Ids of supervisors whose heartbeat is fresh (and whose process is
        alive, when it runs on this machine)."""
        now = time.time()
        out = set()
        for record in self.supervisor_records():
            beat = parse_iso(record.get("heartbeat"))
            if beat is None or now - beat > stale_after_s:
                continue
            if record.get("host") == socket.gethostname() and not _pid_alive(
                record.get("pid")
            ):
                continue
            out.add(str(record.get("id")))
        return out

    def daemon(self) -> dict | None:
        """The live shared supervisor, if one runs."""
        alive = self.live_supervisors()
        for record in self.supervisor_records():
            if record.get("mode") == "daemon" and record.get("id") in alive:
                return record
        return None


def _pid_alive(pid: Any) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        pass
    if os.name == "nt":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
