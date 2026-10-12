"""The command line's supervisor: runs submitted jobs and reports them.

A supervisor claims queued jobs from the state folder
(:mod:`geofuse.jobs.state`), starts each when the jobs it depends on have
completed and its resources are free, and runs it exactly as the GUI does —
built by :func:`geofuse.jobs.plans.build`, then handed to a
:class:`~geofuse.persistence.job_executor.JobExecutor`, so every job runs in
its own spawn process. It mirrors each job's progress into ``status.json``
and applies the pause / resume / cancel requests clients write.

Two kinds run:

* the shared **daemon**, started detached by the first ``geofuse submit``,
  takes every job no one else reserved and exits after a quiet period;
* a **foreground** supervisor (``geofuse run``) takes only the jobs of its
  own run and exits when they end — the form a batch script on a cluster
  uses.

Resource classes cap what runs at once (``[limits]`` in the user
configuration): ``gpu`` and ``streetview`` (one GVI job per machine by
default — Google throttles per address), ``earthengine``, ``fusion`` and
``max_parallel`` overall.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from geofuse.jobs.state import ACTIVE, TERMINAL, StateDir, now_iso

REPO_ROOT = Path(__file__).resolve().parents[2]
PANO_CACHE_DB = str(REPO_ROOT / "logs" / "caches" / "gvi_panos.db")

# Engine extras worth carrying into status.json (small and JSON-safe).
_EXTRA_KEYS = (
    "total_points",
    "point_breakdown",
    "ndvi_tile_bracket",
    "gvi_progress",
    "fusion_study_progress",
    "preaggr_progress",
    "artifacts_dir",
)
_STATUS_EVERY_S = 2.0
_HEARTBEAT_EVERY_S = 5.0
_SCAN_EVERY_S = 1.0


def submit_specs(
    state: StateDir,
    specs: list,
    *,
    run_name: str,
    run_file: str | None = None,
    owner: str | None = None,
    restart_of: dict[int, str] | None = None,
) -> tuple[str, list[str]]:
    """Write ``specs`` (:class:`~geofuse.jobs.plans.JobSpec`) as queued jobs.

    A spec's ``depends_on`` names steps; each becomes the ids of that step's
    jobs in this submission. ``owner`` reserves the jobs for one supervisor.
    Returns ``(run_id, job_ids)``.
    """
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:4]}"
    ids_by_step: dict[str, list[str]] = {}
    job_ids = []
    for i, spec in enumerate(specs):
        depends = [j for step in spec.depends_on for j in ids_by_step.get(step, [])]
        job_id = state.create_job(
            spec.to_dict(),
            run_id=run_id,
            depends_on=depends,
            owner=owner,
            restart_of=(restart_of or {}).get(i),
        )
        ids_by_step.setdefault(spec.step, []).append(job_id)
        job_ids.append(job_id)
    state.write_run(
        run_id,
        {
            "id": run_id,
            "name": run_name,
            "run_file": run_file,
            "jobs": job_ids,
            "owner": owner,
            "submitted_at": now_iso(),
        },
    )
    return run_id, job_ids


def _streetview_key() -> str | None:
    try:
        from geofuse import credentials
    except ImportError:
        return None
    return credentials.get("streetview-key")


class Supervisor:
    """Claims and runs jobs of one state folder (see the module docstring)."""

    def __init__(
        self,
        state: StateDir,
        *,
        mode: str = "daemon",
        limits: dict[str, int] | None = None,
        ee_project: str | None = None,
        executor: Any = None,
        build: Callable | None = None,
        supervisor_id: str | None = None,
        poll_s: float = 0.5,
        log: Callable[[str], None] | None = None,
    ):
        from geofuse.userconfig import LIMIT_DEFAULTS

        self.state = state
        self.mode = mode
        self.id = supervisor_id or f"{mode[0]}{uuid.uuid4().hex[:7]}"
        self.limits = {**LIMIT_DEFAULTS, **(limits or {})}
        self.ee_project = ee_project
        self.poll_s = poll_s
        self._log = log or (lambda msg: None)
        self._executor = executor
        self._build = build
        self._store = None
        self.jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cancel_all = threading.Event()
        self._last_scan = 0.0
        self._last_beat = 0.0
        self._idle_since: float | None = None

    # ── Setup ───────────────────────────────────────────────────

    def _ensure_executor(self):
        if self._executor is None:
            from geofuse.logger import set_log_dir
            from geofuse.persistence.job_executor import JobExecutor
            from geofuse.persistence.job_store import JobStore

            self.state.logs_dir.mkdir(parents=True, exist_ok=True)
            set_log_dir(str(self.state.logs_dir))
            self._store = JobStore(":memory:")
            self._executor = JobExecutor(
                self._store, max_workers=max(4, self.limits["max_parallel"])
            )
        else:
            self._store = self._executor.store
        if self._build is None:
            from geofuse.jobs.plans import build

            self._build = build

    def heartbeat(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_beat < _HEARTBEAT_EVERY_S:
            return
        self._last_beat = now
        self.state.write_supervisor(
            self.id,
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "mode": self.mode,
                "started_at": getattr(self, "_started_at", now_iso()),
                "running": sorted(
                    j for j, e in self.jobs.items() if e["phase"] == "run"
                ),
            },
        )

    # ── Claiming ────────────────────────────────────────────────

    def _scan(self) -> None:
        now = time.monotonic()
        if now - self._last_scan < _SCAN_EVERY_S:
            return
        self._last_scan = now
        for job_id in self.state.job_ids():
            if job_id in self.jobs:
                continue
            spec = self.state.read_spec(job_id)
            if spec is None:
                continue
            owner = spec.get("owner")
            if owner not in (None, self.id):
                continue
            if self.mode != "daemon" and owner != self.id:
                continue
            if self.state.read_status(job_id).get("status") != "queued":
                continue
            if not self.state.claim(job_id, self.id):
                continue
            self.jobs[job_id] = {"spec": spec, "phase": "queued", "record": None}
            self._write(job_id, status="queued", text="Queued")
            self._log(f"claimed {job_id} ({spec.get('kind')} {spec.get('name')})")

    # ── Status ──────────────────────────────────────────────────

    def _write(self, job_id: str, **fields) -> None:
        entry = self.jobs[job_id]
        spec = entry["spec"]
        current = entry.setdefault("status", {})
        current.update(
            {
                "name": spec.get("name"),
                "kind": spec.get("kind"),
                "step": spec.get("step"),
                "run_id": spec.get("run_id"),
                "submitted_at": spec.get("submitted_at"),
                "owner": self.id,
                **fields,
            }
        )
        self.state.write_status(job_id, current)
        entry["written"] = time.monotonic()

    def _mirror(self, job_id: str, *, force: bool = False) -> None:
        entry = self.jobs[job_id]
        rec = entry["record"]
        if rec is None:
            return
        if not force and time.monotonic() - entry.get("written", 0) < _STATUS_EVERY_S:
            return
        extra = {k: rec.extra[k] for k in _EXTRA_KEYS if rec.extra.get(k) is not None}
        status = {"queued": "running"}.get(rec.status, rec.status)
        self._write(
            job_id,
            status=status,
            paused=bool(rec.pause_event.is_set()) and status == "running",
            progress=float(rec.progress),
            text=rec.status_text,
            stage_ledger=rec.stage_ledger,
            extra=extra,
            started_at=rec.started_at or entry["status"].get("started_at"),
            completed_at=rec.completed_at,
            error=rec.error,
            error_detail=rec.extra.get("error_detail"),
            outputs=list(rec.output_paths),
        )

    # ── Scheduling ──────────────────────────────────────────────

    def _in_use(self) -> dict[str, int]:
        use: dict[str, int] = {"max_parallel": 0}
        for entry in self.jobs.values():
            if entry["phase"] in ("build", "run"):
                use["max_parallel"] += 1
                for r in entry["spec"].get("resources") or []:
                    use[r] = use.get(r, 0) + 1
        return use

    def _fits(self, spec: dict, use: dict[str, int]) -> bool:
        if use["max_parallel"] >= self.limits["max_parallel"]:
            return False
        return all(
            use.get(r, 0) < self.limits.get(r, 1) for r in spec.get("resources") or []
        )

    def _dependencies(self, spec: dict) -> tuple[str, str | None]:
        """``("ready" | "wait" | "skip", reason)`` for a job's dependencies."""
        for dep in spec.get("depends_on") or []:
            job = self.state.job(dep)
            state = job.state if job else "missing"
            if state == "completed":
                continue
            if state in TERMINAL or state == "missing":
                return "skip", f"Not run: dependency {dep} ended {state}."
            return "wait", f"Waiting for {dep}"
        return "ready", None

    def _schedule(self) -> None:
        for job_id, entry in list(self.jobs.items()):
            if entry["phase"] != "queued":
                continue
            control = self.state.read_control(job_id)
            if control.get("cancel") or self._cancel_all.is_set():
                entry["phase"] = "done"
                self._write(job_id, status="cancelled", text="", completed_at=now_iso())
                continue
            if control.get("pause"):
                if entry["status"].get("text") != "Held":
                    self._write(job_id, status="queued", paused=True, text="Held")
                continue
            verdict, reason = self._dependencies(entry["spec"])
            if verdict == "skip":
                entry["phase"] = "done"
                self._write(
                    job_id, status="skipped", error=reason, completed_at=now_iso()
                )
                continue
            if verdict == "wait":
                if entry["status"].get("status") != "waiting":
                    self._write(job_id, status="waiting", paused=False, text=reason)
                continue
            if not self._fits(entry["spec"], self._in_use()):
                if entry["status"].get("text") != "Waiting for a free slot":
                    self._write(
                        job_id, status="queued", paused=False,
                        text="Waiting for a free slot",
                    )
                continue
            entry["phase"] = "build"
            self._write(
                job_id, status="running", paused=False, text="Preparing inputs...",
                started_at=now_iso(),
            )
            threading.Thread(
                target=self._start, args=(job_id,), daemon=True,
                name=f"build-{job_id}",
            ).start()

    def _start(self, job_id: str) -> None:
        from geofuse.jobs.plans import JobSpec

        entry = self.jobs[job_id]
        spec_dict = dict(entry["spec"])
        try:
            fields = JobSpec.__dataclass_fields__
            spec = JobSpec.from_dict(
                {k: v for k, v in spec_dict.items() if k in fields}
            )
            if spec.kind == "fusion" and spec.settings.get(
                "clear_preaggregation_cache"
            ):
                from geofuse.persistence import caches

                caches.remove_files(caches.greenery_cache_files(spec.output_dir))
            sub = self._build(
                spec,
                api_key=_streetview_key(),
                pano_cache_db_path=PANO_CACHE_DB,
                ee_project=self.ee_project,
            )
            os.makedirs(spec.output_dir, exist_ok=True)
            with self._lock:
                rec = self._store.submit(
                    sub.record_type, sub.params, name=sub.record_name, job_id=job_id
                )
                method = getattr(self._executor, sub.method)
                if sub.method == "submit_fusion_subprocess":
                    method(rec, sub.kwargs["run_kwargs"])
                else:
                    method(rec, **sub.kwargs)
                entry["record"] = rec
                entry["phase"] = "run"
            self._log(f"started {job_id}")
        except Exception as exc:  # noqa: BLE001 — a bad job must not stop the rest
            import traceback

            entry["phase"] = "done"
            self._write(
                job_id,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                error_detail=traceback.format_exc(),
                completed_at=now_iso(),
            )
            self._log(f"failed to start {job_id}: {exc}")

    def _control(self) -> None:
        for job_id, entry in list(self.jobs.items()):
            if entry["phase"] != "run":
                continue
            rec = entry["record"]
            control = self.state.read_control(job_id)
            if control.get("cancel") or self._cancel_all.is_set():
                if not rec.cancel_event.is_set():
                    self._store.request_cancel(job_id)
                    self._log(f"cancel requested for {job_id}")
            elif control.get("pause") and not rec.pause_event.is_set():
                self._store.request_pause(job_id)
                self._mirror(job_id, force=True)
            elif not control.get("pause") and rec.pause_event.is_set():
                self._store.request_resume(job_id)
                self._mirror(job_id, force=True)
            if rec.status in ("completed", "error", "cancelled", "interrupted"):
                entry["phase"] = "done"
                self._mirror(job_id, force=True)
                self._log(f"{job_id} {rec.status}")
            else:
                self._mirror(job_id)

    # ── Loop ────────────────────────────────────────────────────

    def step(self) -> None:
        """One pass: claim, schedule, apply control, report."""
        stop_file = self.state.supervisors_dir / f"{self.id}.stop"
        if stop_file.exists():
            stop_file.unlink(missing_ok=True)
            self._log("stop requested: cancelling jobs")
            self.cancel_all()
        self.heartbeat()
        self._scan()
        self._schedule()
        self._control()

    def _finished(self) -> bool:
        busy = any(e["phase"] != "done" for e in self.jobs.values())
        if self._cancel_all.is_set() and not busy:
            return True
        if self.mode == "foreground":
            return not busy and self._seen_own_jobs()
        if busy or self._unclaimed_queued():
            self._idle_since = None
            return False
        now = time.monotonic()
        if self._idle_since is None:
            self._idle_since = now
        return now - self._idle_since >= 60 * self.limits["idle_minutes"]

    def _seen_own_jobs(self) -> bool:
        own = [
            j for j in self.state.job_ids()
            if (self.state.read_spec(j) or {}).get("owner") == self.id
        ]
        return all(j in self.jobs for j in own)

    def _unclaimed_queued(self) -> bool:
        for job_id in self.state.job_ids():
            if job_id in self.jobs:
                continue
            spec = self.state.read_spec(job_id) or {}
            if spec.get("owner") is None and (
                self.state.read_status(job_id).get("status") == "queued"
            ):
                return True
        return False

    def cancel_all(self) -> None:
        """Cancel every job this supervisor holds (Ctrl+C, scheduler kill)."""
        self._cancel_all.set()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> int:
        """Run until finished (see the module docstring). Returns 0 when every
        job this supervisor ran completed, 1 otherwise."""
        self._started_at = now_iso()
        self._ensure_executor()
        self.heartbeat(force=True)
        try:
            while not self._stop.is_set():
                self.step()
                if self._finished():
                    break
                time.sleep(self.poll_s)
        finally:
            for job_id, entry in self.jobs.items():
                if entry["phase"] in ("build", "run") and entry["record"] is not None:
                    self._store.request_cancel(job_id)
            if self._executor is not None:
                # Stopped outright, the jobs' processes end with this one
                # (they watch their parent); otherwise let them finish.
                self._executor.shutdown(wait=not self._stop.is_set())
            for job_id, entry in self.jobs.items():
                if entry["record"] is not None:
                    self._mirror(job_id, force=True)
            self.state.remove_supervisor(self.id)
            if self._store is not None:
                self._store.shutdown()
        states = [e.get("status", {}).get("status") for e in self.jobs.values()]
        return 0 if all(s == "completed" for s in states) else 1


def start_daemon(state: StateDir, *, wait_s: float = 20.0) -> dict:
    """Start the shared supervisor detached from this terminal; return its
    record once it is up. A running one is returned as is."""
    running = state.daemon()
    if running:
        return running
    state.root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "geofuse",
        "--state-dir",
        str(state.root),
        "supervisor",
        "--daemon",
    ]
    log = open(state.root / "supervisor.log", "ab")
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": log,
        "stderr": log,
        "cwd": str(REPO_ROOT),
        "close_fds": True,
    }
    try:
        if os.name == "nt":
            flags = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW
                | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
            )
            try:
                subprocess.Popen(cmd, creationflags=flags, **kwargs)
            except OSError:
                flags &= ~getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
                subprocess.Popen(cmd, creationflags=flags, **kwargs)
        else:
            subprocess.Popen(cmd, start_new_session=True, **kwargs)
    finally:
        log.close()
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        running = state.daemon()
        if running:
            return running
        time.sleep(0.2)
    raise RuntimeError(
        f"The supervisor did not start; see {state.root / 'supervisor.log'}."
    )


def serve(
    state: StateDir,
    *,
    mode: str,
    limits: dict[str, int] | None = None,
    ee_project: str | None = None,
    supervisor_id: str | None = None,
    log: Callable[[str], None] | None = None,
) -> int:
    """Run a supervisor in this process until it finishes.

    The first Ctrl+C (or a scheduler's SIGTERM) cancels its jobs and lets them
    wind down; a second ends at once.
    """
    sup = Supervisor(
        state,
        mode=mode,
        limits=limits,
        ee_project=ee_project,
        supervisor_id=supervisor_id,
        log=log,
    )
    hits = {"n": 0}

    def _interrupt(_signum, _frame) -> None:
        hits["n"] += 1
        if hits["n"] == 1:
            (log or (lambda m: None))("stopping: cancelling jobs (again to quit now)")
            sup.cancel_all()
        else:
            sup.stop()

    previous = {}
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                previous[sig] = signal.signal(sig, _interrupt)
            except (ValueError, OSError):
                pass
    try:
        return sup.run()
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
