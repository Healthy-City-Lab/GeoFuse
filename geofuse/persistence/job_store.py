"""Process-level job state with SQLite durability.

Two tiers:
    Hot  (in-memory dict, RLock): every progress tick. Microsecond reads/writes.
    Cold (SQLite): state transitions + a 10s heartbeat flush for dirty rows.

A browser refresh hits the same Python process and reads the same in-memory
dict (the store is held by an ``@st.cache_resource`` factory in the UI).
A Streamlit *process* restart reloads from SQLite and flips any
``status='running'`` rows to ``'interrupted'`` — their threads are gone.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from geofuse.persistence.sqlite_utils import open_wal_connection

_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"completed", "error", "cancelled", "interrupted"}
)
_ACTIVE_STATUSES: frozenset[str] = frozenset({"queued", "running"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class JobRecord:
    """One job's full state. Persisted fields go to SQLite; ``_runtime`` fields don't."""

    id: str
    type: str
    status: str
    params: dict
    name: str = ""
    progress: float = 0.0
    status_text: str = ""
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    output_paths: list[str] = field(default_factory=list)

    # Runtime-only — never persisted.
    cancel_event: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )
    extra: dict = field(default_factory=dict, repr=False, compare=False)
    _dirty: bool = field(default=False, repr=False, compare=False)


class JobStore:
    """In-memory job state mirrored to SQLite on transitions + heartbeat.

    Thread-safe. One ``RLock`` guards both the in-memory dict and the SQLite
    connection; SQLite work is brief enough that this does not bottleneck.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS jobs (
        id                TEXT PRIMARY KEY,
        type              TEXT NOT NULL,
        status            TEXT NOT NULL,
        name              TEXT,
        params_json       TEXT NOT NULL,
        progress          REAL DEFAULT 0,
        status_text       TEXT DEFAULT '',
        started_at        TEXT,
        updated_at        TEXT NOT NULL DEFAULT (datetime('now')),
        completed_at      TEXT,
        error             TEXT,
        output_paths_json TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
    CREATE INDEX IF NOT EXISTS idx_jobs_updated ON jobs(updated_at);
    """

    def __init__(self, db_path: str, heartbeat_interval_s: float = 10.0):
        self._lock = threading.RLock()
        self._records: dict[str, JobRecord] = {}
        self._conn = open_wal_connection(db_path)
        with self._lock:
            self._conn.executescript(self._SCHEMA)
            self._reload_and_mark_interrupted()

        self._stop = threading.Event()
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(heartbeat_interval_s,),
            daemon=True,
            name="JobStoreHeartbeat",
        )
        self._heartbeat.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def submit(self, type: str, params: dict, name: str = "") -> JobRecord:
        """Create a new ``queued`` job, persist it, return the live record."""
        job_id = uuid.uuid4().hex[:8]
        now = _utc_now_iso()
        rec = JobRecord(
            id=job_id,
            type=type,
            status="queued",
            params=params,
            name=name or job_id,
            updated_at=now,
        )
        with self._lock:
            self._records[job_id] = rec
            self._write_row(rec)
        return rec

    def transition(
        self,
        job_id: str,
        status: str,
        *,
        error: str | None = None,
        output_paths: Iterable[str] | None = None,
    ) -> None:
        """Persist a state change. Always writes SQLite immediately."""
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return
            rec.status = status
            rec.updated_at = _utc_now_iso()
            if status == "running" and rec.started_at is None:
                rec.started_at = rec.updated_at
            if status in _TERMINAL_STATUSES:
                rec.completed_at = rec.updated_at
                # Clear stale in-progress text so the monitor doesn't show
                # "Processing (26/425)" next to a Cancelled / Completed badge.
                rec.status_text = ""
            if error is not None:
                rec.error = error
            if output_paths is not None:
                rec.output_paths = list(output_paths)
            rec._dirty = False
            self._write_row(rec)

    # ------------------------------------------------------------------
    # Hot path (no disk I/O)
    # ------------------------------------------------------------------

    def update_progress(
        self,
        job_id: str,
        progress: float | None = None,
        status_text: str | None = None,
        **extra,
    ) -> None:
        """Update progress in memory only. Persisted by next transition/heartbeat."""
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return
            if progress is not None:
                rec.progress = max(0.0, min(1.0, float(progress)))
            if status_text is not None:
                rec.status_text = status_text
            if extra:
                rec.extra.update(extra)
            rec.updated_at = _utc_now_iso()
            rec._dirty = True

    def heartbeat(self, job_id: str) -> None:
        """Bump ``updated_at`` to mark the worker as alive."""
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return
            rec.updated_at = _utc_now_iso()
            rec._dirty = True

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def request_cancel(self, job_id: str) -> None:
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return
            rec.cancel_event.set()

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            rec = self._records.get(job_id)
            return bool(rec and rec.cancel_event.is_set())

    # ------------------------------------------------------------------
    # Reads (return snapshots; safe to use without holding the lock)
    # ------------------------------------------------------------------

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._records.get(job_id)

    def list_all(self) -> list[JobRecord]:
        with self._lock:
            return list(self._records.values())

    def list_active(self) -> list[JobRecord]:
        with self._lock:
            return [r for r in self._records.values() if r.status in _ACTIVE_STATUSES]

    def list_terminal(self) -> list[JobRecord]:
        with self._lock:
            return [r for r in self._records.values() if r.status in _TERMINAL_STATUSES]

    def health(self, stuck_after_s: float = 30.0) -> dict:
        """Snapshot for the sidebar badge."""
        now = datetime.now(timezone.utc)
        stuck_cutoff = now - timedelta(seconds=stuck_after_s)
        err_cutoff = now - timedelta(hours=1)
        active = 0
        stuck = 0
        errored = 0
        oldest_running_s: float | None = None
        with self._lock:
            for rec in self._records.values():
                if rec.status in _ACTIVE_STATUSES:
                    active += 1
                if rec.status == "running" and rec.updated_at:
                    try:
                        ts = datetime.strptime(
                            rec.updated_at, "%Y-%m-%dT%H:%M:%S.%fZ"
                        ).replace(tzinfo=timezone.utc)
                        if ts < stuck_cutoff:
                            stuck += 1
                    except ValueError:
                        pass
                if rec.status == "running" and rec.started_at:
                    try:
                        ts = datetime.strptime(
                            rec.started_at, "%Y-%m-%dT%H:%M:%S.%fZ"
                        ).replace(tzinfo=timezone.utc)
                        age = (now - ts).total_seconds()
                        if oldest_running_s is None or age > oldest_running_s:
                            oldest_running_s = age
                    except ValueError:
                        pass
                if rec.status == "error" and rec.completed_at:
                    try:
                        ts = datetime.strptime(
                            rec.completed_at, "%Y-%m-%dT%H:%M:%S.%fZ"
                        ).replace(tzinfo=timezone.utc)
                        if ts > err_cutoff:
                            errored += 1
                    except ValueError:
                        pass
        return {
            "active": active,
            "stuck": stuck,
            "errored_last_hour": errored,
            "oldest_running_age_s": oldest_running_s,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def dismiss(self, job_id: str) -> None:
        """Drop from in-memory view. SQLite row stays for history."""
        with self._lock:
            self._records.pop(job_id, None)

    def purge(self, job_id: str) -> None:
        """Remove from both in-memory and SQLite."""
        with self._lock:
            self._records.pop(job_id, None)
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def shutdown(self) -> None:
        self._stop.set()
        try:
            self._heartbeat.join(timeout=2.0)
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_row(self, rec: JobRecord) -> None:
        """Caller must hold ``self._lock``."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO jobs (
                id, type, status, name, params_json,
                progress, status_text,
                started_at, updated_at, completed_at,
                error, output_paths_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.id,
                rec.type,
                rec.status,
                rec.name,
                json.dumps(rec.params, default=str),
                rec.progress,
                rec.status_text,
                rec.started_at,
                rec.updated_at,
                rec.completed_at,
                rec.error,
                json.dumps(rec.output_paths) if rec.output_paths else None,
            ),
        )

    def _reload_and_mark_interrupted(self) -> None:
        """Re-populate the in-memory dict from SQLite at startup.

        Anything ``status='running'`` is from a previous Python process — its
        worker thread is gone, so flip it to ``'interrupted'``. The UI shows
        these with a Resubmit button.
        """
        cur = self._conn.execute(
            """
            SELECT id, type, status, name, params_json,
                   progress, status_text,
                   started_at, updated_at, completed_at,
                   error, output_paths_json
            FROM jobs
            ORDER BY updated_at DESC
            LIMIT 200
            """
        )
        to_interrupt: list[str] = []
        for row in cur.fetchall():
            (
                jid,
                jtype,
                status,
                name,
                params_json,
                progress,
                status_text,
                started_at,
                updated_at,
                completed_at,
                error,
                output_paths_json,
            ) = row
            if status in ("running", "queued"):
                status = "interrupted"
                to_interrupt.append(jid)
            try:
                params = json.loads(params_json) if params_json else {}
            except json.JSONDecodeError:
                params = {}
            try:
                outputs = (
                    json.loads(output_paths_json) if output_paths_json else []
                )
            except json.JSONDecodeError:
                outputs = []
            self._records[jid] = JobRecord(
                id=jid,
                type=jtype,
                status=status,
                name=name or jid,
                params=params,
                progress=float(progress or 0.0),
                status_text=status_text or "",
                started_at=started_at,
                updated_at=updated_at,
                completed_at=completed_at,
                error=error,
                output_paths=outputs,
            )
        for jid in to_interrupt:
            rec = self._records[jid]
            rec.completed_at = _utc_now_iso()
            rec.error = rec.error or "Streamlit process restarted while running."
            self._write_row(rec)

    def _heartbeat_loop(self, interval_s: float) -> None:
        """Flush dirty records to SQLite at a bounded rate."""
        while not self._stop.wait(interval_s):
            with self._lock:
                for rec in self._records.values():
                    if rec._dirty:
                        self._write_row(rec)
                        rec._dirty = False
