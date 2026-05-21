"""
Per-job logger with non-blocking queue + listener thread.

Workers call ``log(level, msg)`` and the call returns in microseconds: it
just builds two formatted lines (a coloured one for the UI deque, a plain
one for the file) and pushes them onto an in-process queue. A single
daemon listener thread drains the queue and:

* appends the coloured line to the per-job in-memory deque (the
  Streamlit job-monitor expander reads from this for active jobs), and
* appends the plain line to ``logs/jobs/<job_id>.log`` and flushes so the
  user can open the file in a text editor at any point during the run.

Logs emitted outside of a bound job (e.g. before any job is submitted)
are silently dropped. The :class:`~geofuse.persistence.job_executor.JobExecutor`
binds the worker thread to the job id for the entire runner lifecycle,
so engine initialisation logs land in the same job's log.

Levels
------
    INFO   cyan    — routine progress
    OK     green   — successful result / file written
    WARN   yellow  — skipped step / missing data (non-fatal)
    ERROR  red     — exception or unrecoverable failure
"""

from __future__ import annotations

import os
import queue
import re
import threading
from collections import deque
from typing import IO

_ANSI = {
    "INFO": "\033[36m",  # cyan
    "OK": "\033[32m",  # green
    "WARN": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
}

# Default ring-buffer depth per job (last N log lines kept in memory for the UI).
DEFAULT_JOB_LOG_LINES = 100

# Directory for per-job log files. Kept inside ``logs/`` to stay alongside the
# existing JobStore SQLite + status files.
_LOG_DIR = "logs/jobs"

# Thread-local that names the active job for the current worker thread.
# Anything logged from that thread routes into ``_job_log_buffers[<job_id>]``
# and ``_job_log_files[<job_id>]``.
_current_job: threading.local = threading.local()

# Shared mutable state — protected by ``_job_state_lock``.
_job_log_buffers: dict[str, deque[str]] = {}
_job_log_files: dict[str, IO[str]] = {}
_job_state_lock = threading.RLock()

# Non-blocking producer/consumer pipeline. ``log()`` is the producer (microseconds
# per call); a single daemon listener thread is the consumer that does the
# (potentially blocking) file I/O off the worker's critical path.
_log_queue: queue.Queue[tuple[str, str, str] | None] = queue.Queue()
_listener_thread: threading.Thread | None = None
_listener_lock = threading.Lock()

# Strip ANSI colour codes from the plain (file) line so editors don't show
# escape sequences.
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _ensure_listener() -> None:
    """Start the listener thread on first use. Idempotent."""
    global _listener_thread
    if _listener_thread is not None and _listener_thread.is_alive():
        return
    with _listener_lock:
        if _listener_thread is not None and _listener_thread.is_alive():
            return
        _listener_thread = threading.Thread(
            target=_listener_loop, daemon=True, name="JobLogListener"
        )
        _listener_thread.start()


def _listener_loop() -> None:
    while True:
        item = _log_queue.get()
        if item is None:
            return  # shutdown sentinel (unused today; daemon thread exits with process)
        job_id, colored_line, plain_line = item
        with _job_state_lock:
            buf = _job_log_buffers.get(job_id)
            if buf is not None:
                buf.append(colored_line)
            fh = _job_log_files.get(job_id)
            if fh is not None and not fh.closed:
                try:
                    fh.write(plain_line + "\n")
                    fh.flush()
                except (OSError, ValueError):
                    # File closed mid-flight or disk error: swallow rather than
                    # crash the listener thread for the whole process.
                    pass


def bind_job_log_buffer(job_id: str, max_lines: int = DEFAULT_JOB_LOG_LINES) -> None:
    """Route this thread's ``log(...)`` calls into ``job_id``'s deque + file."""
    _current_job.job_id = job_id
    _ensure_listener()
    with _job_state_lock:
        if job_id not in _job_log_buffers:
            _job_log_buffers[job_id] = deque(maxlen=max_lines)
        if job_id not in _job_log_files:
            try:
                os.makedirs(_LOG_DIR, exist_ok=True)
                path = os.path.join(_LOG_DIR, f"{job_id}.log")
                _job_log_files[job_id] = open(path, "a", encoding="utf-8")
            except OSError:
                # If the file can't be opened, in-memory deque-only is still
                # useful — don't fail the job over a logging side-effect.
                pass


def unbind_job_log_buffer() -> None:
    """Stop capturing this thread's log into any job buffer.

    The per-job file stays open so a restart of the same job id can keep
    appending; :func:`drop_job_log_buffer` does the final close on purge.
    """
    _current_job.job_id = None


def get_job_log_lines(job_id: str) -> list[str]:
    """Return a snapshot of ``job_id``'s captured log lines (oldest → newest)."""
    with _job_state_lock:
        buf = _job_log_buffers.get(job_id)
        return list(buf) if buf is not None else []


def drop_job_log_buffer(job_id: str) -> None:
    """Free the job's deque and close its file (call when the user purges)."""
    with _job_state_lock:
        _job_log_buffers.pop(job_id, None)
        fh = _job_log_files.pop(job_id, None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass


def get_job_log_path(job_id: str) -> str:
    """Filesystem path the listener writes for ``job_id`` (may not yet exist)."""
    return os.path.join(_LOG_DIR, f"{job_id}.log")


def get_logger(engine: str):
    """Return a ``log(level, msg)`` callable prefixed with *engine*."""
    tag = engine.upper()

    def log(level: str, msg: str) -> None:
        color = _ANSI.get(level, "")
        colored_line = f"{color}{_ANSI['BOLD']}[{tag} {level}]{_ANSI['RESET']} {msg}"
        plain_line = _ANSI_ESCAPE_RE.sub("", f"[{tag} {level}] {msg}")
        job_id = getattr(_current_job, "job_id", None)
        if job_id is None:
            return  # Unbound: silently drop. Engines are bound for their
            # entire runner lifecycle so init logs still land in
            # the correct job's file.
        try:
            _log_queue.put_nowait((job_id, colored_line, plain_line))
        except queue.Full:
            pass  # Queue is unbounded so this is defensive only.

    return log
