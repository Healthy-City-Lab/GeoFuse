"""
Coloured console logger for GeoFuse engines.

Usage
-----
    from geofuse.logger import get_logger

    log = get_logger("GVI")
    log("INFO",  "Starting analysis…")
    log("OK",    "Segmentation complete: veg=0.31")
    log("WARN",  "No panoramas found nearby")
    log("ERROR", "Download failed: ConnectError")

Levels
------
    INFO   cyan    — routine progress
    OK     green   — successful result / file written
    WARN   yellow  — skipped step / missing data (non-fatal)
    ERROR  red     — exception or unrecoverable failure

Per-job log capture
-------------------
Every ``log(level, msg)`` call also appends the rendered line (including ANSI
colour codes) to the current thread's job-specific ring buffer if one is bound
via :func:`bind_job_log_buffer`. The job monitor reads from these buffers via
:func:`get_job_log_lines` so each job card can show its own log stream.
"""

from __future__ import annotations

import threading
from collections import deque

_ANSI = {
    "INFO": "\033[36m",  # cyan
    "OK": "\033[32m",  # green
    "WARN": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
}

# Default ring-buffer depth per job (last N log lines).
DEFAULT_JOB_LOG_LINES = 100

# Thread-local that names the active job for the current worker thread.
# Anything logged from that thread routes into ``_job_log_buffers[<job_id>]``.
_current_job: threading.local = threading.local()
_job_log_buffers: dict[str, deque[str]] = {}
_job_log_lock = threading.RLock()


def bind_job_log_buffer(job_id: str, max_lines: int = DEFAULT_JOB_LOG_LINES) -> None:
    """Route this thread's subsequent ``log(...)`` calls into ``job_id``'s buffer."""
    _current_job.job_id = job_id
    with _job_log_lock:
        if job_id not in _job_log_buffers:
            _job_log_buffers[job_id] = deque(maxlen=max_lines)


def unbind_job_log_buffer() -> None:
    """Stop capturing this thread's log into any job buffer."""
    _current_job.job_id = None


def get_job_log_lines(job_id: str) -> list[str]:
    """Return a snapshot of ``job_id``'s captured log lines (oldest → newest)."""
    with _job_log_lock:
        buf = _job_log_buffers.get(job_id)
        return list(buf) if buf is not None else []


def drop_job_log_buffer(job_id: str) -> None:
    """Free the job's log buffer (call when the user permanently purges a job)."""
    with _job_log_lock:
        _job_log_buffers.pop(job_id, None)


def get_logger(engine: str):
    """Return a ``log(level, msg)`` callable prefixed with *engine*.

    Parameters
    ----------
    engine : str
        Short engine name shown in the prefix, e.g. ``"GVI"``, ``"NDVI"``,
        ``"FUSION"``.

    Returns
    -------
    callable
        ``log(level: str, msg: str) -> None``
    """
    tag = engine.upper()

    def log(level: str, msg: str) -> None:
        color = _ANSI.get(level, "")
        line = (
            f"{color}{_ANSI['BOLD']}[{tag} {level}]{_ANSI['RESET']} {msg}"
        )
        print(line, flush=True)
        # Mirror into the active job's buffer if any.
        job_id = getattr(_current_job, "job_id", None)
        if job_id:
            with _job_log_lock:
                buf = _job_log_buffers.get(job_id)
                if buf is not None:
                    buf.append(line)

    return log
