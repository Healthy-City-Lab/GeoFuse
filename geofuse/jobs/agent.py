"""``geofuse agent --stdio``: a state folder's jobs over standard input and output.

One request per line in, one answer per line out, both JSON::

    {"id": 7, "op": "log", "job": "1a2b3c4d", "offset": null}
    {"id": 7, "ok": true, "result": {"text": "...", "offset": 5120}}
    {"id": 8, "ok": false, "error": "No job 'zz'."}

So one SSH login carries everything a remote client needs, password and
second-factor logins included, from any operating system. Operations:

``hello``     protocol, code version, host and state folder
``snapshot``  every job (spec and status) and the supervisor
``log``       a job's log from ``offset`` (its tail when ``null``)
``control``   pause / resume / cancel (``flags``)
``rerun``     queue a job again from its spec
``outputs``   a job's output files (path, size, mtime)
``read``      ``length`` bytes of one of those files from ``offset``, base64

``read`` serves only files the job lists as outputs, so a client cannot read
anything else through the agent. Nothing but answers is written to standard
output while the agent runs.
"""

from __future__ import annotations

import base64
import contextlib
import json
import socket
import sys
from typing import IO

from geofuse.jobs.sources import LocalSource
from geofuse.jobs.state import StateDir, now_iso

PROTOCOL = 1
#: Largest ``read`` a request may ask for.
MAX_READ_BYTES = 1024 * 1024


class AgentError(RuntimeError):
    """A request the agent cannot answer, with the reason."""


def _job(source: LocalSource, request: dict) -> str:
    job_id = str(request.get("job") or "")
    hits = source.state.find(job_id) if job_id else []
    if not hits:
        raise AgentError(f"No job {job_id!r}.")
    if len(hits) > 1:
        raise AgentError(f"{job_id!r} matches several jobs.")
    return hits[0]


def _version() -> str:
    try:
        from geofuse.presets import _geofuse_version

        return str(_geofuse_version())
    except Exception:
        return "unknown"


def handle(source: LocalSource, request: dict):
    """The result of one request (see the module docstring)."""
    op = request.get("op")
    if op == "hello":
        return {
            "protocol": PROTOCOL,
            "version": _version(),
            "host": socket.gethostname(),
            "state_dir": str(source.state.root),
        }
    if op == "snapshot":
        return {
            "jobs": [
                {"id": j.id, "spec": j.spec, "status": j.status} for j in source.jobs()
            ],
            "supervisor": source.supervisor(),
            "at": now_iso(),
        }
    if op == "log":
        text, offset = source.read_log(_job(source, request), request.get("offset"))
        return {"text": text, "offset": offset}
    if op == "control":
        flags = {k: bool(v) for k, v in (request.get("flags") or {}).items()}
        if set(flags) - {"pause", "cancel"}:
            raise AgentError("control takes pause and cancel flags only.")
        return {"message": source.control(_job(source, request), **flags)}
    if op == "rerun":
        return {"message": source.rerun(_job(source, request))}
    if op == "outputs":
        return source.outputs(_job(source, request))
    if op == "read":
        job_id = _job(source, request)
        path = str(request.get("path") or "")
        files = {f["path"]: f for f in source.outputs(job_id)}
        if path not in files:
            raise AgentError(f"{path!r} is not an output of {job_id}.")
        offset = max(0, int(request.get("offset") or 0))
        length = min(int(request.get("length") or MAX_READ_BYTES), MAX_READ_BYTES)
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(length)
        size = files[path]["size"]
        return {
            "data": base64.b64encode(data).decode("ascii"),
            "size": size,
            "eof": offset + len(data) >= size,
        }
    raise AgentError(f"Unknown operation {op!r}.")


def serve(state: StateDir, stdin: IO[bytes], stdout: IO[bytes]) -> int:
    """Answer requests from ``stdin`` until it closes."""
    source = LocalSource(state)
    for raw in stdin:
        if not raw.strip():
            continue
        request: dict = {}
        try:
            request = json.loads(raw)
            result = handle(source, request)
            answer = {"id": request.get("id"), "ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 — every request gets an answer
            if isinstance(exc, AgentError):
                message = str(exc)
            else:
                message = f"{type(exc).__name__}: {exc}"
            answer = {"id": request.get("id"), "ok": False, "error": message}
        stdout.write((json.dumps(answer, default=str) + "\n").encode("utf-8"))
        stdout.flush()
    return 0


def run_stdio(state: StateDir) -> int:
    """Serve on this process's standard streams; anything else printed while
    serving goes to standard error, so the answers stay readable."""
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    with contextlib.redirect_stdout(sys.stderr):
        return serve(state, stdin, stdout)
