"""Remote links: another machine's jobs over one SSH login.

A remote is a table in the user configuration (``geofuse remote add``)::

    [remotes.cluster]
    host = "login.cluster.example.org"
    user = "me"                                # optional
    port = 22                                  # optional
    ssh_options = ["-i", "~/.ssh/cluster"]     # passed to ssh and scp
    command = "conda run -n geofuse geofuse"   # how geofuse starts there
    state_dir = "/scratch/me/geofuse-state"    # optional: its state folder
    ssh = "ssh"                                # the ssh program (or a list)

``geofuse remote attach NAME`` holds the link: it starts ``geofuse agent
--stdio`` there through ``ssh`` (one login, password and second factor
included) and keeps a mirror of the remote jobs in this machine's state
folder::

    remotes/<name>/link.json       connecting / connected / reconnecting /
                                   stopped, with a heartbeat
    remotes/<name>/snapshot.json   the remote jobs and supervisor, last read
    remotes/<name>/logs/<id>.log   each job's log: its tail, then as it grows
    remotes/<name>/requests/       control, re-run, output and fetch requests
    remotes/<name>/responses/      from any client, answered over the link
    remotes/<name>/outputs/<id>/   files fetched over the link

:class:`MirrorSource` reads the mirror as a job source, so ``geofuse watch
--remote NAME`` and the GUI's Jobs tab show remote jobs and send them
requests without logging in themselves. A dropped link is reopened with
growing pauses; meanwhile the mirror reads as stale. Nothing here is specific
to one cluster or scheduler.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

from geofuse.jobs.sources import read_log_file
from geofuse.jobs.state import (
    ACTIVE,
    Job,
    StateDir,
    now_iso,
    parse_iso,
    read_json,
    write_json,
)
from geofuse.userconfig import REMOTE_KEYS

#: How long the first answer may take: the login may ask for a password.
LOGIN_TIMEOUT_S = 300.0
#: How long any later answer may take.
ANSWER_TIMEOUT_S = 60.0
#: Bytes per ``read`` request when a file comes over the link.
CHUNK_BYTES = 1024 * 1024
#: Files larger than this go through ``scp`` when the command line fetches.
SCP_ABOVE_BYTES = 16 * 1024 * 1024


class LinkError(RuntimeError):
    """The link failed: the login, the remote command, or the connection."""


class RemoteRefused(RuntimeError):
    """The remote agent answered with an error (an unknown job, say)."""


# ────────────────────────────────────────────────────────────────────
# A remote and its SSH command
# ────────────────────────────────────────────────────────────────────


@dataclass
class Remote:
    name: str
    host: str
    user: str | None = None
    port: int | None = None
    ssh_options: list[str] = field(default_factory=list)
    command: str = "geofuse"
    state_dir: str | None = None
    ssh: str | list[str] = "ssh"

    @classmethod
    def from_config(cls, name: str, table: dict) -> Remote:
        if not table.get("host"):
            raise ValueError(f"Remote {name!r} has no host.")
        known = {k: v for k, v in table.items() if k in REMOTE_KEYS}
        return cls(name=name, **known)

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def agent_command(self) -> str:
        """The command the remote shell runs: the agent on its state folder."""
        state = f" --state-dir {shlex.quote(self.state_dir)}" if self.state_dir else ""
        return f"{self.command}{state} agent --stdio"

    def _program(self) -> list[str]:
        return list(self.ssh) if isinstance(self.ssh, list) else [self.ssh]

    def ssh_argv(self) -> list[str]:
        port = ["-p", str(self.port)] if self.port else []
        return [
            *self._program(),
            "-T",
            *port,
            *self.ssh_options,
            self.target,
            self.agent_command(),
        ]

    def scp_argv(self, remote_path: str, local_path: str) -> list[str]:
        port = ["-P", str(self.port)] if self.port else []
        source = f"{self.target}:{shlex.quote(remote_path)}"
        return ["scp", "-q", *port, *self.ssh_options, source, local_path]


def remotes(config) -> dict[str, Remote]:
    """The remotes a :class:`~geofuse.userconfig.Config` names."""
    return {
        name: Remote.from_config(name, table)
        for name, table in (config.remotes or {}).items()
    }


# ────────────────────────────────────────────────────────────────────
# The agent at the other end of the link
# ────────────────────────────────────────────────────────────────────


class AgentClient:
    """``geofuse agent --stdio`` behind ``argv`` (an ``ssh`` command), with one
    request in flight at a time. Lines that are not answers (a login banner
    printed by a shell profile, say) are skipped. The console stays shared, so
    ``ssh`` can ask for a password or a second factor there."""

    def __init__(self, argv: list[str]):
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        self._lines: queue.Queue = queue.Queue()
        self._next = 0
        self._lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True, name="agent-read").start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def call(self, op: str, *, timeout: float = ANSWER_TIMEOUT_S, **args):
        """The result of one request; :class:`LinkError` when no answer comes,
        :class:`RemoteRefused` when the agent answers with an error."""
        with self._lock:
            self._next += 1
            request = {"id": self._next, "op": op, **args}
            try:
                self.proc.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
                self.proc.stdin.flush()
            except (OSError, ValueError):
                raise LinkError(self._ended("the link closed")) from None
            deadline = time.monotonic() + timeout
            while True:
                left = max(0.0, deadline - time.monotonic())
                try:
                    line = self._lines.get(timeout=left)
                except queue.Empty:
                    raise LinkError(f"no answer to {op} in {timeout:.0f} s") from None
                if line is None:
                    raise LinkError(self._ended("the remote side closed the link"))
                try:
                    answer = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(answer, dict) or answer.get("id") != self._next:
                    continue
                if answer.get("ok"):
                    return answer.get("result")
                raise RemoteRefused(str(answer.get("error")))

    def _ended(self, what: str) -> str:
        code = self.proc.poll()
        return f"{what} (exit code {code})" if code is not None else what

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(3)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _remote_path(path: str):
    """A path the remote machine wrote, read with that machine's rules."""
    if "\\" in path or ":" in path[:3]:
        return PureWindowsPath(path)
    return PurePosixPath(path)


def fetch_outputs(
    client: AgentClient,
    job_id: str,
    dest: Path,
    *,
    remote: Remote | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    """Copy ``job_id``'s output files into ``dest``, keeping their layout below
    the folder they share. Files reach ``dest`` complete or not at all. With
    ``remote`` given, files above :data:`SCP_ABOVE_BYTES` go through ``scp``
    (its own login) when ``scp`` is installed; everything else comes over the
    link in chunks."""
    files = client.call("outputs", job=job_id) or []
    if not files:
        return []
    paths = [_remote_path(f["path"]) for f in files]
    shared = 0
    if len(paths) > 1:
        for group in zip(*(p.parent.parts for p in paths)):
            if any(part != group[0] for part in group):
                break
            shared += 1
    use_scp = remote is not None and shutil.which("scp") is not None
    written = []
    for f, remote_file in zip(files, paths, strict=True):
        if shared:
            relative = PurePosixPath(*remote_file.parts[shared:])
        else:
            relative = PurePosixPath(remote_file.name)
        local = dest.joinpath(*relative.parts)
        local.parent.mkdir(parents=True, exist_ok=True)
        part = local.with_name(f".{local.name}.part")
        if progress:
            progress(f"{relative.as_posix()} ({f['size']:,} bytes)")
        if use_scp and f["size"] > SCP_ABOVE_BYTES:
            done = subprocess.run(remote.scp_argv(f["path"], str(part)))
            if done.returncode != 0:
                code = done.returncode
                raise LinkError(f"scp of {remote_file.name} failed ({code})")
        else:
            with open(part, "wb") as out:
                offset = 0
                while True:
                    chunk = client.call(
                        "read",
                        job=job_id,
                        path=f["path"],
                        offset=offset,
                        length=CHUNK_BYTES,
                    )
                    data = base64.b64decode(chunk["data"])
                    out.write(data)
                    offset += len(data)
                    if chunk["eof"] or not data:
                        break
        os.replace(part, local)
        written.append(local)
    return written


# ────────────────────────────────────────────────────────────────────
# The mirror of a remote state folder
# ────────────────────────────────────────────────────────────────────


class Mirror:
    """``<state>/remotes/<name>/`` (see the module docstring)."""

    def __init__(self, state: StateDir, name: str):
        self.name = name
        self.root = state.root / "remotes" / name

    @property
    def requests_dir(self) -> Path:
        return self.root / "requests"

    @property
    def responses_dir(self) -> Path:
        return self.root / "responses"

    def log_path(self, job_id: str) -> Path:
        return self.root / "logs" / f"{job_id}.log"

    def outputs_dir(self, job_id: str) -> Path:
        return self.root / "outputs" / job_id

    def read_link(self) -> dict:
        return read_json(self.root / "link.json", {}) or {}

    def write_link(self, state: str, **extra) -> None:
        record = {
            "state": state,
            "pid": os.getpid(),
            "heartbeat": now_iso(),
            **extra,
        }
        write_json(self.root / "link.json", record)

    def read_snapshot(self) -> dict:
        return read_json(self.root / "snapshot.json", {}) or {}

    def write_snapshot(self, snapshot: dict) -> None:
        record = {**snapshot, "received_at": now_iso()}
        write_json(self.root / "snapshot.json", record)

    def write_log(self, job_id: str, text: str, *, append: bool) -> None:
        path = self.log_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "ab" if append else "wb") as f:
            f.write(text.encode("utf-8"))

    def requests(self) -> list[tuple[str, dict]]:
        """Waiting requests, oldest first."""
        if not self.requests_dir.is_dir():
            return []
        dated = []
        for path in self.requests_dir.glob("*.json"):
            try:
                dated.append((path.stat().st_mtime, path))
            except OSError:
                continue
        found = []
        for _mtime, path in sorted(dated):
            request = read_json(path)
            if isinstance(request, dict):
                found.append((path.stem, request))
        return found

    def respond(self, request_id: str, *, result=None, error: str | None = None):
        write_json(
            self.responses_dir / f"{request_id}.json",
            {"ok": error is None, "result": result, "error": error, "at": now_iso()},
        )
        try:
            (self.requests_dir / f"{request_id}.json").unlink()
        except OSError:
            pass


# ────────────────────────────────────────────────────────────────────
# Holding the link
# ────────────────────────────────────────────────────────────────────


def _handle(client: AgentClient, mirror: Mirror, request: dict, heartbeat):
    op = request.get("op")
    job = str(request.get("job") or "")
    if op == "control":
        return client.call("control", job=job, flags=request.get("flags") or {})
    if op == "rerun":
        return client.call("rerun", job=job)
    if op == "outputs":
        return client.call("outputs", job=job)
    if op == "fetch":
        dest = mirror.outputs_dir(job)
        files = fetch_outputs(client, job, dest, progress=lambda _text: heartbeat())
        return {"folder": str(dest), "files": [str(p) for p in files]}
    raise RemoteRefused(f"Unknown request {op!r}.")


def attach(
    remote: Remote,
    state: StateDir,
    *,
    stop: threading.Event | None = None,
    interval_s: float = 2.0,
    retry_s: tuple[float, float] = (2.0, 60.0),
    report: Callable[[str], None] | None = None,
    on_connect: Callable[[AgentClient], None] | None = None,
) -> None:
    """Hold the link to ``remote`` until ``stop`` is set, mirroring its jobs
    into ``state`` every ``interval_s`` and answering mirrored requests."""
    stop = stop or threading.Event()
    report = report or (lambda _text: None)
    mirror = Mirror(state, remote.name)
    delay = retry_s[0]
    connections = 0
    while not stop.is_set():
        mirror.write_link("connecting", host=remote.target, connections=connections)
        client = None
        try:
            client = AgentClient(remote.ssh_argv())
            hello = client.call("hello", timeout=LOGIN_TIMEOUT_S)
            connections += 1
            delay = retry_s[0]
            info = {
                "host": remote.target,
                "remote_host": hello.get("host"),
                "remote_state": hello.get("state_dir"),
                "version": hello.get("version"),
                "connections": connections,
            }

            def heartbeat(info=info) -> None:
                mirror.write_link("connected", **info)

            heartbeat()
            report(f"connected to {remote.target} ({hello.get('state_dir')})")
            if on_connect:
                on_connect(client)
            offsets: dict[str, int] = {}
            synced_states: dict[str, str] = {}
            while not stop.is_set():
                snapshot = client.call("snapshot")
                mirror.write_snapshot(snapshot)
                for record in snapshot.get("jobs") or []:
                    job_id, now_state = record["id"], record["status"].get("status")
                    if job_id in offsets and (
                        now_state not in ACTIVE
                        and synced_states.get(job_id) == now_state
                    ):
                        continue
                    try:
                        answer = client.call(
                            "log", job=job_id, offset=offsets.get(job_id)
                        )
                    except RemoteRefused:
                        continue
                    mirror.write_log(
                        job_id, answer["text"], append=job_id in offsets
                    )
                    offsets[job_id] = answer["offset"]
                    synced_states[job_id] = now_state
                for request_id, request in mirror.requests():
                    try:
                        result = _handle(client, mirror, request, heartbeat)
                        mirror.respond(request_id, result=result)
                    except RemoteRefused as exc:
                        mirror.respond(request_id, error=str(exc))
                heartbeat()
                stop.wait(interval_s)
        except (LinkError, OSError) as exc:
            if client is not None:
                client.close()
                client = None
            if stop.is_set():
                break
            mirror.write_link(
                "reconnecting",
                host=remote.target,
                error=str(exc),
                connections=connections,
                retry_at=time.time() + delay,
            )
            report(f"link lost ({exc}); retrying in {delay:.0f} s")
            stop.wait(delay)
            delay = min(delay * 2, retry_s[1])
        finally:
            if client is not None:
                client.close()
    mirror.write_link("stopped", host=remote.target, connections=connections)
    report("link closed")


# ────────────────────────────────────────────────────────────────────
# Reading the mirror
# ────────────────────────────────────────────────────────────────────


class MirrorSource:
    """A remote's jobs as the mirror holds them (a job source)."""

    kind = "remote"

    def __init__(self, state: StateDir, name: str, *, stale_after_s: float = 20.0):
        self.mirror = Mirror(state, name)
        self.name = name
        self.stale_after_s = stale_after_s

    @property
    def label(self) -> str:
        link = self.link()
        host = link.get("host") or "never attached"
        text = f"{self.name} · {host} · {link['state']}"
        return text + (" · stale" if link["stale"] else "")

    def link(self) -> dict:
        """The link record, with ``stale`` (no fresh heartbeat while
        connected) and ``seen`` (when the jobs were last read)."""
        link = self.mirror.read_link()
        state = link.get("state", "not attached")
        beat = parse_iso(link.get("heartbeat"))
        fresh = beat is not None and time.time() - beat <= self.stale_after_s
        if state == "connected" and not fresh:
            state = "not attached"
        seen = self.mirror.read_snapshot().get("received_at")
        return {**link, "state": state, "stale": state != "connected", "seen": seen}

    def jobs(self) -> list[Job]:
        snapshot = self.mirror.read_snapshot()
        jobs = [
            Job(r["id"], r.get("spec") or {}, r.get("status") or {}, self.mirror.root)
            for r in snapshot.get("jobs") or []
        ]
        return sorted(jobs, key=lambda j: (j.spec.get("submitted_at") or "", j.id))

    def supervisor(self) -> dict | None:
        return self.mirror.read_snapshot().get("supervisor")

    def read_log(self, job_id: str, offset: int | None) -> tuple[str, int]:
        return read_log_file(self.mirror.log_path(job_id), offset)

    def ask(self, op: str, *, wait_s: float = 8.0, **args) -> dict | None:
        """Leave a request for the link; its answer (``{"ok", "result",
        "error"}``) when it comes within ``wait_s``, else ``None``."""
        request_id = uuid.uuid4().hex
        write_json(
            self.mirror.requests_dir / f"{request_id}.json",
            {"op": op, **args, "at": now_iso()},
        )
        path = self.mirror.responses_dir / f"{request_id}.json"
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            answer = read_json(path)
            if answer is not None:
                try:
                    path.unlink()
                except OSError:
                    pass
                return answer
            time.sleep(0.1)
        return None

    def _message(self, op: str, **args) -> str:
        if self.link()["stale"]:
            return (
                f"not sent: the link to {self.name} is not attached "
                f"(geofuse remote attach {self.name})"
            )
        answer = self.ask(op, **args)
        if answer is None:
            return "requested; the link has not answered yet"
        if not answer["ok"]:
            return f"failed: {answer['error']}"
        return str((answer["result"] or {}).get("message", "done"))

    def control(self, job_id: str, **flags: bool) -> str:
        return self._message("control", job=job_id, flags=flags)

    def rerun(self, job_id: str) -> str:
        return self._message("rerun", job=job_id)

    def outputs(self, job_id: str) -> list[dict]:
        if self.link()["stale"]:
            return []
        answer = self.ask("outputs", job=job_id)
        return list(answer["result"] or []) if answer and answer["ok"] else []

    def fetch(self, job_id: str) -> str:
        """Ask the link to copy the job's outputs here; returns at once."""
        if self.link()["stale"]:
            return f"not sent: the link to {self.name} is not attached"
        self.ask("fetch", job=job_id, wait_s=0.0)
        return f"fetching into {self.mirror.outputs_dir(job_id)}"

    def fetched(self, job_id: str) -> list[Path]:
        """Files a fetch has already copied here."""
        folder = self.mirror.outputs_dir(job_id)
        if not folder.is_dir():
            return []
        return sorted(
            p
            for p in folder.rglob("*")
            if p.is_file() and not p.name.endswith(".part")
        )
