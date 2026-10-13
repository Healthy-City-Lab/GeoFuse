"""The ``geofuse`` command line.

::

    geofuse submit RUN.toml         run a run file in the background
    geofuse run RUN.toml            run it here until it ends (batch scripts)
    geofuse plan RUN.toml           show the jobs it becomes; run nothing
    geofuse ndvi | gvi | fusion ... one engine without a run file
    geofuse status [JOB...]         where every job is
    geofuse watch                   full-screen monitor (jobs keep running on quit)
    geofuse logs JOB [-f]           a job's log
    geofuse pause | resume | cancel JOB... | --all
    geofuse wait JOB... | --run RUN
    geofuse rerun JOB...            the same job again (caches resume it)
    geofuse rm JOB... | --finished
    geofuse config template | validate | show | set | unset
    geofuse supervisor [--stop]     the shared background supervisor
    geofuse credentials show | set | remove
    geofuse auth earthengine        sign in (stored in the credential store)
    geofuse doctor                  check this machine is ready to run jobs
    geofuse preset list | show | save | import | remove
    geofuse cache list | purge
    geofuse results | export JOB    a fusion run's headline / per-entity table
    geofuse hpc script RUN.toml     SLURM job arrays (one job per task)

Every setting is also a flag (``--cloud-max 20``, ``--no-water-mask``) that
overrides every step having that setting; ``--set ENGINE[.STEP].KEY=VALUE``
targets one engine or one step. Exit codes: 0 done, 1 a job failed or was
cancelled, 2 invalid settings or inputs, 3 an environment problem.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from geofuse import settings as _settings
from geofuse.jobs.state import ACTIVE, TERMINAL, StateDir

EXIT_OK, EXIT_FAILED, EXIT_INVALID, EXIT_ENVIRONMENT = 0, 1, 2, 3


class CliError(Exception):
    """A problem worth one line to the user, with its exit code."""

    def __init__(self, message: str, code: int = EXIT_INVALID):
        super().__init__(message)
        self.code = code


# ────────────────────────────────────────────────────────────────────
# Shared pieces
# ────────────────────────────────────────────────────────────────────


def _config(args):
    from geofuse import userconfig

    try:
        return userconfig.load(getattr(args, "config", None))
    except userconfig.ConfigError as exc:
        raise CliError(str(exc)) from exc


def _state(args) -> StateDir:
    root = getattr(args, "state_dir", None) or _config(args).state_dir
    return StateDir(root)


def _add_setting_flags(
    parser: argparse.ArgumentParser,
    engines: tuple[str, ...] = ("ndvi", "gvi", "fusion"),
    note: str = "each overrides every step that has the setting",
) -> None:
    group = parser.add_argument_group("settings", note)
    seen: set[str] = set()
    for engine in engines:
        for key, s in _settings.SCHEMAS[engine].items():
            if key in seen:
                continue
            seen.add(key)
            flag = "--" + key.replace("_", "-")
            # argparse formats help with %, so a literal one is doubled.
            text = s.help.replace("%", "%%")
            kw: dict = {"dest": f"setting__{key}", "default": None, "help": text}
            if s.kind == "bool":
                kw["action"] = argparse.BooleanOptionalAction
            elif s.kind == "multichoice":
                kw.update(nargs="+", choices=list(s.choices), metavar="STAT")
            elif s.kind == "choice":
                kw["choices"] = list(s.choices)
            elif s.kind in ("int", "float"):
                kw["type"] = int if s.kind == "int" else float
                kw["metavar"] = "N" if s.kind == "int" else "X"
            group.add_argument(flag, **kw)


def _add_run_args(parser: argparse.ArgumentParser, *, arrays: bool = True) -> None:
    parser.add_argument("runfile", help="the run file (TOML, or JSON)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a key: KEY, ENGINE.KEY or ENGINE.STEP.KEY (repeatable)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="STEP",
        help="run only these steps (ENGINE.STEP or STEP; repeatable); the "
        "steps they depend on are taken as already done",
    )
    if not arrays:
        _add_setting_flags(parser)
        return
    jobs = parser.add_argument_group(
        "job arrays", "pick jobs for one task of a scheduler array (see plan)"
    )
    jobs.add_argument(
        "--level",
        type=int,
        metavar="K",
        help="only jobs at dependency level K (0: they wait for nothing); "
        "earlier levels are taken as done",
    )
    jobs.add_argument(
        "--group",
        choices=["gpu", "fusion", "light"],
        help="only jobs needing these resources (GVI / fusion / the rest)",
    )
    jobs.add_argument(
        "--array-task",
        type=int,
        metavar="N",
        help="only the N-th of the jobs picked so far (from 0)",
    )
    _add_setting_flags(parser)


def _overrides(args) -> list:
    from geofuse.runfile import Override, parse_override

    out = [parse_override(o) for o in args.overrides]
    for name, value in vars(args).items():
        if name.startswith("setting__") and value is not None:
            out.insert(0, Override(("*",), name[len("setting__") :], value))
    return out


def _load(args):
    """``(run, specs)`` for the run file in ``args``, or :class:`CliError`."""
    from geofuse.jobs.plans import PlanError, expand
    from geofuse.runfile import RunFileError, load

    try:
        run = load(args.runfile, _overrides(args))
        specs = expand(run)
    except (RunFileError, PlanError, _settings.SettingError) as exc:
        raise CliError(str(exc)) from exc
    return run, _pick(args, specs)


def _pick(args, specs: list) -> list:
    """The jobs ``--only``, ``--level``, ``--group`` and ``--array-task`` keep."""
    from geofuse.jobs.plans import job_group, levels

    level_of = dict(zip(map(id, specs), levels(specs), strict=True))
    only = getattr(args, "only", None)
    if only:
        specs = [
            s
            for s in specs
            if any(s.step == o or s.step.endswith(f".{o}") for o in only)
        ]
        if not specs:
            raise CliError(f"--only {only} matches no step of {args.runfile}.")
    level = getattr(args, "level", None)
    if level is not None:
        specs = [s for s in specs if level_of[id(s)] == level]
    group = getattr(args, "group", None)
    if group:
        specs = [s for s in specs if job_group(s) == group]
    task = getattr(args, "array_task", None)
    if task is not None:
        if not 0 <= task < len(specs):
            raise CliError(
                f"--array-task {task}: the selection has {len(specs)} job(s), "
                f"0..{len(specs) - 1}."
            )
        specs = [specs[task]]
    if (level is not None or group) and not specs:
        raise CliError("No job matches --level / --group.")
    return specs


def _resolve(state: StateDir, refs: list[str]) -> list[str]:
    ids = []
    for ref in refs:
        hits = state.find(ref)
        if not hits:
            raise CliError(f"No job {ref!r}.")
        if len(hits) > 1:
            raise CliError(f"{ref!r} matches several jobs: {', '.join(hits)}.")
        ids.append(hits[0])
    return ids


def _fmt_progress(job) -> str:
    st = job.status
    if job.state in ("running",) or st.get("progress"):
        return f"{100 * float(st.get('progress') or 0):3.0f}%"
    return ""


def _note(job) -> str:
    st = job.status
    if job.state in TERMINAL and st.get("error"):
        return str(st["error"]).splitlines()[0][:70]
    if job.state == "completed":
        return f"{len(st.get('outputs') or [])} output(s)"
    if job.state in TERMINAL:
        return ""
    bracket = (st.get("extra") or {}).get("ndvi_tile_bracket")
    text = st.get("text") or ""
    if st.get("paused") and job.state in ACTIVE:
        text = f"paused · {text}" if text and text != "Held" else "paused"
    return (f"{bracket} " if bracket else "") + text[:70]


def _shown_state(job) -> str:
    if job.status.get("paused") and job.state in ACTIVE:
        return "paused"
    return job.state


def _table(jobs) -> str:
    rows = [("ID", "KIND", "NAME", "STATUS", "PROG", "NOTE")]
    for j in jobs:
        rows.append(
            (
                j.id,
                str(j.spec.get("kind")),
                j.name[:28],
                _shown_state(j),
                _fmt_progress(j),
                _note(j),
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    return "\n".join(
        "  ".join(r[i].ljust(widths[i]) for i in range(5)) + "  " + r[5] for r in rows
    )


# ────────────────────────────────────────────────────────────────────
# Run files: plan / submit / run
# ────────────────────────────────────────────────────────────────────


def cmd_plan(args) -> int:
    run, specs = _load(args)
    return show_plan(run, specs, as_json=args.json)


def show_plan(run, specs, *, as_json: bool = False) -> int:
    """Print the jobs ``run`` becomes, with the settings that differ from the
    defaults."""
    if as_json:
        print(json.dumps([s.to_dict() for s in specs], indent=1))
        return EXIT_OK
    from geofuse.jobs.plans import job_group, levels

    defaults = {e: _settings.defaults(e) for e in _settings.SCHEMAS}
    source = f" ({run.path})" if run.path else ""
    print(f"Run {run.name!r}{source} -> {run.output_dir}")
    level_of = dict(zip(map(id, specs), levels(specs), strict=True))
    for step in run.steps:
        shard_step = f"{step.key}:shards"
        mine = [s for s in specs if s.step in (step.key, shard_step)]
        if not mine:
            continue
        changed = {
            k: v for k, v in step.settings.items() if v != defaults[step.engine].get(k)
        }
        preset = f"  preset: {step.preset}" if step.preset else ""
        print(f"\n{step.key}{preset}")
        if changed:
            print("  settings: " + ", ".join(f"{k}={v}" for k, v in changed.items()))
        if step.depends_on:
            print("  after: " + ", ".join(step.depends_on))
        for spec in mine:
            label = spec.options.get("output_name") or spec.name
            where = f"level {level_of[id(spec)]}, {job_group(spec)}"
            print(f"  #{specs.index(spec):<3} {spec.kind:12s} {label}  ({where})")
            outs = spec.outputs
            if "years" in outs:
                years = outs["years"]
                first = next(iter(years.values()))
                listed = first if isinstance(first, dict) else years
                print(f"    per year: {', '.join(listed)} -> {outs['folder']}")
            elif "files" in outs:
                files = outs["files"]
                for f in files.values() if isinstance(files, dict) else [files]:
                    print(f"    -> {f}")
            elif "folder" in outs:
                print(f"    -> {outs['folder']}")
    print(f"\n{len(specs)} job(s).")
    return EXIT_OK


def _need_streetview_key(specs: list) -> None:
    """Refuse GVI steps that download through the API when no key is stored."""
    from geofuse.jobs.plans import NO_STREETVIEW_KEY

    gvi = ("gvi", "gvi_column", "gvi_merge")
    if not any(
        s.kind in gvi and s.settings.get("download_mode") == "api" for s in specs
    ):
        return
    from geofuse import credentials

    try:
        stored = credentials.get("streetview-key")
    except credentials.CredentialError:
        stored = None
    if not stored:
        raise CliError(NO_STREETVIEW_KEY, EXIT_ENVIRONMENT)


def cmd_submit(args) -> int:
    return submit(args, *_load(args))


def submit(args, run, specs) -> int:
    """Hand ``specs`` to the shared supervisor, starting it if needed."""
    from geofuse.jobs.supervisor import start_daemon, submit_specs

    _need_streetview_key(specs)
    state = _state(args)
    _, ids = submit_specs(state, specs, run_name=run.name, run_file=run.path)
    try:
        start_daemon(state)
    except RuntimeError as exc:
        raise CliError(str(exc), EXIT_ENVIRONMENT) from exc
    print(_table([state.job(i) for i in ids]))
    print(
        f"\nSubmitted {len(ids)} job(s). Follow them with: geofuse watch  "
        "(or geofuse status, geofuse logs JOB -f)"
    )
    return EXIT_OK


def _follow(state: StateDir, ids: list[str], stop: threading.Event) -> None:
    last: dict[str, tuple] = {}
    while not stop.wait(2.0):
        for job_id in ids:
            job = state.job(job_id)
            if job is None:
                continue
            tenths = int(10 * (job.status.get("progress") or 0))
            key = (_shown_state(job), _note(job), tenths)
            if last.get(job_id) != key:
                last[job_id] = key
                stamp = time.strftime("%H:%M:%S")
                print(
                    f"{stamp}  {job_id}  {job.name[:24]:24s}  {_shown_state(job):10s}"
                    f"  {_fmt_progress(job):4s}  {_note(job)}",
                    flush=True,
                )


def cmd_run(args) -> int:
    return run_here(args, *_load(args))


def run_here(args, run, specs) -> int:
    """Run ``specs`` in a supervisor in this process until they end."""
    from geofuse.jobs.supervisor import serve, submit_specs

    _need_streetview_key(specs)
    state = _state(args)
    cfg = _config(args)
    sup_id = f"f{uuid.uuid4().hex[:7]}"
    _, ids = submit_specs(
        state, specs, run_name=run.name, run_file=run.path, owner=sup_id
    )
    print(f"Running {len(ids)} job(s) here; Ctrl+C cancels them.", flush=True)
    stop = threading.Event()
    follower = threading.Thread(target=_follow, args=(state, ids, stop), daemon=True)
    follower.start()
    try:
        code = serve(
            state,
            mode="foreground",
            limits=cfg.limits,
            ee_project=cfg.ee_project,
            supervisor_id=sup_id,
        )
    finally:
        stop.set()
        follower.join(3)
    print()
    print(_table([state.job(i) for i in ids]))
    return EXIT_OK if code == 0 else EXIT_FAILED


# ────────────────────────────────────────────────────────────────────
# Jobs: status / logs / control / wait / rerun / rm
# ────────────────────────────────────────────────────────────────────


def _selected(args, state: StateDir) -> list:
    jobs = state.jobs()
    if getattr(args, "run", None):
        jobs = [j for j in jobs if str(j.spec.get("run_id", "")).startswith(args.run)]
    if getattr(args, "jobs", None):
        wanted = set(_resolve(state, args.jobs))
        jobs = [j for j in jobs if j.id in wanted]
    return jobs


def cmd_status(args) -> int:
    state = _state(args)
    jobs = _selected(args, state)
    if args.active:
        jobs = [j for j in jobs if j.state in ACTIVE]
    if args.json:
        rows = [{"id": j.id, "spec": j.spec, "status": j.status} for j in jobs]
        print(json.dumps(rows, indent=1, default=str))
        return EXIT_OK
    if not jobs:
        print("No jobs.")
        return EXIT_OK
    print(_table(jobs[-args.limit :] if args.limit else jobs))
    daemon = state.daemon()
    print(
        f"\nState folder: {state.root} · supervisor: "
        + (f"running (pid {daemon.get('pid')})" if daemon else "not running")
    )
    return EXIT_OK


def cmd_watch(args) -> int:
    try:
        from geofuse.cli.watch import LocalSource, WatchApp
    except ImportError as exc:
        raise CliError(
            f"geofuse watch needs the textual package ({exc}); install it with "
            "`pip install textual`, or use geofuse status.",
            EXIT_ENVIRONMENT,
        ) from exc
    from geofuse.jobs.remote import MirrorSource, remotes

    state = _state(args)
    names = list(remotes(_config(args)))
    sources = [LocalSource(state), *(MirrorSource(state, n) for n in names)]
    first = sources[0]
    if args.remote:
        if args.remote not in names:
            raise CliError(
                f"No remote {args.remote!r} (known: {', '.join(names) or 'none'})."
            )
        first = sources[1 + names.index(args.remote)]
    WatchApp(first, sources=sources, run=args.run, active_only=args.active).run()
    return EXIT_OK


def cmd_logs(args) -> int:
    state = _state(args)
    (job_id,) = _resolve(state, [args.job])
    path = state.log_path(job_id)
    pos = 0
    if path.is_file():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-args.lines :] if args.lines else lines:
            print(line)
        pos = path.stat().st_size
    elif not args.follow:
        print(f"No log yet for {job_id} ({path}).")
        return EXIT_OK
    if not args.follow:
        return EXIT_OK
    try:
        while True:
            if path.is_file() and path.stat().st_size > pos:
                with open(path, encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    sys.stdout.write(f.read())
                    sys.stdout.flush()
                    pos = f.tell()
            job = state.job(job_id)
            if job is not None and job.state in TERMINAL:
                return EXIT_OK
            time.sleep(0.5)
    except KeyboardInterrupt:
        return EXIT_OK


def _control(args, **flags) -> int:
    state = _state(args)
    if args.all:
        ids = [j.id for j in state.jobs() if j.state in ACTIVE]
    else:
        if not args.jobs:
            raise CliError("Name the job(s), or use --all.")
        ids = _resolve(state, args.jobs)
    for job_id in ids:
        print(f"{job_id}: {state.control(job_id, **flags)}")
    return EXIT_OK


def cmd_pause(args) -> int:
    return _control(args, pause=True)


def cmd_resume(args) -> int:
    return _control(args, pause=False)


def cmd_cancel(args) -> int:
    return _control(args, cancel=True)


def cmd_wait(args) -> int:
    state = _state(args)
    jobs = _selected(args, state)
    if not jobs:
        raise CliError("No job to wait for; name jobs or --run RUN.")
    ids = [j.id for j in jobs]
    deadline = time.monotonic() + args.timeout if args.timeout else None
    try:
        while True:
            current = [state.job(i) for i in ids]
            if all(j is None or j.state in TERMINAL for j in current):
                break
            if deadline and time.monotonic() > deadline:
                print(_table([j for j in current if j]))
                return EXIT_FAILED
            time.sleep(1.0)
    except KeyboardInterrupt:
        return EXIT_FAILED
    print(_table([j for j in current if j]))
    done = all(j and j.state == "completed" for j in current)
    return EXIT_OK if done else EXIT_FAILED


def cmd_rerun(args) -> int:
    from geofuse.jobs.supervisor import resubmit, serve, start_daemon

    state = _state(args)
    ids = _resolve(state, args.jobs)
    owner = f"f{uuid.uuid4().hex[:7]}" if args.foreground else None
    new_ids = resubmit(state, ids, owner=owner)
    if not args.foreground:
        try:
            start_daemon(state)
        except RuntimeError as exc:
            raise CliError(str(exc), EXIT_ENVIRONMENT) from exc
        print(_table([state.job(i) for i in new_ids]))
        return EXIT_OK
    cfg = _config(args)
    code = serve(
        state,
        mode="foreground",
        limits=cfg.limits,
        ee_project=cfg.ee_project,
        supervisor_id=owner,
    )
    print(_table([state.job(i) for i in new_ids]))
    return EXIT_OK if code == 0 else EXIT_FAILED


def cmd_agent(args) -> int:
    from geofuse.jobs.agent import run_stdio

    return run_stdio(_state(args))


def cmd_rm(args) -> int:
    state = _state(args)
    if args.finished:
        ids = [j.id for j in state.jobs() if j.state in TERMINAL]
    else:
        if not args.jobs:
            raise CliError("Name the job(s), or use --finished.")
        ids = _resolve(state, args.jobs)
    for job_id in ids:
        job = state.job(job_id)
        if job is not None and job.state in ACTIVE and not args.force:
            print(f"{job_id}: still {job.state}; cancel it first (or --force)")
            continue
        state.remove_job(job_id)
        print(f"{job_id}: removed")
    return EXIT_OK


# ────────────────────────────────────────────────────────────────────
# Configuration and the supervisor
# ────────────────────────────────────────────────────────────────────


def cmd_config(args) -> int:
    from geofuse import userconfig
    from geofuse.runfile import parse_value

    if args.action == "template":
        from geofuse.cli.templates import template

        try:
            text = template(args.kind)
        except ValueError as exc:
            raise CliError(str(exc)) from exc
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"Wrote {args.output}")
        else:
            sys.stdout.write(text)
        return EXIT_OK
    if args.action == "validate":
        _, specs = _load(args)
        print(f"OK: {len(specs)} job(s).")
        return EXIT_OK
    try:
        if args.action == "show":
            cfg = userconfig.load(getattr(args, "config", None))
            print(f"Configuration: {cfg.path}")
            print(f"State folder:  {_state(args).root}")
            print(f"Earth Engine project: {cfg.ee_project or '(Earth Engine default)'}")
            print("Limits: " + ", ".join(f"{k}={v}" for k, v in cfg.limits.items()))
            for name, remote in cfg.remotes.items():
                print(f"Remote {name}: {remote}")
            return EXIT_OK
        if args.action == "set":
            path = userconfig.set_value(args.key, parse_value(args.value), args.config)
        else:
            path = userconfig.set_value(args.key, None, args.config)
    except userconfig.ConfigError as exc:
        raise CliError(str(exc)) from exc
    print(f"Updated {path}")
    return EXIT_OK


def cmd_supervisor(args) -> int:
    state = _state(args)
    if args.daemon:
        from geofuse.jobs.supervisor import serve

        cfg = _config(args)

        def log(msg: str) -> None:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)

        log(f"supervisor starting in {state.root}")
        return serve(
            state, mode="daemon", limits=cfg.limits, ee_project=cfg.ee_project, log=log
        )
    daemon = state.daemon()
    if args.stop:
        if not daemon:
            print("No supervisor is running.")
            return EXIT_OK
        state.request_stop(daemon["id"])
        print(f"Stop requested; supervisor {daemon['id']} cancels its jobs and exits.")
        return EXIT_OK
    if daemon:
        print(
            f"Supervisor {daemon['id']} running (pid {daemon.get('pid')}, "
            f"host {daemon.get('host')}), state folder {state.root}"
        )
    else:
        print(f"No supervisor is running (state folder {state.root}).")
    return EXIT_OK


# ────────────────────────────────────────────────────────────────────
# Parser
# ────────────────────────────────────────────────────────────────────


class _Version(argparse.Action):
    """``--version``, looked up only when asked for (it runs ``git``)."""

    def __call__(self, parser, namespace, values, option_string=None):
        from geofuse.presets import _geofuse_version

        print(f"geofuse {_geofuse_version()}")
        parser.exit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geofuse",
        description="GeoFuse: NDVI, GVI and composite greenery index runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run 'geofuse COMMAND --help' for a command's options.",
    )
    parser.add_argument("--state-dir", help="the job state folder (default: per user)")
    parser.add_argument("--config", help="the configuration file (default: per user)")
    parser.add_argument(
        "--version", action=_Version, nargs=0, help="show the code version and exit"
    )
    # The same two options after the command name; SUPPRESS keeps a value given
    # before it from being reset.
    common = argparse.ArgumentParser(add_help=False)
    for flag in ("--state-dir", "--config"):
        common.add_argument(flag, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    add = sub.add_parser

    def add_parser(name, **kw):
        return add(name, parents=[common], **kw)

    sub.add_parser = add_parser

    p = sub.add_parser("plan", help="show the jobs a run file becomes")
    _add_run_args(p)
    p.add_argument("--json", action="store_true", help="print the job specs as JSON")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("submit", help="run a run file in the background")
    _add_run_args(p)
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("run", help="run a run file here until it ends")
    _add_run_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="where every job is")
    p.add_argument("jobs", nargs="*", metavar="JOB", help="job ids (prefixes work)")
    p.add_argument("--run", help="only this run's jobs")
    p.add_argument("--active", action="store_true", help="only unfinished jobs")
    p.add_argument("--limit", type=int, default=0, help="only the newest N")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("watch", help="full-screen monitor: follow and manage jobs")
    p.add_argument("--run", help="only this run's jobs")
    p.add_argument("--active", action="store_true", help="start with unfinished jobs")
    p.add_argument("--remote", metavar="NAME", help="start on a remote's mirrored jobs")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("logs", help="a job's log")
    p.add_argument("job")
    p.add_argument("-f", "--follow", action="store_true", help="keep printing")
    p.add_argument("-n", "--lines", type=int, default=50, help="last N lines (0: all)")
    p.set_defaults(func=cmd_logs)

    for name, func, text in (
        ("pause", cmd_pause, "hold jobs at their next safe point"),
        ("resume", cmd_resume, "continue held jobs"),
        ("cancel", cmd_cancel, "cancel jobs"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("jobs", nargs="*", metavar="JOB")
        p.add_argument("--all", action="store_true", help="every unfinished job")
        p.set_defaults(func=func)

    p = sub.add_parser("wait", help="wait for jobs to end (exit 0 if all completed)")
    p.add_argument("jobs", nargs="*", metavar="JOB")
    p.add_argument("--run", help="every job of this run")
    p.add_argument("--timeout", type=float, default=0, help="give up after S seconds")
    p.set_defaults(func=cmd_wait)

    p = sub.add_parser("rerun", help="run jobs again from their specs")
    p.add_argument("jobs", nargs="+", metavar="JOB")
    p.add_argument(
        "--foreground", action="store_true", help="run here, not in the background"
    )
    p.set_defaults(func=cmd_rerun)

    p = sub.add_parser("rm", help="remove jobs from the list")
    p.add_argument("jobs", nargs="*", metavar="JOB")
    p.add_argument("--finished", action="store_true", help="every finished job")
    p.add_argument("--force", action="store_true", help="also unfinished ones")
    p.set_defaults(func=cmd_rm)

    p = sub.add_parser(
        "agent", help="serve this machine's jobs to a remote client (geofuse remote)"
    )
    p.add_argument(
        "--stdio",
        action="store_true",
        required=True,
        help="answer JSON-line requests on standard input and output",
    )
    p.set_defaults(func=cmd_agent)

    p = sub.add_parser("config", help="run-file templates, validation, user settings")
    actions = p.add_subparsers(dest="action", metavar="ACTION", required=True)
    add_action = actions.add_parser
    actions.add_parser = lambda name, **kw: add_action(name, parents=[common], **kw)
    t = actions.add_parser("template", help="print a commented run file")
    t.add_argument("kind", choices=["pipeline", "ndvi", "gvi", "fusion"])
    t.add_argument("-o", "--output", help="write it to this file")
    v = actions.add_parser("validate", help="check a run file without running it")
    _add_run_args(v)
    actions.add_parser("show", help="the user configuration in force")
    s = actions.add_parser("set", help="set a user configuration key")
    s.add_argument("key", help="e.g. earthengine.project, limits.gpu, state_dir")
    s.add_argument("value")
    u = actions.add_parser("unset", help="remove a user configuration key")
    u.add_argument("key")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("supervisor", help="the shared background supervisor")
    p.add_argument("--stop", action="store_true", help="cancel its jobs and stop it")
    p.add_argument("--daemon", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_supervisor)

    from geofuse.cli import account, hpc, library, quick, remote, results

    quick.add_parsers(sub, common)
    hpc.add_parsers(sub, common)
    account.add_parsers(sub, common)
    library.add_parsers(sub, common)
    results.add_parsers(sub, common)
    remote.add_parsers(sub, common)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except CliError as exc:
        print(f"geofuse: {exc}", file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:
        return EXIT_FAILED
