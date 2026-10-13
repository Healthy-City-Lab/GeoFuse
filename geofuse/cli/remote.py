"""``geofuse remote``: another machine's jobs from this one.

::

    geofuse remote add NAME HOST [--user U] [--port N] [--ssh-option OPTS]
                       [--command CMD] [--state-dir DIR] [--ssh PROGRAM]
    geofuse remote list
    geofuse remote remove NAME [--forget]
    geofuse remote attach NAME [--interval S]
    geofuse remote fetch NAME JOB [-o FOLDER]

``attach`` holds one SSH login open and mirrors the remote jobs here, so
``geofuse watch --remote NAME`` and the GUI's Jobs tab follow and control them
(see ``geofuse.jobs.remote``). ``fetch`` copies a job's output files.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

from geofuse import userconfig
from geofuse.cli.commands import (
    EXIT_ENVIRONMENT,
    EXIT_INVALID,
    EXIT_OK,
    CliError,
    _config,
    _state,
)


def _remote(args):
    from geofuse.jobs.remote import remotes

    found = remotes(_config(args))
    if args.name not in found:
        known = ", ".join(found) or "none yet; add one with geofuse remote add"
        raise CliError(f"No remote {args.name!r} (known: {known}).")
    return found[args.name]


def _config_path(args):
    return getattr(args, "config", None)


def _print(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {text}", flush=True)


def cmd_remote_add(args) -> int:
    table: dict = {"host": args.host}
    if args.user:
        table["user"] = args.user
    if args.port:
        table["port"] = args.port
    options = [part for option in args.ssh_option or [] for part in option.split()]
    if options:
        table["ssh_options"] = options
    if args.remote_command:
        table["command"] = args.remote_command
    if args.remote_state_dir:
        table["state_dir"] = args.remote_state_dir
    if args.ssh:
        table["ssh"] = args.ssh
    try:
        path = userconfig.set_value(f"remotes.{args.name}", table, _config_path(args))
    except userconfig.ConfigError as exc:
        raise CliError(str(exc)) from exc
    print(f"Remote {args.name!r} saved in {path}.")
    print(f"Hold the link with: geofuse remote attach {args.name}")
    return EXIT_OK


def cmd_remote_list(args) -> int:
    from geofuse.jobs.remote import MirrorSource, remotes

    found = remotes(_config(args))
    if not found:
        print("No remotes; add one with geofuse remote add NAME HOST.")
        return EXIT_OK
    state = _state(args)
    for name, remote in found.items():
        link = MirrorSource(state, name).link()
        where = f" ({remote.state_dir})" if remote.state_dir else ""
        print(f"{name}: {remote.target}{where} · runs {remote.command!r}")
        seen = f", jobs read {link['seen']}" if link.get("seen") else ""
        print(f"    link: {link['state']}{seen}")
    return EXIT_OK


def cmd_remote_remove(args) -> int:
    from geofuse.jobs.remote import Mirror

    _remote(args)
    userconfig.set_value(f"remotes.{args.name}", None, _config_path(args))
    mirror = Mirror(_state(args), args.name)
    if args.forget:
        shutil.rmtree(mirror.root, ignore_errors=True)
        print(f"Removed remote {args.name!r} and its mirror.")
    else:
        print(f"Removed remote {args.name!r}; its mirror stays in {mirror.root}.")
    return EXIT_OK


def cmd_remote_attach(args) -> int:
    from geofuse.jobs.remote import attach

    remote = _remote(args)
    state = _state(args)
    stop = threading.Event()
    print(
        f"Linking {remote.name} ({remote.target}); answer any login prompt here. "
        "Ctrl+C closes the link (remote jobs keep running).",
        flush=True,
    )
    worker = threading.Thread(
        target=attach,
        args=(remote, state),
        kwargs={"stop": stop, "interval_s": args.interval, "report": _print},
        daemon=True,
        name=f"remote-{remote.name}",
    )
    worker.start()
    try:
        while worker.is_alive():
            worker.join(0.5)
    except KeyboardInterrupt:
        stop.set()
        worker.join(15)
    return EXIT_OK


def cmd_remote_fetch(args) -> int:
    from geofuse.jobs.remote import (
        LOGIN_TIMEOUT_S,
        AgentClient,
        LinkError,
        RemoteRefused,
        fetch_outputs,
    )

    remote = _remote(args)
    dest = Path(args.output or f"{remote.name}-{args.job}").resolve()
    client = AgentClient(remote.ssh_argv())
    try:
        client.call("hello", timeout=LOGIN_TIMEOUT_S)
        files = fetch_outputs(client, args.job, dest, remote=remote, progress=_print)
    except LinkError as exc:
        message = f"The link to {remote.name} failed: {exc}"
        raise CliError(message, EXIT_ENVIRONMENT) from exc
    except RemoteRefused as exc:
        raise CliError(str(exc), EXIT_INVALID) from exc
    finally:
        client.close()
    if not files:
        print(f"{args.job} has no output files yet.")
    else:
        print(f"{len(files)} file(s) in {dest}")
    return EXIT_OK


def add_parsers(sub, common) -> None:
    """This module's commands on ``sub`` (which adds ``common`` itself)."""
    p = sub.add_parser("remote", help="another machine's jobs over one SSH login")
    actions = p.add_subparsers(dest="action", metavar="ACTION", required=True)
    add_action = actions.add_parser
    actions.add_parser = lambda name, **kw: add_action(name, parents=[common], **kw)

    a = actions.add_parser("add", help="save a remote in the user configuration")
    a.add_argument("name")
    a.add_argument("host")
    a.add_argument("--user")
    a.add_argument("--port", type=int)
    a.add_argument(
        "--ssh-option",
        action="append",
        metavar="OPTS",
        help="options for ssh and scp, e.g. '-i ~/.ssh/cluster' (repeatable)",
    )
    a.add_argument(
        "--command",
        dest="remote_command",
        help="how geofuse starts there (default: geofuse), e.g. "
        "'conda run -n geofuse geofuse'",
    )
    a.add_argument("--remote-state-dir", metavar="DIR", help="the state folder there")
    a.add_argument("--ssh", help="the ssh program (default: ssh)")
    a.set_defaults(func=cmd_remote_add)

    a = actions.add_parser("list", help="the saved remotes and their links")
    a.set_defaults(func=cmd_remote_list)

    a = actions.add_parser("remove", help="forget a remote")
    a.add_argument("name")
    a.add_argument("--forget", action="store_true", help="also delete its mirror")
    a.set_defaults(func=cmd_remote_remove)

    a = actions.add_parser("attach", help="hold the link and mirror the jobs here")
    a.add_argument("name")
    a.add_argument(
        "--interval", type=float, default=2.0, help="seconds between reads (2)"
    )
    a.set_defaults(func=cmd_remote_attach)

    a = actions.add_parser("fetch", help="copy a remote job's output files here")
    a.add_argument("name")
    a.add_argument("job", help="the remote job id (a prefix works)")
    a.add_argument("-o", "--output", help="folder (default: NAME-JOB)")
    a.set_defaults(func=cmd_remote_fetch)
