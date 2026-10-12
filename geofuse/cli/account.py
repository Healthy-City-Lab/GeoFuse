"""Credentials, sign-in and the environment check.

::

    geofuse credentials show
    geofuse credentials set streetview-key            (asks for the key)
    geofuse credentials set earthengine-service-account KEY.json
    geofuse credentials remove NAME
    geofuse auth earthengine [--no-browser] [--port N] [--project ID]
    geofuse doctor [--offline] [--json]

Values are read from a file, from standard input, or typed at a hidden prompt;
never from the command line itself, where shell history and process lists
would keep them.
"""

from __future__ import annotations

import getpass
import json
import os
import platform
import sys
import uuid
from pathlib import Path

from geofuse.cli.commands import (
    EXIT_ENVIRONMENT,
    EXIT_FAILED,
    EXIT_OK,
    CliError,
    _config,
)

# ────────────────────────────────────────────────────────────────────
# geofuse credentials
# ────────────────────────────────────────────────────────────────────


def _read_value(args) -> str:
    if args.file:
        try:
            return Path(args.file).read_text(encoding="utf-8")
        except OSError as exc:
            raise CliError(f"Could not read {args.file}: {exc}") from exc
    if not sys.stdin.isatty():
        return sys.stdin.read()
    if args.name != "streetview-key":
        raise CliError(f"Give the key file: geofuse credentials set {args.name} FILE")
    return getpass.getpass(f"{args.name} (not shown): ")


def cmd_credentials(args) -> int:
    from geofuse import credentials as cr

    try:
        if args.action == "show":
            print(f"Store: {cr.store_description()}")
            have = cr.stored()
            for name, what in cr.NAMES.items():
                print(f"  {name:<28} {have.get(name, 'not stored'):<30} {what}")
            if not cr.file_is_private():
                print(f"Warning: others can read {cr.credentials_path()} (chmod 600).")
            return EXIT_OK
        if args.action == "remove":
            removed = cr.remove(args.name)
            print(f"{args.name}: {'removed' if removed else 'nothing stored'}")
            return EXIT_OK
        where = cr.put(args.name, _read_value(args))
    except cr.CredentialError as exc:
        raise CliError(str(exc)) from exc
    print(f"Stored {args.name} in {where}.")
    if args.file:
        print(f"GeoFuse reads the stored copy; {args.file} can be deleted.")
    return EXIT_OK


# ────────────────────────────────────────────────────────────────────
# geofuse auth earthengine
# ────────────────────────────────────────────────────────────────────


def cmd_auth(args) -> int:
    from geofuse import credentials as cr
    from geofuse import ee_auth, userconfig

    if args.project:
        try:
            userconfig.set_value("earthengine.project", args.project, args.config)
        except userconfig.ConfigError as exc:
            raise CliError(str(exc)) from exc
    project = args.project or _config(args).ee_project
    if not args.no_browser and not ee_auth.can_open_browser():
        print(
            "No browser can open here. Open the address below on your own "
            f"computer after connecting with `ssh -L {args.port}:localhost:"
            f"{args.port} {platform.node()}`, or run again with --no-browser.\n"
        )
    try:
        where = ee_auth.sign_in(browser=not args.no_browser, port=args.port)
    except KeyboardInterrupt:
        print("\nSign-in cancelled; nothing was stored.")
        return EXIT_FAILED
    except (ee_auth.EarthEngineAuthError, cr.CredentialError) as exc:
        raise CliError(str(exc), EXIT_ENVIRONMENT) from exc
    except Exception as exc:
        raise CliError(f"Sign-in failed: {exc}", EXIT_ENVIRONMENT) from exc
    print(f"\nSigned in; stored in {where}.")
    if args.no_browser:
        print(
            "Google expires sign-ins made without a browser after about a "
            "week; a service account key does not expire."
        )
    if cr.get(ee_auth.SERVICE_ACCOUNT):
        print(
            "Note: the stored service account key is used before this sign-in "
            "(geofuse credentials remove earthengine-service-account)."
        )
    try:
        source = ee_auth.initialize(project, sign_in_if_needed=False)
    except ee_auth.EarthEngineAuthError as exc:
        print(f"Earth Engine is not ready yet: {exc}")
        return EXIT_ENVIRONMENT
    print(f"Earth Engine is ready ({source}; project {project or 'from the sign-in'}).")
    return EXIT_OK


# ────────────────────────────────────────────────────────────────────
# geofuse doctor
# ────────────────────────────────────────────────────────────────────

OK, WARN, FAIL = "ok", "warn", "FAIL"


def _writable(folder: Path) -> str | None:
    """``None`` when ``folder`` can be created and written, else the error."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / f".probe-{uuid.uuid4().hex[:8]}"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return str(exc)
    return None


def _check_environment(args, cfg):
    from geofuse.jobs.locks import lock_dir
    from geofuse.jobs.state import StateDir

    yield "Python", OK, f"{platform.python_version()} ({sys.executable})"
    state = StateDir(getattr(args, "state_dir", None) or cfg.state_dir)
    error = _writable(state.root)
    if error:
        yield "State folder", FAIL, f"{state.root}: {error}"
    else:
        daemon = state.daemon()
        running = f"supervisor pid {daemon['pid']}" if daemon else "no supervisor"
        yield "State folder", OK, f"{state.root} ({running})"
    error = _writable(lock_dir())
    if error:
        yield "Resource locks", FAIL, f"{lock_dir()}: {error}"
    else:
        limits = ", ".join(f"{k}={v}" for k, v in cfg.limits.items())
        yield "Resource locks", OK, f"{lock_dir()} ({limits})"


def _check_credentials(args, cfg):
    from geofuse import credentials as cr

    try:
        yield "Credential store", OK, cr.store_description()
        have = cr.stored()
    except cr.CredentialError as exc:
        yield "Credential store", FAIL, str(exc)
        return
    if not cr.file_is_private():
        yield "Credential store", WARN, f"others can read {cr.credentials_path()}"
    if "streetview-key" in have:
        yield "Street View", OK, "API key stored (used when download_mode = 'api')"
    else:
        yield "Street View", OK, "no API key: GVI downloads use the scraper"


def _check_earth_engine(args, cfg):
    from geofuse import ee_auth

    try:
        _, source = ee_auth.stored_credentials()
    except Exception as exc:
        yield "Earth Engine", FAIL, str(exc)
        return
    where = f"{source}; project {cfg.ee_project or '(from the credentials)'}"
    if source == ee_auth.DEFAULT_SOURCE:
        from ee import oauth

        plain = oauth.get_credentials_path()
        if os.path.isfile(plain):
            yield "Earth Engine", WARN, (
                f"signed in through a plain-text file ({plain}); store it with "
                f"`geofuse credentials set earthengine-login {plain}`, then "
                "delete the file if nothing else uses it"
            )
    if args.offline:
        yield "Earth Engine", OK, f"{where} (not contacted)"
        return
    try:
        ee_auth.initialize(cfg.ee_project, sign_in_if_needed=False)
    except Exception as exc:
        yield "Earth Engine", FAIL, str(exc)
        return
    yield "Earth Engine", OK, f"{where}; reachable"


def _check_compute(args, cfg):
    from geofuse.jobs.plans import DEFAULT_MODEL_PATH

    try:
        import torch

        version = f"torch {torch.__version__}"
        if torch.cuda.is_available():
            yield "GPU", OK, f"CUDA, {torch.cuda.get_device_name(0)} ({version})"
        elif torch.backends.mps.is_available():
            yield "GPU", OK, f"Apple MPS ({version})"
        else:
            yield "GPU", WARN, f"none: GVI segments on the CPU, slowly ({version})"
    except ImportError:
        yield "GPU", FAIL, "torch is not installed: GVI cannot run"
    if os.path.isfile(DEFAULT_MODEL_PATH):
        size = os.path.getsize(DEFAULT_MODEL_PATH) / 2**20
        yield "GVI model", OK, f"{DEFAULT_MODEL_PATH} ({size:.0f} MB)"
    else:
        yield "GVI model", FAIL, f"missing {DEFAULT_MODEL_PATH} (see INSTALLATION)"
    try:
        import textual

        yield "Monitor", OK, f"textual {textual.__version__} (geofuse watch)"
    except ImportError:
        yield "Monitor", WARN, "textual is not installed: geofuse watch is unavailable"


def cmd_doctor(args) -> int:
    from geofuse import userconfig

    rows = []

    def report(row):
        rows.append(row)
        if not args.json:
            label, status, detail = row
            print(f"[{status:^4}] {label:<16} {detail}", flush=True)

    try:
        cfg = userconfig.load(args.config)
        note = "" if cfg.path and cfg.path.exists() else " (not created: defaults)"
        report(("Configuration", OK, f"{cfg.path}{note}"))
    except userconfig.ConfigError as exc:
        cfg = userconfig.Config()
        report(("Configuration", FAIL, str(exc)))
    for check in (
        _check_environment,
        _check_credentials,
        _check_earth_engine,
        _check_compute,
    ):
        for row in check(args, cfg):
            report(row)
    failed = sum(status == FAIL for _, status, _ in rows)
    if args.json:
        keys = ("check", "status", "detail")
        print(json.dumps([dict(zip(keys, row)) for row in rows], indent=1))
    else:
        print(f"\n{failed} problem(s)." if failed else "\nReady.")
    return EXIT_ENVIRONMENT if failed else EXIT_OK


# ────────────────────────────────────────────────────────────────────
# Parser
# ────────────────────────────────────────────────────────────────────


def add_parsers(sub, common) -> None:
    """This module's commands on ``sub``; ``common`` holds the options every
    command takes."""
    from geofuse.credentials import NAMES
    from geofuse.ee_auth import DEFAULT_PORT

    p = sub.add_parser("credentials", help="the Street View key, Earth Engine keys")
    actions = p.add_subparsers(dest="action", metavar="ACTION", required=True)
    actions.add_parser("show", parents=[common], help="what is stored, and where")
    s = actions.add_parser(
        "set", parents=[common], help="store one (from FILE, input, or a prompt)"
    )
    s.add_argument("name", choices=list(NAMES))
    s.add_argument("file", nargs="?", help="read the value from this file")
    r = actions.add_parser("remove", parents=[common], help="forget one")
    r.add_argument("name", choices=list(NAMES))
    p.set_defaults(func=cmd_credentials)

    p = sub.add_parser("auth", help="sign in to Earth Engine")
    p.add_argument("service", choices=["earthengine"])
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="approve on another device and paste the code it shows",
    )
    p.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="port of the local sign-in page"
    )
    p.add_argument("--project", help="also set the Earth Engine project")
    p.set_defaults(func=cmd_auth)

    p = sub.add_parser("doctor", help="check this machine is ready to run jobs")
    p.add_argument(
        "--offline", action="store_true", help="skip contacting Earth Engine"
    )
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_doctor)
