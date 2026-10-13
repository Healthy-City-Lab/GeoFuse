"""Earth Engine sign-in: which credentials a job uses, and how to sign in.

A job initialises Earth Engine with, in order: the stored service-account key,
the stored sign-in, or Earth Engine's own default (its credentials file, or
application-default credentials). The project is the one the job names, else
``earthengine.project`` of the user configuration, else the one the
credentials carry.

:func:`sign_in` runs Earth Engine's own OAuth flow, but the flow writes into a
private temporary folder and the token moves into the credential store
(:mod:`geofuse.credentials`), so Earth Engine's plain-text credentials file is
neither written nor needed. Two flows:

* **browser** — a local page receives the sign-in (Earth Engine's
  ``localhost`` mode). On a machine without a browser, forward the port
  first (``ssh -L 8085:localhost:8085 HOST``) and open the printed address on
  your own computer.
* **no browser** — open the printed address on any device and paste the code
  it shows (Earth Engine's ``notebook`` mode; Google expires these sign-ins
  after about a week).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any

LOGIN = "earthengine-login"
SERVICE_ACCOUNT = "earthengine-service-account"
DEFAULT_PORT = 8085
#: The source :func:`stored_credentials` names when nothing is stored.
DEFAULT_SOURCE = "Earth Engine default"


class EarthEngineAuthError(RuntimeError):
    """Earth Engine could not be initialised; the message says what to do."""


def configured_project() -> str | None:
    from geofuse import userconfig

    try:
        return userconfig.load().ee_project
    except userconfig.ConfigError:
        return None


def can_open_browser() -> bool:
    """Whether this session can show a browser window."""
    if sys.platform in ("win32", "darwin"):
        return True
    return any(os.environ.get(v) for v in ("DISPLAY", "WAYLAND_DISPLAY"))


def _login_credentials(info: dict):
    from ee import oauth
    from google.oauth2.credentials import Credentials

    return Credentials(
        None,
        token_uri=oauth.TOKEN_URI,
        refresh_token=info["refresh_token"],
        client_id=info.get("client_id", oauth.CLIENT_ID),
        client_secret=info.get("client_secret", oauth.CLIENT_SECRET),
        scopes=info.get("scopes", oauth.SCOPES),
        quota_project_id=info.get("project"),
    )


def stored_credentials() -> tuple[Any | None, str]:
    """``(credentials, source)``: the stored key or sign-in, or ``None`` and
    :data:`DEFAULT_SOURCE` when neither is stored."""
    from geofuse import credentials

    key = credentials.get(SERVICE_ACCOUNT)
    if key:
        import ee

        email = json.loads(key)["client_email"]
        return ee.ServiceAccountCredentials(email, key_data=key), (
            f"service account {email}"
        )
    login = credentials.get(LOGIN)
    if login:
        return _login_credentials(json.loads(login)), "stored sign-in"
    return None, DEFAULT_SOURCE


def _advice(exc: Exception, source: str) -> str:
    if "no project found" in str(exc):
        return (
            "Earth Engine needs a Google Cloud project: run "
            "`geofuse config set earthengine.project YOUR-PROJECT`."
        )
    if source == DEFAULT_SOURCE:
        return (
            f"Earth Engine is not signed in on this machine ({exc}). Run "
            "`geofuse auth earthengine` (with --no-browser where no browser can "
            "open), or store a service account key with `geofuse credentials "
            "set earthengine-service-account KEY.json`."
        )
    if source.startswith("service account"):
        return (
            f"Earth Engine refused the {source} ({exc}). Check that the account "
            "is registered for Earth Engine, or store another key."
        )
    return (
        f"Earth Engine refused the {source} ({exc}). Sign in again with "
        "`geofuse auth earthengine`."
    )


def initialize(project: str | None = None, *, sign_in_if_needed: bool = True) -> str:
    """Initialise Earth Engine for this process; returns the credential source.

    With nothing stored, Earth Engine's default is tried; when that fails for
    want of a sign-in and a browser can open here, :func:`sign_in` runs first
    (the GUI's first-use path). Otherwise :class:`EarthEngineAuthError` says
    what to do.
    """
    import ee
    from google.auth.exceptions import GoogleAuthError

    project = project or configured_project()
    creds, source = stored_credentials()
    kwargs: dict[str, Any] = {"project": project} if project else {}
    if creds is not None:
        kwargs["credentials"] = creds
    try:
        ee.Initialize(**kwargs)
        return source
    except (ee.EEException, GoogleAuthError) as exc:
        if (
            creds is not None
            or "no project found" in str(exc)
            or not (sign_in_if_needed and can_open_browser())
        ):
            raise EarthEngineAuthError(_advice(exc, source)) from exc
    sign_in(browser=True)
    creds, source = stored_credentials()
    ee.Initialize(**kwargs, credentials=creds)
    return source


def sign_in(*, browser: bool = True, port: int = DEFAULT_PORT) -> str:
    """Run Earth Engine's sign-in and store the result; returns where it went."""
    from ee import oauth

    from geofuse import credentials

    mode = f"localhost:{port}" if browser else "notebook"
    with tempfile.TemporaryDirectory(prefix="geofuse-ee-") as tmp:
        path = os.path.join(tmp, "credentials")
        original = oauth.get_credentials_path
        oauth.get_credentials_path = lambda: path
        try:
            oauth.authenticate(auth_mode=mode, force=True)
        finally:
            oauth.get_credentials_path = original
        if not os.path.isfile(path):
            raise EarthEngineAuthError(
                "The sign-in did not finish; nothing was stored."
            )
        with open(path, encoding="utf-8") as f:
            value = f.read()
    return credentials.put(LOGIN, value)
