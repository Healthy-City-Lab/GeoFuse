"""Credentials GeoFuse uses, kept outside the repository.

::

    streetview-key                 a Google Street View API key (optional:
                                   without one, GVI downloads use the scraper)
    earthengine-service-account    a service account's JSON key
    earthengine-login              the sign-in ``geofuse auth earthengine`` stores

They live in the operating system's keyring when one works (Windows Credential
Manager, macOS Keychain, Secret Service on a Linux desktop), otherwise in
``credentials.toml`` in the user configuration folder, readable by its owner
only; ``[credentials] store`` in the user configuration picks one. Nothing here
reads environment variables, and no value is written to run files, job specs,
status files or logs.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import stat
import tomllib
from pathlib import Path

#: Each credential and what it is.
NAMES: dict[str, str] = {
    "streetview-key": "Google Street View API key (optional; GVI scrapes without)",
    "earthengine-service-account": "Earth Engine service account JSON key",
    "earthengine-login": "Earth Engine sign-in (from `geofuse auth earthengine`)",
}
SERVICE = "geofuse"

# Windows Credential Manager holds at most 1280 characters per entry; longer
# values (service account keys) are split over numbered entries.
_CHUNK = 1000
_CHUNKS = "geofuse-chunks:"

_BACKEND_NAMES = {
    "WinVaultKeyring": "Windows Credential Manager",
    "Keyring": "macOS Keychain",
    "SecretService": "Secret Service",
    "DBusKeyring": "KWallet",
}


class CredentialError(ValueError):
    """A credential that cannot be stored or read, with the reason."""


def _check_name(name: str) -> None:
    if name not in NAMES:
        raise CredentialError(
            f"Unknown credential '{name}' (known: {', '.join(NAMES)})."
        )


def validate(name: str, value: str) -> str:
    """``value`` as it is stored, or :class:`CredentialError` naming the fault."""
    _check_name(name)
    value = value.strip()
    if not value:
        raise CredentialError(f"'{name}' is empty.")
    if name == "streetview-key":
        if any(ch.isspace() for ch in value):
            raise CredentialError("A Street View key is one word, with no spaces.")
        return value
    try:
        doc = json.loads(value)
    except json.JSONDecodeError as exc:
        raise CredentialError(f"'{name}' must be JSON ({exc.msg}).") from None
    if not isinstance(doc, dict):
        raise CredentialError(f"'{name}' must be a JSON object.")
    if name == "earthengine-service-account":
        if doc.get("type") != "service_account" or not (
            doc.get("client_email") and doc.get("private_key")
        ):
            raise CredentialError(
                "Not a service account key: expected the JSON key file of a "
                "service account."
            )
    elif not doc.get("refresh_token"):
        raise CredentialError("Not an Earth Engine sign-in: it has no refresh token.")
    return json.dumps(doc, separators=(",", ":"))


def describe(name: str, value: str) -> str:
    """A line about a stored value that reveals nothing secret."""
    if name == "earthengine-service-account":
        return f"service account {json.loads(value).get('client_email')}"
    if name == "earthengine-login":
        return "signed in"
    return "stored"


# ────────────────────────────────────────────────────────────────────
# Where values live
# ────────────────────────────────────────────────────────────────────


def credentials_path() -> Path:
    """The file store (used when no keyring works, or when configured)."""
    from geofuse.userconfig import config_dir

    return config_dir() / "credentials.toml"


def _setting() -> str:
    from geofuse import userconfig

    try:
        return userconfig.load().credential_store
    except userconfig.ConfigError:
        return "auto"


@functools.cache
def _keyring():
    """The ``keyring`` module when a usable backend exists, else ``None``."""
    try:
        import keyring
    except ImportError:
        return None
    try:
        if getattr(keyring.get_keyring(), "priority", 0) < 1:
            return None
        keyring.get_password(SERVICE, "probe")
    except Exception:
        return None
    return keyring


def store() -> str:
    """Where credentials are kept on this machine: ``keyring`` or ``file``."""
    setting = _setting()
    if setting == "file":
        return "file"
    if _keyring() is not None:
        return "keyring"
    if setting == "keyring":
        raise CredentialError(
            "No usable keyring on this machine; set `[credentials] store` to "
            "'file' or 'auto' in the user configuration."
        )
    return "file"


def store_description() -> str:
    if store() == "keyring":
        name = type(_keyring().get_keyring()).__name__
        return f"{_BACKEND_NAMES.get(name, name)} (keyring)"
    return f"{credentials_path()} (file, readable by you only)"


# ── Keyring store ───────────────────────────────────────────────────


def _chunk_keys(name: str, head: str) -> list[str]:
    """The entries holding the parts of a value split by :func:`_kr_put`."""
    if not head.startswith(_CHUNKS):
        return []
    return [f"{name}#{i}" for i in range(int(head[len(_CHUNKS) :]))]


def _kr_get(kr, name: str) -> str | None:
    head = kr.get_password(SERVICE, name)
    if head is None or not head.startswith(_CHUNKS):
        return head
    parts = [kr.get_password(SERVICE, key) for key in _chunk_keys(name, head)]
    if any(part is None for part in parts):
        raise CredentialError(f"'{name}' is incomplete in the keyring; store it again.")
    return "".join(parts)


def _kr_remove(kr, name: str) -> bool:
    head = kr.get_password(SERVICE, name)
    if head is None:
        return False
    for key in [name, *_chunk_keys(name, head)]:
        with contextlib.suppress(Exception):
            kr.delete_password(SERVICE, key)
    return True


def _kr_put(kr, name: str, value: str) -> None:
    _kr_remove(kr, name)
    if len(value) <= _CHUNK:
        kr.set_password(SERVICE, name, value)
        return
    parts = [value[i : i + _CHUNK] for i in range(0, len(value), _CHUNK)]
    for i, part in enumerate(parts):
        kr.set_password(SERVICE, f"{name}#{i}", part)
    kr.set_password(SERVICE, name, f"{_CHUNKS}{len(parts)}")


# ── File store ──────────────────────────────────────────────────────

_HEADER = (
    "# GeoFuse credentials, readable by you only. "
    "Manage them with `geofuse credentials`.\n"
)


def _file_read() -> dict[str, str]:
    path = credentials_path()
    if not path.is_file():
        return {}
    try:
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CredentialError(f"Could not read {path}: {exc}") from exc
    return {k: v for k, v in doc.items() if k in NAMES and isinstance(v, str)}


def _file_write(values: dict[str, str]) -> None:
    path = credentials_path()
    if not values:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(_HEADER)
        for key, value in values.items():
            f.write(f"{key} = {json.dumps(value)}\n")
    os.replace(tmp, path)


def file_is_private() -> bool:
    """Whether the file store, if any, is readable by its owner only."""
    path = credentials_path()
    if os.name == "nt" or not path.exists():
        return True
    return stat.S_IMODE(path.stat().st_mode) & 0o077 == 0


# ────────────────────────────────────────────────────────────────────
# Reading and writing
# ────────────────────────────────────────────────────────────────────


def get(name: str) -> str | None:
    """The stored value of ``name``, or ``None``."""
    _check_name(name)
    if store() == "keyring":
        return _kr_get(_keyring(), name)
    return _file_read().get(name)


def put(name: str, value: str) -> str:
    """Store ``value`` (checked by :func:`validate`) under ``name``; returns
    where it went."""
    value = validate(name, value)
    if store() == "keyring":
        _kr_put(_keyring(), name, value)
    else:
        values = _file_read()
        values[name] = value
        _file_write(values)
    return store_description()


def remove(name: str) -> bool:
    """Forget ``name``; ``False`` when nothing was stored."""
    _check_name(name)
    if store() == "keyring":
        return _kr_remove(_keyring(), name)
    values = _file_read()
    if name not in values:
        return False
    del values[name]
    _file_write(values)
    return True


def stored() -> dict[str, str]:
    """Each stored credential and its :func:`describe` line."""
    found = {}
    for name in NAMES:
        value = get(name)
        if value:
            found[name] = describe(name, value)
    return found
