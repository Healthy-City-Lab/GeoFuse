"""A private credential store for tests: a temporary file, or an in-memory
keyring that refuses long entries as Windows Credential Manager does."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geofuse import credentials as cr

SERVICE_KEY = json.dumps(
    {
        "type": "service_account",
        "client_email": "runner@project.iam.gserviceaccount.com",
        "private_key": "-----BEGIN PRIVATE KEY-----\n" + "A" * 1600 + "\n-----END",
        "project_id": "project",
    }
)
LOGIN = json.dumps({"refresh_token": "1//token", "scopes": ["s"], "project": "123"})


class FakeKeyring:
    """An in-memory keyring with Windows Credential Manager's entry limit."""

    priority = 5

    def __init__(self):
        self.entries = {}

    def get_keyring(self):
        return self

    def get_password(self, service, key):
        return self.entries.get((service, key))

    def set_password(self, service, key, value):
        if len(value) > 1280:
            raise OSError("The stub received bad data")
        self.entries[(service, key)] = value

    def delete_password(self, service, key):
        del self.entries[(service, key)]


class StoreCase(unittest.TestCase):
    """Each test gets its own store: ``store = "file"`` or ``"auto"`` (the
    fake keyring)."""

    store = "file"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "credentials.toml"
        self.keyring = FakeKeyring()
        patches = [
            mock.patch.object(cr, "credentials_path", lambda: self.path),
            mock.patch.object(cr, "_setting", lambda: self.store),
            mock.patch.object(
                cr, "_keyring", lambda: self.keyring if self.store != "file" else None
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()
