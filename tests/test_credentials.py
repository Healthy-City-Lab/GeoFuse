"""The credential store (``geofuse.credentials``).

Values round-trip through the file store and through a keyring that, like
Windows Credential Manager, refuses long entries; the file is readable by its
owner only; malformed values are refused with the reason.
"""

import json
import os
import stat
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _credential_store import LOGIN, SERVICE_KEY, StoreCase

from geofuse import credentials as cr


class FileStoreTests(StoreCase):
    def test_values_round_trip_and_are_removed(self):
        self.assertIn("file", cr.put("streetview-key", "  AIzaKEY \n"))
        cr.put("earthengine-service-account", SERVICE_KEY)
        self.assertEqual(cr.get("streetview-key"), "AIzaKEY")
        self.assertEqual(
            json.loads(cr.get("earthengine-service-account")), json.loads(SERVICE_KEY)
        )
        self.assertEqual(
            cr.stored(),
            {
                "streetview-key": "stored",
                "earthengine-service-account": "service account "
                "runner@project.iam.gserviceaccount.com",
            },
        )
        self.assertTrue(cr.remove("streetview-key"))
        self.assertFalse(cr.remove("streetview-key"))
        self.assertIsNone(cr.get("streetview-key"))
        cr.remove("earthengine-service-account")
        self.assertFalse(self.path.exists())

    @unittest.skipIf(os.name == "nt", "POSIX permissions")
    def test_the_file_is_readable_by_its_owner_only(self):
        cr.put("streetview-key", "AIzaKEY")
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertTrue(cr.file_is_private())

    def test_malformed_values_are_refused_with_the_reason(self):
        for name, value, reason in (
            ("streetview-key", "two words", "no spaces"),
            ("streetview-key", "  ", "empty"),
            ("earthengine-service-account", "{not json", "must be JSON"),
            ("earthengine-service-account", LOGIN, "service account"),
            ("earthengine-login", SERVICE_KEY, "refresh token"),
            ("google-password", "x", "Unknown credential"),
        ):
            with self.assertRaisesRegex(cr.CredentialError, reason):
                cr.put(name, value)
        self.assertFalse(self.path.exists())


class KeyringStoreTests(StoreCase):
    store = "auto"

    def test_long_values_are_split_over_entries(self):
        self.assertIn("keyring", cr.put("earthengine-service-account", SERVICE_KEY))
        self.assertGreater(len(self.keyring.entries), 2)
        self.assertEqual(
            json.loads(cr.get("earthengine-service-account")), json.loads(SERVICE_KEY)
        )
        cr.put("earthengine-service-account", SERVICE_KEY.replace("A", "B"))
        self.assertIn("BBB", cr.get("earthengine-service-account"))
        self.assertTrue(cr.remove("earthengine-service-account"))
        self.assertEqual(self.keyring.entries, {})
        self.assertFalse(self.path.exists())

    def test_a_required_keyring_that_is_missing_is_an_error(self):
        self.keyring = None
        self.assertEqual(cr.store(), "file")
        self.store = "keyring"
        with self.assertRaisesRegex(cr.CredentialError, "No usable keyring"):
            cr.store()


if __name__ == "__main__":
    unittest.main()
