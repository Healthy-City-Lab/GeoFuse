"""Earth Engine sign-in and initialisation (``geofuse.ee_auth``).

The sign-in flow stores its token without touching Earth Engine's own
credentials file; jobs initialise with a stored service account first, then
the stored sign-in, and the configured project; with nothing stored they say
what to do, or sign in once where a browser can open.
"""

import json
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ee
from _credential_store import LOGIN, SERVICE_KEY, StoreCase
from ee import oauth

from geofuse import credentials as cr
from geofuse import ee_auth


class SignInTests(StoreCase):
    def test_the_token_goes_to_the_store_not_earth_engines_file(self):
        real_path = oauth.get_credentials_path
        seen = []

        def authenticate(auth_mode=None, force=False, **_):
            seen.append((auth_mode, force, oauth.get_credentials_path()))
            with open(oauth.get_credentials_path(), "w", encoding="utf-8") as f:
                f.write(LOGIN)

        with mock.patch.object(oauth, "authenticate", authenticate):
            ee_auth.sign_in(browser=False)
            ee_auth.sign_in(browser=True, port=9001)
        self.assertIs(oauth.get_credentials_path, real_path)
        modes = [s[:2] for s in seen]
        self.assertEqual(modes, [("notebook", True), ("localhost:9001", True)])
        self.assertNotEqual(seen[0][2], real_path())
        self.assertFalse(os.path.exists(seen[0][2]))
        stored = json.loads(cr.get("earthengine-login"))
        self.assertEqual(stored["refresh_token"], "1//token")

    def test_an_unfinished_sign_in_stores_nothing(self):
        with mock.patch.object(oauth, "authenticate", lambda **_: None):
            with self.assertRaisesRegex(ee_auth.EarthEngineAuthError, "did not finish"):
                ee_auth.sign_in(browser=False)
        self.assertIsNone(cr.get("earthengine-login"))


class InitializeTests(StoreCase):
    def init(self, *, side_effect=None, browser=False, project=None):
        calls = []

        def initialize(**kw):
            calls.append(kw)
            if side_effect and len(calls) == 1:
                raise side_effect

        with (
            mock.patch.object(ee, "Initialize", initialize),
            mock.patch.object(ee_auth, "can_open_browser", lambda: browser),
            mock.patch.object(ee_auth, "configured_project", lambda: "cfg-project"),
        ):
            source = ee_auth.initialize(project)
        return source, calls

    def test_the_stored_sign_in_and_the_configured_project_are_used(self):
        cr.put("earthengine-login", LOGIN)
        source, calls = self.init()
        self.assertEqual(source, "stored sign-in")
        creds = calls[0]["credentials"]
        self.assertEqual(creds.refresh_token, "1//token")
        self.assertEqual(creds.quota_project_id, "123")
        self.assertEqual(calls[0]["project"], "cfg-project")
        self.assertEqual(self.init(project="mine")[1][0]["project"], "mine")

    def test_a_service_account_comes_first(self):
        cr.put("earthengine-login", LOGIN)
        cr.put("earthengine-service-account", SERVICE_KEY)
        with mock.patch.object(ee, "ServiceAccountCredentials", lambda e, key_data: e):
            source, calls = self.init()
        self.assertIn("runner@project", source)
        email = "runner@project.iam.gserviceaccount.com"
        self.assertEqual(calls[0]["credentials"], email)

    def test_without_credentials_the_advice_names_the_command(self):
        refused = ee_auth.EarthEngineAuthError
        with self.assertRaisesRegex(refused, "geofuse auth earthengine"):
            self.init(side_effect=ee.EEException("Please authorize access"))
        with self.assertRaisesRegex(refused, "earthengine.project"):
            self.init(side_effect=ee.EEException("ee.Initialize: no project found."))

    def test_a_desktop_signs_in_once_and_retries(self):
        signed = []

        def sign_in(**kw):
            signed.append(kw)
            cr.put("earthengine-login", LOGIN)

        with mock.patch.object(ee_auth, "sign_in", sign_in):
            source, calls = self.init(side_effect=ee.EEException("x"), browser=True)
        self.assertEqual(signed, [{"browser": True}])
        self.assertEqual(source, "stored sign-in")
        self.assertIn("credentials", calls[1])


if __name__ == "__main__":
    unittest.main()
