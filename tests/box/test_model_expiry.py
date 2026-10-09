"""Tests for box/model_expiry.py: a WARNING before OpenRouter retires a model the box uses."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime

from home_guard_project.box import model_expiry as me

NOW = datetime(2026, 10, 9, 23, 0).timestamp()
LIST = [{"id": "qwen/qwen3.5-9b", "expiration_date": "2026-10-21"},
        {"id": "qwen/qwen3-vl-8b-instruct", "expiration_date": None},
        {"id": "google/gemini-3.1-flash-lite", "expiration_date": "2027-03-01"},
        {"id": "google/gemini-2.5-flash-lite", "expiration_date": "2026-10-20"},
        {"id": "google/gemini-3.5-flash-lite"},
        {"id": "qwen/qwen3.7-flash", "expiration_date": "2026-10-05"}]
BOX = {"vlm_provider": "openrouter", "vlm_model": "qwen/qwen3.5-9b",
       "vlm_fallback_provider": "openrouter", "vlm_fallback_model": "qwen/qwen3-vl-8b-instruct",
       "describer_model": "qwen/qwen3.5-9b"}


class ConfiguredModelsTest(unittest.TestCase):
    def test_every_openrouter_role_with_the_defaults(self) -> None:
        roles = dict((m, r) for r, m in me.configured_models(BOX))
        self.assertIn("qwen/qwen3.5-9b", roles)                     # the Eye
        self.assertIn("qwen/qwen3-vl-8b-instruct", roles)           # its fallback
        self.assertIn("google/gemini-3.1-flash-lite", roles)        # the translator (default)
        self.assertIn("google/gemini-3.5-flash-lite", roles)        # its racer (default)

    def test_other_providers_and_an_off_describer_are_not_checked(self) -> None:
        got = me.configured_models({"vlm_provider": "openai", "vlm_model": "gpt-4o", "alert_describer": "off",
                                    "messenger_provider": "google", "agent_model": "openai:gpt-4o",
                                    "agent_fast_model": "openrouter:qwen/qwen3.7-flash"})
        self.assertEqual(got, [("assistant (agent_fast_model)", "qwen/qwen3.7-flash")])

    def test_the_fallback_uses_the_eye_provider_when_it_names_none(self) -> None:
        got = me.configured_models({"vlm_provider": "openrouter", "vlm_model": "a/b", "vlm_fallback_model": "c/d",
                                    "alert_describer": "off", "messenger_provider": "google"})
        self.assertEqual([m for _, m in got], ["a/b", "c/d"])


class FindingsTest(unittest.TestCase):
    def test_within_14_days_past_and_missing_are_found_the_rest_not(self) -> None:
        configured = [("Eye", "qwen/qwen3.5-9b"), ("fallback", "qwen/qwen3-vl-8b-instruct"),
                      ("translator", "google/gemini-3.1-flash-lite"), ("racer", "google/gemini-2.5-flash-lite"),
                      ("x", "qwen/qwen3.7-flash"), ("y", "qwen/qwen3-vl-32b-instruct")]
        found = me.findings(LIST, configured, datetime.fromtimestamp(NOW).date())
        by = {f["model"]: f for f in found}
        self.assertEqual(set(by), {"qwen/qwen3.5-9b", "google/gemini-2.5-flash-lite", "qwen/qwen3.7-flash",
                                   "qwen/qwen3-vl-32b-instruct"})
        self.assertEqual((by["qwen/qwen3.5-9b"]["expires"], by["qwen/qwen3.5-9b"]["days_left"]), ("2026-10-21", 12))
        self.assertEqual(by["qwen/qwen3.7-flash"]["days_left"], -4)
        self.assertTrue(by["qwen/qwen3-vl-32b-instruct"]["gone"])

    def test_a_date_beyond_the_window_is_quiet(self) -> None:
        found = me.findings(LIST, [("Eye", "qwen/qwen3.5-9b")], datetime(2026, 10, 1).date())
        self.assertEqual(found, [])


class CheckTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self.dir.name, "state", me.CACHE_NAME)
        self.calls = 0

    def tearDown(self) -> None:
        self.dir.cleanup()

    def fetch(self):
        self.calls += 1
        return LIST

    def offline(self):
        self.calls += 1
        raise OSError("no network")

    def test_warns_per_finding_and_caches_the_list_for_a_day(self) -> None:
        with self.assertLogs("box.model_expiry", "WARNING") as logs:
            found = me.check(BOX, self.cache, now=NOW, fetch=self.fetch)
        # The Eye and the describer both name qwen3.5-9b; the racer default is no longer 2.5.
        self.assertEqual(sorted(f["role"].split(" ")[0] for f in found), ["Eye", "describer"])
        self.assertEqual({f["model"] for f in found}, {"qwen/qwen3.5-9b"})
        self.assertTrue(any("qwen/qwen3.5-9b" in line and "2026-10-21" in line and "12 days" in line
                            for line in logs.output))
        self.assertTrue(os.path.isfile(self.cache))
        me.check(BOX, self.cache, now=NOW + 3600, fetch=self.fetch)          # within a day: the cache
        self.assertEqual(self.calls, 1)
        me.check(BOX, self.cache, now=NOW + 25 * 3600, fetch=self.fetch)     # a day later: fetched again
        self.assertEqual(self.calls, 2)

    def test_offline_uses_the_old_cache_or_checks_nothing_and_never_raises(self) -> None:
        self.assertEqual(me.check(BOX, self.cache, now=NOW, fetch=self.offline), [])
        os.makedirs(os.path.dirname(self.cache))
        with open(self.cache, "w", encoding="utf-8") as f:
            json.dump({"fetched_at": NOW - 10 * 86400, "models": LIST}, f)
        found = me.check(BOX, self.cache, now=NOW, fetch=self.offline)
        self.assertIn("qwen/qwen3.5-9b", {f["model"] for f in found})
        with open(self.cache, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(me.check(BOX, self.cache, now=NOW, fetch=self.offline), [])

    def test_a_box_without_openrouter_models_fetches_nothing(self) -> None:
        box = {"vlm_provider": "openai", "vlm_model": "gpt-4o", "alert_describer": "off",
               "messenger_provider": "google"}
        self.assertEqual(me.check(box, self.cache, now=NOW, fetch=self.fetch), [])
        self.assertEqual(self.calls, 0)


if __name__ == "__main__":
    unittest.main()
