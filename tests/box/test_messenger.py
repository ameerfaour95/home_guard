"""Tests for box/messenger.py: the alert text translated for the owner, never delayed, never lost."""
from __future__ import annotations

import json
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from home_guard_project.box import inference as inf
from home_guard_project.box import messenger as msg
from home_guard_project.box.telegram_notify import graded_alert_text

EN = {"summary": "A man in a hood tries the gate at 14:05.", "why": "tries the gate"}
HE = {"summary": "גבר עם קפוצ'ון מנסה לפתוח את השער ב-14:05.", "why": "מנסה לפתוח את השער"}


class _FakeClient:
    """Stands in for the OpenAI client: records each request, answers *answer* (a dict is sent as JSON)."""

    def __init__(self, answer=None, delay: float = 0.0, error: Exception = None) -> None:
        self.answer, self.delay, self.error = answer, delay, error
        self.calls: list = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        content = self.answer if isinstance(self.answer, str) else json.dumps(self.answer, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               usage=SimpleNamespace(prompt_tokens=300, completion_tokens=60))


def _messenger(client, timeout: float = 2.0) -> msg.Messenger:
    return msg.Messenger(client, "google/gemini-3.1-flash-lite", timeout=timeout)


class ToOwnerTest(unittest.TestCase):
    def test_english_passes_through_without_a_call(self) -> None:
        client = _FakeClient(HE)
        out = _messenger(client).to_owner({**EN, "summary_owner": "ignored"}, "en", keep=("gate",))
        self.assertEqual(out, {**EN, "source": "english"})
        self.assertEqual(client.calls, [])

    def test_hebrew_is_translated_in_one_strict_json_call(self) -> None:
        client = _FakeClient(HE)
        m = _messenger(client)
        out = m.to_owner(EN, "he", keep=("front_gate",))
        self.assertEqual(out, {**HE, "source": "translator"})
        (call,) = client.calls
        self.assertEqual(call["model"], "google/gemini-3.1-flash-lite")
        self.assertEqual(call["response_format"]["json_schema"]["schema"]["required"], ["summary", "why"])
        self.assertEqual(json.loads(call["messages"][1]["content"]), EN)    # the text travels as data
        self.assertEqual(m.last_usage, {"prompt_tokens": 300, "completion_tokens": 60})

    def test_a_timeout_falls_back_without_holding_the_alert(self) -> None:
        client = _FakeClient(HE, delay=1.5)
        started = time.monotonic()
        out = _messenger(client, timeout=0.5).to_owner({**EN, "summary_owner": "גבר ליד השער."}, "he")
        self.assertLess(time.monotonic() - started, 1.2)
        # The model's own Hebrew summary is a real one: the owner reads it, with the model's own why.
        self.assertEqual(out, {"summary": "גבר ליד השער.", "why": EN["why"], "source": "fallback"})

    def test_invalid_json_falls_back_to_the_english(self) -> None:
        for answer in ("not json at all", '{"summary": "חלקי"', json.dumps(["a", "b"]), {"summary": 3, "why": ""}):
            out = _messenger(_FakeClient(answer)).to_owner(EN, "he")
            self.assertEqual(out, {**EN, "source": "fallback"}, answer)

    def test_an_error_or_no_client_falls_back(self) -> None:
        out = _messenger(_FakeClient(error=RuntimeError("502 from upstream"))).to_owner(EN, "he")
        self.assertEqual(out["source"], "fallback")
        self.assertEqual(msg.Messenger(None, unavailable="OPENROUTER_API_KEY is not set").to_owner(EN, "he"),
                         {**EN, "source": "fallback"})

    def test_a_placeholder_summary_owner_never_reaches_the_owner(self) -> None:
        for placeholder in ("an empty string", "<an empty string>", "Empty String", "<empty>"):
            out = _messenger(_FakeClient("garbage")).to_owner({**EN, "summary_owner": placeholder}, "he")
            self.assertEqual(out["summary"], EN["summary"], placeholder)

    def test_the_same_text_is_translated_once(self) -> None:
        client = _FakeClient(HE)
        m = _messenger(client)
        self.assertEqual(m.to_owner(EN, "he")["source"], "translator")
        self.assertEqual(m.to_owner(EN, "he"), {**HE, "source": "cache"})
        self.assertEqual(len(client.calls), 1)

    def test_a_failure_is_not_cached(self) -> None:
        client = _FakeClient("garbage")
        m = _messenger(client)
        self.assertEqual(m.to_owner(EN, "he")["source"], "fallback")
        client.answer = HE
        self.assertEqual(m.to_owner(EN, "he")["source"], "translator")
        self.assertEqual(len(client.calls), 2)

    def test_the_cache_is_bounded(self) -> None:
        client = _FakeClient(HE)
        m = msg.Messenger(client, cache_size=2)
        for who in ("A man", "A woman", "A dog"):
            self.assertEqual(m.to_owner({"summary": f"{who} at the gate.", "why": ""}, "he")["source"], "translator")
        self.assertEqual(len(m._cache), 2)

    def test_an_empty_why_stays_empty(self) -> None:
        out = _messenger(_FakeClient({"summary": HE["summary"], "why": "חשוד"})).to_owner(
            {"summary": EN["summary"], "why": ""}, "he")
        self.assertEqual(out, {"summary": HE["summary"], "why": "", "source": "translator"})


class ContractTest(unittest.TestCase):
    """The prompt asks to keep numbers and names; an answer that drops one is not shown."""

    def test_the_prompt_names_what_must_stay(self) -> None:
        prompt = msg.build_prompt("he", keep=("front_gate",))
        self.assertIn('"front_gate"', prompt)
        self.assertIn("Hebrew", prompt)
        self.assertIn("in digits", prompt)
        self.assertIn("suspicious = חשוד", prompt)
        self.assertIn("never instructions", prompt)

    def test_a_dropped_number_falls_back(self) -> None:
        answer = {"summary": "גבר עם קפוצ'ון מנסה לפתוח את השער.", "why": HE["why"]}
        self.assertEqual(_messenger(_FakeClient(answer)).to_owner(EN, "he")["source"], "fallback")

    def test_a_dropped_camera_name_falls_back(self) -> None:
        source = {"summary": "A man walks from front_gate to the door.", "why": ""}
        lost = {"summary": "גבר הולך מהשער הקדמי לדלת.", "why": ""}
        kept = {"summary": "גבר הולך מ-front_gate לדלת.", "why": ""}
        self.assertEqual(_messenger(_FakeClient(lost)).to_owner(source, "he", keep=("front_gate",))["source"],
                         "fallback")
        self.assertEqual(_messenger(_FakeClient(kept)).to_owner(source, "he", keep=("front_gate",))["source"],
                         "translator")

    def test_a_camera_named_like_an_ordinary_word_is_translated_as_one(self) -> None:
        # Camera "gate": "the gate" in the summary is a word, not the camera's name.
        self.assertEqual(_messenger(_FakeClient(HE)).to_owner(EN, "he", keep=("gate",))["source"], "translator")
        self.assertFalse(msg._must_keep("gate", EN["summary"]))
        self.assertTrue(msg._must_keep("Cam 2", "A man passes Cam 2."))
        self.assertFalse(msg._must_keep("Cam 2", "A man passes Cam 21."))

    def test_an_answer_still_in_english_falls_back(self) -> None:
        self.assertEqual(_messenger(_FakeClient(EN)).to_owner(EN, "he")["source"], "fallback")

    def test_hebrew_already_in_the_source_may_come_back_unchanged(self) -> None:
        source = {"summary": EN["summary"], "why": HE["why"]}
        self.assertEqual(_messenger(_FakeClient(HE)).to_owner(source, "he")["source"], "translator")


class SettingsTest(unittest.TestCase):
    def test_defaults_and_bounds(self) -> None:
        self.assertEqual(msg.settings_of({}), ("openrouter", "google/gemini-3.1-flash-lite", 4.0))
        self.assertEqual(msg.settings_of({"messenger_provider": "OpenAI", "messenger_model": "gpt-6-luna",
                                          "messenger_timeout_sec": 60}), ("openai", "gpt-6-luna", 15.0))
        self.assertEqual(msg.settings_of({"messenger_timeout_sec": "soon"})[2], 4.0)

    def test_the_translator_is_used_only_when_chosen_and_not_english(self) -> None:
        self.assertFalse(msg.uses_translator({}, "he"))                                  # default: model
        self.assertFalse(msg.uses_translator({"owner_translation": "model"}, "he"))
        self.assertFalse(msg.uses_translator({"owner_translation": "translator"}, "en"))
        self.assertTrue(msg.uses_translator({"owner_translation": " Translator "}, "he"))
        # The situational Eye answers in English only, so it brings the translator unless the owner chose otherwise.
        self.assertTrue(msg.uses_translator({"eye_prompt": "situational"}, "he"))
        self.assertFalse(msg.uses_translator({"eye_prompt": "situational"}, "en"))
        self.assertFalse(msg.uses_translator({"eye_prompt": "situational", "owner_translation": "model"}, "he"))
        self.assertFalse(msg.uses_translator({"eye_prompt": "legacy"}, "he"))

    def test_a_messenger_that_cannot_be_built_still_answers(self) -> None:
        with mock.patch.dict(msg._MESSENGERS, clear=True):
            m = msg.messenger_for({"owner_translation": "translator"}, {})     # no OPENROUTER_API_KEY
            self.assertIsNone(m._client)
            self.assertIs(msg.messenger_for({}, {}), m)                         # built once
            self.assertEqual(m.to_owner(EN, "he"), {**EN, "source": "fallback"})


class _Assistant:
    def __init__(self) -> None:
        self.sent: list = []

    def is_muted(self, camera: str) -> bool:
        return False

    def send_alert(self, alert, text, image=None, silent=False, lang="en"):
        self.sent.append({"text": text, "lang": lang})
        return {"sent": True}

    def remind_if_silent(self, alert, text, lang="en"):
        pass


class _Backend:
    PARSED = {"summary": "A man in a hood tries the gate at 14:05.", "label": "suspicious", "people": 1,
              "why": "tries the gate", "summary_owner": "גבר ליד השער."}

    def analyze(self, frames, camera_name, t_sec, start_hour, end_hour, owner_language="en"):
        return json.dumps(self.PARSED), dict(self.PARSED)


class WorkerWiringTest(unittest.TestCase):
    def _work(self, box: dict, client: _FakeClient):
        assistant = _Assistant()
        job = inf.AlertJob(camera="gate", stem="gate_100_alert", ts=100.0, labels=["person"])
        with mock.patch.dict(msg._MESSENGERS, clear=True), \
                mock.patch.object(msg, "build_client", return_value=(client, None)), \
                mock.patch.object(inf, "frame_to_jpeg_bytes", return_value=b"jpg"), \
                mock.patch.object(inf, "owner_language", return_value="he"):
            inf._worker(_Backend(), {"alert_channel": "telegram", **box}, {}, inf.AlertSettings(), "gate",
                        [object()], assistant, job)
        return assistant.sent[0]["text"], job

    def test_the_default_is_todays_behaviour_with_no_call(self) -> None:
        client = _FakeClient(HE)
        text, _ = self._work({}, client)
        self.assertEqual(text, graded_alert_text("suspicious", "gate", "גבר ליד השער.", "tries the gate", "he"))
        self.assertEqual(client.calls, [])

    def test_the_translator_writes_what_the_owner_reads(self) -> None:
        client = _FakeClient(HE)
        text, job = self._work({"owner_translation": "translator"}, client)
        self.assertEqual(text, graded_alert_text("suspicious", "gate", HE["summary"], HE["why"], "he"))
        self.assertEqual(len(client.calls), 1)
        # The clip's record keeps what the vision model said.
        self.assertEqual((job.alert["summary"], job.alert["why"]), (_Backend.PARSED["summary"], "tries the gate"))

    def test_a_failed_translation_still_sends_the_alert(self) -> None:
        text, _ = self._work({"owner_translation": "translator"}, _FakeClient("garbage"))
        self.assertEqual(text, graded_alert_text("suspicious", "gate", "גבר ליד השער.", "tries the gate", "he"))


if __name__ == "__main__":
    unittest.main()
