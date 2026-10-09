"""2026-10-09 ch1: both vision models timed out (3 tries x 30 s each, about 200 s) and the owner got a bare
"a person or vehicle was detected".

The fallback must answer when the main model times out, one clip must fit a time budget, the logs must say which
path ran, and when both models fail the owner is told honestly that the AI check did not finish (box/inference.py)."""
from __future__ import annotations

import json
import threading
import time
import unittest
from unittest import mock

import httpx
import openai

from home_guard_project.box import alert_texts
from home_guard_project.box import inference as inf
from home_guard_project.box.inference import AlertSettings, FallbackBackend, GptBackend

ANSWER = {"summary": "a man walks to the gate", "label": "normal", "people": 1,
          "vehicle_moving": False, "animals": 0, "why": "", "summary_owner": "a man walks to the gate"}


def _timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("the provider did not answer", request=request)


def http_backend(transport, model: str = "qwen/qwen3.5-9b", timeout: float = 25.0,
                 max_retries: int = 0) -> GptBackend:
    """A real GptBackend and OpenAI client on a fake HTTP transport (no network, no TLS setup)."""
    backend = GptBackend.__new__(GptBackend)
    backend._client = openai.OpenAI(api_key="k", base_url="https://openrouter.ai/api/v1", max_retries=max_retries,
                                    timeout=timeout, http_client=httpx.Client(transport=httpx.MockTransport(transport)))
    backend._timeout = timeout
    backend._model = backend.model_name = backend.last_model = model
    backend._extra_body = None
    backend.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    backend.last_prompt = ""
    backend._response_format = inf.VLM_RESPONSE_FORMAT
    return backend


class Answering:
    model_name = last_model = "qwen/qwen3-vl-8b-instruct"
    last_prompt = "p"
    last_frame_jpegs: list = []
    last_usage = {"prompt_tokens": 1, "completion_tokens": 1}

    def __init__(self, error: Exception = None) -> None:
        self.calls, self.error = 0, error

    def analyze(self, frames, camera_name, t_sec, start_hour, end_hour, **kw):
        self.calls += 1
        if self.error:
            raise self.error
        return json.dumps(ANSWER), dict(ANSWER)


class _Assistant:
    def __init__(self) -> None:
        self.sent: list = []

    def is_muted(self, camera: str) -> bool:
        return False

    def send_alert(self, alert, text, image=None, silent=False, lang="en"):
        self.sent.append({"alert": alert, "text": text, "image": image, "silent": silent, "lang": lang})
        return {"sent": True, "results": [{"chat_id": "1", "ok": True, "message_id": 7}]}


def run_worker(backend, lang: str = "en", labels=("person",), image: bytes = b"jpg"):
    assistant = _Assistant()
    job = inf.AlertJob(camera="front_door", stem="front_door_100_alert", ts=100.0, labels=list(labels))
    with mock.patch.object(inf, "frame_to_jpeg_bytes", return_value=image), \
            mock.patch.object(inf, "owner_language", return_value=lang):
        inf._worker(backend, {"alert_channel": "telegram"}, {}, AlertSettings(), "front_door", [object()],
                    assistant, job)
    return assistant, job


class PrimaryTimeoutTest(unittest.TestCase):
    def test_the_fallback_answers_when_the_primary_times_out(self) -> None:
        fallback = Answering()
        backend = FallbackBackend(http_backend(_timeout), fallback)
        with self.assertLogs("box.inference", "WARNING") as logs:
            raw, parsed = backend.analyze([], "ch1", 0, 0, 0, owner_language="he")
        self.assertEqual(parsed["summary"], ANSWER["summary"])
        self.assertEqual(fallback.calls, 1)
        self.assertTrue(any("VLM fallback to qwen/qwen3-vl-8b-instruct: APITimeoutError" in m for m in logs.output),
                        logs.output)
        self.assertEqual(backend.last_model, "qwen/qwen3-vl-8b-instruct")

    def test_the_worker_alerts_with_the_fallbacks_answer(self) -> None:
        fallback = Answering()
        assistant, job = run_worker(FallbackBackend(http_backend(_timeout), fallback))
        self.assertEqual(fallback.calls, 1)
        self.assertEqual(job.alert["summary"], ANSWER["summary"])
        self.assertNotIn("vlm_failed", job.alert)
        (sent,) = assistant.sent
        self.assertIn("a man walks to the gate", sent["text"])


class BothFailTest(unittest.TestCase):
    def test_both_failed_is_logged_once_with_both_reasons(self) -> None:
        backend = FallbackBackend(http_backend(_timeout), Answering(error=RuntimeError("Error code: 502")))
        with self.assertLogs("box.inference", "WARNING") as logs, self.assertRaises(inf.VlmUnavailable) as caught:
            backend.analyze([], "ch1", 0, 0, 0)
        both = [m for m in logs.output if "VLM: both models failed" in m]
        self.assertEqual(len(both), 1, logs.output)
        self.assertIn("qwen/qwen3.5-9b: APITimeoutError", both[0])
        self.assertIn("qwen/qwen3-vl-8b-instruct: RuntimeError: Error code: 502", both[0])
        self.assertEqual(len(caught.exception.reasons), 2)

    def test_a_junk_fallback_answer_is_logged_as_both_failed(self) -> None:
        junk = Answering()
        junk.analyze = lambda *a, **k: ("sorry", None)
        backend = FallbackBackend(http_backend(_timeout), junk)
        with self.assertLogs("box.inference", "WARNING") as logs:
            self.assertEqual(backend.analyze([], "ch1", 0, 0, 0), ("sorry", None))
        self.assertTrue(any("VLM: both models failed" in m for m in logs.output), logs.output)

    def test_the_owner_is_told_the_ai_check_did_not_finish_hebrew(self) -> None:
        backend = FallbackBackend(http_backend(_timeout), Answering(error=openai.APITimeoutError(
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"))))
        assistant, job = run_worker(backend, lang="he")
        (sent,) = assistant.sent
        self.assertEqual(sent["text"].splitlines()[1], "מה קורה: זוהה אדם. הבדיקה של ה-AI לא הספיקה, הנה התמונה.")
        self.assertTrue(sent["text"].startswith("⚪ "))
        self.assertNotIn("אדם או רכב", sent["text"])
        self.assertEqual(sent["image"], b"jpg")
        # The record keeps what happened: the detector's alert, no model answer.
        self.assertEqual(job.alert["alert_command"], "[send_message]")
        self.assertTrue(job.alert["vlm_failed"])

    def test_the_owner_is_told_the_ai_check_did_not_finish_english(self) -> None:
        backend = FallbackBackend(http_backend(_timeout), Answering(error=RuntimeError("down")))
        assistant, _ = run_worker(backend, lang="en", labels=("car", "person"))
        (sent,) = assistant.sent
        self.assertEqual(sent["text"].splitlines()[1],
                         "What's happening: A person and a vehicle were detected. The AI check did not finish "
                         "in time; "
                         "here is the picture.")

    def test_no_picture_no_promise_of_one(self) -> None:
        self.assertEqual(alert_texts.ai_unavailable({"people"}, "he", with_picture=False),
                         "זוהה אדם. הבדיקה של ה-AI לא הספיקה.")
        self.assertEqual(alert_texts.ai_unavailable(set(), "en"),
                         "Movement was detected. The AI check did not finish in time; here is the picture.")


class TimeBudgetTest(unittest.TestCase):
    def test_box_yaml_budget_and_defaults(self) -> None:
        s = AlertSettings.from_box_settings({})
        self.assertEqual((s.vlm_timeout_sec, s.vlm_max_retries), (25.0, 0))
        s = AlertSettings.from_box_settings({"vlm_timeout_sec": "20", "vlm_max_retries": 1})
        self.assertEqual((s.vlm_timeout_sec, s.vlm_max_retries), (20.0, 1))
        with self.assertLogs("box.inference", "WARNING"):
            s = AlertSettings.from_box_settings({"vlm_timeout_sec": 600, "vlm_max_retries": "lots"})
        self.assertEqual((s.vlm_timeout_sec, s.vlm_max_retries), (25.0, 0))

    def test_both_models_are_built_with_the_budget(self) -> None:
        settings = AlertSettings(vlm_provider="openrouter", vlm_model="qwen/qwen3.5-9b",
                                 vlm_fallback_model="qwen/qwen3-vl-8b-instruct")
        with mock.patch("ssl.create_default_context"), mock.patch("httpx.Client"), \
                mock.patch("openai.OpenAI") as client:
            backend = inf.make_backend(settings, {"OPENROUTER_API_KEY": "k"})
        self.assertIsInstance(backend, FallbackBackend)
        self.assertEqual(client.call_count, 2)
        for call in client.call_args_list:
            self.assertEqual((call.kwargs["timeout"], call.kwargs["max_retries"]), (25.0, 0))

    def test_sdk_retries_are_off_so_a_timeout_costs_one_try(self) -> None:
        tries = []

        def counting(request):
            tries.append(1)
            return _timeout(request)

        with self.assertRaises(openai.APITimeoutError):
            http_backend(counting, max_retries=0).analyze([], "ch1", 0, 0, 0)
        self.assertEqual(len(tries), 1)

    def test_a_trickling_answer_is_cut_at_the_wall_clock(self) -> None:
        release = threading.Event()

        def slow(request):          # bytes keep coming, so no HTTP read timeout would ever fire
            release.wait(5)
            return httpx.Response(200, json={})

        backend = http_backend(slow, timeout=0.2)
        with mock.patch.object(inf, "DEADLINE_GRACE_SEC", 0.1):
            started = time.monotonic()
            with self.assertRaises(inf.VlmDeadline):
                backend.analyze([], "ch1", 0, 0, 0)
            took = time.monotonic() - started
        release.set()
        self.assertLess(took, 2.0)

    def test_the_wall_clock_covers_every_sdk_try(self) -> None:
        seen = {}

        def fake_deadline(fn, seconds, what=""):
            seen["seconds"] = seconds
            return fn()

        backend = GptBackend.__new__(GptBackend)
        backend._client = mock.Mock(max_retries=1)
        backend._timeout, backend._model = 25.0, "m"
        with mock.patch.object(inf, "call_with_deadline", fake_deadline):
            backend._complete([], None)
        self.assertEqual(seen["seconds"], 25.0 * 2 + inf.DEADLINE_GRACE_SEC)

    def test_deadline_passes_answers_and_errors_through(self) -> None:
        self.assertEqual(inf.call_with_deadline(lambda: 7, 1.0), 7)
        with self.assertRaises(KeyError):
            inf.call_with_deadline(lambda: {}["x"], 1.0)


if __name__ == "__main__":
    unittest.main()
