"""2026-10-09: four "both models failed" in under 1.5 h, every one on a big crop (1279-2161 px a side x 16 frames).
When no model answered and the frames are bigger than 768 px, the main model is asked once more with the same
frames shrunk to 768 px (INTER_AREA, aspect kept) within 20 s; an answered call is never touched
(box/inference.py vlm_rescue)."""
from __future__ import annotations

import base64
import json
import unittest
from unittest import mock

import cv2
import httpx
import numpy as np

from home_guard_project.box import inference as inf
from home_guard_project.box.inference import AlertSettings, FallbackBackend
from home_guard_project.data_collection import model_input as mi

from test_vlm_fallback_timeout import ANSWER, _Assistant, http_backend

RESCUED = dict(ANSWER, summary="two workers carry a beam on the pergola")


def ok_response(request: httpx.Request, answer=ANSWER) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": json.loads(request.content)["model"],
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": json.dumps(answer)}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})


def timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("the provider did not answer", request=request)


class Model:
    """A fake provider for one model: answers each request with the next handler, and keeps the requests."""

    def __init__(self, *handlers) -> None:
        self.handlers, self.requests = list(handlers), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        handler = self.handlers[min(len(self.requests) - 1, len(self.handlers) - 1)]
        return handler(request)


def image_sizes(body) -> list:
    out = []
    for part in body["messages"][0]["content"]:
        if part["type"] == "image_url":
            data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            out.append((img.shape[1], img.shape[0]))
    return out


def frames(w=1600, h=1000, n=3):
    rng = np.random.default_rng(7)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def run(primary: Model, fallback: Model, sent_frames, lang="en"):
    backend = FallbackBackend(http_backend(primary, model="qwen/qwen3.5-9b"),
                              http_backend(fallback, model="qwen/qwen3-vl-8b-instruct"))
    assistant = _Assistant()
    job = inf.AlertJob(camera="ameer_week_0_1_ch1", stem="ch1_100_alert", ts=100.0, labels=["person"])
    job.model_input = mi.render_model_input(sent_frames, {"fps": 1.0}, mi.ModelInputConfig(1.0))
    with mock.patch.object(inf, "frame_to_jpeg_bytes", side_effect=mi.encode_jpeg), \
            mock.patch.object(inf, "owner_language", return_value=lang), \
            mock.patch.object(inf, "EVENTS", None), \
            mock.patch.object(inf, "FACTS_PROVIDER", mock.Mock(return_value=[])), \
            mock.patch.object(inf, "TRACKERS", None), \
            mock.patch.object(inf, "_case_memory", return_value=("alert", None, None)), \
            mock.patch.object(inf, "baseline_look", return_value={}), \
            mock.patch.object(inf, "_alert_ground", return_value={}), \
            mock.patch.object(inf.messenger, "uses_translator", return_value=False):
        with mock.patch.object(inf.log, "warning", wraps=inf.log.warning) as warned:
            inf._worker(backend, {"alert_channel": "telegram"}, {}, AlertSettings(), job.camera, job.model_input.frames,
                        assistant, job)
    lines = [c.args[0] % c.args[1:] for c in warned.call_args_list]
    return assistant, job, lines


class AnsweredCallUntouchedTest(unittest.TestCase):
    def test_an_answered_call_is_one_request_with_the_native_pictures(self) -> None:
        primary, fallback = Model(ok_response), Model(ok_response)
        sent = frames()
        assistant, job, lines = run(primary, fallback, sent)
        self.assertEqual(len(primary.requests), 1)
        self.assertEqual(fallback.requests, [])
        body = primary.requests[0]
        # The exact bytes the box sent before the rescue existed: model_input.encode_jpeg of each frame.
        images = [p["image_url"]["url"] for p in body["messages"][0]["content"] if p["type"] == "image_url"]
        self.assertEqual(images, ["data:image/jpeg;base64," + base64.b64encode(mi.encode_jpeg(f)).decode()
                                  for f in sent])
        self.assertEqual(sorted(body), ["messages", "model", "reasoning", "response_format", "temperature"]
                         if "reasoning" in body else ["messages", "model", "response_format", "temperature"])
        self.assertNotIn("vlm_rescued", job.alert)
        self.assertNotIn("vlm_failed", job.alert)
        self.assertEqual(job.rescue, {})
        self.assertNotIn("rescue", inf._model_input_record(job))
        self.assertFalse(any("rescue" in line for line in lines), lines)

    def test_the_fallbacks_answer_is_not_rescued_either(self) -> None:
        primary, fallback = Model(timeout), Model(ok_response)
        _, job, _ = run(primary, fallback, frames())
        self.assertEqual((len(primary.requests), len(fallback.requests)), (1, 1))
        self.assertNotIn("vlm_rescued", job.alert)


class RescueTest(unittest.TestCase):
    def test_both_failed_then_the_main_model_answers_at_768(self) -> None:
        primary = Model(timeout, lambda r: ok_response(r, RESCUED))
        fallback = Model(timeout)
        assistant, job, lines = run(primary, fallback, frames(1600, 1000))
        self.assertEqual((len(primary.requests), len(fallback.requests)), (2, 1))
        self.assertEqual(image_sizes(primary.requests[0]), [(1600, 1000)] * 3)
        self.assertEqual(image_sizes(primary.requests[1]), [(768, 480)] * 3)          # aspect kept
        self.assertEqual(primary.requests[1]["messages"][0]["content"][0]["text"],
                         primary.requests[0]["messages"][0]["content"][0]["text"])  # the same prompt
        self.assertTrue(job.alert["vlm_rescued"])
        self.assertNotIn("vlm_failed", job.alert)
        self.assertEqual(job.alert["summary"], RESCUED["summary"])
        (sent,) = assistant.sent
        self.assertIn(RESCUED["summary"], sent["text"])
        self.assertNotIn("did not finish", sent["text"])
        record = inf._model_input_record(job)["rescue"]
        self.assertEqual((record["max_side"], record["answered"], record["size"]), (768, True, [768, 480]))
        self.assertIn("both", record["reason"]) if "both" in record["reason"] else self.assertTrue(record["reason"])
        self.assertTrue(any("VLM rescue at 768 px: answered" in line for line in lines), lines)
        # The teacher record is the rescue's: the model that answered and the 768 px pictures it saw.
        self.assertEqual(job.teacher["model"], "qwen/qwen3.5-9b")
        self.assertEqual(job.teacher["model_input"]["rescue"]["answered"], True)
        self.assertEqual({cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR).shape[1]
                          for j in job.teacher["frames"]}, {768})

    def test_a_failed_rescue_leaves_the_honest_alert(self) -> None:
        primary, fallback = Model(timeout), Model(timeout)
        assistant, job, lines = run(primary, fallback, frames(1300, 1300), lang="he")
        self.assertEqual((len(primary.requests), len(fallback.requests)), (2, 1))
        self.assertTrue(job.alert["vlm_failed"])
        self.assertNotIn("vlm_rescued", job.alert)
        (sent,) = assistant.sent
        self.assertIn("הבדיקה של ה-AI לא הספיקה", sent["text"])
        record = inf._model_input_record(job)["rescue"]
        self.assertFalse(record["answered"])
        self.assertIn("APITimeoutError", record["error"])
        self.assertTrue(any("VLM rescue at 768 px: failed" in line for line in lines), lines)

    def test_small_pictures_are_not_asked_again(self) -> None:
        primary, fallback = Model(timeout), Model(timeout)
        _, job, lines = run(primary, fallback, frames(768, 600))
        self.assertEqual((len(primary.requests), len(fallback.requests)), (1, 1))
        self.assertTrue(job.alert["vlm_failed"])
        self.assertEqual(job.rescue, {})
        self.assertFalse(any("rescue" in line for line in lines), lines)

    def test_the_rescue_has_its_own_20_s_budget(self) -> None:
        seen = []
        real = inf.call_with_deadline

        def spy(fn, seconds, what=""):
            seen.append(seconds)
            return real(fn, seconds, what)

        primary = Model(timeout, lambda r: ok_response(r, RESCUED))
        with mock.patch.object(inf, "call_with_deadline", side_effect=spy):
            run(primary, Model(timeout), frames())
        self.assertEqual(seen, [25.0 + inf.DEADLINE_GRACE_SEC, 25.0 + inf.DEADLINE_GRACE_SEC,
                                inf.VLM_RESCUE_TIMEOUT_SEC + inf.DEADLINE_GRACE_SEC])

    def test_the_clip_writer_waits_for_the_slowest_worker(self) -> None:
        both = 2 * (inf.VLM_TIMEOUT_SEC + inf.DEADLINE_GRACE_SEC)
        rescue = inf.VLM_RESCUE_TIMEOUT_SEC + inf.DEADLINE_GRACE_SEC
        self.assertGreaterEqual(inf.CLIP_WAIT_SEC, both + rescue + inf.INVESTIGATOR_MAX_WAIT_SEC + 15.0 + 20.0)


class ShrinkTest(unittest.TestCase):
    def test_long_side_768_aspect_kept_small_untouched(self) -> None:
        tall, small = np.zeros((1520, 2161, 3), np.uint8), np.zeros((500, 700, 3), np.uint8)
        out = inf.shrink_to_max_side([tall, small], 768)
        self.assertEqual(out[0].shape[:2], (540, 768))
        self.assertIs(out[1], small)


if __name__ == "__main__":
    unittest.main()
