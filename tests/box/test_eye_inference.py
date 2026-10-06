"""The guard loop with ``eye_prompt: situational`` (box/inference.py + eye_prompt.py + situation.py)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest import mock

import numpy as np

from home_guard_project.box import eye_prompt as eye
from home_guard_project.box import house_state as hs
from home_guard_project.box import inference as inf
from home_guard_project.box.alert_clips import encode_frame

CAM = "front_door"
NIGHT = datetime(2026, 10, 15, 2, 14).timestamp()
DAY = datetime(2026, 10, 15, 14, 5).timestamp()


def answer(**over):
    base = {"summary": "A man walks to the door and knocks.", "category": "N4", "other_text": "",
            "zone": "entrance", "movement": "approaching", "flags": [], "people": 1, "vehicle_moving": False,
            "animals": 0, "visibility": "clear", "evidence_frame": 2, "raw_label": "normal", "label": "normal",
            "applied_fact_id": "", "serious_behaviour": False, "why": ""}
    base.update(over)
    return base


def note(**over):
    base = {"id": "F12", "camera": CAM, "kind": "people", "effect": "lower", "hours": None,
            "text": "my son comes home late", "area": ""}
    base.update(over)
    return base


class SettingTest(unittest.TestCase):
    def test_legacy_is_the_default(self) -> None:
        self.assertEqual(inf.AlertSettings().eye_prompt, "legacy")
        self.assertEqual(inf.AlertSettings.from_box_settings({}).eye_prompt, "legacy")

    def test_box_yaml_value(self) -> None:
        self.assertEqual(inf.AlertSettings.from_box_settings({"eye_prompt": "situational"}).eye_prompt,
                         "situational")
        self.assertEqual(inf.AlertSettings.from_box_settings({"eye_prompt": " Situational "}).eye_prompt,
                         "situational")
        with self.assertLogs("box.inference", level="WARNING"):
            self.assertEqual(inf.AlertSettings.from_box_settings({"eye_prompt": "smart"}).eye_prompt, "legacy")


class WorkerTest(unittest.TestCase):
    def setUp(self) -> None:
        inf._SOFTENED_DAYS.clear()

    def run_worker(self, parsed, ts=NIGHT, facts=(), mode="situational", box=None):
        backend = mock.Mock()
        backend.model_name = "fake"
        backend.last_prompt = "the eye prompt"
        backend.analyze.return_value = (json.dumps(parsed), parsed)
        assistant = mock.Mock()
        assistant.is_muted.return_value = False
        assistant.send_alert.return_value = {"sent": True}
        job = inf.AlertJob(camera=CAM, stem=f"{CAM}_{int(ts)}_alert", ts=ts, labels=["person"],
                           input_meta={"vlm_input": "crop"})
        settings = inf.AlertSettings(eye_prompt=mode)
        with mock.patch.object(inf, "FACTS_PROVIDER", mock.Mock(return_value=list(facts))), \
                mock.patch.object(inf, "owner_language", return_value="en"), \
                mock.patch.object(inf, "_jpegs", return_value=[]), \
                mock.patch.object(hs, "current", side_effect=lambda now, **kw: hs.scheduled(now)):
            inf._worker(backend, dict({"alert_channel": "telegram"}, **(box or {})), {}, settings, CAM, [],
                        assistant, job)
        self.assertTrue(job.ready.is_set())
        return job, assistant, backend

    def test_a_visitor_at_two_at_night_is_sent_as_suspicious_with_the_situation(self) -> None:
        job, assistant, backend = self.run_worker(answer())
        situation = backend.analyze.call_args.kwargs["situation"]
        self.assertEqual((situation.phase, situation.house_state, situation.intent, situation.camera_role),
                         ("late_night", "home_asleep", "alert_triage", "entrance"))
        self.assertEqual(job.alert["label"], "suspicious")
        self.assertEqual(job.alert["raw_label"], "normal")
        self.assertIn("visitor at the door", job.alert["why"])
        self.assertEqual(assistant.send_alert.call_args.kwargs.get("silent"), False)
        self.assertEqual(job.teacher["prompt_version"], eye.EYE_PROMPT_VERSION)
        self.assertEqual(job.teacher["parsed"], dict(answer(), label="normal"))   # the Eye's own words
        for record in (job.teacher, job.input_meta):
            self.assertEqual(record["situation"], {"phase": "late_night", "dark": True,
                                                   "house_state": "home_asleep", "intent": "alert_triage",
                                                   "camera_role": "entrance"})
            self.assertEqual(record["observation"]["category"], "N4")
            self.assertTrue(record["judgement"]["open_case"])
        self.assertEqual(job.input_meta["prompt_version"], eye.EYE_PROMPT_VERSION)
        self.assertEqual(job.input_meta["vlm_input"], "crop")

    def test_the_same_visitor_by_day_is_normal(self) -> None:
        job, assistant, _ = self.run_worker(answer(), ts=DAY)
        self.assertEqual(job.alert["label"], "normal")
        self.assertEqual(job.input_meta["judgement"]["expectation"], "expected")

    def test_the_camera_role_comes_from_box_yaml(self) -> None:
        _, _, backend = self.run_worker(answer(), box={"camera_roles": {CAM: "private"}})
        self.assertEqual(backend.analyze.call_args.kwargs["situation"].camera_role, "private")

    def test_a_note_cannot_soften_escalation_or_a_serious_category(self) -> None:
        job, _, _ = self.run_worker(answer(category="E1", raw_label="escalation", label="normal",
                                           applied_fact_id="F12"), facts=[note()])
        self.assertEqual(job.alert["label"], "escalation")
        self.assertFalse(job.alert["softened"])
        job, _, _ = self.run_worker(answer(category="S1", raw_label="suspicious", label="normal",
                                           applied_fact_id="F12", serious_behaviour=False), facts=[note()])
        self.assertEqual(job.alert["label"], "suspicious")
        self.assertFalse(job.alert["softened"])

    def test_a_note_still_softens_an_unusual_hour(self) -> None:
        job, _, backend = self.run_worker(answer(label="normal", applied_fact_id="F12"), facts=[note()])
        self.assertEqual(job.alert["label"], "normal")
        self.assertTrue(job.alert["softened"])
        self.assertEqual(backend.analyze.call_args.kwargs["facts"], [note()])

    def test_nothing_seen_is_a_false_positive_with_the_records(self) -> None:
        job, assistant, _ = self.run_worker(answer(category="N10", people=0, movement="none",
                                                   summary="No special activity."))
        self.assertTrue(job.false_positive)
        assistant.send_alert.assert_not_called()
        self.assertEqual(job.input_meta["observation"]["category"], "N10")

    def test_no_answer_still_alerts_and_keeps_the_situation(self) -> None:
        backend_answer = None
        job, assistant, _ = self.run_worker(backend_answer)
        assistant.send_alert.assert_called_once()
        self.assertEqual(job.input_meta["situation"]["phase"], "late_night")
        self.assertNotIn("observation", job.input_meta)

    def test_a_failed_call_still_alerts_with_the_situation(self) -> None:
        # An outage, or the gateway's daily cap: the detector's alert goes out in situational mode too.
        backend = mock.Mock()
        backend.model_name = "fake"
        backend.analyze.side_effect = RuntimeError("Error code: 402 - box_daily_cap")
        assistant = mock.Mock()
        assistant.is_muted.return_value = False
        assistant.send_alert.return_value = {"sent": True}
        job = inf.AlertJob(camera=CAM, stem=f"{CAM}_{int(NIGHT)}_alert", ts=NIGHT, labels=["person"])
        with mock.patch.object(inf, "FACTS_PROVIDER", mock.Mock(return_value=[])), \
                mock.patch.object(inf, "owner_language", return_value="en"), \
                mock.patch.object(hs, "current", side_effect=lambda now, **kw: hs.scheduled(now)):
            inf._worker(backend, {"alert_channel": "telegram"}, {}, inf.AlertSettings(eye_prompt="situational"),
                        CAM, [], assistant, job)
        self.assertTrue(job.ready.is_set())
        assistant.send_alert.assert_called_once()
        self.assertEqual(job.alert["summary"], "a person or vehicle was detected")
        self.assertEqual(job.input_meta["situation"]["phase"], "late_night")

    def test_legacy_passes_no_situation(self) -> None:
        job, _, backend = self.run_worker(answer(), mode="legacy")
        self.assertNotIn("situation", backend.analyze.call_args.kwargs)
        self.assertEqual(job.teacher["prompt_version"], inf.PROMPT_VERSION)
        self.assertNotIn("situation", job.input_meta)
        self.assertNotIn("situation", job.teacher)


class BackendTest(unittest.TestCase):
    def backend(self, content):
        backend = object.__new__(inf.GptBackend)
        backend._response_format = inf.VLM_RESPONSE_FORMAT
        backend._model = "m"
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=None)
        return backend, response

    def test_situational_call_sends_the_eye_prompt_and_schema(self) -> None:
        from home_guard_project.box import situation as st

        sit = st.build_situation(CAM, NIGHT, house=hs.scheduled(NIGHT))
        backend, response = self.backend(json.dumps(answer()))
        with mock.patch.object(backend, "_complete", return_value=response) as complete:
            raw, parsed = backend.analyze([], CAM, 0, 0, 0, situation=sit)
        self.assertEqual(parsed, answer())
        self.assertEqual(backend.last_prompt, eye.build_prompt(sit))
        self.assertEqual(complete.call_args.args[1], eye.response_format("alert_triage"))

    def test_legacy_call_is_unchanged(self) -> None:
        backend, response = self.backend("{}")
        with mock.patch.object(backend, "_complete", return_value=response) as complete:
            backend.analyze([], CAM, 0, 0, 0, alert_ts=DAY)
        self.assertIs(complete.call_args.args[1], inf.VLM_RESPONSE_FORMAT)
        self.assertEqual(backend.last_prompt, inf.build_prompt(CAM, 0, "14:05:00", 0, 0, alert_ts=DAY))

    def test_a_model_without_schemas_gets_a_json_object_for_the_eye_too(self) -> None:
        from home_guard_project.box import situation as st

        sit = st.build_situation(CAM, NIGHT, house=hs.scheduled(NIGHT))
        backend, response = self.backend("{}")
        calls = []

        def complete(content, fmt):
            calls.append(fmt)
            if fmt["type"] == "json_schema":
                raise RuntimeError("response_format json_schema is not supported")
            return response

        with mock.patch.object(backend, "_complete", side_effect=complete):
            backend.analyze([], CAM, 0, 0, 0, situation=sit)
            backend.analyze([], CAM, 0, 0, 0, situation=sit)
        self.assertEqual([c["type"] for c in calls], ["json_schema", "json_object", "json_object"])

    def test_null_backend_takes_a_situation(self) -> None:
        self.assertEqual(inf.NullBackend().analyze([], CAM, 0, 0, 0, situation=object())[1], {"summary": ""})


class MetaTest(unittest.TestCase):
    def test_the_records_reach_both_metas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            production, training = os.path.join(tmp, "production_multi"), os.path.join(tmp, "dataset_multi")
            frames = [(NIGHT + i * 0.2, encode_frame(np.zeros((48, 64, 3), dtype=np.uint8))) for i in range(5)]
            extra = {"vlm_input": "crop", "situation": {"phase": "late_night"}, "observation": {"category": "N4"},
                     "judgement": {"open_case": True}, "prompt_version": eye.EYE_PROMPT_VERSION}
            job = inf.AlertJob(camera=CAM, stem=f"{CAM}_1_alert", ts=NIGHT, labels=["person"],
                               alert={"summary": "s", "alert_command": "[send_message]", "labels": ["person"],
                                      "dispatch": {"sent": True}},
                               teacher={"model": "m", "prompt_version": eye.EYE_PROMPT_VERSION, "prompt": "q",
                                        "frames": [b"\xff\xd8one"], "raw": "{}", "parsed": {}},
                               input_meta=extra)
            job.ready.set()
            with mock.patch("home_guard_project.box.alert_clips._to_h264", return_value=False):
                inf._save_clip(job, frames, production, training)
            for root in (production, training):
                path = next(os.path.join(d, n) for d, _, names in os.walk(root) for n in names
                            if n.endswith(".meta.json"))
                with open(path, encoding="utf-8") as f:
                    meta = json.load(f)
                self.assertEqual(meta["situation"], {"phase": "late_night"})
                self.assertEqual(meta["observation"], {"category": "N4"})
                self.assertEqual(meta["prompt_version"], eye.EYE_PROMPT_VERSION)


if __name__ == "__main__":
    unittest.main()
