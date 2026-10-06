from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from home_guard_project.box import inference as inf
from home_guard_project.box.alert_clips import encode_frame, write_alert_clip
from home_guard_project.box.inference import PROMPT_VERSION, VLM_RESPONSE_FORMAT, VLM_SCHEMA, AlertSettings

NOW = 1_800_000_000.0
ANSWER = {"summary": "A person is walking to the door.", "label": "normal", "people": 1, "vehicle_moving": False}


class _TeachingBackend:
    """Stands in for GPT: records the question, answers in the agreed JSON."""

    model_name = "gpt-4o"
    last_prompt = ""

    def analyze(self, frames, camera_name, t_sec, start_hour, end_hour, owner_language="en"):
        self.last_prompt = inf.build_prompt(camera_name, t_sec, "01:00:00", start_hour, end_hour,
                                            owner_language=owner_language)
        return json.dumps(ANSWER), dict(ANSWER)


def _metas(root):
    return sorted(os.path.join(d, n) for d, _, names in os.walk(root) for n in names if n.endswith(".meta.json"))


class TeacherRecordTest(unittest.TestCase):
    """Every answer of the big model is kept with exactly what it saw, so a small model can learn it."""

    def test_the_answer_must_be_json_with_a_summary_and_one_of_three_labels(self) -> None:
        self.assertEqual(VLM_RESPONSE_FORMAT["type"], "json_schema")
        self.assertTrue(VLM_RESPONSE_FORMAT["json_schema"]["strict"])
        self.assertEqual(set(VLM_SCHEMA["required"]),
                         {"summary", "label", "raw_label", "applied_fact_id", "serious_behaviour",
                          "people", "vehicle_moving", "animals", "why", "summary_owner"})
        self.assertEqual(VLM_SCHEMA["properties"]["label"]["enum"], ["normal", "suspicious", "escalation"])
        self.assertFalse(VLM_SCHEMA["additionalProperties"])
        prompt = inf.build_prompt("door", 0, "01:00:00", 0, 0)
        for word in ("normal", "suspicious", "escalation", '"label"'):
            self.assertIn(word, prompt)
        self.assertEqual(inf.LABEL_COMMANDS, {"normal": "[send_message]", "suspicious": "[send_message]",
                                              "escalation": "[call_owner]"})
        self.assertEqual(inf.label_of({"label": "Escalation"}), "escalation")
        self.assertEqual(inf.label_of({"label": "weird"}), "normal")
        self.assertEqual(inf.label_of(None), "normal")

    def test_the_worker_keeps_the_pictures_the_question_and_the_answer(self) -> None:
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(3)]
        job = inf.AlertJob(camera="door", stem="door_100_alert", ts=NOW, labels=["person"])
        with mock.patch.object(inf, "dispatch_alert", return_value={"telegram": {"telegram": {"sent": True}}}):
            inf._worker(_TeachingBackend(), {"alert_channel": "telegram"}, {}, AlertSettings(), "door", frames, None, job)
        self.assertEqual(job.teacher["model"], "gpt-4o")
        self.assertEqual(job.teacher["prompt_version"], PROMPT_VERSION)
        self.assertIn('"summary"', job.teacher["prompt"])
        self.assertEqual(len(job.teacher["frames"]), 3)
        self.assertTrue(all(data.startswith(b"\xff\xd8") for data in job.teacher["frames"]))   # real JPEGs
        self.assertEqual(json.loads(job.teacher["raw"]), ANSWER)
        self.assertEqual(job.teacher["parsed"], ANSWER)

    def test_the_record_names_the_model_that_answered_not_the_alias(self) -> None:
        backend = _TeachingBackend()
        backend.model_name, backend.last_model = "eye", "gpt-4o-2024-08-06"   # asked the gateway for "eye"
        job = inf.AlertJob(camera="door", stem="door_100_alert", ts=NOW, labels=["person"])
        with mock.patch.object(inf, "dispatch_alert", return_value={"telegram": {"telegram": {"sent": True}}}):
            inf._worker(backend, {"alert_channel": "telegram"}, {}, AlertSettings(), "door",
                        [np.zeros((48, 64, 3), dtype=np.uint8)], None, job)
        self.assertEqual(job.teacher["model"], "gpt-4o-2024-08-06")

    def test_the_record_travels_with_the_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frames = [(NOW + i * 0.2, encode_frame(np.zeros((48, 64, 3), dtype=np.uint8))) for i in range(5)]
            teacher = {"model": "gpt-4o", "prompt_version": PROMPT_VERSION, "prompt": "describe",
                       "frames": [b"\xff\xd8one", b"\xff\xd8two"], "raw": json.dumps(ANSWER), "parsed": ANSWER}
            with mock.patch("home_guard_project.box.alert_clips._to_h264", return_value=False):
                meta_path = write_alert_clip(tmp, "door", "door_100_alert", frames,
                                             {"summary": ANSWER["summary"], "alert_command": "[send_message]",
                                              "labels": ["person"]}, kind="alert", teacher=teacher)
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["model_response"], ANSWER)
            self.assertEqual(meta["teacher"]["model"], "gpt-4o")
            self.assertEqual(meta["teacher"]["prompt"], "describe")
            inputs = [os.path.join(tmp, *p.split("\\")) for p in meta["teacher"]["input_frames"]]
            self.assertEqual([os.path.basename(p) for p in inputs], ["door_100_alert_f0.jpg", "door_100_alert_f1.jpg"])
            with open(inputs[1], "rb") as f:
                self.assertEqual(f.read(), b"\xff\xd8two")
            with open(os.path.join(tmp, *meta["teacher"]["raw_path"].split("\\")), encoding="utf-8") as f:
                self.assertEqual(json.loads(f.read()), ANSWER)

            # The outbox takes the pictures and the raw answer along with the clip.
            from home_guard_project.box.outbox import move_clip
            outbox = os.path.join(tmp, "outbox")
            moved = move_clip(meta_path, tmp, outbox)
            self.assertEqual(moved, 5)      # clip, two pictures, raw answer, meta
            self.assertTrue(os.path.isfile(os.path.join(outbox, "vlm_crops", "door",
                                                        os.path.basename(os.path.dirname(meta_path)),
                                                        "door_100_alert_f0.jpg")))

    def test_an_alert_is_saved_for_the_owner_and_for_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            production, training = os.path.join(tmp, "production_multi"), os.path.join(tmp, "dataset_multi")
            frames = [(NOW + i * 0.2, encode_frame(np.zeros((48, 64, 3), dtype=np.uint8))) for i in range(5)]
            job = inf.AlertJob(camera="door", stem="door_100_alert", ts=NOW, labels=["person"],
                               alert={"summary": ANSWER["summary"], "alert_command": "[send_message]",
                                      "labels": ["person"], "dispatch": {"sent": True}},
                               teacher={"model": "gpt-4o", "prompt_version": PROMPT_VERSION, "prompt": "q",
                                        "frames": [b"\xff\xd8one"], "raw": json.dumps(ANSWER), "parsed": ANSWER})
            job.ready.set()
            with mock.patch("home_guard_project.box.alert_clips._to_h264", return_value=False):
                inf._save_clip(job, frames, production, training)
            self.assertEqual([os.path.basename(m) for m in _metas(production)], ["door_100_alert.meta.json"])
            self.assertEqual([os.path.basename(m) for m in _metas(training)], ["door_100_alert.meta.json"])
            with open(_metas(training)[0], encoding="utf-8") as f:
                kept = json.load(f)
            self.assertEqual((kept["kind"], kept["model_response"]), ("alert", ANSWER))


if __name__ == "__main__":
    unittest.main()
