"""eval_prompt.py with ``run --prompt eye``: the situational Eye, scored per category and per situation.

No network and no model: the manifest and frames are written here, the model is FakeBackend(eye=True)."""
from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime
from unittest import mock

import numpy as np

from home_guard_project.box import eval_prompt as ev
from home_guard_project.box import eye_prompt as eye

NIGHT_EPOCH = int(datetime(2026, 10, 15, 2, 30).timestamp())
DAY_EPOCH = int(datetime(2026, 10, 15, 14, 0).timestamp())


def manifest_row(clip_id, ours_label, camera="main_door", category=None, local_time=None):
    row = {"clip_id": clip_id, "batch": "ameer_house_batch_1", "camera": camera, "ours_label": ours_label,
           "ours_text": "A man knocks." if ours_label != "empty" else "No special activity",
           "frames": [f"frames/{clip_id}_{i}.jpg" for i in range(2)], "local_time": local_time}
    if category:
        row["category"] = category
    return row


class _Dir(unittest.TestCase):
    def setUp(self) -> None:
        import cv2

        self.out = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.out, True)
        self.rows = [
            manifest_row(f"main_door_{NIGHT_EPOCH}_trigger", "normal", category="N4"),   # visitor at 02:30
            manifest_row(f"main_door_{DAY_EPOCH}_trigger", "normal", category="N4"),     # visitor at 14:00
            manifest_row(f"front_side_{DAY_EPOCH + 5}_trigger", "empty", camera="front_side"),
            manifest_row("Burglary001", "alert", camera="external"),                     # no time
        ]
        os.makedirs(os.path.join(self.out, "frames"))
        for r in self.rows:
            for rel in r["frames"]:
                cv2.imwrite(os.path.join(self.out, rel), np.full((24, 32, 3), 80, dtype=np.uint8))
        with open(os.path.join(self.out, "manifest.jsonl"), "w", encoding="utf-8") as f:
            for r in self.rows:
                f.write(json.dumps(r) + "\n")


class SituationForClipTest(unittest.TestCase):
    def test_from_the_clip_epoch_with_the_default_schedule(self) -> None:
        sit = ev.eye_situation_for({"clip_id": f"main_door_{NIGHT_EPOCH}_trigger", "camera": "main_door"})
        self.assertEqual((sit.phase, sit.dark, sit.house_state, sit.camera_role, sit.intent),
                         ("late_night", True, "home_asleep", "entrance", "alert_triage"))
        day = ev.eye_situation_for({"clip_id": f"main_door_{DAY_EPOCH}_trigger", "camera": "main_door"})
        self.assertEqual((day.phase, day.house_state), ("day", "home_awake"))

    def test_from_local_time_or_unknown(self) -> None:
        sit = ev.eye_situation_for({"clip_id": "x", "camera": "yard", "local_time": "03:10:00"})
        self.assertEqual((sit.phase, sit.house_state, sit.time), ("late_night", "home_asleep", "03:10"))
        unknown = ev.eye_situation_for({"clip_id": "Burglary001", "camera": "external"})
        self.assertEqual((unknown.phase, unknown.house_state), ("day", "home_awake"))


class FakeEyeTest(unittest.TestCase):
    def test_answers_match_the_eye_schema(self) -> None:
        fake = ev.FakeBackend(eye=True)
        for ours in ("alert", "normal", "empty"):
            raw, parsed = fake.ask({"clip_id": "c_1771696897_x", "camera": "gate", "ours_label": ours}, [])
            self.assertEqual(json.loads(raw), parsed)
            self.assertEqual(set(parsed), set(eye.schema("alert_triage")["properties"]))
        _, parsed = fake.ask({"clip_id": "c", "camera": "gate", "ours_label": "normal", "category": "N3"}, [])
        self.assertEqual(parsed["category"], "N3")
        self.assertIn("SITUATION: time", fake.prompts[-1])


class RunEyeTest(_Dir):
    def test_run_fake_eye_end_to_end(self) -> None:
        summary = ev.run_eval(self.out, ev.FakeBackend(eye=True), eye=True)
        tag = ev.default_tag(None, "fake", fake=True, eye=True)
        self.assertNotEqual(tag, ev.default_tag(None, "fake", fake=True))
        rows = ev.read_jsonl(os.path.join(self.out, "results", f"{tag}.jsonl"))
        self.assertEqual(set(rows[0]), set(ev.ANSWER_COLUMNS) | set(ev.EYE_ANSWER_COLUMNS))
        self.assertEqual(rows[0]["prompt_id"], ev.EYE_PROMPT_ID)
        by_id = {r["clip_id"]: r for r in rows}
        night = by_id[f"main_door_{NIGHT_EPOCH}_trigger"]
        self.assertEqual((night["ai_category"], night["ai_raw_label"], night["ai_label"]), ("N4", "normal",
                                                                                             "suspicious"))
        self.assertEqual((night["sit_phase"], night["sit_house_state"]), ("late_night", "home_asleep"))
        self.assertEqual(by_id[f"main_door_{DAY_EPOCH}_trigger"]["ai_label"], "normal")

        # Each clip is judged by its own situation: the night visitor is not a false alarm.
        self.assertEqual(summary["false_alarms"], [f"main_door_{NIGHT_EPOCH}_trigger"])     # plain score
        self.assertEqual(summary["by_situation"]["night"]["normal_flagged"], 0)
        self.assertEqual(summary["by_situation"]["night"]["alerts_caught"], 1)
        self.assertEqual(summary["by_situation"]["day"]["normal_flagged"], 0)
        self.assertEqual(summary["situation_raised"], 1)
        self.assertEqual(summary["per_category"]["N4"], {"total": 2, "same_category": 2, "label_ok": 2})
        self.assertEqual(summary["ai_categories"]["N4"], 2)
        text = ev.format_summary(summary)
        self.assertIn("by situation", text)
        self.assertIn("N4", text)

        with open(os.path.join(self.out, "results", f"{tag}.csv"), encoding="utf-8", newline="") as f:
            table = list(csv.DictReader(f))
        self.assertEqual(list(table[0]), list(ev.RESULT_COLUMNS) + list(ev.EYE_RESULT_COLUMNS))
        self.assertEqual(ev.load_summary(self.out, tag)["by_situation"], summary["by_situation"])

        again = ev.FakeBackend(eye=True)
        ev.run_eval(self.out, again, eye=True)
        self.assertEqual(again.calls, 0)

    def test_a_box_prompt_results_file_is_not_reused_for_the_eye(self) -> None:
        ev.run_eval(self.out, ev.FakeBackend(), tag="t")
        with self.assertRaises(ev.ResultsConflict):
            ev.run_eval(self.out, ev.FakeBackend(eye=True), tag="t", eye=True)

    def test_the_eye_wording_has_its_own_hash(self) -> None:
        self.assertNotEqual(ev.prompt_sha12_of(None), ev.eye_sha12())
        self.assertEqual(ev.eye_sha12(), ev.eye_sha12())

    def test_main_run_fake_eye_and_summary(self) -> None:
        self.assertEqual(ev.main(["run", "--dir", self.out, "--fake", "--prompt", "eye", "--tag", "cli"]), 0)
        with open(os.path.join(self.out, "results", "cli.jsonl"), encoding="utf-8") as f:
            self.assertEqual(json.loads(f.readline())["prompt_id"], ev.EYE_PROMPT_ID)
        self.assertEqual(ev.main(["summary", "--dir", self.out, "--tag", "cli"]), 0)

    def test_main_refuses_eye_with_a_prompt_file(self) -> None:
        path = os.path.join(self.out, "p.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("x")
        self.assertEqual(ev.main(["run", "--dir", self.out, "--fake", "--prompt", "eye", "--prompt-file", path]), 1)

    def test_real_backend_is_asked_with_the_clip_situation(self) -> None:
        backend = mock.Mock()
        backend.model_name = "ollama:qwen3-vl:4b-instruct-bf16"
        backend.last_usage = {"prompt_tokens": 10, "completion_tokens": 5}
        backend.analyze.return_value = ("{}", {})
        asker = ev.EyeAsker(backend)
        asker.ask(self.rows[0], [])
        sit = backend.analyze.call_args.kwargs["situation"]
        self.assertEqual((sit.phase, sit.house_state), ("late_night", "home_asleep"))
        with mock.patch.object(ev.inference, "build_gpt", return_value=backend):
            self.assertIsInstance(ev._make_gpt("ollama", "qwen3-vl:4b-instruct-bf16", eye=True), ev.EyeAsker)


class SummarizeEyeTest(unittest.TestCase):
    def row(self, cid, ours, ai, raw, cat, phase="day", house="home_awake", ours_cat=None, expected=""):
        return {"clip_id": cid, "ours_label": ours, "ai_label": ai, "ai_summary": "x", "ours_text": "y", "error": "",
                "ai_category": cat, "ai_raw_label": raw, "sit_phase": phase, "sit_house_state": house,
                "sit_dark": phase == "late_night", "sit_camera_role": "entrance", "ours_category": ours_cat or "",
                "ours_expected_label": expected}

    def test_counts_per_situation_and_category(self) -> None:
        rows = [
            self.row("a", "normal", "suspicious", "normal", "N4", "late_night", "home_asleep", "N4", "suspicious"),
            self.row("b", "normal", "suspicious", "suspicious", "S2", ours_cat="N1", expected="normal"),
            self.row("c", "alert", "escalation", "escalation", "E1", ours_cat="E1", expected="escalation"),
            self.row("d", "normal", "suspicious", "normal", "N5", "day", "away"),       # no truth category
        ]
        s = ev.summarize(rows)
        self.assertEqual(s["by_situation"]["night"], {"alerts_caught": 1, "alerts_total": 1, "normal_flagged": 0,
                                                      "normal_total": 0, "errors": 0})
        self.assertEqual(s["by_situation"]["day"]["normal_flagged"], 1)
        self.assertEqual(s["by_situation"]["away"]["normal_flagged"], 1)
        self.assertEqual(s["per_category"]["N1"], {"total": 1, "same_category": 0, "label_ok": 0})
        self.assertEqual(s["confusion"]["N1"], {"S2": 1})
        self.assertEqual(s["no_truth_category"], 1)
        self.assertEqual(s["situation_raised"], 2)

    def test_box_prompt_rows_get_no_eye_fields(self) -> None:
        s = ev.summarize([{"clip_id": "a", "ours_label": "normal", "ai_label": "normal", "ai_summary": "x",
                           "ours_text": "y", "error": ""}])
        self.assertNotIn("by_situation", s)
        self.assertNotIn("by situation", ev.format_summary(s))


if __name__ == "__main__":
    unittest.main()
