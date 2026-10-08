"""Tracker facts in the Eye's prompt: the ``eye_tracker_facts`` switch, both prompts, the version suffix and the
backends (box/inference.py, box/eye_prompt.py, box/situation.py)."""
from __future__ import annotations

import hashlib
import json
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest import mock

from home_guard_project.box import eye_prompt as eye
from home_guard_project.box import house_state as hs
from home_guard_project.box import inference as inf
from home_guard_project.box import tracker as tr
from home_guard_project.box.situation import build_situation

CAM = "front_door"
NIGHT = datetime(2026, 10, 15, 2, 14).timestamp()
LINE = "TRACKER FACTS (from code): person 1 in view 38s, 22s in 'entrance'; came back 2 times in 10 min."
FACT = {"id": "F12", "camera": CAM, "kind": "people", "effect": "lower", "hours": None,
        "text": "my son comes home late", "area": ""}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def situation(**kw):
    return build_situation(CAM, NIGHT, house=hs.scheduled(NIGHT), **kw)


class SwitchTest(unittest.TestCase):
    def test_off_by_default(self) -> None:
        self.assertFalse(inf.AlertSettings().eye_tracker_facts)
        self.assertFalse(inf.AlertSettings.from_box_settings({}).eye_tracker_facts)

    def test_box_yaml_values(self) -> None:
        for value, want in ((True, True), (False, False), ("on", True), (" ON ", True), ("off", False),
                            ("yes", True), ("0", False), (None, False)):
            self.assertEqual(inf.AlertSettings.from_box_settings({"eye_tracker_facts": value}).eye_tracker_facts,
                             want, value)
        with self.assertLogs("box.inference", level="WARNING"):
            self.assertFalse(inf.AlertSettings.from_box_settings({"eye_tracker_facts": "maybe"}).eye_tracker_facts)


class OffIsByteIdenticalTest(unittest.TestCase):
    """Hashes of the prompts without the tracker line (the 2026-10-08 rules, actions not appearance): off must not
    change a byte."""

    def test_legacy_prompt(self) -> None:
        self.assertEqual(sha(inf.build_prompt(CAM, int(NIGHT), "02:14:00", 0, 0)),
                         "651da7ab13100ceaa5d17c4a70efa66c33463de21fd2e3886e92e26db800b589")
        self.assertEqual(sha(inf.build_prompt(CAM, int(NIGHT), "02:14:00", 0, 0, owner_language="he", facts=[FACT],
                                              alert_ts=NIGHT)),
                         "0bf05d13d9f5dd1f72bfb9e1e8ebdcbe691c75a9a407668c22549781c2714c94")
        self.assertEqual(inf.build_prompt(CAM, int(NIGHT), "02:14:00", 0, 0, tracker_facts=""),
                         inf.build_prompt(CAM, int(NIGHT), "02:14:00", 0, 0))

    def test_eye_prompt(self) -> None:
        self.assertEqual(sha(eye.build_prompt(situation())),
                         "8702962736d85368ae7dcb349fd60a508a24c300c58e38e24d76d3bb4a47c6d0")
        self.assertEqual(sha(eye.build_prompt(situation(), facts=[FACT])),
                         "9f710e8cb6d70d5bc9c1f681bbee45f65aca4a5173e952aab57d60eab0a963a5")
        self.assertNotIn("tracker_facts", situation().record())
        self.assertEqual(eye.records(None, situation())["prompt_version"], eye.EYE_PROMPT_VERSION)


class PromptTest(unittest.TestCase):
    def test_legacy_prompt_ends_with_the_line_and_its_rule(self) -> None:
        prompt = inf.build_prompt(CAM, int(NIGHT), "02:14:00", 0, 0, facts=[FACT], alert_ts=NIGHT, tracker_facts=LINE)
        self.assertTrue(prompt.endswith("\n\n" + LINE + "\n" + tr.TRACKER_FACTS_RULE), prompt[-300:])
        self.assertIn("House notes from the owner", prompt)
        self.assertIn("count people from the frames, not from here", tr.TRACKER_FACTS_RULE)

    def test_eye_prompt_puts_it_under_the_situation_and_zone_facts(self) -> None:
        zone = SimpleNamespace(line="ZONE FACTS (from code): person 1 is in 'road' (public)", ground="public",
                               zone="street", crossed_in=False)
        sit = situation(scene_facts=zone, tracker_facts=LINE)
        prompt = eye.build_prompt(sit)
        block = sit.header() + "\n" + zone.line + "\n" + eye.ZONE_FACTS_RULE + "\n" + LINE + "\n" + tr.TRACKER_FACTS_RULE
        self.assertIn(block, prompt)
        self.assertEqual(sit.record()["tracker_facts"], LINE)
        self.assertEqual(eye.records(None, sit)["prompt_version"], eye.EYE_PROMPT_VERSION + "+tf1")

    def test_eye_prompt_without_zone_facts(self) -> None:
        sit = situation(tracker_facts=LINE)
        self.assertIn(sit.header() + "\n" + LINE + "\n" + tr.TRACKER_FACTS_RULE, eye.build_prompt(sit))


def gpt_backend(answer):
    obj = inf.GptBackend.__new__(inf.GptBackend)
    obj._model = obj.model_name = "gpt-local-test"
    obj._response_format = inf.VLM_RESPONSE_FORMAT
    obj._complete = mock.Mock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)))], model="gpt-local-test"))
    return obj


ANSWER = {"summary": "A man waits at the door.", "label": "normal", "raw_label": "normal", "applied_fact_id": "",
          "serious_behaviour": False, "people": 1, "vehicle_moving": False, "animals": 0, "why": "",
          "summary_owner": ""}


class BackendTest(unittest.TestCase):
    def test_gpt_backend_puts_the_line_in_the_legacy_prompt(self) -> None:
        backend = gpt_backend(ANSWER)
        backend.analyze([], CAM, int(NIGHT), 0, 0, tracker_facts=LINE)
        self.assertIn(LINE, backend.last_prompt)
        backend.analyze([], CAM, int(NIGHT), 0, 0)
        self.assertNotIn("TRACKER FACTS", backend.last_prompt)

    def test_fallback_and_null_backends_take_it(self) -> None:
        primary = mock.Mock()
        primary.analyze.return_value = ("{}", {"summary": ""})
        inf.FallbackBackend(primary, mock.Mock()).analyze([], CAM, 1, 0, 0, tracker_facts=LINE)
        self.assertEqual(primary.analyze.call_args.kwargs["tracker_facts"], LINE)
        inf.NullBackend().analyze([], CAM, 1, 0, 0, tracker_facts=LINE)

    def test_takes_tracker_facts(self) -> None:
        class Old:
            def analyze(self, frames, camera_name, t_sec, start_hour, end_hour, owner_language="en"):
                return "", None

        class Open:
            def analyze(self, *a, **kw):
                return "", None

        self.assertFalse(inf._takes_kwarg(Old(), "tracker_facts"))
        self.assertTrue(inf._takes_kwarg(Open(), "tracker_facts"))
        self.assertTrue(inf._takes_kwarg(inf.NullBackend(), "tracker_facts"))
        self.assertTrue(inf._takes_kwarg(gpt_backend(ANSWER), "tracker_facts"))


class WorkerTest(unittest.TestCase):
    def setUp(self) -> None:
        inf._SOFTENED_DAYS.clear()

    def run_worker(self, on: bool, mode: str = "legacy", line: str = LINE, backend=None):
        backend = backend or mock.Mock()
        if isinstance(backend, mock.Mock):
            backend.model_name = "fake"
            backend.last_prompt = "prompt"
            backend.analyze.return_value = (json.dumps(ANSWER), dict(ANSWER))
        assistant = mock.Mock()
        assistant.is_muted.return_value = False
        assistant.send_alert.return_value = {"sent": True}
        job = inf.AlertJob(camera=CAM, stem=f"{CAM}_{int(NIGHT)}_alert", ts=NIGHT, labels=["person"],
                           input_meta={"vlm_input": "crop"})
        job.tracker_line = line
        job.tracker = {"version": "tf1", "line": line}
        settings = inf.AlertSettings(eye_prompt=mode, eye_tracker_facts=on)
        with mock.patch.object(inf, "FACTS_PROVIDER", mock.Mock(return_value=[])), \
                mock.patch.object(inf, "owner_language", return_value="en"), \
                mock.patch.object(inf, "_jpegs", return_value=[]), \
                mock.patch.object(hs, "current", side_effect=lambda now, **kw: hs.scheduled(now)):
            inf._worker(backend, {"alert_channel": "telegram"}, {}, settings, CAM, [], assistant, job)
        return job, backend

    def test_off_sends_nothing_new_but_keeps_the_record(self) -> None:
        job, backend = self.run_worker(on=False)
        self.assertNotIn("tracker_facts", backend.analyze.call_args.kwargs)
        self.assertEqual(job.teacher["prompt_version"], inf.PROMPT_VERSION)
        self.assertEqual(job.teacher["tracker"], {"version": "tf1", "line": LINE})

    def test_on_legacy_passes_the_line_and_suffixes_the_version(self) -> None:
        job, backend = self.run_worker(on=True)
        self.assertEqual(backend.analyze.call_args.kwargs["tracker_facts"], LINE)
        self.assertEqual(job.teacher["prompt_version"], inf.PROMPT_VERSION + "+tf1")

    def test_on_without_a_line_changes_nothing(self) -> None:
        job, backend = self.run_worker(on=True, line="")
        self.assertNotIn("tracker_facts", backend.analyze.call_args.kwargs)
        self.assertEqual(job.teacher["prompt_version"], inf.PROMPT_VERSION)

    def test_on_situational_puts_it_in_the_situation(self) -> None:
        eye_answer = {"summary": "A man waits.", "category": "N4", "other_text": "", "zone": "entrance",
                      "movement": "standing", "flags": [], "people": 1, "vehicle_moving": False, "animals": 0,
                      "visibility": "clear", "evidence_frame": 1, "raw_label": "normal", "label": "normal",
                      "applied_fact_id": "", "serious_behaviour": False, "why": ""}
        backend = mock.Mock()
        backend.model_name = "fake"
        backend.last_prompt = "prompt"
        backend.analyze.return_value = (json.dumps(eye_answer), dict(eye_answer))
        job, backend = self.run_worker(on=True, mode="situational", backend=backend)
        kwargs = backend.analyze.call_args.kwargs
        self.assertNotIn("tracker_facts", kwargs)
        self.assertEqual(kwargs["situation"].tracker_facts, LINE)
        self.assertEqual(job.teacher["prompt_version"], eye.EYE_PROMPT_VERSION + "+tf1")
        self.assertEqual(job.input_meta["situation"]["tracker_facts"], LINE)

    def test_a_backend_that_cannot_take_it_keeps_todays_prompt(self) -> None:
        class Old:
            model_name = "old"
            last_prompt = "prompt"

            def __init__(self):
                self.kwargs = None

            def analyze(self, frames, camera_name, t_sec, start_hour, end_hour, owner_language="en"):
                self.kwargs = {"owner_language": owner_language}
                return json.dumps(ANSWER), dict(ANSWER)

        old = Old()
        job, _ = self.run_worker(on=True, backend=old)
        self.assertEqual(old.kwargs, {"owner_language": "en"})
        self.assertEqual(job.teacher["prompt_version"], inf.PROMPT_VERSION)


if __name__ == "__main__":
    unittest.main()
