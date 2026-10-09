"""Case memory in the guard loop (box/inference.py, case_memory/INTEGRATION.md section 1). No network: a fake memory."""
from __future__ import annotations

import json
import unittest
from datetime import datetime
from unittest import mock

from home_guard_project.box import case_memory as cm
from home_guard_project.box import house_state as hs
from home_guard_project.box import inference as inf
from home_guard_project.box.case_memory import CaseNote, texts

from cm_helpers import memory as real_memory

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


def case_note(kind="softened"):
    return CaseNote(kind, "C1", "רגיל (לפי ההסבר שלך)", "Normal (per your explanation from 5.10): the neighbour.",
                    buttons=(texts.button("not_them", "C1"),), band="high", score=0.91, would_level="quiet")


class FakeMemory:
    """Stands in for CaseMemory: records the calls and answers what the test set."""

    def __init__(self, level="alert", note=None, error=None):
        self.level, self.note, self.error = level, note, error
        self.calls = []

    def apply(self, event, decision):
        self.calls.append((event, dict(decision)))
        if self.error is not None:
            raise self.error
        return self.level, self.note


class StartupTest(unittest.TestCase):
    def tearDown(self) -> None:
        cm.configure(None)

    def test_on_is_the_default(self) -> None:
        self.assertTrue(inf.case_memory_on({}))
        for value in (True, "on", " ON ", "yes", "true"):
            self.assertTrue(inf.case_memory_on({"case_memory": value}), value)
        with self.assertLogs("box.inference", level="WARNING"):
            self.assertTrue(inf.case_memory_on({"case_memory": "maybe"}))

    def test_off_setting(self) -> None:
        for value in (False, "off", "Off", "false", "no", "0"):
            self.assertFalse(inf.case_memory_on({"case_memory": value}), value)

    def test_start_installs_the_default_memory(self) -> None:
        fake = FakeMemory()
        with mock.patch.object(cm, "make_default", return_value=fake) as made:
            self.assertTrue(inf.start_case_memory({}, {"OPENAI_API_KEY": "k"}))
        made.assert_called_once_with(env={"OPENAI_API_KEY": "k"})
        self.assertIs(cm.current(), fake)

    def test_off_leaves_memory_off(self) -> None:
        cm.configure(FakeMemory())
        with mock.patch.object(cm, "make_default") as made:
            self.assertFalse(inf.start_case_memory({"case_memory": False}, {}))
        made.assert_not_called()
        self.assertIsNone(cm.current())

    def test_a_failed_start_leaves_memory_off(self) -> None:
        with mock.patch.object(cm, "make_default", side_effect=OSError("disk")), \
                self.assertLogs("box.inference", level="WARNING") as logs:
            self.assertFalse(inf.start_case_memory({}, {}))
        self.assertIsNone(cm.current())
        self.assertIn("Case memory not started", "\n".join(logs.output))


class WorkerTest(unittest.TestCase):
    def setUp(self) -> None:
        inf._SOFTENED_DAYS.clear()

    def tearDown(self) -> None:
        cm.configure(None)

    def run_worker(self, parsed, memory=None, ts=DAY, muted=(False, False), settings=None):
        cm.configure(memory)
        backend = mock.Mock()
        backend.model_name = "fake-eye"
        backend.last_model = ""
        backend.last_prompt = "the eye prompt"
        backend.analyze.return_value = (json.dumps(parsed), parsed)
        assistant = mock.Mock()
        assistant.is_muted.side_effect = list(muted)
        assistant.send_alert.return_value = {"sent": True}
        job = inf.AlertJob(camera=CAM, stem=f"{CAM}_{int(ts)}_alert", ts=ts, labels=["person"],
                           input_meta={"vlm_input": "crop"})
        with mock.patch.object(inf, "FACTS_PROVIDER", mock.Mock(return_value=[])), \
                mock.patch.object(inf, "owner_language", return_value="en"), \
                mock.patch.object(inf, "_jpegs", return_value=[]), \
                mock.patch.object(hs, "current", side_effect=lambda now, **kw: hs.scheduled(now)):
            inf._worker(backend, {"alert_channel": "telegram"}, {}, settings or inf.AlertSettings(eye_prompt="situational"), CAM, [],
                        assistant, job)
        self.assertTrue(job.ready.is_set())
        return job, assistant

    def sent(self, assistant):
        assistant.send_alert.assert_called_once()
        call = assistant.send_alert.call_args
        return call.args[1], call.kwargs.get("silent")

    def test_memory_sees_the_processed_answer_and_the_decision(self) -> None:
        fake = FakeMemory()
        job, _ = self.run_worker(answer(), fake)
        event, decision = fake.calls[0]
        self.assertEqual(event.event_id, job.stem)
        self.assertEqual(event.signature.camera, CAM)
        self.assertEqual(event.signature.category, "N4")
        self.assertEqual(decision, {"final_label": "normal", "alert_command": "[send_message]",
                                    "serious_behaviour": False})

    def test_alert_without_a_note_is_todays_path(self) -> None:
        job, assistant = self.run_worker(answer(), FakeMemory())
        text, silent = self.sent(assistant)
        self.assertTrue(silent)   # a normal scene was already silent
        self.assertEqual(job.alert["delivery_level"], "alert")
        self.assertIsNone(job.alert["case_memory"])
        self.assertEqual(job.alert["case_buttons"], [])
        self.assertEqual(job.alert["case_signature"]["camera"], CAM)

    def test_alert_with_a_note_adds_one_line_and_records_the_buttons(self) -> None:
        note = case_note("context")
        job, assistant = self.run_worker(answer(), FakeMemory("alert", note), ts=NIGHT)
        text, silent = self.sent(assistant)
        self.assertEqual(job.alert["label"], "suspicious")
        self.assertFalse(silent)
        self.assertTrue(text.endswith("\n" + note.text_en))
        self.assertEqual(job.alert["delivery_level"], "alert")
        self.assertEqual(job.alert["case_memory"], note.record())
        self.assertEqual(job.alert["case_buttons"], [texts.button("not_them", "C1")])
        self.assertNotIn("buttons", assistant.send_alert.call_args.args[0])   # not sent until the callbacks exist

    def test_quiet_is_sent_silently_with_the_note(self) -> None:
        note = case_note()
        job, assistant = self.run_worker(answer(), FakeMemory("quiet", note), ts=NIGHT)
        text, silent = self.sent(assistant)
        self.assertTrue(silent)
        self.assertIn(note.text_en, text)
        self.assertEqual(job.alert["delivery_level"], "quiet")
        self.assertTrue(job.alert["silent"])

    def test_digest_is_sent_quietly_until_the_digest_exists(self) -> None:
        note = case_note()
        with self.assertLogs("box.inference", level="INFO") as logs:
            job, assistant = self.run_worker(answer(), FakeMemory("digest", note), ts=NIGHT)
        text, silent = self.sent(assistant)   # never dropped
        self.assertTrue(silent)
        self.assertIn(note.text_en, text)
        self.assertEqual(job.alert["delivery_level"], "digest")
        self.assertIn("digest, which the box doesn't have yet", "\n".join(logs.output))

    def test_an_exception_in_memory_still_alerts(self) -> None:
        job, assistant = self.run_worker(answer(), FakeMemory(error=RuntimeError("boom")), ts=NIGHT)
        text, silent = self.sent(assistant)
        self.assertFalse(silent)
        self.assertEqual(job.alert["delivery_level"], "alert")
        self.assertIsNone(job.alert["case_memory"])
        self.assertIsNone(job.alert["case_signature"])

    def test_a_failure_building_the_event_still_alerts(self) -> None:
        fake = FakeMemory("quiet", case_note())
        with mock.patch.object(cm.CaseEvent, "build", side_effect=ValueError("bad")):
            job, assistant = self.run_worker(answer(), fake, ts=NIGHT)
        _, silent = self.sent(assistant)
        self.assertFalse(silent)
        self.assertEqual(fake.calls, [])
        self.assertEqual(job.alert["delivery_level"], "alert")

    def test_escalation_never_reaches_memory(self) -> None:
        fake = FakeMemory("digest", case_note())
        job, assistant = self.run_worker(answer(category="E1", raw_label="escalation", label="escalation",
                                                serious_behaviour=True), fake)
        self.assertEqual(job.alert["label"], "escalation")
        self.assertEqual(job.alert["alert_command"], "[call_owner]")
        self.assertEqual(fake.calls, [])
        _, silent = self.sent(assistant)
        self.assertFalse(silent)
        self.assertEqual(job.alert["delivery_level"], "alert")
        self.assertIsNone(job.alert["case_memory"])

    def test_the_guard_refuses_a_call_even_if_memory_would_not(self) -> None:
        fake = FakeMemory("quiet", case_note())
        cm.configure(fake)
        self.assertEqual(inf._case_memory(None, CAM, DAY, answer(), "suspicious", "[call_owner]",
                                          {"serious_behaviour": False}, None, ""), ("alert", None, None))
        self.assertEqual(fake.calls, [])

    def test_off_means_todays_path(self) -> None:
        job, assistant = self.run_worker(answer(), None, ts=NIGHT)
        _, silent = self.sent(assistant)
        self.assertFalse(silent)
        self.assertEqual(job.alert["delivery_level"], "alert")
        self.assertIsNone(job.alert["case_signature"])

    def test_muted_paths_do_not_call_memory(self) -> None:
        fake = FakeMemory("quiet", case_note())
        job, assistant = self.run_worker(answer(), fake, muted=(True,))
        self.assertTrue(job.paused)
        fake2 = FakeMemory("quiet", case_note())
        job2, assistant2 = self.run_worker(answer(), fake2, muted=(False, True))   # muted while the AI looked
        self.assertEqual((fake.calls, fake2.calls), ([], []))
        assistant.send_alert.assert_not_called()
        assistant2.send_alert.assert_not_called()
        self.assertTrue(job2.alert["muted"])

    def test_out_of_window_does_not_call_memory(self) -> None:
        fake = FakeMemory("quiet", case_note())
        job, assistant = self.run_worker(answer(), fake, ts=NIGHT,
                                         settings=inf.AlertSettings(alert_start_hour=8, alert_end_hour=20,
                                                                 eye_prompt="situational"))
        self.assertEqual(fake.calls, [])
        assistant.send_alert.assert_not_called()

    def test_false_positive_does_not_call_memory(self) -> None:
        fake = FakeMemory("quiet", case_note())
        job, assistant = self.run_worker(answer(category="N10", people=0, movement="none"), fake)
        self.assertTrue(job.false_positive)
        self.assertEqual(fake.calls, [])
        assistant.send_alert.assert_not_called()

    def test_the_real_memory_without_cases_alerts_and_records_the_signature(self) -> None:
        job, assistant = self.run_worker(answer(), real_memory(), ts=NIGHT)
        _, silent = self.sent(assistant)
        self.assertFalse(silent)
        self.assertEqual(job.alert["delivery_level"], "alert")
        self.assertIsNone(job.alert["case_memory"])
        sig = job.alert["case_signature"]
        self.assertEqual((sig["camera"], sig["category"], sig["eye_model"]), (CAM, "N4", "fake-eye"))


class LearnFromTheOwnerTest(unittest.TestCase):
    """2026-10-09: precedents of what the owner explained learn in shadow: no line under the alert, and a "would
    quiet" for alerts his week-long memory already kept quiet (case_memory/link.py)."""

    setUp, tearDown, run_worker, sent = WorkerTest.setUp, WorkerTest.tearDown, WorkerTest.run_worker, WorkerTest.sent

    def test_a_shadow_note_adds_no_line_but_is_recorded(self) -> None:
        note = CaseNote("shadow", "C1", "זיהיתי: החשמלאים. בפעם הבאה לא אתריע על זה, בסדר?",
                        "I recognised: the electricians. Next time I won't alert for this, OK?",
                        buttons=(texts.button("confirm", "C1"),), band="high", score=1.0, would_level="quiet")
        job, assistant = self.run_worker(answer(), FakeMemory("alert", note), ts=NIGHT)
        text, _ = self.sent(assistant)
        self.assertNotIn("Next time", text)
        self.assertEqual(job.alert["case_memory"], note.record())

    def test_kept_quiet_by_the_owner_is_learned_without_a_judge(self) -> None:
        seen = []

        class Shadow:
            def apply(self, event, decision, shadow_only=False):
                seen.append((event.signature.actions, dict(decision), shadow_only))
                return "quiet", None

        cm.configure(Shadow())
        decision = {"serious_behaviour": True, "activity_look": {"lowered": True}}
        level, note, sig = inf._case_memory(None, CAM, DAY, answer(), "suspicious", "[send_message]", decision,
                                            mock.Mock(model_name="m", last_model=""), "v",
                                            "A man lies on the stairs.", shadow_only=True)
        self.assertEqual((level, note), ("alert", None), "shadow only never changes the delivery")
        ((actions, facts, shadow_only),) = seen
        self.assertEqual((actions, shadow_only, facts["context_lowered"]), (("lying",), True, True))
        self.assertEqual(sig["actions"], ["lying"])


if __name__ == "__main__":
    unittest.main()
