from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from home_guard_project.box import inference as inf
from home_guard_project.box.inference import AlertSettings, NullBackend


class WindowTest(unittest.TestCase):
    def test_normal_window(self) -> None:
        self.assertTrue(inf.in_alert_window(10, 9, 17))
        self.assertFalse(inf.in_alert_window(8, 9, 17))
        self.assertFalse(inf.in_alert_window(17, 9, 17))  # end exclusive

    def test_wraps_midnight(self) -> None:
        self.assertTrue(inf.in_alert_window(23, 22, 6))
        self.assertTrue(inf.in_alert_window(3, 22, 6))
        self.assertFalse(inf.in_alert_window(12, 22, 6))

    def test_equal_means_always(self) -> None:
        self.assertTrue(inf.in_alert_window(0, 0, 0))
        self.assertTrue(inf.in_alert_window(13, 5, 5))


class ParseTest(unittest.TestCase):
    def test_plain_json(self) -> None:
        self.assertEqual(inf.parse_vlm_json('{"a": 1}'), {"a": 1})

    def test_json_with_surrounding_text(self) -> None:
        self.assertEqual(inf.parse_vlm_json('sure: {"a": 1} done'), {"a": 1})

    def test_garbage(self) -> None:
        self.assertIsNone(inf.parse_vlm_json("not json"))
        self.assertIsNone(inf.parse_vlm_json(""))


class PolicyOverrideTest(unittest.TestCase):
    def test_outside_window_forces_none(self) -> None:
        out = inf.apply_policy_override({"alert_command": "[call_owner]"}, in_window=False, person=True, vehicle=False)
        self.assertEqual(out["alert_command"], "[none]")

    def test_inside_person_none_upgrades(self) -> None:
        out = inf.apply_policy_override({"alert_command": "[none]"}, in_window=True, person=True, vehicle=False)
        self.assertEqual(out["alert_command"], "[send_message]")
        self.assertTrue(out["alert_reason"])

    def test_inside_call_owner_unchanged(self) -> None:
        out = inf.apply_policy_override({"alert_command": "[call_owner]"}, in_window=True, person=True, vehicle=False)
        self.assertEqual(out["alert_command"], "[call_owner]")

    def test_inside_no_trigger_stays_none(self) -> None:
        out = inf.apply_policy_override({"alert_command": "[none]"}, in_window=True, person=False, vehicle=False)
        self.assertEqual(out["alert_command"], "[none]")


class SettingsTest(unittest.TestCase):
    def test_defaults(self) -> None:
        s = AlertSettings.from_box_settings({})
        self.assertEqual(s.alert_channel, "telegram")
        self.assertEqual(s.vlm_backend, "gpt")
        self.assertEqual(s.model, "yolo11s.pt")

    def test_custom(self) -> None:
        s = AlertSettings.from_box_settings({
            "alert_start_hour": 22, "alert_end_hour": 6, "alert_cooldown_sec": 300,
            "alert_channel": "both", "notify_dry_run": True,
        })
        self.assertEqual((s.alert_start_hour, s.alert_end_hour), (22, 6))
        self.assertEqual(s.cooldown_sec, 300.0)
        self.assertEqual(s.alert_channel, "both")
        self.assertTrue(s.dry_run)


class BackendSelectionTest(unittest.TestCase):
    def test_dry_run_is_null(self) -> None:
        s = AlertSettings(dry_run=True)
        self.assertIsInstance(inf.make_backend(s, {"OPENAI_API_KEY": "k"}), NullBackend)

    def test_gpt_without_key_is_null(self) -> None:
        s = AlertSettings(vlm_backend="gpt", dry_run=False)
        self.assertIsInstance(inf.make_backend(s, {}), NullBackend)

    def test_unknown_backend_is_null(self) -> None:
        s = AlertSettings(vlm_backend="llava", dry_run=False)
        self.assertIsInstance(inf.make_backend(s, {}), NullBackend)

    def test_null_backend_returns_summary_field(self) -> None:
        raw, parsed = NullBackend().analyze([], "cam", 0, 0, 0)
        self.assertIn("summary", parsed)


class _FakeBox:
    def __init__(self, cls_id: int) -> None:
        self.cls = [cls_id]


class _FakeResult:
    def __init__(self, cls_ids, names) -> None:
        self.boxes = [_FakeBox(c) for c in cls_ids]
        self.names = names


class DetectTriggerTest(unittest.TestCase):
    NAMES = {0: "person", 2: "car", 15: "cat"}

    def test_person_and_car(self) -> None:
        person, vehicle, labels = inf.detect_trigger(_FakeResult([0, 2], self.NAMES))
        self.assertTrue(person)
        self.assertTrue(vehicle)
        self.assertEqual(labels, ["car", "person"])

    def test_only_cat_no_trigger(self) -> None:
        person, vehicle, labels = inf.detect_trigger(_FakeResult([15], self.NAMES))
        self.assertFalse(person)
        self.assertFalse(vehicle)
        self.assertEqual(labels, [])

    def test_empty(self) -> None:
        self.assertEqual(inf.detect_trigger(_FakeResult([], self.NAMES)), (False, False, []))


class DispatchTest(unittest.TestCase):
    def test_telegram_channel_routes_to_telegram(self) -> None:
        from home_guard_project.box import telegram_notify
        box_settings = {"alert_channel": "telegram", "telegram_chat_ids": "1"}
        with mock.patch.object(telegram_notify, "notify", return_value={"telegram": "ok"}) as n:
            res = inf.dispatch_alert(box_settings, {"TELEGRAM_BOT_TOKEN": "T"}, "[send_message]", "sum", "why")
        n.assert_called_once()
        self.assertEqual(res["channel"], "telegram")
        self.assertEqual(res["telegram"], {"telegram": "ok"})

    def test_dispatch_never_raises(self) -> None:
        from home_guard_project.box import telegram_notify
        with mock.patch.object(telegram_notify, "notify", side_effect=RuntimeError("boom")):
            res = inf.dispatch_alert({"alert_channel": "telegram"}, {}, "[send_message]", "s", "r")
        self.assertIn("error", res)


if __name__ == "__main__":
    unittest.main()


class _FakeAssistant:
    def __init__(self, muted: bool = False) -> None:
        self.muted = muted
        self.sent: list = []

    def is_muted(self, camera: str) -> bool:
        return self.muted

    def send_alert(self, alert, text, image=None):
        self.sent.append({"alert": alert, "text": text, "image": image})
        return {"sent": True, "results": [{"chat_id": "1", "ok": True, "message_id": 7}]}


class _DescribingBackend:
    def analyze(self, frames, camera_name, t_sec, start_hour, end_hour):
        return '{"summary": "a person at the door"}', {"summary": "a person at the door"}


class WorkerWithAssistantTest(unittest.TestCase):
    SETTINGS = AlertSettings()          # 0,0 = always inside the alert window
    BOX = {"alert_channel": "telegram"}

    def _run(self, assistant) -> inf.AlertJob:
        job = inf.AlertJob(camera="front_door", stem="front_door_100_alert", ts=100.0, labels=["person"])
        with mock.patch.object(inf, "frame_to_jpeg_bytes", return_value=b"jpg"):
            inf._worker(_DescribingBackend(), self.BOX, {}, self.SETTINGS, "front_door", [object()], assistant, job)
        return job

    def test_the_alert_goes_out_through_the_assistant_and_is_filed_under_the_clip(self) -> None:
        assistant = _FakeAssistant()
        job = self._run(assistant)

        (sent,) = assistant.sent
        self.assertEqual(sent["alert"], {"alert_id": "front_door_100_alert", "camera": "front_door",
                                         "summary": "a person at the door", "ts": 100.0})
        self.assertIn("front_door: a person at the door", sent["text"])
        self.assertEqual(sent["image"], b"jpg")
        self.assertTrue(job.ready.is_set())
        self.assertEqual((job.alert["summary"], job.alert["alert_command"], job.alert["muted"]),
                         ("a person at the door", "[send_message]", False))
        self.assertEqual(job.alert["labels"], ["person"])

    def test_a_paused_camera_is_described_and_saved_but_not_sent(self) -> None:
        assistant = _FakeAssistant(muted=True)
        job = self._run(assistant)

        self.assertEqual(assistant.sent, [])
        self.assertTrue(job.alert["muted"])
        self.assertEqual(job.alert["summary"], "a person at the door")
        self.assertFalse(job.alert["dispatch"]["sent"])

    def test_the_job_is_released_even_when_the_backend_fails(self) -> None:
        class Broken:
            def analyze(self, *args):
                raise RuntimeError("boom")

        job = inf.AlertJob(camera="front_door", stem="s", ts=1.0)
        inf._worker(Broken(), self.BOX, {}, self.SETTINGS, "front_door", [], _FakeAssistant(), job)
        self.assertTrue(job.ready.is_set())
        self.assertEqual(job.alert, {})

    def test_without_an_assistant_the_old_path_is_used(self) -> None:
        with mock.patch("home_guard_project.box.telegram_notify.notify", return_value={"sent": True}) as notify, \
                mock.patch.object(inf, "frame_to_jpeg_bytes", return_value=b"jpg"):
            inf._worker(_DescribingBackend(), self.BOX, {"TELEGRAM_BOT_TOKEN": "T"}, self.SETTINGS,
                        "front_door", [object()])
        notify.assert_called_once()


class VlmFilterTest(unittest.TestCase):
    def test_what_the_vlm_saw(self) -> None:
        self.assertTrue(inf.vlm_confirms({"summary": "a person walks", "people": 1, "vehicle_moving": False}))
        self.assertTrue(inf.vlm_confirms({"summary": "a car arrives", "people": 0, "vehicle_moving": True}))
        self.assertTrue(inf.vlm_confirms({"summary": "two people", "people": "2"}))
        self.assertFalse(inf.vlm_confirms({"summary": "a parked car", "people": 0, "vehicle_moving": False}))
        self.assertFalse(inf.vlm_confirms({"summary": "nothing", "people": 0}))

    def test_no_verdict_when_the_vlm_did_not_say(self) -> None:
        self.assertIsNone(inf.vlm_confirms(None))
        self.assertIsNone(inf.vlm_confirms({}))
        self.assertIsNone(inf.vlm_confirms({"summary": "something"}))           # an older or other backend
        self.assertIsNone(inf.vlm_confirms({"summary": "x", "people": "several"}))

    def test_the_prompt_asks_for_the_facts_the_decision_needs(self) -> None:
        prompt = inf.build_prompt("front_door", 0, "12:00:00", 22, 6)
        self.assertIn('"people"', prompt)
        self.assertIn('"vehicle_moving"', prompt)


class _ParkedCarBackend:
    def analyze(self, frames, camera_name, t_sec, start_hour, end_hour):
        parsed = {"summary": "a car is parked in the driveway", "people": 0, "vehicle_moving": False}
        return "{}", parsed


class FalsePositiveTest(unittest.TestCase):
    def test_nothing_is_sent_and_the_job_is_marked_for_training(self) -> None:
        assistant = _FakeAssistant()
        job = inf.AlertJob(camera="yard", stem="yard_100_alert", ts=100.0, labels=["car"])
        with mock.patch.object(inf, "dispatch_alert") as dispatch:
            inf._worker(_ParkedCarBackend(), {"alert_channel": "telegram"}, {}, AlertSettings(), "yard",
                        [object()], assistant, job)
        dispatch.assert_not_called()
        self.assertEqual(assistant.sent, [])
        self.assertTrue(job.false_positive)
        self.assertTrue(job.ready.is_set())
        self.assertEqual(job.alert["alert_command"], "[none]")
        self.assertEqual(job.alert["vlm"], {"people": 0, "vehicle_moving": False})

    def test_a_false_positive_clip_goes_to_the_training_folder_not_the_production_one(self) -> None:
        import tempfile

        import numpy as np

        from home_guard_project.box.alert_clips import encode_frame

        with tempfile.TemporaryDirectory() as tmp:
            production, training = os.path.join(tmp, "production_multi"), os.path.join(tmp, "dataset_multi")
            frames = [(100.0 + i * 0.2, encode_frame(np.zeros((48, 64, 3), dtype=np.uint8))) for i in range(5)]
            job = inf.AlertJob(camera="yard", stem="yard_100_alert", ts=100.0, labels=["car"], false_positive=True,
                               alert={"summary": "a parked car", "alert_command": "[none]", "labels": ["car"]})
            job.ready.set()
            with mock.patch("home_guard_project.box.alert_clips._to_h264", return_value=False):
                inf._save_clip(job, frames, production, training)

            self.assertFalse(os.path.exists(production))
            metas = [os.path.join(d, n) for d, _, names in os.walk(training) for n in names if n.endswith(".meta.json")]
            self.assertEqual([os.path.basename(m) for m in metas], ["yard_100_fp.meta.json"])
            with open(metas[0], encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["kind"], "false_positive")
            self.assertEqual(meta["yolo"]["class_counts"], {"car": 1})


class PreviewAdapterTest(unittest.TestCase):
    def test_the_preview_stream_accepts_what_inference_passes_to_its_reader(self) -> None:
        from home_guard_project.box.inference_preview import adapt_stream

        class Reader:
            def __init__(self, name, url, ring=None):
                self.name, self.url, self.ring = name, url, ring

        class Writer:
            enabled = True

            def set_cameras(self, names):
                self.names = list(names)

        writer = Writer()
        stream = adapt_stream(Reader, writer)("front_door", "rtsp://x", ring="RING")
        self.assertEqual((stream.name, stream.ring), ("front_door", "RING"))
        self.assertEqual(writer.names, ["front_door"])
