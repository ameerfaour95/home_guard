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
    NAMES = {0: "person", 2: "car", 14: "bird", 15: "cat", 16: "dog"}

    def test_person_and_car(self) -> None:
        person, vehicle, labels = inf.detect_trigger(_FakeResult([0, 2], self.NAMES))
        self.assertTrue(person)
        self.assertTrue(vehicle)
        self.assertEqual(labels, ["car", "person"])

    def test_a_cat_is_an_animal_not_a_person_or_vehicle(self) -> None:
        person, vehicle, labels = inf.detect_trigger(_FakeResult([15, 16], self.NAMES))
        self.assertFalse(person)
        self.assertFalse(vehicle)
        self.assertEqual(labels, ["cat", "dog"])
        self.assertTrue(inf.has_animal(labels))
        self.assertFalse(inf.has_animal(["car", "person"]))

    def test_birds_never_count(self) -> None:
        self.assertEqual(inf.detect_trigger(_FakeResult([14], self.NAMES)), (False, False, []))

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
        self.assertEqual(sent["alert"], {"alert_id": "front_door_100_alert", "camera": "front_door", "label": "normal",
                                         "summary": "a person at the door", "ts": 100.0})
        self.assertIn("front_door: a person at the door", sent["text"])
        self.assertEqual(sent["image"], b"jpg")
        self.assertTrue(job.ready.is_set())
        self.assertEqual((job.alert["summary"], job.alert["alert_command"], job.alert["muted"]),
                         ("a person at the door", "[send_message]", False))
        self.assertEqual(job.alert["labels"], ["person"])

    def test_a_paused_camera_is_not_analysed_and_nothing_is_sent(self) -> None:
        assistant = _FakeAssistant(muted=True)
        job = self._run(assistant)

        self.assertEqual(assistant.sent, [])
        self.assertTrue(job.alert["muted"] and job.paused)
        self.assertEqual((job.alert["summary"], job.alert["alert_command"]), ("", "[none]"))   # the AI was not asked

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

    def test_only_what_the_owner_alerts_on_counts(self) -> None:
        car = {"summary": "a car arrives", "people": 0, "vehicle_moving": True}
        person = {"summary": "a person walks", "people": 1, "vehicle_moving": False}
        self.assertFalse(inf.vlm_confirms(car, alert_on=("person",)))      # a moving car is not an alert
        self.assertTrue(inf.vlm_confirms(person, alert_on=("person",)))
        self.assertFalse(inf.vlm_confirms(person, alert_on=("vehicle",)))
        self.assertTrue(inf.vlm_confirms(car, alert_on=("vehicle",)))
        self.assertTrue(inf.vlm_confirms(car, alert_on=("person", "vehicle")))

    def test_animals_count_only_when_the_owner_alerts_on_them(self) -> None:
        cat = {"summary": "a cat walks", "people": 0, "vehicle_moving": False, "animals": 1}
        self.assertFalse(inf.vlm_confirms(cat, alert_on=("person",)))
        self.assertTrue(inf.vlm_confirms(cat, alert_on=("animal",)))
        self.assertFalse(inf.vlm_confirms({"summary": "x", "people": 0, "vehicle_moving": False, "animals": 0},
                                          alert_on=("animal",)))

    def test_no_verdict_when_the_vlm_did_not_say(self) -> None:
        self.assertIsNone(inf.vlm_confirms(None))
        self.assertIsNone(inf.vlm_confirms({}))
        self.assertIsNone(inf.vlm_confirms({"summary": "something"}))           # an older or other backend
        self.assertIsNone(inf.vlm_confirms({"summary": "x", "people": "several"}))

    def test_the_prompt_asks_for_the_facts_the_decision_needs(self) -> None:
        prompt = inf.build_prompt("front_door", 0, "12:00:00", 22, 6)
        self.assertIn('"people"', prompt)
        self.assertIn('"animals"', prompt)
        self.assertIn("animals", inf.VLM_SCHEMA["required"])
        self.assertIn('"vehicle_moving"', prompt)

    def test_the_prompt_asks_for_summaries_in_the_style_we_tagged(self) -> None:
        prompt = inf.build_prompt("front_door", 0, "12:00:00", 22, 6)
        self.assertIn('"No special activity."', prompt)          # the taggers' sentence for an empty scene
        for word in ("hood", "knife", "appears to", "one to three short sentences"):
            self.assertIn(word, prompt)
        self.assertNotIn("[alert]", prompt)                       # the label carries that now


class _ParkedCarBackend:
    def analyze(self, frames, camera_name, t_sec, start_hour, end_hour):
        parsed = {"summary": "a car is parked in the driveway", "people": 0, "vehicle_moving": False}
        return "{}", parsed


class PausedCameraTest(unittest.TestCase):
    def test_a_paused_camera_is_not_analysed_and_its_clip_is_kept_for_training(self) -> None:
        class CountingBackend:
            calls = 0

            def analyze(self, *args):
                CountingBackend.calls += 1
                return "{}", {"summary": "a person"}

        assistant = _FakeAssistant(muted=True)
        job = inf.AlertJob(camera="front_door", stem="front_door_100_alert", ts=100.0, labels=["person"])
        decisions = []

        class Status:
            def decision(self, *args, **kwargs):
                decisions.append((args, kwargs))

        inf._worker(CountingBackend(), {"alert_channel": "telegram"}, {}, AlertSettings(), "front_door",
                    [object()], assistant, job, Status())
        self.assertEqual(CountingBackend.calls, 0)                  # no AI call while paused
        self.assertEqual(assistant.sent, [])
        self.assertTrue(job.paused and job.ready.is_set())
        self.assertEqual(job.alert["alert_command"], "[none]")
        self.assertTrue(decisions[0][1]["muted"])

        import tempfile

        import numpy as np

        from home_guard_project.box.alert_clips import encode_frame

        with tempfile.TemporaryDirectory() as tmp:
            production, training = os.path.join(tmp, "production_multi"), os.path.join(tmp, "dataset_multi")
            frames = [(100.0 + i * 0.2, encode_frame(np.zeros((48, 64, 3), dtype=np.uint8))) for i in range(5)]
            with mock.patch("home_guard_project.box.alert_clips._to_h264", return_value=False):
                inf._save_clip(job, frames, production, training, assistant)
            metas = [n for _, _, names in os.walk(training) for n in names if n.endswith(".meta.json")]
            self.assertEqual(metas, ["front_door_100_paused.meta.json"])
            self.assertFalse(os.path.exists(production))


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
        self.assertEqual(job.alert["vlm"], {"people": 0, "vehicle_moving": False, "animals": None})

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


# ---------------------------------------------------------------------------
# Parked vehicles: a vehicle-only trigger escalates only when the vehicles moved
# ---------------------------------------------------------------------------
CAR = (0.50, 0.50, 0.70, 0.60)           # a parked car, normalised xyxy
JITTER = (0.505, 0.502, 0.705, 0.603)    # the same car, the detector's box wobbling a little (IoU ~0.9)
SHIFTED = (0.60, 0.50, 0.80, 0.60)       # the same car half a length further on (IoU ~0.33)
OTHER = (0.10, 0.70, 0.30, 0.80)         # a second car elsewhere in the picture


class _FakeVehicleBox:
    def __init__(self, cls_id, xyxyn) -> None:
        self.cls = [cls_id]
        self.xyxyn = [list(xyxyn)]


class VehicleBoxesTest(unittest.TestCase):
    NAMES = {0: "person", 2: "car", 7: "truck", 15: "cat"}

    def test_only_the_vehicles_boxes_are_kept(self) -> None:
        result = _FakeResult([], self.NAMES)
        result.boxes = [_FakeVehicleBox(0, (0.1, 0.1, 0.2, 0.3)),
                        _FakeVehicleBox(2, CAR),
                        _FakeVehicleBox(15, (0.0, 0.0, 0.1, 0.1)),
                        _FakeVehicleBox(7, OTHER)]
        self.assertEqual(inf.vehicle_boxes(result), [CAR, OTHER])

    def test_no_boxes(self) -> None:
        self.assertEqual(inf.vehicle_boxes(_FakeResult([], self.NAMES)), [])


class VehiclesMovedTest(unittest.TestCase):
    def test_the_first_look_counts_as_moved(self) -> None:
        self.assertTrue(inf.vehicles_moved(None, [CAR]))

    def test_the_same_boxes_have_not_moved(self) -> None:
        self.assertFalse(inf.vehicles_moved([CAR], [CAR]))

    def test_the_detectors_wobble_is_not_movement(self) -> None:
        self.assertFalse(inf.vehicles_moved([CAR], [JITTER]))

    def test_a_car_that_drove_on_has_moved(self) -> None:
        self.assertTrue(inf.vehicles_moved([CAR], [SHIFTED]))

    def test_a_car_arriving_is_movement(self) -> None:
        self.assertTrue(inf.vehicles_moved([CAR], [CAR, OTHER]))
        self.assertTrue(inf.vehicles_moved([], [CAR]))

    def test_a_car_leaving_is_movement(self) -> None:
        self.assertTrue(inf.vehicles_moved([CAR, OTHER], [CAR]))
        self.assertTrue(inf.vehicles_moved([CAR], []))

    def test_no_vehicles_either_time_is_not_movement(self) -> None:
        self.assertFalse(inf.vehicles_moved([], []))

    def test_the_order_of_the_boxes_does_not_matter(self) -> None:
        self.assertFalse(inf.vehicles_moved([CAR, OTHER], [OTHER, CAR]))


class VehicleMemoryTest(unittest.TestCase):
    """One per camera: remembers where the vehicles were, across every look (cooldown looks too)."""

    def test_a_parked_car_is_movement_on_the_first_look_and_never_again(self) -> None:
        mem = inf.VehicleMemory()
        self.assertTrue(mem.look([CAR]))
        self.assertEqual([mem.look([b]) for b in (JITTER, CAR, JITTER, CAR)], [False] * 4)

    def test_a_car_that_creeps_is_caught_against_where_it_was_parked(self) -> None:
        # Each step on its own wobbles too little to count (IoU ~0.82 between neighbours),
        # so comparing only with the previous look would never see this car move.
        mem = inf.VehicleMemory()
        mem.look([CAR], now=0)
        steps = [(0.50 + 0.02 * i, 0.50, 0.70 + 0.02 * i, 0.60) for i in range(1, 8)]
        moved = [mem.look([s], now=0.5 * i) for i, s in enumerate(steps, 1)]
        self.assertFalse(moved[0])
        self.assertTrue(any(moved))

    def test_a_car_that_arrives_and_parks_during_the_cooldown_is_quiet_afterwards(self) -> None:
        mem = inf.VehicleMemory()
        mem.look([], now=0)                           # the empty driveway
        self.assertFalse(mem.look([SHIFTED], now=1))  # a car arrives: one look is not yet movement
        self.assertFalse(mem.look([CAR], now=1.4))    # still there, but not for a second yet
        self.assertTrue(mem.look([CAR], now=2.1))     # a second of change over three looks: it moved
        self.assertFalse(mem.look([CAR], now=2.5))    # parked
        self.assertFalse(mem.look([JITTER], now=120))  # ... two minutes later: no new alert

    def test_a_car_leaving_is_movement_and_the_empty_driveway_is_then_quiet(self) -> None:
        mem = inf.VehicleMemory()
        mem.look([CAR], now=0)
        self.assertFalse(mem.look([CAR], now=1))
        self.assertFalse(mem.look([], now=2))
        self.assertTrue(mem.look([], now=3.1))
        self.assertFalse(mem.look([], now=4))

    def test_the_detector_blinking_on_a_parked_car_is_not_movement(self) -> None:
        # Three looks a second on the graphics chip: a parked car missed for a look or two
        # (under a second) must not wake the AI.
        mem = inf.VehicleMemory()
        mem.look([CAR, OTHER], now=0)
        looks = [([CAR, OTHER], 0.3), ([CAR], 0.6), ([CAR, OTHER], 0.9), ([OTHER], 1.2), ([], 1.5),
                 ([CAR, OTHER], 1.8), ([CAR], 2.1), ([CAR], 2.4), ([CAR, OTHER], 2.7)]
        self.assertEqual([mem.look(b, now=t) for b, t in looks], [False] * len(looks))

    def test_one_slow_look_is_not_movement_but_two_are(self) -> None:
        # On the CPU a camera is looked at only every few seconds: a change must still be
        # seen twice, so one bad look does not count, a car that stays gone does.
        mem = inf.VehicleMemory()
        mem.look([CAR], now=0)
        self.assertFalse(mem.look([], now=3.5))
        self.assertFalse(mem.look([CAR], now=7))
        self.assertFalse(mem.look([], now=10.5))
        self.assertTrue(mem.look([], now=14))


class EscalationTest(unittest.TestCase):
    def test_a_person_always_escalates(self) -> None:
        self.assertTrue(inf.should_escalate(person=True, vehicle=False, vehicles_moved=False))
        self.assertTrue(inf.should_escalate(person=True, vehicle=True, vehicles_moved=False))

    def test_a_vehicle_escalates_only_when_it_moved(self) -> None:
        self.assertTrue(inf.should_escalate(person=False, vehicle=True, vehicles_moved=True))
        self.assertFalse(inf.should_escalate(person=False, vehicle=True, vehicles_moved=False))

    def test_nothing_in_view_does_not_escalate(self) -> None:
        self.assertFalse(inf.should_escalate(person=False, vehicle=False, vehicles_moved=True))

    def test_only_what_the_owner_alerts_on_wakes_the_ai(self) -> None:
        people_only = ("person",)
        self.assertFalse(inf.should_escalate(person=False, vehicle=True, vehicles_moved=True, alert_on=people_only))
        self.assertTrue(inf.should_escalate(person=True, vehicle=True, vehicles_moved=True, alert_on=people_only))
        vehicles_only = ("vehicle",)
        self.assertFalse(inf.should_escalate(person=True, vehicle=False, vehicles_moved=False, alert_on=vehicles_only))
        self.assertTrue(inf.should_escalate(person=False, vehicle=True, vehicles_moved=True, alert_on=vehicles_only))
        self.assertFalse(inf.should_escalate(person=False, vehicle=True, vehicles_moved=False, alert_on=vehicles_only))


class AnimalEscalationTest(unittest.TestCase):
    def test_an_animal_wakes_the_ai_only_where_the_owner_alerts_on_animals(self) -> None:
        self.assertFalse(inf.should_escalate(person=False, vehicle=False, vehicles_moved=False,
                                             alert_on=("person",), animal=True))
        self.assertTrue(inf.should_escalate(person=False, vehicle=False, vehicles_moved=False,
                                            alert_on=("person", "animal"), animal=True))


class QuietReasonTest(unittest.TestCase):
    def test_the_log_says_why_the_ai_was_not_asked(self) -> None:
        self.assertEqual(inf.quiet_reason(True, False, ("person", "vehicle")), "vehicles have not moved")
        self.assertEqual(inf.quiet_reason(True, True, ("person",)), "this camera alerts only on person")
        self.assertEqual(inf.quiet_reason(False, False, ("person", "vehicle")),
                         "this camera alerts only on person, vehicle")


class AlertOnTest(unittest.TestCase):
    def test_people_only_unless_the_owner_chose_otherwise(self) -> None:
        self.assertEqual(inf.AlertSettings().alert_on, ("person",))
        self.assertEqual(inf.AlertSettings.from_box_settings({}).alert_on, ("person",))
        self.assertEqual(inf.AlertSettings.from_box_settings({"alert_on": "vehicle,person"}).alert_on,
                         ("person", "vehicle"))
        self.assertEqual(inf.AlertSettings.from_box_settings({"alert_on": ["vehicle"]}).alert_on, ("vehicle",))
        self.assertEqual(inf.AlertSettings.from_box_settings({"alert_on": "animal,person"}).alert_on,
                         ("person", "animal"))

    def test_a_broken_value_falls_back_to_people(self) -> None:
        for bad in ("", "cats", None, 5):
            self.assertEqual(inf.AlertSettings.from_box_settings({"alert_on": bad}).alert_on, ("person",), bad)

    def test_the_window_sees_the_choice(self) -> None:
        self.assertEqual(inf.AlertSettings(alert_on=("person", "vehicle")).live_values()["alert_on"],
                         ["person", "vehicle"])


class _FakeRing:
    def __init__(self) -> None:
        self.added: list = []

    def wants(self, now: float) -> bool:
        return True

    def add(self, now: float, encoded) -> None:
        self.added.append(encoded)


class StreamMaskTest(unittest.TestCase):
    def test_a_frame_is_masked_once_before_read_and_before_the_clip_ring(self) -> None:
        import threading

        import numpy as np

        from home_guard_project.data_collection.zones import ZoneMask

        stream = inf._Stream.__new__(inf._Stream)      # no capture, no thread
        stream._lock = threading.Lock()
        stream._frame = None
        stream._ring = _FakeRing()
        stream._mask = ZoneMask([(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)])   # the left half
        frame = np.full((10, 20, 3), 255, dtype=np.uint8)

        with mock.patch("home_guard_project.box.alert_clips.encode_frame", side_effect=lambda f: f):
            stream._ingest(frame, now=1.0)

        seen = stream.read()
        self.assertEqual(int(seen[:, :10].min()), 255)
        self.assertEqual(int(seen[:, 11:].max()), 0)
        (ringed,) = stream._ring.added
        self.assertEqual(int(ringed[:, 11:].max()), 0)

    def test_without_a_mask_the_frame_is_kept_whole(self) -> None:
        import threading

        import numpy as np

        stream = inf._Stream.__new__(inf._Stream)
        stream._lock = threading.Lock()
        stream._frame = None
        stream._ring = None
        stream._mask = None
        stream._ingest(np.full((4, 4, 3), 7, dtype=np.uint8), now=1.0)
        self.assertEqual(int(stream.read().min()), 7)

    def test_a_frame_the_mask_cannot_handle_is_dropped_and_the_reader_survives(self) -> None:
        import threading

        import numpy as np

        stream = inf._Stream.__new__(inf._Stream)
        stream.name = "cam"
        stream.url = "rtsp://x"
        stream._lock = threading.Lock()
        stream._frame = None
        stream._ring = None
        stream._running = True
        frames = [np.full((4, 4, 3), 9, dtype=np.uint8), np.full((4, 4, 3), 5, dtype=np.uint8)]

        class _Mask:
            calls = 0

            def apply(self, frame):
                _Mask.calls += 1
                if _Mask.calls == 1:
                    raise RuntimeError("bad frame")
                return frame * 0 + 3

        class _Cap:
            def read(self_inner):
                if frames:
                    return True, frames.pop(0)
                stream._running = False
                return False, None

            def release(self_inner):
                pass

        stream._mask = _Mask()
        stream._cap = _Cap()
        with mock.patch("time.sleep"), mock.patch("cv2.VideoCapture", return_value=_Cap()),                 self.assertLogs("box.inference", "WARNING") as logs:
            stream._loop()
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(int(stream.read().max()), 3)
