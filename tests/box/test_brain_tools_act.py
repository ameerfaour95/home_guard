# tests/box/test_brain_tools_act.py
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from test_brain_events import meta
from test_brain_tools_read import FakeVision, snapshot

from home_guard_project.box.brain.memory import ChatState
from home_guard_project.box.brain.receipts import DONE, FAILED, REQUESTED, ReceiptBook
from home_guard_project.box.brain.tools import (
    TOOLS,
    Services,
    ToolContext,
    check_camera,
    find_events,
    pause_alerts,
    quoted_from,
    record_clip,
    record_verdict,
    resume_alerts,
    send_media,
    set_alias,
    set_camera_active,
)
from home_guard_project.box.feedback import Feedback, MuteState

NOW = dt.datetime(2026, 10, 3, 23, 0).timestamp()
HOUR = 3600.0


class FakeDeliver:
    def __init__(self, ok=True):
        self.ok = ok
        self.sent = []

    def photo(self, chat_id, path, caption=""):
        self.sent.append(("photo", path))
        return {"ok": self.ok, "message_id": 5 if self.ok else None, "error": "" if self.ok else "refused"}

    def video(self, chat_id, path, caption=""):
        self.sent.append(("video", path))
        return {"ok": self.ok, "message_id": 6 if self.ok else None, "error": "" if self.ok else "refused"}


class ActToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = temp.name
        for name in ("LIVE_DIR", "PRODUCTION_ARCHIVE_DIR"):
            isolated = patch("home_guard_project.box.boxconfig." + name, os.path.join(self.root, name))
            isolated.start()
            self.addCleanup(isolated.stop)
        meta(self.root, "main_entrance", "main_entrance_1_alert", NOW - 5 * HOUR,
             summary="A man stands at the door.", label="suspicious", people=1)
        meta(self.root, "back_door", "back_door_2_alert", NOW - 4 * HOUR, summary="A woman.", with_clip=False)
        self.photo = os.path.join(self.root, "live.jpg")
        with open(self.photo, "wb") as f:
            f.write(b"jpg")
        self.deliver = FakeDeliver()
        self.mute = MuteState(os.path.join(self.root, "mute.json"))
        self.camera_calls, self.restarts, self.cuts = [], [], []

    def services(self, **kw) -> Services:
        def cut(clip, clip_start, clip_end, start, seconds, out):
            self.cuts.append((start, seconds))
            with open(out, "wb") as f:
                f.write(b"seg")
            return (start, start + seconds)

        base = dict(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"), feedback_dir=self.root,
                    work_dir=os.path.join(self.root, ".live"), mute=self.mute, deliver=self.deliver,
                    vision=FakeVision(), grab_photo=lambda cam: {"camera": cam, "image": self.photo},
                    record_live=lambda cam, s: {"ok": True, "path": self.photo, "start": NOW, "end": NOW + s},
                    cut_segment=cut,
                    set_camera=lambda cam, active: self.camera_calls.append((cam, active)) or {"ok": True},
                    add_alias=lambda cam, alias, cams: [alias],
                    request_restart=lambda: self.restarts.append(1), now=lambda: NOW)
        base.update(kw)
        return Services(**base)

    def ctx(self, text="", mode="guard", **kw) -> ToolContext:
        return ToolContext(turn_id="t1", chat_id="-5", speaker={"user_id": 1, "name": "A"}, text=text, lang="he",
                           mode=mode, snapshot=snapshot(mode), state=ChatState(), services=self.services(**kw),
                           book=ReceiptBook(os.path.join(self.root, ".receipts"), now=lambda: NOW))

    def test_every_tool_is_registered(self) -> None:
        self.assertEqual(len(TOOLS), 13)

    def test_quoted_from(self) -> None:
        self.assertTrue(quoted_from("stop until six", "it's me, stop until six please"))
        self.assertFalse(quoted_from("this", "this"))
        self.assertFalse(quoted_from("nothing there", "send the video"))

    def test_check_camera_sends_the_photo_and_describes_it(self) -> None:
        ctx = self.ctx()
        out = check_camera(ctx, {"camera": "entrance"})
        self.assertEqual((out["ok"], out["status"], out["quality"], out["label"]), (True, DONE, "clear", "normal"))
        self.assertEqual(self.deliver.sent, [("photo", self.photo)])
        self.assertEqual(ctx.receipts[0].detail["camera"], "main_entrance")

    def test_check_camera_off_or_failed_delivery(self) -> None:
        ctx = self.ctx()
        self.assertEqual(check_camera(ctx, {"camera": "back"})["reason"], "camera_off")
        self.deliver.ok = False
        out = check_camera(ctx, {"camera": "front"})
        self.assertEqual((out["ok"], out["reason"]), (False, "telegram"))
        self.assertIn("description", out)       # facts are still returned

    def test_send_media_whole_clip_part_of_clip_and_gone_clip(self) -> None:
        ctx = self.ctx()
        events = find_events(ctx, {"last_hours": 24})["events"]
        by_cam = {e["camera"]: e["handle"] for e in events}
        whole = send_media(ctx, {"handle": by_cam["main_entrance"]})
        self.assertEqual((whole["status"], ctx.receipts[-1].detail["kind"]), (DONE, "video"))
        part = send_media(ctx, {"handle": by_cam["main_entrance"], "from_sec": -4, "seconds": 4})
        self.assertEqual(part["status"], DONE)
        self.assertEqual(self.cuts, [(NOW - 5 * HOUR - 10, 4.0)])   # trigger_ts (ts - 6) minus 4 seconds
        gone = send_media(ctx, {"handle": by_cam["back_door"]})
        self.assertEqual((gone["status"], gone["reason"]), (FAILED, "not_on_box"))

    def test_send_media_caps_at_three(self) -> None:
        ctx = self.ctx()
        handle = find_events(ctx, {"last_hours": 24, "cameras": ["entrance"]})["events"][0]["handle"]
        for _ in range(3):
            send_media(ctx, {"handle": handle})
        self.assertEqual(send_media(ctx, {"handle": handle})["reason"], "too_many")

    def test_record_clip(self) -> None:
        ctx = self.ctx()
        out = record_clip(ctx, {"camera": "front", "seconds": 99})
        self.assertEqual(out["status"], DONE)
        self.assertEqual(ctx.receipts[0].detail["seconds"], 30)

    def test_pause_needs_the_owners_words_and_resume_one_camera(self) -> None:
        ctx = self.ctx(text="זה אני, תשתיק את הכניסה עד שש")
        self.assertFalse(pause_alerts(ctx, {"owner_words": "please pause", "cameras": ["entrance"]})["ok"])
        out = pause_alerts(ctx, {"owner_words": "תשתיק את הכניסה", "cameras": ["entrance"], "until": "06:00"})
        self.assertEqual(out["status"], DONE)
        self.assertTrue(self.mute.is_muted(NOW, "main_entrance"))
        self.assertFalse(self.mute.is_muted(NOW, "front_side"))
        detail = ctx.receipts[-1].detail
        self.assertEqual((detail["camera"], detail["until"], detail["by"]), ("main_entrance", "06:00", "A"))
        self.assertEqual(detail["before"], {"all": 0.0, "cameras": {}})
        resume_alerts(ctx, {"cameras": ["entrance"]})
        self.assertFalse(self.mute.is_muted(NOW, "main_entrance"))

    def test_record_verdict_needs_a_resolved_event_and_a_real_quote(self) -> None:
        ctx = self.ctx(text="הזה")
        self.assertIn("no alert", record_verdict(ctx, {"verdict": "false_alarm", "owner_words": "הזה"})["error"])
        ctx = self.ctx(text="nobody was there, false alarm")
        ctx.alert_handle = ctx.state.add_handle("event", "main_entrance_1_alert", "main_entrance", NOW, "x")
        self.assertFalse(record_verdict(ctx, {"verdict": "false_alarm", "owner_words": "false"})["ok"])
        self.assertFalse(record_verdict(ctx, {"verdict": "maybe", "owner_words": "false alarm"})["ok"])
        out = record_verdict(ctx, {"verdict": "false_alarm", "owner_words": "false alarm"})
        self.assertEqual(out["status"], DONE)
        saved = glob.glob(os.path.join(self.root, "feedback", "**", "*.feedback.json"), recursive=True)
        with open(saved[0], encoding="utf-8") as stream:
            self.assertEqual(json.load(stream)["verdict"], "false_alarm")
        self.assertEqual(ctx.saved, 1)

    def test_set_camera_active_is_requested_and_restarts_after_the_reply(self) -> None:
        ctx = self.ctx()
        out = set_camera_active(ctx, {"camera": "front", "active": False})
        self.assertEqual(out["status"], REQUESTED)
        self.assertEqual(self.camera_calls, [("front_side", False)])
        self.assertEqual(self.restarts, [])
        for fn in ctx.after_reply:
            fn()
        self.assertEqual(self.restarts, [1])
        self.assertEqual(ctx.receipts[-1].detail["chat_id"], "-5")
        already = set_camera_active(ctx, {"camera": "back", "active": False})
        self.assertEqual(already["status"], DONE)
        again = set_camera_active(ctx, {"camera": "front", "active": True})      # follows this turn's change
        self.assertEqual(again["status"], REQUESTED)
        self.assertEqual(self.restarts, [1])                                      # one restart is enough
        last = self.ctx()
        set_camera_active(last, {"camera": "front", "active": False})
        self.assertEqual(set_camera_active(last, {"camera": "entrance", "active": False})["reason"], "last_camera")

    def test_set_alias(self) -> None:
        ctx = self.ctx()
        self.assertEqual(set_alias(ctx, {"camera": "front", "alias": "street"})["status"], DONE)
        boom = self.ctx(add_alias=lambda cam, alias, cams: (_ for _ in ()).throw(ValueError('"x" already names y')))
        out = set_alias(boom, {"camera": "front", "alias": "x"})
        self.assertEqual((out["status"], out["reason"]), (FAILED, '"x" already names y'))


    def test_acting_entries_contain_malformed_arguments(self):
        for name in ("check_camera", "record_clip", "send_media", "pause_alerts", "resume_alerts",
                     "record_verdict", "set_camera_active", "set_alias"):
            for args in ([], {"bad": object()}, {"bad": float("nan")}, {"bad": float("inf")}):
                with self.subTest(tool=name, args=args):
                    with self.assertLogs("box.brain.tools", level="WARNING") as logs:
                        out = TOOLS[name](self.ctx(), args)
                    self.assertEqual(len(logs.output), 1)
                    self.assertFalse(out["ok"])
                    json.dumps(out, allow_nan=False)

    def test_bad_camera_change_results_and_non_boolean_active(self):
        for result in (None, [], {"ok": False}, {}, {"ok": "yes"}, {"ok": True, "extra": object()}):
            ctx = self.ctx(set_camera=lambda *a: result)
            if result == {"ok": False}:
                self.assertFalse(set_camera_active(ctx, {"camera": "front", "active": False})["ok"])
            else:
                with self.assertLogs("box.brain.tools", level="WARNING") as logs:
                    self.assertFalse(set_camera_active(ctx, {"camera": "front", "active": False})["ok"])
                self.assertEqual(len(logs.output), 1)
            self.assertEqual(ctx.camera_states, {})
            self.assertEqual(ctx.after_reply, [])
        for active in ("false", 0, None, []):
            self.assertFalse(set_camera_active(self.ctx(), {"camera": "front", "active": active})["ok"])
        self.assertEqual(self.camera_calls, [])

    def test_feedback_failure_keeps_receipts_and_prior_mute(self):
        self.mute.apply(Feedback(
            action="mute", mute_until=NOW + HOUR, camera="front_side"), NOW)
        ctx = self.ctx(text="pause alerts please")
        before = self.mute.snapshot()
        def fail(*args):
            self.assertTrue(ctx.receipts)
            raise OSError("disk failed")
        with patch("home_guard_project.box.brain.tools.save_feedback", side_effect=fail):
            with self.assertLogs("box.brain.tools", level="WARNING"):
                out = pause_alerts(ctx, {"owner_words": "pause alerts", "cameras": ["entrance"]})
                self.assertTrue(out["ok"])
                self.assertEqual(ctx.receipts[0].detail["before"], before)
                self.assertTrue(self.mute.is_muted(NOW, "front_side"))
                self.assertTrue(resume_alerts(ctx, {"cameras": ["entrance"]})["ok"])
        self.assertEqual([r.status for r in ctx.receipts], [DONE, DONE])
        self.assertEqual(ctx.saved, 0)

    def test_malformed_live_services_are_contained(self):
        for value in (None, [], {}, {"ok": "yes"}, {"ok": True, "path": [], "start": NOW, "end": NOW}):
            ctx = self.ctx(record_live=lambda *a: value)
            with self.assertLogs("box.brain.tools", level="WARNING") as logs:
                self.assertFalse(record_clip(ctx, {"camera": "front"})["ok"])
            self.assertEqual(len(logs.output), 1)
        for value in (None, [], {}, {"image": []}):
            with self.assertLogs("box.brain.tools", level="WARNING") as logs:
                self.assertFalse(check_camera(self.ctx(grab_photo=lambda *a: value), {"camera": "front"})["ok"])
            self.assertEqual(len(logs.output), 1)
        for value in (None, [], {}, {"ok": "yes"}, {"ok": True, "message_id": object()}):
            self.deliver.photo = lambda *a, **kw: value
            with self.assertLogs("box.brain.tools", level="WARNING") as logs:
                self.assertFalse(check_camera(self.ctx(), {"camera": "front"})["ok"])
            self.assertEqual(len(logs.output), 1)

    def test_nonfinite_numeric_strings_do_not_act(self):
        for value in ("nan", "inf", "-inf", "bad", []):
            with self.assertLogs("box.brain.tools", level="WARNING") as logs:
                self.assertFalse(record_clip(self.ctx(), {"camera": "front", "seconds": value})["ok"])
            self.assertEqual(len(logs.output), 1)
            ctx = self.ctx(text="pause alerts please")
            with self.assertLogs("box.brain.tools", level="WARNING") as logs:
                self.assertFalse(pause_alerts(ctx, {"owner_words": "pause alerts", "minutes": value})["ok"])
            self.assertEqual(len(logs.output), 1)

    def test_malformed_vision_retains_delivery_receipt(self):
        for value in (None, [], {}, {"ok": True}, {"ok": True, "description": object()}):
            vision = FakeVision()
            vision.look = lambda *a, **kw: value
            ctx = self.ctx(vision=vision)
            with self.assertLogs("box.brain.tools", level="WARNING") as logs:
                out = check_camera(ctx, {"camera": "front"})
            self.assertEqual(len(logs.output), 1)
            self.assertTrue(out["ok"])
            self.assertEqual(ctx.receipts[-1].status, DONE)
            self.assertIn("description_error", out)
            json.dumps(out, allow_nan=False)

    def test_bad_video_returns_and_camera_exceptions_have_failure_receipts(self):
        for tool in ("record_clip", "send_media"):
            self.deliver.video = lambda *a, **kw: {"ok": "yes"}
            ctx = self.ctx()
            args = {"camera": "front"} if tool == "record_clip" else {
                "handle": find_events(ctx, {"last_hours": 24, "cameras": ["entrance"]})["events"][0]["handle"]}
            with self.assertLogs("box.brain.tools", level="WARNING"):
                self.assertFalse(TOOLS[tool](ctx, args)["ok"])
        ctx = self.ctx(set_camera=lambda *a: (_ for _ in ()).throw(OSError("bad")))
        with self.assertLogs("box.brain.tools", level="WARNING"):
            out = set_camera_active(ctx, {"camera": "front", "active": False})
        self.assertEqual(out["status"], FAILED)
        self.assertEqual(ctx.receipts[-1].reason, "error")

    def test_bad_segment_numbers_log_once_without_cut_or_send(self):
        for field in ("from_sec", "seconds"):
            for value in ("nan", "inf", "bad", []):
                with self.subTest(field=field, value=value):
                    ctx = self.ctx()
                    handle = find_events(ctx, {"last_hours": 24, "cameras": ["entrance"]})["events"][0]["handle"]
                    with self.assertLogs("box.brain.tools", level="WARNING") as logs:
                        out = send_media(ctx, {"handle": handle, field: value})
                    self.assertEqual(len(logs.output), 1)
                    self.assertFalse(out["ok"])
                    self.assertEqual(self.cuts, [])
                    self.assertEqual(self.deliver.sent, [])

    def test_nonboolean_live_vision_ok_keeps_receipt_without_description(self):
        for ok in ("false", 1):
            vision = FakeVision()
            vision.look = lambda *a, **kw: {"ok": ok, "description": "A person.", "quality": "clear",
                                           "people": 1, "label": "normal", "why": "Ordinary activity."}
            ctx = self.ctx(vision=vision)
            with self.assertLogs("box.brain.tools", level="WARNING") as logs:
                out = check_camera(ctx, {"camera": "front"})
            self.assertEqual(len(logs.output), 1)
            self.assertEqual(out["status"], DONE)
            self.assertIn("description_error", out)
            self.assertNotIn("description", out)
            self.assertEqual(ctx.receipts[-1].status, DONE)

    def test_invalid_file_data_is_contained(self):
        ctx = self.ctx()
        handle = find_events(ctx, {"last_hours": 24, "cameras": ["entrance"]})["events"][0]["handle"]
        for path in glob.glob(os.path.join(self.root, "meta", "**", "*.json"), recursive=True):
            with open(path, "wb") as stream:
                stream.write(b"\xff")
        with self.assertLogs(level="WARNING"):
            json.dumps(send_media(ctx, {"handle": handle}), allow_nan=False)
        photo = ctx.state.add_handle("photo", self.root, "front_side", NOW)
        self.assertEqual(send_media(ctx, {"handle": photo})["reason"], "not_on_box")

    def test_segment_service_receives_six_arguments(self):
        calls = []
        def cut(*args):
            calls.append(args)
            return (args[3], args[3] + args[4])
        ctx = self.ctx(cut_segment=cut)
        handle = find_events(ctx, {"last_hours": 24, "cameras": ["entrance"]})["events"][0]["handle"]
        self.assertTrue(send_media(ctx, {"handle": handle, "from_sec": -4, "seconds": 4})["ok"])
        self.assertEqual(len(calls[0]), 6)
        self.assertEqual(calls[0][2:5], (NOW - 5 * HOUR, NOW - 5 * HOUR - 10, 4.0))


    def test_pause_that_took_effect_keeps_its_receipt_when_saving_fails(self) -> None:
        ctx = self.ctx(text="stop until six please")
        with patch("home_guard_project.box.feedback._write_json", side_effect=OSError("disk full")):
            out = pause_alerts(ctx, {"owner_words": "stop until six", "cameras": ["entrance"], "until": "06:00"})
        self.assertEqual((out["ok"], out["status"]), (True, DONE))
        self.assertEqual((ctx.receipts[0].status, ctx.receipts[0].detail["saved"]), (DONE, False))
        self.assertTrue(self.mute.is_muted(NOW, "main_entrance"))

    def test_resume_that_took_effect_keeps_its_receipt_when_saving_fails(self) -> None:
        self.mute.apply(Feedback(action="mute", mute_until=NOW + HOUR, camera="main_entrance"), NOW)
        ctx = self.ctx(text="resume please")
        with patch("home_guard_project.box.feedback._write_json", side_effect=OSError("disk full")):
            out = resume_alerts(ctx, {"cameras": ["entrance"]})
        self.assertEqual((out["ok"], ctx.receipts[0].status), (True, DONE))
        self.assertFalse(self.mute.is_muted(NOW, "main_entrance"))

    def test_pause_that_did_not_take_effect_is_failed(self) -> None:
        ctx = self.ctx(text="stop until six please")
        with patch.object(MuteState, "apply", side_effect=OSError("boom")):
            out = pause_alerts(ctx, {"owner_words": "stop until six", "cameras": ["entrance"], "until": "06:00"})
        self.assertEqual((out["ok"], ctx.receipts[0].status, ctx.receipts[0].reason), (False, FAILED, "error"))

    def test_pointer_words_are_never_a_quote(self) -> None:
        self.assertFalse(quoted_from("this one", "this one"))
        self.assertFalse(quoted_from("הזה הזה", "הזה הזה"))
        self.assertFalse(quoted_from("هذا هنا", "هذا هنا"))
        self.assertTrue(quoted_from("stop until six", "ok stop until six"))
        ctx = self.ctx(text="this one")
        ctx.alert_handle = ctx.state.add_handle("event", "main_entrance_1_alert", "main_entrance", NOW, "x")
        out = record_verdict(ctx, {"verdict": "false_alarm", "owner_words": "this one"})
        self.assertFalse(out["ok"])
        self.assertEqual(ctx.saved, 0)

    def test_quote_must_be_whole_words(self) -> None:
        self.assertFalse(quoted_from("op unt", "stop until six"))
        self.assertTrue(quoted_from("Until, SIX", "stop until six."))

    def test_live_photos_count_toward_the_media_cap(self) -> None:
        ctx = self.ctx()
        for _ in range(3):
            self.assertEqual(check_camera(ctx, {"camera": "entrance"})["status"], DONE)
        out = check_camera(ctx, {"camera": "entrance"})
        self.assertEqual((out["ok"], out["reason"]), (False, "too_many"))
        self.assertEqual(len(self.deliver.sent), 3)

    def test_repeating_a_camera_change_in_one_turn_is_requested(self) -> None:
        ctx = self.ctx()
        set_camera_active(ctx, {"camera": "front", "active": False})
        set_camera_active(ctx, {"camera": "front", "active": False})
        self.assertEqual([r.status for r in ctx.receipts], [REQUESTED, REQUESTED])
        self.assertTrue(ctx.receipts[1].detail["already"])

    def test_call_key_never_carries_over(self) -> None:
        ctx = self.ctx()
        ctx.call_key = "k1"
        check_camera(ctx, {"camera": "nowhere"})      # fails before any receipt
        self.assertEqual(ctx.call_key, "")


if __name__ == "__main__":
    unittest.main()
