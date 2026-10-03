# tests/box/test_brain_tools_read.py
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import Mock

from test_brain_events import meta

from home_guard_project.box.brain.memory import ChatState
from home_guard_project.box.brain.receipts import ReceiptBook
from home_guard_project.box.brain.registry import CameraState, HouseSnapshot
from home_guard_project.box.brain.vision import VISION_VERSION
from home_guard_project.box.brain.tools import (
    TOOLS,
    MAX_DESCRIBE_PER_TURN,
    Services,
    ToolContext,
    ask_clarification,
    assess_event,
    describe_event,
    find_events,
    summarize_period,
)

NOW = dt.datetime(2026, 10, 3, 23, 0).timestamp()
HOUR = 3600.0


class FakeVision:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def look(self, camera, images, guard, question="", what=""):
        self.calls.append((camera, guard, question))
        if self.result is not None:
            return self.result
        out = {"ok": True, "description": "A courier leaves a parcel at the door.", "quality": "clear", "people": 1}
        if guard:
            out.update(label="normal", why="")
        return out


def snapshot(mode="guard"):
    return HouseSnapshot(now=NOW, mode=mode, mode_ends=NOW + 7 * HOUR, mode_started=NOW - HOUR, start_hour=22,
                         end_hour=6, cameras=(
                             CameraState("main_entrance", True, ("entrance", "front door"), live=True),
                             CameraState("back_door", False, ("back",), live=False),
                             CameraState("front_side", True, ("front",), live=True)))


class ReadToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        meta(self.root, "main_entrance", "main_entrance_1_alert", NOW - 5 * HOUR,
             summary="A man stands at the door looking around.", label="suspicious", people=1)
        meta(self.root, "back_door", "back_door_2_alert", NOW - 4 * HOUR,
             summary="A woman carries bags into the house.", label="normal", people=1)
        meta(self.root, "main_entrance", "main_entrance_3_quiet", NOW - 14 * HOUR, kind="quiet")
        self.vision = FakeVision()

    def ctx(self, mode="guard", vision=None) -> ToolContext:
        services = Services(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"),
                            feedback_dir=self.root, work_dir=os.path.join(self.root, ".live"), mute=None,
                            deliver=None, vision=vision or self.vision, clip_frames=lambda path: [b"jpg"],
                            now=lambda: NOW)
        return ToolContext(turn_id="t1", chat_id="-5", speaker={}, text="", lang="en", mode=mode,
                           snapshot=snapshot(mode), state=ChatState(), services=services,
                           book=ReceiptBook(os.path.join(self.root, ".receipts"), now=lambda: NOW))

    def test_find_events_by_alias_and_time_gives_handles_and_coverage(self) -> None:
        ctx = self.ctx()
        out = find_events(ctx, {"cameras": ["entrance"], "last_hours": 24})
        self.assertEqual([e["handle"] for e in out["events"]], ["E1", "E2"])
        self.assertEqual(out["events"][0]["label"], "suspicious")      # guard: most serious first
        self.assertEqual(out["coverage"]["cameras_off"], ["back_door"])
        self.assertEqual(ctx.shown, ["E1", "E2"])

    def test_unknown_and_ambiguous_cameras(self) -> None:
        self.assertIn("unknown camera", find_events(self.ctx(), {"cameras": ["garage"]})["error"])
        out = find_events(self.ctx(), {"cameras": ["front door and front"]})
        self.assertFalse(out["ok"])
        self.assertEqual(set(out["candidates"]), {"main_entrance", "front_side"})

    def test_meaning_search_includes_unconfirmed_detector_hits(self) -> None:
        out = find_events(self.ctx("assistant"), {"what": "someone at the door", "last_hours": 24})
        summaries = [e["summary"] for e in out["events"]]
        self.assertIn("detector saw a person, not confirmed", summaries)
        self.assertNotIn("A woman carries bags into the house.", summaries)

    def test_summarize_period(self) -> None:
        out = summarize_period(self.ctx(), {"last_hours": 24})
        self.assertEqual(out["total"], 3)
        self.assertEqual(out["by_camera"], {"main_entrance": 2, "back_door": 1})
        self.assertEqual(out["by_kind"], {"alert": 2, "quiet": 1})
        self.assertEqual(out["events"][0]["label"], "suspicious")

    def test_describe_event_answers_the_question_and_caches(self) -> None:
        ctx = self.ctx("assistant")
        handle = find_events(ctx, {"kind": "quiet", "last_hours": 24})["events"][0]["handle"]
        out = describe_event(ctx, {"handle": handle, "question": "what was he holding?"})
        self.assertEqual(out["description"], "A courier leaves a parcel at the door.")
        self.assertEqual(self.vision.calls, [("main_entrance", False, "what was he holding?")])
        describe_event(ctx, {"handle": handle, "question": "what was he holding?"})
        self.assertEqual(len(self.vision.calls), 1)                      # cached
        again = find_events(self.ctx("assistant"), {"kind": "quiet", "last_hours": 24})["events"][0]
        self.assertTrue(again["described"])

    def test_describe_event_is_capped_per_turn(self) -> None:
        ctx = self.ctx("assistant")
        handle = find_events(ctx, {"kind": "quiet", "last_hours": 24})["events"][0]["handle"]
        for i in range(MAX_DESCRIBE_PER_TURN):
            self.assertTrue(describe_event(ctx, {"handle": handle, "question": f"q{i}"})["ok"])
        self.assertFalse(describe_event(ctx, {"handle": handle, "question": "one more"})["ok"])

    def test_assess_event_refusal_is_never_normal(self) -> None:
        ctx = self.ctx(vision=FakeVision({"ok": False, "refused": True, "error": "refused"}))
        handle = find_events(ctx, {"kind": "quiet", "last_hours": 24})["events"][0]["handle"]
        out = assess_event(ctx, {"handle": handle})
        self.assertEqual((out["ok"], out["assessment"]), (True, "unavailable"))
        self.assertNotIn("label", out)

    def test_assess_event_returns_label_and_why(self) -> None:
        ctx = self.ctx()
        handle = find_events(ctx, {"kind": "quiet", "last_hours": 24})["events"][0]["handle"]
        self.assertEqual(assess_event(ctx, {"handle": handle})["label"], "normal")
        self.assertIn("unknown handle", assess_event(ctx, {"handle": "E99"})["error"])

    def test_ask_clarification(self) -> None:
        ctx = self.ctx()
        self.assertFalse(ask_clarification(ctx, {"question": "Which?", "choices": ["a"]})["ok"])
        self.assertTrue(ask_clarification(ctx, {"question": "Which camera?", "choices": ["a", "b"]})["ok"])
        self.assertEqual(ctx.clarification["choices"], ["a", "b"])

    def test_each_tool_contains_malformed_arguments(self) -> None:
        for name, tool in TOOLS.items():
            for args in ([], None, "bad", {"unexpected": object()}, {"unexpected": float("nan")}):
                with self.subTest(tool=name, args=args):
                    with self.assertLogs("box.brain.tools", level="WARNING") as logs:
                        out = tool(self.ctx(), args)
                    self.assertFalse(out["ok"])
                    self.assertEqual(len(logs.output), 1)
                    json.dumps(out, allow_nan=False)

    def test_find_and_summary_reject_bad_hours_and_contain_service_errors(self) -> None:
        for tool in (find_events, summarize_period):
            for hours in ("bad", [], {}, "nan", "inf", float("nan"), float("inf")):
                with self.subTest(tool=tool.__name__, hours=hours):
                    with self.assertLogs("box.brain.tools", level="WARNING"):
                        self.assertFalse(tool(self.ctx(), {"last_hours": hours})["ok"])
            ctx = self.ctx()
            ctx.services.roots = Mock(side_effect=OSError("disk unavailable"))
            with self.assertLogs("box.brain.tools", level="WARNING"):
                self.assertFalse(tool(ctx, {})["ok"])

    def test_clarification_rejects_wrong_choice_container(self) -> None:
        for choices in (42, "ab", {"a": 1, "b": 2}):
            with self.subTest(choices=choices):
                ctx = self.ctx()
                with self.assertLogs("box.brain.tools", level="WARNING"):
                    self.assertFalse(ask_clarification(ctx, {"question": "Which?", "choices": choices})["ok"])
                self.assertIsNone(ctx.clarification)

    def test_describe_and_assess_contain_malformed_handle_entries(self) -> None:
        for tool in (describe_event, assess_event):
            for entry in (["event"], {"kind": "event"}):
                with self.subTest(tool=tool.__name__, entry=entry):
                    ctx = self.ctx()
                    ctx.state.handles["E1"] = entry
                    with self.assertLogs("box.brain.tools", level="WARNING"):
                        self.assertFalse(tool(ctx, {"handle": "E1"})["ok"])

    def test_bad_cached_descriptions_are_skipped(self) -> None:
        for tool, mode in ((describe_event, "assistant"), (assess_event, "guard")):
            for cached in (["bad"], {"text": "stale"},
                           {"text": "stale", "quality": "clear", "people": "many", "ts": NOW},
                           {"text": "stale", "quality": "clear", "people": 1, "ts": float("inf")}):
                with self.subTest(tool=tool.__name__, cached=cached):
                    ctx = self.ctx()
                    handle = find_events(ctx, {"label": "suspicious"})["events"][0]["handle"]
                    os.makedirs(ctx.services.desc_dir, exist_ok=True)
                    path = os.path.join(ctx.services.desc_dir, "main_entrance_1_alert.json")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump({f"{mode}:{VISION_VERSION}:": cached}, f)
                    before = len(self.vision.calls)
                    with self.assertLogs("box.brain.tools", level="WARNING"):
                        out = tool(ctx, {"handle": handle})
                    self.assertTrue(out["ok"])
                    self.assertEqual(len(self.vision.calls), before + 1)
                    self.assertEqual(out["description"], "A courier leaves a parcel at the door.")

    def test_invalid_utf8_cache_is_skipped(self) -> None:
        ctx = self.ctx()
        handle = find_events(ctx, {"label": "suspicious"})["events"][0]["handle"]
        os.makedirs(ctx.services.desc_dir, exist_ok=True)
        with open(os.path.join(ctx.services.desc_dir, "main_entrance_1_alert.json"), "wb") as f:
            f.write(b'\xff')
        with self.assertLogs("box.brain.events", level="WARNING"):
            self.assertTrue(describe_event(ctx, {"handle": handle})["ok"])

    def test_malformed_vision_results_are_not_saved_or_assessed_as_normal(self) -> None:
        valid = {"ok": True, "description": "A person.", "quality": "clear", "people": 1,
                 "label": "normal", "why": ""}
        for tool in (describe_event, assess_event):
            for result in ([], {"ok": True}, {**valid, "description": object()},
                           {**valid, "people": float("nan")}, {**valid, "people": "many"}):
                with self.subTest(tool=tool.__name__, result=result):
                    ctx = self.ctx(vision=FakeVision(result))
                    handle = find_events(ctx, {"kind": "quiet"})["events"][0]["handle"]
                    with self.assertLogs("box.brain.tools", level="WARNING") as logs:
                        out = tool(ctx, {"handle": handle})
                    self.assertEqual(len(logs.output), 1)
                    if tool is assess_event:
                        self.assertEqual(out["assessment"], "unavailable")
                        self.assertNotIn("label", out)
                    else:
                        self.assertFalse(out["ok"])
                    self.assertFalse(os.path.exists(ctx.services.desc_dir))
                    json.dumps(out, allow_nan=False)

    def test_assess_rejects_invalid_guard_labels(self) -> None:
        for label in (None, [], "unknown"):
            ctx = self.ctx(vision=FakeVision({"ok": True, "description": "Activity.",
                                             "quality": "clear", "people": 1, "label": label}))
            handle = find_events(ctx, {"kind": "quiet"})["events"][0]["handle"]
            with self.assertLogs("box.brain.tools", level="WARNING"):
                out = assess_event(ctx, {"handle": handle})
            self.assertEqual(out["assessment"], "unavailable")
            self.assertNotIn("label", out)

    def test_injected_vision_or_frame_errors_are_contained(self) -> None:
        for service in ("vision", "clip_frames"):
            ctx = self.ctx()
            handle = find_events(ctx, {"kind": "quiet"})["events"][0]["handle"]
            if service == "vision":
                ctx.services.vision = Mock(look=Mock(side_effect=RuntimeError("offline")))
            else:
                ctx.services.clip_frames = Mock(side_effect=ValueError("bad clip"))
            with self.assertLogs("box.brain.tools", level="WARNING"):
                out = assess_event(ctx, {"handle": handle})
            self.assertEqual(out["assessment"], "unavailable")
            self.assertNotIn("label", out)


if __name__ == "__main__":
    unittest.main()
