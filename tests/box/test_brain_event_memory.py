"""Stage 2b (2026-10-08): the assistant's memory tools - search_events (EventMemAgent-style: the best 3 past events
by the owner's words, a time or a camera) and get_event (one event in full, with its picture and video handles).
The owner could not tell which video the assistant meant; every result says the camera's name, the day and HH:MM,
never an id. Local fakes only."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, List
from unittest import mock

from home_guard_project.box import event_memory as em
from home_guard_project.box.archive import AlertRecord
from home_guard_project.box.brain import profiles
from home_guard_project.box.brain import tools as brain_tools
from home_guard_project.box.brain.memory import ChatState
from home_guard_project.box.brain.receipts import ReceiptBook
from home_guard_project.box.brain.registry import CameraState, HouseSnapshot
from home_guard_project.box.brain.tools import TOOLS, Services, ToolContext, get_event, search_events, send_media
from home_guard_project.box.events import EventBook

NOW = dt.datetime(2026, 10, 8, 18, 0).timestamp()
YESTERDAY_NOON = dt.datetime(2026, 10, 7, 12, 5).timestamp()
TODAY_NOON = dt.datetime(2026, 10, 8, 12, 20).timestamp()
PERGOLA, GATE = "ameer_week_0_1_ch6", "ameer_week_0_1_ch2"


def house() -> HouseSnapshot:
    return HouseSnapshot(now=NOW, mode="assistant", mode_ends=NOW + 3600, mode_started=NOW - 3600, start_hour=0,
                         end_hour=0, cameras=(CameraState(PERGOLA, True, ("פרגולה",), live=True),
                                              CameraState(GATE, True, ("שער",), live=True)))


class Deliver:
    def __init__(self) -> None:
        self.photos: List[Any] = []

    def photo(self, chat_id, path, caption=""):
        self.photos.append(path)
        return {"ok": True, "message_id": len(self.photos)}


class MemoryToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.dir = os.path.join(self.root, "events")
        self.events = EventBook(self.dir)
        memory = em.memory_for(self.dir)
        memory.place = lambda cam: {PERGOLA: "פרגולה", GATE: "Camera 2"}.get(cam, "")
        em.attach(self.events, memory)
        self.deliver = Deliver()
        self.ids = {}
        self.add(PERGOLA, YESTERDAY_NOON, 3, "Three workers carry a ladder onto the pergola.", known="העובדים")
        self.add(GATE, TODAY_NOON, 1, "A white pickup truck parks by the gate.", keyframe=b"jpeg")
        self.add(GATE, TODAY_NOON + 1800, 1, "The white pickup truck drives away from the gate.")

    def add(self, camera, ts, people, summary, known=None, keyframe=None):
        alert_id = f"{camera}_{int(ts)}_alert"
        d = self.events.decide(camera, ts, "suspicious", people, summary, alert_id=alert_id)
        if keyframe:
            em.save_keyframe(self.dir, d.session_id, keyframe)
        if known:
            self.events.mark_known(camera, known, "Ameer", until=ts + 3600, now=ts + 5)
        self.events.tick(ts + 300)
        self.ids[summary] = (d.session_id, alert_id)

    def ctx(self, text: str) -> ToolContext:
        services = Services(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"),
                            feedback_dir=self.root, work_dir=os.path.join(self.root, ".live"), mute=None,
                            deliver=self.deliver, now=lambda: NOW, events=self.events)
        return ToolContext(turn_id="-5:1", chat_id="-5", speaker={"user_id": 1, "name": "Ameer"}, text=text,
                           lang="he", mode="assistant", snapshot=house(), state=ChatState(), services=services,
                           book=ReceiptBook(os.path.join(self.root, ".r"), now=lambda: NOW))

    # -- registration --------------------------------------------------------------------------------------------
    def test_the_tools_are_registered_with_schemas_in_both_modes(self) -> None:
        self.assertIn("search_events", TOOLS)
        self.assertIn("get_event", TOOLS)
        for mode in ("guard", "assistant"):
            for tier in ("big", "fast"):
                names = profiles.tool_names(mode, tier)
                self.assertIn("search_events", names)
                self.assertIn("get_event", names)
        schemas = {s["function"]["name"]: s["function"] for s in profiles.tools_for("assistant")}
        self.assertEqual(schemas["search_events"]["parameters"]["required"], ["query"])
        self.assertIn("only when", schemas["search_events"]["description"])
        self.assertEqual(schemas["get_event"]["parameters"]["required"], ["handle"])
        prompt = profiles.system_prompt("assistant", 14)
        self.assertIn("search_events", prompt)
        self.assertIn("were the workers here yesterday?", prompt)

    # -- search_events -------------------------------------------------------------------------------------------
    def test_a_hebrew_question_finds_the_english_event_with_names_and_times_never_ids(self) -> None:
        ctx = self.ctx("הטנדר הלבן נסע?")
        out = search_events(ctx, {"query": "הטנדר הלבן נסע?"})
        self.assertTrue(out["ok"])
        self.assertEqual([e["camera"] for e in out["events"]], ["שער", "שער"])     # not the workers who "left"
        first = out["events"][0]
        self.assertEqual((first["camera"], first["day"], first["from"]), ("שער", "היום", "12:50"))
        self.assertIn("drives away", first["what"])
        self.assertIn("at שער.", first["what"])                            # today's name, not "Camera 2"
        text = json.dumps(out, ensure_ascii=False)
        self.assertNotIn("ameer_week", text)
        self.assertNotIn(self.ids["A white pickup truck parks by the gate."][0], text)
        self.assertEqual(ctx.state.resolve(first["handle"])["kind"], "memory")

    def test_yesterday_and_the_cameras_name(self) -> None:
        out = search_events(self.ctx("העובדים היו פה אתמול?"), {"query": "העובדים היו פה אתמול?"})
        (event,) = out["events"]
        self.assertEqual((event["camera"], event["day"], event["most_people"]), ("פרגולה", "אתמול", 3))
        self.assertEqual(event["owner_said"], ["העובדים"])
        gate = search_events(self.ctx("מה היה בשער בצהריים?"), {"query": "מה היה בשער בצהריים?"})
        self.assertEqual([e["camera"] for e in gate["events"]], ["שער", "שער"])

    def test_explicit_camera_and_day_arguments(self) -> None:
        out = search_events(self.ctx("x"), {"query": "workers", "camera": "פרגולה", "day": "yesterday"})
        self.assertEqual(out["found"], 1)
        none = search_events(self.ctx("x"), {"query": "workers", "day": "today"})
        self.assertEqual(none["found"], 0)
        self.assertIn("nothing matching was recorded", none["note"])
        bad = search_events(self.ctx("x"), {"query": "workers", "camera": "the moon"})
        self.assertFalse(bad["ok"])

    def test_an_event_still_going_on_is_found(self) -> None:
        self.events.decide(PERGOLA, NOW - 30, "normal", 1, "A gardener waters the plants.")
        out = search_events(self.ctx("הגנן פה?"), {"query": "gardener"})
        self.assertEqual(out["events"][0]["to"][-len("(still going on)"):], "(still going on)")

    def test_without_the_event_book_it_says_so(self) -> None:
        ctx = self.ctx("x")
        ctx.services.events = None
        self.assertFalse(search_events(ctx, {"query": "workers"})["ok"])
        self.assertFalse(get_event(ctx, {"handle": "E1"})["ok"])

    def test_embeddings_are_used_when_the_box_has_them(self) -> None:
        class Embedder:
            def embed(self, texts):
                return [[1.0, 0.0] if "pickup" in t.lower() or "טנדר" in t else [0.0, 1.0] for t in texts]

        ctx = self.ctx("טנדר")
        ctx.services.embedder = Embedder()
        out = search_events(ctx, {"query": "טנדר"})
        self.assertEqual(len(out["events"]), 2)
        self.assertTrue(all("pickup" in e["what"] for e in out["events"]))

    # -- get_event -----------------------------------------------------------------------------------------------
    def test_get_event_gives_the_details_the_picture_and_the_video(self) -> None:
        ctx = self.ctx("תראה לי")
        found = search_events(ctx, {"query": "pickup parks", "day": "today", "time_from": "12:00", "time_to": "12:30"})
        (event,) = found["events"]
        _, alert_id = self.ids["A white pickup truck parks by the gate."]
        record = AlertRecord(alert_id, GATE, TODAY_NOON + 10, "A white pickup truck parks.", "[send_message]",
                             "clip.mp4", ())
        with mock.patch.object(brain_tools, "load_events", return_value=[record]):
            out = get_event(ctx, {"handle": event["handle"]})
        self.assertTrue(out["ok"])
        self.assertEqual(out["seen"][0]["what"], "A white pickup truck parks by the gate.")
        self.assertIn("after the previous one", out["change_to_next"])          # the drive-away event follows
        self.assertEqual(ctx.state.resolve(out["video"])["ref"], alert_id)
        self.assertEqual(ctx.state.resolve(out["picture"])["kind"], "photo")
        self.assertNotIn("ameer_week", json.dumps(out, ensure_ascii=False))
        sent = send_media(ctx, {"handle": out["picture"]})
        self.assertTrue(sent["ok"])
        self.assertEqual(len(self.deliver.photos), 1)

    def test_get_event_without_a_picture_or_video_on_the_box(self) -> None:
        ctx = self.ctx("x")
        (event,) = search_events(ctx, {"query": "workers"})["events"]
        with mock.patch.object(brain_tools, "load_events", return_value=[]):
            out = get_event(ctx, {"handle": event["handle"]})
        self.assertTrue(out["ok"])
        self.assertNotIn("picture", out)
        self.assertNotIn("video", out)
        self.assertFalse(get_event(ctx, {"handle": "E99"})["ok"])


if __name__ == "__main__":
    unittest.main()
