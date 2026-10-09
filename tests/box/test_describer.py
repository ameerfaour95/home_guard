"""The alert message v2 (owner, 2026-10-09: "explain what the person is doing and how he looks, with P1 P2"):
describer.py draws the tracker's ids on copies of the Eye's frames, asks a SEPARATE model who is who, guards the
answer against invented actions, and composes the professional message; the worker swaps it into the held alert,
and on any failure the old message goes out. Local fakes only: no model, no Telegram."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import mock

import numpy as np

import test_eye_entities as te

from home_guard_project.box import camera_names
from home_guard_project.box import describer as ds
from home_guard_project.box import inference as inf
from home_guard_project.box.events import EventBook
from home_guard_project.box.telegram_agent import OwnerAssistant

CAM = "ameer_week_0_1_ch6"
T0 = datetime(2026, 10, 9, 9, 43).timestamp()
SUMMARY = "A person in a hat walks past a man who is leaning into an open car door."
WHY = "A person looks at another person near an open car."


def track(track_id, kind, boxes, first=None):
    """*boxes*: [(ts, (x1, y1, x2, y2) normalised)]."""
    return {"id": track_id, "kind": kind, "first_seen": float(first if first is not None else boxes[0][0]),
            "last_seen": float(boxes[-1][0]), "boxes": [{"ts": t, "box": list(b)} for t, b in boxes]}


def reply(content: str, model: str = "qwen/qwen3.5-9b", delay: float = 0.0):
    class Completions:
        def __init__(self):
            self.kwargs: List[Dict[str, Any]] = []

        def create(self, **kwargs):
            self.kwargs.append(kwargs)
            if delay:
                time.sleep(delay)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], model=model,
                                   usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=80))

    completions = Completions()
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


ANSWER = {"scene": "Two men by a white car with an open door in the driveway.",
          "entities": [{"id": "P1", "appearance": "man, black hat, white shirt", "action": "walks past the car"},
                       {"id": "P2", "appearance": "man, dark shirt", "action": "leans into the car"},
                       {"id": "CAR1", "appearance": "white pickup truck", "action": "parked, door open"},
                       {"id": "P9", "appearance": "ghost", "action": "not drawn"}],
          "reason": "Someone looks at a man by an open car"}


class GeometryTest(unittest.TestCase):
    def test_box_at_interpolates_between_close_looks_and_stops_at_gaps(self):
        boxes = [{"ts": 10.0, "box": [0.0, 0.0, 0.2, 0.2]}, {"ts": 11.0, "box": [0.2, 0.0, 0.4, 0.2]},
                 {"ts": 20.0, "box": [0.8, 0.0, 1.0, 0.2]}]
        self.assertEqual(tuple(round(v, 3) for v in ds.box_at(boxes, 10.5)), (0.1, 0.0, 0.3, 0.2))
        self.assertEqual(ds.box_at(boxes, 11.8), (0.2, 0.0, 0.4, 0.2))        # nearest, within 1 s
        self.assertIsNone(ds.box_at(boxes, 15.0))                              # a 9 s hole: not there
        self.assertIsNone(ds.box_at(boxes, 8.5))
        self.assertIsNone(ds.box_at(boxes, None))

    def test_ids_keep_the_events_numbers_and_number_the_rest(self):
        tracks = [track(5, "person", [(1, (0, 0, 1, 1))]), track(7, "person", [(2, (0, 0, 1, 1))]),
                  track(9, "vehicle", [(0, (0, 0, 1, 1))]), track(3, "animal", [(0, (0, 0, 1, 1))])]
        mapped = {f"7@{2.0:.3f}": "P1"}
        ids = ds.assign_ids(tracks, mapped)
        self.assertEqual(ids, {"9@0.000": "CAR1", "5@1.000": "P2", "7@2.000": "P1"})

    def test_a_box_lands_in_the_crop_pixels_of_each_frame(self):
        # source 1000 x 500; frame 0 cut from (100, 50)-(500, 450) and sent at 200 x 200
        tracks = [track(1, "person", [(10.0, (0.2, 0.2, 0.3, 0.6))])]
        ids = {"1@10.000": "P1"}
        per = ds.place_marks(tracks, ids, [10.0, 30.0], [(100, 50, 500, 450), None], [(200, 200), (100, 50)],
                             (1000, 500))
        self.assertEqual(per[0], [("P1", "person", (50, 25, 100, 125))])
        self.assertEqual(per[1], [])                                           # 20 s later: not there

    def test_whole_frames_scale_the_normalised_box(self):
        tracks = [track(1, "person", [(10.0, (0.5, 0.5, 1.0, 1.0))])]
        per = ds.place_marks(tracks, {"1@10.000": "P1"}, [10.0], [None], [(64, 48)], None)
        self.assertEqual(per[0], [("P1", "person", (32, 24, 63, 47))])

    def test_only_the_vehicles_closest_to_people_are_kept(self):
        here = [("P1", "person", (0, 0, 10, 10)), ("CAR1", "vehicle", (12, 0, 20, 10)),
                ("CAR2", "vehicle", (500, 0, 520, 10)), ("CAR3", "vehicle", (30, 0, 40, 10))]
        kept = ds.keep_relevant([here], max_vehicles=2)
        self.assertEqual([m[0] for m in kept[0]], ["P1", "CAR1", "CAR3"])

    def test_pick_frames_spreads_over_the_marked_frames(self):
        per = [[]] * 2 + [[1]] * 14 + [[]] * 3
        picked = ds.pick_frames(per, 8)
        self.assertEqual(len(picked), 8)
        self.assertEqual((picked[0], picked[-1]), (2, 15))
        self.assertEqual(ds.pick_frames([[]] * 3, 8), [0, 1, 2])

    def test_frame_times_follow_the_clip_clock(self):
        self.assertEqual(ds.sent_frame_times([0, 10], {"start": 100.0, "end": 110.0, "frames": 11}), [100.0, 110.0])
        self.assertEqual(ds.sent_frame_times([3], {}), [None])


class AnswerTest(unittest.TestCase):
    IDS = {"P1": "person", "P2": "person", "CAR1": "vehicle"}

    def test_parse_drops_unknown_ids_and_orders_people_first(self):
        parsed = ds.parse("```json\n" + json.dumps(ANSWER) + "\n```", self.IDS)
        self.assertEqual([e["id"] for e in parsed["entities"]], ["P1", "P2", "CAR1"])
        self.assertEqual(parsed["scene"], "Two men by a white car with an open door in the driveway")
        with self.assertRaises(ValueError):
            ds.parse("no json here", self.IDS)

    def test_the_guard_keeps_what_the_alert_backs(self):
        checked, dropped = ds.guard(ds.parse(json.dumps(ANSWER), self.IDS), SUMMARY, WHY)
        self.assertEqual(dropped, [])
        self.assertEqual(len(checked["entities"]), 3)

    def test_the_guard_drops_invented_weapons_objects_and_acts(self):
        answer = {"scene": "A man with a knife by the car", "reason": "", "entities": [
            {"id": "P1", "appearance": "man holding a knife", "action": "walks to the gate"},
            {"id": "P2", "appearance": "man, dark shirt", "action": "steals a bag from the car"},
            {"id": "P3", "appearance": "man with a backpack", "action": "puts the backpack down"},
            {"id": "P4", "appearance": "", "action": "carries a ladder"}]}
        checked, dropped = ds.guard(answer, SUMMARY, WHY)
        self.assertEqual(checked["scene"], "")
        by = {e["id"]: e for e in checked["entities"]}
        self.assertEqual(by["P1"]["appearance"], "")                     # the weapon word is gone ...
        self.assertEqual(by["P1"]["action"], "walks to the gate")          # ... the plain action stays
        self.assertEqual((by["P2"]["appearance"], by["P2"]["action"]), ("man, dark shirt", ""))   # appearance only
        self.assertEqual(by["P3"]["action"], "puts the backpack down")    # its own carried object
        self.assertNotIn("P4", by)                                          # nothing left to say
        self.assertEqual(len(dropped), 4)

    def test_a_weapon_the_eye_saw_may_be_named(self):
        answer = {"scene": "", "reason": "", "entities": [{"id": "P1", "appearance": "man with a knife",
                                                          "action": "points the knife at the door"}]}
        checked, dropped = ds.guard(answer, "A man holds a knife at the door.", "")
        self.assertEqual((checked["entities"][0]["action"], dropped), ("points the knife at the door", []))

    def test_the_message_layout(self):
        entities = [{"id": "P1", "appearance": "גבר, כובע שחור, חולצה לבנה", "action": "עומד ליד העציץ ומחזיק חפץ"},
                    {"id": "P2", "appearance": "גבר, חולצה כהה", "action": "רוכן לתוך הרכב"}]
        text = ds.compose("suspicious", "כניסה ראשית", "09:43", "שני אנשים ליד רכב לבן עם דלת פתוחה בחניה",
                          entities, "אדם עם פנים מוסתרות ליד רכב פתוח", "he")
        self.assertEqual(text.split("\n"), [
            "🟡 חשוד · כניסה ראשית · 09:43",
            "מה קורה: שני אנשים ליד רכב לבן עם דלת פתוחה בחניה.",
            "P1 · גבר, כובע שחור, חולצה לבנה: עומד ליד העציץ ומחזיק חפץ.",
            "P2 · גבר, חולצה כהה: רוכן לתוך הרכב.",
            "למה הודעתי: אדם עם פנים מוסתרות ליד רכב פתוח."])

    def test_unsure_ids_give_the_appearance_only_and_more_than_four_are_counted(self):
        many = [{"id": f"P{i}", "appearance": f"גבר בחולצה {i}", "action": ""} for i in range(1, 7)]
        text = ds.compose("normal", "פרגולה", "11:12", "", many, "", "he").split("\n")
        self.assertEqual(text[0], "🟢 נראה תקין · פרגולה · 11:12")
        self.assertEqual(text[1:5], [f"P{i} · גבר בחולצה {i}" for i in range(1, 5)])
        self.assertEqual(text[5], "ועוד 2")
        self.assertEqual(len(text), 6)                                       # no reason line without a reason
        same = [{"id": f"P{i}", "appearance": "אדם בבגדים כהים", "action": "הולך לאורך הקיר"} for i in (1, 2, 3)]
        self.assertEqual(ds.compose("normal", "מצלמה 1", "11:26", "", same, "", "he").split("\n")[1],
                         "P1, P2, P3 · אדם בבגדים כהים: הולך לאורך הקיר.")
        english = ds.compose("escalation", "Gate", "02:10", "A man climbs the gate", [], "Climbing in", "en")
        self.assertEqual(english.split("\n"), ["🔴 ESCALATION · Gate · 02:10", "What's happening: A man climbs the gate.",
                                               "Why I told you: Climbing in."])

    def test_the_translation_goes_field_by_field_and_failure_is_none(self):
        class Messenger:
            def __init__(self, answer):
                self.answer, self.seen = answer, None

            def translate(self, fields, lang, keep=(), timeout=None):
                self.seen = (dict(fields), tuple(keep))
                return self.answer(fields) if self.answer else None

        checked, _ = ds.guard(ds.parse(json.dumps(ANSWER), self.IDS), SUMMARY, WHY)
        m = Messenger(lambda f: {k: f"ע {v}" for k, v in f.items()})
        told = ds.to_owner_language(checked, "he", m, keep=("P1", "P2", "CAR1"))
        self.assertEqual(told["entities"][1], {"id": "P2", "appearance": "ע man, dark shirt",
                                               "action": "ע leans into the car"})
        self.assertIn("P1.action", m.seen[0])
        self.assertIsNone(ds.to_owner_language(checked, "he", Messenger(None)))
        self.assertEqual(ds.to_owner_language(checked, "en", None)["scene"], checked["scene"])


class BrokenEndingTest(unittest.TestCase):
    """2026-10-09 18:16 pergola: "למה הודעתי: ... עם שק לבן גדול. ה." and 18:22 ch6 "... ליד אבן" (from "... next to a
    stone wall"): the describer's reason was cut at 14 words mid-thought, and the translator kept the stump."""

    RAW_1816 = ("A person appears to walk across the driveway carrying a large white bag. The person's face is obscured "
                "by a hood or mask.")
    RAW_1822 = "A person appears to be walking along a paved path next to a stone wall and a black fence."

    def test_a_long_reason_is_kept_whole_or_cut_at_a_full_stop(self):
        ids = {"P1": "person"}
        for raw in (self.RAW_1816, self.RAW_1822):
            with self.subTest(raw=raw):
                parsed = ds.parse(json.dumps({"scene": "", "entities": [], "reason": raw}), ids)
                self.assertEqual(parsed["reason"], raw.rstrip("."))
        too_long = " ".join(["word"] * 30) + ". " + "x " * 5
        self.assertEqual(ds.parse(json.dumps({"reason": too_long}), ids)["reason"], "")       # no stop inside: none
        cut = ds.parse(json.dumps({"reason": "A man walks by. " + " ".join(["more"] * 30)}), ids)["reason"]
        self.assertEqual(cut, "A man walks by")

    def test_a_cut_never_ends_on_a_dangling_word(self):
        self.assertEqual(ds.cut_words(self.RAW_1822, 14), "A person appears to be walking along a paved path next to a stone")
        self.assertEqual(ds.cut_words("A person walks along the long path next to the big stone wall", 12),
                         "A person walks along the long path next to the big stone")
        self.assertEqual(ds.cut_words("A man stands next to the", 4), "A man stands")
        self.assertEqual(ds.cut_words(self.RAW_1816, 14),
                         "A person appears to walk across the driveway carrying a large white bag")

    def test_the_translator_s_stray_letters_are_dropped(self):
        self.assertEqual(ds.tidy_translation("אדם נראה הולך על שביל הכניסה עם שק לבן גדול. ה."),
                         "אדם נראה הולך על שביל הכניסה עם שק לבן גדול.")
        self.assertEqual(ds.tidy_translation("אדם נראה הולך על שביל הכניסה עם שק לבן גדול. ה"),
                         "אדם נראה הולך על שביל הכניסה עם שק לבן גדול.")
        self.assertEqual(ds.tidy_translation("גבר עומד ליד השער. הוא מסתכל."), "גבר עומד ליד השער. הוא מסתכל.")

    def test_a_broken_translated_reason_gives_way_to_the_fallback(self):
        class Messenger:
            def __init__(self, reason):
                self.reason = reason

            def translate(self, fields, lang, keep=(), timeout=None):
                return {k: (self.reason if k == "reason" else f"ע {v}") for k, v in fields.items()}

        answer = {"scene": "A person walks along a paved path", "entities": [],
                  "reason": "A person appears to be walking along a paved path next to a stone wall"}
        self.assertEqual(ds.to_owner_language(answer, "he", Messenger("אדם נראה הולך בשביל מרוצף ליד קיר אבן"))["reason"],
                         "אדם נראה הולך בשביל מרוצף ליד קיר אבן")
        for broken in ("אדם נראה הולך בשביל ה", "אדם הולך", "אדם נראה הולך בשביל מרוצף ליד קיר אבן. ה."):
            with self.subTest(broken=broken):
                told = ds.to_owner_language(answer, "he", Messenger(broken))
                want = "" if broken != "אדם נראה הולך בשביל מרוצף ליד קיר אבן. ה." else "אדם נראה הולך בשביל מרוצף ליד קיר אבן."
                self.assertEqual(told["reason"], want)
                self.assertEqual(told["scene"], "ע A person walks along a paved path")


class TranslateFieldsTest(unittest.TestCase):
    def test_every_field_in_one_call_with_its_own_budget_and_none_on_failure(self):
        from home_guard_project.box.messenger import Messenger

        good = json.dumps({"scene": "שני אנשים ליד רכב", "P1.action": "עובר ליד הרכב"}, ensure_ascii=False)
        client, calls = reply(good)
        told = Messenger(client).translate({"scene": "Two men by a car", "P1.action": "walks past the car"}, "he",
                                           keep=("P1",), timeout=6.0)
        self.assertEqual(told, {"scene": "שני אנשים ליד רכב", "P1.action": "עובר ליד הרכב"})
        self.assertAlmostEqual(calls.kwargs[0]["timeout"], 6.0, places=1)
        self.assertIn("SAME keys", calls.kwargs[0]["messages"][0]["content"])
        english, _ = reply(json.dumps({"scene": "Two men", "P1.action": "walks"}))
        self.assertIsNone(Messenger(english).translate({"scene": "Two men", "P1.action": "walks"}, "he"))
        self.assertIsNone(Messenger(None).translate({"scene": "Two men"}, "he"))
        self.assertEqual(Messenger(None).translate({"scene": "Two men"}, "en"), {"scene": "Two men"})

    def test_a_slow_or_broken_first_call_is_raced_by_a_second(self):
        from home_guard_project.box.messenger import Messenger

        good = json.dumps({"scene": "שני אנשים"}, ensure_ascii=False)

        def client(first_delay, first_content):
            n = {"calls": 0}

            def create(**kwargs):
                n["calls"] += 1
                n.setdefault("models", []).append(kwargs["model"])
                if n["calls"] == 1:
                    time.sleep(first_delay)
                    content = first_content
                else:
                    content = good
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                                       usage=None, model="m")

            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), n

        slow, n = client(3.0, good)
        started = time.monotonic()
        self.assertEqual(Messenger(slow).translate({"scene": "Two men"}, "he", timeout=5.0, hedge_after=0.2),
                         {"scene": "שני אנשים"})
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual(n["models"], ["google/gemini-3.1-flash-lite", "google/gemini-2.5-flash-lite"])   # another model
        broken, n = client(0.0, "not json")
        self.assertEqual(Messenger(broken).translate({"scene": "Two men"}, "he", timeout=5.0, hedge_after=2.0),
                         {"scene": "שני אנשים"})
        self.assertEqual(n["calls"], 2)
        never, _ = client(3.0, "not json")
        self.assertIsNone(Messenger(never).translate({"scene": "Two men"}, "he", timeout=0.5, hedge_after=10.0))


class DescribeTest(unittest.TestCase):
    def setUp(self):
        self.frames = [np.full((120, 160, 3), 40, np.uint8) for _ in range(12)]
        self.copy = [f.copy() for f in self.frames]
        self.tracks = [track(1, "person", [(100.0 + i, (0.1, 0.2, 0.3, 0.9)) for i in range(0, 11)]),
                       track(2, "person", [(103.0 + i, (0.5, 0.2, 0.7, 0.9)) for i in range(0, 6)]),
                       track(3, "vehicle", [(100.0, (0.6, 0.4, 0.95, 0.95)), (110.0, (0.6, 0.4, 0.95, 0.95))])]
        self.mapped = {"1@100.000": "P1", "2@103.000": "P2"}
        self.clock = {"start": 100.0, "end": 111.0, "frames": 12, "size": [160, 120]}

    def run_describe(self, client, timeout=12.0):
        d = ds.Describer(client, "qwen/qwen3.5-9b", timeout)
        return ds.describe(d, self.frames, list(range(12)), [None] * 12, self.clock, self.tracks, self.mapped,
                           SUMMARY, WHY, (), None, [float(i) for i in range(12)])

    def test_a_good_answer_is_recorded_with_what_was_sent(self):
        client, calls = reply(json.dumps(ANSWER))
        record = self.run_describe(client)
        self.assertTrue(record["ok"], record)
        self.assertEqual(record["ids"], {"P1": "person", "P2": "person", "CAR1": "vehicle"})
        self.assertEqual(record["frames"], 8)                       # at most MAX_FRAMES of the Eye's frames
        self.assertEqual(len(record["frames_sha"]), 40)
        self.assertEqual([e["id"] for e in record["answer"]["entities"]], ["P1", "P2", "CAR1"])
        self.assertIn("leaning into an open car door", record["prompt"])
        sent = calls.kwargs[0]
        self.assertEqual(sent["model"], "qwen/qwen3.5-9b")
        self.assertEqual(len([c for c in sent["messages"][0]["content"] if c["type"] == "image_url"]), 8)
        for a, b in zip(self.frames, self.copy):                    # the Eye's frames are never drawn on
            self.assertTrue(np.array_equal(a, b))

    def test_a_slow_model_is_a_timeout_not_a_wait(self):
        client, _ = reply(json.dumps(ANSWER), delay=2.0)
        started = time.monotonic()
        record = self.run_describe(client, timeout=0.3)
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertFalse(record["ok"])
        self.assertIn("TimeoutError", record["error"])

    def test_no_client_bad_json_or_nothing_usable_is_a_failure(self):
        self.assertFalse(self.run_describe(None)["ok"])
        self.assertIn("ValueError", self.run_describe(reply("I see two men.")[0])["error"])
        empty = self.run_describe(reply(json.dumps({"scene": "", "entities": [], "reason": ""}))[0])
        self.assertEqual((empty["ok"], empty["error"]), (False, "nothing usable in the answer"))

    def test_settings(self):
        self.assertEqual(ds.settings_of({}), (True, "openrouter", "qwen/qwen3.5-9b", 12.0))
        self.assertFalse(ds.settings_of({"alert_describer": "off"})[0])
        self.assertEqual(ds.settings_of({"describer_timeout_sec": 99})[3], 30.0)


class HeldTextTest(unittest.TestCase):
    def test_update_held_replaces_only_a_waiting_alert(self):
        assistant = OwnerAssistant.__new__(OwnerAssistant)
        assistant._held, assistant._held_lock = {"a1": {"text": "old"}}, threading.Lock()
        self.assertTrue(assistant.update_held("a1", "new"))
        self.assertEqual(assistant._held["a1"]["text"], "new")
        self.assertFalse(assistant.update_held("a2", "new"))
        self.assertFalse(assistant.update_held("a1", "  "))


class HoldingAssistant:
    """send_alert holds the alert like OwnerAssistant does; update_held swaps its text."""

    def __init__(self, hold=True):
        self.held: Dict[str, str] = {}
        self.sent: List[Dict[str, Any]] = []
        self.hold = hold
        self.index = SimpleNamespace(messages=lambda alert_id: [])

    def is_muted(self, camera):
        return False

    def send_alert(self, alert, text, image=None, silent=False, lang="en", reply_to=None):
        self.sent.append({"text": text})
        if self.hold:
            self.held[alert["alert_id"]] = text
            return {"sent": True, "held": "waiting for the video"}
        return {"sent": True, "results": [{"chat_id": "-5", "ok": True, "message_id": 1}]}

    def update_held(self, alert_id, text):
        if alert_id not in self.held:
            return False
        self.held[alert_id] = text
        return True

    def remind_if_silent(self, alert, text, lang="en"):
        pass


class WorkerTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.book = EventBook(self.dir)
        self.stack = ExitStack()
        for patch in (mock.patch.object(inf, "EVENTS", self.book),
                      mock.patch.object(inf, "KNOWN_CAMERAS", (CAM,)),
                      mock.patch.object(inf, "FACTS_PROVIDER", mock.Mock(return_value=[])),
                      mock.patch.object(inf, "owner_language", return_value="he"),
                      mock.patch.object(inf, "frame_to_jpeg_bytes", return_value=b"jpg"),
                      mock.patch.object(inf, "_jpegs", return_value=[]),
                      mock.patch.object(camera_names, "_load", side_effect=lambda a: {CAM: ["כניסה ראשית"]}
                                        if a is None else a)):
            self.stack.enter_context(patch)
        inf._SOFTENED_DAYS.clear()

    def tearDown(self):
        self.stack.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def work(self, assistant, describe_answer, translate=True, settings=None):
        answer = dict(summary=SUMMARY, label="suspicious", raw_label="suspicious", applied_fact_id="",
                      serious_behaviour=False, people=2, vehicle_moving=False, animals=0, why=WHY, summary_owner="")

        class Backend:
            model_name, last_prompt = "fake", "prompt"

            def analyze(self, frames, camera_name, t_sec, start_hour, end_hour, owner_language="en", **kwargs):
                return json.dumps(answer), answer

        job = inf.AlertJob(camera=CAM, stem=f"{CAM}_{int(T0)}_alert", ts=T0, labels=["person"],
                           input_meta={"vlm_input": "whole"})
        frames = [np.full((90, 160, 3), 50, np.uint8) for _ in range(6)]
        job.model_input = SimpleNamespace(frames=frames, frame_indices=list(range(6)), crops=[None] * 6,
                                          times=[float(i) for i in range(6)], record=lambda: {"vlm_input": "whole"})
        job.input_clock = {"start": T0 - 3, "end": T0 + 2, "frames": 6, "size": [160, 90]}
        # The event's entities come from the same tracks (P1 = track 4, P2 = track 5).
        job.tracker_entities = [te.trk(4, T0 - 3, T0 + 2, start=(0.2, 0.9)), te.trk(5, T0 - 3, T0 + 2, start=(0.6, 0.9))]
        job.tracker_since = T0 - 4
        job.track_boxes = {"tracks": [track(4, "person", [(T0 - 3 + i, (0.1, 0.1, 0.3, 0.9)) for i in range(6)]),
                                      track(5, "person", [(T0 - 3 + i, (0.5, 0.1, 0.7, 0.9)) for i in range(6)])]}
        client, _ = reply(describe_answer) if describe_answer is not None else (None, None)

        class Messenger:
            def translate(self, fields, lang, keep=(), timeout=None):
                return {k: f"[{v}]" for k, v in fields.items()} if translate else None

            def to_owner(self, texts, lang, keep=()):
                return {"summary": "שני אנשים ליד רכב", "why": "אדם מסתכל", "source": "translator"}

        with mock.patch.object(ds, "describer_for", return_value=ds.Describer(client, "qwen/qwen3.5-9b", 5.0)), \
                mock.patch.object(inf.messenger, "messenger_for", return_value=Messenger()):
            inf._worker(Backend(), {"alert_channel": "telegram", **(settings or {})}, {}, inf.AlertSettings(), CAM,
                        frames, assistant, job)
        return job

    def test_the_held_alert_gets_the_new_message_and_the_meta_keeps_the_record(self):
        assistant = HoldingAssistant()
        two = {"scene": "Two men by a car", "reason": "Someone looks at a man by an open car",
               "entities": [{"id": "P1", "appearance": "man, black hat", "action": "walks past the car"},
                            {"id": "P2", "appearance": "man, dark shirt", "action": ""}]}
        job = self.work(assistant, json.dumps(two))
        old = assistant.sent[0]["text"]
        self.assertTrue(old.startswith("🟡 חשוד · כניסה ראשית\n"))             # what was held first
        new = assistant.held[job.stem]
        self.assertEqual(new.split("\n"), [
            "🟡 חשוד · כניסה ראשית · 09:43",
            "מה קורה: [Two men by a car].",
            "P1 · [man, black hat]: [walks past the car].",
            "P2 · [man, dark shirt]",
            "למה הודעתי: [Someone looks at a man by an open car]."])
        record = job.input_meta["describer"]
        self.assertTrue(record["used"])
        self.assertEqual(record["model"], "qwen/qwen3.5-9b")
        self.assertIn("frames_sha", record)
        self.assertIn("seconds", record)
        self.assertEqual(record["answer"]["entities"][0]["id"], "P1")
        self.assertEqual(job.alert["dispatch"]["telegram"]["telegram"]["held"], "waiting for the video")
        entity = {e["id"]: e for e in self.book.session_of_alert(job.stem)["entities"]}["P1"]
        self.assertEqual(entity["notes"][-1]["text"], "[walks past the car]")   # the story can continue from it

    def test_any_failure_keeps_the_old_message(self):
        for answer, translate in (("not json", True), (None, True), (json.dumps(ANSWER), False)):
            self.book = EventBook(tempfile.mkdtemp(dir=self.dir))
            self.stack.enter_context(mock.patch.object(inf, "EVENTS", self.book))
            assistant = HoldingAssistant()
            job = self.work(assistant, answer, translate=translate)
            self.assertEqual(assistant.held[job.stem], assistant.sent[0]["text"])
            self.assertFalse(job.input_meta["describer"]["used"])
            self.assertTrue(job.input_meta["describer"]["error"])

    def test_off_or_not_held_never_asks(self):
        assistant = HoldingAssistant()
        job = self.work(assistant, json.dumps(ANSWER), settings={"alert_describer": "off"})
        self.assertNotIn("describer", job.input_meta)
        self.stack.enter_context(mock.patch.object(inf, "EVENTS", EventBook(tempfile.mkdtemp(dir=self.dir))))
        assistant = HoldingAssistant(hold=False)
        job = self.work(assistant, json.dumps(ANSWER))
        self.assertNotIn("describer", job.input_meta)


if __name__ == "__main__":
    unittest.main()
