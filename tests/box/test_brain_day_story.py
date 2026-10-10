# tests/box/test_brain_day_story.py
"""The day's story (brain/day_story.py), owner 2026-10-10: "a summary of the day tells what happened - at 7:00 a
person with a red hat came, knocked and left... not alerts and numbers"."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import tempfile
import unittest
from typing import Any, Dict, List

from home_guard_project.box import usage_ledger
from home_guard_project.box.brain import day_story as ds
from home_guard_project.box.brain.agent import OwnerAgentV2
from home_guard_project.box.brain.memory import ChatMemory, ChatState
from home_guard_project.box.brain.models import ModelMessage
from home_guard_project.box.brain.receipts import DONE, ReceiptBook
from home_guard_project.box.brain.registry import CameraState, HouseSnapshot
from home_guard_project.box.brain.tools import Services
from home_guard_project.box.events import EventBook

DAY = dt.date(2026, 10, 10)
NOW = dt.datetime.combine(DAY, dt.time(17, 50)).timestamp()
SITE = "ameer_v2"
NAMES = {1: "", 2: "חניה", 3: "פרגולה", 5: "", 6: "כניסה ראשית", 8: "שער"}


def at(hhmm: str) -> float:
    h, m = map(int, hhmm.split(":"))
    return dt.datetime.combine(DAY, dt.time(h, m)).timestamp()


def snapshot() -> HouseSnapshot:
    return HouseSnapshot(now=NOW, mode="guard", mode_ends=None, mode_started=None, start_hour=22, end_hour=6,
                         cameras=tuple(CameraState(f"{SITE}_ch{n}", True, (name,) if name else ())
                                       for n, name in NAMES.items()))


def record(eid: str, ch: int, start: str, summary: str, people: int = 1, minutes: float = 1.0,
           level: str = "normal", site: str = SITE) -> Dict[str, Any]:
    t0 = at(start)
    camera = f"{site}_ch{ch}"
    return {"event_id": eid, "camera": camera, "camera_name": "", "start": t0, "end": t0 + minutes * 60,
            "parent": "", "labels": [level], "level": level, "people_max": people, "reported": False,
            "reported_level": "none", "owner_known": [], "alert_ids": [f"{camera}_{int(t0)}_alert"],
            "keyframe": "", "outcome": "left",
            "observations": [{"ts": t0, "label": level, "people": people, "summary": summary}],
            "caption": summary, "entities": []}


class Scripted:
    """A story model: one answer, and what it was asked."""

    def __init__(self, content: str) -> None:
        self.content, self.calls, self.agents = content, [], []
        self.model_name = "scripted"

    def chat(self, messages, tools, tool_choice=None):
        self.calls.append(messages)
        self.agents.append(usage_ledger._SCOPE.get().get("agent"))
        return ModelMessage(content=self.content, usage=(640, 170))


GOOD = "\n".join([
    "[1] 07:02 · כניסה ראשית: אדם עם כובע אדום ניגש לדלת, דפק ויצא אחרי דקה.",
    "[2] 07:40–17:50 · פרגולה: העובדים עבדו כאן, כרגיל.",       # the example's span: code writes the real one
    "[3] 10:15 · חניה: אישה יצאה מהבית, נכנסה לרכב הלבן ויצאה מהחצר.",
    "היום נרשמו 12 התרעות, מתוכן 2 חשודות.",
    "[4] 14:03 · שער: כנראה שליח, השאיר חבילה ליד הדלת.",
    "[5] 16:40 · חניה: משהו שכדאי לראות: אדם בקפוצ'ון כהה ניסה את ידית הדלת האחורית.",
    "Have a nice evening!",
])


class DayStoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        undescribed = dict(record("n1", 1, "02:45", "a person or vehicle was detected", people=0),
                           entities=[{"id": "P1", "kind": "person"}])
        rows = [
            undescribed,                                                                   # no AI description
            record("n2", 5, "06:30", "A bird lands on the lamp next to the door.", people=0),   # a bird, the lamp
            record("n3", 5, "12:00", "No special activity.", people=0),
            record("e1", 6, "07:02", "A person in a red hat walks to the front door, knocks and leaves."),
            # the workers, under today's ids; the owner's mark was saved under the site's old name (same channel)
            record("w1", 3, "07:40", "Two workers in helmets carry boards in the pergola.", people=2, minutes=20),
            record("w2", 3, "10:00", "Two workers cut wood near the pergola.", people=2, minutes=5),
            record("w3", 3, "17:45", "A worker sweeps the floor.", minutes=2),
            record("e2", 2, "10:15", "A woman walks out of the house to a white car and gets in."),
            record("e3", 8, "10:16", "A white car drives out of the gate.", people=0),
            record("e4", 8, "14:03", "A man in a blue uniform leaves a package at the door and walks away."),
            record("e5", 2, "16:40", "A person in a dark hoodie tries the back door handle and looks around.",
                   level="suspicious"),
        ]
        with open(os.path.join(self.dir, "events_archive.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sessions = [{"id": "e3", "camera": f"{SITE}_ch8", "opened": at("10:16"), "closed": at("10:17"),
                     "observations": [{"ts": at("10:16"), "summary": "A white car drives out of the gate.",
                                       "note": "רכב לבן יוצא מהשער."}],
                     "cross_seen": {"camera": f"{SITE}_ch2", "session": "e2", "mode": "shadow"}},
                    {"id": "e1", "camera": f"{SITE}_ch6", "opened": at("07:02"), "closed": at("07:03"),
                     "observations": [{"ts": at("07:02"), "summary": "x",
                                       "note": "אדם עם כובע אדום ניגש לדלת הכניסה, דופק והולך."}]}]
        with open(os.path.join(self.dir, "events.jsonl"), "w", encoding="utf-8") as f:
            for s in sessions:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        with open(os.path.join(self.dir, "known.json"), "w", encoding="utf-8") as f:
            json.dump([{"text": "העובדים של הפרגולה", "by": "owner", "at": NOW - 2 * 86400, "until": NOW + 5 * 86400,
                        "camera": "ameer_week_0_1_ch3", "id": "k1", "daily_from": "07:00", "daily_to": "18:00"}],
                      f, ensure_ascii=False)
        self.book = EventBook(self.dir)
        self.state = ChatState()

    def tell(self, model=None, words="תן לי סיכום יום") -> ds.Story:
        return ds.tell(self.book, None, snapshot(), words, NOW, model=model, add_handle=self.state.add_handle)

    # -- what goes in ---------------------------------------------------------------------------------------------
    def test_noise_is_dropped(self) -> None:
        records, sessions = ds.gather(self.book, at("00:00"), NOW)
        self.assertEqual(len(records), 11)
        kept = [r["event_id"] for r in records if not ds.is_noise(r)]
        self.assertNotIn("n1", kept)        # "a person or vehicle was detected": no AI description
        self.assertNotIn("n2", kept)        # a bird on the lamp, nobody there
        self.assertNotIn("n3", kept)        # "no special activity"
        self.assertIn("e3", kept)           # a car with nobody in sight is a story

    def test_known_activity_collapses_into_one_line_and_cross_camera_sessions_merge(self) -> None:
        records, sessions = ds.gather(self.book, at("00:00"), NOW)
        eps = ds.episodes_of([r for r in records if not ds.is_noise(r)], sessions)
        ds.explain(eps, ds.marks_of(self.book))
        rows = ds.rows_of(eps, snapshot())
        known = [r for r in rows if r.known]
        self.assertEqual(len(known), 1)
        self.assertEqual((known[0].known, known[0].places), ("העובדים של הפרגולה", ["פרגולה"]))
        self.assertEqual((ds._clock(known[0].start), ds._clock(known[0].end)), ("07:40", "17:47"))
        woman = next(r for r in rows if r.episode and "woman" in " ".join(r.episode.summaries))
        self.assertEqual(woman.places, ["חניה → שער"])          # the woman and the car leaving: one story
        self.assertEqual([r.n for r in rows], [1, 2, 3, 4, 5])
        self.assertIn("CHECK", rows[-1].text)
        prompt = "\n".join(m["content"] for m in ds._prompt(rows, "היום"))
        self.assertLess(len(prompt), 9000)                      # about 3k tokens at most
        self.assertNotIn("detected", prompt)

    # -- what goes out --------------------------------------------------------------------------------------------
    def test_one_call_a_human_story_in_hebrew_with_no_stats(self) -> None:
        model = Scripted(GOOD)
        story = self.tell(model)
        self.assertEqual(len(model.calls), 1)                   # one model call per summary
        self.assertEqual(model.agents, ["day_story"])           # counted as day_story in the usage ledger
        self.assertEqual(story.usage, (640, 170))
        lines = story.text.splitlines()
        self.assertEqual(lines[0], "07:02 · כניסה ראשית: אדם עם כובע אדום ניגש לדלת, דפק ויצא אחרי דקה.")
        self.assertEqual(lines[1], "07:40–17:47 · פרגולה: העובדים עבדו כאן, כרגיל.")
        self.assertEqual(lines[4], "16:40 · חניה: משהו שכדאי לראות: אדם בקפוצ'ון כהה ניסה את ידית הדלת האחורית.")
        self.assertEqual(lines[-1], "רוצה שאשלח את הסרטון של 16:40?")
        self.assertEqual(len(lines), 6)
        for word in ("התרע", "התרא", "חשוד", "נורמל", "כיסוי", "נרשמו"):
            self.assertNotIn(word, story.text)
        self.assertIsNone(re.search(r"[A-Za-z]", story.text))
        self.assertIsNone(re.search(r"\d+\s+(?:אירועים|התרעות|פעמים)", story.text))

    def test_without_a_usable_answer_the_code_tells_it_from_the_hebrew_notes(self) -> None:
        for answer in ("", "היום נרשמו 84 התרעות, מתוכן 7 נורמליות ו-2 חשודות.\nThe coverage is 00:00-17:13."):
            with self.subTest(answer=answer[:20]):
                story = self.tell(Scripted(answer))
                self.assertFalse(story.model_used)
                self.assertIn("07:02 · כניסה ראשית: אדם עם כובע אדום ניגש לדלת הכניסה, דופק והולך.", story.text)
                self.assertIn("07:40–17:47 · פרגולה: העובדים של הפרגולה היו כאן, כרגיל.", story.text)
                self.assertIn("משהו שכדאי לראות", story.text)
                self.assertIsNone(re.search(r"[A-Za-z]", story.text))
                self.assertNotIn("התרע", story.text)
        self.assertIn("16:40", self.tell(None).text)            # no model at all: the same code lines

    def test_a_quiet_period(self) -> None:
        model = Scripted(GOOD)
        story = ds.tell(self.book, None, snapshot(), "מה היה אתמול", NOW, model=model)
        self.assertEqual(story.text, "אתמול היה שקט, לא היה משהו לספר.")
        self.assertEqual(model.calls, [])                       # nothing to tell: no call
        # Never "quiet" over people the AI did not describe: when and where, in one plain line.
        story = ds.tell(self.book, None, snapshot(), "מה היה הלילה", at("05:00"), model=model)
        self.assertEqual(story.text, "הלילה לא היה משהו מיוחד לספר. היו אנשים שלא הצלחתי לראות מה עשו: "
                                     "במצלמה 1 בין 02:45 ל-02:46.")
        self.assertEqual(model.calls, [])

    # -- "שלח את 10:15" ------------------------------------------------------------------------------------------
    def test_each_line_keeps_its_clip_for_a_follow_up(self) -> None:
        story = self.tell(Scripted(GOOD))
        ds.remember(self.state, story, NOW)
        ref = lambda h: self.state.handles[h]["ref"]            # noqa: E731
        self.assertEqual(ref(ds.send_target("שלח את 10:15", self.state, NOW + 60)), f"{SITE}_ch2_{int(at('10:15'))}_alert")
        self.assertEqual(ref(ds.send_target("תראה לי את 14:03", self.state, NOW + 60)),
                         f"{SITE}_ch8_{int(at('14:03'))}_alert")
        self.assertEqual(ref(ds.send_target("תראה לי את זה", self.state, NOW + 60)),
                         f"{SITE}_ch2_{int(at('16:40'))}_alert")     # the story's offer
        self.assertEqual(ref(ds.send_target("כן", self.state, NOW + 60)), f"{SITE}_ch2_{int(at('16:40'))}_alert")
        self.assertEqual(ds.send_target("שלח את 03:00", self.state, NOW + 60), "")
        self.assertEqual(ds.send_target("מה השעה 10:15 בלונדון", self.state, NOW + 60), "")
        self.assertEqual(ds.send_target("שלח את 10:15", self.state, NOW + 7 * 3600), "")   # an old story

    # -- the owner's words ----------------------------------------------------------------------------------------
    def test_which_messages_are_the_day_story(self) -> None:
        for text in ("תן לי סיכום יום", "סיכום יום", "מה היה היום?", "מה קרה הבוקר", "תספר לי מה היה",
                     "מה היה מאתמול בערב", "סכם לי את היום", "what happened today?", "summary of the day"):
            self.assertTrue(ds.asks_day_story(text), text)
        for text in ("כמה התרעות היו היום?", "מה היה בכניסה ב-12:00", "תשלח תמונה של החניה", "מה קורה עכשיו",
                     "היי"):
            self.assertFalse(ds.asks_day_story(text), text)

    def test_periods(self) -> None:
        since, until, name = ds.period_of("מה היה הבוקר", NOW)
        self.assertEqual((ds._clock(since), ds._clock(until), name), ("05:00", "12:00", "הבוקר"))
        since, until, name = ds.period_of("מה היה מאתמול בערב", NOW)
        self.assertEqual((since, until, name), (at("18:00") - 86400, NOW, "מאתמול בערב"))
        self.assertEqual(ds.period_of("סיכום יום", NOW), (at("00:00"), NOW, "היום"))


class DayStoryAgentTest(unittest.TestCase):
    """The router: "תן לי סיכום יום" never reaches the models (nor summarize_period); "שלח את 10:15" sends it."""

    def setUp(self) -> None:
        self.case = DayStoryTest("test_noise_is_dropped")
        self.case.setUp()
        self.addCleanup(shutil.rmtree, self.case.dir, True)
        self.sent: List[Dict[str, Any]] = []

    def run_tool(self, ctx, name, args):
        from home_guard_project.box.brain.tools import TOOLS, _issue, _result  # noqa: PLC0415

        if name == "send_media":
            self.sent.append({"handle": args.get("handle"), "ref": ctx.state.handles[args["handle"]]["ref"]})
            return _result(_issue(ctx, "send_media", DONE, args.get("handle", ""), {"kind": "video", "bounds": "b"}))
        return TOOLS[name](ctx, args)

    def test_the_story_is_answered_in_code_and_its_lines_send_their_clips(self) -> None:
        root = self.case.dir
        story_model = Scripted(GOOD)

        class NoModel:
            model_name = "big"

            def chat(self, *a, **k):
                raise AssertionError("the assistant's models must not answer the day story")

        class Registry:
            def snapshot(self):
                return snapshot()

        services = Services(roots=lambda: [root], desc_dir=os.path.join(root, ".desc"), feedback_dir=root,
                            work_dir=os.path.join(root, ".live"), mute=None, deliver=None,
                            read_settings=lambda: {"owner_language": "he"}, now=lambda: NOW,
                            events=self.case.book, story_model=story_model)
        agent = OwnerAgentV2(NoModel(), Registry(), ChatMemory(os.path.join(root, ".conversations")),
                             ReceiptBook(os.path.join(root, ".receipts"), now=lambda: NOW), services,
                             run_tool=self.run_tool, now=lambda: NOW)
        out = agent.handle("תן לי סיכום יום", "-5", {"user_id": 1})
        self.assertEqual(out.tier, "code")
        self.assertTrue(out.text.startswith("07:02 · כניסה ראשית: אדם עם כובע אדום"), out.text)
        self.assertNotIn("summarize_period", out.tools_called)
        self.assertEqual(out.usage.get("story"), (640, 170))
        again = agent.handle("תן לי סיכום יום", "-5", {"user_id": 1})     # asked twice: the story again
        self.assertEqual(again.text, out.text)
        self.assertEqual(len(story_model.calls), 2)
        sent = agent.handle("שלח את 10:15", "-5", {"user_id": 1})
        self.assertEqual([s["ref"] for s in self.sent], [f"{SITE}_ch2_{int(at('10:15'))}_alert"])
        self.assertEqual(sent.tools_called, ("send_media",))
        agent.handle("תראה לי את זה", "-5", {"user_id": 1})
        self.assertEqual(self.sent[-1]["ref"], f"{SITE}_ch2_{int(at('16:40'))}_alert")


if __name__ == "__main__":
    unittest.main()
