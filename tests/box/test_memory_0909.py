# tests/box/test_memory_0909.py
"""The owner's morning chat of 2026-10-09: the assistant did not read its memory, saved a clip explanation only as a
tag, invented 23:59, kept two marks after a correction, offered what was already marked, and answered with empty
empathy. Models are scripted; nothing reaches a network."""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, List

from home_guard_project.box.brain import known_memory as km
from home_guard_project.box.brain.agent import OwnerAgentV2
from home_guard_project.box.brain.claims import empty_reply
from home_guard_project.box.brain.i18n import t
from home_guard_project.box.brain.memory import ChatMemory, ChatState
from home_guard_project.box.brain.models import ModelMessage, ToolCall
from home_guard_project.box.brain.receipts import ReceiptBook
from home_guard_project.box.brain.registry import CameraState, HouseSnapshot
from home_guard_project.box.brain.style import strip_boilerplate
from home_guard_project.box.brain.tools import Services, ToolContext, mark_known, retag_words
from home_guard_project.box.events import EventBook

NOW = dt.datetime(2026, 10, 9, 9, 22).timestamp()
PERGOLA, ENTRANCE = "ameer_week_0_1_ch3", "ameer_week_0_1_ch6"
ALERT = {"alert_id": f"{PERGOLA}_1791522099_alert", "camera": PERGOLA, "label": "suspicious",
         "ts": dt.datetime(2026, 10, 9, 8, 1).timestamp(), "summary": "Two men by a parked pickup truck."}
ALERT2 = {"alert_id": f"{ENTRANCE}_1791528173_alert", "camera": ENTRANCE, "label": "suspicious",
          "ts": dt.datetime(2026, 10, 9, 9, 42).timestamp(), "summary": "A man in a hat by an open car door."}
EXPLAIN = "זה תקין עובדים על הפרגולה בשעות אלה"
WEEK_END = dt.datetime(2026, 10, 15, 18, 0).timestamp()
OWNER = {"user_id": 1, "name": "Ameer"}


def call(name: str, **args: Any) -> ModelMessage:
    return ModelMessage(tool_calls=(ToolCall(id=f"c_{name}", name=name, arguments=args),), usage=(10, 2))


def reply(answer: str) -> ModelMessage:
    return call("reply", answer=answer)


class Scripted:
    def __init__(self, responses: List[ModelMessage]) -> None:
        self.responses, self.model_name, self.seen = list(responses), "big", []

    def chat(self, messages, tools, tool_choice=None):
        self.seen.append(([m.get("content") for m in messages], [x["function"]["name"] for x in tools]))
        if not self.responses:
            raise ConnectionError("script ended")
        return self.responses.pop(0)


class Registry:
    def __init__(self) -> None:
        self.now = NOW

    def snapshot(self):
        return HouseSnapshot(now=self.now, mode="guard", mode_ends=self.now + 3600, mode_started=self.now - 3600,
                             start_hour=0, end_hour=0,
                             cameras=(CameraState(PERGOLA, True, ("פרגולה",), live=True),
                                      CameraState(ENTRANCE, True, ("כניסה ראשית",), live=True)))


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.events = EventBook(os.path.join(self.root, "events"))
        self.clock = NOW

    def services(self) -> Services:
        return Services(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"),
                        feedback_dir=self.root, work_dir=os.path.join(self.root, ".live"), mute=None, deliver=None,
                        read_settings=lambda: {"owner_language": "he"}, now=lambda: self.clock, events=self.events)

    def agent(self, big: Scripted = None) -> OwnerAgentV2:
        return OwnerAgentV2(big or Scripted([]), Registry(), ChatMemory(os.path.join(self.root, ".conversations")),
                            ReceiptBook(os.path.join(self.root, ".receipts"), now=lambda: self.clock),
                            self.services(), now=lambda: self.clock)

    def ctx(self, text: str, state: ChatState = None) -> ToolContext:
        return ToolContext(turn_id="-5:1", chat_id="-5", speaker=OWNER, text=text, lang="he", mode="guard",
                           snapshot=Registry().snapshot(), state=state or ChatState(), services=self.services(),
                           book=ReceiptBook(os.path.join(self.root, ".r"), now=lambda: self.clock))

    def feedback(self) -> List[dict]:
        out = []
        for path in glob.glob(os.path.join(self.root, "feedback", "**", "*.feedback.json"), recursive=True):
            with open(path, encoding="utf-8") as f:
                out.append(json.load(f))
        return out

    def tap(self, agent, reply_, index):
        return agent.handle_choice("-5", reply_.question_token, index, OWNER)


class ExplanationTest(Base):
    """Point 2 and the owner's additions: the ✏️ explanation is a TAG, and the box draws the conclusion (workers,
    here today, likely all week) and asks only what it cannot know - at most two short button questions."""

    def test_the_explanation_asks_naturally_then_saves_one_short_receipt(self) -> None:
        agent = self.agent()
        self.clock = dt.datetime(2026, 10, 9, 9, 21).timestamp()
        q1 = agent.note_tag("-5", dict(ALERT), EXPLAIN, OWNER, label="normal")
        self.assertEqual(q1.text, "אה, הם של הפרגולה? עד איזו שעה הם עובדים?")
        self.assertEqual(q1.buttons, ("16:00", "17:00", "18:00", "אחר…"))
        self.assertEqual(self.events.list_known(self.clock), [])               # nothing saved before the answer
        q2 = self.tap(agent, q1, 2)                                             # [18:00]
        self.assertEqual(q2.text, "והם עובדים רק בפרגולה, או בכל הבית?")
        self.assertEqual(q2.buttons, ("כל הבית", "רק פרגולה"))
        done = self.tap(agent, q2, 0)                                           # [כל הבית]
        self.assertEqual(done.buttons, ())                                      # no third question
        lines = done.text.split("\n")
        self.assertEqual(lines[0], "אוקי תודה, רשמתי.")
        self.assertEqual(lines[1], "🏷️ תיוג לסרטון 08:01 (פרגולה): תקין: " + EXPLAIN)
        self.assertEqual(lines[2], "🧠 זכרתי: העובדים בכל הבית, כל יום 08:00–18:00, עד יום ה׳ 15.10.")
        (mark,) = self.events.list_known(self.clock)
        self.assertEqual((mark["camera"], mark["daily_from"], mark["daily_to"], mark["until"]),
                         ("", "08:00", "18:00", WEEK_END))
        codes = [c for row in done.rows for _, c in row]
        self.assertIn(f"tu:{ALERT['alert_id']}", codes)                         # the tag's own undo
        self.assertIn(f"kn:x:{mark['id']}", codes)                              # the memory's own undo
        self.assertIn(f"kn:d:{mark['id']}", codes)                              # [רק היום]
        # The workers walk to the main entrance at 09:43: covered (house-wide, inside their hours).
        self.assertFalse(self.events.decide(ENTRANCE, ALERT2["ts"] + 60, "suspicious", 2).notify)
        state = agent.memory.load("-5")
        self.assertEqual(state.prefs.get("group_scope"), "house")              # asked once, kept

    def test_the_owner_types_the_time(self) -> None:
        agent = self.agent()
        q1 = agent.note_tag("-5", dict(ALERT), EXPLAIN, OWNER)
        other = self.tap(agent, q1, 3)                                          # [אחר…]
        self.assertEqual(other.text, t("ask_type_time", "he"))
        q2 = agent.handle("עד 18", "-5", OWNER)
        self.assertEqual(q2.text, "והם עובדים רק בפרגולה, או בכל הבית?")
        done = self.tap(agent, q2, 1)                                           # [רק פרגולה]
        self.assertIn("🧠 זכרתי: העובדים בפרגולה, כל יום 08:00–18:00", done.text)
        self.assertEqual(self.events.list_known(NOW)[0]["camera"], PERGOLA)

    def test_what_the_owner_already_said_is_never_asked(self) -> None:
        agent = self.agent()
        q = agent.note_tag("-5", dict(ALERT), "עובדים אצלי בכל הבית עד 18:00", OWNER)
        self.assertEqual(q.buttons, ("כן", "לא"))                              # one obvious answer: one confirm
        self.assertIn("עד 18:00", q.text)
        done = self.tap(agent, q, 0)
        self.assertIn("🧠 זכרתי: העובדים בכל הבית", done.text)

    def test_a_one_off_or_a_marked_crew_asks_nothing(self) -> None:
        agent = self.agent()
        self.assertIsNone(agent.note_tag("-5", dict(ALERT), "זה הדוור", OWNER))
        self.assertIsNone(agent.note_tag("-5", dict(ALERT), "שני גברים ליד הטנדר", OWNER))
        self.events.mark_known(PERGOLA, "העובדים", "Ameer", WEEK_END, now=NOW - 60)
        self.assertIsNone(agent.note_tag("-5", dict(ALERT), EXPLAIN, OWNER))
        # The explanation is in the chat history: a later turn sees it (the 09:30 "אתה לא קורא?").
        state = agent.memory.load("-5")
        self.assertEqual(state.turns[-1]["kind"], "tag")
        self.assertIn("✏️ " + EXPLAIN, state.turns[-1]["text"])

    def test_a_no_saves_nothing(self) -> None:
        agent = self.agent()
        q1 = agent.note_tag("-5", dict(ALERT), EXPLAIN, OWNER)
        out = agent.handle("לא", "-5", OWNER)
        self.assertEqual(out.text, t("known_not_marked", "he"))
        self.assertEqual(self.events.list_known(NOW), [])
        self.assertIsNotNone(q1)


class NoInventedTimeTest(Base):
    """Point 3: no end time the owner did not give."""

    def test_no_time_asks_until_when(self) -> None:
        for text, until in (("זה השכן", None), ("זה השכן היום", None), ("זה השכן עד שש", "18:00"),
                            ("זה השכן עד 18", "18:00"), ("זה השכן לשעתיים", "11:22")):
            with self.subTest(text=text):
                self.events = EventBook(tempfile.mkdtemp())
                ctx = self.ctx(text)
                ctx.state.set_topic_camera(PERGOLA, "", NOW)
                out = mark_known(ctx, {"who": "השכן", "owner_words": "השכן", "until": "23:59"})
                if until is None:
                    self.assertFalse(out["ok"])                                 # the model's 23:59 is not the owner's
                    self.assertEqual(ctx.clarification["question"], "עד מתי לזכור את השכן?")
                    self.assertEqual(self.events.list_known(NOW), [])
                else:
                    self.assertTrue(out["ok"], out)
                    (mark,) = self.events.list_known(NOW)
                    self.assertEqual(dt.datetime.fromtimestamp(mark["until"]).strftime("%H:%M"), until)

    def test_until_from_words(self) -> None:
        at = dt.datetime(2026, 10, 9, 9, 34).timestamp()
        h = lambda s: dt.datetime.fromtimestamp(km.until_from_words(s, at)).strftime("%d %H:%M")  # noqa: E731
        self.assertEqual(h("הם עובדים עד 18:00 משהו כזה"), "09 18:00")
        self.assertEqual(h("עד 6 בערב"), "09 18:00")
        self.assertEqual(h("עד שש"), "09 18:00")
        self.assertEqual(h("until 5pm"), "09 17:00")
        self.assertIsNone(km.until_from_words("היום", at))
        self.assertIsNone(km.until_from_words("עובדים על הפרגולה בשעות אלה", at))


class CorrectionTest(Base):
    """Point 4: a correction replaces the earlier mark, and the receipt says what it replaced."""

    def test_the_18_correction_replaces_the_23_59_mark(self) -> None:
        old = self.events.mark_known(PERGOLA, "העובדים על הפרגולה", "Ameer",
                                     dt.datetime(2026, 10, 9, 23, 59).timestamp(), now=NOW)
        state = ChatState()
        state.prefs["group_scope"] = "camera"
        ctx = self.ctx("מי אמר עד 23:59? זה נשמע לך הגיוני? הם עובדים עד 18:00 משהו כזה", state)
        ctx.state.set_topic_camera(PERGOLA, "", NOW)
        out = mark_known(ctx, {"who": "העובדים על הפרגולה", "owner_words": "הם עובדים עד 18:00", "until": "18:00"})
        self.assertTrue(out["ok"], out)
        live = self.events.list_known(NOW)
        self.assertEqual(len(live), 1)
        self.assertNotEqual(live[0]["id"], old["id"])
        self.assertEqual(dt.datetime.fromtimestamp(live[0]["until"]).strftime("%H:%M"), "18:00")
        from home_guard_project.box.brain.render import receipt_line
        line = receipt_line(ctx.receipts[-1], "he", snapshot=Registry().snapshot())
        self.assertTrue(line.startswith("🧠 עדכנתי: העובדים על הפרגולה בפרגולה"), line)
        self.assertIn("(במקום 23:59)", line)

    def test_widening_to_the_house_replaces_and_keeps_the_time(self) -> None:
        self.events.mark_known(PERGOLA, "העובדים", "Ameer", dt.datetime(2026, 10, 9, 18, 0).timestamp(), now=NOW)
        ctx = self.ctx("אמרתי לך שיש אנשים שעובדים ליד הפרגולה, תרחיב לכל הבית")
        out = mark_known(ctx, {"who": "העובדים", "owner_words": "שעובדים ליד הפרגולה", "camera": "all",
                               "until": "18:00"})
        self.assertTrue(out["ok"], out)
        (mark,) = self.events.list_known(NOW)
        self.assertEqual(mark["camera"], "")
        from home_guard_project.box.brain.render import receipt_line
        line = receipt_line(ctx.receipts[-1], "he", snapshot=Registry().snapshot())
        self.assertIn("בכל הבית", line)
        self.assertIn("(במקום רק בפרגולה", line)

    def test_a_house_mark_is_not_narrowed_without_only(self) -> None:
        self.events.mark_known("", "העובדים", "Ameer", WEEK_END, now=NOW, daily_from="08:00", daily_to="18:00")
        ctx = self.ctx("שמור מידע\nהמידע זה עובדים אצלי על הפרגולה")
        out = mark_known(ctx, {"who": "העובדים", "owner_words": "עובדים אצלי", "camera": "פרגולה", "scope": "camera"})
        self.assertTrue(out["ok"], out)
        self.assertTrue(ctx.receipts[-1].detail.get("already"))              # "🧠 כבר זוכר", nothing narrowed
        (mark,) = self.events.list_known(NOW)
        self.assertEqual(mark["camera"], "")
        ctx = self.ctx("העובדים רק בפרגולה")
        mark_known(ctx, {"who": "העובדים", "owner_words": "העובדים רק בפרגולה", "camera": "פרגולה"})
        self.assertEqual(self.events.list_known(NOW)[0]["camera"], PERGOLA)  # the owner's "רק" does narrow

    def test_filler_and_promises_are_dropped(self) -> None:
        self.assertEqual(strip_boilerplate("העובדים מסומנים בכל הבית עד 18:00. אם יש צורך בתיקון נוסף, אנא עדכן "
                                           "אותי."), "העובדים מסומנים בכל הבית עד 18:00.")
        self.assertEqual(strip_boilerplate("העובדים מסומנים. אני מבין את התסכול שלך ואשתדל לשפר."),
                         "העובדים מסומנים.")
        self.assertTrue(empty_reply("אני מבין. אני אשתדל להיות יותר ברור ולשאול שאלות כשצריך."))

    def test_the_same_mark_again_is_already_saved(self) -> None:
        self.events.mark_known(PERGOLA, "השכן", "Ameer", dt.datetime(2026, 10, 9, 18, 0).timestamp(), now=NOW)
        ctx = self.ctx("זה השכן עד 18:00")
        ctx.state.set_topic_camera(PERGOLA, "", NOW)
        out = mark_known(ctx, {"who": "השכן", "owner_words": "זה השכן"})
        self.assertTrue(out["ok"])
        self.assertTrue(ctx.receipts[-1].detail["already"])
        self.assertEqual(len(self.events.list_known(NOW)), 1)


class ScopeTest(Base):
    """Point 5: a work crew is asked once - the whole house first - and the answer is kept."""

    def test_a_crew_is_asked_house_first_once(self) -> None:
        ctx = self.ctx("המידע זה עובדים אצלי על הפרגולה עד 18:00")
        out = mark_known(ctx, {"who": "העובדים", "owner_words": "עובדים אצלי"})
        self.assertFalse(out["ok"])                                              # "על הפרגולה" names one camera...
        ctx = self.ctx("עובדים אצלי עד 18:00")
        ctx.state.set_topic_camera(PERGOLA, "", NOW)
        out = mark_known(ctx, {"who": "העובדים", "owner_words": "עובדים אצלי"})
        self.assertEqual(ctx.clarification["choices"], ["כל הבית", "רק פרגולה"])  # ...this one is asked
        state = ctx.state
        state.prefs["group_scope"] = "house"                                     # the kept answer
        ctx = self.ctx("עובדים אצלי עד 18:00", state)
        self.assertTrue(mark_known(ctx, {"who": "העובדים", "owner_words": "עובדים אצלי"})["ok"])
        self.assertEqual(self.events.list_known(NOW)[0]["camera"], "")


class ContextTest(Base):
    """Point 1: every turn reads the live marks, what the owner said today, and the camera no mark covers."""

    def test_the_0945_turn_sees_the_mark_and_the_gap(self) -> None:
        self.events.mark_known(PERGOLA, "העובדים על הפרגולה", "Ameer", dt.datetime(2026, 10, 9, 18, 0).timestamp(),
                               now=NOW)
        self.clock = dt.datetime(2026, 10, 9, 9, 45).timestamp()
        big = Scripted([call("mark_known", who="העובדים", owner_words="שעובדים ליד הפרגולה", camera="all",
                             until="18:00"), reply("")])
        agent = self.agent(big)
        agent.note_alert("-5", dict(ALERT2))
        out = agent.handle("אמרתי לך יא מטומטם שיש אנשים שעובדים ליד הפרגולה, יש אינטראקציה וזה טבעי", "-5", OWNER)
        block = big.seen[0][0][-1]
        self.assertIn("[LIVE MARKS]", block)
        self.assertIn('"העובדים על הפרגולה" at פרגולה until 18:00', block)
        self.assertIn("[NOT COVERED]", block)
        self.assertIn("כניסה ראשית", block.split("[NOT COVERED]")[1])
        self.assertTrue(out.text.startswith("צודק."), out.text)                # names the mistake...
        self.assertIn("🧠 עדכנתי: העובדים בכל הבית", out.text)                  # ...and fixes it
        self.assertIn("(במקום רק בפרגולה", out.text)
        self.assertNotIn("אם תרצה", out.text)

    def test_an_offer_of_a_live_mark_is_rewritten(self) -> None:
        self.events.mark_known(PERGOLA, "העובדים", "Ameer", dt.datetime(2026, 10, 9, 18, 0).timestamp(), now=NOW)
        big = Scripted([reply("אם תרצה, אני יכול לסמן את האנשים ליד הפרגולה כעובדים עד 18:00."),
                        reply("העובדים מסומנים בפרגולה עד 18:00.")])
        out = self.agent(big).handle("למה שלחת התראה?", "-5", OWNER)
        self.assertEqual(out.text, "העובדים מסומנים בפרגולה עד 18:00.")
        self.assertIn("ALREADY marked", big.seen[1][0][-1])

    def test_today_lines_and_readable_receipts_in_the_history(self) -> None:
        agent = self.agent()
        agent.note_tag("-5", dict(ALERT), EXPLAIN, OWNER)
        state = agent.memory.load("-5")
        lines = km.today_lines(state, NOW + 60)
        self.assertEqual(len(lines), 1)
        self.assertIn(EXPLAIN, lines[0])


class EmptyEmpathyTest(Base):
    """Point 6: a reply with no fact, action or question is never sent."""

    def test_detection(self) -> None:
        for text in ("אני מבין אותך.", "אני מבין את התסכול שלך.", "I understand your frustration.", "סליחה על הבלבול."):
            self.assertTrue(empty_reply(text), text)
        for text in ("מה אתה רוצה שאעשה עם זה?", "צודק, סימנתי רק את הפרגולה.", "", "העובדים מסומנים עד 18:00."):
            self.assertFalse(empty_reply(text), text)
        self.assertEqual(strip_boilerplate("אני מבין אותך. ההתראה הייתה בכניסה הראשית."),
                         "ההתראה הייתה בכניסה הראשית.")

    def test_rewritten_once_then_a_concrete_status(self) -> None:
        self.events.mark_known(PERGOLA, "העובדים", "Ameer", dt.datetime(2026, 10, 9, 18, 0).timestamp(), now=NOW)
        out = self.agent(Scripted([reply("אני מבין אותך."), reply("אני מבין את התסכול שלך.")])).handle(
            "היית צריך להגיע למסקנה הזאת לבד ולתשאל אותי", "-5", OWNER)
        self.assertEqual(out.text, "שמור אצלי עכשיו: העובדים בפרגולה, היום עד 18:00. מה לתקן?")
        good = self.agent(Scripted([reply("אני מבין אותך."), reply("סימנתי את העובדים בפרגולה עד 18:00.")]))
        self.assertNotIn("מבין", good.handle("נו?", "-5", OWNER).text)


class TagAndMemoryTest(Base):
    """Point 7 and the routing: a tag, a memory, both, or neither - each in its own store, each with its line."""

    def test_save_info_and_change_tag_does_both(self) -> None:
        text = ("שמור מידע ושנה תיוג\nהתיוג זה אדם עם חולצה לבנה נמצא ליד הטנדר נראה שהוא לוקח משהו מהמטען\n\n"
                "המידע זה עובדים אצלי על הפרגולה עד 18:00")
        self.assertEqual(retag_words(text), "אדם עם חולצה לבנה נמצא ליד הטנדר נראה שהוא לוקח משהו מהמטען")
        big = Scripted([call("mark_known", who="העובדים", owner_words="עובדים אצלי", camera="פרגולה",
                             scope="camera"), reply("")])
        agent = self.agent(big)
        agent.note_alert("-5", dict(ALERT))
        out = agent.handle(text, "-5", OWNER)
        self.assertIn("🏷️ תיוג לסרטון 08:01 (פרגולה): אדם עם חולצה לבנה", out.text)
        self.assertIn("🧠 זכרתי: העובדים בפרגולה", out.text)
        (tag,) = self.feedback()
        self.assertEqual((tag["owner_label"], tag["owner_text"][:8]), ("other", "אדם עם ח"))
        self.assertEqual(len(self.events.list_known(NOW)), 1)

    def test_routing_tag_memory_both_neither(self) -> None:
        # Neither: a complaint is conversation - no feedback/ file, no memory.
        self.agent(Scripted([reply("ההתראה הייתה בפרגולה ב-08:01.")])).handle("למה שלחת את זה?", "-5", OWNER)
        self.assertEqual((self.feedback(), self.events.list_known(NOW)), ([], []))
        # Memory only: who is around, not about a clip - known, no feedback/.
        big = Scripted([call("mark_known", who="השכן", owner_words="השכן", camera="פרגולה"), reply("")])
        self.agent(big).handle("השכן בפרגולה עד 18:00", "-5", OWNER)
        self.assertEqual((len(self.feedback()), len(self.events.list_known(NOW))), (0, 1))
        # Tag only: a one-off ✏️ ("זה הדוור") - no question, no memory (the inbox wrote the feedback/ file).
        self.assertIsNone(self.agent().note_tag("-5", dict(ALERT), "זה הדוור", OWNER))
        self.assertEqual(len(self.events.list_known(NOW)), 1)
        # Both: said in reply to the alert - the clip's tag (normal) and the memory, two lines.
        big = Scripted([call("mark_known", who="הגנן", owner_words="זה הגנן", camera="פרגולה"), reply("")])
        out = self.agent(big).handle("זה בסדר זה הגנן, רק בפרגולה עד 17:00", "-5", OWNER, dict(ALERT), True)
        self.assertEqual([r.tool for r in out.receipts], ["retag_clip", "mark_known"])
        self.assertEqual([x["owner_label"] for x in self.feedback()], ["normal"])
        self.assertTrue(out.text.split("\n")[0].startswith("🏷️"))
        self.assertTrue(out.text.split("\n")[1].startswith("🧠"))

    def test_tag_label(self) -> None:
        self.assertEqual(km.tag_label(EXPLAIN), "normal")
        self.assertEqual(km.tag_label("לא חשוד, זה השכן"), "normal")
        self.assertEqual(km.tag_label("אדם עם כובע עובד ליד הכניסה"), "other")
        self.assertTrue(km.lasting_fact("אדם עם כובע עובד ליד הכניסה"))
        self.assertTrue(km.lasting_fact("הגנן שלנו"))
        self.assertFalse(km.lasting_fact("זה הדוור"))

    def test_what_do_you_remember_and_what_did_i_tag(self) -> None:
        self.events.mark_known("", "העובדים", "Ameer", WEEK_END, now=NOW, daily_from="08:00", daily_to="18:00")
        agent = self.agent()
        out = agent.handle("מה אתה זוכר?", "-5", OWNER)
        self.assertIn("🧠 העובדים בכל הבית: כל יום 08:00–18:00, עד יום ה׳ 15.10", out.text)
        self.assertEqual(out.tier, "code")
        from home_guard_project.box.feedback import Feedback, save_feedback
        save_feedback(self.root, dict(ALERT), Feedback(verdict="expected", owner_label="normal", owner_text=EXPLAIN),
                      EXPLAIN, OWNER, "-5", NOW)
        tags = agent.handle("מה תייגתי היום?", "-5", OWNER)
        self.assertIn("🏷️ תיוג לסרטון 08:01 (פרגולה): תקין: " + EXPLAIN, tags.text)
        self.assertNotIn("🧠", tags.text)


class DailyWindowTest(Base):
    """The owner's third addition: a crew's mark is their hours every day for a week."""

    def setUp(self) -> None:
        super().setUp()
        self.mark = self.events.mark_known(PERGOLA, "העובדים", "Ameer", WEEK_END, now=NOW, daily_from="08:00",
                                           daily_to="18:00")

    def at(self, day: int, hour: int, minute: int = 0) -> float:
        return dt.datetime(2026, 10, day, hour, minute).timestamp()

    def test_inside_outside_night_escalation(self) -> None:
        self.assertFalse(self.events.decide(PERGOLA, self.at(9, 10), "suspicious", 2).notify)
        self.assertTrue(self.events.decide(PERGOLA, self.at(9, 22), "suspicious", 2).notify)       # the night
        self.assertTrue(self.events.decide(PERGOLA, self.at(10, 6, 30), "suspicious", 2).notify)   # before they come
        self.assertTrue(self.events.decide(PERGOLA, self.at(11, 11), "escalation", 2).notify)
        self.assertTrue(self.events.decide(PERGOLA, self.at(16, 10), "suspicious", 2).notify)      # the week is over
        self.assertEqual(len(self.events.list_known(self.at(9, 23))), 1)     # still remembered at night

    def test_the_day_2_arrival_line_once(self) -> None:
        first = self.events.decide(PERGOLA, self.at(9, 12), "suspicious", 2)
        self.assertEqual(first.arrival, {})                                  # the day it was said: no line
        d = self.events.decide(PERGOLA, self.at(10, 7, 40), "suspicious", 2)
        self.assertEqual(d.arrival, {})                                      # 07:40 is before 08:00: sent instead
        self.assertTrue(d.notify)
        d = self.events.decide(PERGOLA, self.at(10, 8, 40), "suspicious", 2)
        self.assertFalse(d.notify)
        self.assertEqual((d.arrival["who"], d.arrival["camera"]), ("העובדים", PERGOLA))
        from home_guard_project.box.inference import arrival_text
        line = arrival_text(d.arrival, "he")             # the camera by the family's name (else "מצלמה 3")
        self.assertTrue(line.startswith("העובדים של ") and line.endswith("הגיעו (08:40)"), line)
        self.assertEqual(self.events.decide(PERGOLA, self.at(10, 9, 40), "suspicious", 2).arrival, {})

    def test_only_today_shortens_it(self) -> None:
        agent = self.agent()
        out = agent.known_button("-5", "d", self.mark["id"], OWNER)
        (mark,) = self.events.list_known(NOW)
        self.assertEqual((mark["daily_from"], mark["until"]), ("", self.at(9, 18)))
        self.assertIn("🧠 עדכנתי: העובדים בפרגולה, היום עד 18:00", out.text)
        self.assertTrue(self.events.decide(PERGOLA, self.at(10, 10), "suspicious", 2).notify)     # tomorrow alerts

    def test_another_last_day(self) -> None:
        agent = self.agent()
        ask = agent.known_button("-5", "o", self.mark["id"], OWNER)
        self.assertEqual(ask.text, t("ask_last_day", "he"))
        out = agent.handle("עד יום ב׳", "-5", OWNER)
        (mark,) = self.events.list_known(NOW)
        self.assertEqual((mark["daily_to"], mark["until"]), ("18:00", self.at(12, 18)))
        self.assertIn("🧠 עדכנתי", out.text)


if __name__ == "__main__":
    unittest.main()
