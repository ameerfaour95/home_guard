"""The owner's 13:00-13:06 of 2026-10-09, through the brain with scripted models.

13:02:01 "למה אתה לא נותן הסבר ?" right after two photos is answered from the photos' looks, per camera with its name
and time, never with a question back. 13:05:16 "אלה אותם אנשים שעבדו מהבוקר או שהתחלפו?" and 13:05:29 "אתה בטוח?"
are answered in code from evidence only: the open event's continuity, the same-day clothes (ReID, shadow too), and
the owner's mark said as his words; "כן, אני בטוח" never comes without strong evidence."""

from __future__ import annotations

import datetime as dt
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List

from home_guard_project.box import entities as ent
from home_guard_project.box.brain import look_explain as le
from home_guard_project.box.brain import same_people as sp
from home_guard_project.box.brain.agent import OwnerAgentV2, _last_photo_camera
from home_guard_project.box.brain.i18n import t
from home_guard_project.box.brain.memory import ChatMemory, ChatState
from home_guard_project.box.brain.models import ModelMessage, ToolCall
from home_guard_project.box.brain.receipts import ReceiptBook
from home_guard_project.box.brain.registry import CameraState, HouseSnapshot
from home_guard_project.box.brain.tools import Services
from home_guard_project.box.events import EventBook, Session

CH1, PERGOLA, ENTRANCE = "ameer_week_0_1_ch1", "ameer_week_0_1_ch3", "ameer_week_0_1_ch6"
T = lambda h, m, s=0: dt.datetime(2026, 10, 9, h, m, s).timestamp()  # noqa: E731
OWNER = {"user_id": 1, "name": "Ameer"}
CHAT = "-5"
MARK = "העובדים של הפרגולה"


def text(content: str) -> ModelMessage:
    return ModelMessage(content=content, usage=(10, 2))


def call(name: str, **args: Any) -> ModelMessage:
    return ModelMessage(tool_calls=(ToolCall(id=f"c_{name}", name=name, arguments=args),), usage=(10, 2))


class Scripted:
    def __init__(self, responses: List[ModelMessage]) -> None:
        self.responses, self.model_name, self.seen = list(responses), "big", []

    def chat(self, messages, tools, tool_choice=None):
        self.seen.append([m.get("content") for m in messages])
        if not self.responses:
            raise ConnectionError("script ended")
        return self.responses.pop(0)


class Registry:
    def __init__(self, clock) -> None:
        self.clock = clock

    def snapshot(self):
        now = self.clock()
        return HouseSnapshot(now=now, mode="guard", mode_ends=now + 3600, mode_started=now - 3600, start_hour=0,
                             end_hour=0, cameras=(CameraState(CH1, True, (), live=True),
                                                  CameraState(PERGOLA, True, ("פרגולה",), live=True),
                                                  CameraState(ENTRANCE, True, ("כניסה ראשית",), live=True)))


def person(pid: str, first: float, last: float, **extra: Any) -> Dict[str, Any]:
    return dict({"id": pid, "kind": "person", "state": "active", "first_seen": first, "last_seen": last}, **extra)


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.events = EventBook(os.path.join(self.root, "events"))
        for cam in (PERGOLA, ENTRANCE):
            self.events.mark_known(cam, MARK, "Ameer", dt.datetime(2026, 10, 15, 18).timestamp(), now=T(9, 34),
                                   daily_from="07:00", daily_to="18:00")
        self.clock = T(13, 0, 53)
        self.memory = ChatMemory(os.path.join(self.root, ".conv"))

    def agent(self, big: Scripted) -> OwnerAgentV2:
        services = Services(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"),
                            feedback_dir=self.root, work_dir=os.path.join(self.root, ".live"), mute=None, deliver=None,
                            read_settings=lambda: {"owner_language": "he"}, now=lambda: self.clock,
                            events=self.events)
        return OwnerAgentV2(big, Registry(lambda: self.clock), self.memory,
                            ReceiptBook(os.path.join(self.root, ".receipts"), now=lambda: self.clock), services,
                            fast_model=None, now=lambda: self.clock)

    def photos(self, at: float, looks: List[tuple]) -> None:
        """The 13:00 look_around: one photo handle per camera with what the vision model saw and its count."""
        state = self.memory.load(CHAT)
        shown = []
        for cam, description, people in looks:
            handle = state.add_handle("photo", f"{cam}.jpg", cam, at)
            state.note_observation(handle, description)
            state.handles[handle]["people"] = people
            shown.append(handle)
        state.add_turn("1", "תגיד לי יש מישהו בחוץ אצלי?", "", shown, [], at)
        self.memory.save(CHAT, state)

    def say(self, agent, words, at):
        self.clock = at
        return agent.handle(words, CHAT, OWNER, None, False)


LOOKS_1300 = [(CH1, "A person wearing a red cap and carrying a bag walks along a paved path.", 1),
              (PERGOLA, "Three workers stand next to a pickup truck under the pergola frame.", 3),
              (ENTRANCE, "A white car with its trunk open is parked; no people.", 0)]


class ExplainTest(Base):
    def test_asks_explanation(self) -> None:
        for words in ("למה אתה לא נותן הסבר ?", "תסביר מה יש שם", "אתה צריך להסביר", "מה אתה רואה?",
                      "why no explanation?", "what do you see"):
            self.assertTrue(le.asks_explanation(words), words)
        for words in ("או מביא סרטון למה תמונה?", "מה קורה?", "יש מישהו בחוץ?", "סבבה"):
            self.assertFalse(le.asks_explanation(words), words)

    def test_1302_why_no_explanation_is_answered_from_the_photos(self) -> None:
        lines = ("במצלמה 1 ב-13:00: אדם בכובע אדום עם תיק הולך בשביל.\n"
                 "בפרגולה ב-13:00: שלושה עובדים ליד הטנדר, כנראה העובדים של הפרגולה.\n"
                 "בכניסה הראשית ב-13:00: רכב לבן עם תא מטען פתוח, אין אנשים.")
        big = Scripted([text(lines)])
        self.photos(T(13, 0, 53), LOOKS_1300)
        out = self.say(self.agent(big), "למה אתה לא נותן הסבר ?", T(13, 2, 1))
        self.assertEqual(out.text, lines)
        self.assertEqual(len(big.seen), 1)                       # one call, no tool loop, no question
        prompt = big.seen[0][1]
        self.assertIn('PLACE "בפרגולה ב-13:00:"', prompt)
        self.assertIn("Three workers stand next to a pickup truck", prompt)
        self.assertIn(f"KNOWN: {MARK}", prompt)
        self.assertIn('PLACE "במצלמה 1 ב-13:00:"', prompt)
        self.assertNotIn("KNOWN", prompt.split("\n")[1])         # camera 1 has no mark

    def test_a_question_back_is_replaced_by_the_plain_lines(self) -> None:
        big = Scripted([text("האם יש משהו מסוים שאתה רוצה לדעת?")])
        self.photos(T(13, 0, 53), LOOKS_1300)
        out = self.say(self.agent(big), "למה אתה לא נותן הסבר ?", T(13, 2, 1))
        self.assertEqual(out.text, "במצלמה 1 ב-13:00: אדם אחד\nבפרגולה ב-13:00: 3 אנשים\n"
                                   "בכניסה הראשית ב-13:00: אין אנשים")

    def test_old_photos_go_to_the_model_as_before(self) -> None:
        big = Scripted([call("reply", answer="אני לא רואה עכשיו תמונות חדשות.")])
        self.photos(T(12, 40), LOOKS_1300)
        out = self.say(self.agent(big), "למה אתה לא נותן הסבר ?", T(13, 2, 1))
        self.assertEqual(out.text, "אני לא רואה עכשיו תמונות חדשות.")
        self.assertNotIn("He asked for an explanation", big.seen[0][0])

    def test_a_video_ask_is_not_an_explanation_and_records_the_busiest_camera(self) -> None:
        self.photos(T(13, 0, 53), LOOKS_1300)
        state = self.memory.load(CHAT)
        self.assertEqual(_last_photo_camera(state, T(13, 2, 13)), PERGOLA)


class SamePeopleTest(Base):
    def visits(self) -> None:
        """The pergola that day: short separate events (the tracker loses everyone after a minute)."""
        for sid, opened, closed in (("s1", T(12, 4, 15), T(12, 10, 49)), ("s2", T(12, 46, 8), T(12, 49, 42)),
                                    ("s3", T(12, 50, 42), T(12, 56, 21))):
            self.events._closed.append(Session(id=sid, camera=PERGOLA, opened=opened, last_active=closed - 60,
                                               closed=closed, entities=[person("P1", opened, closed - 60)]))

    def test_1305_not_seen_without_a_break(self) -> None:
        self.visits()
        self.photos(T(13, 4, 29), [(PERGOLA, "Three workers on the pergola.", 3), (ENTRANCE, "A white car.", 0)])
        big = Scripted([call("reply", answer="כן, אני בטוח. אלה אותם עובדים מהבוקר.")])
        agent = self.agent(big)
        first = self.say(agent, "אלה אותם אנשים שעבדו מהבוקר או שהתחלפו?", T(13, 5, 16)).text
        self.assertEqual(first, "לא בטוח, לא ראיתי אותם ברצף: בפרגולה היו היום 3 ביקורים נפרדים, הראשון ב-12:04 "
                                "והאחרון ב-12:50, ואין לי השוואת בגדים ביניהם.\n"
                                f'מה שכן, יש סימון שלך "{MARK}" עד 18:00, אבל הוא לפי מה שאמרת, לא לפי מה שראיתי.')
        sure = self.say(agent, "אתה בטוח?", T(13, 5, 29)).text
        self.assertEqual(sure, "לא, אני לא בטוח. אין לי רצף שלהם (בפרגולה היו היום 3 ביקורים נפרדים, האחרון "
                               f'ב-12:50) ואין השוואת בגדים.\nהסימון "{MARK}" עד 18:00 הוא לפי מה שאמרת לי, לא בדיקה '
                               "שלי.")
        self.assertEqual(big.seen, [])                            # no model: nothing to overclaim
        self.assertNotIn("אני בטוח", first + sure.replace("לא בטוח", ""))

    def test_seen_all_along_is_said_with_the_ids_and_never_fully_sure(self) -> None:
        self.events._closed.append(Session(id="a", camera=PERGOLA, opened=T(12, 0), last_active=T(12, 30),
                                           closed=T(12, 30), entities=[person("P1", T(12, 0), T(12, 30))]))
        self.events._open[PERGOLA] = Session(id="b", camera=PERGOLA, opened=T(12, 30), last_active=T(13, 5),
                                             parent="a", entities=[person("P1", T(12, 0), T(13, 5)),
                                                                   person("P2", T(12, 40), T(13, 5))])
        self.photos(T(13, 4, 29), [(PERGOLA, "Two workers.", 2)])
        agent = self.agent(Scripted([]))
        first = self.say(agent, "אלה אותם אנשים?", T(13, 5, 16)).text
        self.assertTrue(first.startswith("נראה שכן: ראיתי אותם בפרגולה ברצף מאז 12:00 (P1, P2), בלי הפסקה."), first)
        sure = self.say(agent, "אתה בטוח?", T(13, 5, 29)).text
        self.assertTrue(sure.startswith("די בטוח, אבל לא לגמרי: ראיתי אותם בפרגולה ברצף מאז 12:00"), sure)

    def test_the_clothes_are_a_hint_and_a_split_lowers(self) -> None:
        self.events._open[PERGOLA] = Session(
            id="c", camera=PERGOLA, opened=T(13, 3), last_active=T(13, 5),
            entities=[person("P1", T(13, 3), T(13, 3, 30)),
                      person("P2", T(13, 4), T(13, 5), reid_shadow={"would": "link", "to": "P1", "score": 0.9})])
        self.photos(T(13, 4, 29), [(PERGOLA, "Two workers.", 2)])
        first = self.say(self.agent(Scripted([])), "אלה אותם אנשים?", T(13, 5, 16)).text
        self.assertTrue(first.startswith("לפי הבגדים זה נראה אותם אנשים (P2=P1, 13:03), אבל לא ראיתי אותם ברצף."),
                        first)
        ev = sp.evidence(self.events, PERGOLA, T(13, 5, 16))
        self.assertEqual(ev["level"], "clothes")
        self.events._open[PERGOLA] = Session(
            id="d", camera=PERGOLA, opened=T(12, 0), last_active=T(13, 5),
            entities=[person("P1", T(12, 0), T(13, 5), reid_shadow={"would": "split", "from": "P1", "score": 0.1})])
        self.assertEqual(sp.evidence(self.events, PERGOLA, T(13, 5, 16))["level"], "mixed")
        out = sp.answer(self.events, Registry(lambda: T(13, 5)).snapshot(), PERGOLA, T(13, 5, 16), "he")
        self.assertTrue(out.startswith("לא בטוח, ראיתי אנשים בפרגולה ברצף מאז 12:00, אבל לפי הבגדים P1 נראה שונה"),
                        out)

    def test_are_you_sure_alone_goes_to_the_model(self) -> None:
        big = Scripted([call("reply", answer="על מה?")])
        self.say(self.agent(big), "אתה בטוח?", T(13, 5, 29))
        self.assertEqual(len(big.seen), 1)
        self.assertTrue(sp.asks_sure("אתה בטוח?") and sp.asks_sure("בטוח?") and sp.asks_sure("are you sure?"))
        self.assertFalse(sp.asks_sure("אתה בטוח שהמצלמה דלוקה?"))

    def test_the_camera_in_question_is_the_photo_with_people(self) -> None:
        self.photos(T(13, 4, 29), [(PERGOLA, "Three workers.", 3), (ENTRANCE, "A white car.", 0)])
        self.assertEqual(sp.camera_in_question(self.memory.load(CHAT), None, T(13, 5, 16)), PERGOLA)
        self.assertEqual(sp.camera_in_question(ChatState(), {"camera": ENTRANCE}, T(13, 5, 16)), ENTRANCE)


class AppearanceSaidTest(unittest.TestCase):
    def test_links_and_splits_in_shadow_and_on(self) -> None:
        rows = [person("P1", 0, 1, reid_shadow={"would": "split", "from": "P1", "score": 0.13}),
                person("P4", 0, 1, reid_shadow={"would": "link", "to": "P1", "score": 0.897}),
                person("P5", 0, 1, linked_by="appearance", link_score=0.8, not_of=["P2"], veto_score=0.2),
                {"id": "CAR1", "kind": "vehicle"}, "junk"]
        said = ent.appearance_said(rows)
        self.assertEqual([(x["entity"], x["what"], x["other"], x["acted"]) for x in said],
                         [("P1", "split", "P1", False), ("P4", "link", "P1", False), ("P5", "link", "P5", True),
                          ("P5", "split", "P2", True)])


if __name__ == "__main__":
    unittest.main()
