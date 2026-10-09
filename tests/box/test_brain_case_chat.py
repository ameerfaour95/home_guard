"""The chat side of the long-term memory (brain/case_chat.py, 2026-10-09): the owner's 13:54 explanation also writes a
precedent; "מה אתה זוכר?" lists it apart from the week-long items; the one question about the future days goes out
once with three buttons, and a tap answers it; nightly routines stay in shadow unless box.yaml says on."""
from __future__ import annotations

import datetime as dt
import json
import os
import unittest

from test_activity_chat import CHAT, ENTRANCE, OWNER, PERGOLA, RED_1325, Base, Registry, Scripted, explain, T

from home_guard_project.box.brain import case_chat
from home_guard_project.box.brain.agent import OwnerAgentV2
from home_guard_project.box.brain.memory import ChatMemory
from home_guard_project.box.brain.receipts import ReceiptBook
from home_guard_project.box.brain.tools import Services
from home_guard_project.box.case_memory import CaseStore, link

GATE = "ameer_week_0_1_ch1"     # no mark covers it (a live mark's precedent covers its hours: no proposal there)
DAY = lambda d, h, m=0: dt.datetime(2026, 10, d, h, m).timestamp()  # noqa: E731


class Sent:
    def __init__(self) -> None:
        self.messages = []

    def __call__(self, chat_id, text, rows=()):
        self.messages.append((chat_id, text, rows))
        return {"ok": True, "message_id": len(self.messages)}


class CaseChatBase(Base):
    def setUp(self) -> None:
        super().setUp()
        self.cases = CaseStore.at(os.path.join(self.root, "cases.jsonl"), now=lambda: self.clock)

    def agent(self, big, fast=None) -> OwnerAgentV2:
        self.services = Services(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"),
                                 feedback_dir=self.root, work_dir=os.path.join(self.root, ".live"), mute=None,
                                 deliver=None, read_settings=lambda: {"owner_language": "he"}, now=lambda: self.clock,
                                 events=self.events, activities=self.activities, cases=self.cases)
        agent = OwnerAgentV2(big, Registry(lambda: self.clock), ChatMemory(os.path.join(self.root, ".conv")),
                             ReceiptBook(os.path.join(self.root, ".receipts"), now=lambda: self.clock), self.services,
                             fast_model=fast, now=lambda: self.clock)
        agent.note_alert(CHAT, dict(RED_1325))
        return agent

    def explained(self):
        big = Scripted([explain(event="E1", actions=["lying", "crawling"], place="stairs", place_words="במדרגות",
                                cause="החשמלאים שמתקינים לדים", cause_en="electricians installing LED lights")])
        agent = self.agent(big)
        out = self.say(agent, "אלה מקרים תקינים הם מתקינים זה שני אנשים שמרכיבים את הלדים במדרגות", T(13, 54, 7))
        return agent, out


class CreationTest(CaseChatBase):
    def test_the_1354_explanation_writes_one_shadow_precedent_and_says_nothing_new(self) -> None:
        _, out = self.explained()
        self.assertEqual(out.text, "🧠 הבנתי, השכיבה, הזחילה והכלים ביד במדרגות של הכניסה הראשית זה החשמלאים "
                                   "שמתקינים לדים. עד 18:00 זה ייחשב אצלי רגיל.")
        (fact,) = self.activities.live(self.clock)
        (case,) = [c for c in self.cases.cases() if c.source.get("origin") == "activity"]
        self.assertEqual((case.camera, case.status, case.source["fact_id"], case.source["chat_id"]),
                         (ENTRANCE, "shadow", fact.id, CHAT))
        self.assertEqual(case.scope.actions, tuple(fact.actions))
        self.assertEqual(case.created_by, OWNER["name"])

    def test_cancel_under_the_explanation_ends_its_precedent_too(self) -> None:
        agent, _ = self.explained()
        (fact,) = self.activities.live(self.clock)
        agent.known_button(CHAT, "x", fact.id, OWNER)
        self.assertEqual([c for c in self.cases.cases() if c.source.get("origin") == "activity"], [])

    def test_what_do_you_remember_lists_the_long_term_apart_and_never_repeats(self) -> None:
        agent, _ = self.explained()
        out = self.say(agent, "מה אתה זוכר?", T(13, 56))
        week, _, long_term = out.text.partition("לטווח ארוך (לומד ממך):")
        self.assertIn("🧠 בכניסה הראשית: שכיבה/זחילה/כלים ביד במדרגות = החשמלאים שמתקינים לדים", week)
        self.assertIn("🗂️ במדרגות של הכניסה הראשית: שכיבה/זחילה/כלים ביד = החשמלאים שמתקינים לדים · לומד, 0 מתוך 3 "
                      "אישורים", long_term)
        self.assertNotIn("07:00", long_term, "the week-long line already says its hours")


class QuestionTest(CaseChatBase):
    def test_the_last_day_question_goes_out_once_and_a_tap_answers_it(self) -> None:
        agent, _ = self.explained()
        (fact,) = self.activities.live(self.clock)
        sent = Sent()
        self.assertEqual(case_chat.tick(self.services, sent, [CHAT], {"owner_language": "he"},
                                        agent.registry.snapshot(), DAY(14, 18))["asked"], 0)
        done = case_chat.tick(self.services, sent, [CHAT], {"owner_language": "he"}, agent.registry.snapshot(),
                              DAY(15, 18))
        self.assertEqual(done["asked"], 1)
        (chat, text, rows), = sent.messages
        self.assertEqual(text, "החשמלאים שמתקינים לדים במדרגות של הכניסה הראשית: עוד פעילים אחרי 15.10?")
        self.assertEqual([[label for label, _ in row] for row in rows], [["עוד שבוע", "זה נגמר"], ["קבוע: כל יום חול"]])
        self.assertEqual(rows[0][0][1], f"kn:x:ce.w.{fact.id}")
        self.assertTrue(all(len(code.encode()) <= 64 for row in rows for _, code in row))
        case_chat.tick(self.services, sent, [CHAT], {"owner_language": "he"}, agent.registry.snapshot(), DAY(15, 18, 5))
        self.assertEqual(len(sent.messages), 1, "asked once")
        self.clock = DAY(15, 18, 6)
        reply = agent.known_button(CHAT, "x", f"ce.w.{fact.id}", OWNER)
        self.assertEqual(reply.text, "🧠 סגור, החשמלאים שמתקינים לדים במדרגות של הכניסה הראשית עד יום ה׳ 22.10.")
        self.assertIsNone(agent.known_button(CHAT, "x", f"ce.w.{fact.id}", OWNER), "a second tap says nothing")
        self.assertEqual(self.activities.get(fact.id).until, DAY(22, 18))

    def test_back_after_it_ended_asks_with_the_real_time(self) -> None:
        agent, _ = self.explained()
        (fact,) = self.activities.live(self.clock)
        (case,) = [c for c in self.cases.cases() if c.source.get("origin") == "activity"]
        q = {"key": f"extend:{fact.id}", "kind": "again", "case_id": case.id, "fact_id": fact.id, "chat_id": CHAT,
             "last_day": fact.until, "seen_at": DAY(16, 9, 12)}
        text, _ = case_chat.question(q, self.services, agent.registry.snapshot(), "he")
        self.assertEqual(text, "החשמלאים שמתקינים לדים במדרגות של הכניסה הראשית: שוב כאן היום (09:12). עוד פעילים?")

    def test_standing_and_its_promise(self) -> None:
        agent, _ = self.explained()
        (fact,) = self.activities.live(self.clock)
        case_chat.tick(self.services, Sent(), [CHAT], {"owner_language": "he"}, agent.registry.snapshot(), DAY(15, 18))
        self.clock = DAY(15, 18, 1)
        reply = agent.known_button(CHAT, "x", f"ce.s.{fact.id}", OWNER)
        self.assertTrue(reply.text.startswith("🧠 שמרתי כקבוע: החשמלאים שמתקינים לדים במדרגות של הכניסה הראשית, "
                                              "ימי חול 07:00–18:00."), reply.text)
        self.assertIn("אירוע חמור תמיד יגיע כמו היום", reply.text)

    def test_routine_proposals_are_off_shadow_or_sent_once(self) -> None:
        agent, _ = self.explained()
        with open(self.events.events_path, "w", encoding="utf-8") as f:
            for d in (1, 2, 4, 5, 6, 7, 8):
                ts = DAY(d, 7, 40)
                f.write(json.dumps({"id": f"s{d}", "camera": GATE, "opened": ts, "last_active": ts,
                                    "observations": [{"ts": ts, "alert_id": f"a{d}", "label": "normal",
                                                      "people": 1}]}) + "\n")
        snap = agent.registry.snapshot()
        sent = Sent()
        self.assertEqual(case_chat.tick(self.services, sent, [CHAT], {"owner_language": "he"}, snap, DAY(9, 3, 5))
                         ["routines"], 0, "off by default")
        out = case_chat.tick(self.services, sent, [CHAT], {"owner_language": "he", "routine_proposals": "shadow"},
                             snap, DAY(10, 3, 5))
        self.assertEqual((out["routines"], sent.messages), (1, []), "shadow logs only")
        self.assertEqual(len(self.cases.routine_proposals()), 1)
        # "on" sends a new one once, with the camera's name; the owner's yes makes a shadow case.
        item = {"rid": "R9", "proposal": dict(next(iter(self.cases.routine_proposals().values()))["proposal"])}
        text, rows = case_chat.routine_question(item, snap, "he")
        self.assertEqual(text, "אני רואה מישהו במצלמה 1 ב-7 מתוך 14 הימים האחרונים, בסביבות 07:40. זו שגרה?")
        self.assertNotIn(GATE, text)
        reply = agent.known_button(CHAT, "x", "ce.ry.R1", OWNER)
        self.assertTrue(reply.text.startswith("🧠 שמרתי כשגרה."))
        self.assertIsNone(agent.known_button(CHAT, "x", "ce.ry.R1", OWNER))


if __name__ == "__main__":
    unittest.main()
