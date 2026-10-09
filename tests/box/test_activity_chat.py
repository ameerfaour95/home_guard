"""The owner's afternoon of 2026-10-09 (13:25-13:55), through the brain with scripted models: an explanation of an
action becomes an activity fact with one natural line; a follow-up adds to it; a correction keeps one fact;
"סבבה" gets 👍; no reply repeats; "are these the same people" needs evidence; a plain message is never a tag."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, List

from home_guard_project.box import activity_memory as am
from home_guard_project.box.brain import activity_chat as ac
from home_guard_project.box.brain import reply_guard as rg
from home_guard_project.box.brain.agent import OwnerAgentV2
from home_guard_project.box.brain.i18n import t
from home_guard_project.box.brain.memory import ChatMemory, ChatState
from home_guard_project.box.brain.models import ModelMessage, ToolCall
from home_guard_project.box.brain.receipts import ReceiptBook
from home_guard_project.box.brain.registry import CameraState, HouseSnapshot
from home_guard_project.box.brain.tools import Services
from home_guard_project.box.events import EventBook

PERGOLA, ENTRANCE = "ameer_week_0_1_ch3", "ameer_week_0_1_ch6"
T = lambda h, m, s=0: dt.datetime(2026, 10, 9, h, m, s).timestamp()  # noqa: E731
RED_1325 = {"alert_id": f"{ENTRANCE}_1791541475_alert", "camera": ENTRANCE, "label": "escalation", "ts": T(13, 24),
            "summary": "A man lies on the ground while another person in dark clothing and a cap crawls nearby, "
                       "holding a long metal bar."}
RED_1345 = {"alert_id": f"{PERGOLA}_1791542645_alert", "camera": PERGOLA, "label": "escalation", "ts": T(13, 44),
            "summary": "Two men walk toward a white car. A third man opens the car door and leans inside."}
OWNER = {"user_id": 1, "name": "Ameer"}
CHAT = "-5"


def call(name: str, **args: Any) -> ModelMessage:
    return ModelMessage(tool_calls=(ToolCall(id=f"c_{name}", name=name, arguments=args),), usage=(10, 2))


def text(content: str) -> ModelMessage:
    return ModelMessage(content=content, usage=(10, 2))


def explain(**kw: Any) -> ModelMessage:
    out = {"kind": "explain", "event": "", "camera": "", "fact": "", "actions": [], "place": "", "place_words": "",
           "cause": "", "cause_en": "", "who_mark": "", "until": ""}
    out.update(kw)
    return text(json.dumps(out, ensure_ascii=False))


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
                             end_hour=0, cameras=(CameraState(PERGOLA, True, ("פרגולה",), live=True),
                                                  CameraState(ENTRANCE, True, ("כניסה ראשית",), live=True)))


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.events = EventBook(os.path.join(self.root, "events"))
        self.activities = am.ActivityBook(os.path.join(self.root, "activity.json"))
        self.clock = T(13, 54, 7)
        # The live marks of that afternoon: the pergola workers, at the pergola and at the main entrance, 07-18.
        for cam in (PERGOLA, ENTRANCE):
            self.events.mark_known(cam, "העובדים של הפרגולה", "Ameer", dt.datetime(2026, 10, 15, 18).timestamp(),
                                   now=T(9, 34), daily_from="07:00", daily_to="18:00")

    def agent(self, big: Scripted, fast: Scripted = None) -> OwnerAgentV2:
        services = Services(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"),
                            feedback_dir=self.root, work_dir=os.path.join(self.root, ".live"), mute=None, deliver=None,
                            read_settings=lambda: {"owner_language": "he"}, now=lambda: self.clock,
                            events=self.events, activities=self.activities)
        agent = OwnerAgentV2(big, Registry(lambda: self.clock), ChatMemory(os.path.join(self.root, ".conv")),
                             ReceiptBook(os.path.join(self.root, ".receipts"), now=lambda: self.clock), services,
                             fast_model=fast, now=lambda: self.clock)
        for alert in (RED_1325, RED_1345):
            agent.note_alert(CHAT, dict(alert))
        return agent

    def say(self, agent, words, at, alert=None):
        self.clock = at
        return agent.handle(words, CHAT, OWNER, dict(alert) if alert else None, bool(alert))


class ExplainTest(Base):
    def test_1354_the_explanation_becomes_one_fact_with_one_natural_line(self) -> None:
        big = Scripted([explain(event="E1", actions=["lying", "crawling"], place="stairs", place_words="במדרגות",
                                cause="החשמלאים שמתקינים לדים", cause_en="electricians installing LED lights")])
        out = self.say(self.agent(big), "אלה מקרים תקינים הם מתקינים זה שני אנשים שמרכיבים את הלדים במדרגות",
                       T(13, 54, 7))
        self.assertEqual(out.text, "🧠 הבנתי, השכיבה, הזחילה והכלים ביד במדרגות של הכניסה הראשית זה החשמלאים "
                                   "שמתקינים לדים. עד 18:00 זה ייחשב אצלי רגיל.")
        (fact,) = self.activities.live(self.clock)
        self.assertEqual((fact.cameras, fact.place, fact.daily_from, fact.daily_to), ([ENTRANCE], "stairs", "07:00",
                                                                                     "18:00"))
        self.assertEqual(fact.actions, ["lying", "crawling", "holding_tool"])     # what the 13:25 red showed
        self.assertEqual(out.rows, (((t("btn_cancel_activity", "he"), f"kn:x:{fact.id}"),),))
        self.assertEqual(out.buttons, ())                                         # no question: the window is known
        self.assertEqual(len(big.seen), 1)                                        # one model call, no tool loop
        self.assertEqual(self._tags(), [])                                        # memory, never a tag
        # The guard loop now keeps the 14:03 lying-down quiet at the entrance, and not at the pergola.
        self.assertIsNotNone(self.activities.match(ENTRANCE, T(14, 3), "A person lies on the ground"))
        self.assertIsNone(self.activities.match(PERGOLA, T(14, 3), "A person lies on the ground"))

    def test_1354_follow_ups_add_to_the_same_fact_and_a_correction_keeps_one(self) -> None:
        big = Scripted([
            explain(event="E1", actions=["lying"], place="stairs", cause="החשמלאים שמתקינים לדים"),
            explain(fact="", event="E1", actions=["bending"], cause="החשמלאים"),
            explain(event="E2", actions=["bending"], cause="חשמלאים שמתקינים לד", fact=""),
            {"kind": "correct", "camera": "כניסה ראשית", "place": "stairs", "fact": ""},
        ])
        big.responses[-1] = text(json.dumps(big.responses[-1], ensure_ascii=False))
        agent = self.agent(big)
        self.say(agent, "אלה מקרים תקינים הם מתקינים זה שני אנשים שמרכיבים את הלדים במדרגות", T(13, 54, 7))
        (fact,) = self.activities.live(self.clock)
        big.responses[0] = explain(fact=fact.id, event="E1", actions=["bending"], cause="החשמלאים")
        more = self.say(agent, "אני מדבר על שני האנשים שאחד מהם התכופף זה רגיל", T(13, 54, 31))
        self.assertTrue(more.text.startswith("🧠 הבנתי, גם הכיפוף במדרגות של הכניסה הראשית"), more.text)
        big.responses[0] = explain(fact=fact.id, event="E2", actions=["bending", "car_door"], cause="חשמלאים")
        same = self.say(agent, "חשמלאים שמנסים להתקין לד למדרגות", T(13, 54, 38), alert=RED_1345)
        self.assertTrue(same.text.startswith("🧠 כן, ככה זה אצלי:"), same.text)
        fixed = self.say(agent, "לא בפרגולה אני מתכוון בכניסה הראשית במדרגות", T(13, 54, 47), alert=RED_1345)
        self.assertEqual(fixed.text, "🧠 נכון, במדרגות של הכניסה הראשית.")
        (fact,) = self.activities.live(self.clock)                              # still ONE fact
        self.assertEqual(fact.cameras, [ENTRANCE])                                # never the pergola
        self.assertNotIn("car_door", fact.actions)
        self.assertIn("bending", fact.actions)

    def test_no_known_window_asks_until_when_once(self) -> None:
        for k in self.events.list_known(self.clock):
            self.events.cancel_known(k["id"])
        big = Scripted([explain(event="E1", actions=["lying"], cause="החשמלאים")])
        agent = self.agent(big)
        out = self.say(agent, "זה החשמלאים שעובדים במדרגות, זה רגיל", T(13, 54))
        self.assertTrue(out.text.endswith(t("act_ask_until", "he")))
        self.assertEqual(out.buttons, ("16:00", "17:00", "18:00"))
        done = agent.handle_choice(CHAT, out.question_token, 2, OWNER)
        self.assertEqual(done.text, "🧠 סגור, עד 18:00.")
        (fact,) = self.activities.live(self.clock)
        self.assertEqual(dt.datetime.fromtimestamp(fact.until).strftime("%H:%M"), "18:00")

    def test_the_cancel_button(self) -> None:
        agent = self.agent(Scripted([explain(event="E1", actions=["lying"], cause="החשמלאים")]))
        self.say(agent, "זה החשמלאים שעובדים במדרגות, זה רגיל", T(13, 54))
        (fact,) = self.activities.live(self.clock)
        out = agent.known_button(CHAT, "x", fact.id, OWNER)
        self.assertIn("בוטל", out.text)
        self.assertEqual(self.activities.live(self.clock), [])

    def test_a_question_an_ack_or_none_is_not_an_explanation(self) -> None:
        state = ChatState()
        self.assertFalse(ac.candidate("יש מישהו בחוץ?", state, "E1", T(13, 54), self.activities))
        self.assertFalse(ac.candidate("סבבה", state, "E1", T(13, 54), self.activities))
        self.assertFalse(ac.candidate("זה רגיל", state, None, T(13, 54), self.activities))   # no alert around
        self.assertTrue(ac.candidate("זה רגיל", state, "E1", T(13, 54), self.activities))
        big = Scripted([text('{"kind": "none"}'), call("reply", answer="אני כאן.")])
        out = self.say(self.agent(big), "תעדכן אותי אם משהו משתנה, זה בסדר", T(13, 54))
        self.assertEqual(self.activities.live(self.clock), [])
        self.assertIsNotNone(out)

    def test_what_do_you_remember_lists_the_explained_action(self) -> None:
        agent = self.agent(Scripted([explain(event="E1", actions=["lying", "bending"], place="stairs",
                                             cause="החשמלאים מתקינים לדים")]))
        self.say(agent, "זה החשמלאים שעובדים במדרגות, זה רגיל", T(13, 54))
        out = self.say(agent, "מה אתה זוכר?", T(13, 56))
        self.assertIn("🧠 בכניסה הראשית: שכיבה/כיפוף/זחילה/כלים ביד במדרגות = החשמלאים מתקינים לדים, כל יום עד 18:00",
                      out.text)

    def _tags(self):
        import glob

        return glob.glob(os.path.join(self.root, "feedback", "**", "*.feedback.json"), recursive=True)


class HumanRepliesTest(Base):
    def test_1355_an_ack_gets_a_thumbs_up_never_a_status(self) -> None:
        agent = self.agent(Scripted([]))
        for words in ("סבבה", "בסדר הבנתי", "תודה", "👍"):
            out = self.say(agent, words, T(13, 55), alert=RED_1345)
            self.assertEqual(out.text, "👍", words)

    def test_the_memory_status_goes_out_once_per_half_hour(self) -> None:
        status = "העובדים של הפרגולה כבר מסומנים בפרגולה ובכניסה הראשית עד 18:00."
        big = Scripted([call("reply", answer=status), call("reply", answer=f"כן, אני קורא את ההיסטוריה. {status}")])
        agent = self.agent(big, fast=None)
        first = self.say(agent, "מה קורה עם העובדים בחוץ עכשיו", T(13, 54, 15))
        self.assertEqual(first.text, status)
        again = self.say(agent, "למה אתה חוזר על זה", T(13, 55, 14))
        self.assertNotIn("מסומנים", again.text)

    def test_a_repeated_reply_is_rewritten_once_then_replaced(self) -> None:
        # 13:55:18 and 13:55:31: the same code-written answer twice, word for word.
        big = Scripted([text("כן, קראתי. אמרת שהחשמלאים עובדים במדרגות.")])
        agent = self.agent(big)
        first = self.say(agent, "אתה קורא את ההיסטוריה של השיחה?", T(13, 55, 14)).text
        self.assertTrue(first.startswith("כן."))
        second = self.say(agent, "אתה קורא את ההיסטוריה של השיחה?", T(13, 55, 26)).text
        self.assertEqual(second, "כן, קראתי. אמרת שהחשמלאים עובדים במדרגות.")
        third = self.say(agent, "אתה קורא את ההיסטוריה של השיחה?", T(13, 55, 40)).text
        self.assertEqual(third, t("ack_other", "he"))          # the rewrite failed: a short different line

    def test_same_people_needs_evidence(self) -> None:
        big = Scripted([call("reply", answer="כן, אני בטוח. אלה אותם עובדים מהבוקר.")])
        out = self.say(self.agent(big), "אלה אותם אנשים שעבדו מהבוקר או שהתחלפו?", T(13, 5, 16))
        self.assertEqual(out.text, t("same_unsure", "he"))
        self.assertIn("[SAME PEOPLE EVIDENCE]", "\n".join(big.seen[0]))
        self.assertTrue(rg.asks_same_people("אלה אותם אנשים?"))
        self.assertFalse(rg.overclaims_same("נראה שכן לפי הבגדים", []))

    def test_a_plain_judgement_is_not_a_tag(self) -> None:
        big = Scripted([call("record_verdict", verdict="false_alarm", owner_words="זו אזעקת שווא"),
                        call("reply", answer="הבנתי.")])
        out = self.say(self.agent(big), "זו אזעקת שווא", T(13, 50), alert=RED_1345)
        self.assertFalse(any(r.tool == "record_verdict" for r in out.receipts))
        self.assertTrue(ac.asks_to_tag("תשנה את התיוג לאזעקת שווא"))


class GuardUnitTest(unittest.TestCase):
    def test_near_and_short(self) -> None:
        self.assertTrue(rg.near("כן, אני קורא את ההיסטוריה ובודק את הזיכרון.", "כן, אני קורא את ההיסטוריה ובודק את הזיכרון"))
        self.assertFalse(rg.near("👍", "👍"))

    def test_ack(self) -> None:
        for words in ("סבבה", "בסדר הבנתי", "תודה רבה", "👍", "ok thanks"):
            self.assertTrue(ac.is_ack(words), words)
        for words in ("בסדר, אבל למה?", "הבנתי אתה לא צריך לחזור על זה", "סבבה תשלח תמונה"):
            self.assertFalse(ac.is_ack(words), words)

    def test_definite_hebrew(self) -> None:
        self.assertEqual(ac.definite_he("כניסה ראשית"), "הכניסה הראשית")
        self.assertEqual(ac.definite_he("הפרגולה"), "הפרגולה")
        self.assertEqual(ac.definite_he("מצלמה 6"), "מצלמה 6")


if __name__ == "__main__":
    unittest.main()
