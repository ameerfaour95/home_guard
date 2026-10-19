"""Day activity threads (day_activity.py): by day an ordinary activity is one message, then only its changes.

Owner, 2026-10-19: "send natural message but don't continue sending it if nothing has changed ... you send and then you
ask me (once a day for each unique activity) if you want me to keep updating you. if no then you keep in log. if yes,
if let's say the workers left then update, the workers came back update, etc. if a totally different thing happens
then you update"."""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest import mock

from home_guard_project.box import day_activity as da
from home_guard_project.box import inference
from home_guard_project.box.feedback import AlertIndex, MuteState
from home_guard_project.box.telegram_agent import TelegramInbox
from home_guard_project.box.telegram_notify import TelegramConfig

DAY = dt.datetime(2026, 10, 19, 10, 54).timestamp()      # 10:54 local, daylight
NIGHT = dt.datetime(2026, 10, 19, 21, 0).timestamp()
PERGOLA, ENTRANCE, BACK = "ameer_v2_ch3", "ameer_v2_ch6", "ameer_v2_ch1"
NAMES = {PERGOLA: "הפרגולה", ENTRANCE: "כניסה ראשית", BACK: "מצלמה 1"}
RING = {ENTRANCE: [PERGOLA], PERGOLA: ["ameer_v2_ch8"], "ameer_v2_ch8": [BACK]}


def mins(n: float) -> float:
    return n * 60.0


class Judge:
    """The text model, scripted: each call pops the next answer (None = a failed call)."""

    def __init__(self, *answers: Optional[Dict[str, Any]]) -> None:
        self.answers = list(answers)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, threads: str, camera: str, clock: str, people: int, summary: str) -> Optional[Dict[str, Any]]:
        self.calls.append({"threads": threads, "camera": camera, "clock": clock, "people": people, "summary": summary})
        return self.answers.pop(0) if self.answers else None


class Base(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "day_activity.json")
        self.clock = [DAY]

    def store(self, judge: Any = None, dark: Any = None) -> da.DayActivity:
        return da.DayActivity(self.path, neighbours=RING, judge=judge, names=lambda c: NAMES.get(c, c), lang="he",
                              is_dark=dark or (lambda ts: False), now=lambda: self.clock[0])

    def asked(self, store: da.DayActivity, step: da.Step, ts: float, people: int = 1) -> None:
        """The question reached him (inference calls ``sent`` after a delivery)."""
        store.sent(step.thread["id"], step.action, people, ts, "-5", 100 + len(store.threads))


class NewAndSameTest(Base):
    def test_a_new_activity_is_one_message_with_the_question_and_its_model_name(self) -> None:
        judge = Judge({"same": "", "name": "העובדים בפרגולה", "plural": True})
        s = self.store(judge)
        step = s.look(PERGOLA, DAY, 2, "Two workers carry a ladder", "a1")
        self.assertEqual((step.action, step.new), (da.ASK, True))
        self.assertEqual(step.thread["name_he"], "העובדים בפרגולה")
        self.assertEqual(len(judge.calls), 1)
        self.assertIn("(none yet today)", judge.calls[0]["threads"])

    def test_the_same_activity_still_here_is_quiet_without_a_model_call(self) -> None:
        judge = Judge({"same": "", "name": "העובדים בפרגולה", "plural": True})
        s = self.store(judge)
        first = s.look(PERGOLA, DAY, 1, "a worker on a ladder", "a1")
        self.asked(s, first, DAY)
        again = s.look(PERGOLA, DAY + mins(5), 1, "a worker walks with a hose", "a2")
        self.assertEqual(again.action, da.QUIET)
        self.assertIn("asked, no answer yet", again.reason)
        self.assertEqual(len(judge.calls), 1)                  # only the first look asked the model
        self.assertEqual(again.thread["id"], first.thread["id"])

    def test_a_neighbour_camera_joins_the_thread_still_here_without_a_model_call(self) -> None:
        judge = Judge({"same": "", "name": "העובדים", "plural": True})
        s = self.store(judge)
        first = s.look(PERGOLA, DAY, 1, "a worker", "a1")
        self.asked(s, first, DAY)
        step = s.look(ENTRANCE, DAY + mins(3), 1, "a worker with a cable at the entrance", "a2")
        self.assertEqual((step.action, step.thread["id"]), (da.QUIET, first.thread["id"]))
        self.assertEqual(step.thread["cameras"], [PERGOLA, ENTRANCE])
        self.assertEqual(len(judge.calls), 1)

    def test_a_far_camera_asks_the_model_and_a_totally_different_thing_is_its_own_message(self) -> None:
        judge = Judge({"same": "", "name": "העובדים", "plural": True},
                      {"same": "", "name": "שליח ליד הדלת", "plural": False})
        s = self.store(judge)
        first = s.look(PERGOLA, DAY, 2, "workers", "a1")
        self.asked(s, first, DAY)
        step = s.look(BACK, DAY + mins(4), 1, "a courier leaves a box at the door", "a2")
        self.assertEqual((step.action, step.new), (da.ASK, True))
        self.assertEqual(step.thread["name_he"], "שליח ליד הדלת")
        self.assertIn(first.thread["id"], judge.calls[1]["threads"])
        self.assertIn("העובדים", judge.calls[1]["threads"])

    def test_the_model_says_same_activity_after_a_break(self) -> None:
        s = self.store(Judge({"same": "", "name": "העובדים", "plural": True}, None))
        first = s.look(PERGOLA, DAY, 2, "workers", "a1")
        self.asked(s, first, DAY)
        s.judge = Judge({"same": first.thread["id"], "name": "", "plural": False})
        step = s.look(BACK, DAY + mins(40), 1, "a worker carries planks", "a2")
        self.assertEqual((step.action, step.thread["id"]), (da.QUIET, first.thread["id"]))
        self.assertEqual(step.thread["matched_by"], "model")

    def test_a_failed_model_joins_todays_thread_at_this_camera_else_starts_one(self) -> None:
        s = self.store(Judge(None, None, None))
        first = s.look(PERGOLA, DAY, 1, "a worker", "a1")
        self.assertEqual(first.thread["name_he"], "אדם בפרגולה")           # the rules' own name
        self.asked(s, first, DAY)
        later = s.look(PERGOLA, DAY + mins(50), 1, "a worker again", "a2")
        self.assertEqual((later.thread["id"], later.thread["matched_by"]), (first.thread["id"], "fallback"))
        other = s.look(BACK, DAY + mins(55), 2, "two people", "a3")
        self.assertTrue(other.new)
        self.assertEqual(other.thread["name_he"], "אנשים במצלמה 1")

    def test_no_model_at_all_uses_the_rules(self) -> None:
        s = self.store(judge=None)
        step = s.look(ENTRANCE, DAY, 1, "a man", "a1")
        self.assertEqual((step.action, step.thread["name_he"]), (da.ASK, "אדם בכניסה ראשית"))

    def test_the_model_is_asked_at_most_the_daily_cap(self) -> None:
        judge = Judge(*([None] * (da.MODEL_CALLS_PER_DAY + 5)))
        s = self.store(judge)
        for i in range(da.MODEL_CALLS_PER_DAY + 3):
            s.look(BACK if i % 2 else PERGOLA, DAY + mins(16 * i), 1, "x", f"a{i}")    # same day, never "still here"
        self.assertEqual(len(judge.calls), da.MODEL_CALLS_PER_DAY)

    def test_an_unknown_thread_id_from_the_model_starts_a_new_one(self) -> None:
        s = self.store(Judge({"same": "", "name": "העובדים", "plural": True},
                             {"same": "nosuch", "name": "", "plural": False}))
        first = s.look(PERGOLA, DAY, 1, "a worker", "a1")
        self.asked(s, first, DAY)
        step = s.look(BACK, DAY + mins(30), 1, "someone", "a2")
        self.assertTrue(step.new)


class OnceADayTest(Base):
    def test_the_question_is_asked_once_a_day_per_thread(self) -> None:
        s = self.store()
        first = s.look(PERGOLA, DAY, 1, "a worker", "a1")
        self.asked(s, first, DAY)
        for i in range(1, 8):
            self.assertEqual(s.look(PERGOLA, DAY + mins(2 * i), 1, "still working", f"a{i}").action, da.QUIET)

    def test_a_question_that_did_not_reach_him_is_asked_on_the_next_event(self) -> None:
        s = self.store()
        s.look(PERGOLA, DAY, 1, "a worker", "a1")                    # the send failed: no sent()
        self.assertEqual(s.look(PERGOLA, DAY + mins(2), 1, "a worker", "a2").action, da.ASK)

    def test_a_new_day_starts_empty_and_asks_again(self) -> None:
        s = self.store()
        first = s.look(PERGOLA, DAY, 1, "a worker", "a1")
        self.asked(s, first, DAY)
        s.set_follow(first.thread["id"], False)
        tomorrow = DAY + 24 * 3600
        self.clock[0] = tomorrow
        step = s.look(PERGOLA, tomorrow, 1, "a worker", "b1")
        self.assertEqual((step.action, step.new), (da.ASK, True))
        self.assertEqual(len(s.threads), 1)

    def test_the_file_keeps_today_across_a_restart_and_not_yesterday(self) -> None:
        s = self.store()
        first = s.look(PERGOLA, DAY, 1, "a worker", "a1")
        self.asked(s, first, DAY)
        s.set_follow(first.thread["id"], True)
        again = self.store()
        self.assertEqual((again.threads[0].id, again.threads[0].follow), (first.thread["id"], True))
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["day"], "2026-10-19")
        self.clock[0] = DAY + 24 * 3600
        self.assertEqual(self.store().threads, [])


class FollowTest(Base):
    def start(self, follow: Optional[bool], people: int = 2) -> (da.DayActivity, str):
        s = self.store()
        first = s.look(PERGOLA, DAY, people, "workers", "a1")
        self.asked(s, first, DAY, people)
        if follow is not None:
            s.set_follow(first.thread["id"], follow)
        return s, first.thread["id"]

    def test_no_means_only_the_log_even_when_they_leave_and_come_back(self) -> None:
        s, tid = self.start(False)
        self.assertEqual(s.tick(DAY + mins(25)), [])                   # left, quietly
        self.assertFalse(s.find(tid).present)
        step = s.look(PERGOLA, DAY + mins(40), 5, "workers back", "a2")
        self.assertEqual((step.action, step.thread["id"]), (da.QUIET, tid))
        self.assertIn("he said log only", step.reason)

    def test_yes_tells_when_they_left_with_the_last_time_seen(self) -> None:
        s, tid = self.start(True)
        self.assertEqual(s.tick(DAY + mins(15)), [])                   # not yet
        (notice,) = s.tick(DAY + mins(21))
        self.assertEqual(notice["kind"], da.LEFT)
        self.assertEqual(notice["text"], "⚪ אנשים בפרגולה יצאו · הפרגולה · 11:15 (נראו לאחרונה ב-10:54)")
        self.assertEqual(notice["reply_to"]["message_id"], 101)
        self.assertEqual(s.tick(DAY + mins(30)), [])                   # once

    def test_detector_looks_keep_them_here(self) -> None:
        s, tid = self.start(True)
        for m in range(1, 60, 5):
            s.seen(PERGOLA, DAY + mins(m), 1)
            self.assertEqual(s.tick(DAY + mins(m) + 30), [])
        self.assertTrue(s.find(tid).present)
        s.seen(BACK, DAY + mins(70), 1)                                 # another camera does not keep them
        self.assertEqual(len(s.tick(DAY + mins(56) + mins(21))), 1)

    def test_yes_tells_when_they_came_back_and_then_nothing_while_they_stay(self) -> None:
        s, tid = self.start(True)
        s.tick(DAY + mins(25))
        back = s.look(PERGOLA, DAY + mins(40), 2, "workers carry tools", "a2")
        self.assertEqual((back.action, back.thread["id"]), (da.CAME_BACK, tid))
        s.sent(tid, back.action, 2, DAY + mins(40), "-5", 300)
        self.assertEqual(s.look(PERGOLA, DAY + mins(42), 2, "still working", "a3").action, da.QUIET)
        text = s.update_text(back, PERGOLA, DAY + mins(40), "שני עובדים נושאים כלים.", 2)
        self.assertEqual(text, "⚪ אנשים בפרגולה חזרו · הפרגולה · 11:34\nמה קורה: שני עובדים נושאים כלים.")

    def test_yes_tells_once_about_clearly_more_people(self) -> None:
        s, tid = self.start(True, people=2)
        self.assertEqual(s.look(PERGOLA, DAY + mins(3), 3, "three workers", "a2").action, da.QUIET)
        more = s.look(PERGOLA, DAY + mins(4), 4, "four workers", "a3")
        self.assertEqual(more.action, da.MORE)
        self.assertEqual(s.update_text(more, PERGOLA, DAY + mins(4), "", 4),
                         "⚪ אנשים בפרגולה: עוד 2 אנשים הגיעו · הפרגולה · 10:58")
        s.sent(tid, more.action, 4, DAY + mins(4), "-5", 301)
        self.assertEqual(s.look(PERGOLA, DAY + mins(5), 4, "four workers", "a4").action, da.QUIET)

    def test_unanswered_is_quiet_too(self) -> None:
        s, tid = self.start(None)
        s.tick(DAY + mins(25))
        self.assertEqual(s.look(PERGOLA, DAY + mins(40), 2, "workers", "a2").action, da.QUIET)

    def test_no_left_line_after_dark(self) -> None:
        s = self.store(dark=lambda ts: ts >= NIGHT)
        first = s.look(PERGOLA, NIGHT - mins(30), 1, "a worker", "a1")
        self.asked(s, first, NIGHT - mins(30))
        s.set_follow(first.thread["id"], True)
        self.assertEqual(s.tick(NIGHT + mins(1)), [])
        self.assertFalse(s.find(first.thread["id"]).present)

    def test_changing_his_mind_and_the_receipts(self) -> None:
        s, tid = self.start(None)
        yes = s.set_follow(tid, True, "Ameer")
        self.assertEqual((yes["changed"], yes["receipt"]), (True, "סבבה, אעדכן רק כשהם יוצאים או חוזרים."))
        self.assertFalse(s.set_follow(tid, True)["changed"])           # the same tap again: no second receipt
        no = s.set_follow(tid, False)
        self.assertEqual((no["changed"], no["receipt"]), (True, "סבבה, היום רק ביומן."))
        self.assertFalse(s.set_follow("gone1234", True)["found"])

    def test_the_buttons_mark_his_choice_and_keep_the_other(self) -> None:
        s, tid = self.start(None)
        rows = s.buttons(tid, True)["inline_keyboard"]
        self.assertEqual([r[0]["callback_data"] for r in rows], [f"da:y:{tid}", f"da:n:{tid}"])
        self.assertTrue(rows[0][0]["text"].startswith("✓ 🔔"))
        self.assertEqual(rows[1][0]["text"], "🔕 לא, רק ביומן")
        self.assertLessEqual(len(f"da:y:{tid}".encode()), 64)


class KnownMarkTest(Base):
    def test_the_owners_mark_makes_it_log_only(self) -> None:
        s = self.store()
        step = s.look(PERGOLA, DAY, 3, "workers", "a1", known_text="העובדים של הפרגולה")
        self.assertEqual((step.action, step.thread["follow"]), (da.QUIET, False))
        self.assertEqual(s.look(PERGOLA, DAY + mins(2), 3, "workers", "a2").action, da.QUIET)

    def test_his_yes_wins_over_the_mark(self) -> None:
        s = self.store()
        step = s.look(PERGOLA, DAY, 3, "workers", "a1", known_text="העובדים")
        self.assertTrue(s.set_follow(step.thread["id"], True)["changed"])


class OwnerWordsTest(Base):
    def setUp(self) -> None:
        super().setUp()
        self.s = self.store(Judge({"same": "", "name": "העובדים בפרגולה", "plural": True}))
        first = self.s.look(PERGOLA, DAY, 2, "workers", "a1")
        self.asked(self.s, first, DAY)
        self.tid = first.thread["id"]

    def test_stop_by_name(self) -> None:
        self.assertEqual(self.s.owner_words("תפסיק לעדכן על העובדים"), "סבבה, היום רק ביומן.")
        self.assertIs(self.s.find(self.tid).follow, False)

    def test_start_by_name(self) -> None:
        self.assertEqual(self.s.owner_words("תעדכן אותי על העובדים"), "סבבה, אעדכן רק כשהם יוצאים או חוזרים.")
        self.assertIs(self.s.find(self.tid).follow, True)

    def test_a_reply_to_its_message(self) -> None:
        self.assertIsNotNone(self.s.owner_words("אל תעדכן", {"alert_id": "a1", "day_thread": self.tid}))
        self.assertIs(self.s.find(self.tid).follow, False)

    def test_not_a_watch_request_nor_a_long_message_nor_another_subject(self) -> None:
        self.assertIsNone(self.s.owner_words("תעדכן אותי כשהטנדר יוצא"))
        self.assertIsNone(self.s.owner_words("תעדכן אותי על הרכב הלבן"))
        self.assertIsNone(self.s.owner_words(
            "תפסיק לעדכן על העובדים הם שלי עד 17:00 כל השבוע ככה זה"))
        self.assertIsNone(self.s.owner_words("מה קורה בחוץ?"))
        self.assertIsNone(self.s.find(self.tid).follow)

    def test_v3_answers_his_words_without_a_model(self) -> None:
        from home_guard_project.box.assistant_v3 import agent as v3

        da.set_current(self.s)
        self.addCleanup(da.set_current, None)
        self.assertEqual(v3._day_activity_words("תפסיק לעדכן על העובדים", None, {"name": "Ameer"}),
                         "סבבה, היום רק ביומן.")
        self.assertEqual(v3._day_activity_words("מה קורה?", None, {}), "")


class ButtonTest(Base):
    def test_a_tap_sets_follow_marks_the_button_and_says_the_receipt_once(self) -> None:
        s = self.store()
        first = s.look(PERGOLA, DAY, 2, "workers", "a1")
        self.asked(s, first, DAY)
        tid = first.thread["id"]
        da.set_current(s)
        self.addCleanup(da.set_current, None)
        calls: List[Dict[str, Any]] = []

        def post(token: str, method: str, fields: Dict[str, str], timeout: float = 15.0) -> Dict[str, Any]:
            calls.append({"method": method, "fields": fields})
            return {"ok": True, "result": {"message_id": 900 + len(calls)}}

        d = tempfile.mkdtemp()
        inbox = TelegramInbox(TelegramConfig(bot_token="T", chat_ids=["-5"]), None,
                              AlertIndex(os.path.join(d, "i.json")), MuteState(os.path.join(d, "m.json")), d,
                              os.path.join(d, "o.json"), post=post, now=lambda: DAY)
        tap = {"id": "q1", "data": f"da:y:{tid}", "from": {"id": 1, "first_name": "Ameer"},
               "message": {"chat": {"id": -5}, "message_id": 101}}
        inbox._on_button(tap)
        self.assertIs(s.find(tid).follow, True)
        methods = [c["method"] for c in calls]
        self.assertEqual(methods, ["answerCallbackQuery", "editMessageReplyMarkup", "sendMessage"])
        keyboard = json.loads(calls[1]["fields"]["reply_markup"])["inline_keyboard"]
        self.assertTrue(keyboard[0][0]["text"].startswith("✓ 🔔"))
        self.assertEqual(calls[2]["fields"]["text"], "סבבה, אעדכן רק כשהם יוצאים או חוזרים.")
        calls.clear()
        inbox._on_button(dict(tap, id="q2"))                           # the same tap again: no second receipt
        self.assertNotIn("sendMessage", [c["method"] for c in calls])
        calls.clear()
        inbox._on_button(dict(tap, id="q3", data=f"da:n:{tid}"))       # changed his mind
        self.assertIs(s.find(tid).follow, False)
        self.assertEqual(calls[-1]["fields"]["text"], "סבבה, היום רק ביומן.")

    def test_a_tap_on_yesterdays_message_only_says_so(self) -> None:
        s = self.store()
        da.set_current(s)
        self.addCleanup(da.set_current, None)
        calls: List[str] = []
        d = tempfile.mkdtemp()
        inbox = TelegramInbox(TelegramConfig(bot_token="T", chat_ids=["-5"]), None,
                              AlertIndex(os.path.join(d, "i.json")), MuteState(os.path.join(d, "m.json")), d,
                              os.path.join(d, "o.json"),
                              post=lambda t, m, f, timeout=15.0: calls.append(m) or {"ok": True}, now=lambda: DAY)
        inbox._on_button({"id": "q", "data": "da:n:old12345", "from": {"id": 1},
                          "message": {"chat": {"id": -5}, "message_id": 5}})
        self.assertEqual(calls, ["answerCallbackQuery"])


class InferenceTest(Base):
    """The guard's side: scope, the message format, and that suspicious / night are untouched."""

    def setUp(self) -> None:
        super().setUp()
        self.s = self.store()
        patcher = mock.patch.object(inference, "DAY_ACTIVITY", self.s)
        patcher.start()
        self.addCleanup(patcher.stop)
        events = mock.patch.object(inference, "EVENTS", None)
        events.start()
        self.addCleanup(events.stop)
        self.sent: List[Dict[str, Any]] = []

        def message(cfg: Any, text: str, **kw: Any) -> Dict[str, Any]:
            self.sent.append({"text": text, **kw})
            return {"sent": True, "results": [{"chat_id": "-5", "ok": True, "message_id": 700 + len(self.sent)}]}

        def photo(cfg: Any, image: bytes, caption: str = "", **kw: Any) -> Dict[str, Any]:
            return message(cfg, caption, photo=True, **kw)

        for name, fake in (("send_photo", photo), ("send_message", message)):
            p = mock.patch(f"home_guard_project.box.telegram_notify.{name}", side_effect=fake)
            p.start()
            self.addCleanup(p.stop)
        for name, value in (("camera_display", lambda c, lang: NAMES.get(c, c)), ("owner_guard", lambda t, c, l: t)):
            p = mock.patch.object(inference, name, value)
            p.start()
            self.addCleanup(p.stop)

    def look(self, label: str = "normal", ts: float = DAY, notify: bool = False, where: Optional[dict] = None,
             people: int = 1, summary: str = "A worker carries a ladder", alert_id: str = "a1",
             assistant: Any = None) -> Optional[Dict[str, Any]]:
        event = SimpleNamespace(notify=notify, reason="normal: kept in the event, not sent")
        return inference._day_activity_look(PERGOLA, ts, label, people, summary, "עובד נושא סולם", event,
                                            where if where is not None else {"on": "mine"}, ("person",), None, [],
                                            alert_id, "he", {"owner_language": "he"}, {}, assistant)

    def test_scope(self) -> None:
        ev = SimpleNamespace(notify=False)
        scope = inference.day_activity_out_of_scope
        self.assertEqual(scope("normal", ev, {"on": "mine"}, 1, ("person",), DAY, self.s), "")
        self.assertEqual(scope("suspicious", ev, {}, 1, ("person",), DAY, self.s), "label suspicious")
        self.assertEqual(scope("escalation", ev, {}, 1, ("person",), DAY, self.s), "label escalation")
        self.assertEqual(scope("normal", SimpleNamespace(notify=True), {}, 1, ("person",), DAY, self.s),
                         "no event, or already sent")
        self.assertEqual(scope("normal", ev, {"off_our_ground": True}, 1, ("person",), DAY, self.s), "not our ground")
        self.assertEqual(scope("normal", ev, {"on": "neighbour"}, 1, ("person",), DAY, self.s), "not our ground")
        self.assertEqual(scope("normal", ev, {}, 0, ("person",), DAY, self.s), "no person to tell about")
        self.assertEqual(scope("normal", ev, {}, 1, ("vehicle",), DAY, self.s), "no person to tell about")
        self.assertEqual(scope("normal", ev, {}, 1, ("person",), DAY, None), "day activity off")
        night = self.store(dark=lambda ts: True)
        self.assertEqual(scope("normal", ev, {}, 1, ("person",), NIGHT, night), "dark")

    def test_the_first_message_in_the_agreed_format_then_quiet(self) -> None:
        out = self.look()
        self.assertEqual((out["action"], out["sent"]), (da.ASK, True))
        (msg,) = self.sent
        self.assertEqual(msg["text"], "⚪ פעילות · הפרגולה · 10:54\nמה קורה: עובד נושא סולם.\n"
                                      "למה הודעתי: אדם בשטח שלנו ביום.\nלהמשיך לעדכן אותך על זה היום?")
        self.assertTrue(msg["silent"])
        self.assertIn("da:y:", msg["reply_markup"])
        self.assertEqual(self.s.threads[0].messages[0]["message_id"], 701)
        again = self.look(ts=DAY + mins(2), alert_id="a2")
        self.assertEqual((again["action"], len(self.sent)), (da.QUIET, 1))

    def test_suspicious_and_night_are_untouched(self) -> None:
        self.assertIsNone(self.look(label="suspicious"))
        self.assertIsNone(self.look(label="escalation"))
        with mock.patch.object(self.s, "is_dark", lambda ts: True):
            self.assertIsNone(self.look(ts=NIGHT))
        self.assertEqual((self.sent, self.s.threads), ([], []))

    def test_followed_came_back_replies_in_the_thread(self) -> None:
        self.look()
        self.s.set_follow(self.s.threads[0].id, True)
        self.s.tick(DAY + mins(25))
        out = self.look(ts=DAY + mins(40), alert_id="a2")
        self.assertEqual(out["action"], da.CAME_BACK)
        self.assertTrue(self.sent[-1]["text"].startswith("⚪ אדם בפרגולה חזר · הפרגולה · 11:34"))
        self.assertEqual(self.sent[-1]["reply_to"]["message_id"], 701)

    def test_the_message_is_filed_with_the_assistant(self) -> None:
        index = AlertIndex(os.path.join(tempfile.mkdtemp(), "i.json"))
        notes: List[Any] = []
        assistant = SimpleNamespace(index=index, note_line=lambda *a: notes.append(a))
        self.look(assistant=assistant)
        self.assertEqual(index.lookup("-5", 701)["day_thread"], self.s.threads[0].id)
        self.assertEqual(notes[0][0], PERGOLA)

    def test_the_tick_sends_the_left_line_off_thread(self) -> None:
        self.look()
        self.s.set_follow(self.s.threads[0].id, True)
        with mock.patch.object(inference, "_day_ticked", 0.0), \
                mock.patch.object(inference.threading, "Thread") as thread:
            inference.day_activity_tick(DAY + mins(21), {}, {})
            target = thread.call_args.kwargs["target"]
        target()
        self.assertTrue(self.sent[-1]["text"].startswith("⚪ אדם בפרגולה יצא · הפרגולה · 11:15"))
        self.assertEqual(self.s.threads[0].left_told_at, DAY + mins(21))


class ModelJudgeTest(unittest.TestCase):
    def test_the_prompt_and_the_answer(self) -> None:
        seen: List[Any] = []

        class Model:
            def chat(self, messages: Any, tools: Any, json_schema: Any = None, max_tokens: int = 0) -> Any:
                seen.append(messages[0]["content"])
                return SimpleNamespace(error="", content='{"same": "ab12cd34", "name": "", "plural": false}')

        out = da.model_judge(Model())("- ab12cd34: \"העובדים\"", "הפרגולה", "11:30", 2, "a worker")
        self.assertEqual(out, {"same": "ab12cd34", "name": "", "plural": False})
        self.assertIn('"הפרגולה" at 11:30 (2 people in view)', seen[0])

    def test_a_failure_is_none(self) -> None:
        model = SimpleNamespace(chat=lambda *a, **k: SimpleNamespace(error="timeout", content=""))
        self.assertIsNone(da.model_judge(model)("", "x", "11:00", 1, "y"))


class SwitchTest(unittest.TestCase):
    def test_on_by_default_and_off_by_box_yaml(self) -> None:
        self.assertTrue(da.on_of({}))
        self.assertFalse(da.on_of({"day_activity": "off"}))
        self.assertFalse(da.on_of({"day_activity": False}))
        self.assertEqual(da.minutes_of({"day_activity_left_min": 30}, "day_activity_left_min", da.LEFT_SEC), 1800.0)
        self.assertEqual(da.minutes_of({}, "day_activity_left_min", da.LEFT_SEC), da.LEFT_SEC)

    def test_build_with_the_model_off(self) -> None:
        d = tempfile.mkdtemp()
        store = da.build({"day_activity_model": "off", "camera_neighbours": RING}, {}, names=str, lang="he",
                         path=os.path.join(d, "x.json"))
        self.addCleanup(da.set_current, None)
        self.assertIs(da.current(), store)
        self.assertIsNone(store.judge)
        self.assertIn(PERGOLA, store.neighbours[ENTRANCE])
        self.assertIsNone(da.build({"day_activity": "off"}, {}, names=str, lang="he"))
        self.assertIsNone(da.current())


if __name__ == "__main__":
    unittest.main()
