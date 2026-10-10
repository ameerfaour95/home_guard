# -*- coding: utf-8 -*-
"""Assistant v3 (home_guard_project/box/assistant_v3/): the deterministic parts, with stub models (no network).

- the acts' quoted slots (no invented time can reach memory), the time words, the supervisor's named checks;
- whole turns through the golden suite's stubbed box (eval_chat.World): a place is kept for good with no question,
  an ack is a 👍 with no model call, a pause files nothing in feedback/, a crew with no hour gets ONE question with
  buttons and the tapped hour completes the mark in code, a complaint re-does what the last statement should have
  saved, the box.yaml switch picks v3 only when asked.
"""
import datetime as dt
import json
import os
import unittest
from typing import Any, Callable, Dict, List

from home_guard_project.box import eval_chat as ec
from home_guard_project.box.assistant_v3 import supervisor as sv
from home_guard_project.box.assistant_v3 import wants_v3
from home_guard_project.box.assistant_v3.acts import validate
from home_guard_project.box.assistant_v3.timeparse import parse_until
from home_guard_project.box.brain.models import ModelMessage

DAY = "2026-10-10"


def at(clock: str, day: str = DAY) -> float:
    return dt.datetime.strptime(f"{day} {clock}", "%Y-%m-%d %H:%M").timestamp()


class FakeModel:
    """A chat model that answers from a function of the last user message (and counts its calls)."""

    def __init__(self, answer: Callable[[str, Dict[str, Any]], str]) -> None:
        self.answer, self.calls = answer, []

    def chat(self, messages: List[Dict[str, Any]], tools: Any = None, tool_choice: Any = None,
             **kwargs: Any) -> ModelMessage:
        user = str(messages[-1].get("content") or "")
        self.calls.append({"user": user, "kwargs": kwargs, "tools": bool(tools)})
        return ModelMessage(content=self.answer(user, kwargs), usage=(100, 20))


def acts_for(table: Dict[str, Dict[str, Any]]) -> FakeModel:
    """Understanding: the acts JSON for the owner's message (found in the context's last section)."""

    def answer(user: str, kwargs: Dict[str, Any]) -> str:
        message = user.rsplit(":\n", 1)[-1].split("\n")[0].strip()
        return json.dumps(table.get(message, {"emotion": "neutral", "acts": [{"act": "unclear"}]}), ensure_ascii=False)

    return FakeModel(answer)


def act(kind: str, **slots: Any) -> Dict[str, Any]:
    out = {"act": kind, "quote": None, "camera": None, "event": None, "subject": None, "until_quote": None,
           "scope": "", "command": "", "value": None, "verdict": "", "issue": "", "routine": False,
           "look_again": False, "earlier": False}
    out.update(slots)
    return out


CAMS = ["ameer_v2_ch2", "ameer_v2_ch3", "ameer_v2_ch6"]
HOUSE = {"cameras": CAMS, "aliases": {"ameer_v2_ch3": ["פרגולה"], "ameer_v2_ch6": ["כניסה ראשית"]}}
ALERT = {"id": "ameer_v2_ch2_1_alert", "camera": "ameer_v2_ch2", "time": "13:35:21", "label": "suspicious",
         "summary": "אדם הולך לאורך הקיר ליד חלון ומביט פנימה."}


def run(case: Dict[str, Any], understand: FakeModel, writer: FakeModel, critic: Any = None) -> Dict[str, Any]:
    base = {"id": "t", "source": "test", "date": DAY, "house": HOUSE, "expect": {"intent": "ack", "ideal": "x",
                                                                                  "max_questions": 0}}
    base.update(case)
    ec.V3.clear()
    ec.V3.update(write=writer, understand=understand, critic=critic, escalate=None)
    try:
        return ec.run_case(base, None, None)
    finally:
        ec.V3.clear()


class ActsTests(unittest.TestCase):
    def test_a_time_he_never_said_is_dropped(self):
        raw = {"emotion": "neutral", "acts": [act("person_mark", quote="עובדים על הפרגולה", subject="העובדים",
                                                   until_quote="עד 23:59", camera="cam3")]}
        und = validate(raw, "זה תקין עובדים על הפרגולה", [], {"cam3": "ameer_v2_ch3"}, [])
        self.assertEqual(und.acts[0].until_quote, "")
        self.assertIn("until_quote", und.acts[0].dropped)
        self.assertEqual(und.acts[0].camera, "ameer_v2_ch3")

    def test_an_earlier_act_may_quote_his_earlier_words_only(self):
        raw = {"emotion": "confused", "acts": [act("place_fact", quote="זה הבית של השכן", earlier=True),
                                                act("place_fact", quote="זה הבית של השכן")]}
        und = validate(raw, "מה קשר? לא הבנתי", ["זה הבית של השכן"], {}, [])
        self.assertEqual(len(und.acts), 1)               # a memory act with no real quote is dropped whole
        self.assertEqual(und.acts[0].quote, "זה הבית של השכן")

    def test_unknown_camera_and_handle_are_dropped(self):
        und = validate({"acts": [act("question_live", camera="cam9", event="E44")]}, "מה קורה", [], {}, ["E1"])
        self.assertEqual((und.acts[0].camera, und.acts[0].event), ("", ""))

    def test_nothing_parsed_is_unclear(self):
        self.assertEqual(validate({}, "x", [], {}, []).kinds(), ["unclear"])


class TimeTests(unittest.TestCase):
    def test_words(self):
        noon = at("12:00")
        self.assertEqual(dt.datetime.fromtimestamp(parse_until("עד 18:00", noon)).strftime("%H:%M"), "18:00")
        self.assertEqual(dt.datetime.fromtimestamp(parse_until("הם מסיימים ב17", noon)).strftime("%H:%M"), "17:00")
        self.assertEqual(dt.datetime.fromtimestamp(parse_until("עד שש", noon)).strftime("%H:%M"), "18:00")
        late = parse_until("תכבה עד היום בלילה ותדליק אותם שוב ב 12 בלילה", at("17:11"))
        self.assertEqual(dt.datetime.fromtimestamp(late).strftime("%d %H:%M"), "11 00:00")
        self.assertIsNone(parse_until("עובדים על הפרגולה", noon))


class SupervisorTests(unittest.TestCase):
    def check(self, text: str, **kw: Any) -> sv.Verdict:
        args = dict(lang="he", last_replies=[], allowed_times="", receipts=[], memory_subjects=[], owner_text="",
                    may_ask=False, act_kinds=["question_live"])
        args.update(kw)
        return sv.code_checks(text, **args)

    def test_hard_failures_are_named(self):
        self.assertIn("english", self.check("I could not work on that right now").reason())
        self.assertIn("internal_id", self.check("במצלמה ameer_v2_ch6 שקט").reason())
        self.assertIn("generic_question", self.check("מה לתקן?").reason())
        self.assertIn("closing_offer", self.check("שקט בחוץ. אם תרצה משהו נוסף, אני כאן!").reason())
        self.assertIn("empty_empathy", self.check("אני מבין את התסכול שלך.").reason())
        self.assertIn("invented_time", self.check("הם עובדים עד 23:59.").reason())
        self.assertTrue(self.check("הם עובדים עד 18:00.", allowed_times="18:00").ok)

    def test_repetition_of_a_recent_reply(self):
        said = "העובדים של הפרגולה כבר מסומנים בפרגולה ובכניסה הראשית עד 18:00."
        self.assertIn("repetition", self.check(said, last_replies=[said], allowed_times="18:00").reason())

    def test_memory_recital_vs_saying_what_is_seen(self):
        subject = "העובדים של הפרגולה"
        dump = "אגב, העובדים של הפרגולה מסומנים עד יום ה׳."
        self.assertIn("memory_recital", self.check(dump, memory_subjects=[subject], owner_text="זה הבית של השכן",
                                                   act_kinds=["place_fact"]).reason())
        self.assertIn("memory_recital", self.check("סגור, אשאל על העובדים בדרך אגב.", memory_subjects=[subject],
                                                   owner_text="תשאל בדרך אגב", act_kinds=["preference"],
                                                   strict_memory="single").reason())
        seen = "בפרגולה שלושה עובדים ליד הטריילר."
        self.assertTrue(self.check(seen, memory_subjects=[subject], owner_text="יש מישהו בחוץ?").ok)

    def test_an_unbacked_save_claim(self):
        self.assertIn("unbacked_claim", self.check("שמרתי את זה.", act_kinds=["preference"]).reason())
        self.assertTrue(self.check("אני זוכר שבמצלמה 2 זה הבית של השכן.", act_kinds=["question_memory"]).ok)

    def test_salvage_keeps_the_passing_sentences(self):
        from home_guard_project.box.assistant_v3.agent import salvage

        check = lambda text: self.check(text)  # noqa: E731
        self.assertEqual(salvage("בפרגולה שלושה עובדים. הם עובדים עד 23:59.", check), "בפרגולה שלושה עובדים.")

    def test_sanitize(self):
        self.assertNotIn("E7", sv.sanitize("ההתראה (E7) נסגרה", [], "he"))


class TurnTests(unittest.TestCase):
    def test_a_place_is_kept_for_good_with_no_question(self):
        understand = acts_for({"זה הבית של השכן": {"emotion": "neutral", "acts": [
            act("place_fact", quote="זה הבית של השכן", camera="cam2", subject="הבית של השכן")]}})
        writer = FakeModel(lambda user, kw: "הבנתי, זה הבית של השכן. מה שקורה שם לא יקפיץ לך התראה.")
        out = run({"time": "13:48:00", "alerts": [ALERT],
                   "message": {"text": "זה הבית של השכן", "reply_to": ALERT["id"]}}, understand, writer)
        self.assertEqual(out["error"], "")
        places = [w for w in out["writes"] if w["type"] == "place_fact"]
        self.assertEqual(len(places), 1)
        self.assertEqual(places[0]["camera"], "ameer_v2_ch2")
        self.assertIsNone(places[0]["until"])
        self.assertNotIn("?", out["text"])
        self.assertFalse([w for w in out["writes"] if w["type"] in ("tag", "person_mark")])

    def test_an_ack_is_a_thumb_with_no_writer_call(self):
        understand = acts_for({"סבבה": {"emotion": "happy", "acts": [act("ack", quote="סבבה")]}})
        writer = FakeModel(lambda user, kw: "should not be called")
        out = run({"time": "13:55:04", "message": {"text": "סבבה"}}, understand, writer)
        self.assertEqual(out["text"], "👍")
        self.assertEqual(writer.calls, [])

    def test_a_pause_files_nothing_in_feedback(self):
        understand = acts_for({"תכבה את המצלמות עד 18:00": {"emotion": "neutral", "acts": [
            act("command", quote="תכבה את המצלמות עד 18:00", command="pause", camera="house",
                until_quote="עד 18:00")]}})
        writer = FakeModel(lambda user, kw: "סגור, אין התראות עד 18:00.")
        out = run({"time": "15:08:16", "alerts": [ALERT],
                   "message": {"text": "תכבה את המצלמות עד 18:00", "reply_to": ALERT["id"]}}, understand, writer)
        kinds = [w["type"] for w in out["writes"]]
        self.assertIn("pause", kinds)
        self.assertNotIn("tag", kinds)
        pause = next(w for w in out["writes"] if w["type"] == "pause")
        self.assertEqual(dt.datetime.fromtimestamp(pause["until"]).strftime("%H:%M"), "18:00")

    def test_a_crew_with_no_hour_gets_one_question_and_no_invented_time(self):
        understand = acts_for({"זה בסדר אלה העובדים של הפרגולה": {"emotion": "neutral", "acts": [
            act("person_mark", quote="אלה העובדים של הפרגולה", subject="העובדים של הפרגולה", camera="cam3")]}})
        writer = FakeModel(lambda user, kw: "הבנתי, אלה העובדים של הפרגולה. עד איזו שעה הם עובדים פה?")
        alert = dict(ALERT, id="ameer_v2_ch3_1_alert", camera="ameer_v2_ch3", time="09:51:33")
        out = run({"time": "09:52:41", "alerts": [alert],
                   "message": {"text": "זה בסדר אלה העובדים של הפרגולה", "reply_to": alert["id"]}},
                  understand, writer)
        self.assertGreaterEqual(len(out["buttons"]), 2)
        self.assertEqual(ec.count_questions(out["text"], out["buttons"]), 1)
        self.assertFalse([w for w in out["writes"] if w["type"] == "person_mark"])    # no time he did not say
        self.assertNotIn("23:59", out["text"])

    def test_the_tapped_hour_completes_the_mark_in_code(self):
        from home_guard_project.box.assistant_v3 import AssistantV3  # noqa: F401

        understand = acts_for({"זה בסדר אלה העובדים של הפרגולה": {"emotion": "neutral", "acts": [
            act("person_mark", quote="אלה העובדים של הפרגולה", subject="העובדים של הפרגולה", camera="cam3")]}})
        writer = FakeModel(lambda user, kw: "סגור, העובדים של הפרגולה עד 18:00.")
        alert = dict(ALERT, id="ameer_v2_ch3_1_alert", camera="ameer_v2_ch3", time="09:51:33")
        case = {"id": "t", "source": "test", "date": DAY, "house": HOUSE, "time": "09:52:41", "alerts": [alert],
                "message": {"text": "זה בסדר אלה העובדים של הפרגולה", "reply_to": alert["id"]},
                "expect": {"intent": "person_mark", "ideal": "x", "max_questions": 1}}
        ec.V3.clear()
        ec.V3.update(write=writer, understand=understand, critic=None, escalate=None)
        world = ec.World(case, None, None)
        try:
            ec._local.world = world
            world.seed()
            first = world.agent.handle(case["message"]["text"], ec.CHAT, ec.OWNER,
                                       {"alert_id": alert["id"], "camera": alert["camera"], "ts": at("09:51"),
                                        "label": "suspicious", "summary": alert["summary"]}, True)
            self.assertIn("18:00", first.buttons)
            calls = len(understand.calls)
            world.clock["now"] += 20
            second = world.agent.handle_choice(ec.CHAT, first.question_token, list(first.buttons).index("18:00"),
                                               ec.OWNER)
            self.assertIsNotNone(second)
            self.assertEqual(len(understand.calls), calls)          # the answer is read in code
            marks = world.events.list_known(world.clock["now"])
            self.assertEqual(len(marks), 1)
            self.assertEqual(marks[0]["daily_to"], "18:00")
            self.assertEqual(marks[0]["camera"], "ameer_v2_ch3")
        finally:
            ec._local.world = None
            ec.V3.clear()

    def test_in_a_complaint_a_crew_with_no_hour_is_kept_for_today_without_a_question(self):
        understand = acts_for({"אמרתי לך שהם עובדים פה": {"emotion": "angry", "acts": [  # angry: no question
            act("complaint", quote="אמרתי לך", issue="ignored_memory"),
            act("person_mark", quote="שהם עובדים פה", subject="העובדים", camera="cam3")]}})
        writer = FakeModel(lambda user, kw: "צודק, סימנתי אותם להיום.")
        out = run({"time": "12:56:55", "message": {"text": "אמרתי לך שהם עובדים פה"}}, understand, writer)
        marks = [w for w in out["writes"] if w["type"] == "person_mark"]
        self.assertEqual(len(marks), 1)
        self.assertEqual(dt.datetime.fromtimestamp(marks[0]["until"]).date().isoformat(), DAY)
        self.assertEqual(out["buttons"], [])

    def test_a_complaint_redoes_the_mishandled_place(self):
        understand = acts_for({
            "מה קשר ? לא הבנתי": {"emotion": "confused", "acts": [act("complaint", quote="מה קשר ? לא הבנתי",
                                                                       issue="unwanted_question")]},
            "זה הבית של השכן": {"emotion": "neutral", "acts": [act("place_fact", quote="זה הבית של השכן",
                                                                     camera="cam2", subject="הבית של השכן")]}})
        writer = FakeModel(lambda user, kw: "סליחה, שאלה מיותרת. שמרתי שבמצלמה 2 זה הבית של השכן, בלי תאריך.")
        out = run({"time": "13:49:34", "alerts": [ALERT],
                   "history": [{"time": "13:48:00", "owner": "זה הבית של השכן", "bot": "עד מתי לזכור את הבית של השכן?",
                                "about": ALERT["id"]}],
                   "message": {"text": "מה קשר ? לא הבנתי"}}, understand, writer)
        self.assertEqual([w["type"] for w in out["writes"]], ["place_fact"])

    def test_a_failing_draft_is_rewritten_once_then_a_hebrew_template(self):
        understand = acts_for({"מה קורה": {"emotion": "neutral", "acts": [act("question_live", quote="מה קורה")]}})
        writer = FakeModel(lambda user, kw: "All quiet outside, nothing to report.")
        out = run({"time": "12:00:00", "message": {"text": "מה קורה"}}, understand, writer)
        self.assertEqual(len(writer.calls), 2)
        self.assertFalse(ec.english_words(out["text"]))
        self.assertTrue(out["text"].strip())


class SmallRulesTests(unittest.TestCase):
    def test_pause_then_resume_at_midnight_is_one_pause(self):
        from home_guard_project.box.assistant_v3.acts import Act, Understanding
        from home_guard_project.box.assistant_v3.agent import merge_pause_resume

        und = Understanding(acts=[Act("command", command="pause", until_quote="עד היום בלילה"),
                                  Act("command", command="resume", until_quote="ב 12 בלילה")])
        merge_pause_resume(und)
        self.assertEqual([a.command for a in und.acts], ["pause"])
        self.assertEqual(und.acts[0].until_quote, "ב 12 בלילה")

    def test_people_in_a_description(self):
        from home_guard_project.box.assistant_v3.handlers import _people_in

        self.assertEqual(_people_in("Three men work under the pergola"), 3)
        self.assertEqual(_people_in("A man kneels by the low wall"), 1)
        self.assertEqual(_people_in("A quiet yard, no people"), 0)

    def test_reporting_what_was_not_done_is_internal_state(self):
        verdict = sv.code_checks("הבנתי, זה היה הדוור. לא שיניתי כלום.", lang="he", last_replies=[], allowed_times="",
                                 receipts=[], memory_subjects=[], owner_text="", may_ask=False, act_kinds=["ack"])
        self.assertIn("internal_state", verdict.reason())

    def test_the_postman_closes_the_alert_and_is_not_remembered(self):
        alert = dict(ALERT, id="ameer_v2_ch6_9_alert", camera="ameer_v2_ch6", time="12:10:00")
        understand = acts_for({"זה היה הדוור": {"emotion": "neutral", "acts": [
            act("person_mark", quote="זה היה הדוור", subject="הדוור", camera="cam6")]}})
        writer = FakeModel(lambda user, kw: "👍 סגרתי, זה היה הדוור.")
        out = run({"time": "12:15:00", "alerts": [alert], "message": {"text": "זה היה הדוור", "reply_to": alert["id"]}},
                  understand, writer)
        self.assertEqual(out["writes"], [])
        self.assertEqual(out["buttons"], [])


class SwitchTests(unittest.TestCase):
    def test_v3_only_when_asked(self):
        self.assertFalse(wants_v3({}))
        self.assertFalse(wants_v3({"assistant": "v2"}))
        self.assertTrue(wants_v3({"assistant": "V3"}))


if __name__ == "__main__":
    unittest.main()
