# -*- coding: utf-8 -*-
"""The owner's chat of 2026-10-20 08:53-10:15, replayed (stub models, no network).

He asked for "the video of the child who fell" (an 08:23 clip at the main entrance, in the journal and even sent to
him) eight times. He got last night's clip from camera 5, then 10 live seconds, then live photos, then "there is no
such video", while each complaint ("לא הבאת לי סרטון ...", "שוב פעם אתה מביא לי תמונה חיה") was answered with an
apology or one more live photo. Only "ביקשתי סרטון מ 8:23" got it.
"""
import json
import unittest
from typing import Any, Dict, List

from home_guard_project.box import eval_chat as ec
from home_guard_project.box.assistant_v3.acts import Act, Understanding
from home_guard_project.box.assistant_v3.agent import media_request
from home_guard_project.box.assistant_v3.handlers import complains_about_photo
from home_guard_project.box.brain.models import ModelMessage
from test_assistant_v3 import act

DAY = "2026-10-20"
HOUSE = {"cameras": ["ameer_v2_ch2", "ameer_v2_ch3", "ameer_v2_ch5", "ameer_v2_ch6"],
         "aliases": {"ameer_v2_ch3": ["פרגולה"], "ameer_v2_ch6": ["כניסה ראשית"]}}
JOURNAL = [
    {"time": "2026-10-19 22:42:00", "camera": "ameer_v2_ch5", "clip": True, "label": "suspicious", "reported": True,
     "summary": "A person in dark clothes walks along the side of the building at night, stops near some crates."},
    {"time": "08:23:00", "camera": "ameer_v2_ch6", "clip": True, "label": "normal", "reported": True,
     "summary": "A woman walks across the patio holding an item. A child appears, follows her briefly, then falls "
                "to the ground."},
    {"time": "08:53:00", "camera": "ameer_v2_ch6", "clip": True, "label": "normal", "reported": True,
     "summary": "A woman exits a white car parked in the driveway and walks towards the house holding her phone."},
]

class Understander:
    """The acts JSON for the owner's message, and the clip finder's answer (a function of its request)."""

    def __init__(self, acts: Dict[str, Any], find: Any = None) -> None:
        self.acts, self.find, self.finds = acts, find, []

    def chat(self, messages: List[Dict[str, Any]], tools: Any = None, tool_choice: Any = None,
             **kwargs: Any) -> ModelMessage:
        system = str(messages[0].get("content") or "")
        user = str(messages[-1].get("content") or "")
        if "recorded clip a home owner asks to see" in system:
            self.finds.append(user)
            return ModelMessage(content=json.dumps(self.find(user) if self.find else {"described": False,
                                                                                       "index": -1}), usage=(10, 5))
        return ModelMessage(content=json.dumps(self.acts, ensure_ascii=False), usage=(100, 20))


class Writer:
    def __init__(self) -> None:
        self.calls: List[str] = []

    def chat(self, messages: List[Dict[str, Any]], tools: Any = None, tool_choice: Any = None,
             **kwargs: Any) -> ModelMessage:
        self.calls.append(str(messages[-1].get("content") or ""))
        return ModelMessage(content="בסדר.", usage=(100, 20))


def fall_line(user: str) -> Dict[str, Any]:
    """The finder picks the clip whose line mentions the fall."""
    for line in user.split("\n"):
        if line.startswith("[") and "falls" in line:
            return {"described": True, "index": int(line[1:line.index("]")])}
    return {"described": True, "index": -1}


def run(text: str, und: Understander, history: Any = None, time: str = "08:55:00", writer: Any = None,
        **extra: Any) -> Dict[str, Any]:
    case = {"id": "t", "source": "test", "date": DAY, "house": HOUSE, "time": time, "journal": JOURNAL,
            "history": history or [], "message": {"text": text},
            "expect": {"intent": "command", "ideal": "x", "max_questions": 0}}
    case.update(extra)
    ec.V3.clear()
    ec.V3.update(write=writer or Writer(), understand=und, critic=None, escalate=None)
    try:
        return ec.run_case(case, None, None)
    finally:
        ec.V3.clear()


def fall_clip(out: Dict[str, Any]) -> bool:
    ts = ec.ts_of({"date": DAY}, "08:23:00")
    return out["videos"] == [f"ameer_v2_ch6_{int(ts)}_alert.mp4"]


class DescribedClipTests(unittest.TestCase):
    def test_the_video_of_the_fall_is_found_in_the_journal(self):
        und = Understander({"emotion": "neutral", "acts": [act("command", command="send_video", subject="המפילה",
                                                               quote="תראה לי את הסרטון של המפילה")]}, fall_line)
        out = run("תראה לי את הסרטון של המפילה", und)
        self.assertEqual(out["error"], "")
        self.assertTrue(fall_clip(out), out["videos"])
        self.assertNotIn("record_clip", out["tools"])
        self.assertTrue(und.finds and "falls" in und.finds[0])

    def test_a_described_video_at_a_named_camera_is_never_live_seconds(self):
        und = Understander({"emotion": "neutral", "acts": [act("command", command="send_video",
                                                               camera="ameer_v2_ch6")]}, fall_line)
        out = run("תראה לי סרטון של התינוק נפל בכניסה הראשית", und)
        self.assertTrue(fall_clip(out), out["videos"])
        self.assertNotIn("record_clip", out["tools"])

    def test_nothing_shows_it_then_nothing_else_is_sent(self):
        und = Understander({"emotion": "neutral", "acts": [act("command", command="send_video")]},
                           lambda user: {"described": True, "index": -1})
        out = run("תראה לי את הסרטון של הגנן שנפל מהסולם", und)
        self.assertEqual(out["videos"], [])
        self.assertNotIn("record_clip", out["tools"])

    def test_a_complaint_about_the_missing_video_sends_it(self):
        und = Understander({"emotion": "angry", "acts": [act("complaint", issue="ignored_memory",
                                                             command="send_video",
                                                             quote="לא הבאת לי סרטון של נפילת הילד")]}, fall_line)
        out = run("לא הבאת לי סרטון של נפילת הילד יא מטומטם", und)
        self.assertTrue(fall_clip(out), out["videos"])

    def test_a_complaint_about_a_live_photo_brings_the_video_not_another_photo(self):
        history = [{"time": "08:59:00", "owner": "אני ביקשתי משהו פשוט, סרטון של הילד נופל",
                    "bot": "שלחתי תמונה.", "photos": ["ameer_v2_ch6"]}]
        und = Understander({"emotion": "angry", "acts": [act("complaint", issue="irrelevant"),
                                                         act("command", command="send_photo")]}, fall_line)
        out = run("שוב פעם אתה מביא לי תמונה חיה יא אהבללל", und, history, time="09:00:05")
        self.assertEqual(out["photos"], [])
        self.assertTrue(fall_clip(out), out["videos"])
        self.assertIn("סרטון של הילד נופל", und.finds[0])          # his earlier words describe it

    def test_a_time_he_names_still_finds_the_clip(self):
        und = Understander({"emotion": "neutral", "acts": [act("complaint", command="send_video",
                                                               quote="ביקשתי סרטון מ 8:23")]},
                           lambda user: {"described": False, "index": -1})
        out = run("ביקשתי סרטון מ 8:23", und, time="09:02:19")
        self.assertTrue(fall_clip(out), out["videos"])


class MediaWordsTests(unittest.TestCase):
    def und(self, *acts: Act) -> Understanding:
        return Understanding(acts=list(acts))

    def test_a_video_ask_never_becomes_a_photo(self):
        u = self.und(Act(act="complaint", issue="other"), Act(act="command", command="send_photo"))
        media_request(u, "הרי הבאת לי תמונה אז אני לא מבין למה אתה לא מביא לי את הסרטון")
        self.assertEqual([(a.act, a.command) for a in u.acts], [("complaint", ""), ("command", "send_video")])

    def test_photo_and_video_both_asked(self):
        u = self.und(Act(act="command", command="send_photo"))
        media_request(u, "תשלח סרטון וגם תמונה")
        self.assertEqual(sorted(a.command for a in u.acts if a.act == "command"), ["send_photo", "send_video"])

    def test_a_complaint_carrying_the_request_gets_the_act(self):
        u = self.und(Act(act="complaint", issue="ignored_memory", command="send_video", subject="נפילת הילד"))
        media_request(u, "מבקש את הסרטון יא אהבלללל")
        self.assertEqual([a.command for a in u.acts if a.act == "command"], ["send_video"])

    def test_no_video_word_changes_nothing(self):
        u = self.und(Act(act="command", command="send_photo"))
        media_request(u, "תשלח תמונה")
        self.assertEqual([a.command for a in u.acts], ["send_photo"])

    def test_complaints_about_a_photo(self):
        self.assertTrue(complains_about_photo("שוב פעם אתה מביא לי תמונה חיה"))
        self.assertTrue(complains_about_photo("מה קשר תמונה עכשיו"))
        self.assertFalse(complains_about_photo("למה אתה לא שולח תמונה"))
        self.assertFalse(complains_about_photo("תשלח תמונה מהכניסה"))


class PlateClaimTests(unittest.TestCase):
    """08:55 "בדקתי שוב את המספר בפינה השמאלית העליונה, והוא תואם לרכב שלך": the camera's clock, read as a plate."""

    def check(self, text: str, evidence: str = "") -> Any:
        from home_guard_project.box.assistant_v3 import supervisor as sv

        return sv.code_checks(text, lang="he", last_replies=[], allowed_times="", receipts=[], memory_subjects=[],
                              owner_text="", may_ask=False, act_kinds=["complaint"], evidence_text=evidence)

    def test_a_plate_claim_needs_the_reader(self):
        bad = self.check("צודק, בדקתי שוב את המספר בפינה השמאלית העליונה, והוא תואם לרכב שלך.")
        self.assertIn("plate_claim", [c.name for c in bad.failed])
        self.assertFalse(bad.ok)
        self.assertFalse(self.check("המספר של הרכב תואם לרכב שלך.").ok)
        ok = self.check("כן, לפי קורא הלוחיות זה הרכב שלך.",
                        "קורא הלוחיות של הקופסה קרא באירוע של 08:53 בכניסה ראשית: ••••804 = הרכב שלך")
        self.assertTrue(ok.ok, ok.failed)

    def test_no_read_is_no_evidence(self):
        self.assertFalse(self.check("בדקתי את המספר והוא תואם.",
                                    "קורא הלוחיות של הקופסה לא קרא מספר רכב באירוע של 08:53; מספר לא נקרא מתמונה.").ok)


class UntilTests(unittest.TestCase):
    def test_durations(self):
        import datetime as dt
        from home_guard_project.box.assistant_v3.timeparse import parse_until

        now = dt.datetime(2026, 10, 19, 22, 44).timestamp()
        hm = lambda w: dt.datetime.fromtimestamp(parse_until(w, now)).strftime("%d %H:%M")  # noqa: E731
        self.assertEqual(hm("שעה בערך"), "19 23:44")
        self.assertEqual(hm("עוד חצי שעה"), "19 23:14")
        self.assertEqual(hm("שעתיים"), "20 00:44")
        self.assertEqual(hm("3 שעות"), "20 01:44")
        self.assertEqual(hm("עד הבוקר"), "20 07:00")
        self.assertEqual(hm("ב3"), "20 03:00")                  # a clock hour stays a clock hour
        self.assertIsNone(parse_until("מכינים הזמנות במחסן", now))

    def test_it_is_never_asked_twice(self):
        # 22:43 "זה אנחנו מכינים הזמנות במחסן" -> "עד מתי הם פה היום?" -> "אנחנו" -> the same question again
        history = [{"time": "22:43:24", "owner": "זה אנחנו מכינים הזמנות במחסן", "bot": "עד מתי הם פה היום?"}]
        und = Understander({"emotion": "neutral", "acts": [act("person_mark", quote="אנחנו", subject="אנחנו",
                                                               camera="ameer_v2_ch5")]})
        out = run("אנחנו", und, history, time="22:43:36", date="2026-10-19", journal=[])
        self.assertEqual(out["buttons"], [])
        self.assertNotIn("?", out["text"])
        self.assertTrue(out["marks_after"], out)

    def test_we_are_asked_as_we(self):
        und = Understander({"emotion": "neutral", "acts": [act("person_mark", quote="זה אנחנו מכינים הזמנות במחסן",
                                                               subject="אנחנו", camera="ameer_v2_ch5")]})
        out = run("זה אנחנו מכינים הזמנות במחסן", und, time="22:43:24", date="2026-10-19", journal=[])
        self.assertIn("עד מתי אתם פה היום?", out["text"])


class PastPresenceTests(unittest.TestCase):
    MANSOUR = {"plate": "1609875", "label": "הרכב של החשמלאי מנסור", "description": "לבן", "camera": "ameer_v2_ch3"}

    def test_was_here_is_a_history_question_not_a_mark(self):
        und = Understander({"emotion": "neutral", "acts": [act("person_mark", quote="מנסור החשמלאי היה פה",
                                                               subject="מנסור החשמלאי",
                                                               until_quote="ביומיים האחרונים")]})
        out = run("מנסור החשמלאי היה פה ביומיים האחרונים", und, time="10:54:22", memory={"vehicles": [self.MANSOUR]})
        self.assertEqual(out["buttons"], [])
        self.assertEqual(out["marks_after"], [])
        self.assertNotIn("עד איזו שעה", out["text"])

    def test_the_plate_reader_answers_when_his_car_was_here(self):
        seen = dict(JOURNAL[2], time="07:40:00", camera="ameer_v2_ch3", summary="A white car parks by the pergola.",
                    entities=[{"id": "CAR1", "kind": "vehicle", "vehicle_id": "", "owner_label": "הרכב של החשמלאי מנסור",
                               "plate_masked": "****875"}])
        writer = Writer()
        und = Understander({"emotion": "neutral", "acts": [act("question_history", quote="מה עם מנסור הגיע היום?",
                                                               subject="מנסור", camera="house")]})
        out = run("מה עם מנסור הגיע היום?", und, time="10:37:50", writer=writer, journal=JOURNAL + [seen],
                  memory={"vehicles": [self.MANSOUR]})
        self.assertEqual(out["error"], "")
        task = writer.calls[-1]
        self.assertIn("קורא הלוחיות של הקופסה זיהה את", task)
        self.assertIn("07:40", task)
        self.assertIn("••••875", task)

    def test_not_read_is_said(self):
        writer = Writer()
        und = Understander({"emotion": "neutral", "acts": [act("question_history", quote="מתי מנסור היה פה לאחרונה",
                                                               subject="מנסור", camera="house")]})
        run("מתי מנסור היה פה לאחרונה", und, time="10:38:12", writer=writer, memory={"vehicles": [self.MANSOUR]})
        self.assertIn("לא זיהה את", writer.calls[-1])


class PhotoCameraTests(unittest.TestCase):
    def test_the_camera_he_just_named(self):
        # 13:14:30 "מה עם כניסה הראשית" then 13:15:32 "תראה לי תמונה": camera 8 was sent
        history = [{"time": "13:13:51", "owner": "תשלח תמונה", "bot": "שלחתי תמונות.", "photos": ["ameer_v2_ch5"]},
                   {"time": "13:14:30", "owner": "מה עם כניסה הראשית", "bot": "בכניסה הראשית עומדת מכונית לבנה."}]
        und = Understander({"emotion": "neutral", "acts": [act("command", command="send_photo", camera="house")]})
        out = run("תראה לי תמונה", und, history, time="13:15:32")
        self.assertEqual(out["photo_cameras"], ["ameer_v2_ch6"])


if __name__ == "__main__":
    unittest.main()
