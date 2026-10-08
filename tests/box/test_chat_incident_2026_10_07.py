# tests/box/test_chat_incident_2026_10_07.py
"""Regression: the owner's real Telegram chat of 2026-10-07/08 with the box's assistant (logs/telegram_chat.jsonl).

Of 25 replies, 13 ended with "אם יש משהו נוסף… אני כאן" and 13 were "רשמתי את זה כהתרעה…" - complaints, questions
and "which video?" were filed as verdicts, a complaint after "אחר…" was saved as a clip's explanation, a camera
name was "saved" on an id that no longer existed, and "these are my workers" changed nothing (142 more alerts).
Every message below goes through the code paths that do not need a model: the pre-checks, the claims rewrite,
the boilerplate filter, the "Other…" capture rules, and the Memory Keeper's receipt and its buttons."""

from __future__ import annotations

import glob
import json
import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

import numpy as np

from home_guard_project.box.agent import AgentReply
from home_guard_project.box.alert_clips import encode_frame, write_alert_clip
from home_guard_project.box.brain.agent import AgentReply as AgentReplyV2
from home_guard_project.box.brain.claims import honest_answer, unbacked_claims
from home_guard_project.box.brain.i18n import t
from home_guard_project.box.brain.profiles import asks_which_event, says_known
from home_guard_project.box.brain.receipts import DONE, Receipt
from home_guard_project.box.brain.style import clean_outgoing, strip_boilerplate
from home_guard_project.box.feedback import AlertIndex, MuteState, not_a_judgement
from home_guard_project.box.telegram_agent import TelegramInbox, is_tag_answer, reply_fields, send_alert
from home_guard_project.box.telegram_notify import TelegramConfig

NOW = 1_791_000_000.0
CHAT = "-5326761586"
OLD_ID, NEW_ID = "ameer_tes2_ch6", "ameer_week_0_1_ch6"
ALERT_ID = f"{NEW_ID}_1790999940_alert"
ALERT = {"alert_id": ALERT_ID, "camera": NEW_ID, "summary": "a person at the entrance", "label": "suspicious",
         "ts": NOW - 60}
AMEER = {"id": 42, "first_name": "Ameer"}

# What the owner wrote (in order), and whether it may ever be filed as a judgement of an alert by itself.
OWNER = [
    ("אל תקרא לה יותר ככה תקרא לה כניסה ראשית", False),        # a command (a camera name)
    ("מדי פעם אני (עמיר) יוצא החוצה", True),
    ("זה בסדר זה עובדים אצלי שעובדים על הפרגולה", True),
    ("איזה תזכורת , אתה מטומטם ומגזים", False),
    ("אני לא יודע על איזה סרטון אתה מדבר בכלל", False),
    ("תראה לי אותו", False),
    ("אין בעיה אבל על איזה פאקינג סרטון דיברת", False),
    ("תפסיק עם ההודעה המטומטת הזאת של אם יש עוד משהו שתרצה לדעת אני כאן זה מעצבן", False),
    ("אם זה תקין אז למה אתה ממשיך לשלוח לי הודעות יא מטומטם", False),
    ("מה הקשר התרעה צפויה", False),
    ("די עם ההודעה המטופשת הזאת", False),
    ("אתה כבר אמור לדעת איפה זה", False),
    ("אישה יוצאת מהכניסה לבית כאשר היא מחזיקה בשני סקוטרים", True),
    ("זה לא התרעה זה תקין", True),
    ("לא חתיכת מטומטם לא !", False),
]

# What the bot answered, and what the owner must read instead when no receipt of that action exists.
BOT = [
    ('שמרתי את המצלמה ameer_tes2_ch6 כ"כניסה ראשית". אם יש משהו נוסף שתרצה לשאול או לשנות, אני כאן!',
     t("not_saved_yet", "he")),
    ("רשמתי את זה כהתרעה צפויה. אם יש משהו נוסף שתרצה לדעת, אני כאן!", t("not_saved_verdict", "he")),
    ("רשמתי את זה כהתרעה אמיתית, אבל התיאור היה לא מדויק. תודה על ההבהרה!", t("not_saved_verdict", "he")),
    ("רשמתי את זה כהתרעה צפויה. תודה על ההבהרה!", t("not_saved_verdict", "he")),
]


def receipt(tool: str) -> Receipt:
    return Receipt(id="R1", turn="t", tool=tool, status=DONE)


class PreCheckTest(unittest.TestCase):
    def test_questions_complaints_and_commands_are_never_a_verdict(self) -> None:
        for text, judgement in OWNER:
            with self.subTest(text=text):
                self.assertEqual(not_a_judgement(text), not judgement)

    def test_which_video_and_who_is_there_are_recognised(self) -> None:
        self.assertTrue(asks_which_event("אני לא יודע על איזה סרטון אתה מדבר בכלל"))
        self.assertTrue(asks_which_event("אין בעיה אבל על איזה פאקינג סרטון דיברת"))
        self.assertFalse(asks_which_event("זה לא התרעה זה תקין"))
        for text in ("זה בסדר זה עובדים אצלי שעובדים על הפרגולה", "מדי פעם אני (עמיר) יוצא החוצה", "זה אני",
                     "זה הגנן", "these are my workers", "it's me"):
            self.assertTrue(says_known(text), text)
        self.assertFalse(says_known("יש אנשים שעובדים ליד הפרגולה?"))   # a question about workers


class BotRepliesTest(unittest.TestCase):
    def test_no_closing_boilerplate_ever(self) -> None:
        cases = {
            "רשמתי את זה כהתרעה צפויה. אם יש משהו נוסף שתרצה לדעת, אני כאן!": "רשמתי את זה כהתרעה צפויה.",
            "רשמתי את זה כהתרעה צפויה. תודה על ההבהרה!": "רשמתי את זה כהתרעה צפויה.",
            "רשמתי את זה, ואם יש משהו נוסף אני כאן!": "רשמתי את זה.",
            "There was one event at 22:00. If there's anything else you need, let me know!":
                "There was one event at 22:00.",
            "Done. Let me know if you need anything else. I'm here.": "Done.",
        }
        for said, cleaned in cases.items():
            with self.subTest(said=said):
                self.assertEqual(strip_boilerplate(said), cleaned)
        self.assertEqual(clean_outgoing("אם יש משהו נוסף, אני כאן!"), "👍")      # never an empty message

    def test_camera_ids_become_names(self) -> None:
        out = clean_outgoing('שמרתי את המצלמה ameer_tes2_ch6 כ"כניסה ראשית".', [NEW_ID])
        self.assertNotIn("ameer", out)
        self.assertIn("מצלמה 6", out)
        self.assertEqual(clean_outgoing("ב-12:46 במצלמה ameer_week_0_1_ch6 נראו שני אנשים."),
                         "ב-12:46 במצלמה 6 נראו שני אנשים.")
        with mock.patch("home_guard_project.box.brain.aliases.load_aliases",
                        return_value={OLD_ID: ["כניסה ראשית"]}):          # a name saved before the site rename
            self.assertEqual(clean_outgoing("תנועה ב-ameer_week_0_1_ch6"), "תנועה ב-כניסה ראשית")

    def test_saved_wording_needs_the_receipt_of_that_very_action(self) -> None:
        for said, honest in BOT:
            with self.subTest(said=said):
                self.assertTrue(unbacked_claims(said, []))
                self.assertEqual(clean_outgoing(honest_answer(said, [], "he", "תקרא לה כניסה ראשית")), honest)
        # A receipt of another action backs nothing: a keeper's note is not a verdict, a verdict is not a name.
        self.assertEqual(unbacked_claims("רשמתי את זה כהתרעה צפויה.", [receipt("mark_known")]), ["verdict"])
        self.assertIn("alias", unbacked_claims('שמרתי את המצלמה כ"כניסה ראשית".', [receipt("record_verdict")]))
        self.assertEqual(unbacked_claims("רשמתי שאלה העובדים שלך. לא אשלח עליהם הודעות.", [receipt("record_verdict")]),
                         ["known"])
        self.assertEqual(unbacked_claims("רשמתי את זה כהתרעה צפויה.", [receipt("record_verdict")]), [])


class FakeTelegram:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._next = 900

    def post(self, token: str, method: str, fields: Dict[str, str], timeout: float = 15.0) -> Dict[str, Any]:
        self._next += 1
        self.calls.append({"method": method, "fields": dict(fields), "message_id": self._next})
        return {"ok": True, "result": {"message_id": self._next}}

    def post_multipart(self, token, method, fields, files, timeout=20.0):
        return self.post(token, method, fields, timeout)

    def texts(self) -> List[str]:
        return [c["fields"]["text"] for c in self.calls if c["method"] == "sendMessage"]


class Agent:
    """The v2 agent's surface the inbox uses; it answers like the model of 2026-10-07 did."""

    version = 2

    def __init__(self, answer: str = "רשמתי את זה כהתרעה צפויה. אם יש משהו נוסף שתרצה לדעת, אני כאן!") -> None:
        self.seen: List[tuple] = []
        self.buttons: List[tuple] = []
        self.answer = answer

    def handle(self, text, chat_id, who=None, alert=None, threaded=False):
        self.seen.append((text, (alert or {}).get("alert_id")))
        return AgentReplyV2(text=f"{self.answer} ({NEW_ID})", lang="he")

    def known_button(self, chat_id, action, known_id, who=None):
        self.buttons.append((action, known_id))
        return AgentReplyV2(text="בוטל. אשלח שוב הודעות על פרגולה.", lang="he")


def tap(update_id: int, code: str, on_message: int) -> dict:
    return {"update_id": update_id, "callback_query": {"id": f"cb{update_id}", "data": code, "from": dict(AMEER),
                                                       "message": {"message_id": on_message, "chat": {"id": int(CHAT)}}}}


def text(update_id: int, words: str, reply_to: Optional[int] = None) -> dict:
    msg: Dict[str, Any] = {"message_id": 500 + update_id, "chat": {"id": int(CHAT)}, "text": words,
                           "from": dict(AMEER, is_bot=False)}
    if reply_to is not None:
        msg["reply_to_message"] = {"message_id": reply_to}
    return {"update_id": update_id, "message": msg}


class InboxTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.production = os.path.join(self.dir, "production_multi")
        frames = [(NOW - 60 + i * 0.2, encode_frame(np.zeros((48, 64, 3), dtype=np.uint8))) for i in range(5)]
        with mock.patch("home_guard_project.box.alert_clips._to_h264", return_value=False):
            write_alert_clip(self.production, NEW_ID, ALERT_ID, frames,
                             {"summary": "a person", "labels": ["person"], "dispatch": {"sent": True}})
        self.index = AlertIndex(os.path.join(self.dir, "alert_index.json"))
        self.index.remember(CHAT, 77, ALERT)
        self.tg = FakeTelegram()
        self.clock = NOW
        self.heard = ""

    def inbox(self, agent: Any) -> TelegramInbox:
        return TelegramInbox(TelegramConfig(bot_token="T", chat_ids=[CHAT]), agent, self.index,
                             MuteState(os.path.join(self.dir, "mute.json")), self.production,
                             os.path.join(self.dir, "offset.json"), post=self.tg.post,
                             post_multipart=self.tg.post_multipart, now=lambda: self.clock,
                             training_dir=os.path.join(self.dir, "dataset_multi"),
                             archive_dir=os.path.join(self.dir, "archive"), lang=lambda: "he",
                             transcriber=lambda audio, name, lang: self.heard,
                             fetch_voice=lambda file_id: (b"ogg", "voice.oga"), cameras=lambda: [NEW_ID])

    def explanations(self) -> List[str]:
        out = []
        for path in sorted(glob.glob(os.path.join(self.production, "feedback", "*", "*", "*.feedback.json"))):
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("owner_label") == "other":
                out.append(saved["owner_text"])
        return out

    def test_after_other_a_complaint_goes_to_the_assistant_and_a_statement_is_the_explanation(self) -> None:
        agent = Agent()
        inbox = self.inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        self.clock += 30
        inbox.handle_update(text(2, "איזה תזכורת , אתה מטומטם ומגזים"))          # 12:53, right after "אחר…"
        self.assertEqual(self.explanations(), [])
        self.assertEqual(agent.seen, [("איזה תזכורת , אתה מטומטם ומגזים", None)])   # no alert bound either
        self.assertNotIn(t("tag_saved_explanation", "he", time="12:46")[:10], " ".join(self.tg.texts()))
        self.clock += 30
        inbox.handle_update(text(3, "אישה יוצאת מהכניסה לבית כאשר היא מחזיקה בשני סקוטרים"))
        self.assertEqual(self.explanations(), ["אישה יוצאת מהכניסה לבית כאשר היא מחזיקה בשני סקוטרים"])

    def test_the_wait_takes_a_message_that_replies_to_nothing_only_for_three_minutes(self) -> None:
        agent = Agent()
        inbox = self.inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        question = self.tg.calls[0]["message_id"]
        self.clock += 200
        inbox.handle_update(text(2, "שני ילדים על אופניים"))
        self.assertEqual(self.explanations(), [])
        self.assertEqual(agent.seen[-1][0], "שני ילדים על אופניים")
        inbox.handle_update(text(3, "מה זה? שני ילדים", reply_to=question))   # a reply to the question always is
        self.assertEqual(self.explanations(), ["מה זה? שני ילדים"])

    def test_a_reply_to_the_alert_that_asks_something_goes_to_the_assistant(self) -> None:
        agent = Agent()
        inbox = self.inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        inbox.handle_update(text(2, "למה אתה ממשיך לשלוח לי הודעות?", reply_to=77))
        self.assertEqual(self.explanations(), [])
        self.assertEqual(agent.seen, [("למה אתה ממשיך לשלוח לי הודעות?", ALERT_ID)])
        self.assertTrue(is_tag_answer("שני ילדים", 77, {"prompt_ids": ["901"]}))
        self.assertFalse(is_tag_answer("איזה סרטון?", 77, {"prompt_ids": ["901"]}))
        self.assertTrue(is_tag_answer("איזה סרטון?", 901, {"prompt_ids": ["901"]}))

    def test_a_spoken_complaint_is_read_by_the_assistant_not_saved_as_the_clip(self) -> None:
        agent = Agent()
        inbox = self.inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        self.heard = "די עם ההודעה המטופשת הזאת"
        inbox.handle_update({"update_id": 2, "message": {"message_id": 600, "chat": {"id": int(CHAT)},
                                                         "voice": {"file_id": "v1"}, "from": dict(AMEER)}})
        self.assertEqual(self.explanations(), [])
        self.assertEqual(agent.seen, [("די עם ההודעה המטופשת הזאת", None)])

    def test_every_answer_goes_out_without_boilerplate_and_without_the_camera_id(self) -> None:
        inbox = self.inbox(Agent())
        inbox.handle_update(text(1, "זה לא התרעה זה תקין"))
        (answer,) = self.tg.texts()
        self.assertEqual(answer, "רשמתי את זה כהתרעה צפויה. (מצלמה 6)")
        self.assertNotIn("אני כאן", answer)

    def test_v1_answers_are_cleaned_too_and_a_question_is_not_bound_to_the_latest_alert(self) -> None:
        class V1:
            version = 1
            seen: List[Any] = []

            def handle(self, text, chat_id, who=None, alert=None):
                self.seen.append((text, (alert or {}).get("alert_id")))
                return AgentReply(text="רשמתי את זה כהתרעה צפויה. תודה על ההבהרה!")

        agent = V1()
        inbox = self.inbox(agent)
        inbox.handle_update(text(1, "מה הקשר התרעה צפויה"))
        inbox.handle_update(text(2, "זה לא התרעה זה תקין"))
        self.assertEqual(agent.seen, [("מה הקשר התרעה צפויה", None), ("זה לא התרעה זה תקין", ALERT_ID)])
        self.assertEqual(self.tg.texts(), ["רשמתי את זה כהתרעה צפויה."] * 2)

    def test_the_keepers_buttons_reach_the_agent_and_cancel_clears_them(self) -> None:
        agent = Agent()
        inbox = self.inbox(agent)
        inbox.handle_update(tap(1, "kn:x:abc123", 950))
        self.assertEqual(agent.buttons, [("x", "abc123")])
        edit = next(c for c in self.tg.calls if c["method"] == "editMessageReplyMarkup")
        self.assertEqual(json.loads(edit["fields"]["reply_markup"]), {"inline_keyboard": []})
        self.assertIn("בוטל. אשלח שוב הודעות על פרגולה.", self.tg.texts())
        inbox.handle_update(tap(2, "kn:bad", 950))                    # malformed: ignored
        self.assertEqual(agent.buttons, [("x", "abc123")])

    def test_the_keepers_receipt_goes_out_with_its_buttons(self) -> None:
        rows = ((("ביטול", "kn:x:k1"), ("כל השבוע", "kn:w:k1")),)

        class Keeper(Agent):
            def handle(self, text, chat_id, who=None, alert=None, threaded=False):
                return AgentReplyV2(text="שמרתי: האנשים בפרגולה הם העובדים, עד 23:59. לא אשלח עליהם הודעות, "
                                         "חוץ מדבר חריג.", lang="he", rows=rows)

        self.inbox(Keeper()).handle_update(text(1, "זה בסדר זה עובדים אצלי שעובדים על הפרגולה", reply_to=77))
        sent = next(c for c in self.tg.calls if c["method"] == "sendMessage")
        self.assertEqual(json.loads(sent["fields"]["reply_markup"])["inline_keyboard"],
                         [[{"text": "ביטול", "callback_data": "kn:x:k1"},
                           {"text": "כל השבוע", "callback_data": "kn:w:k1"}]])


class ThreadedAlertTest(unittest.TestCase):
    def test_an_event_update_is_a_reply_to_its_first_alert_in_that_chat_only(self) -> None:
        self.assertEqual(reply_fields(CHAT, {"chat_id": CHAT, "message_id": 123}),
                         {"reply_to_message_id": "123", "allow_sending_without_reply": "true"})
        self.assertEqual(reply_fields("-1", {"chat_id": CHAT, "message_id": 123}), {})
        self.assertEqual(reply_fields(CHAT, None), {})
        tg = FakeTelegram()
        with tempfile.TemporaryDirectory() as root:
            index = AlertIndex(os.path.join(root, "index.json"))
            send_alert(TelegramConfig(bot_token="T", chat_ids=[CHAT, "-1"]), index, ALERT, "🟡 עוד אדם", post=tg.post,
                       post_multipart=tg.post_multipart, lang="he", reply_to={"chat_id": CHAT, "message_id": 123})
        first, other = tg.calls
        self.assertEqual(first["fields"]["reply_to_message_id"], "123")
        self.assertNotIn("reply_to_message_id", other["fields"])

    def test_a_held_alert_keeps_its_thread(self) -> None:
        from home_guard_project.box.telegram_agent import OwnerAssistant  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as root:
            index = AlertIndex(os.path.join(root, "index.json"))
            assistant = OwnerAssistant(TelegramConfig(bot_token="T", chat_ids=[CHAT]), index, None, None,
                                       video_wait=3600)
            with mock.patch("home_guard_project.box.telegram_agent.threading.Timer"):
                assistant.send_alert(ALERT, "🟡 עוד אדם", reply_to={"chat_id": CHAT, "message_id": 123})
            with mock.patch("home_guard_project.box.telegram_agent.send_alert", return_value={"sent": True}) as sent:
                assistant.send_clip(ALERT_ID, "")
        self.assertEqual(sent.call_args.kwargs["reply_to"], {"chat_id": CHAT, "message_id": 123})
        self.assertIs(assistant.index, index)                      # the guard reads assistant.index.messages()


if __name__ == "__main__":
    unittest.main()
