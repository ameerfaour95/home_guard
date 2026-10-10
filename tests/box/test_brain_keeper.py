# tests/box/test_brain_keeper.py
"""Stage 1, the assistant side (2026-10-08): the Memory Keeper (mark_known), recent_activity, "which video?",
the event of a replied-to alert in the context, claims per action, no closing boilerplate, and the v2 build with
the box's real settings. Models are scripted; nothing reaches a network."""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, List
from unittest.mock import Mock, patch

from home_guard_project.box.brain.agent import OwnerAgentV2
from home_guard_project.box.brain.i18n import t
from home_guard_project.box.brain.memory import ChatMemory, ChatState
from home_guard_project.box.brain.models import ModelMessage, ToolCall
from home_guard_project.box.brain.receipts import DONE, ReceiptBook
from home_guard_project.box.brain.registry import CameraState, HouseSnapshot
from home_guard_project.box.brain.tools import (
    TOOLS, Services, ToolContext, known_until, mark_known, record_verdict, recent_activity,
)
from home_guard_project.box.events import EventBook

NOW = dt.datetime(2026, 10, 7, 9, 52).timestamp()
PERGOLA, GATE = "ameer_week_0_1_ch6", "ameer_week_0_1_ch2"
ALERT = {"alert_id": f"{PERGOLA}_1791000000_alert", "camera": PERGOLA, "ts": NOW - 120, "label": "suspicious",
         "summary": "Two men work on the pergola with a ladder."}
WORKERS = "זה בסדר זה עובדים אצלי שעובדים על הפרגולה"
END_OF_DAY = dt.datetime(2026, 10, 7, 23, 59).timestamp()
WORKERS_18 = "זה בסדר זה עובדים אצלי שעובדים רק בפרגולה עד 18:00"
WEEK_END = dt.datetime(2026, 10, 13, 18, 0).timestamp()


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


def house(mode: str = "guard") -> HouseSnapshot:
    return HouseSnapshot(now=NOW, mode=mode, mode_ends=NOW + 3600, mode_started=NOW - 3600, start_hour=0,
                         end_hour=0, cameras=(CameraState(PERGOLA, True, ("פרגולה",), live=True),
                                              CameraState(GATE, True, (), live=True)))


class Registry:
    def snapshot(self):
        return house()


class KeeperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.events = EventBook(os.path.join(self.root, "events"))
        self.sent: List[Any] = []

    def services(self) -> Services:
        return Services(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"),
                        feedback_dir=self.root, work_dir=os.path.join(self.root, ".live"), mute=None, deliver=None,
                        read_settings=lambda: {"owner_language": "he"}, now=lambda: NOW, events=self.events)

    def agent(self, big: Scripted, run_tool=None) -> OwnerAgentV2:
        return OwnerAgentV2(big, Registry(), ChatMemory(os.path.join(self.root, ".conversations")),
                            ReceiptBook(os.path.join(self.root, ".receipts"), now=lambda: NOW), self.services(),
                            run_tool=run_tool, now=lambda: NOW)

    def ctx(self, text: str, alert: bool = False) -> ToolContext:
        state = ChatState()
        ctx = ToolContext(turn_id="-5:1", chat_id="-5", speaker={"user_id": 1, "name": "Ameer"}, text=text,
                          lang="he", mode="guard", snapshot=house(), state=state, services=self.services(),
                          book=ReceiptBook(os.path.join(self.root, ".r"), now=lambda: NOW), threaded=alert)
        if alert:
            ctx.alert_handle = state.add_handle("event", ALERT["alert_id"], PERGOLA, ALERT["ts"], ALERT["summary"])
        return ctx

    def verdicts(self) -> List[str]:
        out = []
        for path in glob.glob(os.path.join(self.root, "feedback", "**", "*.feedback.json"), recursive=True):
            with open(path, encoding="utf-8") as f:
                out.append(json.load(f)["verdict"])
        return out

    # -- the tool ---------------------------------------------------------------------------------------------
    def test_until_follows_the_owners_words(self) -> None:
        self.assertEqual(known_until("", NOW), END_OF_DAY)                 # default: the end of today
        self.assertEqual(known_until("today", NOW), END_OF_DAY)
        self.assertEqual(known_until("רק עכשיו", NOW), NOW + 3600)          # "only now": an hour
        self.assertEqual(known_until("week", NOW), NOW + 7 * 86400)
        self.assertEqual(known_until("18:00", NOW), dt.datetime(2026, 10, 7, 18, 0).timestamp())
        late = dt.datetime(2026, 10, 7, 23, 59, 30).timestamp()
        self.assertEqual(known_until("", late), late + 3600)               # today is over: an hour
        with self.assertRaises(ValueError):
            known_until("whenever", NOW)

    def test_said_on_an_alert_it_silences_that_camera_and_files_the_alert_as_expected(self) -> None:
        ctx = self.ctx(WORKERS, alert=True)
        out = mark_known(ctx, {"who": "העובדים על הפרגולה", "owner_words": "עובדים אצלי"})
        self.assertFalse(out["ok"])                               # 2026-10-09: no time said - asked, never 23:59
        self.assertEqual(ctx.clarification["choices"], ["16:00", "17:00", "18:00", "אחר…"])
        self.assertEqual(self.events.list_known(NOW), [])
        ctx = self.ctx(WORKERS_18, alert=True)
        out = mark_known(ctx, {"who": "העובדים על הפרגולה", "owner_words": "עובדים אצלי"})
        self.assertTrue(out["ok"])
        (known,) = self.events.list_known(NOW)
        # A crew: their hours (first seen 09:50 -> from 09:00) every day for a week.
        self.assertEqual((known["camera"], known["text"], known["until"], known["daily_from"], known["daily_to"]),
                         (PERGOLA, "העובדים על הפרגולה", WEEK_END, "09:00", "18:00"))
        self.assertEqual(self.verdicts(), ["expected"])                  # the alert's TAG (normal), its own line
        self.assertEqual([r.tool for r in ctx.receipts], ["retag_clip", "mark_known"])
        self.assertTrue(self.events.decide(PERGOLA, NOW + 13 * 3600, "suspicious", people=2).notify)  # the night
        decision = self.events.decide(PERGOLA, NOW + 600, "suspicious", people=2, alert_id="later")
        self.assertFalse(decision.notify)                               # the 142 alerts of 2026-10-07 stop
        self.assertTrue(self.events.decide(GATE, NOW + 600, "suspicious", people=1).notify)   # other cameras alert
        self.assertTrue(self.events.decide(PERGOLA, NOW + 700, "escalation", people=2).notify)  # never an escalation

    def test_without_an_alert_it_uses_the_topic_camera_or_asks_which_camera(self) -> None:
        ctx = self.ctx("זה אני עד 18:00")
        out = mark_known(ctx, {"who": "עמיר", "owner_words": "זה אני"})
        self.assertFalse(out["ok"])
        self.assertEqual(ctx.clarification["choices"], [PERGOLA, GATE])   # buttons; the inbox shows their names
        self.assertEqual(self.events.list_known(NOW), [])
        ctx = self.ctx("זה אני עד 18:00")
        ctx.state.set_topic_camera(GATE, "", NOW)
        self.assertTrue(mark_known(ctx, {"who": "עמיר", "owner_words": "זה אני"})["ok"])
        self.assertEqual(self.events.list_known(NOW)[0]["camera"], GATE)
        self.assertEqual(self.verdicts(), [])                            # no alert: no verdict

    def test_it_needs_the_owners_words_and_the_event_book(self) -> None:
        self.assertFalse(mark_known(self.ctx(WORKERS, alert=True), {"who": "x", "owner_words": "גנבים בחצר"})["ok"])
        ctx = self.ctx(WORKERS, alert=True)
        ctx.services.events = None
        self.assertFalse(mark_known(ctx, {"who": "x", "owner_words": "עובדים אצלי"})["ok"])
        self.assertIn("mark_known", TOOLS)

    def test_record_verdict_refuses_questions_and_insults(self) -> None:
        for text, words in (("אם זה תקין אז למה אתה ממשיך לשלוח לי הודעות יא מטומטם", "זה תקין"),
                            ("מה הקשר התרעה צפויה", "התרעה צפויה"), ("די עם ההודעה המטופשת הזאת", "ההודעה המטופשת")):
            with self.subTest(text=text):
                ctx = self.ctx(text)
                ctx.alert_handle = ctx.state.add_handle("event", ALERT["alert_id"], PERGOLA, ALERT["ts"], "")
                out = record_verdict(ctx, {"verdict": "expected", "owner_words": words})
                self.assertFalse(out["ok"])
                self.assertEqual(ctx.receipts, [])
        self.assertEqual(self.verdicts(), [])
        ctx = self.ctx("זה לא התרעה זה תקין")
        ctx.alert_handle = ctx.state.add_handle("event", ALERT["alert_id"], PERGOLA, ALERT["ts"], "")
        self.assertTrue(record_verdict(ctx, {"verdict": "expected", "owner_words": "זה תקין"})["ok"])

    def test_recent_activity_names_cameras_and_says_what_the_owner_said(self) -> None:
        self.events.decide(PERGOLA, NOW - 300, "suspicious", people=2, summary="Two men on a ladder.", alert_id="a1")
        self.events.mark_known(PERGOLA, "העובדים", "Ameer", END_OF_DAY, now=NOW - 200)
        out = recent_activity(self.ctx("מה הם עושים עכשיו בפרגולה?"), {"camera": "פרגולה", "minutes": 30})
        self.assertTrue(out["ok"])
        (event,) = out["events"]
        self.assertEqual((event["camera"], event["most_people"]), ("פרגולה", 2))
        self.assertEqual(event["seen"][0]["what"], "Two men on a ladder.")
        self.assertEqual(out["owner_said"], [{"camera": "פרגולה", "who": "העובדים", "until": "23:59"}])
        self.assertNotIn("ameer_week", json.dumps(out, ensure_ascii=False))
        noon = recent_activity(self.ctx("מה קרה בצהריים?"), {"time_from": "09:00"})
        self.assertEqual(noon["since"], "09:00")

    # -- the agent -----------------------------------------------------------------------------------------------
    def test_the_reply_is_the_keepers_receipt_only_with_cancel_and_week_buttons(self) -> None:
        big = Scripted([call("mark_known", who="העובדים", owner_words="עובדים אצלי"),
                        reply("רשמתי את זה כהתרעה צפויה. אם יש משהו נוסף, אני כאן!")])
        out = self.agent(big).handle(WORKERS_18, "-5", {"user_id": 1, "name": "Ameer"}, dict(ALERT), True)
        # Two lines, two stores: the clip's TAG (🏷️) and the MEMORY (🧠) with the week assumption said plainly.
        self.assertEqual(out.text, "🏷️ נשמר כתיוג לסרטון 09:50 (פרגולה): תקין: עובדים אצלי\n"
                                   "🧠 זכרתי: העובדים בפרגולה, כל יום 09:00–18:00, עד יום ג׳ 13.10.")
        known_id = self.events.list_known(NOW)[0]["id"]
        self.assertEqual(out.rows, ((("רק היום", f"kn:d:{known_id}"), ("שבוע ✓", f"kn:w:{known_id}"),
                                     ("אחר…", f"kn:o:{known_id}")), (("↩ זיכרון", f"kn:x:{known_id}"),),
                                    (("↩ תיוג", f"tu:{ALERT['alert_id']}"),)))
        self.assertIn("[OWNER SAYS WHO IS THERE]", big.seen[0][0][-1])
        self.assertIn("mark_known", big.seen[0][1])

    def test_cancel_and_all_week_buttons(self) -> None:
        agent = self.agent(Scripted([call("mark_known", who="השכן", owner_words="זה השכן"), reply("")]))
        agent.handle("זה השכן עד 18:00", "-5", {"user_id": 1}, dict(ALERT), True)
        known_id = self.events.list_known(NOW)[0]["id"]
        week = agent.known_button("-5", "w", known_id, {"user_id": 1})
        (known,) = self.events.list_known(NOW)
        self.assertEqual(known["until"], NOW + 7 * 86400)
        self.assertIn("🧠 זכרתי: השכן בפרגולה", week.text)
        self.assertEqual(week.rows, ((("ביטול", f"kn:x:{known['id']}"),),))
        cancel = agent.known_button("-5", "x", known["id"], {"user_id": 1})
        self.assertEqual(cancel.text, "בוטל. אשלח שוב הודעות על פרגולה.")
        self.assertEqual(self.events.list_known(NOW), [])
        self.assertEqual(agent.known_button("-5", "x", known["id"], {"user_id": 1}).text, t("known_gone", "he"))

    def test_which_video_is_answered_in_code_with_the_event_being_discussed(self) -> None:
        sent = []

        def run_tool(ctx, name, args):
            sent.append((name, args))
            from home_guard_project.box.brain.tools import _issue, _result  # noqa: PLC0415

            return _result(_issue(ctx, "send_media", DONE, args["handle"], {"kind": "video", "bounds": "12:46"}))

        big = Scripted([])                                        # no model call at all
        agent = self.agent(big, run_tool=run_tool)
        handle = agent.note_alert("-5", dict(ALERT, ts=dt.datetime(2026, 10, 7, 9, 46).timestamp()))
        out = agent.handle("אני לא יודע על איזה סרטון אתה מדבר בכלל", "-5", {"user_id": 1})
        self.assertEqual(out.text, "התכוונתי לסרטון של 09:46 בפרגולה: Two men work on the pergola with a ladder.")
        self.assertEqual(out.buttons, ("📹 שלח את הסרטון",))
        self.assertEqual(big.seen, [])
        video = agent.handle_choice("-5", out.question_token, 0, {"user_id": 1})
        self.assertEqual(sent, [("send_media", {"handle": handle})])
        self.assertEqual(video.text, "✓ הסרטון נשלח (12:46)")

    def test_the_event_of_a_replied_alert_is_in_the_context(self) -> None:
        self.events.decide(PERGOLA, NOW - 300, "suspicious", people=3, summary="Three men on the roof.",
                           alert_id=ALERT["alert_id"])
        big = Scripted([reply("שלושה אנשים על הגג.")])
        self.agent(big).handle("מה הם עושים?", "-5", {"user_id": 1}, dict(ALERT), True)
        block = big.seen[0][0][-1]
        self.assertIn("[EVENT OF THIS ALERT]", block)
        self.assertIn("Three men on the roof.", block)
        self.assertIn('"most_people": 3', block)

    def test_no_closing_offer_and_no_unbacked_verdict_claim(self) -> None:
        out = self.agent(Scripted([reply("היו שני אירועים בבוקר. אם יש משהו נוסף שתרצה לדעת, אני כאן!")])).handle(
            "מה היה בכניסה הבוקר", "-5", {"user_id": 1})
        self.assertEqual(out.text, "היו שני אירועים בבוקר.")
        big = Scripted([reply("רשמתי את זה כהתרעה צפויה. תודה על ההבהרה!"), reply("רשמתי את זה כהתרעה צפויה.")])
        out = self.agent(big).handle("מדי פעם אני יוצא החוצה בלילה", "-5", {"user_id": 1})
        self.assertEqual(out.text, t("not_saved_verdict", "he"))
        self.assertEqual(self.verdicts(), [])           # conversation: in the chat log, never a feedback/ file


class BuildTest(unittest.TestCase):
    def test_v2_builds_with_the_boxs_real_settings(self) -> None:
        # The box (2026-10-08): vlm_provider openrouter, owner_language he, the three keys in the environment.
        from home_guard_project.box.brain.agent import build_owner_agent  # noqa: PLC0415

        settings = {"vlm_provider": "openrouter", "vlm_model": "qwen/qwen3.5-9b", "owner_language": "he"}
        env = {"OPENAI_API_KEY": "sk-test", "OPENROUTER_API_KEY": "or-test", "TELEGRAM_BOT_TOKEN": "1:abc"}
        with tempfile.TemporaryDirectory() as root:
            book = EventBook(os.path.join(root, "events"))
            with patch("home_guard_project.box.embeddings.make_embedder", return_value=None), \
                    patch("home_guard_project.box.brain.agent._event_book", return_value=book):
                agent, deliverer = build_owner_agent(settings, env, None, Mock(bot_token="1:abc"), root, root, root)
            self.assertIsNotNone(agent)
            self.assertIsNotNone(deliverer)
            self.assertIs(agent.services.events, book)
            self.assertIsNotNone(agent.services.vision)                 # the Eye's provider, OpenRouter


if __name__ == "__main__":
    unittest.main()
