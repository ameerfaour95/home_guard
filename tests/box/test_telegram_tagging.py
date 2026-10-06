"""Task 18b: tagging an alert from Telegram - label buttons, and "Other…" that waits for the owner's words."""

from __future__ import annotations

import datetime as dt
import json
import os
import urllib.error
import tempfile
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

import numpy as np

from home_guard_project.box.agent import AgentReply
from home_guard_project.box.alert_clips import encode_frame, write_alert_clip
from home_guard_project.box.brain.i18n import t
from home_guard_project.box.feedback import (
    OWNER_LABELS, AlertIndex, Feedback, MuteState, undo_training_tag, verdict_for,
)
from home_guard_project.box.telegram_agent import PendingTags, TelegramInbox, feedback_keyboard, send_alert
from home_guard_project.box.telegram_notify import TelegramConfig

NOW = 1_800_000_000.0
CHAT = "-1001"
ALERT_ID = "door_1800000000_alert"
ALERT = {"alert_id": ALERT_ID, "camera": "door", "summary": "a person at the door", "label": "suspicious",
         "ts": NOW - 60}
ALERT_META = {"summary": "A person is at the door.", "alert_command": "[send_message]", "alert_reason": "",
              "labels": ["person"], "dispatch": {"sent": True}}
DANA = {"id": 42, "first_name": "Dana"}
OMER = {"id": 43, "first_name": "Omer"}


class FakeTelegram:
    """Records every Bot API call and answers like Telegram would."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._next_message_id = 900

    def post(self, token: str, method: str, fields: Dict[str, str], timeout: float = 15.0) -> Dict[str, Any]:
        self._next_message_id += 1
        self.calls.append({"method": method, "fields": fields, "message_id": self._next_message_id})
        return {"ok": True, "result": {"message_id": self._next_message_id}}

    def post_multipart(self, token: str, method: str, fields: Dict[str, str], files: Dict[str, Any],
                       timeout: float = 20.0) -> Dict[str, Any]:
        return self.post(token, method, fields, timeout)

    def sent(self, method: str) -> List[Dict[str, Any]]:
        return [c for c in self.calls if c["method"] == method]

    def texts(self) -> List[str]:
        return [c["fields"]["text"] for c in self.sent("sendMessage")]


class FakeAgentV1:
    version = 1

    def __init__(self) -> None:
        self.seen: List[str] = []

    def handle(self, text: str, chat_id: Any, who: Any = None, alert: Any = None) -> AgentReply:
        self.seen.append(text)
        return AgentReply(text="agent answer")


class FakeAgentV2:
    version = 2

    def __init__(self) -> None:
        self.seen: List[str] = []

    def handle(self, text: str, chat_id: Any, who: Any = None, alert: Any = None, threaded: bool = False) -> Any:
        self.seen.append(text)

        class Reply:
            text = "agent answer"

        return Reply()


def tap(update_id: int, code: str, on_message: int, sender: Optional[Dict[str, Any]] = None) -> dict:
    return {"update_id": update_id, "callback_query": {
        "id": f"cb{update_id}", "data": code, "from": dict(sender or DANA),
        "message": {"message_id": on_message, "chat": {"id": int(CHAT)}},
    }}


def text(update_id: int, words: str, sender: Optional[Dict[str, Any]] = None,
         reply_to: Optional[int] = None) -> dict:
    msg: Dict[str, Any] = {"message_id": 500 + update_id, "chat": {"id": int(CHAT)}, "text": words,
                           "from": dict(sender or DANA, is_bot=False)}
    if reply_to is not None:
        msg["reply_to_message"] = {"message_id": reply_to}
    return {"update_id": update_id, "message": msg}


def markup(call: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(call["fields"].get("reply_markup") or "{}")


def callback_codes(call: Dict[str, Any]) -> List[str]:
    return [b["callback_data"] for row in markup(call).get("inline_keyboard", []) for b in row]


class VerdictForTest(unittest.TestCase):
    def test_every_owner_label_against_every_ai_label(self) -> None:
        self.assertEqual(OWNER_LABELS, ("normal", "suspicious", "escalation", "empty", "other", "rule_mismatch"))
        expected = {
            "normal": {"normal": "expected", "suspicious": "expected", "escalation": "expected", "": "expected"},
            "empty": {"normal": "false_alarm", "suspicious": "false_alarm", "escalation": "false_alarm",
                      "": "false_alarm"},
            "suspicious": {"normal": "real_but_wrong", "suspicious": "true_alert", "escalation": "real_but_wrong",
                           "": "real_but_wrong"},
            "escalation": {"normal": "real_but_wrong", "suspicious": "real_but_wrong", "escalation": "true_alert",
                           "": "real_but_wrong"},
            "other": {"normal": "real_but_wrong", "suspicious": "real_but_wrong", "escalation": "real_but_wrong",
                      "": "real_but_wrong"},
            "rule_mismatch": {"normal": "expected", "suspicious": "expected", "escalation": "expected",
                              "": "expected"},
        }
        for owner, by_ai in expected.items():
            for ai, verdict in by_ai.items():
                self.assertEqual(verdict_for(owner, ai), verdict, (owner, ai))

    def test_feedback_has_the_tag_fields_with_empty_defaults(self) -> None:
        fb = Feedback()
        self.assertEqual((fb.owner_label, fb.owner_text, fb.tagged_by), ("", "", ""))


class KeyboardTest(unittest.TestCase):
    def _rows(self, raw: str) -> List[List[Dict[str, str]]]:
        return json.loads(raw)["inline_keyboard"]

    def test_one_row_suspicious_normal_other(self) -> None:
        rows = self._rows(feedback_keyboard("en"))
        self.assertEqual([[b["callback_data"] for b in row] for row in rows],
                         [["tag:suspicious", "tag:normal", "tag:other"]])
        self.assertEqual([[b["text"] for b in row] for row in rows], [["🟡 Suspicious", "🟢 Normal", "✏️ Other…"]])

    def test_the_ai_label_gets_a_tick(self) -> None:
        texts = [b["text"] for row in self._rows(feedback_keyboard("en", ai_label="suspicious")) for b in row]
        self.assertEqual(texts[0], "✓ 🟡 Suspicious")
        self.assertEqual([x for x in texts if x.startswith("✓")], ["✓ 🟡 Suspicious"])
        unticked = [b["text"] for row in self._rows(feedback_keyboard("en", ai_label="bogus")) for b in row]
        self.assertFalse(any(x.startswith("✓") for x in unticked))

    def test_hebrew_texts(self) -> None:
        texts = [b["text"] for row in self._rows(feedback_keyboard("he", ai_label="normal")) for b in row]
        self.assertEqual(texts, [
            f"🟡 {t('btn_tag_suspicious', 'he')}", f"✓ 🟢 {t('btn_tag_normal', 'he')}", f"✏️ {t('btn_tag_other', 'he')}",
        ])
        self.assertNotEqual(t("btn_tag_normal", "he"), t("btn_tag_normal", "en"))

    def test_the_english_strings(self) -> None:
        self.assertEqual([t(k, "en") for k in ("btn_tag_normal", "btn_tag_suspicious", "btn_tag_escalation",
                                               "btn_tag_empty", "btn_tag_other")],
                         ["Normal", "Suspicious", "Escalation", "Nothing there", "Other…"])
        self.assertEqual(t("tag_ask_text", "en"), "What's happening in the clip? Write or send a voice message.")
        self.assertEqual(t("tag_saved", "en", label="Normal"), "✓ Saved as Normal.")
        self.assertEqual(t("tag_receipt", "en", label="Normal"), "✓ Saved: Normal")
        self.assertEqual(t("tag_saved_explanation", "en", time="02:14"), "✓ Saved your explanation for the 02:14 clip.")
        self.assertEqual(t("tag_undone", "en"), "Tag removed.")
        self.assertEqual(t("tag_expired", "en"), "That tag request expired; tap Other… again.")

    def test_a_hebrew_alert_gets_hebrew_tag_buttons_with_the_ai_tick(self) -> None:
        # A Hebrew alert whose VLM parse failed still shows an English summary under a Hebrew header;
        # its buttons are in the box language all the same.
        tg = FakeTelegram()
        index = AlertIndex(os.path.join(tempfile.mkdtemp(), "index.json"))
        cfg = TelegramConfig(bot_token="T", chat_ids=[CHAT])
        send_alert(cfg, index, ALERT, "🟡 חשוד · door\na person at the door", post=tg.post, lang="he")
        (call,) = tg.sent("sendMessage")
        self.assertEqual(call["fields"]["reply_markup"], feedback_keyboard("he", ai_label="suspicious"))


class PendingTagsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = os.path.join(tempfile.mkdtemp(), "production_multi", ".pending_tags.json")

    def test_add_take_once(self) -> None:
        pending = PendingTags(self.path)
        pending.add(CHAT, 42, ALERT_ID, NOW)
        self.assertIsNone(pending.take(CHAT, 43, NOW + 5))           # another user
        self.assertEqual(pending.take(CHAT, 42, NOW + 5), ALERT_ID)
        self.assertIsNone(pending.take(CHAT, 42, NOW + 6))           # taken: gone

    def test_survives_a_restart(self) -> None:
        PendingTags(self.path).add(CHAT, 42, ALERT_ID, NOW)
        self.assertEqual(PendingTags(self.path).take(CHAT, 42, NOW + 30), ALERT_ID)
        self.assertIsNone(PendingTags(self.path).take(CHAT, 42, NOW + 31))

    def test_expires_after_ten_minutes(self) -> None:
        pending = PendingTags(self.path, ttl=600)
        pending.add(CHAT, 42, ALERT_ID, NOW)
        self.assertIsNone(pending.take(CHAT, 42, NOW + 601))
        self.assertIsNone(PendingTags(self.path).take(CHAT, 42, NOW + 1))   # dropped, not just skipped

    def test_cancel(self) -> None:
        pending = PendingTags(self.path)
        pending.add(CHAT, 42, ALERT_ID, NOW)
        pending.cancel(CHAT, 42)
        self.assertIsNone(pending.take(CHAT, 42, NOW + 1))

    def test_a_reply_picks_its_own_wait(self) -> None:
        pending = PendingTags(self.path)
        pending.add(CHAT, 42, "a", NOW, prompt_id=901)
        pending.add(CHAT, 42, "b", NOW + 1, prompt_id=902)
        self.assertIsNone(pending.take(CHAT, 42, NOW + 2, reply_to=555))              # some other message
        self.assertEqual(pending.take(CHAT, 42, NOW + 2, reply_to=555, reply_alert="a"), "a")
        self.assertEqual(pending.take(CHAT, 42, NOW + 3, reply_to=902), "b")
        self.assertIsNone(pending.take(CHAT, 42, NOW + 4))

    def test_the_expired_prompt_belongs_to_its_user(self) -> None:
        pending = PendingTags(self.path, ttl=600)
        pending.add(CHAT, 42, ALERT_ID, NOW, prompt_id=901)
        self.assertIsNone(pending.take(CHAT, 42, NOW + 601))
        self.assertFalse(pending.expired_prompt(CHAT, 901, 43, NOW + 602))
        self.assertTrue(pending.expired_prompt(CHAT, 901, 42, NOW + 603))
        self.assertFalse(pending.expired_prompt(CHAT, 901, 42, NOW + 604))         # once

    def test_a_damaged_file_is_an_empty_store(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{not json")
        pending = PendingTags(self.path)
        self.assertIsNone(pending.take(CHAT, 42, NOW))
        pending.add(CHAT, 42, ALERT_ID, NOW)
        self.assertEqual(pending.take(CHAT, 42, NOW + 1), ALERT_ID)

    def test_superseding_a_wait_is_scoped_to_its_user_and_chat(self) -> None:
        pending = PendingTags(self.path)
        pending.add(CHAT, 42, "a", NOW, prompt_id=901)
        pending.add(CHAT, 43, "omer", NOW, prompt_id=902)
        pending.add("other_chat", 42, "elsewhere", NOW, prompt_id=903)
        pending.add(CHAT, 42, "b", NOW, prompt_id=904)
        pending = PendingTags(self.path)
        self.assertEqual(pending.take(CHAT, 42, NOW + 1, reply_to=904), "b")
        self.assertIsNone(pending.take(CHAT, 42, NOW + 1))
        self.assertEqual(pending.take(CHAT, 43, NOW + 1), "omer")
        self.assertEqual(pending.take("other_chat", 42, NOW + 1), "elsewhere")
        self.assertEqual(pending.take(CHAT, 42, NOW + 1, reply_to=901), "a")


class InboxTaggingTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.production = os.path.join(self.dir, "production_multi")
        self.training = os.path.join(self.dir, "dataset_multi")
        self.archive = os.path.join(self.dir, "production_archive")
        self.frames = [(NOW - 60 + i * 0.2, encode_frame(np.zeros((48, 64, 3), dtype=np.uint8))) for i in range(5)]
        with mock.patch("home_guard_project.box.alert_clips._to_h264", return_value=False):
            write_alert_clip(self.production, "door", ALERT_ID, self.frames, ALERT_META)
        self.cfg = TelegramConfig(bot_token="T", chat_ids=[CHAT])
        self.index = AlertIndex(os.path.join(self.dir, "alert_index.json"))
        self.index.remember(CHAT, 77, ALERT)
        self.mute = MuteState(os.path.join(self.dir, "alert_mute.json"))
        self.clock = NOW
        self.lang = "en"
        self.tg = FakeTelegram()

    def _inbox(self, agent: Any = None) -> TelegramInbox:
        return TelegramInbox(self.cfg, agent, self.index, self.mute, self.production,
                             os.path.join(self.dir, "telegram_offset.json"),
                             post=self.tg.post, post_multipart=self.tg.post_multipart, now=lambda: self.clock,
                             training_dir=self.training, archive_dir=self.archive, lang=lambda: self.lang)

    def _saved(self) -> List[dict]:
        paths = []
        for dirpath, _, names in os.walk(os.path.join(self.production, "feedback")):
            paths += [os.path.join(dirpath, n) for n in names]
        found = []
        for path in sorted(paths, key=os.path.basename):
            with open(path, encoding="utf-8") as f:
                found.append(json.load(f))
        return found

    def _training_meta(self) -> Optional[Dict[str, Any]]:
        for dirpath, _, names in os.walk(os.path.join(self.training, "meta")):
            for name in names:
                if name == f"{ALERT_ID}.meta.json":
                    with open(os.path.join(dirpath, name), encoding="utf-8") as f:
                        return json.load(f)
        return None

    def _training_clip(self) -> str:
        return os.path.join(self.training, "clips", "door",
                            os.path.basename(os.path.dirname(self._production_clip())), f"{ALERT_ID}.mp4")

    def _production_clip(self) -> str:
        for dirpath, _, names in os.walk(os.path.join(self.production, "clips")):
            if f"{ALERT_ID}.mp4" in names:
                return os.path.join(dirpath, f"{ALERT_ID}.mp4")
        raise AssertionError("no production clip")

    # -- a label tap ---------------------------------------------------------------
    def test_a_suspicious_tag_on_a_suspicious_alert_is_a_true_alert_with_the_clip_and_an_undo(self) -> None:
        inbox = self._inbox()
        inbox.handle_update(tap(1, "tag:suspicious", 77))

        (saved,) = self._saved()
        self.assertEqual((saved["verdict"], saved["owner_label"], saved["source"]), ("true_alert", "suspicious", "button"))
        self.assertEqual((saved["tagged_by"], saved["owner_text"]), ("Dana", ""))
        self.assertEqual(saved["alert"]["alert_id"], ALERT_ID)
        meta = self._training_meta()
        self.assertIsNotNone(meta)
        self.assertEqual(meta["owner_feedback"][-1]["owner_label"], "suspicious")
        self.assertEqual(meta["owner_feedback"][-1]["ai_label"], "suspicious")
        self.assertTrue(os.path.isfile(self._training_clip()))
        self.assertEqual(len(self.tg.sent("answerCallbackQuery")), 1)
        self.assertEqual(self.tg.sent("sendMessage"), [])
        (edit,) = self.tg.sent("editMessageReplyMarkup")
        self.assertEqual((edit["fields"]["chat_id"], edit["fields"]["message_id"]), (CHAT, "77"))
        self.assertEqual(markup(edit), {"inline_keyboard": [[
            {"text": "✓ Saved: Suspicious", "callback_data": "tag:noop"},
            {"text": "↩ Undo", "callback_data": f"tu:{ALERT_ID}"}]]})
        self.assertIn(ALERT_ID, inbox.answered)                   # the escalation reminder stops

    def test_a_receipt_telegram_will_not_edit_is_a_message_with_undo(self) -> None:
        post = self.tg.post

        def refuse_edits(token, method, fields, timeout=15.0):
            if method == "editMessageReplyMarkup":
                return {"ok": False, "description": "Bad Request: message can't be edited"}
            return post(token, method, fields, timeout)

        self.tg.post = refuse_edits
        self._inbox().handle_update(tap(1, "tag:suspicious", 77))
        (said,) = self.tg.sent("sendMessage")
        self.assertEqual(said["fields"]["text"], "✓ Saved as Suspicious.")
        self.assertEqual(callback_codes(said), [f"tu:{ALERT_ID}"])

    def test_undo_on_the_receipt_brings_the_buttons_back_without_a_message(self) -> None:
        inbox = self._inbox()
        inbox.handle_update(tap(1, "tag:normal", 77))
        inbox.handle_update(tap(2, f"tu:{ALERT_ID}", 77))
        self.assertEqual([s["note"] for s in self._saved()], ["", "tag undone"])
        restored = self.tg.sent("editMessageReplyMarkup")[-1]
        self.assertEqual(restored["fields"]["message_id"], "77")
        self.assertEqual(restored["fields"]["reply_markup"], feedback_keyboard("en", ai_label="suspicious"))
        self.assertEqual(self.tg.sent("answerCallbackQuery")[-1]["fields"]["text"], "Tag removed.")
        self.assertEqual(self.tg.sent("sendMessage"), [])

    def test_tapping_the_receipt_itself_does_nothing(self) -> None:
        self._inbox().handle_update(tap(1, "tag:noop", 77))
        self.assertEqual(self._saved(), [])
        self.assertEqual([c["method"] for c in self.tg.calls], ["answerCallbackQuery"])

    def test_a_different_label_is_real_but_wrong_and_the_newest_tag_is_last(self) -> None:
        inbox = self._inbox()
        inbox.handle_update(tap(1, "tag:escalation", 77))
        self.clock += 5
        inbox.handle_update(tap(2, "tag:empty", 77))
        self.assertEqual([(s["owner_label"], s["verdict"]) for s in self._saved()],
                         [("escalation", "real_but_wrong"), ("empty", "false_alarm")])

    def test_a_tag_never_pauses(self) -> None:
        inbox = self._inbox()
        for i, label in enumerate(OWNER_LABELS):
            inbox.handle_update(tap(i + 1, f"tag:{label}", 77))
        self.assertEqual(self.mute.snapshot(), {"all": 0.0, "cameras": {}})

    def test_a_tag_confirmation_speaks_the_box_language(self) -> None:
        self.lang = "he"
        self._inbox().handle_update(tap(1, "tag:normal", 77))
        (edit,) = self.tg.sent("editMessageReplyMarkup")
        receipt, undo = markup(edit)["inline_keyboard"][0]
        self.assertEqual(receipt["text"], t("tag_receipt", "he", label=t("btn_tag_normal", "he")))
        self.assertEqual(undo["text"], t("undo_button", "he"))

    def test_a_tag_on_an_unknown_message_saves_nothing(self) -> None:
        self._inbox().handle_update(tap(1, "tag:normal", 12345))
        self.assertEqual(self._saved(), [])
        self.assertEqual(self.tg.texts(), [t("button_unused", "en")])

    def test_a_failure_never_raises_and_answers_the_tap_with_a_short_error(self) -> None:
        with mock.patch("home_guard_project.box.telegram_agent.save_feedback", side_effect=OSError("disk full")):
            self._inbox().handle_update(tap(1, "tag:suspicious", 77))
        answers = self.tg.sent("answerCallbackQuery")
        self.assertTrue(answers)
        self.assertTrue(answers[-1]["fields"].get("text"))

    # -- "Other…" ---------------------------------------------------------------------
    def test_other_asks_with_force_reply_and_the_next_text_is_the_tag(self) -> None:
        agent = FakeAgentV1()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))

        (ask,) = self.tg.sent("sendMessage")
        self.assertIn(t("tag_ask_text", "en"), ask["fields"]["text"])
        self.assertEqual(ask["fields"]["reply_to_message_id"], "77")
        self.assertEqual(markup(ask), {"force_reply": True, "selective": True})
        mentions = json.loads(ask["fields"].get("entities") or "[]")
        self.assertEqual([m["user"]["id"] for m in mentions if m["type"] == "text_mention"], [42])
        self.assertIn(ALERT_ID, inbox.answered)
        self.assertEqual(self._saved(), [])

        self.clock += 30
        inbox.handle_update(text(2, "  two kids on bikes  ", reply_to=ask["message_id"]))
        (saved,) = self._saved()
        self.assertEqual((saved["owner_label"], saved["owner_text"], saved["verdict"]),
                         ("other", "two kids on bikes", "real_but_wrong"))
        self.assertEqual(saved["tagged_by"], "Dana")
        self.assertEqual(self._training_meta()["owner_feedback"][-1]["owner_text"], "two kids on bikes")
        said = self.tg.sent("sendMessage")[-1]
        clock = dt.datetime.fromtimestamp(ALERT["ts"]).strftime("%H:%M")
        self.assertEqual(said["fields"]["text"], f"✓ Saved your explanation for the {clock} clip.")
        self.assertEqual(callback_codes(said), [f"tu:{ALERT_ID}"])
        (receipt,) = self.tg.sent("editMessageReplyMarkup")              # the alert shows it too
        self.assertEqual((receipt["fields"]["message_id"], callback_codes(receipt)), ("77", ["tag:noop", f"tu:{ALERT_ID}"]))
        self.assertEqual(agent.seen, [])

    def test_the_v2_agent_does_not_see_the_tag_text_either(self) -> None:
        agent = FakeAgentV2()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        inbox.handle_update(text(2, "my brother"))
        self.assertEqual(agent.seen, [])
        self.assertEqual(self._saved()[0]["owner_text"], "my brother")

    def test_another_users_text_goes_to_the_agent(self) -> None:
        agent = FakeAgentV1()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        inbox.handle_update(text(2, "who is that?", sender=OMER))
        self.assertEqual(agent.seen, ["who is that?"])
        self.assertEqual(self._saved(), [])
        inbox.handle_update(text(3, "the gardener"))                # Dana's tag still waits
        self.assertEqual(self._saved()[0]["owner_text"], "the gardener")

    def test_a_long_text_is_trimmed_to_500_characters(self) -> None:
        inbox = self._inbox(FakeAgentV1())
        inbox.handle_update(tap(1, "tag:other", 77))
        inbox.handle_update(text(2, "x" * 900))
        self.assertEqual(len(self._saved()[0]["owner_text"]), 500)

    def test_a_command_cancels_the_pending_tag(self) -> None:
        agent = FakeAgentV1()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        inbox.handle_update(text(2, "/status"))
        inbox.handle_update(text(3, "hello"))
        self.assertEqual(agent.seen, ["/status", "hello"])
        self.assertEqual(self._saved(), [])

    def test_another_button_tap_by_that_user_cancels_the_pending_tag(self) -> None:
        agent = FakeAgentV1()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        inbox.handle_update(tap(2, "fb:true", 77))
        inbox.handle_update(text(3, "hello"))
        self.assertEqual(agent.seen, ["hello"])

    def test_after_half_an_hour_a_plain_message_goes_to_the_assistant(self) -> None:
        agent = FakeAgentV1()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        self.clock += 1801
        inbox.handle_update(text(2, "it's me, pause until six"))
        self.assertEqual(agent.seen, ["it's me, pause until six"])
        self.assertEqual(self._saved(), [])
        self.assertNotIn(t("tag_expired", "en"), self.tg.texts())

    def test_a_reply_to_the_question_or_the_clip_is_the_answer_for_a_day(self) -> None:
        agent = FakeAgentV1()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        self.clock += 23 * 3600
        inbox.handle_update(text(2, "the neighbour's son", reply_to=77))      # a reply to the video itself
        self.assertEqual(agent.seen, [])
        self.assertEqual(self._saved()[0]["owner_text"], "the neighbour's son")

    def test_a_reply_to_an_expired_prompt_says_it_expired(self) -> None:
        agent = FakeAgentV1()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        prompt_id = self.tg.sent("sendMessage")[0]["message_id"]    # the "Other…" question
        self.clock += 86401
        inbox.handle_update(text(2, "a delivery", reply_to=prompt_id))
        self.assertEqual(agent.seen, [])
        self.assertEqual(self._saved(), [])
        self.assertEqual(self.tg.texts()[-1], t("tag_expired", "en"))

    def test_the_pending_tag_survives_an_inbox_restart(self) -> None:
        self._inbox().handle_update(tap(1, "tag:other", 77))
        agent = FakeAgentV1()
        self._inbox(agent).handle_update(text(2, "a delivery"))
        self.assertEqual(agent.seen, [])
        self.assertEqual(self._saved()[0]["owner_text"], "a delivery")

    # -- Undo ---------------------------------------------------------------------------
    def test_undo_appends_a_record_and_removes_the_clip_the_tag_copied(self) -> None:
        inbox = self._inbox()
        inbox.handle_update(tap(1, "tag:suspicious", 77))
        self.assertTrue(os.path.isfile(self._training_clip()))
        self.clock += 10
        inbox.handle_update(tap(2, f"tu:{ALERT_ID}", 902))

        first, undo = self._saved()
        self.assertEqual((undo["verdict"], undo["owner_label"], undo["note"], undo["source"]),
                         ("none", "", "tag undone", "button"))
        self.assertEqual(undo["alert"]["alert_id"], ALERT_ID)
        self.assertFalse(os.path.isfile(self._training_clip()))
        self.assertIsNone(self._training_meta())
        self.assertTrue(os.path.isfile(self._production_clip()))    # the owner's copy stays
        self.assertEqual(self.tg.texts()[-1], "Tag removed.")

    def test_undo_keeps_a_training_copy_the_tag_did_not_create(self) -> None:
        # Inference already keeps every real alert in the training folder (kind "alert").
        with mock.patch("home_guard_project.box.alert_clips._to_h264", return_value=False):
            write_alert_clip(self.training, "door", ALERT_ID, self.frames, ALERT_META, kind="alert")
        inbox = self._inbox()
        inbox.handle_update(tap(1, "tag:normal", 77))
        inbox.handle_update(tap(2, f"tu:{ALERT_ID}", 902))
        self.assertTrue(os.path.isfile(self._training_clip()))
        answers = self._training_meta()["owner_feedback"]
        self.assertEqual([(a["verdict"], a.get("note")) for a in answers],
                         [("expected", ""), ("none", "tag undone")])

    # -- the old buttons ---------------------------------------------------------------
    def test_old_fb_true_taps_still_work(self) -> None:
        inbox = self._inbox()
        inbox.handle_update(tap(1, "fb:true", 77))
        (saved,) = self._saved()
        self.assertEqual((saved["verdict"], saved["source"], saved["owner_label"]), ("true_alert", "button", ""))
        self.assertEqual(self.tg.texts(), [t("verdict_saved", "en", verdict=t("verdict_true_alert", "en"))])
        self.assertIn(ALERT_ID, inbox.answered)

    def test_old_button_confirmations_speak_the_box_language(self) -> None:
        self.lang = "he"
        inbox = self._inbox()
        inbox.handle_update(tap(1, "fb:false", 77))
        inbox.handle_update(tap(2, "fb:mute60", 77))
        inbox.handle_update(tap(3, "fb:whatever", 77))
        said = self.tg.texts()
        self.assertEqual(said[0], t("verdict_saved", "he", verdict=t("verdict_false_alarm", "he")))
        self.assertTrue(said[1].startswith(t("paused_all", "he", until="")[:10]))
        self.assertEqual(said[2], t("button_unused", "he"))
        self.assertTrue(self.mute.is_muted(NOW + 60, "door"))


    # -- review fixes: a tag never swallows a message meant for the assistant ----------------
    B_ID = "front_1800000100_alert"

    def _remember_b(self) -> None:
        self.index.remember(CHAT, 88, {"alert_id": self.B_ID, "camera": "front", "label": "normal", "ts": NOW})

    def _prompts(self) -> List[int]:
        return [c["message_id"] for c in self.tg.sent("sendMessage")
                if t("tag_ask_text", "en") in c["fields"]["text"]]

    def test_a_reply_to_another_alert_reaches_the_agent_and_the_wait_stays(self) -> None:
        self._remember_b()
        agent = FakeAgentV1()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        inbox.handle_update(text(2, "who is this at the front?", reply_to=88))
        self.assertEqual(agent.seen, ["who is this at the front?"])
        self.assertEqual(self._saved(), [])
        (prompt,) = self._prompts()
        inbox.handle_update(text(3, "the gardener", reply_to=prompt))   # the wait is still there
        (saved,) = self._saved()
        self.assertEqual((saved["alert"]["alert_id"], saved["owner_text"]), (ALERT_ID, "the gardener"))

    def test_a_reply_to_an_unrelated_message_reaches_the_agent(self) -> None:
        agent = FakeAgentV2()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        inbox.handle_update(text(2, "and the back door?", reply_to=4321))
        self.assertEqual(agent.seen, ["and the back door?"])
        self.assertEqual(self._saved(), [])

    def test_a_reply_to_the_pending_alert_itself_is_the_tag(self) -> None:
        agent = FakeAgentV1()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        inbox.handle_update(text(2, "the neighbour", reply_to=77))
        self.assertEqual(agent.seen, [])
        self.assertEqual(self._saved()[0]["owner_text"], "the neighbour")

    def test_a_reply_to_the_first_prompt_saves_under_the_first_alert(self) -> None:
        self._remember_b()
        inbox = self._inbox(FakeAgentV1())
        inbox.handle_update(tap(1, "tag:other", 77))
        inbox.handle_update(tap(2, "tag:other", 88))
        prompt_a, prompt_b = self._prompts()
        inbox.handle_update(text(3, "the gardener", reply_to=prompt_a))
        inbox.handle_update(text(4, "a fox", reply_to=prompt_b))
        self.assertEqual(sorted((s["alert"]["alert_id"], s["owner_text"]) for s in self._saved()),
                         [(ALERT_ID, "the gardener"), (self.B_ID, "a fox")])

    def test_a_plain_text_answers_the_newest_wait_only_once(self) -> None:
        self._remember_b()
        agent = FakeAgentV1()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        inbox.handle_update(tap(2, "tag:other", 88))
        inbox.handle_update(text(3, "a fox"))                       # the newest question: B
        inbox.handle_update(text(4, "is the door locked?"))         # not a second tag
        self.assertEqual([(s["alert"]["alert_id"], s["owner_text"]) for s in self._saved()], [(self.B_ID, "a fox")])
        self.assertEqual(agent.seen, ["is the door locked?"])
        prompt_a = self._prompts()[0]
        inbox.handle_update(text(5, "the gardener", reply_to=prompt_a))   # A still takes a reply to its question
        self.assertEqual(sorted((s["alert"]["alert_id"], s["owner_text"]) for s in self._saved()),
                         [(ALERT_ID, "the gardener"), (self.B_ID, "a fox")])
        self.assertEqual(agent.seen, ["is the door locked?"])

    def test_another_users_reply_to_an_expired_prompt_reaches_the_agent(self) -> None:
        agent = FakeAgentV1()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        (prompt,) = self._prompts()
        self.clock += 86401
        inbox.handle_update(text(2, "is the door locked?", sender=OMER, reply_to=prompt))
        self.assertEqual(agent.seen, ["is the door locked?"])
        self.assertNotIn(t("tag_expired", "en"), self.tg.texts())
        inbox.handle_update(text(3, "a delivery", reply_to=prompt))       # Dana's own late reply
        self.assertEqual(self.tg.texts()[-1], t("tag_expired", "en"))
        self.assertEqual(agent.seen, ["is the door locked?"])

    def test_an_expired_prompt_keeps_its_user_across_a_restart(self) -> None:
        self._inbox().handle_update(tap(1, "tag:other", 77))
        (prompt,) = self._prompts()
        self.clock += 86401
        self._inbox(FakeAgentV1()).handle_update(text(2, "hi", sender=OMER))      # the purge writes the file
        agent = FakeAgentV1()
        inbox = self._inbox(agent)
        inbox.handle_update(text(3, "who?", sender=OMER, reply_to=prompt))
        inbox.handle_update(text(4, "a delivery", reply_to=prompt))
        self.assertEqual(agent.seen, ["who?"])
        self.assertEqual(self.tg.texts()[-1], t("tag_expired", "en"))

    def test_undo_with_a_corrupt_training_meta_reports_partial_completion(self) -> None:
        inbox = self._inbox()
        inbox.handle_update(tap(1, "tag:suspicious", 77))
        meta_path = None
        for dirpath, _, names in os.walk(os.path.join(self.training, "meta")):
            if f"{ALERT_ID}.meta.json" in names:
                meta_path = os.path.join(dirpath, f"{ALERT_ID}.meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.clock += 10
        inbox.handle_update(tap(2, f"tu:{ALERT_ID}", 902))
        self.assertEqual([s["note"] for s in self._saved()], ["", "tag undone"])
        self.assertEqual(self.tg.texts()[-1], "Part of the undo failed; tap Undo again.")
        self.assertEqual(callback_codes(self.tg.sent("sendMessage")[-1]), [f"tu:{ALERT_ID}"])

    def test_undo_keeps_the_button_when_both_steps_fail(self) -> None:
        inbox = self._inbox()
        inbox.handle_update(tap(1, "tag:suspicious", 77))
        with mock.patch("home_guard_project.box.telegram_agent.save_feedback", side_effect=OSError("disk full")), \
                mock.patch("home_guard_project.box.telegram_agent.undo_training_tag", side_effect=OSError("disk")):
            inbox.handle_update(tap(2, f"tu:{ALERT_ID}", 902))
        self.assertNotEqual(self.tg.texts()[-1], "Tag removed.")
        self.assertEqual(self.tg.texts()[-1], "Part of the undo failed; tap Undo again.")
        self.assertEqual(callback_codes(self.tg.sent("sendMessage")[-1]), [f"tu:{ALERT_ID}"])

    def test_undo_still_removes_the_copy_when_the_record_cannot_be_saved(self) -> None:
        inbox = self._inbox()
        inbox.handle_update(tap(1, "tag:suspicious", 77))
        with mock.patch("home_guard_project.box.telegram_agent.save_feedback", side_effect=OSError("disk full")):
            inbox.handle_update(tap(2, f"tu:{ALERT_ID}", 902))
        self.assertIsNone(self._training_meta())
        self.assertEqual(self.tg.texts()[-1], "Part of the undo failed; tap Undo again.")
        self.assertEqual(callback_codes(self.tg.sent("sendMessage")[-1]), [f"tu:{ALERT_ID}"])
        inbox = self._inbox()
        inbox.handle_update(tap(3, f"tu:{ALERT_ID}", 902))
        inbox.handle_update(tap(4, f"tu:{ALERT_ID}", 902))
        self.assertEqual(self.tg.texts()[-2:], ["Tag removed.", "Tag removed."])
        self.assertEqual([s["note"] for s in self._saved()], ["", "tag undone"])

    def test_when_the_other_question_cannot_be_sent_the_tap_says_so_and_nothing_waits(self) -> None:
        self.lang = "he"
        real_post = self.tg.post

        def post(token: str, method: str, fields: Dict[str, str], timeout: float = 15.0) -> Dict[str, Any]:
            if method == "sendMessage":
                raise urllib.error.URLError("offline")
            return real_post(token, method, fields, timeout)

        agent = FakeAgentV1()
        inbox = self._inbox(agent)
        inbox._post = post
        with mock.patch("home_guard_project.box.telegram_agent.time.sleep"):
            inbox.handle_update(tap(1, "tag:other", 77))
        answers = self.tg.sent("answerCallbackQuery")
        self.assertEqual([a["fields"].get("text") for a in answers], [t("tag_ask_failed", "he")])
        inbox._post = real_post
        inbox.handle_update(text(2, "hello"))
        self.assertEqual(agent.seen, ["hello"])
        self.assertEqual(self._saved(), [])

    def test_the_ask_failed_strings(self) -> None:
        for lang in ("en", "he", "ar"):
            self.assertTrue(t("tag_ask_failed", lang))
        self.assertNotEqual(t("tag_ask_failed", "he"), t("tag_ask_failed", "en"))
        self.assertNotEqual(t("tag_ask_failed", "ar"), t("tag_ask_failed", "en"))

    def test_newest_prompt_reply_never_rearms_an_older_wait(self) -> None:
        self._remember_b()
        for restart in (False, True):
            with self.subTest(restart=restart):
                agent = FakeAgentV1()
                inbox = self._inbox(agent)
                inbox.handle_update(tap(1, "tag:other", 77))
                inbox.handle_update(tap(2, "tag:other", 88))
                pa, pb = self._prompts()[-2:]
                inbox.handle_update(text(3, "fox at B", reply_to=pb))
                if restart:
                    inbox = self._inbox(agent)
                inbox.handle_update(text(4, "is the door locked?"))
                self.assertEqual(agent.seen, ["is the door locked?"])
                inbox.handle_update(text(5, "gardener at A", reply_to=pa))
                self.assertEqual([s for s in self._saved() if s["alert"]["alert_id"] == ALERT_ID][-1]["owner_text"],
                                 "gardener at A")

    def test_failed_new_prompt_demotes_older_wait_even_after_restart(self) -> None:
        self._remember_b()
        agent = FakeAgentV2()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        pa = self._prompts()[-1]
        with mock.patch.object(inbox, "_ask_for_tag", side_effect=OSError("offline")):
            inbox.handle_update(tap(2, "tag:other", 88))
        self.assertEqual(self.tg.sent("answerCallbackQuery")[-1]["fields"]["text"], t("tag_ask_failed", "en"))
        inbox.handle_update(text(3, "why can you not ask for the tag?"))
        self.assertEqual(agent.seen, ["why can you not ask for the tag?"])
        inbox = self._inbox(agent)
        inbox.handle_update(text(4, "is the door locked?"))
        self.assertEqual(agent.seen[-1], "is the door locked?")
        self.assertEqual(self._saved(), [])
        inbox.handle_update(text(5, "gardener", reply_to=pa))
        self.assertEqual(self._saved()[0]["owner_text"], "gardener")

    def test_completed_wait_is_reconciled_when_pending_write_fails(self) -> None:
        agent = FakeAgentV1()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        with mock.patch("home_guard_project.box.telegram_agent._write_json", side_effect=OSError("disk full")):
            inbox.handle_update(text(2, "the gardener"))
            inbox.handle_update(text(3, "first question"))
        self.assertEqual(agent.seen, ["first question"])
        inbox = self._inbox(agent)
        inbox.handle_update(text(4, "is the door locked?"))
        self.assertEqual(agent.seen, ["first question", "is the door locked?"])
        self.assertEqual([s["owner_text"] for s in self._saved()], ["the gardener"])
        self.assertTrue(self._saved()[0]["request_id"])
        # Completion belongs to this request, not all future tags of the same alert.
        inbox.handle_update(tap(5, "tag:other", 77))
        self._inbox(agent).handle_update(text(6, "the neighbour"))
        self.assertEqual(len({s["request_id"] for s in self._saved()}), 2)

    def test_undo_training_failure_is_partial_and_retry_survives_restart(self) -> None:
        inbox = self._inbox()
        inbox.handle_update(tap(1, "tag:suspicious", 77))
        with mock.patch("home_guard_project.box.feedback.os.remove", side_effect=PermissionError("locked")):
            inbox.handle_update(tap(2, f"tu:{ALERT_ID}", 902))
        self.assertEqual(self._saved()[-1]["note"], "tag undone")
        self.assertEqual(self.tg.texts()[-1], "Part of the undo failed; tap Undo again.")
        self.assertEqual(callback_codes(self.tg.sent("sendMessage")[-1]), [f"tu:{ALERT_ID}"])
        inbox = self._inbox()
        inbox.handle_update(tap(3, f"tu:{ALERT_ID}", 902))
        inbox.handle_update(tap(4, f"tu:{ALERT_ID}", 902))
        self.assertEqual(self.tg.texts()[-2:], ["Tag removed.", "Tag removed."])
        self.assertFalse(os.path.exists(self._training_clip()))
        self.assertIsNone(self._training_meta())
        self.assertEqual(len(self._saved()), 2)

    def test_media_to_a_wait_requests_text_and_preserves_the_wait(self) -> None:
        for kind in ("voice", "photo"):
            for reply in (False, True):
                with self.subTest(kind=kind, reply=reply):
                    agent = FakeAgentV2()
                    inbox = self._inbox(agent)
                    inbox.handle_update(tap(1, "tag:other", 77))
                    msg = text(2, "", reply_to=self._prompts()[-1] if reply else None)
                    del msg["message"]["text"]
                    msg["message"][kind] = {"file_id": "fake"} if kind == "voice" else [{"file_id": "fake"}]
                    msg["message"]["caption"] = "Can you identify this?"
                    before = len(self._saved())
                    inbox.handle_update(msg)
                    # No transcriber here: a voice answer is asked for in writing, like a photo.
                    asked = t("tag_voice_failed", "en") if kind == "voice" else "Please write the tag as text."
                    self.assertEqual(self.tg.texts()[-1], asked)
                    self.assertEqual(len(self._saved()), before)
                    inbox = self._inbox(agent)
                    inbox.handle_update(text(3, "gardener"))
                    self.assertEqual(self._saved()[-1]["owner_text"], "gardener")
                    self.assertEqual(agent.seen, [])

    def test_unrelated_media_keeps_existing_behaviour(self) -> None:
        inbox = self._inbox(FakeAgentV2())
        inbox.handle_update(tap(1, "tag:other", 77))
        for uid, sender, reply_to in ((2, OMER, None), (3, DANA, 9999), (4, OMER, self._prompts()[-1])):
            msg = text(uid, "", sender=sender, reply_to=reply_to)
            del msg["message"]["text"]
            msg["message"]["voice"] = {"file_id": "fake"}
            before = len(self.tg.calls)
            inbox.handle_update(msg)
            self.assertEqual(len(self.tg.calls), before)
        inbox.handle_update(text(5, "gardener"))
        self.assertEqual(self._saved()[0]["owner_text"], "gardener")

    def test_probe_original_defects_restart_two_users(self) -> None:
        self._remember_b()
        agent = FakeAgentV2()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        inbox.handle_update(text(2, "question about B", reply_to=88))
        self.assertEqual(agent.seen, ["question about B"])
        inbox.handle_update(tap(3, "tag:other", 88))
        pa, pb = self._prompts()
        inbox = self._inbox(agent)
        inbox.handle_update(text(4, "A label", reply_to=pa))
        inbox.handle_update(text(5, "B label", reply_to=pb))
        self.assertEqual({r["alert"]["alert_id"]: r["owner_text"] for r in self._saved()},
                         {ALERT_ID: "A label", self.B_ID: "B label"})
        inbox.handle_update(tap(6, "tag:other", 77))
        expired = self._prompts()[-1]
        self.clock += 86401
        inbox = self._inbox(agent)
        inbox.handle_update(tap(7, "tag:other", 88, sender=OMER))
        inbox.handle_update(text(8, "Omer question", sender=OMER, reply_to=expired))
        self.assertEqual(agent.seen, ["question about B", "Omer question"])
        inbox = self._inbox(agent)
        inbox.handle_update(text(9, "Dana late tag", reply_to=expired))
        self.assertEqual(self.tg.texts()[-1], t("tag_expired", "en"))
        inbox.handle_update(text(10, "Omer B label", sender=OMER))
        self.assertEqual(self._saved()[-1]["owner_text"], "Omer B label")

    def test_probe_commands_cancel_all_waits_only_for_sender(self) -> None:
        self._remember_b()
        agent = FakeAgentV2()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        inbox.handle_update(tap(2, "tag:other", 88))
        inbox.handle_update(tap(3, "tag:other", 77, sender=OMER))
        inbox.handle_update(text(4, " /status", reply_to=self._prompts()[0]))
        inbox = self._inbox(agent)
        inbox.handle_update(text(5, "hello"))
        inbox.handle_update(text(6, "Omer label", sender=OMER))
        self.assertEqual(agent.seen, ["/status", "hello"])
        self.assertEqual([r["owner_text"] for r in self._saved()], ["Omer label"])

    def test_new_tag_messages_are_translated(self) -> None:
        for key, english in (("tag_undo_partial", "Part of the undo failed; tap Undo again."),
                             ("tag_need_text", "Please write the tag as text.")):
            self.assertEqual(t(key, "en"), english)
            for lang in ("he", "ar"):
                self.assertTrue(t(key, lang))
                self.assertNotEqual(t(key, lang), english)

    def test_a_failed_tag_save_keeps_only_explicit_retry_open(self) -> None:
        agent = FakeAgentV1()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        prompt = self._prompts()[-1]
        with mock.patch("home_guard_project.box.telegram_agent.save_feedback", side_effect=OSError("disk full")):
            inbox.handle_update(text(2, "gardener"))
        self.assertEqual(self._saved(), [])
        self.assertNotIn("Saved as", self.tg.texts()[-1])
        inbox = self._inbox(agent)
        inbox.handle_update(text(3, "why did that fail?"))
        self.assertEqual(agent.seen, ["why did that fail?"])
        inbox.handle_update(text(4, "gardener", reply_to=prompt))
        self.assertEqual(self._saved()[0]["owner_text"], "gardener")

    def test_legacy_wait_completion_survives_a_failed_pending_write(self) -> None:
        inbox = self._inbox()
        inbox.handle_update(tap(1, "tag:other", 77))
        path = inbox.pending.path
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        for entry in state["pending"].values():
            entry.pop("request_id", None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)
        agent = FakeAgentV2()
        inbox = self._inbox(agent)
        with mock.patch("home_guard_project.box.telegram_agent._write_json", side_effect=OSError("disk full")):
            inbox.handle_update(text(2, "gardener"))
        self._inbox(agent).handle_update(text(3, "is the door locked?"))
        self.assertEqual(agent.seen, ["is the door locked?"])
        self.assertEqual([s["owner_text"] for s in self._saved()], ["gardener"])


class UndoTrainingTagTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.training = tmp.name
        self.meta_dir = os.path.join(self.training, "meta", "door", "2027-01-15")
        self.clip_dir = os.path.join(self.training, "clips", "door", "2027-01-15")
        os.makedirs(self.meta_dir)
        os.makedirs(self.clip_dir)
        self.meta_path = os.path.join(self.meta_dir, f"{ALERT_ID}.meta.json")
        self.clip_path = os.path.join(self.clip_dir, f"{ALERT_ID}.mp4")
        with open(self.clip_path, "wb") as f:
            f.write(b"mp4")

    def _meta(self, meta: Any) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as f:
            f.write(meta if isinstance(meta, str) else json.dumps(meta))

    def _tagged(self) -> None:
        self._meta({"kind": "owner_feedback", "clip_path": f"clips/door/2027-01-15/{ALERT_ID}.mp4",
                    "owner_feedback": [{"owner_label": "suspicious"}]})

    def test_a_corrupt_meta_does_not_raise(self) -> None:
        self._meta("{not json")
        self.assertEqual(undo_training_tag(ALERT, "", {"name": "Dana"}, NOW, self.training), "failed")

    def test_a_clip_that_cannot_be_removed_still_drops_the_meta(self) -> None:
        self._tagged()
        real_remove = os.remove

        def remove(path: str) -> None:
            if path.endswith(".mp4"):
                raise PermissionError("in use")
            real_remove(path)

        with mock.patch("home_guard_project.box.feedback.os.remove", side_effect=remove):
            self.assertEqual(undo_training_tag(ALERT, "", {"name": "Dana"}, NOW, self.training), "failed")
        self.assertFalse(os.path.exists(self.meta_path))
        self.assertEqual(undo_training_tag(ALERT, "", {"name": "Dana"}, NOW, self.training), "removed")
        self.assertFalse(os.path.exists(self.clip_path))
        self.assertEqual(undo_training_tag(ALERT, "", {"name": "Dana"}, NOW, self.training), "none")

    def test_a_meta_that_cannot_be_removed_still_drops_the_clip(self) -> None:
        self._tagged()
        real_remove = os.remove

        def remove(path: str) -> None:
            if path.endswith(".meta.json"):
                raise PermissionError("in use")
            real_remove(path)

        with mock.patch("home_guard_project.box.feedback.os.remove", side_effect=remove):
            self.assertEqual(undo_training_tag(ALERT, "", {"name": "Dana"}, NOW, self.training), "failed")
        self.assertFalse(os.path.exists(self.clip_path))
        self.assertEqual(undo_training_tag(ALERT, "", {"name": "Dana"}, NOW, self.training), "removed")
        self.assertFalse(os.path.exists(self.meta_path))

    def test_repeated_undo_of_an_existing_copy_adds_only_one_note(self) -> None:
        self._meta({"kind": "alert", "clip_path": f"clips/door/2027-01-15/{ALERT_ID}.mp4", "owner_feedback": []})
        for now in (NOW, NOW + 1):
            self.assertEqual(undo_training_tag(ALERT, "", {"name": "Dana"}, now, self.training), "noted")
        with open(self.meta_path, encoding="utf-8") as f:
            self.assertEqual(len(json.load(f)["owner_feedback"]), 1)

    def test_nothing_removable_is_failed(self) -> None:
        self._tagged()
        with mock.patch("home_guard_project.box.feedback.os.remove", side_effect=PermissionError("in use")):
            self.assertEqual(undo_training_tag(ALERT, "", {"name": "Dana"}, NOW, self.training), "failed")

    def test_a_note_that_cannot_be_written_is_failed(self) -> None:
        self._meta({"kind": "alert", "clip_path": f"clips/door/2027-01-15/{ALERT_ID}.mp4", "owner_feedback": []})
        with mock.patch("home_guard_project.box.feedback._write_json", side_effect=OSError("disk full")):
            self.assertEqual(undo_training_tag(ALERT, "", {"name": "Dana"}, NOW, self.training), "failed")


if __name__ == "__main__":
    unittest.main()
