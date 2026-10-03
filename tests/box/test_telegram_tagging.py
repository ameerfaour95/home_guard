"""Task 18b: tagging an alert from Telegram - label buttons, and "Other…" that waits for the owner's words."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

import numpy as np

from home_guard_project.box.agent import AgentReply
from home_guard_project.box.alert_clips import encode_frame, write_alert_clip
from home_guard_project.box.brain.i18n import t
from home_guard_project.box.feedback import OWNER_LABELS, AlertIndex, Feedback, MuteState, verdict_for
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
        self.assertEqual(OWNER_LABELS, ("normal", "suspicious", "escalation", "empty", "other"))
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

    def test_two_rows_six_buttons_with_the_tag_codes(self) -> None:
        rows = self._rows(feedback_keyboard("en"))
        self.assertEqual([[b["callback_data"] for b in row] for row in rows], [
            ["tag:normal", "tag:suspicious", "tag:escalation"],
            ["tag:empty", "tag:other", "fb:mute60"],
        ])
        self.assertEqual([[b["text"] for b in row] for row in rows], [
            ["🟢 Normal", "🟡 Suspicious", "🔴 Escalation"],
            ["⚪ Nothing there", "✏️ Other…", "⏸ Pause 1 hour"],
        ])

    def test_the_ai_label_gets_a_tick(self) -> None:
        texts = [b["text"] for row in self._rows(feedback_keyboard("en", ai_label="suspicious")) for b in row]
        self.assertEqual(texts[1], "✓ 🟡 Suspicious")
        self.assertEqual([x for x in texts if x.startswith("✓")], ["✓ 🟡 Suspicious"])
        unticked = [b["text"] for row in self._rows(feedback_keyboard("en", ai_label="bogus")) for b in row]
        self.assertFalse(any(x.startswith("✓") for x in unticked))

    def test_hebrew_texts(self) -> None:
        texts = [b["text"] for row in self._rows(feedback_keyboard("he", ai_label="escalation")) for b in row]
        self.assertEqual(texts, [
            f"🟢 {t('btn_tag_normal', 'he')}", f"🟡 {t('btn_tag_suspicious', 'he')}",
            f"✓ 🔴 {t('btn_tag_escalation', 'he')}",
            f"⚪ {t('btn_tag_empty', 'he')}", f"✏️ {t('btn_tag_other', 'he')}", f"⏸ {t('btn_mute60', 'he')}",
        ])
        self.assertNotEqual(t("btn_tag_normal", "he"), t("btn_tag_normal", "en"))

    def test_the_english_strings(self) -> None:
        self.assertEqual([t(k, "en") for k in ("btn_tag_normal", "btn_tag_suspicious", "btn_tag_escalation",
                                               "btn_tag_empty", "btn_tag_other")],
                         ["Normal", "Suspicious", "Escalation", "Nothing there", "Other…"])
        self.assertEqual(t("tag_ask_text", "en"), "Write the correct tag for this clip.")
        self.assertEqual(t("tag_saved", "en", label="Normal"), "✓ Saved as Normal.")
        self.assertEqual(t("tag_saved_text", "en", text="a cat"), "✓ Saved as: \"a cat\"")
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

    def test_a_damaged_file_is_an_empty_store(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{not json")
        pending = PendingTags(self.path)
        self.assertIsNone(pending.take(CHAT, 42, NOW))
        pending.add(CHAT, 42, ALERT_ID, NOW)
        self.assertEqual(pending.take(CHAT, 42, NOW + 1), ALERT_ID)


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
        (said,) = self.tg.sent("sendMessage")
        self.assertEqual(said["fields"]["text"], "✓ Saved as Suspicious.")
        self.assertEqual(callback_codes(said), [f"tu:{ALERT_ID}"])
        self.assertIn(ALERT_ID, inbox.answered)                   # the escalation reminder stops

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
        (said,) = self.tg.sent("sendMessage")
        self.assertEqual(said["fields"]["text"], t("tag_saved", "he", label=t("btn_tag_normal", "he")))
        self.assertEqual(markup(said)["inline_keyboard"][0][0]["text"], t("undo_button", "he"))

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
        inbox.handle_update(text(2, "  two kids on bikes  ", reply_to=901))
        (saved,) = self._saved()
        self.assertEqual((saved["owner_label"], saved["owner_text"], saved["verdict"]),
                         ("other", "two kids on bikes", "real_but_wrong"))
        self.assertEqual(saved["tagged_by"], "Dana")
        self.assertEqual(self._training_meta()["owner_feedback"][-1]["owner_text"], "two kids on bikes")
        said = self.tg.sent("sendMessage")[-1]
        self.assertEqual(said["fields"]["text"], "✓ Saved as: \"two kids on bikes\"")
        self.assertEqual(callback_codes(said), [f"tu:{ALERT_ID}"])
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

    def test_a_pending_tag_older_than_ten_minutes_is_not_consumed(self) -> None:
        agent = FakeAgentV1()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        self.clock += 601
        inbox.handle_update(text(2, "what happened today?"))
        self.assertEqual(agent.seen, ["what happened today?"])
        self.assertEqual(self._saved(), [])
        self.assertNotIn(t("tag_expired", "en"), self.tg.texts())

    def test_a_reply_to_an_expired_prompt_says_it_expired(self) -> None:
        agent = FakeAgentV1()
        inbox = self._inbox(agent)
        inbox.handle_update(tap(1, "tag:other", 77))
        prompt_id = self.tg.sent("sendMessage")[0]["message_id"]    # the "Other…" question
        self.clock += 601
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


if __name__ == "__main__":
    unittest.main()
