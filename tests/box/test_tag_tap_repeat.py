"""2026-10-09 12:48: eleven taps on "✏️ אחר…" (and one "🟢 תקין") on one alert each asked "מה קורה בסרטון?" again.

Why: a 40 s "מה קורה" turn held the inbox; the taps waited in Telegram's queue with their spinners running, so the
owner tapped again and again; when they were handled at last Telegram refused the late answers (HTTP 400), and the
🟢's refusal even stopped its receipt. Now a tap is answered at once (and never raises), a repeat of the same button
on the same alert within a minute asks nothing, and taps waiting behind a long update are answered from a side look
at the queue.
"""

from __future__ import annotations

import time
import unittest
import urllib.error
from typing import Any, Dict, List

import test_telegram_tagging as tt
from test_telegram_tagging import CHAT, tap, text

from home_guard_project.box.brain.i18n import t
from home_guard_project.box.telegram_agent import TAP_REPEAT_SEC

OTHER, NORMAL = "tag:other", "tag:normal"
# The owner's taps 12:48:28-12:48:40 (seconds after the first), all on the same alert message.
SEQUENCE = [(0.0, OTHER), (1.0, OTHER), (2.0, OTHER), (3.4, OTHER), (4.5, OTHER), (5.5, NORMAL),
            (8.4, OTHER), (9.3, OTHER), (10.1, OTHER), (11.0, OTHER), (11.7, OTHER), (12.8, OTHER)]


class _Inbox(unittest.TestCase):
    """The tagging tests' inbox and stores (test_telegram_tagging.InboxTaggingTest), without its tests."""
    setUp = tt.InboxTaggingTest.setUp
    _inbox = tt.InboxTaggingTest._inbox
    _saved = tt.InboxTaggingTest._saved
    _production_clip = tt.InboxTaggingTest._production_clip


class TapRepeatTest(_Inbox):
    def _asks(self) -> List[str]:
        return [x for x in self.tg.texts() if t("tag_ask_text", self.lang) in x]

    def test_the_exact_tap_sequence_asks_once_and_saves_the_normal_once(self) -> None:
        self.lang = "he"
        inbox = self._inbox()
        start = self.clock
        for i, (dt_s, code) in enumerate(SEQUENCE, start=1):
            self.clock = start + dt_s
            inbox.handle_update(tap(i, code, 77))
        self.assertEqual(len(self._asks()), 1)                       # one question, not eleven
        answers = self.tg.sent("answerCallbackQuery")
        self.assertEqual(len(answers), len(SEQUENCE))                # every tap's spinner stopped
        toasts = [a["fields"].get("text") for a in answers]
        self.assertEqual(toasts.count(t("tag_already_waiting", "he")), 10)
        saved = self._saved()
        self.assertEqual([s["owner_label"] for s in saved], ["normal"])
        self.assertEqual(len(self.tg.sent("editMessageReplyMarkup")), 1)     # the 🟢 receipt is shown

    def test_the_wait_survives_the_normal_tap_and_takes_the_words(self) -> None:
        inbox = self._inbox()
        inbox.handle_update(tap(1, OTHER, 77))
        inbox.handle_update(tap(2, NORMAL, 77))
        self.clock += 5
        inbox.handle_update(tap(3, OTHER, 77))                       # still waiting: no second question
        self.assertEqual(len(self._asks()), 1)
        inbox.handle_update(text(4, "the gardener"))
        self.assertEqual([s["owner_label"] for s in self._saved()], ["normal", "other"])
        self.assertEqual(self._saved()[-1]["owner_text"], "the gardener")

    def test_after_a_minute_another_tap_asks_again(self) -> None:
        inbox = self._inbox()
        inbox.handle_update(tap(1, OTHER, 77))
        self.clock += TAP_REPEAT_SEC + 1
        inbox.handle_update(tap(2, OTHER, 77))
        self.assertEqual(len(self._asks()), 2)

    def test_after_the_words_were_saved_another_tap_asks_again(self) -> None:
        inbox = self._inbox()
        inbox.handle_update(tap(1, OTHER, 77))
        inbox.handle_update(text(2, "my brother"))
        inbox.handle_update(tap(3, OTHER, 77))
        self.assertEqual(len(self._asks()), 2)

    def test_a_different_label_within_the_minute_is_saved(self) -> None:
        inbox = self._inbox()
        inbox.handle_update(tap(1, NORMAL, 77))
        inbox.handle_update(tap(2, NORMAL, 77))
        inbox.handle_update(tap(3, "tag:suspicious", 77))
        inbox.handle_update(tap(4, NORMAL, 77))
        self.assertEqual([s["owner_label"] for s in self._saved()], ["normal", "suspicious", "normal"])

    def test_a_late_answer_refused_by_telegram_still_shows_the_receipt(self) -> None:
        real_post = self.tg.post

        def post(token: str, method: str, fields: Dict[str, str], timeout: float = 15.0) -> Dict[str, Any]:
            if method == "answerCallbackQuery":
                raise urllib.error.HTTPError("u", 400, "Bad Request: query is too old", None, None)
            return real_post(token, method, fields, timeout)

        inbox = self._inbox()
        inbox._post = post
        inbox.handle_update(tap(1, NORMAL, 77))
        self.assertEqual([s["owner_label"] for s in self._saved()], ["normal"])
        self.assertEqual(len(self.tg.sent("editMessageReplyMarkup")), 1)

    def test_a_callback_is_answered_once(self) -> None:
        inbox = self._inbox()
        query = tap(1, NORMAL, 77)["callback_query"]
        self.assertTrue(inbox._answer_callback(query))
        self.assertFalse(inbox._answer_callback(query, "again"))
        self.assertEqual(len(self.tg.sent("answerCallbackQuery")), 1)


class BusyWatcherTest(_Inbox):
    def test_taps_waiting_behind_a_long_turn_get_their_spinner_stopped(self) -> None:
        waiting = [tap(10, OTHER, 77), tap(11, OTHER, 77)]
        polls: List[Dict[str, str]] = []
        real_post = self.tg.post

        def post(token: str, method: str, fields: Dict[str, str], timeout: float = 15.0) -> Dict[str, Any]:
            if method == "getUpdates":
                polls.append(dict(fields))
                if fields.get("timeout") == "0":              # the side look: the taps are still queued
                    return {"ok": True, "result": waiting}
                return {"ok": True, "result": [{"update_id": 9, "message": {
                    "message_id": 600, "chat": {"id": int(CHAT)}, "text": "מה קורה",
                    "from": {"id": 42, "first_name": "Dana", "is_bot": False}}}]}
            return real_post(token, method, fields, timeout)

        class SlowAgent:
            version = 1

            def handle(self, text: str, chat_id: Any, who: Any = None, alert: Any = None) -> Any:
                time.sleep(0.5)                                   # the 40 s look_around, shortened

                class Reply:
                    text = "answer"
                    clips = ()

                return Reply()

        inbox = self._inbox(SlowAgent())
        inbox._post = post
        inbox.busy_ack_sec = 0.1
        self.assertEqual(inbox.poll_once(timeout=1), 1)
        acked = [c for c in self.tg.sent("answerCallbackQuery")]
        self.assertEqual(sorted(c["fields"]["callback_query_id"] for c in acked), ["cb10", "cb11"])
        self.assertTrue(all(c["fields"].get("text") == t("busy_ack", self.lang) for c in acked))
        side = [p for p in polls if p.get("timeout") == "0"]
        self.assertTrue(side)
        self.assertTrue(all(p["offset"] == "10" for p in side))            # nothing taken off the queue
        self.assertTrue(all("message" in p["allowed_updates"] for p in side))   # the sticky list is not narrowed
        # Handled next as always: one question, and no second answer to the already answered taps.
        for update in waiting:
            inbox.handle_update(update)
        self.assertEqual(len([x for x in self.tg.texts() if t("tag_ask_text", self.lang) in x]), 1)
        self.assertEqual(len(self.tg.sent("answerCallbackQuery")), 2)

    def test_a_quick_update_starts_no_side_look(self) -> None:
        polls: List[str] = []
        real_post = self.tg.post

        def post(token: str, method: str, fields: Dict[str, str], timeout: float = 15.0) -> Dict[str, Any]:
            if method == "getUpdates":
                polls.append(fields.get("timeout", ""))
                return {"ok": True, "result": [tap(5, NORMAL, 77)]} if len(polls) == 1 else {"ok": True, "result": []}
            return real_post(token, method, fields, timeout)

        inbox = self._inbox()
        inbox._post = post
        inbox.poll_once(timeout=1)
        self.assertEqual(polls, ["1"])


if __name__ == "__main__":
    unittest.main()
