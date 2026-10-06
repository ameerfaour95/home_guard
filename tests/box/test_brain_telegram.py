# tests/box/test_brain_telegram.py
from __future__ import annotations

import json
import os
import tempfile
import unittest

from home_guard_project.box.brain.agent import AgentReply
from home_guard_project.box.feedback import AlertIndex, MuteState
from home_guard_project.box.telegram_agent import TelegramInbox
from home_guard_project.box.telegram_notify import TelegramConfig

NOW = 1_790_000_000.0
CFG = TelegramConfig(bot_token="t", chat_ids=["-5"], dry_run=False)


class FakeAgent:
    version = 2

    def __init__(self):
        self.calls = []
        self.order = []

    def handle(self, text, chat_id, who=None, alert=None, threaded=False):
        self.calls.append(("handle", text, (alert or {}).get("alert_id"), threaded))
        return AgentReply(text="answer", undo_token="123", after=(lambda: self.order.append("after"),))

    def handle_choice(self, chat_id, token, index, who=None):
        self.calls.append(("choice", token, index))
        return AgentReply(text="picked")

    def undo_turn(self, chat_id, token, who=None):
        self.calls.append(("undo", token))
        return AgentReply(text="✓ Alerts are back on.")

    def answer_proposal(self, chat_id, proposal_id, approve, who=None):
        self.calls.append(("proposal", proposal_id, approve))
        return AgentReply(text="✓ Approved: set the house to home, awake")


class FakeDeliverer:
    def __init__(self):
        self.typing_to = []

    def typing(self, chat_id):
        self.typing_to.append(chat_id)


class InboxV2Test(unittest.TestCase):
    def setUp(self) -> None:
        d = tempfile.mkdtemp()
        self.index = AlertIndex(os.path.join(d, "index.json"))
        self.posts = []
        self.agent = FakeAgent()
        self.deliverer = FakeDeliverer()

        def post(token, method, fields, timeout=15.0):
            self.posts.append((method, fields))
            self.agent.order.append(method)
            return {"ok": True, "result": {"message_id": 99}}

        self.inbox = TelegramInbox(CFG, self.agent, self.index, MuteState(os.path.join(d, "m.json")), d,
                                   os.path.join(d, "offset.json"), post=post, post_multipart=post,
                                   now=lambda: NOW, deliverer=self.deliverer)

    def message(self, text, reply_to=None, update_id=1):
        msg = {"chat": {"id": -5}, "from": {"id": 7, "first_name": "A"}, "text": text, "message_id": 10}
        if reply_to is not None:
            msg["reply_to_message"] = {"message_id": reply_to}
        return {"update_id": update_id, "message": msg}

    def test_a_reply_to_an_alert_is_threaded(self) -> None:
        self.index.remember("-5", 50, {"alert_id": "a1", "ts": NOW - 4000})
        self.inbox.handle_update(self.message("nothing there", reply_to=50))
        self.assertEqual(self.agent.calls[0], ("handle", "nothing there", "a1", True))
        self.assertEqual(self.deliverer.typing_to, ["-5"])

    def test_an_unthreaded_message_binds_only_one_recent_alert(self) -> None:
        self.index.remember("-5", 50, {"alert_id": "a1", "ts": NOW - 60})
        self.inbox.handle_update(self.message("who is it", update_id=1))
        self.index.remember("-5", 51, {"alert_id": "a2", "ts": NOW - 30})
        self.inbox.handle_update(self.message("and now", update_id=2))
        self.assertEqual([c[2] for c in self.agent.calls], ["a1", None])

    def test_the_undo_button_and_after_reply_actions(self) -> None:
        self.inbox.handle_update(self.message("pause"))
        method, fields = [p for p in self.posts if p[0] == "sendMessage"][0]
        self.assertEqual(json.loads(fields["reply_markup"])["inline_keyboard"][0][0]["callback_data"], "u:123")
        self.assertLess(self.agent.order.index("sendMessage"), self.agent.order.index("after"))

    def test_clarification_and_undo_taps(self) -> None:
        tap = lambda data, uid: {"update_id": uid, "callback_query": {  # noqa: E731
            "id": "q", "data": data, "from": {"id": 7}, "message": {"chat": {"id": -5}, "message_id": 99}}}
        self.inbox.handle_update(tap("cl:ab12:1", 5))
        self.inbox.handle_update(tap("u:123", 6))
        self.assertEqual(self.agent.calls, [("choice", "ab12", 1), ("undo", "123")])

    def test_house_proposal_buttons_go_out_and_their_taps_come_back(self) -> None:
        rows = ((("✓ Yes", "hs:P1:y"), ("✗ No", "hs:P1:n")),)
        self.agent.handle = lambda *a, **k: AgentReply(text="🏠 House: home, asleep", rows=rows)
        self.inbox.handle_update(self.message("status"))
        markup = json.loads(next(f for m, f in self.posts if m == "sendMessage")["reply_markup"])
        self.assertEqual(markup["inline_keyboard"], [[{"text": "✓ Yes", "callback_data": "hs:P1:y"},
                                                      {"text": "✗ No", "callback_data": "hs:P1:n"}]])
        tap = lambda data, uid: {"update_id": uid, "callback_query": {  # noqa: E731
            "id": "q", "data": data, "from": {"id": 7}, "message": {"chat": {"id": -5}, "message_id": 99}}}
        self.inbox.handle_update(tap("hs:P1:y", 7))
        self.inbox.handle_update(tap("hs:P1:n", 8))
        self.inbox.handle_update(tap("hs:P1:maybe", 9))                 # malformed: ignored
        self.assertEqual([c for c in self.agent.calls if c[0] == "proposal"],
                         [("proposal", "P1", True), ("proposal", "P1", False)])

    def test_a_repeated_update_is_ignored(self) -> None:
        self.inbox.handle_update(self.message("hi", update_id=9))
        self.inbox.handle_update(self.message("hi", update_id=9))
        self.assertEqual(len(self.agent.calls), 1)



class WiringTest(InboxV2Test):
    def tap(self, data, uid):
        return {"update_id": uid, "callback_query": {
            "id": "q", "data": data, "from": {"id": 7},
            "message": {"chat": {"id": -5}, "message_id": 99}}}

    def test_raising_agent_does_not_stop_next_update_in_poll(self):
        from unittest.mock import Mock
        for method, update in (("handle", self.message("hi")),
                               ("handle_choice", self.tap("cl:abc:0", 1)),
                               ("undo_turn", self.tap("u:123", 1))):
            with self.subTest(method=method):
                self.inbox._seen.clear()
                setattr(self.agent, method, Mock(side_effect=ValueError("broken agent")))
                self.agent.handle = Mock(return_value=AgentReply(text="next")) if method != "handle" else Mock(
                    side_effect=[ValueError("broken agent"), AgentReply(text="next")])
                self.inbox._post = Mock(return_value={"ok": True, "result": [update, self.message("next", update_id=2)]})
                with self.assertLogs("box.telegram_agent", level="WARNING"):
                    self.inbox.poll_once()
                self.assertEqual(self.agent.handle.call_args.args[0], "next")

    def test_malformed_polled_update_does_not_block_the_next(self):
        from unittest.mock import Mock
        self.inbox._post = Mock(return_value={"ok": True, "result": [
            None, {"update_id": "nan"}, {"update_id": float("inf")},
            self.message("next", update_id=9)]})
        self.inbox.poll_once()
        self.assertEqual(self.agent.calls[-1][1], "next")

    def test_bad_offset_files_use_zero(self):
        for raw in (b"[]", b'{"offset": Infinity}', b"\xff"):
            with open(self.inbox.offset_path, "wb") as f:
                f.write(raw)
            self.assertEqual(self.inbox._load_offset(), 0)

    def test_malformed_updates_do_not_escape(self):
        for update in (None, 3, [], {"update_id": [], "message": []},
                       self.tap("cl:abc:nan", 21), self.tap("cl:missing", 22)):
            self.inbox.handle_update(update)
        self.inbox.handle_update(self.message("next", update_id=30))
        self.assertEqual(self.agent.calls[-1][1], "next")

    def test_stale_choice_sends_nothing(self):
        self.agent.handle_choice = lambda *a: None
        self.inbox.handle_update(self.tap("cl:old:0", 20))
        self.assertEqual([m for m, f in self.posts], ["answerCallbackQuery"])

    def test_choices_keyboard_and_independent_after_actions(self):
        def bad():
            raise ValueError("bad action")
        self.inbox._send_v2("-5", AgentReply(text="Which?", buttons=("front", "back"),
                            question_token="abc", undo_token="123",
                            after=(bad, lambda: self.agent.order.append("after"))), 10)
        fields = self.posts[-1][1]
        self.assertEqual(json.loads(fields["reply_markup"])["inline_keyboard"][0][0]["callback_data"], "cl:abc:0")
        self.assertEqual(self.agent.order[-1], "after")

    def test_a_question_with_an_undo_shows_both(self):
        self.inbox._send_v2("-5", AgentReply(text="Which?", buttons=("front", "back"), question_token="abc",
                                             undo_token="123"), 10)
        rows = json.loads(self.posts[-1][1]["reply_markup"])["inline_keyboard"]
        self.assertEqual([[b["callback_data"] for b in row] for row in rows], [["cl:abc:0"], ["cl:abc:1"], ["u:123"]])
        self.assertEqual(rows[-1][0]["text"], "↩ Undo")

    def test_a_failed_spinner_stop_does_not_lose_the_tap(self):
        from unittest.mock import Mock
        sent = []

        def post(token, method, fields, timeout=15.0):
            if method == "answerCallbackQuery":
                raise OSError("connection reset")
            sent.append(method)
            return {"ok": True, "result": {"message_id": 99}}
        self.inbox._post = post
        with self.assertLogs("box.telegram_agent", level="WARNING"):
            self.inbox.handle_update(self.tap("u:123", 40))
            self.inbox.handle_update(self.tap("cl:ab12:1", 41))
        self.assertEqual(self.agent.calls, [("undo", "123"), ("choice", "ab12", 1)])
        self.assertEqual(sent, ["sendMessage", "sendMessage"])

    def test_v2_taps_are_noted_in_the_chat_window(self):
        notes = []

        class Feed:
            def add(self, who, kind, text, name="", camera="", alert_id="", now=0.0):
                notes.append((who, kind, text, name))
        self.inbox.feed = Feed()
        tap = self.tap("cl:ab12:1", 50)
        tap["callback_query"]["from"]["first_name"] = "Dana"
        tap["callback_query"]["message"]["reply_markup"] = {"inline_keyboard": [
            [{"text": "front", "callback_data": "cl:ab12:0"}], [{"text": "back", "callback_data": "cl:ab12:1"}]]}
        self.inbox.handle_update(tap)
        undo = self.tap("u:123", 51)
        undo["callback_query"]["from"]["first_name"] = "Dana"
        self.inbox.handle_update(undo)
        owner = [n for n in notes if n[0] == "owner"]
        self.assertEqual(owner, [("owner", "button", "back", "Dana"), ("owner", "button", "↩ Undo", "Dana")])

    def test_after_actions_run_even_when_send_fails(self):
        from unittest.mock import patch
        with patch.object(self.inbox, "_say", side_effect=ValueError("send failed")):
            self.inbox.handle_update(self.message("change"))
        self.assertIn("after", self.agent.order)

class StartupTest(unittest.TestCase):
    def test_version_switch_and_follow_up_only_at_start(self):
        from unittest.mock import Mock, patch
        from home_guard_project.box import telegram_agent as module
        with tempfile.TemporaryDirectory() as root:
            for version in (2, "2", 1, "nonsense", [], float("inf")):
                with (
                    self.subTest(version=version),
                    patch.object(module.telegram_notify, "load_telegram_config", return_value=CFG),
                    patch.object(module.threading, "Thread") as thread,
                    patch.object(module, "make_chat_model", return_value=None) as v1,
                    patch("home_guard_project.box.brain.agent.build_owner_agent", return_value=(Mock(), Mock())) as v2,
                ):
                    out = module.start({"agent_version": version}, {}, [], log_dir=root, live_dir=root, archive_dir=root)
                    if version in (2, "2"):
                        v2.assert_called_once()
                        v1.assert_not_called()
                        self.assertEqual([c.kwargs["name"] for c in thread.call_args_list],
                                         ["camera-follow-up", "telegram-inbox"])
                        self.assertIsNotNone(out.deliverer)
                    else:
                        v2.assert_not_called()
                        v1.assert_called_once()

    def test_without_any_model_unavailable_still_works(self):
        from unittest.mock import patch
        from home_guard_project.box import telegram_agent as module
        with (
            tempfile.TemporaryDirectory() as root,
            patch.object(module.telegram_notify, "load_telegram_config", return_value=CFG),
            patch.object(module.threading, "Thread"),
            patch("home_guard_project.box.brain.agent.build_owner_agent", return_value=(None, FakeDeliverer())),
        ):
            out = module.start({"agent_version": 2}, {}, [], log_dir=root, live_dir=root, archive_dir=root)
            self.assertIsNone(out.inbox.agent)

    def test_announcement_never_raises_and_uses_thread(self):
        from unittest.mock import Mock, patch
        from home_guard_project.box.telegram_agent import OwnerAssistant
        assistant = OwnerAssistant(CFG, None, None, None)
        with patch("home_guard_project.box.telegram_agent.threading.Thread") as thread:
            assistant.announce("mode")
            thread.assert_called_once()
            with patch("home_guard_project.box.telegram_agent.telegram_notify._http_post", side_effect=ValueError("bad response")):
                thread.call_args.kwargs["target"]()
        with patch("home_guard_project.box.telegram_agent.threading.Thread", side_effect=RuntimeError("no thread")):
            assistant.announce("mode")
        assistant.cfg = []
        assistant.announce("mode")

    def test_a_delivered_alert_goes_into_each_chats_history(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from home_guard_project.box.telegram_agent import OwnerAssistant

        noted = []
        agent = SimpleNamespace(version=2, note_alert=lambda chat_id, alert: noted.append((chat_id, alert["alert_id"])))
        assistant = OwnerAssistant(CFG, None, None, SimpleNamespace(agent=agent))
        sent = {"sent": True, "results": [{"chat_id": "-5", "ok": True}, {"chat_id": "-6", "ok": False}]}

        class Now:                               # runs the background note at once
            def __init__(self, target, **kw):
                self.target = target

            def start(self):
                self.target()

        with patch("home_guard_project.box.telegram_agent.send_alert", return_value=sent), \
                patch("home_guard_project.box.telegram_agent.threading.Thread", Now):
            self.assertEqual(assistant.send_alert({"alert_id": "a1", "camera": "gate"}, "text"), sent)
            self.assertEqual(noted, [("-5", "a1")])                   # only where it arrived
            agent.note_alert = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
            self.assertEqual(assistant.send_alert({"alert_id": "a2"}, "text"), sent)   # never breaks an alert
            assistant.inbox = SimpleNamespace(agent=SimpleNamespace(version=1))
            self.assertEqual(assistant.send_alert({"alert_id": "a3"}, "text"), sent)

    def test_no_cameras_hands_over_to_the_assistant_only_wait(self):
        # serve_without_cameras itself (assistant started even if it raises, waits) is
        # covered in test_inference.py; here run() must hand over to it, not exit.
        from types import SimpleNamespace
        from unittest.mock import patch
        from home_guard_project.box import inference
        with (
            patch("home_guard_project.box.boxconfig.load_box_settings", return_value={}),
            patch("home_guard_project.data_collection.config.load_config", return_value=SimpleNamespace(CAMERAS={})),
            patch.object(inference, "make_backend"),
            patch.object(inference, "load_detector", return_value=(None, None)),
            patch("home_guard_project.box.alert_settings.camera_names", return_value=["yard"]),
            patch.object(inference, "serve_without_cameras", return_value=0) as wait,
        ):
            self.assertEqual(inference.run(), 0)
        wait.assert_called_once()
        self.assertEqual(list(wait.call_args.args[2]), ["yard"])


if __name__ == "__main__":
    unittest.main()
