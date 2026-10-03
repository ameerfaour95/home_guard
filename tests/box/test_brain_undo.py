# tests/box/test_brain_undo.py
from __future__ import annotations

import datetime as dt
import os
import tempfile
import unittest

from test_brain_agent import FakeRegistry, Scripted, call, reply

from home_guard_project.box.brain.agent import OwnerAgentV2
from home_guard_project.box.brain.memory import ChatMemory
from home_guard_project.box.brain.receipts import ReceiptBook
from home_guard_project.box.brain.tools import Services
from home_guard_project.box.feedback import MuteState

NOW = dt.datetime(2026, 10, 3, 23, 0).timestamp()


class UndoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.mute = MuteState(os.path.join(self.root, "mute.json"))
        self.store = {"alert_start_hour": 22, "alert_end_hour": 6, "owner_language": "en"}
        self.cameras, self.restarts = [], []

    def agent(self, big) -> OwnerAgentV2:
        services = Services(
            roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"), feedback_dir=self.root,
            work_dir=self.root, mute=self.mute, deliver=None, now=lambda: NOW,
            set_camera=lambda cam, active: self.cameras.append((cam, active)) or {"ok": True},
            request_restart=lambda: self.restarts.append(1),
            set_option=lambda k, v: self.store.__setitem__(k, int(v) if v.isdigit() else v),
            read_settings=lambda: dict(self.store))
        return OwnerAgentV2(big, FakeRegistry("guard"), ChatMemory(os.path.join(self.root, "c")),
                            ReceiptBook(os.path.join(self.root, "r"), now=lambda: NOW), services, now=lambda: NOW)

    def test_undo_keeps_an_earlier_pause(self) -> None:
        from home_guard_project.box.feedback import Feedback  # noqa: PLC0415

        self.mute.apply(Feedback(action="mute", mute_until=NOW + 7200, camera="back_door"), NOW)
        agent = self.agent(Scripted([call("pause_alerts", owner_words="stop until six", until="06:00"),
                                     reply("")]))
        first = agent.handle("it's us, stop until six", "-5", {"user_id": 1})
        agent.undo_turn("-5", first.undo_token, {})
        self.assertTrue(self.mute.is_muted(NOW, "back_door"))
        self.assertFalse(self.mute.is_muted(NOW, "main_entrance"))

    def test_undo_a_pause(self) -> None:
        agent = self.agent(Scripted([call("pause_alerts", owner_words="stop until six", until="06:00"),
                                     reply("")]))
        first = agent.handle("it's us, stop until six", "-5", {"user_id": 1, "name": "A"})
        self.assertTrue(first.undo_token)
        self.assertTrue(self.mute.is_muted(NOW, "main_entrance"))
        undone = agent.undo_turn("-5", first.undo_token, {"user_id": 1})
        self.assertFalse(self.mute.is_muted(NOW, "main_entrance"))
        self.assertEqual(undone.text, "✓ Alerts are back on.")
        self.assertEqual(agent.undo_turn("-5", first.undo_token, {}).text, "There is nothing left to undo here.")

    def test_undo_a_camera_change_and_a_setting(self) -> None:
        agent = self.agent(Scripted([call("set_camera_active", camera="front", active=False),
                                     call("change_setting", setting="alert_hours", value="23-07",
                                          owner_words="watch 23 to 7"),
                                     reply("")]))
        first = agent.handle("turn off the front camera and watch 23 to 7", "-5", {})
        agent.undo_turn("-5", first.undo_token, {})
        self.assertEqual(self.cameras, [("front_side", False), ("front_side", True)])
        self.assertEqual((self.store["alert_start_hour"], self.store["alert_end_hour"]), (22, 6))

    def test_no_undo_button_when_nothing_changed(self) -> None:
        agent = self.agent(Scripted([reply("Nothing was recorded today.")]))
        self.assertEqual(agent.handle("anything today?", "-5", {}).undo_token, "")



class UndoRobustnessTest(UndoTest):
    def test_malformed_pause_restore_does_not_clear_current_pauses(self):
        from home_guard_project.box.feedback import Feedback
        agent = self.agent(Scripted([]))
        self.mute.apply(Feedback(action="mute", mute_until=NOW + 7200), NOW)
        for i, before in enumerate(([], {"all": "bad", "cameras": {}}, {"all": 0, "cameras": []})):
            token = str(123 + i)
            agent.book.issue("-5:" + token, "pause_alerts", "done", detail={"before": before})
            agent.undo_turn("-5", token)
            self.assertTrue(self.mute.is_muted(NOW, "front_side"))
            self.assertEqual(agent.book.turn_receipts("-5:" + token)[0].status, "done")

    def test_both_renderers_fail_but_after_actions_survive(self):
        from unittest.mock import patch
        from home_guard_project.box.brain.i18n import t
        agent = self.agent(Scripted([call("set_camera_active", camera="front", active=False), reply("")]))
        with patch("home_guard_project.box.brain.agent.render_reply", side_effect=ValueError("bad render")), patch(
                "home_guard_project.box.brain.agent.receipt_line", side_effect=ValueError("bad receipt")):
            out = agent.handle("turn off front", "-5")
        self.assertEqual(out.text, t("unavailable", "en"))
        self.assertTrue(out.after)

    def test_undo_never_raises_on_load_failure_or_bad_input(self):
        from unittest.mock import patch
        from home_guard_project.box.brain.i18n import t
        agent = self.agent(Scripted([]))
        with patch.object(agent.memory, "load", side_effect=ValueError("bad state")):
            self.assertEqual(agent.undo_turn("-5", "123", []).text, t("unavailable", "en"))
        for token in ([], {}, float("nan"), "bad:token"):
            self.assertIsInstance(agent.undo_turn("-5", token, {"user_id": []}).text, str)
        agent.services.read_settings = lambda: []
        self.assertIsInstance(agent.undo_turn("-5", "123", []).text, str)
        agent._now = lambda: float("inf")
        self.assertIsInstance(agent.undo_turn("-5", "123").text, str)

    def test_failed_camera_undo_stays_retryable(self):
        agent = self.agent(Scripted([call("set_camera_active", camera="front", active=False), reply("")]))
        first = agent.handle("turn off front", "-5")
        agent.services.set_camera = lambda *a: {"ok": False}
        undone = agent.undo_turn("-5", first.undo_token)
        self.assertIn("could not", undone.text.lower())
        self.assertEqual(undone.after, ())
        self.assertEqual(agent.book.turn_receipts("-5:" + first.undo_token)[0].status, "requested")

    def test_missing_settings_restore_defaults(self):
        self.store.clear()
        agent = self.agent(Scripted([call("change_setting", setting="alert_hours", value="23-07",
                                         owner_words="watch 23 to 7"), reply("")]))
        first = agent.handle("watch 23 to 7", "-5")
        self.assertEqual(first.receipts[0].detail["restore"], {"alert_start_hour": 0, "alert_end_hour": 0})
        agent.undo_turn("-5", first.undo_token)
        self.assertEqual((self.store["alert_start_hour"], self.store["alert_end_hour"]), (0, 0))

    def test_receipt_reader_skips_malformed_disk_records(self):
        import json
        from dataclasses import asdict
        agent = self.agent(Scripted([]))
        r = agent.book.issue("-5:123", "pause_alerts", "done")
        path = agent.book._path(dt.datetime.fromtimestamp(NOW).date())
        with open(path, "ab") as f:
            f.write(b"\xff\n")
            for bad in ([], dict(asdict(r), id=[]), dict(asdict(r), id="broken"),
                        dict(asdict(r), detail=[]), dict(asdict(r), ts=float("nan"))):
                f.write((json.dumps(bad) + "\n").encode())
        self.assertEqual([r.id for r in agent.book.turn_receipts("-5:123")], ["R1"])
        for days in ([], "bad", float("inf")):
            self.assertEqual(agent.book.turn_receipts("-5:123", days), [])

    def test_render_failure_keeps_receipts_and_after_actions(self):
        from unittest.mock import patch
        agent = self.agent(Scripted([call("set_camera_active", camera="front", active=False), reply("")]))
        with patch("home_guard_project.box.brain.agent.render_reply", side_effect=ValueError("bad render")), patch(
                "home_guard_project.box.brain.agent.save_feedback") as saved:
            out = agent.handle("turn off front", "-5")
        saved.assert_called_once()
        self.assertTrue(out.after)
        self.assertIn("front_side", out.text)
        self.assertTrue(out.undo_token)
        out.after[0]()
        self.assertEqual(self.restarts, [1])
        with patch("home_guard_project.box.brain.agent.render_reply", side_effect=ValueError("bad render")):
            undone = agent.undo_turn("-5", out.undo_token)
        self.assertTrue(undone.after)

    def test_effective_numeric_calls_run_once(self):
        from home_guard_project.box.brain.tools import _issue, _result
        for name, first, second in (
            ("record_clip", {"camera": "front"}, {"camera": "front", "seconds": 10.0}),
            ("record_clip", {"seconds": "99"}, {"seconds": 30}),
            ("record_clip", {"seconds": 10.9}, {"seconds": 10}),
            ("send_media", {"from_sec": "2", "seconds": "99"}, {"from_sec": 2.0, "seconds": 60}),
            ("send_media", {"from_sec": "2"}, {"from_sec": 2.0, "seconds": 10}),
        ):
            with self.subTest(name=name, first=first):
                calls = []
                def run(ctx, tool, args):
                    calls.append(tool)
                    return _result(_issue(ctx, tool, "done", "", {}))
                agent = self.agent(Scripted([call(name, **first), call(name, **second), reply("")]))
                agent._run_tool = run
                agent.handle("video please", "-5")
                self.assertEqual(calls, [name])

    def test_hebrew_cancel_word_does_not_match_phone(self):
        from home_guard_project.box.brain.profiles import needs_big
        for text in ("תשלח לי לטלפון", "בטלפון שלי", "ובטלפון"):
            self.assertFalse(needs_big(text), text)
        for text in ("בטל את ההשתקה", "ובטל!", "(בטל)", "תבטל", "שבטל", "הבטל"):
            self.assertTrue(needs_big(text), text)


if __name__ == "__main__":
    unittest.main()
