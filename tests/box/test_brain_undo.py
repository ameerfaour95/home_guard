# tests/box/test_brain_undo.py
from __future__ import annotations

import datetime as dt
import os
import tempfile
import unittest

import dataclasses

from test_brain_agent import FakeRegistry, Scripted, call, reply

from home_guard_project.box.brain.agent import OwnerAgentV2
from home_guard_project.box.brain.memory import ChatMemory
from home_guard_project.box.brain.receipts import ReceiptBook
from home_guard_project.box.brain.tools import Services
from home_guard_project.box.feedback import MuteState

NOW = dt.datetime(2026, 10, 3, 23, 0).timestamp()
START = NOW


class TrackingRegistry(FakeRegistry):
    """The fake house, with each camera's on/off as the last set_camera call left it (like cameras.yaml)."""

    def __init__(self, changes, mode="guard"):
        super().__init__(mode)
        self.changes = changes

    def snapshot(self):
        snap = super().snapshot()
        state = dict(self.changes)
        cams = tuple(dataclasses.replace(c, enabled=state.get(c.name, c.enabled)) for c in snap.cameras)
        return dataclasses.replace(snap, cameras=cams)


class UndoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.mute = MuteState(os.path.join(self.root, "mute.json"))
        self.store = {"alert_start_hour": 22, "alert_end_hour": 6, "owner_language": "en"}
        self.cameras, self.restarts = [], []
        self.addCleanup(self.tick, 0)

    def tick(self, seconds: float) -> None:
        """Move the clock to START + *seconds* (each turn gets its own turn id and receipt time)."""
        global NOW
        NOW = START + seconds

    def agent(self, big) -> OwnerAgentV2:
        services = Services(
            roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"), feedback_dir=self.root,
            work_dir=self.root, mute=self.mute, deliver=None, now=lambda: NOW,
            set_camera=lambda cam, active: self.cameras.append((cam, active)) or {"ok": True},
            request_restart=lambda: self.restarts.append(1),
            set_option=lambda k, v: self.store.__setitem__(k, int(v) if v.isdigit() else v),
            read_settings=lambda: dict(self.store))
        return OwnerAgentV2(big, TrackingRegistry(self.cameras), ChatMemory(os.path.join(self.root, "c")),
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
        self.assertEqual(undone.text, "✗ Could not undo the change to front_side: something went wrong on the box")
        self.assertEqual(undone.after, ())
        self.assertEqual(undone.receipts[0].detail.get("undo_of"), "set_camera_active")
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


class UndoRaceTest(UndoTest):
    """Undo restores only what that turn changed; anything changed since stays as it is."""

    def pause(self, words: str, **args):
        agent = self.agent(Scripted([call("pause_alerts", owner_words=words, **args), reply("")]))
        return agent, agent.handle(words, "-5", {"user_id": 1})

    def run_turn(self, text: str, *calls):
        agent = self.agent(Scripted(list(calls) + [reply("")]))
        return agent, agent.handle(text, "-5", {"user_id": 1})

    def test_undo_of_an_older_pause_keeps_a_newer_resume(self) -> None:
        from home_guard_project.box.feedback import Feedback  # noqa: PLC0415

        self.mute.apply(Feedback(action="mute", mute_until=NOW + 7200), NOW)       # the house, until 01:00
        a, first = self.pause("stop the back door until six", camera="back_door", until="06:00")
        self.tick(60)
        self.run_turn("turn everything back on", call("resume_alerts", owner_words="turn everything back on"))
        self.tick(120)
        undone = a.undo_turn("-5", first.undo_token, {"user_id": 1})
        self.assertFalse(self.mute.is_muted(NOW, "main_entrance"))
        self.assertFalse(self.mute.is_muted(NOW, "back_door"))
        self.assertEqual(undone.text, "Changed since - nothing to undo for the pause of alerts from back_door.")
        self.assertEqual(a.book.turn_receipts("-5:" + first.undo_token)[0].status, "done")

    def test_undo_removes_only_its_own_entry(self) -> None:
        a, first = self.pause("stop the back door until six", camera="back_door", until="06:00")
        self.tick(60)
        self.pause("stop everything until one", until="01:00")
        self.tick(120)
        undone = a.undo_turn("-5", first.undo_token, {"user_id": 1})
        self.assertEqual(self.mute.snapshot()["cameras"], {})
        self.assertEqual(self.mute.muted_until(NOW, "main_entrance"), START + 7200)
        self.assertEqual(undone.text, "✓ Undone, but alerts from back_door stay paused until 01:00 by another pause.")

    def test_undo_puts_back_an_earlier_pause_of_the_same_camera(self) -> None:
        from home_guard_project.box.feedback import Feedback  # noqa: PLC0415

        self.mute.apply(Feedback(action="mute", mute_until=NOW + 7200, camera="back_door"), NOW)
        a, first = self.pause("stop the back door until six", camera="back_door", until="06:00")
        undone = a.undo_turn("-5", first.undo_token, {"user_id": 1})
        self.assertEqual(self.mute.muted_until(NOW, "back_door"), START + 7200)
        self.assertEqual(undone.text, "✓ Undone, but alerts from back_door stay paused until 01:00 by another pause.")

    def test_undo_of_a_house_pause_names_what_stays_paused(self) -> None:
        from home_guard_project.box.feedback import Feedback  # noqa: PLC0415

        self.mute.apply(Feedback(action="mute", mute_until=NOW + 7200, camera="back_door"), NOW)
        a, first = self.pause("stop everything until six", until="06:00")
        undone = a.undo_turn("-5", first.undo_token, {})
        self.assertEqual(undone.text, "✓ Alerts are back on, except back_door until 01:00 (paused separately).")

    def test_the_same_pause_set_again_later_is_not_undone(self) -> None:
        a, first = self.pause("stop everything until six", until="06:00")
        self.tick(60)
        self.run_turn("turn everything back on", call("resume_alerts", owner_words="turn everything back on"))
        self.tick(120)
        self.pause("stop everything until six again", until="06:00")
        self.tick(180)
        undone = a.undo_turn("-5", first.undo_token, {})
        self.assertTrue(self.mute.is_muted(NOW, "main_entrance"))
        self.assertEqual(undone.text, "Changed since - nothing to undo for the pause of all alerts.")

    def test_undo_of_an_older_setting_keeps_the_newer_value(self) -> None:
        c, first = self.run_turn("watch 23 to 7", call("change_setting", setting="alert_hours", value="23-07",
                                                       owner_words="watch 23 to 7"))
        self.tick(60)
        self.run_turn("watch 1 to 5", call("change_setting", setting="alert_hours", value="01-05",
                                           owner_words="watch 1 to 5"))
        self.tick(120)
        undone = c.undo_turn("-5", first.undo_token, {})
        self.assertEqual((self.store["alert_start_hour"], self.store["alert_end_hour"]), (1, 5))
        self.assertEqual(undone.text, "Changed since - nothing to undo for Alert hours.")
        self.assertEqual(first.receipts[0].detail["wrote"], {"alert_start_hour": "23", "alert_end_hour": "7"})

    def test_the_same_setting_written_again_later_is_not_undone(self) -> None:
        c, first = self.run_turn("watch 23 to 7", call("change_setting", setting="alert_hours", value="23-07",
                                                       owner_words="watch 23 to 7"))
        self.tick(60)
        self.run_turn("watch 1 to 5", call("change_setting", setting="alert_hours", value="01-05",
                                           owner_words="watch 1 to 5"))
        self.tick(120)
        self.run_turn("watch 23 to 7 again", call("change_setting", setting="alert_hours", value="23-07",
                                                  owner_words="watch 23 to 7 again"))
        self.tick(180)
        c.undo_turn("-5", first.undo_token, {})
        self.assertEqual((self.store["alert_start_hour"], self.store["alert_end_hour"]), (23, 7))

    def test_undo_of_an_older_camera_change_keeps_the_newer_state(self) -> None:
        off = lambda: call("set_camera_active", camera="front", active=False)  # noqa: E731
        on = lambda: call("set_camera_active", camera="front", active=True)  # noqa: E731
        a, first = self.run_turn("turn off front", off())
        self.tick(60)
        self.run_turn("turn on front", on())
        self.tick(120)
        undone = a.undo_turn("-5", first.undo_token, {})
        self.assertEqual(self.cameras, [("front_side", False), ("front_side", True)])
        self.assertEqual(undone.text, "Changed since - nothing to undo for the change to front_side.")
        self.assertEqual(undone.after, ())
        self.tick(180)
        self.run_turn("turn off front again", off())
        self.tick(240)
        a.undo_turn("-5", first.undo_token, {})
        self.assertEqual(self.cameras[-1], ("front_side", False))          # off, on, off: the camera stays off
        self.assertEqual(len(self.cameras), 3)

    def test_undo_newest_first_then_the_older_one(self) -> None:
        a, first = self.run_turn("turn off front", call("set_camera_active", camera="front", active=False))
        self.tick(60)
        b, second = self.run_turn("turn on front", call("set_camera_active", camera="front", active=True))
        self.tick(120)
        b.undo_turn("-5", second.undo_token, {})
        self.tick(180)
        undone = a.undo_turn("-5", first.undo_token, {})
        self.assertEqual(self.cameras[-1], ("front_side", True))
        self.assertIn("front_side", undone.text)
        self.assertNotIn("Changed since", undone.text)

    def test_second_tap_after_a_straight_undo_does_nothing(self) -> None:
        a, first = self.run_turn("turn off the front camera and watch 23 to 7",
                                 call("set_camera_active", camera="front", active=False),
                                 call("change_setting", setting="alert_hours", value="23-07",
                                      owner_words="watch 23 to 7"))
        self.tick(60)
        undone = a.undo_turn("-5", first.undo_token, {})
        self.assertNotIn("Changed since", undone.text)
        self.tick(120)
        self.assertEqual(a.undo_turn("-5", first.undo_token, {}).text, "There is nothing left to undo here.")
        self.assertEqual(self.cameras, [("front_side", False), ("front_side", True)])
        self.assertEqual((self.store["alert_start_hour"], self.store["alert_end_hour"]), (22, 6))

    def test_failed_setting_undo_has_its_own_line(self) -> None:
        a, first = self.run_turn("watch 23 to 7", call("change_setting", setting="alert_hours", value="23-07",
                                                       owner_words="watch 23 to 7"))
        def refuse(key, value):
            raise ValueError("disk full")
        a.services.set_option = refuse
        undone = a.undo_turn("-5", first.undo_token, {})
        self.assertEqual(undone.text, "✗ Could not undo Alert hours: something went wrong on the box")
        self.assertEqual(undone.receipts[0].detail["undo_of"], "change_setting")

    def test_undo_lines_in_hebrew_and_arabic(self) -> None:
        from home_guard_project.box.brain.i18n import t  # noqa: PLC0415

        self.assertEqual(t("undo_changed_since", "he", what="X"), "השתנה מאז - אין מה לבטל עבור X.")
        self.assertEqual(t("undo_changed_since", "en", what="X"), "Changed since - nothing to undo for X.")
        self.assertTrue(t("undo_changed_since", "ar", what="X"))
        self.assertEqual(t("undo_failed", "en", what="X", reason="r"), "✗ Could not undo X: r")
        self.assertTrue(t("undo_failed", "he", what="X", reason="r").startswith("✗"))


if __name__ == "__main__":
    unittest.main()
