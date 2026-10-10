from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import unittest

from home_guard_project.box.feedback import (
    FEEDBACK_BUTTONS,
    AlertIndex,
    Feedback,
    MuteState,
    button_feedback,
    confirmation_text,
    parse_feedback,
    save_feedback,
)

# 2026-10-02 15:00 local time, whatever the machine's time zone is.
NOW = dt.datetime(2026, 10, 2, 15, 0).timestamp()
HOUR = 3600.0
CAMERAS = ["front_door", "back_yard"]


def local(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def parse(obj: dict, **kwargs) -> Feedback:
    result = parse_feedback(json.dumps(obj), NOW, CAMERAS, **kwargs)
    assert result is not None
    return result


class ParseFeedbackTest(unittest.TestCase):
    def test_false_alarm(self) -> None:
        fb = parse({"verdict": "false_alarm", "action": "none", "note": "nothing was there"})
        self.assertEqual((fb.verdict, fb.action, fb.mute_until), ("false_alarm", "none", None))
        self.assertEqual(fb.note, "nothing was there")

    def test_every_verdict_is_accepted_and_unknown_ones_become_none(self) -> None:
        for verdict in ("true_alert", "false_alarm", "real_but_wrong", "expected", "missed_event", "none"):
            self.assertEqual(parse({"verdict": verdict}).verdict, verdict)
        self.assertEqual(parse({"verdict": "delete all clips"}).verdict, "none")
        self.assertEqual(parse({"verdict": "none", "action": "format_disk"}).action, "none")

    def test_mute_until_a_clock_time_means_its_next_occurrence(self) -> None:
        fb = parse({"verdict": "expected", "action": "mute", "mute_until": "18:00"})
        self.assertEqual(local(fb.mute_until), "2026-10-02 18:00")
        fb = parse({"action": "mute", "mute_until": "08:30"})  # already past today -> tomorrow
        self.assertEqual(local(fb.mute_until), "2026-10-03 08:30")

    def test_mute_for_some_minutes(self) -> None:
        fb = parse({"action": "mute", "mute_minutes": 90})
        self.assertEqual(fb.mute_until, NOW + 90 * 60)

    def test_mute_without_an_end_and_too_long_mutes_are_capped(self) -> None:
        self.assertEqual(parse({"action": "mute"}).mute_until, NOW + 24 * HOUR)
        self.assertEqual(parse({"action": "mute", "mute_minutes": 100000}).mute_until, NOW + 24 * HOUR)
        self.assertEqual(parse({"action": "mute"}, max_mute_hours=2).mute_until, NOW + 2 * HOUR)

    def test_nonsense_mute_values_fall_back_to_the_cap(self) -> None:
        for bad in ({"mute_until": "soon"}, {"mute_minutes": -5}, {"mute_minutes": "abc"}, {"mute_until": "25:99"}):
            self.assertEqual(parse({"action": "mute", **bad}).mute_until, NOW + 24 * HOUR, bad)

    def test_mute_fields_are_ignored_unless_the_action_is_mute(self) -> None:
        self.assertIsNone(parse({"verdict": "true_alert", "action": "none", "mute_minutes": 60}).mute_until)
        self.assertIsNone(parse({"action": "resume", "mute_until": "18:00"}).mute_until)

    def test_camera_must_be_one_of_the_box_cameras(self) -> None:
        self.assertEqual(parse({"action": "mute", "camera": "Back_Yard"}).camera, "back_yard")
        self.assertIsNone(parse({"action": "mute", "camera": "the moon"}).camera)

    def test_note_is_trimmed_and_bounded(self) -> None:
        self.assertEqual(len(parse({"verdict": "none", "note": "x" * 5000}).note), 300)

    def test_json_inside_other_text_is_found_and_garbage_is_rejected(self) -> None:
        fb = parse_feedback('Sure! {"verdict": "true_alert"} hope that helps', NOW, CAMERAS)
        self.assertEqual(fb.verdict, "true_alert")
        self.assertIsNone(parse_feedback("I could not understand", NOW, CAMERAS))
        self.assertIsNone(parse_feedback("[1, 2]", NOW, CAMERAS))
        self.assertIsNone(parse_feedback("", NOW, CAMERAS))


class FindQueryTest(unittest.TestCase):
    def _query(self, find: dict):
        fb = parse({"action": "find", "find": find})
        self.assertEqual(fb.action, "find")
        return fb.query

    def test_no_details_means_the_last_day(self) -> None:
        q = self._query({})
        self.assertEqual((q.start_ts, q.end_ts), (NOW - 24 * HOUR, NOW))
        self.assertTrue(q.want_video)

    def test_a_clock_time_means_the_hour_from_its_most_recent_occurrence(self) -> None:
        q = self._query({"from": "14:00"})   # earlier today
        self.assertEqual((local(q.start_ts), local(q.end_ts)), ("2026-10-02 14:00", "2026-10-02 15:00"))
        q = self._query({"from": "22:00"})   # not reached yet today -> last night
        self.assertEqual(local(q.start_ts), "2026-10-01 22:00")

    def test_a_range_over_midnight(self) -> None:
        q = self._query({"day": "yesterday", "from": "22:00", "to": "06:00"})
        self.assertEqual((local(q.start_ts), local(q.end_ts)), ("2026-10-01 22:00", "2026-10-02 06:00"))

    def test_a_whole_day_and_the_last_hours(self) -> None:
        q = self._query({"day": "2026-09-30"})
        self.assertEqual((local(q.start_ts), local(q.end_ts)), ("2026-09-30 00:00", "2026-10-01 00:00"))
        self.assertEqual(self._query({"last_hours": 3}).start_ts, NOW - 3 * HOUR)

    def test_the_range_never_leaves_what_is_kept_or_reaches_the_future(self) -> None:
        q = self._query({"day": "2020-01-01"})
        self.assertEqual(q.start_ts, NOW - 14 * 24 * HOUR)
        self.assertLessEqual(self._query({"day": "today"}).end_ts, NOW)
        self.assertEqual(self._query({"last_hours": 99999}).start_ts, NOW - 14 * 24 * HOUR)

    def test_latest_camera_words_and_text_only(self) -> None:
        q = self._query({"latest": True, "camera": "front_door", "what": "the delivery man", "want": "text"})
        self.assertTrue(q.latest)
        self.assertEqual((q.camera, q.what, q.want_video), ("front_door", "the delivery man", False))
        self.assertEqual(q.start_ts, NOW - 14 * 24 * HOUR)

    def test_no_query_unless_the_action_is_find(self) -> None:
        self.assertIsNone(parse({"verdict": "true_alert", "find": {"latest": True}}).query)


class ButtonAndTextTest(unittest.TestCase):
    def test_every_button_gives_a_feedback(self) -> None:
        codes = [code for row in FEEDBACK_BUTTONS for _, code in row]
        self.assertEqual(len(codes), len(set(codes)))
        for code in codes:
            self.assertIsNotNone(button_feedback(code, NOW), code)
            self.assertLessEqual(len(code.encode()), 64)  # Telegram's callback_data limit
        self.assertIsNone(button_feedback("fb:whatever", NOW))

    def test_buttons_mean_what_they_say(self) -> None:
        self.assertEqual(button_feedback("fb:true", NOW).verdict, "true_alert")
        self.assertEqual(button_feedback("fb:false", NOW).verdict, "false_alarm")
        self.assertEqual(button_feedback("fb:expected", NOW).verdict, "expected")
        pause = button_feedback("fb:mute60", NOW)
        self.assertEqual((pause.action, pause.mute_until), ("mute", NOW + HOUR))

    def test_confirmation_says_what_was_understood(self) -> None:
        self.assertIn("false alarm", confirmation_text(Feedback(verdict="false_alarm")))
        self.assertIn("real", confirmation_text(Feedback(verdict="true_alert")))
        paused = confirmation_text(Feedback(verdict="expected", action="mute", mute_until=NOW + 3 * HOUR))
        self.assertIn("18:00", paused)
        self.assertIn("continue", paused)
        one_camera = confirmation_text(Feedback(action="mute", mute_until=NOW + HOUR, camera="back_yard"))
        self.assertIn("back_yard", one_camera)
        self.assertIn("back on", confirmation_text(Feedback(action="resume")))
        self.assertTrue(confirmation_text(Feedback()))


class StateTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def test_mute_all_expires_and_survives_a_restart(self) -> None:
        path = os.path.join(self.dir, "alert_mute.json")
        state = MuteState(path)
        self.assertFalse(state.is_muted(NOW, "front_door"))
        state.apply(Feedback(action="mute", mute_until=NOW + HOUR), NOW)

        reloaded = MuteState(path)
        self.assertTrue(reloaded.is_muted(NOW + 10, "front_door"))
        self.assertTrue(reloaded.is_muted(NOW + 10, "back_yard"))
        self.assertFalse(reloaded.is_muted(NOW + HOUR + 1, "front_door"))

    def test_mute_one_camera_leaves_the_others_on(self) -> None:
        state = MuteState(os.path.join(self.dir, "m.json"))
        state.apply(Feedback(action="mute", mute_until=NOW + HOUR, camera="back_yard"), NOW)
        self.assertTrue(state.is_muted(NOW + 10, "back_yard"))
        self.assertFalse(state.is_muted(NOW + 10, "front_door"))

    def test_resume_turns_everything_back_on(self) -> None:
        state = MuteState(os.path.join(self.dir, "m.json"))
        state.apply(Feedback(action="mute", mute_until=NOW + HOUR), NOW)
        state.apply(Feedback(action="mute", mute_until=NOW + HOUR, camera="back_yard"), NOW)
        state.apply(Feedback(action="resume"), NOW)
        self.assertFalse(state.is_muted(NOW + 10, "back_yard"))
        self.assertFalse(MuteState(os.path.join(self.dir, "m.json")).is_muted(NOW + 10, "front_door"))

    def test_feedback_without_an_action_changes_nothing(self) -> None:
        state = MuteState(os.path.join(self.dir, "m.json"))
        state.apply(Feedback(verdict="false_alarm"), NOW)
        self.assertFalse(state.is_muted(NOW, "front_door"))

    def test_a_damaged_state_file_means_not_muted(self) -> None:
        path = os.path.join(self.dir, "m.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertFalse(MuteState(path).is_muted(NOW, "front_door"))

    def test_alert_index_finds_the_alert_a_reply_belongs_to(self) -> None:
        path = os.path.join(self.dir, "alert_index.json")
        index = AlertIndex(path)
        first = {"alert_id": "front_door_100_alert", "camera": "front_door", "summary": "a person", "ts": NOW - 600}
        second = {"alert_id": "back_yard_200_alert", "camera": "back_yard", "summary": "a car", "ts": NOW - 60}
        index.remember("-1001", 11, first)
        index.remember("-1001", 12, second)
        index.remember("555", 12, first)

        reloaded = AlertIndex(path)
        self.assertEqual(reloaded.lookup("-1001", 11)["alert_id"], "front_door_100_alert")
        self.assertEqual(reloaded.lookup("-1001", 12)["alert_id"], "back_yard_200_alert")
        self.assertIsNone(reloaded.lookup("-1001", 99))
        self.assertEqual(reloaded.latest("-1001", NOW)["alert_id"], "back_yard_200_alert")
        self.assertIsNone(reloaded.latest("-1001", NOW + 24 * HOUR))  # too old to guess
        self.assertIsNone(reloaded.latest("777", NOW))

    def test_alert_index_keeps_only_the_most_recent_alerts(self) -> None:
        index = AlertIndex(os.path.join(self.dir, "i.json"), keep=3)
        for i in range(5):
            index.remember("1", i, {"alert_id": f"a{i}", "camera": "c", "summary": "", "ts": NOW + i})
        self.assertIsNone(index.lookup("1", 0))
        self.assertEqual(index.lookup("1", 4)["alert_id"], "a4")

    def test_save_feedback_writes_one_file_next_to_the_production_clips(self) -> None:
        alert = {"alert_id": "front_door_100_alert", "camera": "front_door", "summary": "a person", "ts": NOW - 60}
        fb = Feedback(verdict="false_alarm", note="nothing there", source="text")
        path = save_feedback(self.dir, alert, fb, "no there was nothing", {"user_id": 42, "name": "Dana"}, "-1001", NOW)

        rel = os.path.relpath(path, self.dir).replace("\\", "/")
        self.assertTrue(rel.startswith("feedback/front_door/2026-10-02/front_door_100_alert_"), rel)
        self.assertTrue(rel.endswith(".feedback.json"), rel)
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["verdict"], "false_alarm")
        self.assertEqual(saved["raw_text"], "no there was nothing")
        self.assertEqual(saved["alert"]["alert_id"], "front_door_100_alert")
        self.assertEqual(saved["from"], {"user_id": 42, "name": "Dana"})
        self.assertEqual(saved["chat_id"], "-1001")
        self.assertIsNone(saved["mute_until_utc"])

    def test_feedback_that_belongs_to_no_alert_is_saved_too(self) -> None:
        fb = Feedback(verdict="missed_event", note="someone came at 3 and no alert")
        path = save_feedback(self.dir, None, fb, "someone came at 3", {"user_id": 1, "name": ""}, "5", NOW)
        rel = os.path.relpath(path, self.dir).replace("\\", "/")
        self.assertTrue(rel.startswith("feedback/_general/2026-10-02/general_"), rel)

    def test_two_feedbacks_on_one_alert_do_not_overwrite_each_other(self) -> None:
        alert = {"alert_id": "a", "camera": "c", "summary": "", "ts": NOW}
        one = save_feedback(self.dir, alert, Feedback(verdict="false_alarm"), "no", {}, "1", NOW)
        two = save_feedback(self.dir, alert, Feedback(verdict="true_alert"), "sorry, yes", {}, "1", NOW)
        self.assertNotEqual(one, two)
        self.assertTrue(os.path.isfile(one) and os.path.isfile(two))


if __name__ == "__main__":
    unittest.main()
