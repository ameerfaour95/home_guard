"""The two owner-approved system notices (2026-10-10): the box was down, and the AI is not available (with its one
"back" line). 2026-10-10 the box had no vision AI 00:25-09:05 (OpenRouter 402), was stopped 09:06-12:33 and lost power
at 09:23, and nobody knew (box/system_notices.py, wired in box/inference.py and box/control.py)."""
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime
from unittest import mock

import numpy as np

from home_guard_project.box import control
from home_guard_project.box import inference as inf
from home_guard_project.box import system_notices as sn

from test_guard_events import CAM, T0, Backend, GuardCase, answer


def at(day, hour, minute, second=0):
    return datetime(2026, 10, day, hour, minute, second).timestamp()


ERR_402 = ("qwen/qwen3.5-9b: APIStatusError: Error code: 402 - {'error': {'message': 'This request requires more "
           "credits, or fewer max_tokens.'}}")
ERR_401 = "qwen/qwen3.5-9b: AuthenticationError: Error code: 401 - {'error': {'message': 'No auth credentials found'}}"
ERR_TIMEOUT = "qwen/qwen3.5-9b: VlmDeadline: qwen/qwen3.5-9b gave no answer in 25 s"


class NoticeCase(unittest.TestCase):
    lang = "he"

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.state = os.path.join(self.dir, "state")
        self.logs = os.path.join(self.dir, "logs")
        self.sent = []
        self.now = [at(10, 2, 10)]
        self.booted = [None]

    def notices(self, on=True, provider="openrouter"):
        return sn.SystemNotices(self.state, self.logs, lambda kind, text: self.sent.append((kind, text)),
                                lambda: self.lang, provider=provider, on=on, clock=lambda: self.now[0],
                                boot=lambda: self.booted[0])

    def alive_at(self, ts):
        os.makedirs(self.state, exist_ok=True)
        with open(os.path.join(self.state, sn.ALIVE_NAME), "w", encoding="utf-8") as f:
            json.dump({"at": ts}, f)

    def alive_written(self):
        with open(os.path.join(self.state, sn.ALIVE_NAME), encoding="utf-8") as f:
            return json.load(f)["at"]


class BoxDownTest(NoticeCase):
    def start_at(self, ts, **kwargs):
        self.now[0] = ts
        with self.assertLogs("box.system_notices", "INFO") as logs:
            self.notices(**kwargs).box_started()
        return logs.output

    def test_the_very_first_run_sends_nothing_and_writes_the_alive_time(self):
        logs = self.start_at(at(10, 2, 47))
        self.assertEqual(self.sent, [])
        self.assertEqual(self.alive_written(), at(10, 2, 47))
        self.assertTrue(any("first run" in line for line in logs), logs)

    def test_a_gap_below_ten_minutes_is_not_news(self):
        self.alive_at(at(10, 2, 10))
        logs = self.start_at(at(10, 2, 19, 59))
        self.assertEqual(self.sent, [])
        self.assertIn("INFO:box.system_notices:Box-down notice: gap 02:10-02:19 (owner stop: no, power cut: unknown)"
                      " -> not sent (gap 9 min, below 10 min)", logs)

    def test_a_gap_of_ten_minutes_or_more_is_sent_once(self):
        self.alive_at(at(10, 2, 10))
        logs = self.start_at(at(10, 2, 47))
        self.assertEqual(self.sent, [(sn.BOX_DOWN, "⚠️ המערכת לא פעלה בין 02:10 ל-02:47.")])
        self.assertIn("INFO:box.system_notices:Box-down notice: gap 02:10-02:47 (owner stop: no, power cut: unknown)"
                      " -> sent", logs)
        self.assertEqual(self.alive_written(), at(10, 2, 47))
        self.start_at(at(10, 2, 48))                           # the next start: the gap is a minute
        self.assertEqual(len(self.sent), 1)

    def test_a_boot_after_the_last_alive_time_names_the_power_cut(self):
        self.alive_at(at(10, 9, 23))
        self.booted[0] = at(10, 12, 30)
        self.start_at(at(10, 12, 33))
        self.assertEqual(self.sent, [(sn.BOX_DOWN, "⚠️ המערכת לא פעלה בין 09:23 ל-12:33 (הקופסה כבתה, כנראה הפסקת חשמל).")])

    def test_a_boot_before_the_last_alive_time_says_no_cause(self):
        self.alive_at(at(10, 9, 23))
        self.booted[0] = at(10, 8, 0)                          # the program died, Windows did not
        logs = self.start_at(at(10, 12, 33))
        self.assertEqual(self.sent, [(sn.BOX_DOWN, "⚠️ המערכת לא פעלה בין 09:23 ל-12:33.")])
        self.assertTrue(any("power cut: no" in line for line in logs), logs)

    def test_the_owners_stop_is_not_news(self):
        self.alive_at(at(10, 9, 5, 30))
        control.stop(self.logs)                               # the app / Telegram / the CLI stop
        sn.record_owner_stop(self.logs, now=at(10, 9, 6))     # (its time, fixed for the test)
        control.start(self.logs)
        self.assertEqual(sn.owner_stop_at(self.logs), at(10, 9, 6))   # the start keeps the record
        logs = self.start_at(at(10, 12, 33))
        self.assertEqual(self.sent, [])
        self.assertTrue(any("owner stop: yes" in line and "not sent (the owner stopped the box)" in line
                            for line in logs), logs)

    def test_an_alive_time_just_after_the_stop_is_still_the_owners_stop(self):
        sn.record_owner_stop(self.logs, now=at(10, 9, 6))
        self.alive_at(at(10, 9, 6, 2))                        # the runner took two seconds to end the program
        self.start_at(at(10, 12, 33))
        self.assertEqual(self.sent, [])

    def test_an_old_stop_then_a_crash_is_news(self):
        sn.record_owner_stop(self.logs, now=at(9, 18, 0))     # stopped and started again yesterday
        self.alive_at(at(10, 2, 10))
        self.start_at(at(10, 2, 47))
        self.assertEqual(self.sent, [(sn.BOX_DOWN, "⚠️ המערכת לא פעלה בין 02:10 ל-02:47.")])

    def test_a_gap_across_midnight_carries_the_dates(self):
        self.alive_at(at(9, 23, 50))
        self.start_at(at(10, 0, 30))
        self.assertEqual(self.sent, [(sn.BOX_DOWN, "⚠️ המערכת לא פעלה בין 9.10 23:50 ל-10.10 00:30.")])

    def test_a_gap_longer_than_a_day_carries_the_dates(self):
        self.alive_at(at(8, 2, 10))
        self.start_at(at(10, 2, 47))
        self.assertEqual(self.sent, [(sn.BOX_DOWN, "⚠️ המערכת לא פעלה בין 8.10 02:10 ל-10.10 02:47.")])

    def test_english_box(self):
        self.lang = "en"
        self.alive_at(at(9, 23, 50))
        self.booted[0] = at(10, 0, 20)
        self.start_at(at(10, 0, 30))
        self.assertEqual(self.sent, [(sn.BOX_DOWN, "⚠️ The system was not running between Oct 9 23:50 and Oct 10 00:30 "
                                                   "(the box was off, probably a power cut).")])

    def test_system_notices_off_sends_nothing_but_keeps_the_alive_time(self):
        self.alive_at(at(10, 2, 10))
        logs = self.start_at(at(10, 2, 47), on=False)
        self.assertEqual(self.sent, [])
        self.assertTrue(any("-> not sent (system_notices off)" in line for line in logs), logs)
        self.assertEqual(self.alive_written(), at(10, 2, 47))

    def test_a_zeroed_alive_file_falls_back_to_its_time(self):
        os.makedirs(self.state, exist_ok=True)
        path = os.path.join(self.state, sn.ALIVE_NAME)
        with open(path, "wb") as f:
            f.write(b"\0" * 20)                               # 2026-10-10 09:23: replaced, never synced
        os.utime(path, (at(10, 9, 23), at(10, 9, 23)))
        self.start_at(at(10, 12, 33))
        self.assertEqual(self.sent, [(sn.BOX_DOWN, "⚠️ המערכת לא פעלה בין 09:23 ל-12:33.")])

    def test_the_alive_time_is_written_about_once_a_minute(self):
        self.start_at(at(10, 2, 0))
        notices = self.notices()
        notices.box_started(at(10, 2, 0))
        notices.tick(at(10, 2, 0, 59))
        self.assertEqual(self.alive_written(), at(10, 2, 0))
        notices.tick(at(10, 2, 1))
        self.assertEqual(self.alive_written(), at(10, 2, 1))
        self.assertFalse(os.path.exists(os.path.join(self.state, sn.ALIVE_NAME + ".tmp")))


class AiDownTest(NoticeCase):
    def fail(self, notices, ts, error=ERR_402):
        self.now[0] = ts
        notices.eye(False, error)

    def test_three_failures_inside_ten_minutes_wait_for_the_ten_minutes(self):
        notices = self.notices()
        for minute in (25, 27, 30):
            self.fail(notices, at(10, 0, minute))
        self.assertEqual(self.sent, [])
        notices.tick(at(10, 0, 34, 59))
        self.assertEqual(self.sent, [])
        with self.assertLogs("box.system_notices", "INFO") as logs:
            notices.tick(at(10, 0, 35))                      # no new Eye call: the guard loop's beat tells
        self.assertEqual(self.sent, [(sn.AI_DOWN, "⚠️ ה-AI לא זמין מ-00:25 (נגמר הקרדיט ב-OpenRouter). הקופסה ממשיכה "
                                                  "להקליט ולשמור, אבל בלי בדיקת AI לא נשלחות התרעות עד שזה יחזור.")])
        self.assertIn("INFO:box.system_notices:AI-down notice: no answer since 00:25, 3 failed Eye calls, cause: credit"
                      " -> sent", logs.output)

    def test_ten_minutes_with_only_two_failures_is_not_an_outage(self):
        notices = self.notices()
        self.fail(notices, at(10, 0, 25))
        self.fail(notices, at(10, 0, 50))
        notices.tick(at(10, 1, 30))
        self.assertEqual(self.sent, [])
        self.fail(notices, at(10, 1, 31))                    # the third: due at once
        self.assertEqual([kind for kind, _ in self.sent], [sn.AI_DOWN])

    def test_an_answer_breaks_the_streak(self):
        notices = self.notices()
        self.fail(notices, at(10, 0, 25))
        self.fail(notices, at(10, 0, 26))
        notices.eye(True, now=at(10, 0, 27))
        self.fail(notices, at(10, 0, 40))
        self.fail(notices, at(10, 0, 41))
        notices.tick(at(10, 1, 0))
        self.assertEqual(self.sent, [])                      # 2 in a row since 00:40, not 4
        notices.eye(True, now=at(10, 1, 1))
        self.assertEqual(self.sent, [])                      # never announced: no "back" line either

    def test_the_cause_words(self):
        self.assertEqual(sn.ai_cause(ERR_402), sn.CREDIT)
        self.assertEqual(sn.ai_cause("insufficient_quota"), sn.CREDIT)
        self.assertEqual(sn.ai_cause(ERR_401), sn.KEY)
        self.assertEqual(sn.ai_cause("Incorrect API key provided"), sn.KEY)
        self.assertEqual(sn.ai_cause(ERR_TIMEOUT), sn.OTHER)
        self.assertEqual(sn.ai_cause(""), sn.OTHER)
        self.assertEqual(sn.ai_cause(f"{ERR_402}; {ERR_401}"), sn.CREDIT)
        since, now = at(10, 0, 25), at(10, 0, 35)
        self.assertIn("(נגמר הקרדיט ב-OpenRouter)", sn.ai_down_text(since, now, sn.CREDIT, "openrouter", "he"))
        self.assertIn("(בעיה במפתח ה-API)", sn.ai_down_text(since, now, sn.KEY, "openrouter", "he"))
        self.assertIn("(הספק לא עונה)", sn.ai_down_text(since, now, sn.OTHER, "openrouter", "he"))
        self.assertIn("(the OpenRouter credit ran out)", sn.ai_down_text(since, now, sn.CREDIT, "openrouter", "en"))
        self.assertIn("(a problem with the API key)", sn.ai_down_text(since, now, sn.KEY, "openrouter", "en"))
        self.assertIn("(the provider is not answering)", sn.ai_down_text(since, now, sn.OTHER, "openrouter", "en"))
        self.assertIn("(נגמר הקרדיט אצל ספק ה-AI)", sn.ai_down_text(since, now, sn.CREDIT, "vllm", "he"))

    def test_each_cause_reaches_the_message(self):
        for error, words in ((ERR_401, "בעיה במפתח ה-API"), (ERR_TIMEOUT, "הספק לא עונה")):
            with self.subTest(error=error):
                shutil.rmtree(self.state, ignore_errors=True)
                self.sent.clear()
                notices = self.notices()
                for minute in (25, 30, 35):
                    self.fail(notices, at(10, 0, minute), error)
                self.assertEqual(len(self.sent), 1)
                self.assertIn(f"({words})", self.sent[0][1])

    def test_a_restart_mid_outage_does_not_send_again_and_the_back_line_goes_once(self):
        notices = self.notices()
        for minute in (25, 31, 36):
            self.fail(notices, at(10, 0, minute))
        self.assertEqual(len(self.sent), 1)
        again = self.notices()                               # the program restarted at 01:00
        for minute in (5, 40):
            self.fail(again, at(10, 1, minute))
        again.tick(at(10, 2, 0))
        self.assertEqual(len(self.sent), 1)
        with self.assertLogs("box.system_notices", "INFO") as logs:
            again.eye(True, now=at(10, 9, 5))
        self.assertEqual(self.sent[1], (sn.AI_BACK, "✅ ה-AI חזר לעבוד. לא היה זמין 00:25–09:05."))
        self.assertIn("INFO:box.system_notices:AI-back notice: not available 00:25-09:05, 5 failed Eye call(s) -> sent", logs.output)
        again.eye(True, now=at(10, 9, 6))
        self.notices().eye(True, now=at(10, 9, 7))           # and after a restart
        self.assertEqual(len(self.sent), 2)

    def test_the_back_line_in_english_and_across_midnight(self):
        self.assertEqual(sn.ai_back_text(at(9, 23, 50), at(10, 0, 40), 3, "en"),
                         "✅ The AI is working again. It was not available Oct 9 23:50–Oct 10 00:40, and 3 events were "
                         "saved without a check in that time.")
        self.assertEqual(sn.ai_back_text(at(10, 0, 25), at(10, 9, 5), 0, "he"),
                         "✅ ה-AI חזר לעבוד. לא היה זמין 00:25–09:05.")

    def test_system_notices_off_sends_neither(self):
        notices = self.notices(on=False)
        with self.assertLogs("box.system_notices", "INFO") as logs:
            for minute in (25, 31, 36):
                self.fail(notices, at(10, 0, minute))
        notices.eye(True, now=at(10, 9, 5))
        self.assertEqual(self.sent, [])
        self.assertTrue(any("AI-down notice:" in line and "-> not sent (system_notices off)" in line
                            for line in logs.output), logs.output)

    def test_a_failing_send_never_raises(self):
        notices = sn.SystemNotices(self.state, self.logs, mock.Mock(side_effect=OSError("no network")), lambda: "he",
                                   clock=lambda: self.now[0], boot=lambda: None)
        for minute in (25, 31, 36):
            self.fail(notices, at(10, 0, minute))
        notices.eye(True, now=at(10, 9, 5))


CAMS = ("ameer_v2_ch2", "ameer_v2_ch3", "ameer_v2_ch5")


class CamerasDownTest(NoticeCase):
    """2026-10-19: power back 06:58, the recorder on a new address, no picture from any camera until 10:45."""

    def frames(self, ts=0.0, **some):
        return {**{name: ts for name in CAMS}, **some}

    def test_ten_minutes_with_no_picture_is_sent_once(self):
        notices = self.notices()
        started = at(18, 6, 59)
        notices.cameras(at(18, 7, 8, 59), self.frames(), started)
        self.assertEqual(self.sent, [])
        with self.assertLogs("box.system_notices", "INFO") as logs:
            notices.cameras(at(18, 7, 9), self.frames(), started)
        self.assertEqual(self.sent, [(sn.CAMERAS_DOWN, "⚠️ אין תמונה מאף מצלמה מ-06:59. הקופסה פועלת, אבל בלי תמונה "
                                                       "לא נשלחות התרעות. כדאי לבדוק שמכשיר ההקלטה דלוק ומחובר לרשת.")])
        self.assertIn("INFO:box.system_notices:Cameras-down notice: no picture from 3 camera(s) since 06:59 -> sent",
                      logs.output)
        notices.cameras(at(18, 12, 0), self.frames(), started)
        self.assertEqual(len(self.sent), 1)

    def test_the_outage_counts_from_the_last_picture(self):
        notices = self.notices()
        notices.cameras(at(19, 9, 0), self.frames(at(19, 8, 55)), at(19, 8, 0))
        notices.cameras(at(19, 9, 5), self.frames(at(19, 8, 55)), at(19, 8, 0))
        self.assertEqual(self.sent[0][0], sn.CAMERAS_DOWN)
        self.assertIn("מ-08:55", self.sent[0][1])

    def test_one_camera_sending_is_no_outage(self):
        notices = self.notices()
        for minute in range(0, 60, 5):
            notices.cameras(at(19, 9, minute), self.frames(ameer_v2_ch3=at(19, 9, minute)), at(19, 8, 0))
        self.assertEqual(self.sent, [])

    def test_the_back_line_after_a_restart_names_the_moved_recorder(self):
        notices = self.notices()
        started = at(18, 6, 59)
        notices.cameras(at(18, 7, 10), self.frames(), started)
        notices.recorder_moved("192.168.68.106", "192.168.68.108", 6)
        again = self.notices()                                         # the finder's restart
        self.now[0] = at(19, 10, 45)
        self.alive_at(at(19, 10, 44, 30))
        again.box_started()
        again.cameras(at(19, 10, 45), self.frames(), at(19, 10, 45))   # nothing yet: the outage is still on
        self.assertEqual(len(self.sent), 1)
        with self.assertLogs("box.system_notices", "INFO") as logs:
            again.cameras(at(19, 10, 46), self.frames(ameer_v2_ch2=at(19, 10, 46)), at(19, 10, 45))
        self.assertEqual(self.sent[1], (sn.CAMERAS_BACK, "✅ המצלמות חזרו לעבוד. לא הייתה תמונה 18.10 06:59–19.10 10:46 "
                                                         "(מכשיר ההקלטה קיבל כתובת חדשה ברשת והקופסה מצאה אותו לבד)."))
        self.assertIn("INFO:box.system_notices:Cameras-back notice: no picture Oct 18 06:59-Oct 19 10:46 "
                      "(recorder moved: yes) -> sent", logs.output)
        self.notices().cameras(at(19, 10, 47), self.frames(at(19, 10, 47)), at(19, 10, 45))
        self.assertEqual(len(self.sent), 2)

    def test_a_short_outage_fixed_by_the_finder_sends_nothing(self):
        notices = self.notices()
        notices.cameras(at(19, 10, 43), self.frames(), at(19, 10, 41))
        notices.recorder_moved("192.168.68.106", "192.168.68.108", 6)
        again = self.notices()
        with self.assertLogs("box.system_notices", "INFO") as logs:
            again.cameras(at(19, 10, 46), self.frames(at(19, 10, 46)), at(19, 10, 45))
        self.assertEqual(self.sent, [])
        self.assertTrue(any("never announced -> nothing to send" in line for line in logs.output), logs.output)

    def test_an_unannounced_outage_is_not_carried_over_a_box_down(self):
        notices = self.notices()
        notices.cameras(at(16, 14, 33), self.frames(), at(16, 14, 30))
        self.alive_at(at(16, 14, 34))
        self.now[0] = at(18, 6, 59)
        again = self.notices()
        again.box_started()                                            # the box-down notice tells this part
        again.cameras(at(18, 7, 0), self.frames(), at(18, 6, 59))
        again.cameras(at(18, 7, 9), self.frames(), at(18, 6, 59))
        self.assertEqual([kind for kind, _ in self.sent], [sn.BOX_DOWN, sn.CAMERAS_DOWN])
        self.assertIn("מ-06:59", self.sent[1][1])

    def test_english_texts(self):
        self.assertEqual(sn.cameras_down_text(at(19, 6, 59), at(19, 7, 9), "en"),
                         "⚠️ No camera has sent a picture since 06:59. The box is running, but without pictures no "
                         "alerts are sent. Please check that the recorder is on and connected to the network.")
        self.assertEqual(sn.cameras_back_text(at(19, 6, 59), at(19, 10, 46), False, "en"),
                         "✅ The cameras are working again. There was no picture 06:59–10:46.")

    def test_system_notices_off_sends_neither(self):
        notices = self.notices(on=False)
        notices.cameras(at(19, 7, 0), self.frames(), at(19, 6, 0))
        notices.cameras(at(19, 8, 0), self.frames(at(19, 8, 0)), at(19, 6, 0))
        self.assertEqual(self.sent, [])

    def test_the_ai_outage_state_survives_beside_it(self):
        notices = self.notices()
        for minute in (25, 31, 36):
            notices.eye(False, ERR_402, now=at(10, 0, minute))
        notices.cameras(at(10, 0, 40), self.frames(), at(10, 0, 0))
        again = self.notices()
        again.eye(True, now=at(10, 9, 5))
        self.assertEqual([kind for kind, _ in self.sent], [sn.AI_DOWN, sn.CAMERAS_DOWN, sn.AI_BACK])

    def test_a_failing_send_never_raises(self):
        notices = sn.SystemNotices(self.state, self.logs, mock.Mock(side_effect=OSError("no network")), lambda: "he",
                                   clock=lambda: self.now[0], boot=lambda: None)
        notices.cameras(at(19, 7, 0), self.frames(), at(19, 6, 0))
        notices.cameras(at(19, 8, 0), self.frames(at(19, 8, 0)), at(19, 6, 0))


class SwitchTest(unittest.TestCase):
    def test_default_on_and_off_spellings(self):
        self.assertTrue(sn.enabled({}))
        self.assertTrue(sn.enabled({"system_notices": "on"}))
        self.assertTrue(sn.enabled({"system_notices": True}))
        self.assertFalse(sn.enabled({"system_notices": False}))      # YAML's off
        self.assertFalse(sn.enabled({"system_notices": "off"}))

    def test_start_reads_the_switch_and_checks_the_gap(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with mock.patch.dict(os.environ, {"HOMEGUARD_SYSTEM_NOTICES": ""}):
            notices = sn.start({"system_notices": "off"}, {}, lambda: "he", state_dir=tmp, log_dir=tmp)
        self.assertFalse(notices.on)
        self.assertTrue(os.path.exists(os.path.join(tmp, sn.ALIVE_NAME)))
        self.assertIsNone(sn.start({}, {}, lambda: "he", state_dir=tmp, log_dir=tmp))     # the suite's switch


class SenderTest(unittest.TestCase):
    def test_retries_until_the_network_is_up(self):
        results, waits = [False, False, True], []
        send = mock.Mock(side_effect=lambda text: results.pop(0))
        sender = sn.Sender(send, tries=5, wait=30.0, sleep=waits.append)
        self.assertTrue(sender.deliver(sn.BOX_DOWN, "x"))
        self.assertEqual(send.call_count, 3)
        self.assertEqual(waits, [30.0, 30.0])

    def test_gives_up_after_the_tries_and_never_raises(self):
        send = mock.Mock(side_effect=OSError("down"))
        sender = sn.Sender(send, tries=3, wait=1.0, sleep=lambda s: None)
        self.assertFalse(sender.deliver(sn.BOX_DOWN, "x"))
        self.assertEqual(send.call_count, 3)

    def test_telegram_not_set_up_is_not_retried(self):
        send = mock.Mock(return_value=None)
        self.assertFalse(sn.Sender(send, tries=5, sleep=lambda s: None).deliver(sn.AI_DOWN, "x"))
        self.assertEqual(send.call_count, 1)

    def test_a_normal_message_with_no_buttons_on_the_alert_chats(self):
        with mock.patch("home_guard_project.box.telegram_notify.send_message",
                        return_value={"sent": True}) as send_message:
            send = sn.telegram_send({"alert_channel": "telegram", "telegram_chat_ids": ["-5"]},
                                    {"TELEGRAM_BOT_TOKEN": "t"})
            self.assertTrue(send("שלום"))
        cfg, text = send_message.call_args.args
        self.assertEqual((cfg.chat_ids, text), (["-5"], "שלום"))
        self.assertEqual(send_message.call_args.kwargs, {})               # not silent, no reply_markup
        self.assertIsNone(sn.telegram_send({"alert_channel": "twilio"}, {})("x"))


class GuardLoopWiringTest(GuardCase):
    """_worker tells the notices how each Eye call ended, after the fallback and the rescue."""

    def setUp(self):
        super().setUp()
        self.notices = mock.Mock()
        patcher = mock.patch.object(inf, "NOTICES", self.notices)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_failed_call_carries_its_error(self):
        with self.assertLogs("box.inference", "INFO"):
            self.work(Backend(None), T0)
        self.notices.eye.assert_called_once()
        answered, error = self.notices.eye.call_args.args
        self.assertFalse(answered)
        self.assertEqual(sn.ai_cause(error), sn.CREDIT)

    def test_an_answer_is_an_answer(self):
        self.work(Backend(answer("normal", people=1)), T0)
        self.notices.eye.assert_called_once_with(True, "")

    def test_no_model_at_all_is_not_an_eye_call(self):
        self.work(inf.NullBackend(), T0)
        self.notices.eye.assert_not_called()

    def test_the_rescue_error_counts_too(self):
        inf.note_eye(Backend(), False, "", {"error": "APIStatusError: Error code: 401 - bad key"})
        self.assertEqual(sn.ai_cause(self.notices.eye.call_args.args[1]), sn.KEY)

    def test_the_guard_loops_beat(self):
        inf.notices_tick(123.0)
        self.notices.tick.assert_called_once_with(123.0)
        self.notices.tick.side_effect = RuntimeError("boom")
        inf.notices_tick(124.0)                                          # never stops the loop

    def test_the_cameras_beat_reads_each_cameras_last_picture(self):
        finder = mock.Mock()
        streams = {"a": mock.Mock(last_ts=100.0), "b": mock.Mock(last_ts=0.0)}
        inf.cameras_tick(finder, streams, 200.0, 50.0)
        self.notices.cameras.assert_called_once_with(200.0, {"a": 100.0, "b": 0.0}, 50.0)
        finder.check.assert_called_once_with(200.0, {"a": 100.0, "b": 0.0})


class TenTenNightReplayTest(NoticeCase):
    """2026-10-10: AI failing from 00:25 (402), answering again at 09:05; last alive 09:23 (power cut), start 12:33 with
    no owner stop. Exactly three messages: at 00:39 (the third failed call, 14 min in), 09:05 and 12:33."""

    def test_the_night(self):
        log = []
        notices = sn.SystemNotices(self.state, self.logs, lambda kind, text: log.append((self.now[0], kind, text)),
                                   lambda: "he", provider="openrouter", clock=lambda: self.now[0],
                                   boot=lambda: self.booted[0])
        self.now[0] = at(9, 22, 0)
        notices.box_started()
        failed = 0
        while self.now[0] < at(10, 9, 5):
            self.now[0] += 60                       # the guard loop's beat, every minute
            notices.tick()
            if self.now[0] >= at(10, 0, 25) and (self.now[0] - at(10, 0, 25)) % (7 * 60) == 0:
                notices.eye(False, ERR_402)         # a detection every 7 minutes from 00:25, every Eye call 402
                failed += 1
        notices.eye(True)                           # 09:05: the credit is back
        while self.now[0] < at(10, 9, 23):
            self.now[0] += 60
            notices.tick()                          # the last alive time: 09:23, then the power cut
        self.booted[0] = at(10, 12, 30)
        self.now[0] = at(10, 12, 33)
        sn.SystemNotices(self.state, self.logs, lambda kind, text: log.append((self.now[0], kind, text)),
                         lambda: "he", provider="openrouter", clock=lambda: self.now[0],
                         boot=lambda: self.booted[0]).box_started()
        clock = [(datetime.fromtimestamp(ts).strftime("%H:%M"), kind) for ts, kind, _ in log]
        self.assertEqual(clock, [("00:39", sn.AI_DOWN), ("09:05", sn.AI_BACK), ("12:33", sn.BOX_DOWN)])
        self.assertEqual(log[0][2], "⚠️ ה-AI לא זמין מ-00:25 (נגמר הקרדיט ב-OpenRouter). הקופסה ממשיכה להקליט ולשמור, "
                                    "אבל בלי בדיקת AI לא נשלחות התרעות עד שזה יחזור.")
        self.assertEqual(log[1][2], "✅ ה-AI חזר לעבוד. לא היה זמין 00:25–09:05.")
        self.assertEqual(log[2][2], "⚠️ המערכת לא פעלה בין 09:23 ל-12:33 (הקופסה כבתה, כנראה הפסקת חשמל).")


if __name__ == "__main__":
    unittest.main()
