# -*- coding: utf-8 -*-
"""The house recorder's recording (box/recorder.py) and "when did the truck come" (assistant_v3/arrival.py).

2026-10-12 13:37-13:39, about the pickup at the pergola: "תן לי את הוידיו מתי הבנאדם יצא ממנו", "אני צריך מתי הבנאדם
יצא מהטנדר", "הטנדר לא היה פה בבוקר ... אני צריך בדיוק את הזמן שהוא הגיע": 10 live seconds and two watches he never
asked for (one on camera 2). The recorder had the truck coming in at 12:56 (13:01 on its own clock, 5 min ahead).
"""
import os
import tempfile
import unittest

import numpy as np

from home_guard_project.box import recorder as rc
from home_guard_project.box.assistant_v3 import arrival

from test_assistant_v3 import FakeModel, act, acts_for, at, run

PERGOLA = "ameer_v2_ch3"
LIVE = "rtsp://admin:pw@192.168.68.106:554/unicast/c3/s0/live"
ONVIF = ("<SOAP-ENV:Body><tds:GetSystemDateAndTimeResponse><tds:SystemDateAndTime><tt:DateTimeType>Manual"
         "</tt:DateTimeType><tt:UTCDateTime><tt:Time><tt:Hour>11</tt:Hour><tt:Minute>4</tt:Minute><tt:Second>51"
         "</tt:Second></tt:Time><tt:Date><tt:Year>2026</tt:Year><tt:Month>10</tt:Month><tt:Day>12</tt:Day></tt:Date>"
         "</tt:UTCDateTime><tt:LocalDateTime><tt:Time><tt:Hour>14</tt:Hour><tt:Minute>4</tt:Minute><tt:Second>51"
         "</tt:Second></tt:Time></tt:LocalDateTime></tds:SystemDateAndTime></tds:GetSystemDateAndTimeResponse>")
TRUCK = [0.15, 0.35, 0.35, 0.55]


class AddressTests(unittest.TestCase):
    def test_a_uniview_channel_plays_back_any_moment(self):
        self.assertEqual(rc.replay_url(LIVE, 1791799200, 1791799220),
                         "rtsp://admin:pw@192.168.68.106:554/c3/b1791799200/e1791799220/replay/")

    def test_another_address_has_no_recording(self):
        self.assertIsNone(rc.replay_url("rtsp://a:b@10.0.0.5:554/Streaming/Channels/101", 1, 2))

    def test_the_recorders_clock_from_onvif(self):
        self.assertEqual(rc.onvif_utc(ONVIF), 1791803091.0)          # 2026-10-12 11:04:51 UTC, not the local 14:04
        self.assertIsNone(rc.onvif_utc("<html>no</html>"))

    def test_the_offset_moves_every_playback(self):
        now = [1791802789.0]                                          # the box: 10:59:49 UTC
        r = rc.Recorder("x", tempfile.mkdtemp(), post=lambda host: ONVIF, now=lambda: now[0], url_of=lambda c: LIVE)
        self.assertEqual(r.offset(PERGOLA), 302.0)


class SearchTests(unittest.TestCase):
    """The truck stands on its place from 12:56:00; at 10:35 another vehicle stood elsewhere."""

    def setUp(self):
        self.r = rc.Recorder("x", tempfile.mkdtemp(), post=lambda host: "", url_of=lambda c: LIVE)
        self.arrived = at("12:56")
        self.looks = []

        def vehicles_at(camera, ts, detect):
            self.looks.append(ts)
            if ts >= self.arrived:
                return [list(TRUCK)]
            return [[0.6, 0.3, 0.8, 0.5]] if at("10:30") <= ts <= at("10:45") else []

        self.r.vehicles_at = vehicles_at

    def test_the_arrival_is_found_to_fifteen_seconds(self):
        out = self.r.arrival(PERGOLA, lambda p: [], now=at("13:39") + 20)
        self.assertTrue(out["ok"])
        (row,) = out["vehicles"]
        self.assertLessEqual(row["before"], self.arrived)
        self.assertGreaterEqual(row["arrived"], self.arrived)
        self.assertLessEqual(row["arrived"] - row["before"], rc.ARRIVAL_PRECISION_SEC)
        self.assertLess(out["looks"], 40)

    def test_a_vehicle_there_all_along(self):
        self.arrived = at("00:00")
        (row,) = self.r.arrival(PERGOLA, lambda p: [], now=at("13:39") + 20)["vehicles"]
        self.assertNotIn("arrived", row)
        self.assertIn("since_before", row)

    def test_no_recording_no_answer(self):
        r = rc.Recorder("x", tempfile.mkdtemp(), post=lambda host: "", url_of=lambda c: "rtsp://a/b")
        self.assertEqual(r.arrival(PERGOLA, lambda p: []), {"ok": False, "error": "no_recorder"})


class FramesTests(unittest.TestCase):
    def test_two_frames_a_second_apart_at_the_recorders_time(self):
        opened = []

        class Cap:
            def read(self):
                return True, np.zeros((8, 8, 3), np.uint8)

            def release(self):
                pass

        def open_capture(url):
            opened.append(url)
            return Cap()

        r = rc.Recorder("x", tempfile.mkdtemp(), post=lambda host: ONVIF, open_capture=open_capture,
                        now=lambda: 1791802789.0, url_of=lambda c: LIVE)
        r._mask = lambda camera: (None, True)
        self.assertEqual(len(r.frames(PERGOLA, 1791799000.0)), 2)
        self.assertIn("/c3/b1791799302/", opened[0])                 # box time + 302 s

    def test_a_zone_that_cannot_be_read_gives_no_picture(self):
        r = rc.Recorder("x", tempfile.mkdtemp(), post=lambda host: "", open_capture=lambda u: None,
                        url_of=lambda c: LIVE)
        r._mask = lambda camera: (None, False)
        self.assertEqual(r.frames(PERGOLA, 1.0), [])


class QuestionTests(unittest.TestCase):
    def test_when_did_it_come_is_a_question_not_a_watch(self):
        self.assertTrue(arrival.asks_arrival("אני לריך מתי הבנאדם יצא מהטנדר", "הטנדר"))
        self.assertTrue(arrival.asks_arrival("הוא הגיע מתישהו אני צריך בדיוק את הזמן שהוא הגיע", "הטנדר"))
        self.assertTrue(arrival.asks_arrival("תגיד לי מתי הטנדר הגיע", "הטנדר"))
        self.assertFalse(arrival.asks_arrival("תגיד לי מתי היא יוצאת", "הרכב"))
        self.assertFalse(arrival.asks_arrival("כשהרכב יוצא מהכניסה תגיד לי", "הרכב"))
        self.assertFalse(arrival.asks_arrival("מתי הם הגיעו", ""))


class ArrivalTurnTests(unittest.TestCase):
    PHOTO_REPLY = "בתמונה החיה מהפרגולה עכשיו נראים טנדר עם דלת פתוחה, נגרר וחומרי בנייה; אין שם אנשים."
    HISTORY = [{"time": "13:36:44", "owner": "תן לי תמונה", "bot": PHOTO_REPLY, "photos": [PERGOLA]}]

    def turn(self, msg, acts, recorder, time="13:39:17"):
        understand = acts_for({msg: {"emotion": "annoyed", "acts": acts}})
        writer = FakeModel(lambda user, kw: "הטנדר הגיע לפרגולה ב־12:56.")
        out = run({"time": time, "history": list(self.HISTORY), "recorder": recorder, "message": {"text": msg}},
                  understand, writer)
        return out, " ".join(c["user"] for c in writer.calls)

    def test_when_did_the_truck_come_is_answered_from_the_recording(self):
        msg = "יצא\nהטנדר לא היה פה בבוקר\nהוא הגיע מתישהו  אני צריך בדיוק את הזמן שהוא הגיע או שירד ממנו אדם?"
        out, prompt = self.turn(msg, [act("question_history", quote="הוא הגיע מתישהו"),
                                      act("watch_request", quote="הוא הגיע מתישהו", camera="ameer_v2_ch2",
                                          change="arrives")],
                                {PERGOLA: {"arrived": "12:56:05", "before": "12:55:50"}})
        self.assertEqual(out["error"], "")
        self.assertEqual([w for w in out["writes"] if "watch" in str(w)], [])          # no watch on camera 2
        self.assertEqual(len(out["videos"]), 1)
        self.assertTrue(out["videos"][0].endswith("_rec.mp4"))
        self.assertIn("12:56", prompt)
        self.assertIn("המקליט", prompt)

    def test_the_video_of_when_he_got_out_is_the_recording_not_live(self):
        msg = "תן לי את הוידיו מתי הבנאדם יצא ממנו"
        out, _ = self.turn(msg, [act("command", quote=msg, command="send_video", camera=PERGOLA)],
                           {PERGOLA: {"arrived": "12:56:05", "before": "12:55:50"}}, time="13:37:15")
        self.assertEqual(len(out["videos"]), 1)
        self.assertNotIn("live", out["videos"][0])

    def test_no_recording_is_said_and_nothing_is_watched(self):
        msg = "אני לריך מתי הבנאדם יצא מהטנדר"
        out, prompt = self.turn(msg, [act("watch_request", quote=msg, camera=PERGOLA, change="leaves")],
                                {PERGOLA: {"error": "no_playback"}}, time="13:38:34")
        self.assertEqual(out["videos"], [])
        self.assertEqual([w for w in out["writes"] if "watch" in str(w)], [])
        self.assertIn("לא הצלחתי לקרוא את ההקלטה", prompt)


if __name__ == "__main__":
    unittest.main()
