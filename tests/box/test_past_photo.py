# -*- coding: utf-8 -*-
"""A picture of a moment that has passed comes from the box's recordings (assistant_v3/past_photo.py).

2026-10-12 11:43-11:45: "אתה יכול להראות לי תמונה שהם היו שם ב 10:24", "לא של עכשיו יא מטומטם אני מתכוון שהם היו",
"תראה לי תמונה שהם שם בשעות האלה": each got a LIVE picture of the empty pergola and "no picture of those times was
kept", while the box's journal held the workers' events and their pictures. And the critic, reading only the first
1500 characters of the evidence, turned the right "seen until 10:43" into a wrong "until 10:24".
"""
import datetime as dt
import unittest

from home_guard_project.box.assistant_v3 import past_photo as pp
from home_guard_project.box.assistant_v3.supervisor import critic_messages

from test_assistant_v3 import DAY, FakeModel, act, acts_for, at, run

PERGOLA = "ameer_v2_ch3"
JOURNAL = [
    {"camera": PERGOLA, "time": "10:06:34", "end": "10:12:00", "people": 1,
     "summary": "A man in a black shirt stands on a chair holding an object by the wall."},
    {"camera": PERGOLA, "time": "10:18:39", "end": "10:24:15", "people": 1,
     "summary": "A man walks across the patio, opens a large metal container."},
    {"camera": PERGOLA, "time": "10:35:46", "end": "10:43:00", "people": 2,
     "summary": "Two men load boards into a pickup truck."},
    {"camera": "ameer_v2_ch6", "time": "10:22:10", "end": "10:24:00", "people": 1,
     "summary": "A man walks across the main entrance."},
]
ASK_1024 = "אתה יכול להראות לי תמונה שהם היו שם ב 10:24"
REPLY_HOURS = "צודק, התכוונתי שהעובדים נראו בפרגולה בין 10:06 ל־10:12 ובין 10:18 ל־10:24."


def turn(msg, acts, history=(), journal=JOURNAL, time="11:44:30", writer_text="הנה התמונה מ־10:24 בפרגולה."):
    understand = acts_for({msg: {"emotion": "annoyed", "acts": acts}})
    writer = FakeModel(lambda user, kw: writer_text)
    out = run({"time": time, "journal": list(journal), "history": list(history), "message": {"text": msg}},
              understand, writer)
    return out, " ".join(c["user"] for c in writer.calls)


class PastPhotoTurnTests(unittest.TestCase):
    def test_a_picture_at_a_past_time_is_the_recording_not_a_live_one(self):
        out, prompt = turn(ASK_1024, [act("question_history", quote=ASK_1024, camera=PERGOLA),
                                      act("command", quote=ASK_1024, command="send_photo")])
        self.assertEqual(out["error"], "")
        self.assertEqual(out["photo_cameras"], [])                 # no live picture of the empty pergola
        self.assertEqual(out["photos"], ["j1.jpg"])                # the 10:18-10:24 event's picture
        self.assertIn("לא חיה", prompt)

    def test_not_now_i_mean_when_they_were_here_sends_the_past_picture(self):
        msg = "לא של עכשיו יא מטומטם אני מתכוון שהם היו"
        history = [{"time": "11:43:55", "owner": ASK_1024, "bot": "בתמונה החיה ששלחתי עכשיו אין אנשים בפרגולה.",
                    "photos": [PERGOLA]}]
        out, _ = turn(msg, [act("complaint", quote=msg, issue="wrong_fact"), act("question_history", quote=msg)],
                      history)
        self.assertEqual(out["photo_cameras"], [])
        self.assertEqual(out["photos"], ["j1.jpg"])

    def test_those_hours_are_the_ones_the_assistant_just_cited(self):
        msg = "אבל אני רוצה שתראה לי אם הם היו היום תראה לי תמונה שהם שם בשעות האלה"
        history = [{"time": "11:44:20", "owner": "מה עם העובדים", "bot": REPLY_HOURS}]
        out, _ = turn(msg, [act("question_history", quote=msg), act("command", quote=msg, command="send_photo")],
                      history, time="11:44:50")
        self.assertEqual(out["photo_cameras"], [])
        self.assertEqual(sorted(out["photos"]), ["j0.jpg", "j1.jpg"])

    def test_no_recording_at_that_time_is_said_and_no_live_picture_is_sent(self):
        msg = "תראה לי תמונה שהם היו בפרגולה ב 08:00"
        out, prompt = turn(msg, [act("command", quote=msg, command="send_photo", camera=PERGOLA)])
        self.assertEqual(out["photos"], [])
        self.assertEqual(out["photo_cameras"], [])
        self.assertIn("לא מצאתי הקלטה", prompt)

    def test_a_plain_ask_is_still_a_live_picture(self):
        msg = "תראה לי תמונה מהפרגולה"
        out, _ = turn(msg, [act("command", quote=msg, command="send_photo", camera=PERGOLA)])
        self.assertEqual(out["photo_cameras"], [PERGOLA])


class PastPhotoUnitTests(unittest.TestCase):
    NOW = at("11:44")

    def test_past_words_and_clock_times(self):
        self.assertTrue(pp.asks_past(ASK_1024))
        self.assertTrue(pp.asks_past("לא של עכשיו"))
        self.assertFalse(pp.asks_past("תראה לי תמונה"))
        self.assertFalse(pp.asks_past("יש מישהו בפרגולה עכשיו?"))

    def test_a_later_clock_today_is_yesterday(self):
        (t,) = pp.clock_times("ב 23:30", self.NOW)
        self.assertEqual(dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M"),
                         (dt.date.fromisoformat(DAY) - dt.timedelta(days=1)).isoformat() + " 23:30")

    def test_the_event_spanning_the_moment_wins(self):
        records = [{"event_id": "a", "start": at("10:06"), "end": at("10:12"), "people_max": 1},
                   {"event_id": "b", "start": at("10:18"), "end": at("10:24"), "people_max": 1},
                   {"event_id": "c", "start": at("10:25"), "end": at("10:26"), "people_max": 0}]
        picked = pp.pick_events(records, [at("10:24")], self.NOW)
        self.assertEqual([r["event_id"] for r, _ in picked], ["b"])
        self.assertEqual(pp.pick_events(records, [at("07:00")], self.NOW), [])


class CriticEvidenceTests(unittest.TestCase):
    def test_the_critic_reads_all_the_evidence(self):
        lines = [f"מהיומן של הקופסה: 10:{i:02d}–10:{i + 1:02d} בפרגולה: " + "x" * 280 for i in range(8)]
        lines.append("מהיומן של הקופסה: 10:35–10:43 בפרגולה: שני גברים מעמיסים קרשים")
        user = critic_messages("sys", "איפה העובדים", [], [], "\n".join(lines), "draft")[1]["content"]
        self.assertIn("10:35–10:43", user)


if __name__ == "__main__":
    unittest.main()
