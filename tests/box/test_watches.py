# -*- coding: utf-8 -*-
"""Owner watches (box/watches.py) and the v3 turns around them, 2026-10-11 07:57:

"תבדוק במצלמה הראשית אם יש רכב לבן" -> photo -> "מעולה ברגע שהיא יוצאת תן לי דיווח" was understood as a camera NAME
and "אשת בעל הבית" was saved as an alias of the main entrance, twice. Now it is a one-shot watch: the vehicles seen
when he asked are checked every minute by the detector, and one photo goes out when the car is gone on two checks in
a row. A camera name is saved only when he says to call / name it. "מה המספר של הרכב של הבעלים" is a memory question:
the writer is told to say plainly when it is not saved, never to answer from a clip.
"""
import os
import tempfile
import unittest
from typing import Any, Dict, List

from home_guard_project.box import eval_chat as ec
from home_guard_project.box import watches as wt

from test_assistant_v3 import ALERT, FakeModel, act, acts_for, run

CAR = {"label": "car", "conf": 0.8, "box": [0.40, 0.50, 0.60, 0.80]}
OTHER_CAR = {"label": "car", "conf": 0.8, "box": [0.05, 0.50, 0.20, 0.70]}


class FakeBox:
    """What the camera shows now (a list of detector objects per check), and the photos sent."""

    def __init__(self, frames: List[List[Dict[str, Any]]]) -> None:
        self.frames, self.sent, self.clock = list(frames), [], {"now": 1_000_000.0}

    def grab(self, camera: str) -> Dict[str, Any]:
        return {"camera": camera, "image": f"/live/{camera}_{int(self.clock['now'])}.jpg"}

    def detect(self, path: str) -> Any:
        return self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]

    def send(self, chat_id: str, image: str, caption: str) -> bool:
        self.sent.append((chat_id, image, caption))
        return True


def store(box: FakeBox, tmp: str) -> wt.Watches:
    return wt.Watches(os.path.join(tmp, "watches.json"), box.grab, box.detect, box.send, now=lambda: box.clock["now"])


class WatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def step(self, box: FakeBox, w: wt.Watches, times: int = 1) -> None:
        for _ in range(times):
            box.clock["now"] += wt.CHECK_SEC
            w.tick()

    def test_the_car_leaving_sends_one_photo_after_two_checks(self):
        box = FakeBox([[CAR], [CAR], [], [], []])
        w = store(box, self.tmp)
        seen = w.look("ameer_v2_ch6")
        w.add("-1", "ameer_v2_ch6", "כניסה ראשית", "leaves", "הרכב הלבן", seen["vehicles"])
        self.step(box, w)                       # still there
        self.step(box, w)                       # gone once: not yet
        self.assertEqual(box.sent, [])
        self.step(box, w)                       # gone twice: told
        self.assertEqual(len(box.sent), 1)
        self.assertTrue(box.sent[0][2].startswith("🚗 הרכב הלבן יצא מכניסה ראשית · "))
        self.step(box, w, 3)                    # one-shot
        self.assertEqual(len(box.sent), 1)
        self.assertEqual(w.active(), [])

    def test_a_flicker_is_not_leaving(self):
        box = FakeBox([[CAR], [], [CAR], [], [CAR], [CAR]])
        w = store(box, self.tmp)
        w.add("-1", "ameer_v2_ch6", "כניסה ראשית", "leaves", "הרכב הלבן", w.look("ameer_v2_ch6")["vehicles"])
        self.step(box, w, 5)
        self.assertEqual(box.sent, [])

    def test_another_car_does_not_hide_ours_leaving(self):
        box = FakeBox([[CAR, OTHER_CAR], [OTHER_CAR], [OTHER_CAR]])
        w = store(box, self.tmp)
        w.add("-1", "ameer_v2_ch6", "כניסה ראשית", "leaves", "", w.look("ameer_v2_ch6")["vehicles"])
        self.step(box, w, 2)
        self.assertEqual(len(box.sent), 1)
        self.assertIn("אחד הרכבים יצא", box.sent[0][2])

    def test_arrives(self):
        box = FakeBox([[], [CAR], [CAR]])
        w = store(box, self.tmp)
        w.add("-1", "ameer_v2_ch2", "מצלמה 2", "arrives", "הטנדר", w.look("ameer_v2_ch2")["vehicles"])
        self.step(box, w, 2)
        self.assertEqual(len(box.sent), 1)
        self.assertIn("הטנדר הגיע למצלמה 2", box.sent[0][2])

    def test_no_picture_counts_as_nothing_and_the_watch_expires_quietly(self):
        box = FakeBox([[CAR]])
        w = store(box, self.tmp)
        w.add("-1", "ameer_v2_ch6", "כניסה ראשית", "leaves", "", [CAR["box"]], until=box.clock["now"] + 150)
        w.detect = lambda path: None             # the detector cannot run
        self.step(box, w, 4)
        self.assertEqual(box.sent, [])
        self.assertEqual(w.active(), [])

    def test_the_watch_survives_a_restart(self):
        box = FakeBox([[CAR]])
        store(box, self.tmp).add("-1", "ameer_v2_ch6", "כניסה ראשית", "leaves", "", [CAR["box"]])
        self.assertEqual(len(store(box, self.tmp).active("-1")), 1)


class WatchTurnTests(unittest.TestCase):
    """Whole turns through the golden suite's stubbed box: its watches stand on the case's live "vehicles" count."""

    MSG = "מעולה ברגע שהיא יוצאת תן לי דיווח"
    CAR_AT_ENTRANCE = {"ameer_v2_ch6": {"description": "A white car parked at the main entrance.", "vehicles": 1}}

    def tearDown(self) -> None:
        wt._current = None

    def watches(self, out):
        return [w for w in out["writes"] if w["type"] == "watch"]

    def test_tell_me_when_she_leaves_is_a_watch_not_a_camera_name(self):
        understand = acts_for({self.MSG: {"emotion": "happy", "acts": [
            act("watch_request", quote="ברגע שהיא יוצאת תן לי דיווח", camera="cam6", subject="אשת בעל הבית",
                change="leaves")]}})
        writer = FakeModel(lambda user, kw: "סגור, אשלח לך תמונה ברגע שהרכב הלבן יוצא מהכניסה הראשית.")
        out = run({"time": "07:57:50", "live": self.CAR_AT_ENTRANCE,
                   "history": [{"time": "07:57:26", "owner": "תבדוק במצלמה הראשית אם יש רכב לבן שם",
                                "bot": "כן, הרכב הלבן עדיין חונה בכניסה."}],
                   "message": {"text": self.MSG}}, understand, writer)
        self.assertEqual(out["error"], "")
        self.assertFalse([w for w in out["writes"] if w["type"] == "alias"])
        rows = self.watches(out)
        self.assertEqual(len(rows), 1)
        # what is watched is the car he named earlier, never the person label the understanding gave
        self.assertEqual((rows[0]["camera"], rows[0]["change"], rows[0]["text"]),
                         ("ameer_v2_ch6", "leaves", "הרכב הלבן"))
        self.assertIn("אשלח לך תמונה ברגע שהרכב הלבן יוצא מכניסה ראשית", " ".join(c["user"] for c in writer.calls))
        self.assertIn("הרכב הלבן", out["text"])

    def test_a_promise_with_no_watch_is_not_sent(self):
        understand = acts_for({self.MSG: {"emotion": "happy", "acts": [act("chit_chat", quote=self.MSG)]}})
        writer = FakeModel(lambda user, kw: "סגור, אעדכן אותך כשהיא יוצאת.")
        out = run({"time": "07:57:50", "live": self.CAR_AT_ENTRANCE, "message": {"text": self.MSG}},
                  understand, writer)
        self.assertNotIn("אעדכן אותך כש", out["text"])

    def test_a_misread_alias_without_naming_words_is_dropped(self):
        understand = acts_for({self.MSG: {"emotion": "happy", "acts": [
            act("preference", quote="ברגע שהיא יוצאת", command="alias", camera="cam6", value="אשת בעל הבית")]}})
        writer = FakeModel(lambda user, kw: "👍")
        out = run({"time": "07:57:50", "message": {"text": self.MSG}}, understand, writer)
        self.assertFalse([w for w in out["writes"] if w["type"] == "alias"])
        self.assertNotIn("אשת בעל הבית", out["text"])

    def test_a_real_name_is_still_saved(self):
        understand = acts_for({"תקרא לה חניה מעכשיו": {"emotion": "neutral", "acts": [
            act("preference", quote="תקרא לה חניה מעכשיו", command="alias", camera="cam2", value="חניה")]}})
        writer = FakeModel(lambda user, kw: "סגור.")
        out = run({"time": "08:00:00", "message": {"text": "תקרא לה חניה מעכשיו"}}, understand, writer)
        self.assertTrue([w for w in out["writes"] if w["type"] == "alias"])

    def test_no_vehicle_now_sets_nothing(self):
        understand = acts_for({self.MSG: {"emotion": "happy", "acts": [
            act("watch_request", quote="ברגע שהיא יוצאת תן לי דיווח", camera="cam6", change="leaves")]}})
        writer = FakeModel(lambda user, kw: "לא רואה רכב בכניסה הראשית עכשיו, אולי היא כבר יצאה.")
        out = run({"time": "07:57:50", "message": {"text": self.MSG}}, understand, writer)
        self.assertEqual(self.watches(out), [])


class MemoryQuestionTests(unittest.TestCase):
    def test_one_thing_not_saved_is_said_plainly(self):
        q = "מה המספר של הרכב של הבעלים"
        understand = acts_for({q: {"emotion": "neutral", "acts": [
            act("question_memory", quote=q, subject="מספר הרכב של בעל הבית")]}})
        writer = FakeModel(lambda user, kw: "אין לי שמור את מספר הרכב שלך. אם תכתוב לי אותו, אזכור.")
        out = run({"time": "23:55:24", "alerts": [ALERT], "message": {"text": q}}, understand, writer)
        prompt = " ".join(c["user"] for c in writer.calls)
        self.assertIn("you do not have it saved", prompt)
        self.assertNotIn("בלי לדלג על אף אחד", prompt)          # not "list everything you remember"
        self.assertEqual(out["error"], "")

    def test_what_do_you_remember_still_lists(self):
        q = "מה אתה זוכר?"
        understand = acts_for({q: {"emotion": "neutral", "acts": [act("question_memory", quote=q)]}})
        writer = FakeModel(lambda user, kw: "שום דבר מיוחד עדיין.")
        run({"time": "12:00:00", "message": {"text": q}}, understand, writer)
        self.assertIn("בלי לדלג על אף אחד", " ".join(c["user"] for c in writer.calls))


if __name__ == "__main__":
    unittest.main()
