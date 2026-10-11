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

from test_assistant_v3 import ALERT, HOUSE as HOUSE_ALL, FakeModel, act, acts_for, run

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
        msg = "כשהרכב יוצא מהכניסה תגיד לי"
        understand = acts_for({msg: {"emotion": "happy", "acts": [
            act("watch_request", quote=msg, camera="cam6", change="leaves")]}})
        writer = FakeModel(lambda user, kw: "לא רואה רכב בכניסה הראשית עכשיו, אולי הוא כבר יצא.")
        out = run({"time": "07:57:50", "message": {"text": msg}}, understand, writer)
        self.assertEqual(self.watches(out), [])

    def test_a_courier_is_an_event_watch_on_the_whole_house(self):
        # 2026-10-11 13:00, answering a camera 8 alert: it watched camera 8 for a car; the courier's van stood at
        # the main entrance and his men carried the boxes out, all kept as "normal", and he was never told.
        msg = "אם מגיע שליח לקחת את החבילות תעדכן אותי"
        understand = acts_for({msg: {"emotion": "neutral", "acts": [
            act("watch_request", quote=msg, camera="cam8", event="E1", subject="שליח", change="arrives")]}})
        writer = FakeModel(lambda user, kw: "סגור, אשלח לך תמונה ברגע שאראה שליח באחת המצלמות.")
        alert = dict(ALERT, id="ameer_v2_ch8_1_alert", camera="ameer_v2_ch8")
        out = run({"time": "13:00:06", "alerts": [alert],
                   "message": {"text": msg, "reply_to": alert["id"]}}, understand, writer)
        self.assertEqual(out["error"], "")
        rows = self.watches(out)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["camera"], rows[0]["change"]), ("every camera", "event"))
        self.assertIn("אשלח לך תמונה ברגע שאראה שליח, בכל מצלמה שהיא", " ".join(c["user"] for c in writer.calls))


class EventWatchTests(unittest.TestCase):
    """"אם מגיע שליח לקחת את החבילות תעדכן אותי": each event the box records is put to the judge with his words."""

    ABOUT = "אם מגיע שליח לקחת את החבילות תעדכן אותי (שליח)"

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.box = FakeBox([[]])
        self.w = store(self.box, self.tmp)
        self.asked: List[tuple] = []

        def judge(about, camera, summary):
            self.asked.append((about, camera, summary))
            return "שליח סוחב ארגז ליד הטנדר" if "box" in summary else None

        self.w.judge = judge
        self.row = self.w.add("-1", "", "", wt.EVENT, "שליח", [], about=self.ABOUT)
        self.notes: List[tuple] = []

    def event(self, summary, camera="ameer_v2_ch6", dt=120.0, sent=False):
        return self.w.on_event(camera, "כניסה ראשית", self.box.clock["now"] + dt, summary, b"jpeg", sent,
                               "ameer_v2_ch6_1_alert", lambda *a: self.notes.append(a))

    def test_the_first_matching_event_sends_its_picture_once(self):
        self.assertEqual(self.event("Two men are working near ladders."), [])
        self.assertEqual(self.box.sent, [])
        self.assertEqual(self.event("A man carries a large cardboard box towards the van."), [self.row["id"]])
        self.assertEqual(len(self.box.sent), 1)
        chat, image, caption = self.box.sent[0]
        self.assertTrue(caption.startswith("🔔 ביקשת שאעדכן: שליח סוחב ארגז"))
        self.assertIn("כניסה ראשית · ", caption)
        with open(image, "rb") as f:
            self.assertEqual(f.read(), b"jpeg")
        self.assertEqual(self.notes[0][0], "ameer_v2_ch6")
        self.assertEqual(self.notes[0][3], "ameer_v2_ch6_1_alert")
        self.assertEqual(self.w.active(), [])
        self.event("Another box is carried out.")          # one-shot
        self.assertEqual(len(self.box.sent), 1)
        self.assertEqual(self.asked[0][0], self.ABOUT)

    def test_an_event_from_before_the_ask_is_not_judged(self):
        self.event("A man carries a box.", dt=-600)
        self.assertEqual((self.asked, self.box.sent), ([], []))

    def test_an_alert_already_sent_ends_the_watch_without_a_second_message(self):
        self.assertEqual(self.event("A man carries a box out.", sent=True), [self.row["id"]])
        self.assertEqual(self.box.sent, [])
        self.assertEqual(self.w.active(), [])

    def test_a_named_camera_watches_only_that_camera(self):
        self.w.add("-2", "ameer_v2_ch8", "מצלמה 8", wt.EVENT, "", [], about="מישהו ליד החבילות")
        self.w.on_event("ameer_v2_ch6", "כניסה ראשית", self.box.clock["now"] + 60, "A man carries a box.", b"j")
        self.assertEqual([r["chat_id"] for r in self.w.active()], ["-2"])

    def test_a_failed_send_keeps_the_watch(self):
        self.box.send = lambda *a: False
        self.w.send = self.box.send
        self.event("A man carries a box.")
        self.assertEqual(len(self.w.active()), 1)

    def test_the_timer_never_checks_an_event_watch(self):
        self.box.clock["now"] += wt.CHECK_SEC * 3
        self.w.tick()
        self.assertEqual(len(self.w.active()), 1)
        self.assertEqual(self.box.sent, [])


class JudgeTests(unittest.TestCase):
    def judge(self, content, error=""):
        class Msg:
            pass

        msg = Msg()
        msg.content, msg.error = content, error
        model = type("M", (), {"chat": lambda self, *a, **k: msg})()
        return wt.model_judge(model)("שליח", "כניסה ראשית", "A man carries a box to a van.")

    def test_answers(self):
        self.assertEqual(self.judge('{"match": true, "line": "שליח לוקח ארגז"}'), "שליח לוקח ארגז")
        self.assertIsNone(self.judge('{"match": false, "line": ""}'))
        self.assertIsNone(self.judge("not json"))
        self.assertIsNone(self.judge("", error="timeout"))
        self.assertEqual(self.judge('{"match": true, "line": ""}'), "A man carries a box to a van.")


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


class PhotoAskTests(unittest.TestCase):
    """2026-10-11 10:08-10:14: "העובדים של הפרגולה הגיעו (10:07)", then four asks for a picture; the first went to a
    camera that is off, the next two were read as complaints and got "צודק, סליחה."."""

    ARRIVAL = {"id": "arrival_ameer_v2_ch3_1791702420", "camera": "ameer_v2_ch3", "time": "10:08:18", "label": "normal",
               "people": 2, "summary": "העובדים של הפרגולה הגיעו לפרגולה (10:07)"}

    def tearDown(self) -> None:
        ec.V3.clear()

    def test_a_picture_right_after_the_workers_line_is_of_the_pergola(self):
        understand = acts_for({"תן לי תמונה": {"emotion": "neutral", "acts": [
            act("command", quote="תן לי תמונה", command="send_photo")]}})
        writer = FakeModel(lambda user, kw: "הנה הפרגולה עכשיו: שני עובדים ליד הסולם.")
        out = run({"time": "10:11:34", "alerts": [self.ARRIVAL], "message": {"text": "תן לי תמונה"}},
                  understand, writer)
        self.assertEqual(out["photo_cameras"], ["ameer_v2_ch3"])

    def test_an_angry_ask_read_as_a_complaint_still_gets_a_picture(self):
        msg = "נוווווו תן לי פאקינג תמונהה"
        understand = acts_for({msg: {"emotion": "angry", "acts": [act("complaint", quote=msg, issue="other")]}})
        writer = FakeModel(lambda user, kw: "הנה, הפרגולה עכשיו.")
        out = run({"time": "10:13:14", "alerts": [self.ARRIVAL], "message": {"text": msg}}, understand, writer)
        self.assertEqual(out["photo_cameras"], ["ameer_v2_ch3"])
        self.assertNotEqual(out["text"].strip(), "צודק, סליחה.")

    def test_a_past_picture_question_is_not_a_new_picture(self):
        msg = "מה רואים בתמונה?"
        understand = acts_for({msg: {"emotion": "neutral", "acts": [act("question_history", quote=msg)]}})
        writer = FakeModel(lambda user, kw: "שני עובדים ליד הסולם.")
        out = run({"time": "10:15:00", "alerts": [self.ARRIVAL], "message": {"text": msg}}, understand, writer)
        self.assertEqual(out["photo_cameras"], [])

    def test_an_empty_big_model_falls_back_to_the_writer(self):
        from home_guard_project.box.assistant_v3.agent import AssistantV3

        msg = "מה סליחה תן לי תמונהה יא טיפש"
        understand = acts_for({msg: {"emotion": "angry", "acts": [act("complaint", quote=msg, issue="other")]}})
        writer = FakeModel(lambda user, kw: "הנה הפרגולה עכשיו: שני עובדים ליד הסולם.")
        empty = FakeModel(lambda user, kw: "")
        old = AssistantV3._escalate
        AssistantV3._escalate = lambda self, turn, trace: True
        try:
            ec.V3.clear()
            ec.V3.update(write=writer, understand=understand, critic=None, escalate=empty)
            case = {"id": "t", "source": "test", "date": "2026-10-10", "house": HOUSE_ALL, "time": "10:14:39",
                    "alerts": [self.ARRIVAL], "message": {"text": msg},
                    "expect": {"intent": "command", "ideal": "x", "max_questions": 0}}
            out = ec.run_case(case, None, None)
        finally:
            AssistantV3._escalate = old
        self.assertTrue(empty.calls)
        self.assertIn("שני עובדים", out["text"])


class ArrivalLineNotedTests(unittest.TestCase):
    def test_the_arrival_line_goes_into_the_chat_history_and_feed(self):
        from types import SimpleNamespace

        from home_guard_project.box.telegram_agent import OwnerAssistant

        noted, fed = [], []
        agent = SimpleNamespace(version=2, note_alert=lambda chat, item: noted.append((chat, item)))
        feed = SimpleNamespace(add=lambda *a, **k: fed.append((a, k)))
        assistant = OwnerAssistant(cfg=SimpleNamespace(chat_ids=["-5"]), index=None, mute=None,
                                   inbox=SimpleNamespace(agent=agent), feed=feed)
        assistant.note_line("ameer_v2_ch3", 1791702420.0, "העובדים של הפרגולה הגיעו (10:07)", "arrival_ameer_v2_ch3_1")
        self.assertEqual(noted[0][0], "-5")
        self.assertEqual((noted[0][1]["camera"], noted[0][1]["alert_id"]), ("ameer_v2_ch3", "arrival_ameer_v2_ch3_1"))
        self.assertEqual(fed[0][0][:2], ("box", "message"))


class LookSpeedTests(unittest.TestCase):
    def test_one_look_per_camera_per_turn(self):
        from types import SimpleNamespace
        from unittest import mock

        from home_guard_project.box.assistant_v3 import skills

        calls: List[str] = []

        def check_camera(ctx, args):
            calls.append(args["camera"])
            return {"ok": True, "camera": args["camera"], "description": "שקט", "people": 0}

        ctx = SimpleNamespace(text="מה עם אזור התרנגולות", snapshot=None, lang="he")
        box = skills.Toolbox(ctx, [], lambda words: "ameer_v2_ch8")
        with mock.patch.dict(skills.TOOLS, {"check_camera": check_camera}),                 mock.patch.object(skills, "_look_line", lambda c, out: "line"):
            first = box.run("look_now", {"camera": "מצלמה 8"})
            second = box.run("look_now", {"camera": "8"})
        self.assertEqual(calls, ["ameer_v2_ch8"])           # one photo, not two
        self.assertEqual(first, second)

    def test_the_cameras_are_looked_at_together(self):
        import threading
        import time as _time

        from home_guard_project.box.eval_chat import World

        seen: List[str] = []
        lock = threading.Lock()
        original = World._grab_photo

        def slow(self, camera):
            with lock:
                seen.append(threading.current_thread().name)
            _time.sleep(0.3)
            return original(self, camera)

        World._grab_photo = slow
        try:
            understand = acts_for({"מה קורה בחוץ": {"emotion": "neutral", "acts": [act("question_live", quote="מה קורה בחוץ",
                                                                                  camera="house")]}})
            writer = FakeModel(lambda user, kw: "שקט בחוץ.")
            started = _time.monotonic()
            run({"time": "11:03:54", "message": {"text": "מה קורה בחוץ"}}, understand, writer)
            took = _time.monotonic() - started
        finally:
            World._grab_photo = original
        self.assertGreaterEqual(len(seen), 3)
        self.assertLess(took, 0.3 * len(seen) - 0.2)          # not one after another
