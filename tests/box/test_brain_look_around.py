"""2026-10-08 live replay of the owner's chat through the stage-1 assistant (real gpt-4o): three fixes.
- "יש משהו מעניין בחוץ?" with no camera got "על איזו מצלמה?": look_around looks at every camera.
- "מדי פעם אני (עמיר) יוצא החוצה" was saved as known people on the entrance: a routine is not mark_known.
- "הוא עומד ליד הרכב בחניון, נראה שהוא מדבר עם הידיים" was saved as known people: a description is not either."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from typing import Any, List

from home_guard_project.box.brain.memory import ChatState
from home_guard_project.box.brain.receipts import DONE, ReceiptBook
from home_guard_project.box.brain.registry import CameraState, HouseSnapshot
from home_guard_project.box.brain.tools import (
    Services, ToolContext, identifies_people, is_routine, look_around, mark_known,
)
from home_guard_project.box.events import EventBook

NOW = 1_791_355_961.0
PERGOLA, ENTRANCE, BACK = "ameer_week_0_1_ch3", "ameer_week_0_1_ch6", "ameer_week_0_1_ch8"


def house() -> HouseSnapshot:
    return HouseSnapshot(now=NOW, mode="guard", mode_ends=NOW + 3600, mode_started=NOW - 3600, start_hour=0,
                         end_hour=0, cameras=(CameraState(PERGOLA, True, ("פרגולה",), live=True),
                                              CameraState(ENTRANCE, True, ("כניסה ראשית",), live=True),
                                              CameraState(BACK, False, (), live=True)))


class Vision:
    def __init__(self, people: dict) -> None:
        self.people = people

    def look(self, camera, images, guard=False):
        n = self.people.get(camera, 0)
        return {"ok": True, "description": f"{n} people" if n else "empty yard", "quality": "clear", "people": n,
                "label": "normal", "why": ""}


class Deliver:
    def __init__(self) -> None:
        self.photos: List[Any] = []

    def photo(self, chat_id, path, caption=""):
        self.photos.append(path)
        return {"ok": True, "message_id": len(self.photos)}


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.events = EventBook(os.path.join(self.root, "events"))
        self.deliver = Deliver()
        self.grabbed: List[str] = []

    def grab(self, camera: str):
        self.grabbed.append(camera)
        path = os.path.join(self.root, f"{camera}.jpg")
        with open(path, "wb") as f:
            f.write(b"\xff\xd8jpeg")
        return {"image": path}

    def ctx(self, text: str, vision=None, alert: bool = False) -> ToolContext:
        services = Services(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"),
                            feedback_dir=self.root, work_dir=os.path.join(self.root, ".live"), mute=None,
                            deliver=self.deliver, read_settings=lambda: {"owner_language": "he"}, now=lambda: NOW,
                            events=self.events, grab_photo=self.grab, vision=vision)
        state = ChatState()
        ctx = ToolContext(turn_id="-5:1", chat_id="-5", speaker={"user_id": 1, "name": "Ameer"}, text=text,
                          lang="he", mode="guard", snapshot=house(), state=state, services=services,
                          book=ReceiptBook(os.path.join(self.root, ".r"), now=lambda: NOW), threaded=alert)
        if alert:
            ctx.alert_handle = state.add_handle("event", f"{ENTRANCE}_1791439138_alert", ENTRANCE, NOW - 60,
                                                "A man stands near a car and raises his hands.")
        return ctx


class LookAroundTest(Base):
    def test_every_camera_that_is_on_is_looked_at_and_only_people_get_a_photo(self) -> None:
        out = look_around(self.ctx("יש משהו מעניין בחוץ?", Vision({PERGOLA: 3})), {})
        self.assertTrue(out["ok"])
        self.assertEqual(self.grabbed, [PERGOLA, ENTRANCE])                 # ch8 is off
        self.assertEqual(len(self.deliver.photos), 1)                       # only the pergola had people
        names = [row["camera"] for row in out["cameras"]]
        self.assertEqual(names, ["פרגולה", "כניסה ראשית"])                 # names, never ids
        self.assertNotIn("ameer_", repr(out))

    def test_no_vision_is_said_per_camera(self) -> None:
        out = look_around(self.ctx("מה קורה מסביב לבית"), {})
        self.assertFalse(out["ok"])                                  # no picture is not "all quiet"
        self.assertIn("never say it is quiet", out["error"])
        self.assertTrue(all("error" in row for row in out["cameras"]))
        self.assertEqual(self.deliver.photos, [])


class KnownGuardTest(Base):
    def test_routine_is_not_saved(self) -> None:
        text = "מדי פעם אני (עמיר) יוצא החוצה"
        self.assertTrue(is_routine(text))
        out = mark_known(self.ctx(text), {"who": "עמיר", "owner_words": "אני (עמיר)", "camera": "כניסה ראשית"})
        self.assertFalse(out["ok"])
        self.assertEqual(self.events.list_known(NOW), [])

    def test_description_is_not_saved(self) -> None:
        text = "הוא עומד ליד הרכב בחניון, נראה שהוא מדבר עם הידיים"
        self.assertFalse(identifies_people(text))
        out = mark_known(self.ctx(text, alert=True), {"who": "אדם מדבר עם הידיים", "owner_words": "מדבר עם הידיים"})
        self.assertFalse(out["ok"])
        self.assertEqual(self.events.list_known(NOW), [])

    def test_who_they_are_is_saved(self) -> None:
        for text in ("זה בסדר זה עובדים אצלי שעובדים על הפרגולה", "זה אחד אנשים שעובדים מחוץ לבית אמרתי לך כבר",
                     "זה אני", "these are my workers"):
            self.assertTrue(identifies_people(text), text)
        out = mark_known(self.ctx("זה אני", alert=True), {"who": "עמיר", "owner_words": "זה אני"})
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["status"], DONE)


if __name__ == "__main__":
    unittest.main()
