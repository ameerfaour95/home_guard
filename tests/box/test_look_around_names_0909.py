"""2026-10-09 12:47: the owner wrote "מה קורה"; the bot sent 2 photos, then "במצלמה 1 יש אדם... במצלמה 3 יש אדם...
במצלמה 6 יש שני אנשים... / ✓ התמונה נשלחה (מצלמה 1) / ✓ התמונה נשלחה (פרגולה)".

- Cameras 3 and 6 have names (פרגולה, כניסה ראשית): every outgoing text says the name, never "מצלמה N".
- A photo that went out in the same turn needs no "✓ התמונה נשלחה" line under the answer.
- Each photo's caption is the camera's name and the time.
"""

from __future__ import annotations

import datetime as dt
import unittest
from typing import Any, List
from unittest import mock

import test_brain_agent as ba
import test_brain_look_around as bl
from test_brain_agent import Scripted, call, reply
from test_brain_look_around import ENTRANCE, NOW, PERGOLA, Vision

from home_guard_project.box.brain.receipts import DONE, FAILED, Receipt
from home_guard_project.box.brain.style import clean_outgoing, name_numbered_cameras
from home_guard_project.box.brain.tools import look_around

GARDEN = "ameer_week_0_1_ch1"
NAMES = {PERGOLA: ["פרגולה"], ENTRANCE: ["כניסה ראשית"]}
SAID = ("במצלמה 1 יש אדם שעומד בחצר ליד קיר נמוך. במצלמה 3 יש אדם בחולצה לבנה שעובר בחניה ליד רכב כסוף. "
        "במצלמה 6 יש שני אנשים עומדים ליד רכב לבן עם תא מטען פתוח.")


def names(_: Any = None) -> dict:
    return dict(NAMES)


class NumberedCameraGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch("home_guard_project.box.camera_names._load", side_effect=names)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_answer_of_12_47_names_the_pergola_and_the_entrance(self) -> None:
        out = clean_outgoing(SAID, [GARDEN, PERGOLA, ENTRANCE])
        self.assertIn("במצלמה 1 יש אדם", out)               # no name: the number stays
        self.assertIn("בפרגולה יש אדם בחולצה לבנה", out)
        self.assertIn("בכניסה ראשית יש שני אנשים", out)
        self.assertNotIn("מצלמה 3", out)
        self.assertNotIn("מצלמה 6", out)

    def test_without_the_camera_list_the_named_ids_still_count(self) -> None:
        self.assertEqual(name_numbered_cameras("מצלמה 3 ריקה."), "פרגולה ריקה.")

    def test_prefixes_and_english(self) -> None:
        with mock.patch("home_guard_project.box.camera_names._load", return_value={PERGOLA: ["הפרגולה"]}):
            self.assertEqual(name_numbered_cameras("במצלמה 3 ובמצלמה 33", [PERGOLA]), "בפרגולה ובמצלמה 33")
            self.assertEqual(name_numbered_cameras("המצלמה 3", [PERGOLA]), "הפרגולה")
        self.assertEqual(name_numbered_cameras("Camera 6 sees two people; camera 2 is empty.", [ENTRANCE]),
                         "כניסה ראשית sees two people; camera 2 is empty.")

    def test_two_sites_naming_one_channel_differently_change_nothing(self) -> None:
        with mock.patch("home_guard_project.box.camera_names._load",
                        return_value={PERGOLA: ["פרגולה"], "ameer_tes2_ch3": ["מחסן"]}):
            self.assertEqual(name_numbered_cameras("במצלמה 3 יש אדם."), "במצלמה 3 יש אדם.")


class Deliver:
    def __init__(self) -> None:
        self.photos: List[Any] = []

    def photo(self, chat_id, path, caption=""):
        self.photos.append((path, caption))
        return {"ok": True, "message_id": len(self.photos)}


class CaptionTest(unittest.TestCase):
    setUp, grab, ctx = bl.Base.setUp, bl.Base.grab, bl.Base.ctx

    def test_each_look_around_photo_is_captioned_with_the_name_and_the_time(self) -> None:
        self.deliver = Deliver()
        look_around(self.ctx("מה קורה", vision=Vision({PERGOLA: 1, ENTRANCE: 2})), {})
        clock = dt.datetime.fromtimestamp(NOW).strftime("%H:%M")
        self.assertEqual([c for _, c in self.deliver.photos], [f"פרגולה · {clock}", f"כניסה ראשית · {clock}"])


class NoPhotoReceiptTest(unittest.TestCase):
    setUp, agent = ba.AgentTest.setUp, ba.AgentTest.agent

    def look(self, ctx, name, args):
        from home_guard_project.box.brain.tools import _issue, _result  # noqa: PLC0415

        if name == "look_around":
            _issue(ctx, "check_camera", DONE, PERGOLA, {"camera": PERGOLA, "message_id": 1})
            _issue(ctx, "check_camera", FAILED, ENTRANCE, {"camera": ENTRANCE}, "telegram")
            return {"ok": True, "cameras": []}
        return _result(_issue(ctx, name, DONE, ""))

    def test_a_photo_sent_in_the_turn_gets_no_receipt_line_but_a_failure_does(self) -> None:
        big = Scripted([call("look_around"), reply("בפרגולה יש אדם.")])
        out = self.agent(big, run_tool=self.look).handle("מה קורה", "-5", {"user_id": 1})
        lines = out.text.split("\n")
        self.assertEqual(lines[0], "בפרגולה יש אדם.")
        self.assertFalse(any(line.startswith("✓") for line in lines))
        self.assertEqual(len(lines), 2)                      # the failed photo is still said

    def test_without_an_answer_the_receipt_is_the_reply(self) -> None:
        def only_photo(ctx, name, args):
            from home_guard_project.box.brain.tools import _issue  # noqa: PLC0415

            _issue(ctx, "check_camera", DONE, PERGOLA, {"camera": PERGOLA, "message_id": 1})
            return {"ok": True}

        big = Scripted([call("look_around"), reply("")])
        out = self.agent(big, run_tool=only_photo).handle("תמונה מהפרגולה", "-5", {"user_id": 1})
        self.assertTrue(out.text.startswith("✓"))


if __name__ == "__main__":
    unittest.main()
