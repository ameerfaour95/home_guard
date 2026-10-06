# tests/box/test_brain_house_parse.py
"""House-state commands from Telegram, in Hebrew and English, parsed in code (no model)."""
from __future__ import annotations

import datetime as dt
import unittest

from home_guard_project.box.brain.house import parse_command

EVENING = dt.datetime(2026, 10, 5, 22, 30).timestamp()      # a Monday
MORNING = dt.datetime(2026, 10, 6, 8, 0).timestamp()


def at(*args) -> float:
    return dt.datetime(*args).timestamp()


class ParseTest(unittest.TestCase):
    def kind(self, text: str, now: float = EVENING):
        cmd = parse_command(text, now)
        return cmd.kind if cmd else None

    def test_going_to_sleep(self) -> None:
        for text in ("הולכים לישון", "אנחנו הולכים לישון", "הולכת לישון 😴", "הלכנו לישון", "לילה טוב!",
                     "going to sleep", "Going to bed now", "good night", "Goodnight 🌙", "we're going to sleep"):
            with self.subTest(text=text):
                cmd = parse_command(text, EVENING)
                self.assertEqual((cmd.kind, cmd.until), ("sleep", at(2026, 10, 6, 6, 0)))   # the morning
        after_midnight = parse_command("הולכים לישון", at(2026, 10, 6, 1, 30))
        self.assertEqual(after_midnight.until, at(2026, 10, 6, 6, 0))

    def test_were_up(self) -> None:
        for text in ("קמנו", "קמנו!", "התעוררנו", "בוקר טוב", "אנחנו ערים", "we're up", "We are up!", "good morning"):
            with self.subTest(text=text):
                cmd = parse_command(text, at(2026, 10, 6, 5, 30))
                self.assertEqual((cmd.kind, cmd.until), ("up", at(2026, 10, 7, 0, 0)))    # until the night

    def test_we_left_and_were_back(self) -> None:
        for text in ("יצאנו", "יצאנו מהבית", "אנחנו יוצאים", "עזבנו את הבית", "we left", "We're leaving now",
                     "nobody's home"):
            with self.subTest(text=text):
                self.assertEqual(self.kind(text, MORNING), "left")
                self.assertIsNone(parse_command(text, MORNING).until)          # away until they say otherwise
        for text in ("חזרנו", "חזרנו הביתה", "הגענו הביתה", "אנחנו בבית", "we're back", "back home", "I'm home"):
            with self.subTest(text=text):
                self.assertEqual(self.kind(text, MORNING), "back")

    def test_vacation_until_a_date(self) -> None:
        end = at(2026, 10, 20, 23, 59)
        for text in ("אנחנו בחופשה עד 20.10", "בחופשה עד ה-20/10", "יוצאים לחופשה עד 20.10.2026",
                     "on vacation until 20.10", "We're on holiday until October 20", "on vacation till 20/10",
                     "אנחנו בחופשה עד 20 באוקטובר"):
            with self.subTest(text=text):
                cmd = parse_command(text, EVENING)
                self.assertEqual((cmd.kind, cmd.until), ("vacation", end))
        self.assertEqual(parse_command("בחופשה עד יום ראשון", EVENING).until, at(2026, 10, 11, 23, 59))
        self.assertEqual(parse_command("we left until Sunday", EVENING).kind, "vacation")
        self.assertEqual(parse_command("בחופשה עד מחר", EVENING).until, at(2026, 10, 6, 23, 59))
        self.assertEqual(parse_command("on vacation until 3.1", EVENING).until, at(2027, 1, 3, 23, 59))  # next year
        self.assertIsNone(parse_command("אנחנו בחופשה", EVENING))             # no date: the model asks

    def test_expecting(self) -> None:
        cmd = parse_command("מצפים לחבילה היום", MORNING)
        self.assertEqual((cmd.kind, cmd.note, cmd.until), ("expect", "חבילה", at(2026, 10, 6, 23, 59)))
        cmd = parse_command("שרברב מגיע ב-10", MORNING)
        self.assertEqual((cmd.kind, cmd.note, cmd.until), ("expect", "שרברב ב-10:00", at(2026, 10, 6, 13, 0)))
        cmd = parse_command("השרברב אמור להגיע בשעה 10:30", MORNING)
        self.assertEqual((cmd.note, cmd.until), ("השרברב ב-10:30", at(2026, 10, 6, 13, 30)))
        cmd = parse_command("expecting a package today", MORNING)
        self.assertEqual((cmd.kind, cmd.note, cmd.until), ("expect", "a package", at(2026, 10, 6, 23, 59)))
        cmd = parse_command("the plumber is coming at 10", MORNING)
        self.assertEqual((cmd.note, cmd.until), ("the plumber at 10:00", at(2026, 10, 6, 13, 0)))
        self.assertEqual(parse_command("מחכים לטכנאי מחר", MORNING).until, at(2026, 10, 7, 23, 59))
        late = parse_command("שרברב מגיע ב-10", EVENING)                     # 10:00 already passed: tomorrow
        self.assertEqual(late.until, at(2026, 10, 6, 13, 0))
        for text in ("מי מגיע?", "האוטו מגיע לחניה", "who is coming?", "a car comes by every day"):
            with self.subTest(text=text):
                self.assertIsNone(parse_command(text, MORNING))              # no time, a question: not a note

    def test_cancel(self) -> None:
        self.assertEqual((parse_command("בטל את החופשה", MORNING).kind, parse_command("בטל את החופשה", MORNING).what),
                         ("cancel", "state"))
        self.assertEqual(parse_command("cancel the vacation", MORNING).what, "state")
        cmd = parse_command("בטל את החבילה", MORNING)
        self.assertEqual((cmd.kind, cmd.what, cmd.words), ("cancel", "expect", "חבילה"))
        self.assertEqual(parse_command("cancel the plumber", MORNING).words, "plumber")
        for text in ("בטל", "cancel", "תבטל!"):
            with self.subTest(text=text):
                self.assertEqual(parse_command(text, MORNING).what, "")         # the last house command
        self.assertIsNone(parse_command("בטל את ההשתקה", MORNING))            # a pause, not the house
        self.assertIsNone(parse_command("cancel the pause on the gate", MORNING))

    def test_status(self) -> None:
        for text in ("מה המצב?", "מה המצב", "מה המצב בבית?", "מה מצב הבית", "סטטוס", "status", "Status?",
                     "what's the status?", "house status"):
            with self.subTest(text=text):
                self.assertEqual(self.kind(text), "status")
        self.assertIsNone(parse_command("מה המצב בכניסה?", EVENING))           # a camera question

    def test_not_commands(self) -> None:
        for text in ("הולכים לישון?", "מה קורה בכניסה", "send me a picture of the gate", "תביא לי תמונה", "",
                     None, 5, "did they go to sleep?"):
            with self.subTest(text=text):
                self.assertIsNone(parse_command(text, EVENING))

    def test_more_than_the_command_goes_on_to_the_model(self) -> None:
        self.assertTrue(parse_command("יצאנו, תשלח לי תמונה מהכניסה", MORNING).rest)
        self.assertFalse(parse_command("הולכים לישון, תודה", EVENING).rest)
        self.assertFalse(parse_command("ok we left now", MORNING).rest)


if __name__ == "__main__":
    unittest.main()
