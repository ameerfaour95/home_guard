"""The guards on the model's label (stage 1, 2026-10-08): appearance is not suspicious; which red needs a second look."""
import unittest

from home_guard_project.box import alert_guards as g
from home_guard_project.box import alert_texts
from home_guard_project.box import inference as inf


class AppearanceOnlyTest(unittest.TestCase):
    def test_real_reasons_from_the_owner_s_two_days_are_appearance_only(self):
        for why in ("wearing a mask", "a man in a hoodie with a covered face", "face hidden by a hood",
                    "dark clothes and a hat", "wearing sunglasses and a cap", "the face is pixelated",
                    "hiding his face", "עובד עם מסכה", "גבר רעול פנים עם כובע", "לובש קפוצ'ון", "פנים מכוסות",
                    "בגדים כהים ומשקפי שמש", "הפנים מפוקסלות"):
            with self.subTest(why=why):
                self.assertTrue(g.appearance_only(why))

    def test_an_action_keeps_the_suspicious(self):
        for why in ("a masked man tries the door handle", "hooded man looks into the windows",
                    "a man in a mask climbs the fence", "hiding behind the car with a hood",
                    "masked person takes a package and leaves", "a hooded man walks around at night",
                    "masked man approaching the entrance", "covers the camera with a cloth while masked",
                    "גבר עם מסכה מציץ לחלון", "רעול פנים מנסה את הידית", "גבר בקפוצ'ון לוקח חבילה", "מסכה בלילה"):
            with self.subTest(why=why):
                self.assertFalse(g.appearance_only(why))

    def test_nothing_to_judge(self):
        for why in ("", "a man walks to the door", "that is all", "standing for a long time"):
            self.assertFalse(g.appearance_only(why), why)

    def test_inference_exposes_the_guard(self):
        self.assertTrue(inf.appearance_only("wearing a hoodie"))


class VerifyClassTest(unittest.TestCase):
    def test_the_three_classes(self):
        cases = {"man holds a possible weapon": "weapon", "a long tool that could be a rifle": "weapon",
                 "אוחז בסכין": "weapon", "a man smashed the car window": "vehicle", "breaking into a car": "vehicle",
                 "ניפץ את חלון הרכב": "vehicle", "two men fight in the yard": "violence",
                 "a man hits another man": "violence", "תוקף אדם": "violence"}
        for text, want in cases.items():
            with self.subTest(text=text):
                self.assertEqual(g.verify_class(text), want)

    def test_clear_classes_go_out_at_once(self):
        for text in ("a man breaks into the house with a knife", "fire and smoke at the gate",
                     "a person lying motionless", "climbing in through the window", "forcing the front door",
                     "האש מתפשטת בחצר", "פורץ לבית"):
            with self.subTest(text=text):
                self.assertTrue(g.clear_class(text))
                self.assertIsNone(g.verify_class(text))

    def test_no_second_look_for_other_reds(self):
        for text in ("a man breaks the window", "hits the window with a hammer", "steals a bicycle",
                     "a man gets out of his car", "ראש"):
            with self.subTest(text=text):
                self.assertIsNone(g.verify_class(text))

    def test_several_classes_one_question(self):
        self.assertEqual(g.verify_classes("a man with a knife smashes the car window"), ["weapon", "vehicle"])
        question = g.verify_question(["weapon", "vehicle"])
        self.assertIn("ANY", question)
        self.assertIn(g.VERIFY_QUESTIONS["weapon"], question)
        self.assertEqual(g.verify_question(["violence"]), g.VERIFY_QUESTIONS["violence"])
        self.assertIn("NOT weapons", g.VERIFY_QUESTIONS["weapon"])

    def test_the_prompt_asks_for_strict_json(self):
        prompt = g.verify_prompt("Is it a gun?", 4)
        for part in ("Is it a gun?", '"confirmed"', '"what_it_is"', '"evidence_frame"', "1 to 4"):
            self.assertIn(part, prompt)


class AlertTextsTest(unittest.TestCase):
    def test_more_people(self):
        self.assertEqual(alert_texts.more_people(3, "he"), "עוד 3 אנשים הגיעו")
        self.assertEqual(alert_texts.more_people(1, "he"), "עוד אדם אחד הגיע")
        self.assertEqual(alert_texts.more_people(2, "en"), "2 more people arrived")
        self.assertEqual(alert_texts.more_people(1, "en"), "1 more person arrived")

    def test_second_look(self):
        self.assertEqual(alert_texts.second_look("weapon", "מוט ארוך", 3, "he"), "בדקתי שוב: מוט ארוך (פריים 3), לא נשק")
        self.assertEqual(alert_texts.second_look("vehicle", "a man leaving his car", 0, "en"),
                         "Second look: a man leaving his car, not a car break-in")
        self.assertEqual(alert_texts.second_look("violence", "", "x", "en"), "Second look: not violence")


if __name__ == "__main__":
    unittest.main()
