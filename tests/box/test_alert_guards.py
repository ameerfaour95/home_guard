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

    def test_a_long_or_blunt_thing_is_a_tool_weapon(self):
        # 2026-10-09 13:25 ch6, the owner's pavers workers: a red for "a long metal bar" had no second look.
        for text in ("A man lies on the ground while another person crawls nearby, holding a long metal bar.",
                     "התנהגות חריגה: אדם נמצא על הקרקע ואדם אחר מחזיק מוט מתכתי ארוך לידו.",
                     "a man with a crowbar at the door", "holding a metal pipe", "a man with a wooden stick",
                     "carrying an axe", "raises a shovel", "a wooden plank in his hand", "a club", "two iron rods",
                     "a long pole", "a baseball bat", "אוחז אלה", "מחזיק פטיש", "מחזיק מחבט", "עם צינור", "גרזן",
                     "את חפירה", "קרש עץ", "עם לום", "מקל ארוך"):
            with self.subTest(text=text):
                self.assertEqual(g.verify_classes(text), ["tool_weapon"])

    def test_words_that_only_look_like_a_tool(self):
        for text in ("a man climbs the drain pipe", "sticks his hand through the gate", "near the light pole",
                     "cuts the bars on the window", "a man lies on the ground", "אדם מוטל על הקרקע", "אומר שלום",
                     "לא עושה כלום", "נכנס למקלחת", "ליד המקלט", "האנשים האלה מסתובבים", "שאלה", "robotic arm"):
            with self.subTest(text=text):
                self.assertNotIn("tool_weapon", g.verify_classes(text))

    def test_a_red_that_names_the_act_gets_no_tool_question(self):
        # eval_set_v2 2026-10-09: asked the tool question, the 9B said "no" on 11 of 14 real tool alerts; a red that
        # says what was done goes on as before.
        for text in ("swings a baseball bat at the door", "hits the window with a hammer", "a man hits another man with a pipe",
                     "smashing a window with a metal bar", "prying the door with a crowbar", "threatening with a bat",
                     "attacking a man with a stick", "loading a crowbar into a van", "takes a hammer and runs away",
                     "cuts the lock with a saw and a bar", "using a crowbar on the car door", "throws a brick and a bar",
                     "מכה במקל", "מניף מחבט", "תוקף עם מוט", "שובר את החלון עם פטיש", "גונב פטיש"):
            with self.subTest(text=text):
                self.assertNotIn("tool_weapon", g.verify_classes(text))
        self.assertEqual(g.verify_classes("a man hits another man with a pipe"), ["violence"])

    def test_the_tool_must_be_the_reason(self):
        summary = "Two people load bags into a van; one holds a metal bar."
        self.assertEqual(g.verify_classes(f"suspicious night visit {summary}", reason="suspicious night visit"), [])
        self.assertEqual(g.verify_classes(f"holding a metal bar near a person {summary}",
                                          reason="holding a metal bar near a person"), [])   # "load" in the summary
        live_why = "התנהגות חריגה: אדם נמצא על הקרקע ואדם אחר מחזיק מוט מתכתי ארוך לידו."
        live_summary = "A man lies on the ground while another person crawls nearby, holding a long metal bar."
        self.assertEqual(g.verify_classes(f"{live_why}  {live_summary}", reason=f"{live_why} "), ["tool_weapon"])
        self.assertEqual(g.verify_classes(f"  {live_summary}", reason="  "), ["tool_weapon"])  # no why: the summary

    def test_a_gun_and_a_bar_get_both_questions(self):
        self.assertEqual(g.verify_classes("a man with a knife and a metal pipe"), ["weapon", "tool_weapon"])
        question = g.verify_question(["weapon", "tool_weapon"])
        self.assertIn(g.VERIFY_QUESTIONS["weapon"], question)
        self.assertIn(g.VERIFY_QUESTIONS["tool_weapon"], question)
        self.assertIn("threaten or hit a person", g.VERIFY_QUESTIONS["tool_weapon"])
        self.assertIn("as work is NOT", g.VERIFY_QUESTIONS["tool_weapon"])

    def test_a_clear_class_still_goes_out_at_once(self):
        for text in ("forcing the front door with a crowbar", "a man lying motionless next to a bat"):
            with self.subTest(text=text):
                self.assertEqual(g.verify_classes(text), [])

    def test_a_tool_s_no_names_the_tool_without_contradicting_itself(self):
        for what in ("a worker laying pavers with a metal bar", "פועל עובד עם מוט מתכת על הקרקע",
                     "a metal bar used on the ground, not to threaten anyone", "מוט ברזל, ללא איום",
                     "a man breaking paving stones with a hammer", ""):
            with self.subTest(what=what):
                self.assertFalse(g.answer_names(["tool_weapon"], what))

    def test_a_tool_s_no_that_names_the_act_contradicts_itself(self):
        for what in ("swinging the bar at another man", "hitting a person with a pipe", "threatening with a bat",
                     "prying the car door with a crowbar", "מאיים עם מוט", "מכה אדם במקל", "פורץ את הדלת עם לום",
                     "two men fighting with sticks", "a gun"):
            with self.subTest(what=what):
                self.assertTrue(g.answer_names(["tool_weapon"], what))

    def test_the_other_classes_keep_their_own_contradiction_rule(self):
        self.assertTrue(g.answer_names(["violence"], "a physical fight between two men"))
        self.assertFalse(g.answer_names(["weapon"], "מוט ארוך"))
        self.assertTrue(g.answer_names(["weapon", "tool_weapon"], "a man swinging a pipe at a woman"))

    def test_several_classes_one_question(self):
        self.assertEqual(g.verify_classes("a man with a knife smashes the car window"), ["weapon", "vehicle"])
        question = g.verify_question(["weapon", "vehicle"])
        self.assertIn("ANY", question)
        self.assertIn(g.VERIFY_QUESTIONS["weapon"], question)
        self.assertEqual(g.verify_question(["violence"]), g.VERIFY_QUESTIONS["violence"])
        self.assertIn("NOT weapons", g.VERIFY_QUESTIONS["weapon"])

    # 2026-10-09 14:03 ch6: the owner's pavers workers went out red for "a person lies on the ground".
    DOWN_WHY = "אדם שוכב על הקרקע"
    DOWN_SUMMARY = "A person lies on the ground while two others stand nearby. One person appears to be wearing a hat."

    def test_a_person_down_gets_its_own_question(self):
        self.assertEqual(g.verify_classes(f"{self.DOWN_WHY}  {self.DOWN_SUMMARY}", reason=f"{self.DOWN_WHY} "),
                         ["person_down"])
        for text in ("a man in a mask kneels on the ground", "two people crouching", "one lying on the pavement",
                     "a person is on the ground", "a worker on his knees", "squatting in the yard", "אדם כורע על הקרקע",
                     "פועל רכון", "גבר על הרצפה"):
            with self.subTest(text=text):
                self.assertEqual(g.verify_classes(text), ["person_down"])
        question = g.VERIFY_QUESTIONS["person_down"]
        for words in ("hurt, collapsed, unconscious", "attacked or held down", "to WORK", "laying tiles or pavers", "is NOT"):
            self.assertIn(words, question)

    def test_a_person_down_red_that_names_more_keeps_today_s_way(self):
        # Lying motionless / unconscious is CLEAR (out at once); violence, a weapon, a break-in, a way in, hiding, night,
        # a fall or an injury, a child, a theft or a robbery around it: no person-down question.
        for text in ("a man lying motionless", "a person unconscious on the floor", "גבר שוכב ללא תנועה",
                     "a person fell and lies on the ground", "a man is beaten while lying on the ground",
                     "a man crouching by the car", "crouching near the window at night", "a child lying on the grass",
                     "a man lies on the floor bleeding", "squatting at the door", "kneeling and picking the lock",
                     "a person lying on the floor while another takes the cash", "two men kneel on top of a man, holding him down",
                     "אדם שוכב על הקרקע ואחר מכה אותו", "גבר נפל ושוכב על הקרקע", "ילד שוכב על הדשא",
                     "throwing items on the floor"):
            with self.subTest(text=text):
                self.assertNotIn("person_down", g.verify_classes(text))
        self.assertEqual(g.verify_classes("a man with a knife kneels"), ["weapon"])
        self.assertEqual(g.verify_classes("two men fight, one is on the ground"), ["violence"])

    def test_the_person_down_must_be_the_reason(self):
        self.assertEqual(g.verify_classes(f"wearing masks {self.DOWN_SUMMARY}", reason="wearing masks"), [])

    def test_a_person_down_s_no_names_the_ground_without_contradicting_itself(self):
        for what in ("a worker kneeling on the pavement laying pavers", "פועל כורע על הקרקע ומניח אבנים",
                     "a person lying on the ground fixing a pipe", "not hurt, working on the ground",
                     "a worker hitting pavers with a rubber mallet", ""):
            with self.subTest(what=what):
                self.assertFalse(g.answer_names(["person_down"], what))
        for what in ("a man collapsed on the ground", "a person being held down", "an injured man", "a man fell",
                     "two men fighting on the ground", "אדם פצוע על הקרקע", "גבר התמוטט"):
            with self.subTest(what=what):
                self.assertTrue(g.answer_names(["person_down"], what))

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
