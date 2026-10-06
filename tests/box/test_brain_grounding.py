# tests/box/test_brain_grounding.py
from __future__ import annotations

import unittest

from home_guard_project.box.brain.grounding import ungrounded_details


class GroundingTest(unittest.TestCase):
    """A visual detail in an answer must come from the observation or a vision answer (§10)."""

    def test_details_not_in_the_evidence_are_flagged(self) -> None:
        self.assertEqual(ungrounded_details("The car is red.", "A car parks at the gate."), ["red"])
        self.assertEqual(ungrounded_details("המכונית אדומה.", "A car parks at the gate."), ["red"])
        self.assertEqual(ungrounded_details("He was holding a knife.", "A man walks to the door."), ["knife"])
        self.assertEqual(ungrounded_details("הוא לבש קפוצ'ון שחור", "A man walks."), ["black", "hoodie"])
        self.assertEqual(ungrounded_details("The plate is 12-345-67.", "A white car."), ["plate 12-345-67"])

    def test_details_in_the_evidence_pass_in_any_language(self) -> None:
        self.assertEqual(ungrounded_details("White.", "A white car parks at the gate."), [])
        self.assertEqual(ungrounded_details("המכונית לבנה.", "A white car parks at the gate."), [])
        self.assertEqual(ungrounded_details("בטלפון", 'asked "what was in his hand?": A phone.'), [])
        self.assertEqual(ungrounded_details("12-345-67", "plate 12-345-67"), [])

    def test_answers_without_visual_details_pass(self) -> None:
        for text in ("There was one event at 22:00.", "The box keeps clips for 14 days.", "היו שני אירועים היום.",
                     "The picture is blacked out outside the zone.", "", None):
            with self.subTest(text=text):
                self.assertEqual(ungrounded_details(text, ""), [])


if __name__ == "__main__":
    unittest.main()
