from __future__ import annotations

import json
import unittest

from home_guard_project.box import inference
from home_guard_project.box import taxonomy as tx

DAY = tx.Context()
NIGHT = tx.Context(phase="late_night", house_state="home_asleep", dark=True)
EVENING_ASLEEP = tx.Context(phase="evening", house_state="home_asleep", dark=True)
AWAY = tx.Context(house_state="away")


class TaxonomyListTest(unittest.TestCase):
    def test_ids_are_fixed_and_complete(self) -> None:
        self.assertEqual([c.id for c in tx.CATEGORIES if c.group == "N"], [f"N{i}" for i in range(1, 11)])
        self.assertEqual([c.id for c in tx.CATEGORIES if c.group == "S"], [f"S{i}" for i in range(1, 10)])
        self.assertEqual([c.id for c in tx.CATEGORIES if c.group == "E"], [f"E{i}" for i in range(1, 9)])
        self.assertEqual(tx.CATEGORY_IDS[-1], "other")

    def test_labels_match_the_box(self) -> None:
        self.assertEqual(tx.LABELS, inference.LABELS)

    def test_every_category_has_text_in_both_languages(self) -> None:
        for c in tx.CATEGORIES:
            self.assertTrue(c.name and c.definition and c.he, c.id)

    def test_normalize(self) -> None:
        self.assertEqual(tx.normalize_id(" n3 "), "N3")
        self.assertEqual(tx.normalize_id("S10"), "other")
        self.assertEqual(tx.normalize_id(None), "other")
        self.assertEqual(tx.group_of("e2"), "E")
        self.assertEqual(tx.group_of("other"), "")

    def test_table_is_json(self) -> None:
        table = json.loads(json.dumps(tx.as_table(), ensure_ascii=False))
        self.assertEqual(len(table["categories"]), 28)
        self.assertEqual(table["version"], tx.TAXONOMY_VERSION)

    def test_prompt_list_names_every_id(self) -> None:
        text = tx.prompt_list()
        for cid in tx.CATEGORY_IDS:
            self.assertIn(cid, text)


class PriorsTest(unittest.TestCase):
    def test_columns(self) -> None:
        self.assertEqual(tx.column("day", "home_awake"), tx.DAY)
        self.assertEqual(tx.column("evening", "home_awake"), tx.DAY)
        self.assertEqual(tx.column("late_night", "home_awake"), tx.NIGHT)
        self.assertEqual(tx.column("day", "home_asleep"), tx.NIGHT)
        self.assertEqual(tx.column("late_night", "away"), tx.AWAY)

    def test_priors_table_from_the_plan(self) -> None:
        m = tx.priors_matrix()
        expected = {
            "N1": ("expected", "expected", "expected"),
            "N2": ("expected", "unusual", "unusual"),
            "N3": ("expected", "unusual", "expected"),
            "N4": ("expected", "unusual", "expected"),
            "N5": ("expected", "unusual", "unusual"),
            "N6": ("expected", "unusual", "unusual"),
            "N7": ("expected", "unusual", "unusual"),
            "N8": ("expected", "expected", "expected"),
            "N10": ("expected", "expected", "expected"),
            "S1": ("serious", "serious", "serious"),
            "S3": ("unusual", "serious", "serious"),
            "S9": ("serious", "serious", "serious"),
            "S4": ("unusual", "serious", "serious"),
            "E1": ("escalation", "escalation", "escalation"),
            "E8": ("escalation", "escalation", "escalation"),
        }
        for cid, cols in expected.items():
            self.assertEqual((m[cid]["day"], m[cid]["night"], m[cid]["away"]), cols, cid)

    def test_courier_before_midnight_is_fine_after_is_not(self) -> None:
        self.assertEqual(tx.expectation("N3", EVENING_ASLEEP), "expected")
        self.assertEqual(tx.expectation("N3", NIGHT), "unusual")

    def test_coming_home_at_night_needs_a_key(self) -> None:
        with_key = tx.Context(phase="late_night", house_state="home_asleep",
                              flags=("key_or_door_opened_from_inside",))
        self.assertEqual(tx.expectation("N2", with_key), "expected")
        self.assertEqual(tx.expectation("N2", NIGHT), "unusual")

    def test_expecting_or_a_fact_covers_away(self) -> None:
        self.assertEqual(tx.expectation("N5", tx.Context(house_state="away", expecting=True)), "expected")
        self.assertEqual(tx.expectation("N2", tx.Context(house_state="away", fact_covers=True)), "expected")
        self.assertEqual(tx.expectation("N5", AWAY), "unusual")

    def test_work_in_the_dark_needs_a_fact(self) -> None:
        dark_evening = tx.Context(phase="evening", dark=True)
        self.assertEqual(tx.expectation("N5", dark_evening), "unusual")
        self.assertEqual(tx.expectation("N5", tx.Context(phase="evening", dark=True, fact_covers=True)), "expected")

    def test_household_life_at_night_only_in_private_areas(self) -> None:
        self.assertEqual(tx.expectation("N6", tx.Context(phase="late_night", camera_role="private")), "expected")
        self.assertEqual(tx.expectation("N6", tx.Context(phase="late_night", camera_role="entrance")), "unusual")

    def test_vehicle_and_soldier_at_night_depend_on_movement(self) -> None:
        self.assertEqual(tx.expectation("N7", tx.Context(phase="late_night", movement="passing")), "expected")
        self.assertEqual(tx.expectation("N7", tx.Context(phase="late_night", movement="staying")), "unusual")
        self.assertEqual(tx.expectation("N9", tx.Context(phase="late_night", movement="passing")), "expected")
        self.assertEqual(tx.expectation("N9", tx.Context(phase="late_night", movement="approaching")), "unusual")

    def test_other_and_unknown_are_unusual(self) -> None:
        self.assertEqual(tx.expectation("other", DAY), "unusual")
        self.assertEqual(tx.expectation("Z9", DAY), "unusual")


class ContextualLabelTest(unittest.TestCase):
    def test_courier_by_day_is_normal_and_quiet(self) -> None:
        j = tx.contextual_label("N3", "normal", DAY)
        self.assertEqual((j.label, j.open_case, j.escalation_candidate), ("normal", False, False))

    def test_visitor_at_night_is_suspicious_and_opens_a_case(self) -> None:
        j = tx.contextual_label("N4", "normal", NIGHT)
        self.assertEqual((j.label, j.expectation, j.open_case), ("suspicious", "unusual", True))

    def test_serious_s_at_night_is_a_candidate_not_an_escalation(self) -> None:
        j = tx.contextual_label("S1", "suspicious", NIGHT)
        self.assertEqual((j.label, j.escalation_candidate), ("suspicious", True))
        self.assertFalse(tx.contextual_label("S1", "suspicious", DAY).escalation_candidate)

    def test_escalation_is_never_lowered(self) -> None:
        for ctx in (DAY, NIGHT, AWAY):
            self.assertEqual(tx.contextual_label("E3", "normal", ctx).label, "escalation")
        # The Eye's raw escalation survives a normal category (and the mismatch opens a case).
        j = tx.contextual_label("N1", "escalation", DAY)
        self.assertEqual((j.label, j.open_case), ("escalation", True))

    def test_never_below_the_raw_label(self) -> None:
        j = tx.contextual_label("N6", "suspicious", DAY)
        self.assertEqual(j.label, "suspicious")
        self.assertTrue(j.open_case)

    def test_partial_visibility_on_serious_opens_a_case(self) -> None:
        j = tx.contextual_label("S2", "suspicious", DAY, visibility="partial")
        self.assertIn("partial visibility on a serious category", j.reasons)

    def test_other_opens_a_case(self) -> None:
        j = tx.contextual_label("other", "normal", DAY)
        self.assertEqual((j.label, j.open_case), ("suspicious", True))

    def test_expectation_lines_only_show_what_changed(self) -> None:
        self.assertEqual(tx.expectation_lines(DAY), [])
        lines = tx.expectation_lines(NIGHT)
        self.assertTrue(any(line.startswith("N4 ") for line in lines))
        self.assertFalse(any(line.startswith("N1 ") for line in lines))


if __name__ == "__main__":
    unittest.main()
