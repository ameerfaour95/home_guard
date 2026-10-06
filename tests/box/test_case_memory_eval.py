from __future__ import annotations

import unittest

from cm_helpers import OWNER, confirm_times, memory, saved_case

from home_guard_project.box.case_memory import evaluation as ev


class BoundTest(unittest.TestCase):
    def test_zero_of_n(self) -> None:
        self.assertAlmostEqual(ev.upper_bound(0, 100), 0.0295, places=4)       # "0 of 100 still allows ~3%"
        self.assertLess(ev.upper_bound(0, 300), 0.01)
        self.assertGreater(ev.upper_bound(0, 298), 0.01)
        self.assertAlmostEqual(ev.upper_bound(0, 100), 3 / 100, delta=0.002, msg="about the rule of three")

    def test_known_clopper_pearson_values(self) -> None:
        self.assertAlmostEqual(ev.upper_bound(1, 100), 0.04656, places=4)
        self.assertAlmostEqual(ev.upper_bound(5, 50), 0.19883, places=4)   # scipy beta.ppf(0.95, 6, 45)
        self.assertEqual(ev.upper_bound(7, 7), 1.0)
        self.assertEqual(ev.upper_bound(0, 0), 1.0)
        with self.assertRaises(ValueError):
            ev.upper_bound(3, 2)

    def test_bound_shrinks_with_n_and_grows_with_k(self) -> None:
        self.assertLess(ev.upper_bound(1, 200), ev.upper_bound(1, 100))
        self.assertLess(ev.upper_bound(1, 100), ev.upper_bound(2, 100))

    def test_negatives_needed(self) -> None:
        self.assertEqual(ev.negatives_needed(0.01), 299)
        self.assertEqual(ev.negatives_needed(0.03), 99)


class OperatingPointTest(unittest.TestCase):
    def test_lowest_threshold_at_target(self) -> None:
        scored = [(0.95, True), (0.9, True), (0.8, True), (0.7, True)] + [(0.85, False)] + [(None, False)] * 99
        point = ev.saved_at_false_silence(scored, target=0.01)
        self.assertEqual(point.threshold, 0.7, "1 of 100 softened is still at the 1% target")
        self.assertEqual(point.false_silence, 0.01)
        self.assertEqual(point.saved_rate, 1.0)
        self.assertFalse(point.bound_ok, "but 1 of 100 is not shown to be under 1%")
        strict = ev.saved_at_false_silence(scored, target=0.0)
        self.assertEqual((strict.threshold, strict.saved_rate), (0.9, 0.5))

    def test_no_threshold_works(self) -> None:
        point = ev.saved_at_false_silence([(0.9, True), (0.95, False)], target=0.01)
        self.assertIsNone(point.threshold)
        self.assertEqual(point.saved_rate, 0.0)

    def test_gated_negatives_never_count(self) -> None:
        scored = [(0.9, True)] + [(None, False)] * 300
        point = ev.saved_at_false_silence(scored)
        self.assertEqual((point.saved_rate, point.false_silence), (1.0, 0.0))
        self.assertTrue(point.bound_ok)


class PairsTest(unittest.TestCase):
    def test_pairs_cover_every_level(self) -> None:
        mem = memory()
        case = saved_case(mem)
        pairs = ev.make_pairs(case)
        self.assertEqual({p.level for p in pairs}, set(ev.LEVELS))
        self.assertTrue(all(p.event.signature.template for p in pairs))
        night = next(p for p in pairs if p.variant == "02:00 asleep").event.signature
        self.assertEqual((night.minute, night.phase), (120, "late_night"))

    def test_the_pipeline_never_silences_a_must_alert_pair(self) -> None:
        mem = memory()
        case = saved_case(mem)
        confirm_times(mem, case.id, 3)
        report = ev.evaluate(mem, ev.make_pairs(case))
        self.assertEqual(report["false_silence"]["softened"], 0, report["failures"])
        self.assertEqual(report["false_silence"]["pairs"], 10)
        self.assertAlmostEqual(report["false_silence"]["upper_95"], ev.upper_bound(0, 10))
        self.assertEqual(report["saved_repeats"], {"softened": 2, "pairs": 2, "rate": 1.0})
        self.assertEqual(report["allowed_matches"]["pairs"], 1)
        self.assertEqual(report["per_level"]["staged_suspicious"], {"pairs": 4, "softened": 0})
        self.assertEqual(mem.store.matches(), [], "an evaluation run writes nothing")

    def test_shadow_counts_as_would_have_silenced(self) -> None:
        mem = memory()
        case = saved_case(mem)
        report = ev.evaluate(mem, ev.make_pairs(case))
        self.assertEqual(report["saved_repeats"]["softened"], 2)
        self.assertEqual(report["false_silence"]["softened"], 0)

    def test_a_too_wide_case_and_a_careless_judge_show_up_as_false_silence(self) -> None:
        def always_same(request):
            return {"verdict": "same", "case_id": request["case_ids"][0], "matched_fields": ["path"],
                    "mismatched_fields": [], "reason": ""}

        mem = memory(judge=always_same)
        case = saved_case(mem, when="when:every_day")
        wide = mem.store.propose_widen(case.id, case.scope.__class__.from_dict(
            {**case.scope.to_dict(), "hours": ["00:00", "00:00"], "house_states": ["home_awake", "home_asleep"],
             "night": True}), OWNER)
        mem.store.answer_widen(wide["wid"], True, OWNER)
        confirm_times(mem, case.id, 3)
        report = ev.evaluate(mem, ev.make_pairs(mem.store.get(case.id)))
        self.assertEqual(report["false_silence"]["softened"], 2, "02:00 and mid-afternoon now reach the judge")
        self.assertEqual({f["level"] for f in report["failures"]}, {"wrong_hour"})
        self.assertGreater(report["false_silence"]["upper_95"], 0.4)


class BandProtectsTest(unittest.TestCase):
    def test_without_a_judge_a_far_hour_stays_in_the_middle_band(self) -> None:
        mem = memory()
        case = saved_case(mem, when="when:every_day")
        wide = mem.store.propose_widen(case.id, case.scope.__class__.from_dict(
            {**case.scope.to_dict(), "hours": ["00:00", "00:00"]}), OWNER)
        mem.store.answer_widen(wide["wid"], True, OWNER)
        confirm_times(mem, case.id, 3)
        report = ev.evaluate(mem, ev.make_pairs(mem.store.get(case.id)))
        self.assertEqual(report["false_silence"]["softened"], 0)


if __name__ == "__main__":
    unittest.main()
