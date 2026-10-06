"""Case memory fed with the real Eye answer (eye_prompt.postprocess), the way the guard loop will pass ``parsed``."""
from __future__ import annotations

import unittest

from cm_helpers import NORMAL, OWNER, at, confirm_times, memory

from home_guard_project.box import eye_prompt as eye
from home_guard_project.box import house_state as hs
from home_guard_project.box import situation as st
from home_guard_project.box.case_memory import ALERT, QUIET, CaseEvent, build_signature, keeper
from home_guard_project.box.case_memory import interviewer as iv

TRACKER = {"time_in_view_s": 9.0, "path": ["gate", "street"]}


def processed(ts: float, **over):
    """What inference holds after the Eye answered: eye_prompt.postprocess on a raw alert_triage answer."""
    raw = {"summary": "A person in a dark coat leaves through the gate.", "category": "N2", "other_text": "",
           "zone": "gate", "movement": "leaving", "flags": [], "people": 1, "vehicles": 0, "vehicle_moving": False,
           "animals": 0, "visibility": "clear", "appearance": ["dark coat", "backpack"], "evidence_frame": 2,
           "raw_label": "normal", "label": "normal", "applied_fact_id": "", "serious_behaviour": False, "why": ""}
    raw.update(over)
    situation = st.build_situation("gate", ts, "alert_triage", house=hs.scheduled(ts))
    return eye.postprocess(raw, situation)


def event_from(answer, event_id: str = "gate_ev", tracker=TRACKER) -> CaseEvent:
    """As INTEGRATION.md says: ``observation=parsed``, no separate situation."""
    return CaseEvent.build(event_id, "gate", at(4, 7, 40) if event_id == "gate_ev" else at(5, 7, 45),
                           observation=answer, tracker=tracker, label=answer["label"])


class SignatureFromEyeTest(unittest.TestCase):
    def test_cleaned_nested_values_win(self) -> None:
        answer = processed(at(4, 7, 40), zone=" GATE", movement="walking out", flags=["Running ", "made-up"])
        self.assertEqual(answer["zone"], " GATE", "the model's raw copy stays at the top level")
        sig = build_signature("gate", at(4, 7, 40), answer)
        self.assertEqual((sig.zone, sig.movement), ("gate", "none"))
        self.assertEqual((sig.category, sig.people, sig.vehicles), ("N2", 1, 0))
        self.assertEqual(sig.appearance, ("backpack", "dark coat"))
        self.assertIn("running", sig.flags)

    def test_situation_comes_from_the_answer_unless_given(self) -> None:
        night = processed(at(4, 2, 30))
        sig = build_signature("gate", at(4, 2, 30), night)
        self.assertEqual((sig.phase, sig.house_state), ("late_night", "home_asleep"))
        given = build_signature("gate", at(4, 2, 30), night, situation={"phase": "day", "house_state": "away"})
        self.assertEqual((given.phase, given.house_state), ("day", "away"))

    def test_veto_sees_the_union_of_both_flag_lists(self) -> None:
        answer = processed(at(5, 7, 45), flags=["touching_handle"])
        only_top = dict(answer, observation=dict(answer["observation"], flags=[]))
        only_nested = dict(answer, flags=[])
        for case in (only_top, only_nested):
            self.assertIn("touching_handle", build_signature("gate", at(5, 7, 45), case).flags)

    def test_legacy_answer_without_nested_dicts_still_builds(self) -> None:
        sig = build_signature("gate", at(4, 7, 40), {"people": 1, "label": "normal", "flags": ["crouching"]})
        self.assertEqual((sig.category, sig.flags, sig.phase), ("", ("crouching",), ""))


class EyeToMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mem = memory()
        first = event_from(processed(at(4, 7, 40)))
        interview, step = iv.start(first.event_id, first.signature)
        self.assertEqual(step.slot, "who")
        interview.answer("who:neighbour")
        step = interview.answer("when:workweek")
        self.assertEqual(step.slot, "recognise", "the Eye's appearance words reach the interview")
        self.assertIn("backpack, dark coat", step.text_he)
        interview.answer("rec:yes")
        result = keeper.save_interview(self.mem.store, interview.answer("card:save"), OWNER, embed=self.mem.embed)
        self.case = result.case
        self.assertEqual(self.case.recognise, ("backpack", "dark coat"))
        self.assertEqual(self.case.scope.house_states, ("home_awake",))

    def test_the_routine_next_day_is_softened(self) -> None:
        confirm_times(self.mem, self.case.id, 3)
        answer = processed(at(5, 7, 45))
        level, note = self.mem.apply(event_from(answer, "next"), {**NORMAL, "final_label": answer["label"]})
        self.assertEqual((level, note.kind), (QUIET, "softened"))

    def test_a_handle_flag_from_the_eye_vetoes(self) -> None:
        confirm_times(self.mem, self.case.id, 12)
        answer = processed(at(5, 7, 45), flags=["touching_handle"])
        result = self.mem.assess(event_from(answer, "next"), {**NORMAL, "final_label": answer["label"]})
        self.assertEqual(result.level, ALERT)
        self.assertIn("flags touching_handle", result.vetoed)

    def test_an_s_category_vetoes_even_with_a_normal_decision(self) -> None:
        confirm_times(self.mem, self.case.id, 12)
        answer = processed(at(5, 7, 45), category="S3", raw_label="suspicious")
        self.assertTrue(answer["serious_behaviour"] or answer["label"] != "normal")
        result = self.mem.assess(event_from(answer, "next"), NORMAL)
        self.assertEqual(result.level, ALERT)
        self.assertTrue(result.vetoed)

    def test_the_same_walk_at_night_is_gated_by_the_answers_situation(self) -> None:
        confirm_times(self.mem, self.case.id, 12)
        answer = processed(at(5, 2, 30))
        ev = CaseEvent.build("night", "gate", at(5, 2, 30), observation=answer, tracker=TRACKER,
                             label=answer["label"])
        self.assertEqual(answer["label"], "suspicious", "the priors: leaving at night without a key is unusual")
        result = self.mem.assess(ev, {**NORMAL, "final_label": answer["label"]})
        self.assertEqual((result.level, result.note), (ALERT, None))
        self.assertEqual(result.gated_out[self.case.id],
                         ["02:30 is outside 07:10-08:10", "house is home_asleep", "night is not in scope"])


if __name__ == "__main__":
    unittest.main()
