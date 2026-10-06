from __future__ import annotations

import time
import unittest
from dataclasses import replace

from cm_helpers import (NORMAL, OWNER, SUSPICIOUS, at, confirm_times, event, memory, observation, saved_case,
                        tracker)

from home_guard_project.box.case_memory import (ALERT, DIGEST, QUIET, CaseMemoryConfig, apply_case_memory,
                                                build_signature, configure)
from home_guard_project.box.case_memory.gates import (gate_failures, hours_around, in_window, path_distance,
                                                      veto)
from home_guard_project.box.case_memory.scorer import combine, rank, time_score

STRICT_HIGH = CaseMemoryConfig(high=0.999)       # everything that passes the gates is judged


class SignatureTest(unittest.TestCase):
    def test_template_is_fixed_and_ignores_free_text(self) -> None:
        a = build_signature("gate", at(4, 7, 40), observation(summary="A man in a dark jacket strolls out"),
                            tracker(), {"phase": "day"})
        b = build_signature("gate", at(5, 7, 45), observation(summary="Person exits via the gate"),
                            tracker(), {"phase": "day"})
        self.assertEqual(a.template, b.template)
        self.assertEqual(a.template, "camera=gate; category=N2 coming home or leaving; zone=gate; path=gate>street; "
                                     "movement=leaving; people=1; vehicles=0; animals=0; "
                                     "appearance=backpack, dark jacket; dwell=short")

    def test_body_words_never_kept(self) -> None:
        sig = build_signature("gate", at(4, 7, 40), observation(appearance=["tall man", "beard", "Red Cap", "red cap",
                                                                            "white van"]))
        self.assertEqual(sig.appearance, ("red cap", "white van"))

    def test_path_cleanup_and_edges(self) -> None:
        sig = build_signature("gate", at(4, 7, 40), observation(), {"path": "Gate>gate>street", "time_in_view_s": 7})
        self.assertEqual(sig.path, ("gate", "street"))
        self.assertEqual((sig.entry_edge, sig.exit_edge), ("gate", "street"))

    def test_todays_gpt4o_answer_still_builds(self) -> None:
        sig = build_signature("gate", at(4, 7, 40), {"people": 1, "label": "normal", "vehicle_moving": True})
        self.assertEqual((sig.category, sig.path, sig.vehicles, sig.dwell_s), ("", (), 1, None))


class VetoTest(unittest.TestCase):
    def check(self, expect: bool, **kw) -> None:
        sig = event(**kw).signature
        self.assertEqual(bool(veto(sig)), expect, kw)

    def test_veto(self) -> None:
        self.check(False)
        self.check(True, obs=observation(category="S1"))
        self.check(True, obs=observation(category="E3"))
        self.check(True, obs=observation(flags=["touching_handle"]))
        self.check(True, obs=observation(flags=["face_covered"]))
        self.check(False, obs=observation(flags=["uniform_or_helmet"]))
        self.check(True, obs=observation(serious_behaviour=True))
        self.check(True, label="escalation")
        self.check(True, cameras_in_incident=2, situation={"phase": "late_night", "house_state": "home_asleep"})
        self.check(True, cameras_in_incident=3, situation={"phase": "day", "house_state": "away"})
        self.check(False, cameras_in_incident=2, situation={"phase": "day", "house_state": "home_awake"})


class GatesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mem = memory()
        self.case = saved_case(self.mem)

    def fails(self, ev) -> list:
        return gate_failures(self.case, ev.signature)

    def test_the_same_situation_passes(self) -> None:
        self.assertEqual(self.fails(event(ts=at(5, 7, 50))), [])

    def test_each_gate(self) -> None:
        self.assertTrue(self.fails(event(camera="yard")))
        self.assertTrue(self.fails(event(ts=at(5, 8, 30))), "outside 07:10-08:10 + 15 min")
        self.assertEqual(self.fails(event(ts=at(5, 8, 25))), [], "inside the margin")
        self.assertTrue(self.fails(event(ts=at(10, 7, 40))), "Saturday is not in scope")
        self.assertTrue(self.fails(event(situation={"phase": "day", "house_state": "away"})))
        self.assertTrue(self.fails(event(situation={"phase": "late_night", "house_state": "home_awake"})))
        self.assertTrue(self.fails(event(obs=observation(people=2))))
        self.assertTrue(self.fails(event(obs=observation(vehicle_moving=True))))
        self.assertTrue(self.fails(event(obs=observation(category="N4"))), "door family is not transit")
        self.assertEqual(self.fails(event(obs=observation(category="N1"))), [], "N1 and N2 are one family")
        self.assertTrue(self.fails(event(obs=observation(category="other"))))
        self.assertTrue(self.fails(event(trk=tracker(path=["gate", "entrance"]))))
        self.assertTrue(self.fails(event(trk=tracker(path=["street", "gate"]))))
        self.assertTrue(self.fails(event(trk={})), "no tracker path, no match")
        self.assertTrue(self.fails(event(trk=tracker(time_in_view_s=45))))

    def test_a_category_from_a_new_eye_is_required_once_the_case_has_one(self) -> None:
        self.assertTrue(self.fails(event(obs={k: v for k, v in observation().items() if k != "category"})))

    def test_window_helpers(self) -> None:
        self.assertEqual(hours_around(7 * 60 + 40), ("07:10", "08:10"))
        self.assertTrue(in_window(23 * 60 + 55, ("23:30", "00:30")))
        self.assertTrue(in_window(10, ("23:30", "00:05"), margin=15))
        self.assertFalse(in_window(30, ("23:30", "00:05"), margin=15))
        self.assertAlmostEqual(path_distance(("gate", "yard", "street"), ("gate", "street")), 1 / 3)


class ScorerTest(unittest.TestCase):
    def test_closest_example_wins_and_missing_parts_are_left_out(self) -> None:
        mem = memory()
        case = saved_case(mem)
        later = replace(case.examples[0], event_id="late",
                        signature=event(ts=at(5, 8, 5)).signature)
        case = replace(case, examples=case.examples + [later])
        sig = event(ts=at(6, 8, 5)).signature
        (_, detail), = rank([case], sig, mem.embed(sig.template))
        self.assertEqual(detail.example_id, "late")
        self.assertNotIn("appearance", detail.components, "the owner said the clothes change")
        self.assertGreater(detail.score, 0.95)

    def test_time_and_combine(self) -> None:
        self.assertAlmostEqual(time_score(100, 100), 1.0)
        self.assertAlmostEqual(time_score(10, 1430), time_score(0, 20), "circular")
        self.assertAlmostEqual(combine({"path": 1.0, "time": 0.0}), 0.35 / 0.55)
        self.assertEqual(combine({}), 0.0)


class PolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mem = memory()
        self.case = saved_case(self.mem)

    def apply(self, ev=None, decision=NORMAL):
        return self.mem.apply(ev or event(ts=at(5, 7, 45), event_id="next"), decision)

    def test_shadow_alerts_logs_and_asks(self) -> None:
        level, note = self.apply()
        self.assertEqual(level, ALERT)
        self.assertEqual(note.kind, "shadow")
        self.assertEqual(note.would_level, QUIET)
        self.assertIn("בפעם הבאה לא אתריע", note.text_he)
        self.assertEqual([b["action"] for b in note.buttons], ["confirm", "not_them", "keep_alerting"])
        logged = self.mem.store.matches(self.case.id)
        self.assertEqual((logged[0]["event_id"], logged[0]["shadow"]), ("next", True))
        self.assertEqual(self.mem.store.get(self.case.id).confirmations, 0, "a match is not a confirmation")

    def test_trust_ladder_quiet_then_digest(self) -> None:
        confirm_times(self.mem, self.case.id, 3)
        level, note = self.apply()
        self.assertEqual((level, note.kind), (QUIET, "softened"))
        self.assertIn("רגיל (לפי ההסבר שלך מ-4.10)", note.text_he)
        self.assertEqual([b["action"] for b in note.buttons], ["not_them"])
        confirm_times(self.mem, self.case.id, 7)
        level, note = self.apply()
        self.assertEqual(level, DIGEST)
        self.assertTrue(note.digest_he.startswith("07:45 gate: רגיל"))

    def test_owner_ceiling_quiet_never_reaches_digest(self) -> None:
        self.mem.store.set_effect(self.case.id, QUIET, OWNER)
        confirm_times(self.mem, self.case.id, 12)
        self.assertEqual(self.apply()[0], QUIET)

    def test_keep_alerting_is_context_only(self) -> None:
        confirm_times(self.mem, self.case.id, 12)
        self.mem.store.keep_alerting(self.case.id, OWNER)
        level, note = self.apply()
        self.assertEqual((level, note.kind), (ALERT, "keep_alerting"))

    def test_calls_and_escalations_are_never_touched(self) -> None:
        confirm_times(self.mem, self.case.id, 12)
        self.assertEqual(self.apply(decision={"final_label": "escalation", "alert_command": "[call_owner]"}),
                         (ALERT, None))
        self.assertEqual(self.apply(decision={"final_label": "normal", "alert_command": "[call_owner]"}),
                         (ALERT, None))
        self.assertEqual(self.mem.store.matches(), [], "memory was not even consulted")

    def test_suspicious_signs_always_win(self) -> None:
        confirm_times(self.mem, self.case.id, 12)
        for obs in (observation(flags=["touching_handle"]), observation(category="S2"),
                    observation(serious_behaviour=True)):
            self.assertEqual(self.apply(event(ts=at(5, 7, 45), obs=obs)), (ALERT, None))
        self.assertEqual(self.apply(decision={**NORMAL, "serious_behaviour": True}), (ALERT, None))

    def test_a_suspicious_label_gets_context_but_is_never_softened(self) -> None:
        confirm_times(self.mem, self.case.id, 12)
        level, note = self.apply(decision=SUSPICIOUS)
        self.assertEqual((level, note.kind), (ALERT, "context"))
        self.assertIn("ההתרעה בתוקף", note.text_he)

    def test_gated_out_is_the_normal_flow(self) -> None:
        confirm_times(self.mem, self.case.id, 12)
        result = self.mem.assess(event(ts=at(5, 7, 45), obs=observation(people=2)), NORMAL)
        self.assertEqual((result.level, result.note), (ALERT, None))
        self.assertIn("2 people, the case has 1", result.gated_out[self.case.id])

    def test_high_band_needs_a_path(self) -> None:
        mem = memory(judge=None)
        case = saved_case(mem, ev=event(trk={}, obs=observation(category="")))   # learned on today's gpt-4o
        confirm_times(mem, case.id, 12)
        result = mem.assess(event(ts=at(5, 7, 45), trk={}, obs=observation(category="")), NORMAL)
        self.assertEqual((result.band, result.level), ("mid", ALERT), "no tracker: judged, and no judge alerts")

    def test_low_band_is_the_normal_flow(self) -> None:
        config = CaseMemoryConfig(high=0.99, mid=0.98)
        mem = memory(config=config)
        case = saved_case(mem)
        confirm_times(mem, case.id, 12)
        result = mem.assess(event(ts=at(5, 8, 20)), NORMAL)
        self.assertEqual((result.band, result.level, result.note), ("low", ALERT, None))

    def test_failure_inside_means_alert(self) -> None:
        def broken(_):
            raise RuntimeError("disk")

        self.mem.store.live_cases = broken
        with self.assertLogs("box.case_memory", "WARNING"):
            self.assertEqual(self.apply(), (ALERT, None))

    def test_module_entry_point(self) -> None:
        configure(None)
        self.assertEqual(apply_case_memory(event(), NORMAL), (ALERT, None))
        confirm_times(self.mem, self.case.id, 3)
        configure(self.mem)
        try:
            self.assertEqual(apply_case_memory({"event_id": "x", "camera": "gate", "ts": at(5, 7, 45),
                                                "observation": observation(), "tracker": tracker(),
                                                "situation": {"phase": "day"}}, NORMAL)[0], QUIET)
        finally:
            configure(None)


class JudgeBandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.calls = []
        self.answer = {}

        def judge(request):
            self.calls.append(request)
            if isinstance(self.answer, Exception):
                raise self.answer
            if callable(self.answer):
                return self.answer(request)
            return self.answer

        self.mem = memory(judge=judge, config=STRICT_HIGH)
        self.case = saved_case(self.mem)
        confirm_times(self.mem, self.case.id, 3)

    def run_it(self, **kw):
        return self.mem.assess(event(ts=at(5, 7, 45), event_id="next", **kw), NORMAL)

    def test_same_lowers_one_step(self) -> None:
        self.answer = {"verdict": "same", "case_id": self.case.id, "matched_fields": ["path", "hour"],
                       "mismatched_fields": [], "reason": "same walk"}
        result = self.run_it()
        self.assertEqual((result.band, result.level, result.matched), ("mid", QUIET, self.case.id))
        self.assertEqual(self.mem.store.matches()[0]["verdict"], "same")

    def test_similar_but_different_alerts_with_context(self) -> None:
        self.answer = {"verdict": "similar_but_different", "case_id": self.case.id, "matched_fields": ["hour"],
                       "mismatched_fields": ["dwell"], "reason": "stayed longer"}
        result = self.run_it(trk=tracker(time_in_view_s=17))
        self.assertEqual((result.level, result.note.kind), (ALERT, "similar"))
        self.assertEqual(result.note.text_he, "נראה כמו שכן, אבל הפעם שהה 17 שניות")
        self.assertEqual(result.note.text_en, "Looks like a neighbour, but this time stayed 17 seconds")

    def test_unsure_invalid_and_error_alert(self) -> None:
        bad = [
            {"verdict": "unsure", "case_id": "", "matched_fields": [], "mismatched_fields": [], "reason": ""},
            {"verdict": "same", "case_id": "C99", "matched_fields": ["path"], "mismatched_fields": [], "reason": ""},
            "not json at all",
            RuntimeError("429"),
        ]
        for answer in bad:
            self.answer = answer
            result = self.run_it()
            self.assertEqual((result.level, result.note), (ALERT, None), answer)

    def test_timeout_alerts(self) -> None:
        mem = memory(judge=lambda request: time.sleep(1.0), config=CaseMemoryConfig(high=0.999, judge_timeout=0.05))
        case = saved_case(mem)
        confirm_times(mem, case.id, 3)
        result = mem.assess(event(ts=at(5, 7, 45)), NORMAL)
        self.assertEqual((result.level, result.verdict.error), (ALERT, "timeout"))

    def test_few_cases_send_all_gated_cases_many_send_three(self) -> None:
        self.answer = {"verdict": "unsure", "case_id": "", "matched_fields": [], "mismatched_fields": [], "reason": ""}
        for minute in (35, 50, 55):
            self.mem.store.add(replace(self.case, scope=replace(self.case.scope, people=1),
                                       note=f"twin {minute}"), OWNER)
        self.run_it()
        self.assertEqual(len(self.calls[-1]["case_ids"]), 4)
        small = replace(self.mem.config, few_cases=2)
        self.mem.config = small
        self.run_it()
        self.assertEqual(len(self.calls[-1]["case_ids"]), 3)


if __name__ == "__main__":
    unittest.main()
