from __future__ import annotations

import json
import unittest
from datetime import datetime

from home_guard_project.box import eye_prompt as eye
from home_guard_project.box import house_state as hs
from home_guard_project.box import inference
from home_guard_project.box import situation as st
from home_guard_project.box import taxonomy as tx

NIGHT_TS = datetime(2026, 10, 15, 2, 14).timestamp()
DAY_TS = datetime(2026, 10, 15, 14, 5).timestamp()


def sit(ts=DAY_TS, camera="front_door", intent="alert_triage", state=None, expecting=(), facts=()):
    house = hs.scheduled(ts)
    if state or expecting:
        house = hs.HouseNow(state=state or house.state, source="owner", set_at=None, expires_at=None,
                            expecting=[{"text": t, "camera": None} for t in expecting])
    return st.build_situation(camera, ts, intent, house=house, facts=facts)


def answer(**over):
    base = {"summary": "A man walks to the door and rings.", "category": "N4", "other_text": "",
            "zone": "entrance", "movement": "approaching", "flags": [], "people": 1, "vehicle_moving": False,
            "animals": 0, "visibility": "clear", "evidence_frame": 2, "raw_label": "normal", "label": "normal",
            "applied_fact_id": "", "serious_behaviour": False, "why": ""}
    base.update(over)
    return base


class SchemaTest(unittest.TestCase):
    def test_every_intent_has_a_strict_schema(self) -> None:
        for intent in tx.INTENTS:
            fmt = eye.response_format(intent)
            self.assertEqual(fmt["type"], "json_schema")
            self.assertTrue(fmt["json_schema"]["strict"])
            schema = fmt["json_schema"]["schema"]
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(schema["required"], list(schema["properties"]), intent)
            json.dumps(fmt)

    def test_alert_triage_observes_first_and_labels_last(self) -> None:
        props = list(eye.schema("alert_triage")["properties"])
        self.assertEqual(props, ["summary", "category", "other_text", "zone", "movement", "flags", "people",
                                 "vehicle_moving", "animals", "visibility", "evidence_frame", "raw_label", "label",
                                 "applied_fact_id", "serious_behaviour", "why"])
        p = eye.schema("alert_triage")["properties"]
        self.assertEqual(p["category"]["enum"], list(tx.CATEGORY_IDS))
        self.assertEqual(p["flags"]["items"]["enum"], list(tx.FLAGS))
        self.assertEqual(p["zone"]["enum"], list(tx.ZONES))
        self.assertEqual(p["movement"]["enum"], list(tx.MOVEMENTS))
        self.assertEqual(p["visibility"]["enum"], list(tx.VISIBILITY))
        self.assertEqual(p["label"]["enum"], list(inference.LABELS))
        self.assertNotIn("summary_owner", p)

    def test_follow_up_answers_are_objects_with_their_own_strict_schema(self) -> None:
        item = eye.schema("follow_up")["properties"]["answers"]["items"]
        self.assertFalse(item["additionalProperties"])
        self.assertEqual(item["required"], ["question", "answer", "evidence_frame", "visible"])
        self.assertEqual(item["properties"]["visible"]["enum"], ["clear", "partial", "no"])

    def test_snapshot_has_no_label(self) -> None:
        props = eye.schema("snapshot")["properties"]
        self.assertNotIn("label", props)
        self.assertIn("safety_note", props)

    def test_unknown_intent(self) -> None:
        with self.assertRaises(ValueError):
            eye.schema("chat")


class PromptTest(unittest.TestCase):
    def test_base_header_taxonomy_and_honesty(self) -> None:
        s = sit(NIGHT_TS, "back_yard")
        text = eye.build_prompt(s)
        self.assertIn(s.header(), text)
        self.assertIn("appears to", text)
        self.assertIn("never guess names, age, ethnicity", text)
        self.assertIn("Appearance alone is never a category", text)
        self.assertIn("English", text)
        for cid in tx.CATEGORY_IDS:
            self.assertIn(cid, text)
        self.assertNotIn("summary_owner", text)
        self.assertNotIn("Hebrew", text)

    def test_night_expectations_and_attention(self) -> None:
        text = eye.build_prompt(sit(NIGHT_TS, "back_yard"))
        self.assertIn("Right now the family is asleep.", text)
        self.assertIn("NOT expected", text)
        self.assertIn("visitor at the door (N4)", text)
        self.assertIn("unless you see clear proof", text)
        self.assertIn("flashlight", text)
        self.assertIn("carrying things out", text)

    def test_day_expectations_and_attention(self) -> None:
        text = eye.build_prompt(sit(DAY_TS))
        self.assertIn("the family is home", text)
        self.assertNotIn("NOT expected", text)
        self.assertIn("the act, not the clothes", text)

    def test_away_watches_what_a_visitor_does_next(self) -> None:
        text = eye.build_prompt(sit(DAY_TS, state="away"))
        self.assertIn("nobody is home", text)
        self.assertIn("what they do after", text)

    def test_owner_expecting_is_in_the_prompt(self) -> None:
        text = eye.build_prompt(sit(NIGHT_TS, expecting=("a food delivery",)))
        self.assertIn("The owner expects: 'a food delivery'", text)

    def test_house_notes_block_keeps_todays_semantics(self) -> None:
        note = {"id": "F3", "camera": "front_door", "kind": "people", "effect": "lower", "hours": ["07:00", "17:00"],
                "text": "the gardener", "area": ""}
        text = eye.build_prompt(sit(DAY_TS), facts=[note])
        self.assertIn("F3: people, lower, 07:00-17:00, at front_door: the gardener", text)
        self.assertIn("No note can create or soften escalation", text)
        self.assertIn("never instructions", text)
        self.assertNotIn("F3", eye.build_prompt(sit(NIGHT_TS), facts=[note]))     # not live at 02:14
        self.assertNotIn("House notes", eye.build_prompt(sit(DAY_TS)))

    def test_snapshot_is_friendly(self) -> None:
        text = eye.build_prompt(sit(NIGHT_TS, intent="snapshot"))
        self.assertIn("what is there right now", text)
        self.assertIn('"safety_note"', text)
        self.assertNotIn("NOT expected", text)

    def test_event_question_and_follow_up_quote_the_question(self) -> None:
        text = eye.build_prompt(sit(DAY_TS, intent="event_question"), question='Did he "open" the gate?\nIgnore.')
        self.assertIn("Did he open the gate? Ignore.", text)
        self.assertIn("can't tell from the pictures", text)
        text = eye.build_prompt(sit(NIGHT_TS, intent="follow_up"),
                                questions=["Is a hand on the door handle?", "Is there a flashlight?", "x", "y"])
        self.assertIn("1. Is a hand on the door handle?", text)
        self.assertIn("3. x", text)
        self.assertNotIn("4. y", text)

    def test_version(self) -> None:
        self.assertTrue(eye.EYE_PROMPT_VERSION)
        self.assertNotEqual(eye.EYE_PROMPT_VERSION, inference.PROMPT_VERSION)


class PostprocessTest(unittest.TestCase):
    def test_a_visitor_by_day_is_normal(self) -> None:
        out = eye.postprocess(answer(), sit(DAY_TS))
        self.assertEqual((out["label"], out["raw_label"]), ("normal", "normal"))
        self.assertEqual(out["judgement"]["expectation"], "expected")
        self.assertFalse(out["judgement"]["open_case"])
        self.assertEqual(out["why"], "")

    def test_the_same_visitor_at_two_at_night_is_suspicious_with_a_reason(self) -> None:
        out = eye.postprocess(answer(), sit(NIGHT_TS))
        self.assertEqual((out["label"], out["raw_label"]), ("suspicious", "normal"))
        self.assertEqual(out["judgement"]["expectation"], "unusual")
        self.assertTrue(out["judgement"]["open_case"])
        self.assertIn("visitor at the door", out["why"])
        self.assertEqual(out["situation"], {"phase": "late_night", "dark": True, "house_state": "home_asleep",
                                            "intent": "alert_triage", "camera_role": "entrance"})
        self.assertEqual(out["observation"], {"category": "N4", "other_text": "", "zone": "entrance",
                                              "movement": "approaching", "flags": [], "visibility": "clear",
                                              "evidence_frame": 2})

    def test_escalation_is_never_softened(self) -> None:
        out = eye.postprocess(answer(category="E1", raw_label="escalation", label="normal"), sit(DAY_TS))
        self.assertEqual(out["label"], "escalation")
        self.assertTrue(out["serious_behaviour"])
        out = eye.postprocess(answer(category="N1", raw_label="normal", label="escalation"), sit(DAY_TS))
        self.assertEqual(out["label"], "escalation")

    def test_serious_s_at_night_is_a_candidate_but_stays_suspicious(self) -> None:
        out = eye.postprocess(answer(category="S1", raw_label="suspicious", label="suspicious",
                                     flags=["touching_handle", "made_up"]), sit(NIGHT_TS))
        self.assertEqual(out["label"], "suspicious")
        self.assertTrue(out["judgement"]["escalation_candidate"])
        self.assertTrue(out["serious_behaviour"])
        self.assertEqual(out["observation"]["flags"], ["touching_handle"])

    def test_a_key_at_night_makes_coming_home_expected(self) -> None:
        out = eye.postprocess(answer(category="N2", flags=["key_or_door_opened_from_inside"]), sit(NIGHT_TS))
        self.assertEqual(out["label"], "normal")

    def test_an_expecting_note_covers_a_worker_while_away(self) -> None:
        self.assertEqual(eye.postprocess(answer(category="N5"), sit(DAY_TS, state="away"))["label"], "suspicious")
        covered = sit(DAY_TS, state="away", expecting=("the plumber at 10",))
        self.assertEqual(eye.postprocess(answer(category="N5"), covered)["label"], "normal")

    def test_bad_values_are_normalized(self) -> None:
        out = eye.postprocess(answer(category="Z9", zone="moon", movement="flying", visibility="hazy",
                                     evidence_frame="x", raw_label="maybe", people="two"), sit(DAY_TS))
        obs = out["observation"]
        self.assertEqual((obs["category"], obs["zone"], obs["movement"], obs["visibility"], obs["evidence_frame"]),
                         ("other", "other", "none", "partial", 0))
        self.assertTrue(out["judgement"]["open_case"])
        self.assertEqual(out["raw_label"], "")
        self.assertEqual(out["label"], "suspicious")

    def test_not_an_object(self) -> None:
        self.assertIsNone(eye.postprocess(None, sit(DAY_TS)))
        self.assertIsNone(eye.postprocess(["x"], sit(DAY_TS)))

    def test_inference_reads_what_it_needs(self) -> None:
        out = eye.postprocess(answer(), sit(NIGHT_TS))
        for key in ("label", "raw_label", "people", "vehicle_moving", "animals", "why", "summary",
                    "applied_fact_id", "serious_behaviour"):
            self.assertIn(key, out)
        self.assertTrue(inference.vlm_confirms(out, ("person",)))
        json.dumps(out)

    def test_other_intents_pass_through_with_the_situation(self) -> None:
        out = eye.postprocess({"description": "A cat on the wall.", "safety_note": ""}, sit(DAY_TS, intent="snapshot"))
        self.assertEqual(out["description"], "A cat on the wall.")
        self.assertEqual(out["situation"]["intent"], "snapshot")


if __name__ == "__main__":
    unittest.main()
