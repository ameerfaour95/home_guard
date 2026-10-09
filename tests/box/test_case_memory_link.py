"""One memory for the owner (case_memory/link.py, 2026-10-09): what he explains in the chat is kept for the week AND
learned for the long term, in shadow, through the Keeper; one question about the future days; nightly routines in
shadow. The owner's afternoon: 13:54 "the lying down on the entrance stairs is the electricians installing LEDs"."""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
import unittest

from cm_helpers import Clock, fake_embed

from home_guard_project.box import activity_memory as am
from home_guard_project.box.case_memory import (CaseEvent, CaseMemory, CaseStore, MemoryBackend, keeper, link)
from home_guard_project.box.case_memory.gates import veto
from home_guard_project.box.case_memory.models import ALL_DAYS, WORKWEEK

ENTRANCE, PERGOLA = "ameer_week_0_1_ch6", "ameer_week_0_1_ch3"
OWNER = "Ameer"
T = lambda d, h, m=0: dt.datetime(2026, 10, d, h, m).timestamp()  # noqa: E731  (9 Oct 2026 is a Friday)
RED_1325 = {"ref": f"{ENTRANCE}_1791541475_alert", "camera": ENTRANCE, "ts": T(9, 13, 25), "label": "escalation",
            "summary": "A man lies on the ground while another person in dark clothing and a cap crawls nearby, "
                       "holding a long metal bar."}
NORMAL = {"final_label": "normal", "alert_command": "[send_message]", "serious_behaviour": False}
SUSPICIOUS = {"final_label": "suspicious", "alert_command": "[send_message]", "serious_behaviour": False}


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.clock = Clock(T(9, 13, 54))
        self.store = CaseStore(MemoryBackend(), now=self.clock)
        self.memory = CaseMemory(self.store, embed=fake_embed)
        self.activities = am.ActivityBook(os.path.join(self.root, "activity.json"))

    def explain(self, until=None, daily=("07:00", "18:00"), actions=("lying", "crawling", "holding_tool"), **kw):
        """The 13:54 explanation, saved as the week-long fact (activity_chat does this from the owner's words)."""
        fact = self.activities.add([ENTRANCE], actions, "החשמלאים שמתקינים לדים", until or T(15, 18), self.clock(),
                                   place="stairs", place_words="במדרגות", daily_from=daily[0], daily_to=daily[1],
                                   owner_words="אלה מקרים תקינים הם מתקינים זה שני אנשים שמרכיבים את הלדים במדרגות",
                                   by=OWNER, **kw)
        results = link.on_fact(self.store, fact, OWNER, RED_1325, chat_id="-5", now=self.clock())
        return fact, results

    def event(self, words, ts, label="suspicious", **obs):
        observation = {"people": 2, "label": label, **obs}
        return CaseEvent.build(f"ev_{int(ts)}", ENTRANCE, ts, observation, None, None, label=label, text=words)


class CreationTest(Base):
    def test_an_explained_action_also_becomes_a_shadow_precedent(self) -> None:
        fact, (result,) = self.explain()
        case = result.case
        self.assertEqual(result.kind, "saved")
        self.assertEqual((case.camera, case.scope.hours, case.scope.weekdays), (ENTRANCE, ("07:00", "18:00"), ALL_DAYS))
        self.assertEqual(case.scope.actions, ("lying", "crawling", "holding_tool"))
        self.assertEqual((case.scope.place, case.scope.until, case.scope.people_range), ("stairs", T(15, 18), (1, 0)))
        self.assertEqual((case.status, case.confirmations, case.effect), ("shadow", 0, "quiet"))
        self.assertEqual(case.source["fact_id"], fact.id)
        self.assertEqual(case.title, "החשמלאים שמתקינים לדים")
        self.assertEqual(case.examples[0].signature.actions, ("lying", "crawling", "holding_tool"))
        self.assertEqual(case.created_by, OWNER)

    def test_a_follow_up_updates_the_same_precedent_with_history(self) -> None:
        fact, (first,) = self.explain()
        self.clock.ts = T(9, 13, 55)
        fact = self.activities.update(fact.id, self.clock(), actions=fact.actions + ["bending"])
        (again,) = link.on_fact(self.store, fact, OWNER, None, now=self.clock())
        self.assertEqual((again.kind, again.case.id), ("updated", first.case.id))
        self.assertIn("bending", again.case.scope.actions)
        self.assertEqual(again.case.history[-1]["op"], "update")
        self.assertNotIn("bending", again.case.history[-1]["before"]["actions"])
        self.assertEqual(len(self.store.cases()), 1)
        (same,) = link.on_fact(self.store, fact, OWNER, None, now=self.clock())
        self.assertEqual(same.kind, "unchanged")

    def test_moved_to_another_camera_keeps_the_old_one_as_history(self) -> None:
        fact, (first,) = self.explain()
        fact = self.activities.update(fact.id, self.clock(), cameras=[PERGOLA])
        link.on_fact(self.store, fact, OWNER, None, now=self.clock())
        old = self.store.get(first.case.id)
        self.assertEqual((old.status, old.invalid_reason), ("invalid", "the owner moved it to another camera"))
        self.assertEqual([c.camera for c in self.store.cases()], [PERGOLA])

    def test_cancel_marks_it_not_valid_since_and_keeps_it(self) -> None:
        fact, (first,) = self.explain()
        self.assertEqual(link.on_fact_cancelled(self.store, fact.id, OWNER), 1)
        self.assertEqual(self.store.cases(), [])
        self.assertEqual(self.store.cases(include_invalid=True)[0].invalid_reason, "cancelled by the owner")

    def test_a_known_mark_becomes_a_precedent_per_camera(self) -> None:
        mark = {"id": "k1", "camera": "", "text": "העובדים", "until": T(15, 18), "at": T(9, 9, 34),
                "daily_from": "07:00", "daily_to": "18:00", "people": 3}
        out = link.on_mark(self.store, mark, OWNER, [ENTRANCE, PERGOLA], now=T(9, 9, 34))
        self.assertEqual([r.kind for r in out], ["saved", "saved"])
        self.assertEqual(out[0].case.scope.people_range, (1, 5))
        self.assertEqual(out[0].case.scope.actions, ())
        # Replaced by a corrected mark (the owner's "עד 17"): updated, never a second precedent.
        fixed = dict(mark, id="k2", daily_to="17:00")
        again = link.on_mark(self.store, fixed, OWNER, [ENTRANCE, PERGOLA], replaces=["k1"], now=T(9, 10))
        self.assertEqual([r.kind for r in again], ["updated", "updated"])
        self.assertEqual({c.scope.hours for c in self.store.cases()}, {("07:00", "17:00")})
        self.assertEqual(link.on_mark_cancelled(self.store, "k2", OWNER), 2)

    def test_sync_gives_older_memories_their_precedent_once(self) -> None:
        fact = self.activities.add([ENTRANCE], ["lying"], "החשמלאים", T(15, 18), T(9, 13, 54), daily_from="07:00",
                                   daily_to="18:00", by="Hello_24")
        marks = [{"id": "k9", "camera": PERGOLA, "text": "העובדים של הפרגולה", "until": T(15, 18), "at": T(9, 9),
                  "daily_from": "07:00", "daily_to": "18:00", "by": "Hello_24"}]
        self.assertEqual(link.sync(self.store, [fact], marks, [ENTRANCE, PERGOLA], T(9, 20)), 2)
        self.assertEqual(link.sync(self.store, [fact], marks, [ENTRANCE, PERGOLA], T(9, 21)), 0)
        self.assertEqual({c.created_by for c in self.store.cases()}, {"Hello_24"})


class ShadowMatchTest(Base):
    def test_the_next_day_the_same_action_is_logged_would_quiet_and_still_alerts(self) -> None:
        self.explain()
        self.clock.ts = T(10, 10, 3)
        ev = self.event("A man lies on the stairs while another stands nearby.", T(10, 10, 3),
                        flags=["crouching"])
        with self.assertLogs("box.case_memory", level="INFO") as logs:
            level, note = self.memory.apply(ev, SUSPICIOUS)
        self.assertEqual((level, note.kind, note.would_level), ("alert", "shadow", "quiet"))
        self.assertEqual(note.owner_text("he"), "")                       # logged, never a line to the owner
        (logged,) = self.store.matches()
        self.assertEqual((logged["shadow"], logged["level"], logged["band"]), (True, "quiet", "high"))
        self.assertIn("would quiet (case C1, score 1.00)", "\n".join(logs.output))

    def test_shadow_only_never_asks_the_judge(self) -> None:
        self.explain()
        asked = []
        self.memory.judge = lambda request: asked.append(request)
        ev = self.event("A person lies on the ground.", T(10, 10, 3))
        self.assertEqual(self.memory.apply(ev, SUSPICIOUS, shadow_only=True)[0], "alert")
        self.assertEqual(asked, [])

    def test_what_an_explained_action_never_covers(self) -> None:
        self.explain()
        cases = (("A person lies on the ground, not moving.", {}),                 # harm
                 ("A man opens the car door and leans inside.", {}),              # a car door
                 ("A person lies on the ground and tries the door handle.", {"flags": ["touching_handle"]}),
                 ("A person lies on the stairs.", {"flags": ["face_covered"]}),
                 ("Two people walk by.", {}),                                     # no explained action
                 ("A person lies near the window.", {}))                           # another way in
        for words, extra in cases:
            ev = self.event(words, T(10, 10, 3), **extra)
            level, note = self.memory.apply(ev, SUSPICIOUS)
            self.assertEqual((level, note), ("alert", None), words)
        self.assertEqual(self.store.matches(), [])
        # Outside its hours, and an escalation, never.
        self.assertEqual(self.memory.apply(self.event("A person lies on the stairs.", T(10, 20)), SUSPICIOUS),
                         ("alert", None))
        red = {**SUSPICIOUS, "final_label": "escalation", "alert_command": "[call_owner]"}
        self.assertEqual(self.memory.apply(self.event("A person lies on the stairs.", T(10, 10)), red), ("alert", None))

    def test_a_red_its_context_look_lowered_is_learned_never_a_red_as_it_is(self) -> None:
        sig = self.event("A person lies on the ground.", T(10, 10), serious_behaviour=True).signature
        self.assertIn("serious behaviour", veto(sig, "suspicious", explained=("lying",)))
        self.assertEqual(veto(sig, "suspicious", explained=("lying",), context_lowered=True), [])
        self.explain()
        lowered = {**SUSPICIOUS, "serious_behaviour": True, "context_lowered": True}
        level, note = self.memory.apply(self.event("A person lies on the ground.", T(10, 10),
                                                   serious_behaviour=True), lowered)
        self.assertEqual((level, note.kind), ("alert", "shadow"))

    def test_after_its_end_the_same_event_is_seen_again_not_matched(self) -> None:
        self.explain(until=T(9, 18))
        level, note = self.memory.apply(self.event("A person lies on the stairs.", T(10, 9, 30)), SUSPICIOUS)
        self.assertEqual((level, note), ("alert", None))
        self.assertEqual([m["event_ts"] for m in self.store.seen_after_end()], [T(10, 9, 30)])


class LadderTest(Base):
    def test_three_owner_confirmations_make_it_a_quiet_message_never_more(self) -> None:
        _, (result,) = self.explain()
        case_id = result.case.id
        for i in range(3):
            self.store.confirm(case_id, OWNER, f"c{i}")
        self.assertEqual(self.store.get(case_id).status, "active")
        level, note = self.memory.apply(self.event("A person lies on the stairs.", T(10, 10)), SUSPICIOUS)
        self.assertEqual((level, note.kind), ("quiet", "softened"))
        for i in range(10):
            self.store.confirm(case_id, OWNER, f"d{i}")
        self.store.set_effect(case_id, "digest", OWNER)
        self.assertEqual(self.memory.apply(self.event("A person lies on the stairs.", T(10, 11)), SUSPICIOUS)[0],
                         "quiet", "an explained action's suspicious is at most a quiet message")

    def test_the_owner_explaining_it_again_on_another_day_is_one_confirmation(self) -> None:
        _, (first,) = self.explain()
        self.clock.ts = T(10, 9)
        other = self.activities.add([ENTRANCE], ["lying", "kneeling"], "החשמלאים", T(16, 18), self.clock(),
                                    place="stairs", daily_from="07:00", daily_to="18:00", by=OWNER)
        (again,) = link.on_fact(self.store, other, OWNER, None, now=self.clock())
        self.assertEqual((again.kind, again.case.id, again.case.confirmations), ("reinforced", first.case.id, 1))
        self.assertEqual(again.case.scope.until, T(16, 18))                # the end follows the newest
        self.assertEqual(again.case.source["fact_id"], other.id)
        (twice,) = link.on_fact(self.store, other, OWNER, None, now=T(10, 9, 5))
        self.assertEqual(self.store.get(first.case.id).confirmations, 1, "once a day")


class KeeperOpsTest(Base):
    def test_not_them_narrows_at_once_and_steps_back(self) -> None:
        _, (result,) = self.explain()
        case_id = result.case.id
        for i in range(3):
            self.store.confirm(case_id, OWNER, f"c{i}")
        sig = self.event("A person lies on the stairs.", T(10, 17, 10)).signature
        from home_guard_project.box.case_memory import Example

        out = keeper.on_button(self.store, "not_them", case_id, OWNER, "ev", example=Example("ev", sig))
        self.assertEqual(out.scope.hours, ("07:00", "17:10"))
        self.assertEqual((out.status, out.contradictions), ("shadow", 1))
        self.assertEqual(out.history[-1]["op"], "narrow")

    def test_a_widening_the_box_infers_still_waits(self) -> None:
        _, (result,) = self.explain()
        kind, proposal = keeper.correct(self.store, result.case.id, OWNER, hours=["06:00", "19:00"])
        self.assertEqual(kind, "proposed")
        self.assertEqual(self.store.get(result.case.id).scope.hours, ("07:00", "18:00"))
        kind, narrowed = keeper.correct(self.store, result.case.id, OWNER, weekdays=list(WORKWEEK))
        self.assertEqual((kind, narrowed.scope.weekdays), ("narrowed", WORKWEEK))

    def test_remember_needs_the_owner(self) -> None:
        fact = self.activities.add([ENTRANCE], ["lying"], "x", T(15, 18), self.clock())
        with self.assertRaises(ValueError):
            keeper.remember(self.store, link.precedent_from_fact(fact, ENTRANCE), "")


class QuestionTest(Base):
    def test_the_last_day_asks_once_at_18(self) -> None:
        fact, _ = self.explain()
        self.assertEqual(link.questions_due(self.store, self.activities, T(14, 18)), [])     # not the last day
        self.assertEqual(link.questions_due(self.store, self.activities, T(15, 17, 59)), [])
        (q,) = link.questions_due(self.store, self.activities, T(15, 18))
        self.assertEqual((q["kind"], q["fact_id"], q["chat_id"], q["last_day"]), ("ending", fact.id, "-5", T(15, 18)))
        self.store.note_asked(q["key"], case_id=q["case_id"], fact_id=q["fact_id"])
        self.assertEqual(link.questions_due(self.store, self.activities, T(15, 18, 5)), [], "once")

    def test_a_one_day_explanation_asks_only_when_it_comes_back(self) -> None:
        fact, _ = self.explain(until=T(9, 18), daily=("", ""))
        self.assertEqual(link.questions_due(self.store, self.activities, T(9, 18)), [])
        self.memory.apply(self.event("A person lies on the stairs.", T(10, 9, 30)), SUSPICIOUS)
        (q,) = link.questions_due(self.store, self.activities, T(10, 9, 35))
        self.assertEqual((q["kind"], q["seen_at"]), ("again", T(10, 9, 30)))

    def test_another_week(self) -> None:
        fact, (result,) = self.explain()
        (q,) = link.questions_due(self.store, self.activities, T(15, 18))
        self.store.note_asked(q["key"], case_id=q["case_id"], fact_id=q["fact_id"])
        out = link.answer(self.store, self.activities, q["key"], link.WEEK, OWNER, T(15, 18, 2))
        self.assertEqual((out["kind"], out["until"]), ("week", T(22, 18)))
        self.assertEqual(self.activities.get(fact.id).until, T(22, 18))
        case = self.store.get(result.case.id)
        self.assertEqual((case.scope.until, case.confirmations), (T(22, 18), 1))
        self.assertFalse(link.answer(self.store, self.activities, q["key"], link.WEEK, OWNER, T(15, 18, 3))["ok"])

    def test_it_is_over(self) -> None:
        fact, (result,) = self.explain()
        (q,) = link.questions_due(self.store, self.activities, T(15, 18))
        self.store.note_asked(q["key"], case_id=q["case_id"], fact_id=q["fact_id"])
        self.clock.ts = T(15, 17)
        out = link.answer(self.store, self.activities, q["key"], link.ENDED, OWNER, T(15, 17))
        self.assertEqual(out["kind"], "ended")
        self.assertEqual(self.activities.live(T(15, 17, 1)), [])
        self.assertEqual(self.store.cases(include_invalid=True)[0].invalid_reason, "ended (the owner)")

    def test_standing_every_weekday(self) -> None:
        fact, (result,) = self.explain()
        (q,) = link.questions_due(self.store, self.activities, T(15, 18))
        self.store.note_asked(q["key"], case_id=q["case_id"], fact_id=q["fact_id"])
        out = link.answer(self.store, self.activities, q["key"], link.STANDING, OWNER, T(15, 18, 1))
        case = out["case"]
        self.assertEqual((case.scope.until, case.scope.weekdays, case.scope.hours), (None, WORKWEEK, ("07:00", "18:00")))
        self.assertEqual((case.confirmations, case.status), (1, "shadow"))
        self.assertEqual(self.activities.get(fact.id).until, T(15, 18), "the week-long explanation keeps its end")
        # After it the precedent still matches on a weekday, in shadow until two more confirmations.
        level, note = self.memory.apply(self.event("A person lies on the stairs.", T(18, 10)), SUSPICIOUS)
        self.assertEqual((level, note.kind), ("alert", "shadow"))
        self.assertEqual(self.memory.apply(self.event("A person lies on the stairs.", T(17, 10)), SUSPICIOUS),
                         ("alert", None), "Saturday is not a weekday")


class RoutineTest(Base):
    def write_events(self, days):
        path = os.path.join(self.root, "events.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for d in days:
                ts = dt.datetime(2026, 10, d, 7, 40).timestamp()
                f.write(json.dumps({"id": f"s{d}", "camera": "gate", "opened": ts, "last_active": ts + 30,
                                    "observations": [{"ts": ts, "alert_id": f"a{d}", "label": "normal",
                                                      "people": 1, "summary": "A man walks out."}]}) + "\n")
        return path

    def test_routines_are_off_by_default_and_shadow_logs_once(self) -> None:
        self.assertEqual(link.routine_mode({}), "off")
        self.assertEqual(link.routine_mode({"routine_proposals": "shadow"}), "shadow")
        self.assertEqual(link.routine_mode({"routine_proposals": True}), "on")
        self.assertEqual(link.routine_mode({"routine_proposals": "maybe"}), "off")
        path = self.write_events([1, 2, 4, 5, 6, 7, 8])
        self.assertEqual(link.nightly_routines(self.store, path, "off", T(9, 3)), [])
        (item,) = link.nightly_routines(self.store, path, "shadow", T(9, 3))
        self.assertEqual((item["rid"], item["proposal"]["camera"], item["proposal"]["minute"]), ("R1", "gate", 460))
        self.assertEqual(link.nightly_routines(self.store, path, "shadow", T(10, 3)), [], "once")
        self.assertEqual(self.store.cases(), [], "a proposal changes nothing")

    def test_an_answered_proposal_makes_a_shadow_case(self) -> None:
        path = self.write_events([1, 2, 4, 5, 6, 7, 8])
        (item,) = link.nightly_routines(self.store, path, "on", T(9, 3))
        case = link.answer_routine(self.store, item["rid"], True, OWNER)
        self.assertEqual((case.status, case.source["origin"]), ("shadow", "routine"))
        self.assertIsNone(link.answer_routine(self.store, item["rid"], True, OWNER), "once")


if __name__ == "__main__":
    unittest.main()
