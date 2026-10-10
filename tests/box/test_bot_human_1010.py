# tests/box/test_bot_human_1010.py
"""The owner's chat of 2026-10-09 / 2026-10-10: the bot leaked its bookkeeping, asked "until when?" about the
neighbour's house, dumped its memory, talked about the workers out of nowhere, asked generic questions and answered
a 402 in English. Models are scripted; nothing reaches a network."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, List
from unittest import mock

from home_guard_project.box.agent import AgentContext, OwnerAgent, UNAVAILABLE_REPLY, unavailable_reply
from home_guard_project.box.brain.models import no_ai_access
from home_guard_project.box.brain import human
from home_guard_project.box.brain.agent import OwnerAgentV2
from home_guard_project.box.brain.memory import ChatMemory
from home_guard_project.box.brain.models import ModelMessage, ToolCall
from home_guard_project.box.brain.receipts import ReceiptBook
from home_guard_project.box.brain.registry import CameraState, HouseSnapshot
from home_guard_project.box.brain.style import clean_outgoing, strip_internals
from home_guard_project.box.brain.tools import Services
from home_guard_project.box.camera_profiles import CameraProfiles
from home_guard_project.box.events import Decision, EventBook
from home_guard_project.box.place_facts import place_statement, quiet_place
from home_guard_project.box.scene_map import Track

LEAK_0909 = ("במצלמת הפרגולה יש אדם אחד שנראה עובד או זז ליד הטריילר וחומרי הבניין. התמונה ברורה.\n"
             "✓ התמונה נשלחה (פרגולה)\n"
             "[handles: E30=photo פרגולה Fri 09 Oct 18:41 | receipts: R1 check_camera פרגולה done]")


class InternalsGuardTests(unittest.TestCase):
    def test_the_0909_leak_is_stripped(self) -> None:
        out = clean_outgoing(LEAK_0909, [], "he")
        self.assertEqual(out, "במצלמת הפרגולה יש אדם אחד שנראה עובד או זז ליד הטריילר וחומרי הבניין.")
        for word in ("handles", "receipts", "R1", "check_camera", "✓", "התמונה ברורה", "E30"):
            self.assertNotIn(word, out)

    def test_bookkeeping_lines_and_inline_brackets(self) -> None:
        self.assertEqual(strip_internals("כן, יש שני אנשים. [receipts: R1 check_camera x done]"), "כן, יש שני אנשים.")
        self.assertEqual(strip_internals("שלום\nreceipts: R2 send_media done\nR3 look_around הכל done"), "שלום")
        self.assertEqual(strip_internals("The picture is clear. Two men by the gate."), "Two men by the gate.")

    def test_a_photo_receipt_alone_stays(self) -> None:
        self.assertEqual(strip_internals("✓ התמונה נשלחה (פרגולה)"), "✓ התמונה נשלחה (פרגולה)")

    def test_plain_text_is_untouched(self) -> None:
        text = "הבנתי, זה הבית של השכן.\nמה שקורה אצלו לא יגיע אליך."
        self.assertEqual(strip_internals(text), text)



class Refused402(Exception):
    pass


ERR_402 = Refused402("Error code: 402 - {'error': {'message': 'This request requires more credits, or fewer "
                     "max_tokens.'}}")
ERR_429 = Refused402("Error code: 429 - {'error': {'message': 'You have no credits remaining.', 'type': "
                     "'insufficient_quota'}}")


class FailureTextTests(unittest.TestCase):
    def test_402_and_quota_are_no_access(self) -> None:
        self.assertTrue(no_ai_access(ERR_402))
        self.assertTrue(no_ai_access(ERR_429))
        self.assertFalse(no_ai_access(TimeoutError("read timed out")))

    def test_hebrew_and_honest(self) -> None:
        self.assertEqual(unavailable_reply("he", ERR_402), "אין לי כרגע גישה ל-AI, אני בודק ומעדכן.")
        self.assertEqual(unavailable_reply("he"), "לא הצלחתי לטפל בזה כרגע, אבל ההודעה שלך נשמרה.")
        self.assertNotEqual(unavailable_reply("he", ERR_402), UNAVAILABLE_REPLY)

    def test_v1_agent_answers_a_402_in_hebrew(self) -> None:
        import tempfile  # noqa: PLC0415

        from home_guard_project.box.feedback import MuteState  # noqa: PLC0415

        class Broke:
            def chat(self, *a, **k):
                raise ERR_429

        root = tempfile.mkdtemp()
        ctx = AgentContext(camera_names=["ameer_v2_ch6"], mute_state=MuteState(root + "/mute.json"),
                           feedback_dir=root, roots=lambda: [root], embedder=False, look_now=lambda c: {},
                           set_camera=lambda c, a: {}, conversations_dir=root + "/conv", language=lambda: "he")
        reply = OwnerAgent(Broke(), ctx).handle("מה קורה הם יש משהו מחוץ לכניסה הראשית", "-1", {}, None)
        self.assertEqual(reply.text, "אין לי כרגע גישה ל-AI, אני בודק ומעדכן.")

    def test_detector_only_summary_is_in_the_box_language(self) -> None:
        from home_guard_project.box.inference import DETECTED_ONLY, owner_summary  # noqa: PLC0415

        self.assertEqual(owner_summary(DETECTED_ONLY, "", "he"), "זוהה אדם או רכב")
        self.assertEqual(owner_summary(DETECTED_ONLY, "", "en"), DETECTED_ONLY)


class NoAccessRegexTests(unittest.TestCase):
    def test_a_bare_402_is_no_access_but_not_a_number_inside_a_word(self) -> None:
        self.assertTrue(no_ai_access(Exception("Error code: 402")))
        self.assertFalse(no_ai_access(Exception("read 4021 bytes")))


# ---------- today's chat, through the brain (scripted models) ----------
CH2, CH3, CH6 = "ameer_v2_ch2", "ameer_v2_ch3", "ameer_v2_ch6"
T = lambda h, m=0, s=0: dt.datetime(2026, 10, 10, h, m, s).timestamp()  # noqa: E731
ALERT_ID = "ameer_v2_ch2_1791628521_alert"
ALERT = {"alert_id": ALERT_ID, "camera": CH2, "label": "suspicious", "ts": T(13, 35, 21),
         "summary": "A person wearing a white hat and dark clothing walks along the wall and appears to look into a "
                    "window."}
TRACKS = {"camera": CH2, "tracks": [{"id": 9, "kind": "person", "entity": "P1", "boxes": [
    {"frame": 18, "ts": T(13, 35, 20), "box": [0.0813, 0.5517, 0.1228, 0.7298]},
    {"frame": 24, "ts": T(13, 35, 21), "box": [0.1208, 0.5896, 0.1843, 0.7863]}]}]}
OWNER = {"user_id": 1, "name": "Hello_24"}
CONFIRM = "הבנתי, זה הבית של השכן. מה שקורה אצלו לא יגיע אליך, רק אם מישהו עובר לשטח שלך."
ROBOTIC = ("מה לתקן?", "מה תרצה לשנות או להוסיף?", "במה אוכל לעזור?", "הבנתי אותך.", "עד מתי")


def call(name: str, **args: Any) -> ModelMessage:
    return ModelMessage(tool_calls=(ToolCall(id=f"c_{name}", name=name, arguments=args),), usage=(10, 2))


class Scripted:
    def __init__(self, responses: List[ModelMessage]) -> None:
        self.responses, self.model_name, self.seen = list(responses), "big", []

    def chat(self, messages, tools, tool_choice=None):
        self.seen.append([m.get("content") for m in messages])
        if not self.responses:
            raise ConnectionError("script ended")
        return self.responses.pop(0)


class Registry:
    def __init__(self, clock) -> None:
        self.clock = clock

    def snapshot(self):
        now = self.clock()
        return HouseSnapshot(now=now, mode="guard", mode_ends=now + 3600, mode_started=now - 3600, start_hour=0,
                             end_hour=0, cameras=(CameraState(CH2, True, (), live=True),
                                                  CameraState(CH3, True, ("פרגולה",), live=True),
                                                  CameraState(CH6, True, ("כניסה ראשית",), live=True)))


class TodayChatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.clock = T(13, 48)
        self.events = EventBook(os.path.join(self.root, "events"))
        for cam in (CH3, CH6):           # the pergola workers, as carried over the rename (memory_rename)
            self.events.mark_known(cam, "העובדים של הפרגולה", "Hello_24", T(15, 18) + 5 * 86400, now=T(9, 9),
                                   daily_from="07:00", daily_to="18:00")
        day = os.path.join(self.root, "responses", CH2, "2026-10-10")
        os.makedirs(day)
        with open(os.path.join(day, ALERT_ID + ".tracks.json"), "w", encoding="utf-8") as f:
            json.dump(TRACKS, f)
        self.profiles = CameraProfiles(os.path.join(self.root, "events", "camera_profiles.json"))

    def agent(self, big: Scripted) -> OwnerAgentV2:
        now = lambda: self.clock  # noqa: E731
        services = Services(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"),
                            feedback_dir=self.root, work_dir=os.path.join(self.root, ".live"), mute=None,
                            deliver=None, read_settings=lambda: {"owner_language": "he"}, now=now, events=self.events)
        return OwnerAgentV2(big, Registry(now), ChatMemory(os.path.join(self.root, ".conv")),
                            ReceiptBook(os.path.join(self.root, ".receipts"), now=now), services, now=now)

    def say(self, agent, text, at, alert=None):
        self.clock = at
        return agent.handle(text, "-5326761586", OWNER, dict(alert) if alert else None, False)

    def test_the_whole_afternoon(self) -> None:
        # The models answer the way they did on the box: the memory status and robotic questions.
        big = Scripted([call("reply", answer="העובדים של הפרגולה כבר מסומנים בפרגולה ובכניסה ראשית עד 18:00. מה לתקן?"),
                        call("reply", answer="העובדים של הפרגולה כבר מסומנים בפרגולה ובכניסה ראשית עד 18:00. מה לתקן?")])
        agent = self.agent(big)
        self.clock = T(13, 36, 2)
        agent.note_alert("-5326761586", dict(ALERT))
        out = self.say(agent, "זה הבית של השכן", T(13, 48), ALERT)
        self.assertEqual(out.text, CONFIRM)
        (place,) = self.profiles.places(CH2)
        self.assertEqual((place["text"], place["owner"]), ("הבית של השכן", "neighbour"))
        self.assertEqual(place["region"], [0.0, 0.4317, 0.3043, 0.9063])          # P1's boxes, widened
        self.assertEqual(self.events.list_known(T(13, 48))[-1]["text"], "העובדים של הפרגולה")   # no new mark
        self.assertEqual(len(self.events.list_known(T(13, 48))), 2)

        lines = [out.text]
        lines.append(self.say(agent, "מה קשר ? לא הבנתי", T(13, 49, 34)).text)
        sorry = self.say(agent, "מה הקשררררררר העבדים של הפרגולה יא חתכית מטומטם", T(13, 49, 55)).text
        self.assertEqual(sorry, "סליחה, טעיתי. זה לא קשור לעובדים. רשמתי: הבית של השכן במצלמה 2.")
        keep = self.say(agent, "הזכרון שלך שמור אצלך אתה לא צריך לחשוף לי אותו", T(15, 14, 8)).text
        self.assertEqual(keep, "צודק. מה ששמור אצלי נשאר אצלי, ואפרט רק אם תשאל.")
        ask = self.say(agent, "אתה יכול לשאול בצורה דרך אגב אבל לא לחשוף", T(15, 14, 23)).text
        self.assertEqual(ask, "סגור. אם אצטרך לברר משהו אשאל בקצרה, בלי לפרט מה שמור אצלי.")
        again = self.say(agent, "מה הקקשרררררר עובדייים יא מטומטם", T(15, 14, 30)).text
        self.assertEqual(again, "צודק, אני מתנצל. העובדים לא קשורים לזה בכלל. כל מה שרשמתי הוא שהבית של השכן "
                                "במצלמה 2.")
        lines += [sorry, keep, ask, again]
        for line in lines:
            for bad in ROBOTIC + ("שמור אצלי עכשיו", "מסומנים"):
                self.assertNotIn(bad, line)
        for line in (lines[1], keep, ask):
            self.assertNotIn("עובדים", line)
            self.assertNotIn("העובדים", line)

    def test_the_memory_is_shown_when_he_asks(self) -> None:
        out = self.say(self.agent(Scripted([])), "מה אתה זוכר?", T(13, 50))
        self.assertIn("העובדים", out.text)

    def test_a_place_is_never_saved_as_people(self) -> None:
        from home_guard_project.box.brain.tools import ToolContext, mark_known  # noqa: PLC0415

        agent = self.agent(Scripted([]))
        ctx = ToolContext(turn_id="t", chat_id="c", speaker=OWNER, text="זה הבית של השכן", lang="he",
                          mode="guard", snapshot=Registry(lambda: self.clock).snapshot(), state=agent.memory.load("c"),
                          services=agent.services, book=agent.book)
        result = mark_known(ctx, {"who": "הבית של השכן", "owner_words": "השכן", "camera": CH2})
        self.assertFalse(result["ok"])
        self.assertIn("PLACE", result["error"])
        self.assertIsNone(ctx.clarification)


class PlaceStatementTests(unittest.TestCase):
    def test_places(self) -> None:
        for text, words, owner in (("זה הבית של השכן", "הבית של השכן", "neighbour"),
                                   ("זה השטח של השכן", "השטח של השכן", "neighbour"),
                                   ("זו החצר של השכן", "החצר של השכן", "neighbour"),
                                   ("זה הרחוב", "הרחוב", "public"),
                                   ("זה המחסן שלנו", "המחסן שלנו", "mine"),
                                   ("לא, זה בבית של השכן", "הבית של השכן", "neighbour"),
                                   ("זה אצל השכן", "אצל השכן", "neighbour")):
            got = place_statement(text)
            self.assertEqual((got or {}).get("words"), words, text)
            self.assertEqual(got["owner"], owner, text)

    def test_not_places(self) -> None:
        for text in ("זה השכן", "זה הבית של השכן?", "מה קשר ? לא הבנתי", "זה העובדים של הפרגולה", "זה תקין",
                     "זה הדוור", "זה אני", "זה השכן עד שש"):
            self.assertIsNone(place_statement(text), text)


class NeighbourPlaceGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.profiles = CameraProfiles(os.path.join(self.root, "camera_profiles.json"))
        self.profiles.add_place(CH2, "הבית של השכן", "neighbour", [0.0, 0.43, 0.30, 0.91], alert_id=ALERT_ID,
                                now=T(13, 48))

    def test_staying_at_the_neighbours_is_quiet(self) -> None:
        along_the_wall = Track("person", [(1.0, 0.10, 0.73), (2.0, 0.15, 0.79), (3.0, 0.22, 0.80)])
        place = quiet_place(CH2, [along_the_wall], self.profiles)
        self.assertEqual(place["text"], "הבית של השכן")

    def test_crossing_onto_our_ground_goes_out(self) -> None:
        crossing = Track("person", [(1.0, 0.10, 0.73), (2.0, 0.25, 0.80), (3.0, 0.55, 0.85), (4.0, 0.75, 0.80)])
        self.assertIsNone(quiet_place(CH2, [crossing], self.profiles))
        other_camera = Track("person", [(1.0, 0.10, 0.73)])
        self.assertIsNone(quiet_place(CH3, [other_camera], self.profiles))

    def test_the_guard_loop_keeps_it_quiet_and_says_why(self) -> None:
        from home_guard_project.box import inference as inf  # noqa: PLC0415

        job = mock.Mock(tracker_tracks=[Track("person", [(1.0, 0.10, 0.73), (2.0, 0.15, 0.79)])])
        event = Decision(notify=True, session_id="s1", reason="suspicious")
        decision: dict = {}
        events = mock.Mock(directory=self.root)
        with mock.patch.object(inf, "EVENTS", events), \
                mock.patch("home_guard_project.box.camera_profiles.PROFILES_NAME", "camera_profiles.json"):
            out = inf._owner_place_quiet(event, job, CH2, decision)
        self.assertFalse(out.notify)
        self.assertEqual(out.reason, "neighbour's place (owner said)")
        self.assertEqual(decision["owner_place"]["text"], "הבית של השכן")
        job.tracker_tracks = [Track("person", [(1.0, 0.10, 0.73), (2.0, 0.60, 0.85)])]
        with mock.patch.object(inf, "EVENTS", events):
            self.assertTrue(inf._owner_place_quiet(event, job, CH2, {}).notify)


class RoboticLineTests(unittest.TestCase):
    def test_generic_questions_are_found(self) -> None:
        for text in ("מה לתקן?", "מה תרצה לשנות או להוסיף?", "במה אוכל לעזור?", "הבנתי אותך.", "What should I fix?"):
            self.assertEqual(human.drop_generic(text), "", text)
        self.assertEqual(human.drop_generic("ההתראה הייתה בכניסה. מה לתקן?"), "ההתראה הייתה בכניסה.")
        self.assertEqual(human.drop_generic("מה לתקן בהתראה של 13:35?"), "מה לתקן בהתראה של 13:35?")

    def test_anger(self) -> None:
        self.assertTrue(human.is_angry("מה הקשררררררר העבדים של הפרגולה יא חתכית מטומטם"))
        self.assertTrue(human.is_angry("מה הקקשרררררר עובדייים יא מטומטם"))
        self.assertFalse(human.is_angry("מה הקשר?"))
        self.assertFalse(human.is_angry("מה קשר ? לא הבנתי"))


if __name__ == "__main__":
    unittest.main()
