"""Stage 2a in the guard loop: box.yaml ``eye_entities`` (the P1/P2 roster line in the legacy prompt, off by default),
the event's entities from the job's tracker tracks, and the story in an event's update; plus the owner's own
three-message example (2026-10-08) from synthetic tracker tracks. Local fakes only: no model, no Telegram."""
import json
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime
from types import SimpleNamespace
from unittest import mock

import numpy as np

from home_guard_project.box import camera_names
from home_guard_project.box import entities as ent
from home_guard_project.box import inference as inf
from home_guard_project.box.events import ESCALATION_REPEAT_SEC, EventBook

CAM = "ameer_week_0_1_ch3"
ALIASES = {CAM: ["פרגולה"]}
T0 = datetime(2026, 10, 8, 9, 42).timestamp()
HE = {"alert_channel": "telegram", "owner_translation": "model"}
LINE = "PEOPLE/VEHICLES IN VIEW (from the tracker): P1 in view 3 min, gate>patio; P2 new"


def trk(track_id, first, last, start=(0.5, 0.8), active=True, path=()):
    return {"id": track_id, "kind": "person", "first_seen": float(first), "last_seen": float(last), "prev_id": None,
            "first_foot": tuple(start), "last_foot": tuple(start), "moved": 0.2, "active": active, "path": list(path),
            "entry_edge": "", "exit_edge": ""}


def answer(label="suspicious", people=2, summary="Two men near the entrance.", per_entity=None, why=""):
    out = dict(summary=summary, label=label, raw_label=label, applied_fact_id="", serious_behaviour=False,
               people=people, vehicle_moving=False, animals=0, why=why, summary_owner="")
    if per_entity is not None:
        out["per_entity"] = per_entity
    return out


class Backend:
    def __init__(self, *answers):
        self.answers, self.calls, self.kwargs = list(answers), 0, []
        self.model_name, self.last_prompt = "fake", "prompt"

    def analyze(self, frames, camera_name, t_sec, start_hour, end_hour, owner_language="en", **kwargs):
        self.kwargs.append(kwargs)
        parsed = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return json.dumps(parsed), parsed


class Assistant:
    def __init__(self):
        self.sent = []
        self.index = SimpleNamespace(messages=lambda alert_id: [])

    def is_muted(self, camera):
        return False

    def send_alert(self, alert, text, image=None, silent=False, lang="en", reply_to=None):
        message_id = 100 + len(self.sent)
        self.sent.append({"text": text, "reply_to": reply_to, "id": message_id})
        return {"sent": True, "results": [{"chat_id": "-5", "ok": True, "message_id": message_id}]}

    def remind_if_silent(self, alert, text, lang="en"):
        pass


class SettingsAndPromptTest(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(inf.AlertSettings().eye_entities)
        self.assertFalse(inf.AlertSettings.from_box_settings({}).eye_entities)
        self.assertTrue(inf.AlertSettings.from_box_settings({"eye_entities": "on"}).eye_entities)
        self.assertFalse(inf.AlertSettings.from_box_settings({"eye_entities": "off"}).eye_entities)

    def test_off_leaves_the_prompt_byte_identical(self):
        base = inf.build_prompt(CAM, 0, "09:42:00", 0, 0)
        self.assertEqual(inf.build_prompt(CAM, 0, "09:42:00", 0, 0, entities_line=""), base)
        self.assertNotIn("per_entity", base)
        self.assertNotIn("PEOPLE/VEHICLES", base)

    def test_on_adds_the_roster_its_rule_and_per_entity(self):
        prompt = inf.build_prompt(CAM, 0, "09:42:00", 0, 0, owner_language="he", entities_line=LINE)
        base = inf.build_prompt(CAM, 0, "09:42:00", 0, 0, owner_language="he")
        self.assertTrue(prompt.startswith(base))
        tail = prompt[len(base):]
        self.assertTrue(tail.startswith("\n\n" + LINE + "\n" + ent.ROSTER_RULE + "\n"), tail)
        self.assertIn('"per_entity"', tail)
        self.assertIn("in Hebrew", tail)
        self.assertIn("Count people from the frames", tail)

    def test_the_schema_takes_per_entity_only_when_on(self):
        self.assertNotIn("per_entity", inf.VLM_SCHEMA["properties"])
        schema = inf.VLM_RESPONSE_FORMAT_ENTITIES["json_schema"]["schema"]
        self.assertIn("per_entity", schema["required"])
        self.assertEqual(set(schema["required"]) - {"per_entity"}, set(inf.VLM_SCHEMA["required"]))

    def test_gpt_backend_uses_the_entities_schema_with_a_line(self):
        obj = inf.GptBackend.__new__(inf.GptBackend)
        obj._model = obj.model_name = "gpt-local-test"
        obj._response_format = inf.VLM_RESPONSE_FORMAT
        obj._complete = mock.Mock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer())))], model="m"))
        obj.analyze([], CAM, 0, 0, 0, entities_line=LINE)
        self.assertIs(obj._complete.call_args.args[1], inf.VLM_RESPONSE_FORMAT_ENTITIES)
        self.assertIn(LINE, obj.last_prompt)
        obj.analyze([], CAM, 0, 0, 0)
        self.assertIs(obj._complete.call_args.args[1], inf.VLM_RESPONSE_FORMAT)
        inf.NullBackend().analyze([], CAM, 0, 0, 0, entities_line=LINE)


class WorkerCase(unittest.TestCase):
    lang = "he"

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.book = EventBook(self.dir)
        self.stack = ExitStack()
        for patch in (mock.patch.object(inf, "EVENTS", self.book),
                      mock.patch.object(inf, "KNOWN_CAMERAS", (CAM,)),
                      mock.patch.object(inf, "FACTS_PROVIDER", mock.Mock(return_value=[])),
                      mock.patch.object(inf, "owner_language", side_effect=lambda: self.lang),
                      mock.patch.object(inf, "frame_to_jpeg_bytes", return_value=b"jpg"),
                      mock.patch.object(inf, "_jpegs", return_value=[]),
                      mock.patch.object(camera_names, "_load", side_effect=lambda a: ALIASES if a is None else a)):
            self.stack.enter_context(patch)
        inf._SOFTENED_DAYS.clear()
        self.assistant = Assistant()

    def tearDown(self):
        self.stack.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def work(self, backend, ts, tracks, on=True, mode="legacy"):
        job = inf.AlertJob(camera=CAM, stem=f"{CAM}_{int(ts)}_alert", ts=ts, labels=["person"],
                           input_meta={"vlm_input": "crop"})
        job.tracker_entities, job.tracker_since = tracks, (ts - 4 if tracks is not None else None)
        inf._worker(backend, HE, {}, inf.AlertSettings(eye_entities=on, eye_prompt=mode), CAM,
                    [np.zeros((4, 4, 3), np.uint8)] * 4, self.assistant, job)
        return job

    def keep_alive(self, start, end):
        for t in range(int(start), int(end), 20):
            self.book.activity(CAM, t, people=2)


class WorkerTest(WorkerCase):
    def test_off_the_eye_gets_nothing_new_but_the_event_has_its_entities(self):
        backend = Backend(answer())
        job = self.work(backend, T0, [trk(1, T0 - 2, T0)], on=False)
        self.assertNotIn("entities_line", backend.kwargs[0])
        self.assertEqual(job.teacher["prompt_version"], inf.PROMPT_VERSION)
        self.assertEqual(job.alert["event"]["entities"], ["P1"])
        self.assertEqual(self.book.session_of_alert(job.stem)["reported_entities"], ["P1"])

    def test_on_passes_the_roster_and_suffixes_the_version(self):
        backend = Backend(answer(per_entity=[{"id": "P1", "action": "מנקה את הרצפה"}]))
        job = self.work(backend, T0, [trk(1, T0 - 2, T0)], on=True)
        self.assertEqual(backend.kwargs[0]["entities_line"], "PEOPLE/VEHICLES IN VIEW (from the tracker): P1 new")
        self.assertEqual(job.teacher["prompt_version"], inf.PROMPT_VERSION + "+ent1")
        notes = self.book.session_of_alert(job.stem)["entities"][0]["notes"]
        self.assertEqual((notes[0]["text"], notes[0]["source"]), ("מנקה את הרצפה", "eye"))

    def test_on_without_tracks_or_with_the_situational_prompt_changes_nothing(self):
        backend = Backend(answer())
        job = self.work(backend, T0, None, on=True)
        self.assertNotIn("entities_line", backend.kwargs[0])
        self.assertEqual(job.teacher["prompt_version"], inf.PROMPT_VERSION)
        self.assertEqual(job.alert["event"]["counted_by"], "head-count")
        backend = Backend(answer())
        with mock.patch.object(inf, "_eye_situation", return_value=None):
            self.work(backend, T0 + 600, [trk(1, T0 + 598, T0 + 600)], on=True, mode="situational")
        self.assertIn("entities_line", backend.kwargs[0])      # no situation was built: the legacy prompt
        backend = Backend(answer())
        with mock.patch.object(inf, "_eye_situation", return_value=SimpleNamespace(intent="x")), \
                mock.patch.object(inf, "_eye_answer", side_effect=lambda parsed, s: (parsed, {})):
            self.work(backend, T0 + 1200, [trk(1, T0 + 1198, T0 + 1200)], on=True, mode="situational")
        self.assertNotIn("entities_line", backend.kwargs[0])

    def test_an_update_tells_who_is_new(self):
        self.work(Backend(answer(people=1, summary="A man walks by the gate.")), T0, [trk(1, T0 - 2, T0)], on=False)
        self.keep_alive(T0, T0 + 60)
        tracks = [trk(1, T0 - 2, T0 + 60), trk(2, T0 + 50, T0 + 60, start=(0.1, 0.2))]
        self.work(Backend(answer(people=2)), T0 + 60, tracks, on=False)
        self.assertEqual(len(self.assistant.sent), 2)
        update = self.assistant.sent[1]
        self.assertEqual(update["reply_to"]["message_id"], self.assistant.sent[0]["id"])
        # 2026-10-09 message v2: the Eye's English is never quoted inside the Hebrew story line.
        self.assertEqual(update["text"].splitlines()[:2], ["עוד אדם אחד הגיע (P2)", "בתמונה עכשיו: P1 ו-P2."])
        self.assertNotIn("P1", self.assistant.sent[0]["text"])         # the first message keeps today's format

    def test_workers_marked_then_a_new_one_is_not_theirs(self):
        workers = lambda ts: [trk(1, T0 - 2, ts), trk(2, T0 - 2, ts, start=(0.2, 0.8))]   # noqa: E731
        self.work(Backend(answer()), T0, workers(T0), on=False)
        self.book.mark_known(CAM, "עובדים בפרגולה", "owner", until=T0 + 8 * 3600, now=T0 + 10)
        self.keep_alive(T0, T0 + 120)
        self.work(Backend(answer(people=5)), T0 + 60, workers(T0 + 60), on=False)          # the Eye counts 5
        self.assertEqual(len(self.assistant.sent), 1)
        stranger = trk(3, T0 + 110, T0 + 120, start=(0.9, 0.2))
        self.work(Backend(answer("normal", people=3)), T0 + 120, workers(T0 + 120) + [stranger], on=False)
        self.assertEqual(len(self.assistant.sent), 1)                                      # normal never sends
        self.keep_alive(T0 + 120, T0 + 180)
        self.work(Backend(answer(people=3, why="looks into the car")), T0 + 180,
                  workers(T0 + 180) + [dict(stranger, last_seen=T0 + 180)], on=False)
        self.assertEqual(len(self.assistant.sent), 2)
        self.assertEqual(self.assistant.sent[1]["text"].splitlines()[0],
                         "אדם חדש הגיע (P3), לא מאלה שסימנת (עובדים בפרגולה)")


class OwnersExampleTest(WorkerCase):
    """The owner's words (2026-10-08): msg1 'a man in a white shirt walks near the entrance with another man cleaning
    the floor', msg2 'both moved now toward the pergola', msg3 'the one who was cleaning put the broom in the pickup'.
    Labels are chosen so that each of the three looks is a message under stage 1's policy (a first suspicious, a
    level rise, an escalation again after ESCALATION_REPEAT_SEC); what is tested is the text the owner reads."""

    def test_three_messages_continue_one_story(self):
        white = lambda ts, path: trk(1, T0 - 2, ts, start=(0.4, 0.8), path=path)        # noqa: E731
        cleaner = lambda ts, path: trk(2, T0 - 2, ts, start=(0.6, 0.8), path=path)      # noqa: E731
        one = Backend(answer("suspicious", summary="A man in a white shirt walks near the entrance while another "
                             "man cleans the floor.", why="walks around the entrance",
                             per_entity=[{"id": "P1", "action": "הולך ליד הכניסה בחולצה לבנה"},
                                         {"id": "P2", "action": "מנקה את הרצפה"}]))
        self.work(one, T0, [white(T0, ["כניסה"]), cleaner(T0, ["כניסה"])])
        self.keep_alive(T0, T0 + 60)
        two = Backend(answer("escalation", summary="Both men walk to the pergola.", per_entity=[]))
        self.work(two, T0 + 60, [white(T0 + 60, ["כניסה", "פרגולה"]), cleaner(T0 + 60, ["כניסה", "פרגולה"])])
        t3 = T0 + 60 + ESCALATION_REPEAT_SEC + 1
        self.keep_alive(T0 + 60, t3)
        three = Backend(answer("escalation", summary="A man puts a broom in the pickup.",
                               per_entity=[{"id": "P2", "action": "שם את המטאטא בטנדר"}]))
        self.work(three, t3, [white(t3, ["כניסה", "פרגולה"]), cleaner(t3, ["כניסה", "פרגולה"])])

        msgs = self.assistant.sent
        self.assertEqual(len(msgs), 3)
        self.assertIsNone(msgs[0]["reply_to"])
        self.assertNotIn("P1", msgs[0]["text"])                      # the first message keeps today's format
        self.assertEqual(msgs[1]["reply_to"]["message_id"], msgs[0]["id"])
        self.assertEqual(msgs[1]["text"].splitlines()[0], "שניהם (P1, P2) עברו לפרגולה.")
        self.assertEqual(msgs[2]["reply_to"]["message_id"], msgs[0]["id"])
        self.assertIn("P2 (קודם: מנקה את הרצפה): שם את המטאטא בטנדר.", msgs[2]["text"].splitlines()[0])
        self.assertNotIn("עברו", msgs[2]["text"])                    # they did not move again
        self.assertNotIn(CAM, "".join(m["text"] for m in msgs))
        self.assertIn("A man puts a broom in the pickup.", msgs[2]["text"])   # the new observation follows


if __name__ == "__main__":
    unittest.main()
