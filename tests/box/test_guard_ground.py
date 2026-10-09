"""Scene map stage 2c in the guard loop: whose ground decides, in code, before the event book."""
import unittest
from unittest import mock

import numpy as np

from home_guard_project.box import inference as inf
from home_guard_project.box import scene_map as sm
from test_guard_events import CAM, HE, T0, Backend, GuardCase, answer

LEFT = ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))
RIGHT = ((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0))
SCENE = sm.SceneMap(CAM, areas=(sm.Area("yard", sm.MINE, "yard", LEFT),
                                sm.Area("their driveway", sm.WATCH, "parking", RIGHT, owner="neighbour")),
                    lines=(sm.Line("המעקה", (0.5, 0.0), (0.5, 1.0), inward="right"),))


def walk(xs, t0=T0):
    return [sm.Track("person", [(t0 + i * 0.5, x, 0.6) for i, x in enumerate(xs)])]


STAYS_THERE = walk([0.8, 0.82, 0.85, 0.8, 0.78, 0.8])
COMES_IN = walk([0.8, 0.75, 0.7, 0.6, 0.55, 0.45, 0.4, 0.35, 0.3])


class GroundGuardTest(GuardCase):
    def setUp(self):
        super().setUp()
        self.stack.enter_context(mock.patch.object(sm, "load_scene_map", return_value=SCENE))

    def work_with(self, backend, tracks, ts=T0):
        job = inf.AlertJob(camera=CAM, stem=f"{CAM}_{int(ts)}_alert", ts=ts, labels=["person"],
                           input_meta={"vlm_input": "crop"})
        job.tracker_tracks = list(tracks)
        inf._worker(backend, HE, {}, inf.AlertSettings(), CAM, [np.zeros((4, 4, 3), np.uint8)] * 4,
                    self.assistant, job)
        self.assertTrue(job.ready.is_set())
        return job

    def test_a_suspicious_walk_on_the_neighbours_ground_is_not_sent(self):
        job = self.work_with(Backend(answer("suspicious", people=1, why="a man walks around at night")), STAYS_THERE)
        self.assertEqual(self.assistant.sent, [])
        self.assertIn("not ours", job.alert["not_sent_reason"])
        self.assertEqual(job.alert["ground"]["on"], "neighbour")
        self.assertEqual(job.input_meta["ground"]["off_our_ground"], True)

    def test_trying_the_neighbours_car_door_is_sent(self):
        self.work_with(Backend(answer("suspicious", people=1, why="tries the car door handle")), STAYS_THERE)
        self.assertEqual(len(self.assistant.sent), 1)

    def test_coming_in_from_the_neighbours_side_is_a_message_even_when_normal(self):
        job = self.work_with(Backend(answer("normal", people=1, summary="A man walks.")), COMES_IN)
        self.assertEqual(len(self.assistant.sent), 1)
        self.assertIn("נכנס מהצד של השכן אל השטח שלנו (דרך המעקה)", self.assistant.sent[0]["text"])
        self.assertEqual(job.alert["label"], "suspicious")
        self.assertEqual(job.alert["raised"], "came onto our ground")
        self.assertIn("came onto our ground", job.alert["event"]["reason"])
        self.assertNotIn(CAM, self.assistant.sent[0]["text"])          # the camera's name, never its id

    def test_without_tracks_or_map_everything_is_as_before(self):
        job = self.work_with(Backend(answer("suspicious", people=1, why="a man walks around at night")), [])
        self.assertEqual(len(self.assistant.sent), 1)
        self.assertNotIn("ground", job.alert)

    @mock.patch.object(inf, "AI_FAILED_NOTIFY", True)
    def test_an_unanswered_look_off_our_ground_is_not_forced_out(self):
        job = self.work_with(Backend(None), STAYS_THERE)
        self.assertEqual(self.assistant.sent, [])
        self.assertIn("not ours", job.alert["not_sent_reason"])
        self.work_with(Backend(None), COMES_IN, ts=T0 + 600)            # on our ground: the detector's alert goes
        self.assertEqual(len(self.assistant.sent), 1)

    def test_a_broken_map_never_stops_an_alert(self):
        with mock.patch.object(sm, "load_scene_map", side_effect=OSError("disk")), \
                self.assertLogs("box.inference", level="WARNING"):
            self.work_with(Backend(answer("suspicious", people=1, why="a man walks around at night")), STAYS_THERE)
        self.assertEqual(len(self.assistant.sent), 1)


if __name__ == "__main__":
    unittest.main()
