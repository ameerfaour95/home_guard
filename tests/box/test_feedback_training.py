from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from home_guard_project.box.alert_clips import encode_frame, write_alert_clip
from home_guard_project.box.feedback import Feedback, keep_for_training, save_feedback

NOW = 1_800_000_000.0
ALERT_META = {"summary": "A person is at the door.", "alert_command": "[send_message]", "alert_reason": "",
              "labels": ["person"], "dispatch": {"sent": True}}


class KeepForTrainingTest(unittest.TestCase):
    """An owner's answer keeps the alert's video, with what the detector and the AI said, for tagging."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.production = os.path.join(tmp.name, "production_multi")
        self.training = os.path.join(tmp.name, "dataset_multi")
        self.archive = os.path.join(tmp.name, "production_archive")
        frames = [(NOW + i * 0.2, encode_frame(np.zeros((48, 64, 3), dtype=np.uint8))) for i in range(5)]
        from unittest import mock
        with mock.patch("home_guard_project.box.alert_clips._to_h264", return_value=False):
            self.meta_path = write_alert_clip(self.production, "door", "door_100_alert", frames, ALERT_META)
        self.alert = {"alert_id": "door_100_alert", "camera": "door", "summary": "A person is at the door.", "ts": NOW}

    def _training_metas(self):
        return [os.path.join(d, n) for d, _, names in os.walk(self.training) for n in names if n.endswith(".meta.json")]

    def test_a_false_alarm_answer_copies_the_clip_with_everything_a_tagger_needs(self) -> None:
        feedback = Feedback(verdict="false_alarm", note="it was the neighbour's cat", source="text")
        path = save_feedback(self.production, self.alert, feedback, "no, that was the neighbour's cat",
                             {"user_id": 5, "name": "Ameer"}, "-100", NOW + 60,
                             training_dir=self.training, archive_dir=self.archive)
        self.assertTrue(os.path.isfile(path))                      # the answer itself, as before

        metas = self._training_metas()
        self.assertEqual([os.path.basename(m) for m in metas], ["door_100_alert.meta.json"])
        with open(metas[0], encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["kind"], "owner_feedback")
        self.assertEqual(meta["yolo"]["class_counts"], {"person": 1})            # what the detector saw
        self.assertEqual(meta["alert"]["summary"], "A person is at the door.")   # what the AI said
        self.assertEqual(meta["owner_feedback"][0]["verdict"], "false_alarm")    # what the owner said
        self.assertEqual(meta["owner_feedback"][0]["raw_text"], "no, that was the neighbour's cat")
        self.assertEqual(meta["owner_feedback"][0]["from"], "Ameer")
        clip = os.path.join(self.training, *meta["clip_path"].replace("\\", "/").split("/"))
        self.assertTrue(os.path.isfile(clip))                                    # and the video
        self.assertTrue(os.path.isfile(self.meta_path))                          # the production copy stays

    def test_a_second_answer_is_added_to_the_same_copy(self) -> None:
        keep_for_training(self.alert, Feedback(verdict="true_alert", source="button"), "", {"name": "Ameer"},
                          NOW + 10, self.production, self.training, self.archive)
        keep_for_training(self.alert, Feedback(verdict="real_but_wrong", note="two people, not one"),
                          "there were two of them", {"name": "Ameer"}, NOW + 90, self.production, self.training,
                          self.archive)
        metas = self._training_metas()
        self.assertEqual(len(metas), 1)
        with open(metas[0], encoding="utf-8") as f:
            verdicts = [a["verdict"] for a in json.load(f)["owner_feedback"]]
        self.assertEqual(verdicts, ["true_alert", "real_but_wrong"])

    def test_an_answer_that_judges_nothing_keeps_nothing(self) -> None:
        save_feedback(self.production, self.alert, Feedback(verdict="none", action="mute", mute_until=NOW + 3600),
                      "stop for an hour", {"name": "Ameer"}, "-100", NOW + 5,
                      training_dir=self.training, archive_dir=self.archive)
        self.assertEqual(self._training_metas(), [])

    def test_an_archived_clip_is_found_too_and_a_gone_clip_is_harmless(self) -> None:
        import shutil
        site = os.path.join(self.archive, "test")
        shutil.move(self.production, site)
        os.makedirs(self.production)
        kept = keep_for_training(self.alert, Feedback(verdict="expected"), "it's my brother", {"name": "Ameer"},
                                 NOW + 5, self.production, self.training, self.archive)
        self.assertTrue(kept and os.path.isfile(kept))
        gone = keep_for_training({"alert_id": "door_999_alert", "camera": "door"}, Feedback(verdict="expected"),
                                 "", {}, NOW + 6, self.production, self.training, self.archive)
        self.assertIsNone(gone)


if __name__ == "__main__":
    unittest.main()
