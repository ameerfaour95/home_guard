"""An owner's answer is kept for good: a copy outside the S3 folder that expires, with everything training needs."""

from __future__ import annotations

import glob
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from home_guard_project.box import boxconfig
from home_guard_project.box.alert_clips import encode_frame, write_alert_clip
from home_guard_project.box.feedback import Feedback, save_feedback

NOW = 1_800_000_000.0
STEM = "door_1800000000_alert"
ALERT = {"alert_id": STEM, "camera": "door", "summary": "A person is at the door.", "label": "suspicious",
         "ts": NOW}
ALERT_META = {"summary": "A person is at the door.", "alert_command": "[send_message]", "alert_reason": "",
              "labels": ["person"], "raw_label": "normal", "label": "suspicious", "why": "at night",
              "dispatch": {"sent": True}}
SITUATION = {"phase": "late_night", "dark": True, "house_state": "home_asleep", "intent": "alert_triage",
             "camera_role": "entrance"}
TEACHER = {"model": "gpt-4o", "prompt_version": "eye-v3", "prompt": "what do you see", "frames": [b"\xff\xd8one"],
           "raw": "{}", "parsed": {"summary": "A person is at the door.", "category": "S2"}}
DANA = {"user_id": 42, "name": "Dana"}


class _Crop:
    """What vlm_crop.crop_clip returns, as far as write_alert_clip reads it."""

    width, height = 32, 24
    frames = [np.zeros((24, 32, 3), dtype=np.uint8) for _ in range(3)]


def _write_event(root: str) -> str:
    """The production copy of one alert, as inference writes it: clip, crop, teacher pictures, situation."""
    frames = [(NOW + i * 0.2, encode_frame(np.zeros((48, 64, 3), dtype=np.uint8))) for i in range(5)]
    with mock.patch("home_guard_project.box.alert_clips._to_h264", return_value=False), \
            mock.patch("home_guard_project.data_collection.vlm_crop.crop_meta",
                       side_effect=lambda crop, settings, rel, fps: {"vlm_crop_path": rel.replace("/", "\\"),
                                                                     "fps": fps}):
        return write_alert_clip(root, "door", STEM, frames, ALERT_META, teacher=TEACHER,
                                extra={"trigger_ts": NOW, "mode": "guard", "vlm_input": "crop",
                                       "situation": SITUATION, "observation": {"category": "S2"},
                                       "prompt_version": "eye-v3"},
                                crop=_Crop(), crop_settings=None, crop_fps=5.0)


def _files(root: str, pattern: str):
    return sorted(glob.glob(os.path.join(root, *pattern.split("/")), recursive=True))


class DurableAnswerTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.production = os.path.join(self.dir, "production_multi")
        self.training = os.path.join(self.dir, "dataset_multi")
        self.archive = os.path.join(self.dir, "production_archive")
        self.meta_path = _write_event(self.production)

    def _save(self, feedback: Feedback, raw: str = "", now: float = NOW + 60, training: bool = True) -> str:
        return save_feedback(self.production, ALERT, feedback, raw, DANA, "-100", now,
                             training_dir=self.training if training else None, archive_dir=self.archive)

    def test_the_answer_is_also_written_to_the_training_folder_under_the_same_name(self) -> None:
        path = self._save(Feedback(verdict="expected", owner_label="normal", tagged_by="Dana", source="button"))
        (durable,) = _files(self.training, "feedback/**/*.feedback.json")
        self.assertEqual(os.path.basename(durable), os.path.basename(path))
        self.assertEqual(os.path.relpath(durable, self.training), os.path.relpath(path, self.production))
        with open(path, encoding="utf-8") as f:
            operational = json.load(f)
        with open(durable, encoding="utf-8") as f:
            kept = json.load(f)
        self.assertEqual({k: v for k, v in kept.items() if k != "training"}, operational)

    def test_the_kept_answer_carries_everything_training_needs(self) -> None:
        self._save(Feedback(verdict="real_but_wrong", owner_label="other", owner_text="the gardener",
                            transcript="the gardener", tagged_by="Dana", source="voice"), raw="the gardener")
        (durable,) = _files(self.training, "feedback/**/*.feedback.json")
        with open(durable, encoding="utf-8") as f:
            record = json.load(f)["training"]
        self.assertEqual(record["event_id"], STEM)
        self.assertEqual(record["answer"], {
            "label": "other", "verdict": "real_but_wrong", "text": "the gardener", "transcript": "the gardener",
            "source": "voice", "by": DANA, "time_utc": "2027-01-15T08:01:00Z"})
        day = os.path.basename(os.path.dirname(self.meta_path))
        self.assertEqual(record["media"]["clip_path"], f"clips\\door\\{day}\\{STEM}.mp4")
        self.assertEqual(record["media"]["vlm_crop_path"], f"vlm_crops\\door\\{day}\\{STEM}.mp4")
        self.assertEqual(record["media"]["meta_path"], f"meta/door/{day}/{STEM}.meta.json")
        self.assertEqual(record["media"]["vlm_input"], "crop")
        self.assertEqual(record["yolo"]["class_counts"], {"person": 1})
        self.assertEqual(record["model"], {
            "description": "A person is at the door.", "category": "S2", "raw_label": "normal",
            "label": "suspicious", "why": "at night", "model": "gpt-4o", "prompt_version": "eye-v3"})
        self.assertEqual(record["situation"]["house_state"], "home_asleep")
        self.assertEqual(record["situation"]["camera_role"], "entrance")
        self.assertEqual(record["situation"]["camera"], "door")
        self.assertTrue(record["situation"]["time_local"].startswith("2027-01-15 "))

    def test_a_late_answer_copies_the_crop_and_the_teacher_pictures_with_the_clip(self) -> None:
        # The training copy inference wrote has long been uploaded and removed from the box.
        self._save(Feedback(verdict="expected", owner_label="normal", source="button"))
        day = os.path.basename(os.path.dirname(self.meta_path))
        for rel in (f"clips/door/{day}/{STEM}.mp4", f"vlm_crops/door/{day}/{STEM}.mp4",
                    f"vlm_crops/door/{day}/{STEM}_f0.jpg", f"meta/door/{day}/{STEM}.meta.json"):
            self.assertTrue(os.path.isfile(os.path.join(self.training, *rel.split("/"))), rel)
        with open(os.path.join(self.training, "meta", "door", day, f"{STEM}.meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["teacher"]["model"], "gpt-4o")
        self.assertEqual(meta["vlm_crop"]["vlm_crop_path"], f"vlm_crops\\door\\{day}\\{STEM}.mp4")

    def test_an_answer_whose_clip_is_gone_is_still_kept(self) -> None:
        shutil.rmtree(self.production)
        self._save(Feedback(verdict="expected", owner_label="normal", source="button"))
        (durable,) = _files(self.training, "feedback/**/*.feedback.json")
        with open(durable, encoding="utf-8") as f:
            record = json.load(f)["training"]
        self.assertEqual((record["event_id"], record["answer"]["label"]), (STEM, "normal"))
        self.assertEqual(record["model"]["description"], "A person is at the door.")   # from the alert itself
        self.assertIsNone(record["media"]["clip_path"])

    def test_a_pause_request_is_kept_too(self) -> None:
        self._save(Feedback(action="mute", mute_until=NOW + 3600), raw="it's me, pause for an hour")
        self.assertEqual(len(_files(self.training, "feedback/**/*.feedback.json")), 1)

    def test_no_training_folder_named_and_not_the_box_folder_writes_no_copy(self) -> None:
        # Tests and tools that name only a scratch folder must never write into the box's dataset.
        live = os.path.join(self.dir, "box_dataset")
        with mock.patch.object(boxconfig, "LIVE_DIR", live):
            self._save(Feedback(action="mute", mute_until=NOW + 3600), training=False)
        self.assertFalse(os.path.exists(os.path.join(live, "feedback")))

    def test_the_box_production_folder_keeps_its_answers_in_the_box_dataset(self) -> None:
        live = os.path.join(self.dir, "box_dataset")
        with mock.patch.object(boxconfig, "LIVE_DIR", live), \
                mock.patch.object(boxconfig, "PRODUCTION_LIVE_DIR", self.production), \
                mock.patch.object(boxconfig, "PRODUCTION_ARCHIVE_DIR", self.archive):
            self._save(Feedback(action="mute", mute_until=NOW + 3600), training=False)
        self.assertEqual(len(_files(live, "feedback/**/*.feedback.json")), 1)


class ProductionCopyTest(unittest.TestCase):
    def test_the_owner_copy_of_an_alert_names_the_model_and_keeps_what_it_saw(self) -> None:
        # A late answer re-creates the training copy from this one: it must not lose the teacher's answer.
        from home_guard_project.box import inference as inf

        with tempfile.TemporaryDirectory() as tmp:
            production, training = os.path.join(tmp, "production_multi"), os.path.join(tmp, "dataset_multi")
            frames = [(NOW + i * 0.2, encode_frame(np.zeros((48, 64, 3), dtype=np.uint8))) for i in range(5)]
            job = inf.AlertJob(camera="door", stem=STEM, ts=NOW, labels=["person"], alert=dict(ALERT_META),
                               teacher=dict(TEACHER))
            job.ready.set()
            with mock.patch("home_guard_project.box.alert_clips._to_h264", return_value=False):
                inf._save_clip(job, frames, production, training)
            (meta_path,) = _files(production, "meta/**/*.meta.json")
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        self.assertEqual((meta["teacher"]["model"], meta["teacher"]["prompt_version"]), ("gpt-4o", "eye-v3"))
        self.assertEqual(meta["model_response"], TEACHER["parsed"])
        self.assertEqual(meta["alert"]["label"], "suspicious")      # the owner's copy keeps the final label


class FeedbackKeyTest(unittest.TestCase):
    """The S3 keys an answer is uploaded under: one copy must live outside ``production_`` (14-day expiry)."""

    def test_the_kept_answer_is_uploaded_under_the_dataset_prefix(self) -> None:
        from home_guard_project.box.__main__ import run_upload
        from home_guard_project.box.boxconfig import BoxConfig, production_prefix
        from home_guard_project.s3_upload.s3_upload import _s3_key

        with tempfile.TemporaryDirectory() as tmp:
            production, training = os.path.join(tmp, "production_multi"), os.path.join(tmp, "dataset_multi")
            _write_event(production)
            save_feedback(production, ALERT, Feedback(verdict="expected", owner_label="normal", source="button"),
                          "", DANA, "-100", NOW + 60, training_dir=training,
                          archive_dir=os.path.join(tmp, "production_archive"))
            keys = []

            def uploader(dataset_dir, bucket, prefix, **kwargs):
                for dirpath, _, names in os.walk(dataset_dir):
                    keys.extend(_s3_key(os.path.join(dirpath, n), dataset_dir, prefix) for n in names)

            cfg = BoxConfig(site="house2", min_age_minutes=0.0, mode="inference")
            run_upload(cfg, training, os.path.join(tmp, "dataset_outbox"), "bucket", 1, uploader=uploader)
            run_upload(cfg, production, os.path.join(tmp, "production_archive"), "bucket", 1, uploader=uploader,
                       prefix_for=production_prefix, keep_local=True)

        answers = [k for k in keys if k.endswith(".feedback.json")]
        kept = [k for k in answers if not k.startswith("production_")]
        self.assertEqual(len(kept), 1, answers)
        self.assertTrue(kept[0].startswith("dataset_house2/feedback/door/"), kept[0])
        self.assertIn(f"dataset_house2/meta/door/", " ".join(keys))        # the clip's meta goes along


if __name__ == "__main__":
    unittest.main()
