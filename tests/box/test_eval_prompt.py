"""Tests for box/eval_prompt.py: score the VLM prompt against our human tags.

No network: S3 is a fake client serving files from a temp dir, the model is FakeBackend.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import unittest
from typing import Dict, List

import numpy as np

from home_guard_project.box import eval_prompt as ev
from home_guard_project.box import inference


def write_video(path: str, levels: List[int], size=(64, 48)) -> None:
    """A tiny mp4 whose frame i is filled with grey level levels[i]."""
    import cv2

    w, h = size
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for level in levels:
        writer.write(np.full((h, w, 3), level, dtype=np.uint8))
    writer.release()


class TruthTest(unittest.TestCase):
    def test_alert_tags(self) -> None:
        self.assertEqual(ev.parse_truth("Two men break the door. [alert]"), ("alert", "Two men break the door."))
        self.assertEqual(ev.parse_truth("[ALERT] A man climbs the fence."), ("alert", "A man climbs the fence."))
        self.assertEqual(ev.parse_truth("A man hides his face [alet]"), ("alert", "A man hides his face"))

    def test_empty(self) -> None:
        self.assertEqual(ev.parse_truth("No special activity"), ("empty", "No special activity"))
        self.assertEqual(ev.parse_truth("no special activity."), ("empty", "no special activity."))
        self.assertEqual(ev.parse_truth("  No special activity.  "), ("empty", "No special activity."))

    def test_normal(self) -> None:
        self.assertEqual(ev.parse_truth("A woman carries a bag inside."), ("normal", "A woman carries a bag inside."))

    def test_other_tag_is_stripped_but_not_alert(self) -> None:
        self.assertEqual(ev.parse_truth("[note] A man walks by."), ("normal", "A man walks by."))

    def test_dropped(self) -> None:
        self.assertTrue(ev.is_dropped("[delete] blurry"))
        self.assertTrue(ev.is_dropped(""))
        self.assertFalse(ev.is_dropped("A man walks by. [alert]"))


class IndexTest(unittest.TestCase):
    def test_preference_order(self) -> None:
        keys = [
            "tagging/b1/dataset_multi/clips/cam/d/x.mp4",
            "dataset_ameer_house/clips/cam/x.mp4",
            "dataset_multi/clips/cam/d/x.mp4",
            "tagging/b1/dataset_multi/clips/cam/d/y.mp4",
            "dataset_ameer_house/vlm_crops/cam/y.mp4",
            "dataset_ameer_house/clips/cam/y.mp4",
            "tagging/b1/dataset_multi/vlm_crops/cam/d/z.mp4",
            "tagging/b1/dataset_multi/clips/cam/d/z.mp4",
            "tagging/b2/clips/w.mp4",
            "tagging/b1/clips/w.mp4",
            "tagging/b1/analysis_output/vlm_training.jsonl",
        ]
        index = ev.build_index(keys)
        self.assertEqual(index["x"], "dataset_multi/clips/cam/d/x.mp4")
        self.assertEqual(index["y"], "dataset_ameer_house/clips/cam/y.mp4")   # full clip beats crop
        self.assertEqual(index["z"], "tagging/b1/dataset_multi/clips/cam/d/z.mp4")
        self.assertEqual(index["w"], "tagging/b1/clips/w.mp4")                # sorted, deterministic
        self.assertNotIn("vlm_training", index)

    def test_a_crop_never_beats_a_full_clip_across_prefix_tiers(self) -> None:
        keys = ["dataset_ameer_house/vlm_crops/cam/q.mp4", "tagging/b1/dataset_multi/clips/cam/d/q.mp4"]
        for ordered in (keys, list(reversed(keys))):
            self.assertEqual(ev.build_index(ordered)["q"], "tagging/b1/dataset_multi/clips/cam/d/q.mp4")

    def test_order_of_keys_does_not_matter(self) -> None:
        keys = ["tagging/b2/clips/w.mp4", "tagging/b1/clips/w.mp4"]
        self.assertEqual(ev.build_index(keys), ev.build_index(list(reversed(keys))))

    def test_match_reports_unmatched(self) -> None:
        index = {"a": "dataset_multi/clips/a.mp4"}
        rows = [{"clip_id": "a"}, {"clip_id": "b"}, {"clip_id": "a.mp4"}]
        matched, unmatched = ev.match_rows(rows, index)
        self.assertEqual([key for _, key in matched], ["dataset_multi/clips/a.mp4"] * 2)
        self.assertEqual(unmatched, ["b"])


class SamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_indices(self) -> None:
        self.assertEqual(ev.sample_indices(20), [0, 5, 10, 14, 19])
        self.assertEqual(ev.sample_indices(5), [0, 1, 2, 3, 4])
        self.assertEqual(ev.sample_indices(1), [0, 0, 0, 0, 0])
        self.assertEqual(ev.sample_indices(0), [])

    def test_samples_five_evenly_spaced_frames(self) -> None:
        path = os.path.join(self.tmp, "clip.mp4")
        write_video(path, [i * 10 for i in range(20)])
        frames = ev.sample_frames(path)
        self.assertEqual(len(frames), 5)
        for frame, want in zip(frames, [0, 5, 10, 14, 19]):
            # Neighbouring frames differ by 10 grey levels; mp4v shifts them by about 4.
            self.assertLess(abs(float(frame.mean()) - want * 10), 5, f"frame {want}")

    def test_unreadable_video_gives_no_frames(self) -> None:
        path = os.path.join(self.tmp, "bad.mp4")
        with open(path, "wb") as f:
            f.write(b"not a video")
        self.assertEqual(ev.sample_frames(path), [])

    def test_long_side_resize(self) -> None:
        big = np.zeros((1000, 2000, 3), dtype=np.uint8)
        self.assertEqual(ev.fit_long_side(big).shape, (640, 1280, 3))
        tall = np.zeros((2560, 1000, 3), dtype=np.uint8)
        self.assertEqual(ev.fit_long_side(tall).shape, (1280, 500, 3))
        small = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertIs(ev.fit_long_side(small), small)

    def test_clip_local_time(self) -> None:
        import datetime as dt

        want = dt.datetime.fromtimestamp(1771696897).strftime("%H:%M:%S")
        self.assertEqual(ev.clip_local_time("front_side_1771696897_trigger"), want)
        self.assertIsNone(ev.clip_local_time("Burglary001_x264_3"))


class FakeS3:
    """Serves keys from a dict {key: local file}; counts downloads."""

    def __init__(self, files: Dict[str, str]) -> None:
        self.files = files
        self.downloads: List[str] = []

    def get_paginator(self, name: str):
        assert name == "list_objects_v2", name
        files = self.files

        class Pager:
            def paginate(self, Bucket: str, Prefix: str = ""):  # noqa: N803
                keys = sorted(k for k in files if k.startswith(Prefix))
                for i in range(0, len(keys), 2):           # small pages, to exercise paging
                    yield {"Contents": [{"Key": k} for k in keys[i:i + 2]]}
                if not keys:
                    yield {"KeyCount": 0}

        return Pager()

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:  # noqa: N803
        if Key not in self.files:
            raise KeyError(Key)
        self.downloads.append(Key)
        shutil.copyfile(self.files[Key], Filename)


def jsonl_file(path: str, rows: List[dict]) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


class PreparedDirMixin:
    """A fake bucket with two home batches and one public batch, prepared into self.out."""

    def make_bucket(self) -> FakeS3:
        self.src = tempfile.mkdtemp()
        self.out = tempfile.mkdtemp()
        video = os.path.join(self.src, "v.mp4")
        write_video(video, [i * 10 for i in range(20)])
        b1 = [
            {"clip_id": "cam_a_1771696897_trigger", "camera_name": "cam_a", "description": "A man walks in.",
             "video_s3_path": "s3://bucket/stale/cam_a_1771696897_trigger.mp4"},
            {"clip_id": "cam_a_1771696900_trigger", "camera_name": "cam_a", "description": "No special activity"},
            {"clip_id": "cam_a_missing", "camera_name": "cam_a", "description": "Lost clip."},
        ]
        b2 = [
            {"clip_id": "cam_b_1771700000_trigger", "camera_name": "cam_b",
             "description": "Two men break the door. [alert]"},
            {"clip_id": "cam_b_1771700001_trigger", "camera_name": "cam_b", "description": "[delete] blurry"},
        ]
        pub = [{"clip_id": "Burglary001", "camera_name": "uca", "description": "A man steals. [alert]"}]
        files = {
            "tagging/ameer_house_batch_1/analysis_output/vlm_training.jsonl":
                jsonl_file(os.path.join(self.src, "b1.jsonl"), b1),
            "tagging/ameer_house_batch_2/analysis_output/vlm_training.jsonl":
                jsonl_file(os.path.join(self.src, "b2.jsonl"), b2),
            "tagging/uca_dataset_batch/analysis_output/vlm_training.jsonl":
                jsonl_file(os.path.join(self.src, "pub.jsonl"), pub),
            "dataset_multi/clips/cam_a/d/cam_a_1771696897_trigger.mp4": video,
            "tagging/ameer_house_batch_1/dataset_multi/vlm_crops/cam_a/d/cam_a_1771696900_trigger.mp4": video,
            "tagging/ameer_house_batch_1/dataset_multi/clips/cam_a/d/cam_a_1771696900_trigger.mp4": video,
            "dataset_ameer_house/clips/cam_b/cam_b_1771700000_trigger.mp4": video,
            "dataset_ameer_house/clips/cam_b/cam_b_1771700001_trigger.mp4": video,
            "tagging/uca_dataset_batch/dataset_uca/Burglary001.mp4": video,
        }
        return FakeS3(files)

    def cleanup(self) -> None:
        shutil.rmtree(self.src, ignore_errors=True)
        shutil.rmtree(self.out, ignore_errors=True)


def read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class PrepareTest(PreparedDirMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.s3 = self.make_bucket()

    def tearDown(self) -> None:
        self.cleanup()

    def test_prepare_end_to_end_and_cache(self) -> None:
        counts = ev.prepare(self.out, client=self.s3)
        self.assertEqual(counts["rows"], 5)
        self.assertEqual(counts["dropped"], 1)
        self.assertEqual(counts["matched"], 3)
        self.assertEqual(counts["unmatched"], ["cam_a_missing"])
        self.assertEqual(counts["new"], 3)
        self.assertEqual(counts["cached"], 0)
        self.assertEqual(counts["failed"], [])

        manifest = read_jsonl(os.path.join(self.out, "manifest.jsonl"))
        self.assertEqual([(r["batch"], r["clip_id"]) for r in manifest], [
            ("ameer_house_batch_1", "cam_a_1771696897_trigger"),
            ("ameer_house_batch_1", "cam_a_1771696900_trigger"),
            ("ameer_house_batch_2", "cam_b_1771700000_trigger"),
        ])
        first, empty, alert = manifest
        self.assertEqual(first["camera"], "cam_a")
        self.assertEqual(first["ours_label"], "normal")
        self.assertEqual(first["ours_text"], "A man walks in.")
        self.assertEqual(first["s3_key"], "dataset_multi/clips/cam_a/d/cam_a_1771696897_trigger.mp4")
        self.assertEqual(first["frames"], [f"frames/cam_a_1771696897_trigger_{i}.jpg" for i in range(5)])
        self.assertEqual(empty["s3_key"],
                         "tagging/ameer_house_batch_1/dataset_multi/clips/cam_a/d/cam_a_1771696900_trigger.mp4")
        self.assertEqual(empty["ours_label"], "empty")
        self.assertEqual((alert["ours_label"], alert["ours_text"]), ("alert", "Two men break the door."))
        for row in manifest:
            for rel in row["frames"]:
                self.assertTrue(os.path.isfile(os.path.join(self.out, rel)), rel)
        # No mp4 is left behind in the output folder.
        leftovers = [n for _, _, names in os.walk(self.out) for n in names if n.endswith(".mp4")]
        self.assertEqual(leftovers, [])
        self.assertNotIn("tagging/uca_dataset_batch/dataset_uca/Burglary001.mp4", self.s3.downloads)

        before = len(self.s3.downloads)
        again = ev.prepare(self.out, client=self.s3)
        self.assertEqual(len(self.s3.downloads) - before, 2, "only the two home jsonl files are fetched again")
        self.assertTrue(all(k.endswith(".jsonl") for k in self.s3.downloads[before:]))
        self.assertEqual((again["new"], again["cached"]), (0, 3))
        self.assertEqual(read_jsonl(os.path.join(self.out, "manifest.jsonl")), manifest)

    def test_temp_mp4_lives_in_the_system_temp_dir_and_is_removed(self) -> None:
        made: List[tuple] = []
        real = ev.tempfile.mkstemp

        def spy(*args, **kwargs):
            fd, path = real(*args, **kwargs)
            if kwargs.get("suffix") == ".mp4":
                made.append((kwargs.get("dir"), path))
            return fd, path

        ev.tempfile.mkstemp = spy
        try:
            ev.prepare(self.out, client=self.s3)
        finally:
            ev.tempfile.mkstemp = real
        self.assertEqual(len(made), 3)
        for directory, path in made:
            self.assertEqual(directory, tempfile.gettempdir())
            self.assertFalse(os.path.exists(path))

    def test_limit_and_batches(self) -> None:
        counts = ev.prepare(self.out, client=self.s3, limit=1)
        self.assertEqual(counts["new"], 1)
        self.assertEqual(len(read_jsonl(os.path.join(self.out, "manifest.jsonl"))), 1)

        counts = ev.prepare(self.out, client=self.s3, batches=["uca_dataset_batch"])
        self.assertEqual((counts["rows"], counts["new"]), (1, 1))
        manifest = read_jsonl(os.path.join(self.out, "manifest.jsonl"))
        self.assertEqual([r["clip_id"] for r in manifest], ["Burglary001"])

    def test_counts_are_printable(self) -> None:
        counts = ev.prepare(self.out, client=self.s3)
        text = ev.format_prepare_counts(counts)
        self.assertIn("cam_a_missing", text)
        self.assertIn("matched", text)


class RunTest(PreparedDirMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.s3 = self.make_bucket()
        ev.prepare(self.out, client=self.s3)

    def tearDown(self) -> None:
        self.cleanup()

    def test_run_fake_end_to_end_and_resume(self) -> None:
        backend = ev.FakeBackend()
        summary = ev.run_eval(self.out, backend)
        self.assertEqual(backend.calls, 3)
        tag = ev.default_tag(None, "fake", fake=True)
        results = os.path.join(self.out, "results")
        rows = read_jsonl(os.path.join(results, f"{tag}.jsonl"))
        self.assertEqual([r["clip_id"] for r in rows], [
            "cam_a_1771696897_trigger", "cam_a_1771696900_trigger", "cam_b_1771700000_trigger"])
        self.assertEqual(set(rows[0]), set(ev.RESULT_COLUMNS))
        by_id = {r["clip_id"]: r for r in rows}
        self.assertEqual(by_id["cam_a_1771696897_trigger"]["ai_label"], "normal")
        self.assertEqual(by_id["cam_a_1771696897_trigger"]["ai_summary"], "A person walks to the door.")
        self.assertEqual(by_id["cam_a_1771696900_trigger"]["ai_summary"], "No special activity.")
        self.assertEqual(by_id["cam_a_1771696900_trigger"]["ai_people"], 0)
        self.assertIn(by_id["cam_b_1771700000_trigger"]["ai_label"], ("suspicious", "escalation"))
        self.assertEqual(rows[0]["prompt_id"], inference.PROMPT_VERSION)
        self.assertEqual(rows[0]["model"], "fake")
        self.assertEqual(rows[0]["error"], "")

        with open(os.path.join(results, f"{tag}.csv"), encoding="utf-8", newline="") as f:
            table = list(csv.DictReader(f))
        self.assertEqual(len(table), 3)
        self.assertEqual(list(table[0]), list(ev.RESULT_COLUMNS))
        with open(os.path.join(results, f"{tag}.summary.json"), encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["alerts_total"], 1)
        self.assertEqual(saved["alerts_caught"], 1)
        self.assertEqual(saved["empty_exact"], 1)
        self.assertEqual(summary["alerts_caught"], 1)

        again = ev.FakeBackend()
        ev.run_eval(self.out, again)
        self.assertEqual(again.calls, 0, "a finished run is not asked again")
        self.assertEqual(len(read_jsonl(os.path.join(results, f"{tag}.jsonl"))), 3)

    def test_one_error_is_recorded_and_the_run_continues(self) -> None:
        backend = ev.FakeBackend(fail_ids={"cam_a_1771696900_trigger"})
        summary = ev.run_eval(self.out, backend, tag="t1")
        self.assertEqual(backend.calls, 3)
        rows = {r["clip_id"]: r for r in read_jsonl(os.path.join(self.out, "results", "t1.jsonl"))}
        self.assertIn("forced failure", rows["cam_a_1771696900_trigger"]["error"])
        self.assertEqual(rows["cam_b_1771700000_trigger"]["error"], "")
        self.assertEqual(summary["errors"], 1)

        # The next run retries only the failed clip.
        retry = ev.FakeBackend()
        summary = ev.run_eval(self.out, retry, tag="t1")
        self.assertEqual(retry.calls, 1)
        self.assertEqual(summary["errors"], 0)
        rows = read_jsonl(os.path.join(self.out, "results", "t1.jsonl"))
        self.assertEqual(len(rows), 3)

    def test_a_different_prompt_or_model_is_asked_again(self) -> None:
        ev.run_eval(self.out, ev.FakeBackend(), tag="t2")
        other = ev.FakeBackend()
        other.model_name = "fake-2"
        with self.assertRaises(ev.ResultsConflict):
            ev.run_eval(self.out, other, tag="t2")
        self.assertEqual(other.calls, 0)
        ev.run_eval(self.out, other, tag="t2", overwrite=True)
        self.assertEqual(other.calls, 3)

    def test_default_tag_names_prompt_and_model(self) -> None:
        self.assertEqual(ev.default_tag(None, "gpt-4o", fake=False), f"{inference.PROMPT_VERSION}__gpt-4o")
        self.assertEqual(ev.default_tag(None, "ft:gpt/4o mini", fake=False),
                         f"{inference.PROMPT_VERSION}__ft_gpt_4o_mini")
        self.assertEqual(ev.default_tag(None, "fake", fake=True), f"fake-{inference.PROMPT_VERSION}__fake")

    def _backend(self, model: str) -> "ev.FakeBackend":
        b = ev.FakeBackend()
        b.model_name = model
        return b

    def test_two_models_under_the_default_tag_keep_two_files(self) -> None:
        ev.run_eval(self.out, self._backend("model-a"))
        ev.run_eval(self.out, self._backend("model-b"))
        results = os.path.join(self.out, "results")
        files = sorted(n for n in os.listdir(results) if n.endswith(".jsonl"))
        self.assertEqual(files, [f"fake-{inference.PROMPT_VERSION}__model-a.jsonl",
                                 f"fake-{inference.PROMPT_VERSION}__model-b.jsonl"])
        for name, model in zip(files, ("model-a", "model-b")):
            rows = read_jsonl(os.path.join(results, name))
            self.assertEqual(len(rows), 3)
            self.assertEqual({r["model"] for r in rows}, {model})

    def test_an_explicit_tag_with_another_model_is_refused_unless_overwrite(self) -> None:
        ev.run_eval(self.out, self._backend("model-a"), tag="shared")
        path = os.path.join(self.out, "results", "shared.jsonl")
        with open(path, "rb") as f:
            before = f.read()
        refused = self._backend("model-b")
        with self.assertRaises(ev.ResultsConflict) as ctx:
            ev.run_eval(self.out, refused, tag="shared")
        self.assertIn("model-a", str(ctx.exception))
        self.assertIn("model-b", str(ctx.exception))
        self.assertEqual(refused.calls, 0)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), before)

        replaced = self._backend("model-b")
        ev.run_eval(self.out, replaced, tag="shared", overwrite=True)
        self.assertEqual(replaced.calls, 3)
        self.assertEqual({r["model"] for r in read_jsonl(path)}, {"model-b"})

    def test_main_exits_2_and_leaves_the_file_when_the_prompt_differs(self) -> None:
        ev.run_eval(self.out, ev.FakeBackend(), tag="shared")
        path = os.path.join(self.out, "results", "shared.jsonl")
        with open(path, "rb") as f:
            before = f.read()
        prompt_file = os.path.join(self.src, "other.txt")
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write("Other wording for {camera_name}")
        args = ["run", "--dir", self.out, "--fake", "--tag", "shared", "--prompt-file", prompt_file]
        self.assertEqual(ev.main(args), 2)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), before)
        self.assertEqual(ev.main(args + ["--overwrite"]), 0)
        self.assertTrue(all(r["prompt_id"].startswith("file-") for r in read_jsonl(path)))

    def test_changed_prompt_wording_with_the_same_version_asks_every_clip_again(self) -> None:
        original = inference.build_prompt
        try:
            inference.build_prompt = lambda *a, **k: "wording one {x}"
            ev.run_eval(self.out, ev.FakeBackend(), tag="w")
            inference.build_prompt = lambda *a, **k: "wording two {x}"
            with self.assertRaises(ev.ResultsConflict):
                ev.run_eval(self.out, ev.FakeBackend(), tag="w")
            second = ev.FakeBackend()
            ev.run_eval(self.out, second, tag="w", overwrite=True)
            self.assertEqual(second.calls, 3)
            third = ev.FakeBackend()
            ev.run_eval(self.out, third, tag="w")
            self.assertEqual(third.calls, 0)
        finally:
            inference.build_prompt = original
        rows = read_jsonl(os.path.join(self.out, "results", "w.jsonl"))
        self.assertRegex(rows[0]["prompt_sha12"], r"^[0-9a-f]{12}$")

    def test_a_truncated_last_line_is_skipped_with_a_warning(self) -> None:
        path = os.path.join(self.out, "rows.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"clip_id": "a"}\n{"clip_id": "b"')
        with self.assertLogs("box.eval_prompt", level="WARNING"):
            self.assertEqual(ev.read_jsonl(path), [{"clip_id": "a"}])
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"clip_id": "a"}\n{"clip_id": "b"\n{"clip_id": "c"}\n')
        with self.assertRaises(json.JSONDecodeError):
            ev.read_jsonl(path)

    def test_a_run_over_a_truncated_results_file_still_works(self) -> None:
        ev.run_eval(self.out, ev.FakeBackend(), tag="cut")
        path = os.path.join(self.out, "results", "cut.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"clip_id": "half')
        again = ev.FakeBackend()
        ev.run_eval(self.out, again, tag="cut")
        self.assertEqual(again.calls, 0)
        self.assertEqual(len(read_jsonl(path)), 3)

    def test_the_run_says_what_it_will_ask(self) -> None:
        lines: List[str] = []
        ev.run_eval(self.out, ev.FakeBackend(), tag="m1", progress=lines.append)
        self.assertEqual(lines[0], "asking 3 clips (0 already answered, 0 with errors to retry)")
        ev.run_eval(self.out, ev.FakeBackend(fail_ids={"cam_a_1771696900_trigger"}), tag="m2")
        lines = []
        ev.run_eval(self.out, ev.FakeBackend(), tag="m2", progress=lines.append)
        self.assertEqual(lines[0], "asking 1 clips (2 already answered, 1 with errors to retry)")

    def test_a_big_real_run_names_the_model_and_waits_unless_yes(self) -> None:
        slept: List[float] = []
        saved = (ev.time.sleep, ev.CONFIRM_OVER, ev._make_gpt)

        class Stub:
            model_name = "gpt-test"

            def ask(self, row, frames):
                parsed = {"summary": "x", "label": "normal", "people": 0, "vehicle_moving": False}
                return json.dumps(parsed), parsed

        ev.time.sleep = slept.append
        ev.CONFIRM_OVER = 2
        ev._make_gpt = lambda model: Stub()
        try:
            self.assertEqual(ev.main(["run", "--dir", self.out, "--tag", "big1"]), 0)
            self.assertEqual(slept, [5])
            self.assertEqual(ev.main(["run", "--dir", self.out, "--tag", "big2", "--yes"]), 0)
            self.assertEqual(slept, [5])
            self.assertEqual(ev.main(["run", "--dir", self.out, "--fake", "--tag", "big3"]), 0)
            self.assertEqual(slept, [5])
            lines: List[str] = []
            ev.run_eval(self.out, Stub(), tag="big4", progress=lines.append, wait=True)
            self.assertIn("gpt-test", lines[1])
        finally:
            ev.time.sleep, ev.CONFIRM_OVER, ev._make_gpt = saved

    def test_limit(self) -> None:
        backend = ev.FakeBackend()
        ev.run_eval(self.out, backend, tag="t3", limit=2)
        self.assertEqual(backend.calls, 2)

    def test_missing_frame_is_an_error_not_a_crash(self) -> None:
        os.remove(os.path.join(self.out, "frames", "cam_a_1771696897_trigger_3.jpg"))
        backend = ev.FakeBackend()
        summary = ev.run_eval(self.out, backend, tag="t4")
        self.assertEqual(backend.calls, 2)
        self.assertEqual(summary["errors"], 1)

    def test_fake_is_told_the_clip_time(self) -> None:
        backend = ev.FakeBackend()
        ev.run_eval(self.out, backend, tag="t5")
        self.assertIn(ev.clip_local_time("cam_a_1771696897_trigger"), backend.prompts[0])

    def test_summary_command_reprints(self) -> None:
        ev.run_eval(self.out, ev.FakeBackend(), tag="t6")
        summary = ev.load_summary(self.out, "t6")
        self.assertEqual(summary["alerts_total"], 1)
        self.assertIn("alerts caught", ev.format_summary(summary))

    def test_main_run_fake(self) -> None:
        code = ev.main(["run", "--dir", self.out, "--fake", "--tag", "cli"])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.isfile(os.path.join(self.out, "results", "cli.summary.json")))
        self.assertEqual(ev.main(["summary", "--dir", self.out, "--tag", "cli"]), 0)


class PromptFileTest(PreparedDirMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.s3 = self.make_bucket()
        ev.prepare(self.out, client=self.s3)
        self.original = inference.build_prompt
        self.prompt_file = os.path.join(self.src, "prompt.txt")
        with open(self.prompt_file, "w", encoding="utf-8") as f:
            f.write('Camera {camera_name} at {local_time_str}. Reply {"summary": "..."}')

    def tearDown(self) -> None:
        inference.build_prompt = self.original
        self.cleanup()

    def test_prompt_file_is_used_and_restored(self) -> None:
        seen: List[str] = []

        class Spy(ev.FakeBackend):
            def ask(self, row, frames):
                seen.append(inference.build_prompt(row["camera"], 0, "99:99:99", 0, 0))
                return super().ask(row, frames)

        with open(self.prompt_file, encoding="utf-8") as f:
            text = f.read()
        ev.run_eval(self.out, Spy(), prompt_text=text, tag="pf")
        self.assertTrue(seen[0].startswith("Camera cam_a at "))
        self.assertIn('Reply {"summary": "..."}', seen[0])
        self.assertIn(ev.clip_local_time("cam_a_1771696897_trigger"), seen[0])
        self.assertNotIn("{camera_name}", seen[0])
        self.assertIs(inference.build_prompt, self.original)
        rows = read_jsonl(os.path.join(self.out, "results", "pf.jsonl"))
        self.assertEqual(rows[0]["prompt_id"], ev.prompt_id_of(text))
        self.assertRegex(ev.prompt_id_of(text), r"^file-[0-9a-f]{12}$")
        self.assertEqual(ev.default_tag(text, "gpt-4o", fake=False), ev.prompt_id_of(text) + "__gpt-4o")

    def test_restored_after_an_exception(self) -> None:
        class Boom(ev.FakeBackend):
            def ask(self, row, frames):
                raise KeyboardInterrupt  # not an ordinary model error: it ends the run

        with self.assertRaises(KeyboardInterrupt):
            ev.run_eval(self.out, Boom(), prompt_text="x {camera_name}", tag="boom")
        self.assertIs(inference.build_prompt, self.original)

    def test_override_context_manager(self) -> None:
        with ev.prompt_override("cam={camera_name} t={local_time_str}") as state:
            state.local_time = "21:15:00"
            self.assertEqual(inference.build_prompt("porch", 0, "10:00:00", 0, 0), "cam=porch t=21:15:00")
            state.local_time = None
            self.assertEqual(inference.build_prompt("porch", 0, "10:00:00", 0, 0), "cam=porch t=10:00:00")
        self.assertIs(inference.build_prompt, self.original)

    def test_default_prompt_gets_the_clip_time(self) -> None:
        with ev.prompt_override(None) as state:
            state.local_time = "03:04:05"
            text = inference.build_prompt("porch", 0, "10:00:00", 0, 0)
        self.assertIn("local time 03:04:05", text)
        self.assertEqual(text, self.original("porch", 0, "03:04:05", 0, 0))

    def test_real_backend_sends_the_prompt_file_text_and_five_images(self) -> None:
        calls: List[dict] = []
        answer = {"summary": "A man walks.", "label": "normal", "people": 1, "vehicle_moving": False}

        class Message:
            content = json.dumps(answer)

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        class Completions:
            def create(self, **kw):
                calls.append(kw)
                return Response()

        class Chat:
            completions = Completions()

        class Client:
            chat = Chat()

        backend = inference.GptBackend.__new__(inference.GptBackend)
        backend._client = Client()
        backend._model = "gpt-stub"
        backend.model_name = "gpt-stub"
        backend.last_prompt = ""
        backend._response_format = inference.VLM_RESPONSE_FORMAT
        text = "Look at camera {camera_name} and answer in JSON."
        ev.run_eval(self.out, ev.GptAsker(backend), prompt_text=text, tag="real", limit=1)
        self.assertEqual(len(calls), 1)
        content = calls[0]["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Look at camera cam_a and answer in JSON."})
        self.assertEqual(sum(1 for part in content if part["type"] == "image_url"), 5)
        self.assertEqual(len(content), 6)
        rows = read_jsonl(os.path.join(self.out, "results", "real.jsonl"))
        self.assertEqual((rows[0]["ai_label"], rows[0]["model"], rows[0]["error"]), ("normal", "gpt-stub", ""))

    def test_main_prompt_file(self) -> None:
        code = ev.main(["run", "--dir", self.out, "--fake", "--prompt-file", self.prompt_file])
        self.assertEqual(code, 0)
        with open(self.prompt_file, encoding="utf-8") as f:
            tag = ev.default_tag(f.read(), "fake", fake=True)
        self.assertTrue(os.path.isfile(os.path.join(self.out, "results", f"{tag}.jsonl")))


def row(ours: str, ai: str, summary: str = "A man walks.", ours_text: str = "A man walks by.", error: str = "") -> dict:
    return {"ours_label": ours, "ai_label": ai, "ai_summary": summary, "ours_text": ours_text, "error": error}


class SummarizeTest(unittest.TestCase):
    def test_exact_numbers(self) -> None:
        rows = [
            row("alert", "escalation", "Two men break the door."),                     # caught, escalation
            row("alert", "suspicious", "A man in a hood looks around."),               # caught
            row("alert", "normal", "A man walks without a bag."),                      # missed; padding
            row("alert", "escalation", "", error="timeout"),                           # error: not scored
            row("normal", "normal", "A woman walks to the door."),
            row("normal", "suspicious", "A man stands near a parked car."),            # flagged; padding
            row("normal", "normal", "No one is visible."),                             # padding (x2 words, once)
            row("empty", "normal", "No special activity.", ours_text="No special activity"),
            row("empty", "normal", " No special activity. ", ours_text="No special activity"),
            row("empty", "normal", "Nothing happens in the background.", ours_text="No special activity"),
        ]
        s = ev.summarize(rows)
        self.assertEqual(s["rows"], 10)
        self.assertEqual(s["errors"], 1)
        self.assertEqual((s["alerts_caught"], s["alerts_total"]), (2, 3))
        self.assertAlmostEqual(s["alerts_caught_ratio"], 2 / 3)
        self.assertAlmostEqual(s["escalation_share"], 0.5)
        self.assertEqual((s["normal_flagged"], s["normal_total"]), (1, 3))
        self.assertAlmostEqual(s["normal_flagged_ratio"], 1 / 3)
        self.assertEqual((s["empty_exact"], s["empty_total"]), (2, 3))
        self.assertEqual((s["padding"], s["padding_total"]), (4, 9))
        # ai words: 5+7+6+6+7+4+3+3+5 = 46 over 9; ours: 6 rows x 4 + 3 rows x 3 = 33 over 9.
        self.assertAlmostEqual(s["words_ai_mean"], 46 / 9)
        self.assertAlmostEqual(s["words_ours_mean"], 33 / 9)

    def test_empty_input(self) -> None:
        s = ev.summarize([])
        self.assertEqual(s["alerts_total"], 0)
        self.assertIsNone(s["alerts_caught_ratio"])
        self.assertIsNone(s["escalation_share"])
        self.assertIsNone(s["words_ai_mean"])
        self.assertIn("alerts caught", ev.format_summary(s))


class FakeBackendTest(unittest.TestCase):
    def test_alert_label_follows_the_clip_id_hash(self) -> None:
        fake = ev.FakeBackend()
        labels = set()
        for i in range(10):
            clip_id = f"clip_{i}"
            _, parsed = fake.ask({"clip_id": clip_id, "camera": "c", "ours_label": "alert"}, [])
            want = "escalation" if ev.clip_hash(clip_id) % 2 == 0 else "suspicious"
            self.assertEqual(parsed["label"], want)
            labels.add(parsed["label"])
        self.assertEqual(labels, {"escalation", "suspicious"})

    def test_answers_match_the_schema(self) -> None:
        raw, parsed = ev.FakeBackend().ask({"clip_id": "x", "camera": "c", "ours_label": "empty"}, [])
        self.assertEqual(json.loads(raw), parsed)
        self.assertEqual(set(parsed), {"summary", "label", "people", "vehicle_moving"})
        self.assertEqual((parsed["summary"], parsed["people"], parsed["label"]), ("No special activity.", 0, "normal"))


if __name__ == "__main__":
    unittest.main()
