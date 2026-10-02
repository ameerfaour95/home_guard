from pathlib import Path
import tempfile
import unittest
from home_guard_project.box.app.model import (
    State,
    parse_activity,
    read_activity,
    tail,
    redact,
)
from home_guard_project.box.app.backend import (
    Answers,
    Sequence,
    SimulatedBackend,
    STEPS,
)
from home_guard_project.box.app.strings import tr


class AppModelTest(unittest.TestCase):
    def test_heartbeat_combines_live_and_outbox_without_mutation(self):
        payload = dict(
            site="home",
            collector_running=True,
            mode="inference",
            clips_live=3,
            clips_outbox=4,
            disk_free_gb=25.5,
        )
        state = State.from_heartbeat(payload, cameras=["front"])
        self.assertEqual(state.waiting, 7)
        self.assertTrue(state.collecting)
        self.assertEqual(state.mode, "inference")
        self.assertEqual(payload["clips_live"], 3)
        self.assertEqual(State.from_heartbeat({}).waiting, 0)
        self.assertFalse(State.from_heartbeat({}).collecting)

    def test_collector_activity(self):
        for kind in ("trigger", "random"):
            event = parse_activity(
                f"12:00:00 INFO [front_door] {kind} saved: clip.mp4 (no VLM)"
            )
            self.assertEqual(event.text, "Clip saved from front door")
        self.assertEqual(
            parse_activity("Starting data_collection (overlay: config)").text,
            tr("restart"),
        )
        self.assertIsNone(parse_activity("person x1 score=1.0"))
        self.assertIsNone(parse_activity("[front] VLM crop saved: clip.mp4"))

    def test_upload_files_are_not_mislabeled_as_clips(self):
        event = parse_activity(
            "Done. Uploaded: 13  |  Skipped (same size): 0  |  Failed: 0"
        )
        self.assertEqual(event.text, "13 files sent to the online folder")
        self.assertTrue(event.upload)
        self.assertEqual(
            parse_activity("Moved 4 finished clip(s) (16 files) to the outbox.").text,
            "4 clips ready for the online folder",
        )
        event = parse_activity(
            "Done. Uploaded: 13  |  Skipped (same size): 0  |  Failed: 2"
        )
        self.assertFalse(event.upload)
        self.assertEqual(event.text, tr("upload_error"))
        self.assertIsNone(parse_activity("Completed 1.5 MiB/9.0 MiB (15 MiB/s)"))

    def test_redaction_applies_to_details(self):
        value = redact("ERROR rtsp://example.invalid/path password=sample token=sample")
        self.assertNotIn("example.invalid", value)
        self.assertNotIn("sample", value)
        self.assertLessEqual(len(redact("x" * 1000)), 500)

    def test_rotating_logs_and_time_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runner.log").write_text(
                "2026-10-01 10:00:00 Starting data_collection"
            )
            (root / "collector-2026-10-02.log").write_text(
                "12:10:00 INFO [front] trigger saved: clip.mp4"
            )
            (root / "upload-2026-10-02.log").write_text(
                "2026-10-02 12:00:00 Done. Uploaded: 13 | Failed: 0"
            )
            events, upload = read_activity(root)
            self.assertEqual(events[0].text, "Clip saved from front")
            self.assertEqual(upload, "2026-10-02 12:00:00")
            self.assertEqual(len(events), 3)
            (root / "upload-2026-10-03.log").write_text(
                "2026-10-03 12:00:00 Done. Uploaded: 3 | Failed: 0"
            )
            self.assertEqual(read_activity(root)[1], "2026-10-03 12:00:00")

    def test_bounded_tail_discards_partial_line_and_handles_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log"
            self.assertEqual(tail(path), [])
            path.write_text("x" * 100 + "\nlast line\n")
            self.assertEqual(tail(path, 20), ["last line"])


class WizardSequenceTest(unittest.TestCase):
    def test_steps_run_in_order_and_warnings_do_not_stop_setup(self):
        sequence = Sequence(SimulatedBackend(), Answers(find_cameras=False))
        while not sequence.done:
            sequence.advance()
        self.assertEqual([result.step for result in sequence.results], list(STEPS))
        self.assertEqual(sequence.results[3].status, "WARN")
        self.assertEqual(sequence.results[-1].status, "WARN")
        self.assertTrue(sequence.results[-1].checks)
        self.assertFalse(sequence.failed)
        self.assertIsNone(sequence.advance())

    def test_network_failure_stops_future_steps_and_new_sequence_retries(self):
        sequence = Sequence(SimulatedBackend(failure=True), Answers())
        while not sequence.done:
            sequence.advance()
        self.assertEqual(len(sequence.results), 3)
        self.assertTrue(sequence.failed)
        retry = Sequence(SimulatedBackend(), Answers())
        while not retry.done:
            retry.advance()
        self.assertFalse(retry.failed)

    def test_backend_exception_becomes_recoverable_failure(self):
        class Broken:
            def execute(self, step, answers):
                raise RuntimeError("sample backend failure")

        sequence = Sequence(Broken(), Answers())
        self.assertEqual(sequence.advance().message_key, "step_error")
        self.assertTrue(sequence.done)

    def test_answers_repr_does_not_expose_secrets(self):
        answers = Answers(wifi_password="sample", camera_password="sample")
        self.assertNotIn("sample", repr(answers))
