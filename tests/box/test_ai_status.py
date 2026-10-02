from __future__ import annotations

import os
import tempfile
import unittest

from home_guard_project.box.ai_status import AiStatus, objects_from_result, read_status

NOW = 1_800_000_000.0
PERSON = {"label": "person", "conf": 0.71, "box": [0.1, 0.2, 0.3, 0.9]}


class _Box:
    def __init__(self, cls_id, conf, xyxyn):
        self.cls, self.conf, self.xyxyn = [cls_id], [conf], [xyxyn]


class _Result:
    names = {0: "person", 2: "car"}

    def __init__(self, boxes):
        self.boxes = boxes


class AiStatusTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "logs", "ai_status.json")

    def test_objects_carry_label_confidence_and_a_box_in_fractions(self) -> None:
        result = _Result([_Box(0, 0.7123, [0.1, 0.2, 0.30004, 0.9]), _Box(2, 0.5, [0.5, 0.5, 0.75, 0.8])])
        self.assertEqual(objects_from_result(result), [
            {"label": "person", "conf": 0.71, "box": [0.1, 0.2, 0.3, 0.9]},
            {"label": "car", "conf": 0.5, "box": [0.5, 0.5, 0.75, 0.8]},
        ])
        self.assertEqual(objects_from_result(_Result([])), [])
        self.assertEqual(objects_from_result(object()), [])

    def test_a_detection_is_kept_when_later_pictures_are_empty(self) -> None:
        status = AiStatus(self.path, min_interval=0)
        status.detection("front", [PERSON], now=NOW)
        status.detection("front", [], now=NOW + 5)
        camera = read_status(self.path)["cameras"]["front"]
        self.assertEqual((camera["ts"], camera["checked_ts"], camera["objects"]), (NOW, NOW + 5, [PERSON]))

    def test_detections_are_not_written_more_than_once_an_interval(self) -> None:
        status = AiStatus(self.path, min_interval=1.0)
        status.detection("front", [PERSON], now=NOW)
        status.detection("front", [], now=NOW + 0.3)            # too soon: not written
        self.assertEqual(read_status(self.path)["cameras"]["front"]["checked_ts"], NOW)
        status.detection("front", [], now=NOW + 1.5)
        self.assertEqual(read_status(self.path)["cameras"]["front"]["checked_ts"], NOW + 1.5)

    def test_a_decision_is_written_at_once_and_only_the_newest_are_kept(self) -> None:
        status = AiStatus(self.path, min_interval=60)
        status.detection("front", [PERSON], now=NOW)
        for i in range(55):
            status.decision("front", ["person"], f"event {i}", "[send_message]", sent=i % 2 == 0, now=NOW + i)
        decisions = read_status(self.path)["decisions"]
        self.assertEqual(len(decisions), 50)
        self.assertEqual((decisions[-1]["summary"], decisions[-1]["sent"]), ("event 54", True))
        self.assertEqual(decisions[0]["summary"], "event 5")

    def test_a_decision_records_why_nothing_was_sent(self) -> None:
        status = AiStatus(self.path)
        status.decision("yard", ["car"], "A car is parked.", "[none]", sent=False, false_positive=True, now=NOW)
        status.decision("door", ["person"], "A person at the door.", "[send_message]", sent=False,
                        error="Forbidden: bot was kicked from the group chat", now=NOW + 1)
        first, second = read_status(self.path)["decisions"]
        self.assertTrue(first["false_positive"])
        self.assertEqual(second["error"], "Forbidden: bot was kicked from the group chat")

    def test_the_ai_is_shown_thinking_until_it_decides(self) -> None:
        status = AiStatus(self.path, min_interval=60)
        status.thinking("door", ["person"], now=NOW)
        self.assertEqual(read_status(self.path)["thinking"], {"camera": "door", "labels": ["person"], "ts": NOW})
        status.decision("door", ["person"], "A person at the door.", "[send_message]", sent=True, now=NOW + 3)
        self.assertIsNone(read_status(self.path)["thinking"])

    def test_a_missing_or_damaged_file_reads_as_empty(self) -> None:
        self.assertEqual(read_status(self.path), {})
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(read_status(self.path), {})


if __name__ == "__main__":
    unittest.main()
