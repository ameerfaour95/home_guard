import unittest

import numpy as np

from home_guard_project.analysis import replay_tracks as rt


class Box:
    def __init__(self, cls, conf, xyxy):
        self.cls, self.conf, self.xyxy = cls, conf, [np.array(xyxy, dtype=float)]


class Result:
    def __init__(self, boxes):
        self.boxes = boxes


def walker(frames):
    """A fake detector: a person walking left to right, missing from frames 10-11 (a short dropout)."""
    def detect(frame):
        i = detect.i
        detect.i += 1
        if i in (10, 11):
            return Result([])
        x = 50 + 8 * i
        return Result([Box(0, 0.9, [x, 100, x + 40, 300])])
    detect.i = 0
    return detect


class ReplayTests(unittest.TestCase):
    def test_a_short_dropout_keeps_one_track(self):
        frames = [np.zeros((576, 704, 3), np.uint8)] * 30
        times = [1000 + i / 7 for i in range(30)]
        result = rt.replay(frames, times, walker(frames))
        people = [t for t in result["tracks"] if t["kind"] == "person"]
        self.assertEqual(len(people), 1)
        self.assertEqual(len(people[0]["boxes"]), 28)
        stats = rt.split_stats(result, eye_people=1)
        self.assertEqual((stats["person_tracks"], stats["fragments"], stats["duplicates"], stats["excess_vs_eye"]),
                         (1, [], [], 0))

    def test_fragments_and_duplicates_are_counted(self):
        def track(tid, t0, t1, x, confirmed=True):
            boxes = [{"frame": i, "ts": t0 + i * 0.5, "box": [x, 0.2, x + 0.1, 0.6]} for i in range(int((t1 - t0) / 0.5) + 1)]
            return {"id": tid, "kind": "person", "cls": 0, "first_seen": t0, "last_seen": t1, "hits": len(boxes),
                    "confirmed": confirmed, "prev_id": None, "returns": 0, "boxes": boxes}
        result = {"looks": [], "tracks": [track(1, 0, 3, 0.4), track(2, 4, 6, 0.42), track(3, 4, 6, 0.41),
                                          track(4, 10, 10, 0.9, confirmed=False)]}
        stats = rt.split_stats(result, eye_people=1)
        self.assertEqual([(f["from"], f["to"]) for f in stats["fragments"]], [(1, 2), (1, 3)])
        self.assertEqual(stats["duplicates"][0]["tracks"], [2, 3])
        self.assertEqual((stats["flickers"], stats["excess_vs_eye"], stats["most_people_at_once"]), (1, 2, 2))

    def test_frame_times_spread_like_the_box(self):
        self.assertEqual(rt.frame_times({"clip_start_ts": 10, "clip_end_ts": 12}, 3), [10, 11, 12])


if __name__ == "__main__":
    unittest.main()
