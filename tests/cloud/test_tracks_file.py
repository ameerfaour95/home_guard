"""The box tracker's <stem>.tracks.json read as editable tracks (tagstudio/tracks_file.py), on real files written by
home_guard analysis/replay_tracks.py (oct7_tracks): returns join the track they came back to, flicker is left out,
gaps and ends are hidden keyframes."""
import json
import shutil
from pathlib import Path

from home_guard_project.cloud.tagstudio import tracks_file as tf
from home_guard_project.fleet_contract.tracks import box_at, validate_tracks

FIXTURES = Path(__file__).parent / "fixtures" / "tracks"


def _doc(name):
    return json.loads((FIXTURES / f"{name}.tracks.json").read_text(encoding="utf-8"))


def test_return_continues_the_track_it_came_back_to():
    doc = _doc("ameer_week_0_1_ch3_1791355310_alert")      # tracker: car 1, person 2, person 3 (prev_id 2), person 4
    tracks = tf.read_tracks(doc, 7.0)
    assert [(t.track_id, t.label, t.source) for t in tracks] == [("t-1", "car", "yolo"), ("t-2", "person", "yolo"),
                                                                ("t-3", "person", "yolo")]
    person = tracks[1]
    # one person: seen at 11, missing 12-15, seen at 16, gone 17-43, back at 44 (the tracker's return), gone at 46
    assert [(k.frame, k.enabled) for k in person.keyframes] == [(11, True), (12, False), (16, True), (17, False),
                                                                (44, True), (45, True), (46, False)]
    assert box_at(person, 14 / 7.0) is None and box_at(person, 44 / 7.0) is not None
    # the car is in every look up to the clip's last frame: no hidden end
    assert tracks[0].keyframes[-1].frame == 59 and tracks[0].keyframes[-1].enabled
    assert validate_tracks(tracks, doc["frames"] / 7.0) == []


def test_keyframes_are_thinned_but_boxes_stay_within_tolerance():
    doc = _doc("ameer_week_0_1_ch3_1791355310_alert")
    car = tf.read_tracks(doc, 7.0)[0]
    raw = next(t for t in doc["tracks"] if t["id"] == 1)["boxes"]
    assert len(car.keyframes) < len(raw) / 4
    for b in raw:
        got = box_at(car, b["frame"] / 7.0)
        assert max(abs(x - y) for x, y in zip(got, b["box"])) <= tf.TOLERANCE + 1e-6


def test_person_and_vehicle_and_times_without_fps():
    doc = _doc("ameer_week_0_1_ch2_1791439138_alert")
    tracks = tf.read_tracks(doc)          # no fps: times from the looks' own timestamps
    assert [t.label for t in tracks] == ["person", "car"]
    first_ts = min(lk["ts"] for lk in doc["looks"])
    look8 = next(lk for lk in doc["looks"] if lk["frame"] == 8)
    assert abs(tracks[0].keyframes[0].t_sec - (look8["ts"] - first_ts)) < 1e-3


def test_empty_file_and_unconfirmed_flicker():
    assert tf.read_tracks(_doc("ameer_week_0_1_ch3_1791461339_alert"), 7.0) == []
    doc = _doc("ameer_week_0_1_ch2_1791439138_alert")
    for t in doc["tracks"]:
        t["confirmed"] = False
    assert tf.read_tracks(doc, 7.0) == []
    assert len(tf.read_tracks(doc, 7.0, confirmed_only=False)) == 2
    assert tf.read_tracks({"tracks": [{"id": 1, "cls": 56, "confirmed": True, "boxes": []}]}) == []
    assert tf.read_tracks("not a document") == []


def _live_doc():
    """A live file as box clip_tracks.py writes it: source, params, sparse looks, shown / max_conf / entity."""
    def box(frame, x):
        return {"frame": frame, "ts": 1000.0 + frame / 7.0, "box": [x, 0.3, x + 0.1, 0.8]}
    return {"clip": "clips/cam/2026-10-09/cam_1000_alert.mp4", "camera": "cam", "start_local": "10:00",
            "source": "live", "params": {"person_conf": 0.5, "alert_person_conf": 0.8, "tracker": "tb1"},
            "clip_start_ts": 1000.0, "clip_end_ts": 1000.0 + 34 / 7.0, "frames": 35,
            "looks": [{"frame": f, "ts": 1000.0 + f / 7.0, "detections": 2, "tracks": []}
                      for f in (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30)],
            "tracks": [
                {"id": 41, "kind": "person", "cls": 0, "first_seen": 1000.0, "last_seen": 1001.3, "hits": 4,
                 "confirmed": True, "shown": True, "max_conf": 0.91, "prev_id": None, "returns": 0, "entity": "P1",
                 "boxes": [box(0, 0.1), box(3, 0.15), box(6, 0.2), box(9, 0.25)]},
                # the same person after an occlusion: a new tracker track the event book re-attached to P1
                {"id": 57, "kind": "person", "cls": 0, "first_seen": 1003.0, "last_seen": 1004.3, "hits": 4,
                 "confirmed": True, "shown": True, "max_conf": 0.88, "prev_id": None, "returns": 0, "entity": "P1",
                 "boxes": [box(24, 0.55), box(27, 0.6), box(30, 0.65)]},
                {"id": 12, "kind": "vehicle", "cls": 2, "first_seen": 1000.0, "last_seen": 1004.3, "hits": 11,
                 "confirmed": True, "shown": False, "max_conf": 0.95, "prev_id": None, "returns": 0,
                 "boxes": [{"frame": f, "ts": 1000.0 + f / 7.0, "box": [0.6, 0.6, 0.9, 0.9]}
                           for f in (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30)]},
                {"id": 58, "kind": "person", "cls": 0, "first_seen": 1002.0, "last_seen": 1002.0, "hits": 1,
                 "confirmed": False, "shown": True, "max_conf": 0.52, "prev_id": None, "returns": 0,
                 "boxes": [box(15, 0.8)]}],
            "stats": {"person_tracks": 2, "entities": ["P1"]}}


def test_a_live_file_from_the_box():
    tracks = tf.read_tracks(_live_doc(), 7.0)
    # P1 over two tracker tracks is one track; the parked car is kept; the one-look flicker is left out
    assert [t.label for t in tracks] == ["person", "car"]
    person = tracks[0]
    assert [(k.frame, k.enabled) for k in person.keyframes] == [(0, True), (9, True), (12, False), (24, True),
                                                                (30, True), (31, False)]
    assert box_at(person, 15 / 7.0) is None and box_at(person, 25 / 7.0) is not None
    assert validate_tracks(tracks, 35 / 7.0) == []


def test_found_next_to_meta_or_video(tmp_path):
    name = "ameer_week_0_1_ch2_1791439138_alert"
    meta = tmp_path / "meta" / f"{name}.meta.json"
    video = tmp_path / "clips" / f"{name}.mp4"
    assert tf.local_tracks([str(meta), str(video)], 7.0) == []
    video.parent.mkdir()
    shutil.copy(FIXTURES / f"{name}.tracks.json", video.parent / f"{name}.tracks.json")
    assert [t.label for t in tf.local_tracks([str(meta), str(video)], 7.0)] == ["person", "car"]
