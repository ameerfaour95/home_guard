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


def test_found_next_to_meta_or_video(tmp_path):
    name = "ameer_week_0_1_ch2_1791439138_alert"
    meta = tmp_path / "meta" / f"{name}.meta.json"
    video = tmp_path / "clips" / f"{name}.mp4"
    assert tf.local_tracks([str(meta), str(video)], 7.0) == []
    video.parent.mkdir()
    shutil.copy(FIXTURES / f"{name}.tracks.json", video.parent / f"{name}.tracks.json")
    assert [t.label for t in tf.local_tracks([str(meta), str(video)], 7.0)] == ["person", "car"]
