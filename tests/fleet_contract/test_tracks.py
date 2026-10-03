import pytest

from home_guard_project.fleet_contract.tracks import (
    Keyframe, Track, box_at, boxes_at, frame_at, frame_time, tracks_from_weak_labels, validate_tracks)

B1 = [0.1, 0.1, 0.3, 0.3]
B5 = [0.5, 0.3, 0.7, 0.5]


def kf(frame, t, box, enabled=True):
    return Keyframe(frame=frame, t_sec=t, xyxy=list(box), enabled=enabled)


def approx(a, b):
    assert a == pytest.approx(b)


def test_frame_helpers():
    assert frame_time(10, 5.0) == 2.0
    assert frame_at(2.0, 5.0) == 10
    assert frame_at(2.09, 5.0) == 10


def test_halfway_in_time_for_kept_box():
    tr = Track("a", "person", [kf(1, 1.0, B1), kf(5, 5.0, B5)])
    mid = box_at(tr, 3.0)
    for got, want in zip(mid, [0.3, 0.2, 0.5, 0.4]):
        approx(got, want)


def test_interpolation_follows_time_not_frame_count():
    # frame 3 is "halfway" by frame count but sits at t=1.5 of a 1.0..5.0 span: weight 0.125
    tr = Track("a", "person", [kf(1, 1.0, B1), kf(5, 5.0, B5)])
    approx(box_at(tr, 1.5)[0], 0.1 + 0.4 * 0.125)
    uneven = Track("a", "person", [kf(1, 0.0, [0, 0, 0.1, 0.1]), kf(5, 4.0, [0.4, 0.4, 0.5, 0.5])])
    approx(box_at(uneven, 1.0)[0], 0.1)


def test_hidden_segment():
    tr = Track("a", "person", [kf(1, 1.0, B1, enabled=False), kf(5, 5.0, B5)])
    assert box_at(tr, 1.0) is None
    for t in (2.0, 3.0, 4.0):
        assert box_at(tr, t) is None
    assert box_at(tr, 5.0) == B5


def test_before_first_and_after_last():
    tr = Track("a", "person", [kf(1, 1.0, B1), kf(5, 5.0, B5)])
    assert box_at(tr, 0.5) is None
    assert box_at(tr, 1.0) == B1
    assert box_at(tr, 9.0) == B5
    gone = Track("a", "person", [kf(1, 1.0, B1), kf(5, 5.0, B5, enabled=False)])
    assert box_at(gone, 9.0) is None
    assert box_at(Track("a", "person", []), 1.0) is None


def test_single_keyframe():
    tr = Track("a", "person", [kf(2, 2.0, B1)])
    assert box_at(tr, 1.9) is None
    assert box_at(tr, 2.0) == B1
    assert box_at(tr, 100.0) == B1


def test_boxes_at():
    a = Track("a", "person", [kf(1, 1.0, B1)])
    b = Track("b", "dog", [kf(1, 3.0, B5)])
    assert boxes_at([a, b], 2.0) == [("person", B1)]
    assert boxes_at([a, b], 3.0) == [("person", B1), ("dog", B5)]


def _frames(n, fn, label="person"):
    return [(i, i * 0.5, fn(i, label)) for i in range(n)]


def test_weak_one_moving_person_is_one_sparse_track():
    frames = _frames(10, lambda i, l: [(l, [0.1 + 0.05 * i, 0.2, 0.2 + 0.05 * i, 0.4])])
    tracks = tracks_from_weak_labels(frames)
    assert len(tracks) == 1 and tracks[0].label == "person" and tracks[0].source == "suggestion"
    kfs = tracks[0].keyframes
    assert len(kfs) <= 3 and kfs[0].frame == 0 and kfs[-1].frame == 9 and all(k.enabled for k in kfs)
    for i in range(10):
        got = box_at(tracks[0], i * 0.5)
        assert max(abs(g - w) for g, w in zip(got, frames[i][2][0][1])) <= 0.02 + 1e-9


def test_weak_two_people_two_tracks():
    frames = _frames(6, lambda i, l: [(l, [0.1, 0.1, 0.2, 0.3]), (l, [0.7, 0.5, 0.9, 0.9])])
    tracks = tracks_from_weak_labels(frames)
    assert len(tracks) == 2
    assert len({t.track_id for t in tracks}) == 2


def test_weak_person_leaving_gets_disabled_keyframe():
    frames = [(i, i * 0.5, [("person", [0.1, 0.1, 0.3, 0.3])] if i < 4 else []) for i in range(8)]
    tracks = tracks_from_weak_labels(frames)
    assert len(tracks) == 1
    last = tracks[0].keyframes[-1]
    assert last.enabled is False and last.frame == 4
    assert box_at(tracks[0], 1.5) is not None and box_at(tracks[0], 2.0) is None


def test_weak_different_classes_do_not_link():
    frames = [(0, 0.0, [("person", B1)]), (1, 0.5, [("dog", B1)])]
    assert len(tracks_from_weak_labels(frames)) == 2


def test_validate_tracks():
    ok = Track("a", "person", [kf(1, 1.0, B1), kf(5, 5.0, B5)])
    assert validate_tracks([ok], 10.0) == []
    bad = Track("b", "unicorn", [kf(1, 2.0, [0.5, 0.5, 0.4, 0.6]), kf(2, 2.0, [0, 0, 1.2, 1]), kf(3, 99.0, B1)])
    text = " ".join(validate_tracks([bad], 10.0))
    for needle in ("label", "sorted", "x1<x2", "0..1", "clip"):
        assert needle in text, needle


# ---------------------------------------------------------------- box_at is the one interpolation (fix round D2)
# The review's counterexamples: the analyzer must agree with these, not the other way round.

def test_box_moves_towards_a_disabled_keyframe_by_time():
    # an enabled box at x=.1 (t=0) followed by a disabled keyframe at x=.5 (t=1): the segment still interpolates
    # towards the next keyframe's position; the disabled keyframe hides the box from its own time on
    tr = Track("a", "person", [kf(0, 0.0, [0.1, 0.1, 0.2, 0.2]), kf(1, 1.0, [0.5, 0.1, 0.6, 0.2], enabled=False)])
    approx(box_at(tr, 0.5)[0], 0.3)
    assert box_at(tr, 1.0) is None and box_at(tr, 3.0) is None


def test_nonuniform_timestamps_interpolate_by_time_not_frame():
    # keyframes (frame 0, t=0, x=.1) and (frame 10, t=2, x=.5); native frame 5 decoded at t=.5 -> .2, not .3
    tr = Track("a", "person", [kf(0, 0.0, [0.1, 0.1, 0.2, 0.2]), kf(10, 2.0, [0.5, 0.1, 0.6, 0.2])])
    approx(box_at(tr, 0.5)[0], 0.2)


def test_disabled_keyframe_hides_until_the_next_keyframe_then_reappears():
    tr = Track("a", "person", [kf(0, 0.0, B1), kf(2, 1.0, B1, enabled=False), kf(4, 2.0, B5), kf(6, 3.0, B5)])
    assert box_at(tr, 1.5) is None
    assert box_at(tr, 2.0) == B5 and box_at(tr, 2.5) == B5


# ---------------------------------------------------------------- weak-label linking (fix round L2)

def _person(x, y=0.2, w=0.1, h=0.3):
    return ("person", [x, y, x + w, y + h])


def test_weak_person_missed_on_two_frames_is_one_track():
    frames = [(i, i / 7, [] if i in (3, 6) else [_person(0.1 + 0.02 * i)]) for i in range(10)]
    tracks = tracks_from_weak_labels(frames)
    assert len(tracks) == 1
    assert all(k.enabled for k in tracks[0].keyframes)  # a short miss is bridged, never hidden
    assert box_at(tracks[0], 3 / 7) is not None and box_at(tracks[0], 6 / 7) is not None


def test_weak_two_people_crossing_stay_two_tracks():
    frames = [(i, i / 7, [_person(0.1 + 0.05 * i, y=0.20), _person(0.8 - 0.05 * i, y=0.25)]) for i in range(15)]
    tracks = tracks_from_weak_labels(frames)
    assert len(tracks) == 2
    assert all(len(t.keyframes) >= 2 and all(k.enabled for k in t.keyframes) for t in tracks)


def test_weak_one_frame_flicker_is_dropped_but_a_short_clip_keeps_it():
    frames = [(i, i / 7, [_person(0.1)] + ([_person(0.7)] if i == 4 else [])) for i in range(10)]
    assert len(tracks_from_weak_labels(frames)) == 1
    assert len(tracks_from_weak_labels([(0, 0.0, [_person(0.1)]), (2, 0.3, [_person(0.7)])])) == 2


def test_weak_small_box_matches_by_centre_when_iou_is_low():
    # a far-away person (2% of the frame wide) shifting by its own width: IoU 0, centre distance small
    frames = [(i, i / 7, [("person", [0.5 + 0.02 * i, 0.1, 0.52 + 0.02 * i, 0.15])]) for i in range(6)]
    assert len(tracks_from_weak_labels(frames)) == 1


def _fixture_frames():
    import re
    from pathlib import Path

    out = []
    for path in sorted((Path(__file__).parent / "fixtures" / "collect_labels").glob("*.txt")):
        index = int(re.search(r"_f(\d+)\.txt$", path.name).group(1))
        boxes = []
        for line in path.read_text().splitlines():
            cls, xc, yc, w, h = line.split()
            if cls == "0":
                xc, yc, w, h = map(float, (xc, yc, w, h))
                boxes.append(("person", [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]))
        out.append((index, index / 7.0, boxes))
    return out


def test_weak_real_back_door_clip_is_not_fragmented():
    # collect.meta.json's clip (dataset_ameer_house back_door_1790944263_trigger): 3-5 people per sampled frame,
    # its 32 real weak-label files are fixtures/collect_labels/
    frames = _fixture_frames()
    assert len(frames) == 32
    tracks = tracks_from_weak_labels(frames)
    assert 3 <= len(tracks) <= 6, [(t.track_id, [(k.frame, k.enabled) for k in t.keyframes]) for t in tracks]
