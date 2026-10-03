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
