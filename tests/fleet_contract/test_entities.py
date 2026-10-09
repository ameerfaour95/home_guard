"""Track entity ids (P1, CAR2, A1): valid shapes, the next free id per kind, filling unnamed tracks in order of first
appearance, and validation of a bad id."""
from home_guard_project.fleet_contract.tracks import (Keyframe, Track, fill_entities, next_entity, valid_entity,
                                                      validate_tracks)


def tr(tid, label, t, entity=None):
    return Track(tid, label, [Keyframe(int(t * 10), t, [0.1, 0.1, 0.2, 0.2])], entity=entity)


def test_valid_shapes():
    assert all(valid_entity(v) for v in ("P1", "P999", "CAR2", "A1"))
    assert not any(valid_entity(v) for v in ("P0x", "p1", "X1", "P1000", "CAR", "", None, 3, "CAR12345"))


def test_next_free_per_kind_and_fill_in_order():
    tracks = [tr("a", "person", 2.0, "P1"), tr("b", "car", 1.0), tr("c", "person", 0.5), tr("d", "dog", 3.0),
              tr("e", "bicycle", 0.1), tr("f", "truck", 4.0, "CAR1")]
    assert next_entity(tracks, "person") == "P2" and next_entity(tracks, "bus") == "CAR2"
    assert next_entity(tracks, "bicycle") is None
    fill_entities(tracks)
    assert [t.entity for t in tracks] == ["P1", "CAR2", "P2", "A1", None, "CAR1"]


def test_a_bad_entity_is_a_problem():
    assert validate_tracks([tr("a", "person", 0.0, "P1")], 5.0) == []
    problems = validate_tracks([tr("a", "person", 0.0, "Z9")], 5.0)
    assert problems and "entity 'Z9'" in problems[0]
