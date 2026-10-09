from home_guard_project.fleet_contract.classes import COCO_NAMES, CONTIGUOUS, name_to_coco


def test_class_maps():
    assert COCO_NAMES == {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 14: "bird", 15: "cat", 16: "dog"}
    assert CONTIGUOUS == {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 7: 5, 14: 6, 15: 7, 16: 8}
    assert name_to_coco("truck") == 7
    assert name_to_coco("horse") is None


def test_coco_contiguous_round_trip():
    from home_guard_project.fleet_contract.classes import (COCO_IDS, CONTIGUOUS_IDS, class_name, coco_to_contiguous,
                                                           contiguous_to_coco)
    for coco, name in COCO_NAMES.items():
        index = coco_to_contiguous(coco)
        assert contiguous_to_coco(index) == coco
        assert class_name(coco, COCO_IDS) == name == class_name(index, CONTIGUOUS_IDS)
    # the ids that mean different classes in the two systems: never read one as the other
    assert class_name(7, COCO_IDS) == "truck" and class_name(7, CONTIGUOUS_IDS) == "cat"
    assert class_name(5, COCO_IDS) == "bus" and class_name(5, CONTIGUOUS_IDS) == "truck"
    assert class_name(16, COCO_IDS) == "dog" and class_name(16, CONTIGUOUS_IDS) is None
    assert class_name(4, COCO_IDS) is None and class_name(4, CONTIGUOUS_IDS) == "bus"
    assert coco_to_contiguous(4) is None and contiguous_to_coco(9) is None
