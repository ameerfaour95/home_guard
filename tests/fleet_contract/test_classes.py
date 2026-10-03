from home_guard_project.fleet_contract.classes import COCO_NAMES, CONTIGUOUS, name_to_coco


def test_class_maps():
    assert COCO_NAMES == {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 14: "bird", 15: "cat", 16: "dog"}
    assert CONTIGUOUS == {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 7: 5, 14: 6, 15: 7, 16: 8}
    assert name_to_coco("truck") == 7
    assert name_to_coco("horse") is None
