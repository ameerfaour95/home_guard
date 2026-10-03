"""The supported COCO classes and their contiguous training-export IDs."""

from typing import Optional


COCO_NAMES: dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
    7: "truck", 14: "bird", 15: "cat", 16: "dog",
}
CONTIGUOUS: dict[int, int] = {coco: index for index, coco in enumerate(sorted(COCO_NAMES))}


def name_to_coco(name: str) -> Optional[int]:
    return next((coco for coco, label in COCO_NAMES.items() if label == name), None)
