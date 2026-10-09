"""The supported COCO classes and their contiguous training-export IDs."""

from typing import Optional


COCO_NAMES: dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
    7: "truck", 14: "bird", 15: "cat", 16: "dog",
}
CONTIGUOUS: dict[int, int] = {coco: index for index, coco in enumerate(sorted(COCO_NAMES))}


def name_to_coco(name: str) -> Optional[int]:
    return next((coco for coco, label in COCO_NAMES.items() if label == name), None)


# The two class-id systems of YOLO label files. Weak labels (the box's data collection, dataset_*/yolo/labels) keep
# the sparse COCO ids; training labels (exports, the unified dataset) use the contiguous 0-8 ids. Every conversion
# between them goes through these functions.
COCO_IDS = "coco"
CONTIGUOUS_IDS = "contiguous"
COCO_OF_CONTIGUOUS: dict[int, int] = {index: coco for coco, index in CONTIGUOUS.items()}


def coco_to_contiguous(coco: int) -> Optional[int]:
    """COCO id -> contiguous training id; None for a class outside the nine."""
    return CONTIGUOUS.get(coco)


def contiguous_to_coco(index: int) -> Optional[int]:
    """Contiguous training id -> COCO id; None outside 0-8."""
    return COCO_OF_CONTIGUOUS.get(index)


def class_name(cls_id: int, system: str) -> Optional[str]:
    """The class name of a label-file id read in `system` (COCO_IDS or CONTIGUOUS_IDS); None when unknown."""
    if system == COCO_IDS:
        return COCO_NAMES.get(cls_id)
    if system == CONTIGUOUS_IDS:
        coco = contiguous_to_coco(cls_id)
        return COCO_NAMES[coco] if coco is not None else None
    raise ValueError(f"unknown class-id system {system!r}")
