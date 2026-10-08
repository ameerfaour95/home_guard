"""Label folders in the two class-id systems: weak labels (sparse COCO ids) and training labels (contiguous 0-8) are
each read as what they are, never one as the other (tagstudio/boxes.py over fleet_contract.classes)."""
from home_guard_project.cloud.tagstudio import boxes
from home_guard_project.fleet_contract.classes import COCO_IDS, CONTIGUOUS_IDS

ROW = "{} 0.5 0.5 0.2 0.4\n"


def _dataset(tmp_path, folder, rows_by_frame, classes_txt=True, declare=None):
    labels = tmp_path / "yolo" / "labels" / folder
    labels.mkdir(parents=True)
    if classes_txt:
        (tmp_path / "yolo" / "classes.txt").write_text(
            "\n".join(f"{i} {n}" for i, n in enumerate(boxes.DEFAULT_CLASSES)), encoding="utf-8")
    if declare:
        (labels / boxes.ID_FILE).write_text(declare, encoding="utf-8")
    for frame, ids in rows_by_frame.items():
        (labels / f"clip_a_f{frame:04d}.txt").write_text("".join(ROW.format(i) for i in ids), encoding="utf-8")
    return str(tmp_path), str(labels)


def _labels(dataset):
    return sorted({tr.label for tr in boxes.dataset_tracks(dataset, "clip_a", 7.0)})


def test_contiguous_training_labels(tmp_path):
    # 5 = truck, 7 = cat in the training ids
    dataset, folder = _dataset(tmp_path, "house", {0: [5], 1: [5], 2: [5]})
    assert boxes.id_system(dataset, folder) == CONTIGUOUS_IDS
    assert _labels(dataset) == ["truck"]


def test_coco_weak_labels_are_recognised_by_a_coco_only_id(tmp_path):
    # 16 = dog exists only in COCO: the folder's 7 is a truck, not a cat
    dataset, folder = _dataset(tmp_path, "box_cam", {0: [7, 16], 1: [7, 16], 2: [7, 16]})
    assert boxes.id_system(dataset, folder) == COCO_IDS
    assert _labels(dataset) == ["dog", "truck"]


def test_declared_system_wins(tmp_path):
    dataset, folder = _dataset(tmp_path, "box_cam", {0: [5], 1: [5], 2: [5]}, declare="coco\n")
    assert boxes.id_system(dataset, folder) == COCO_IDS
    assert _labels(dataset) == ["bus"]


def test_ambiguous_folder_follows_the_dataset_kind(tmp_path):
    # only ids that exist in both systems: a training dataset (classes.txt) reads them contiguous ...
    dataset, folder = _dataset(tmp_path / "a", "house", {0: [7], 1: [7], 2: [7]})
    assert boxes.id_system(dataset, folder) == CONTIGUOUS_IDS
    assert _labels(dataset) == ["cat"]
    # ... a box's weak-label folder (no classes.txt) reads them as COCO
    dataset, folder = _dataset(tmp_path / "b", "box_cam", {0: [7], 1: [7], 2: [7]}, classes_txt=False)
    assert boxes.id_system(dataset, folder) == COCO_IDS
    assert _labels(dataset) == ["truck"]


def test_contiguous_only_id_marks_the_folder(tmp_path):
    dataset, folder = _dataset(tmp_path, "house", {0: [8], 1: [8], 2: [8]}, classes_txt=False)
    assert boxes.id_system(dataset, folder) == CONTIGUOUS_IDS
    assert _labels(dataset) == ["dog"]


def test_unknown_coco_classes_are_dropped(tmp_path):
    dataset, folder = _dataset(tmp_path, "box_cam", {0: [0, 9, 56], 1: [0], 2: [0]}, declare="coco")
    assert [n for n, _ in boxes.read_boxes(folder + "/clip_a_f0000.txt", boxes.DEFAULT_CLASSES, COCO_IDS)] == ["person"]
