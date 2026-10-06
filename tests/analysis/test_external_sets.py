"""Outside YOLO sets: class mapping, Roboflow duplicates, untagged-object check, dataset build with gray copies."""
import json
import os

import cv2
import numpy as np
import yaml

from home_guard_project.analysis import external_sets as ex


def _img(path, color=(0, 0, 255)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.zeros((64, 64, 3), np.uint8)
    img[:] = color
    cv2.imwrite(path, img)


def _lbl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("".join(" ".join(str(v) for v in r) + "\n" for r in rows))


def test_map_names_sorts_ours_ignored_and_ambiguous():
    m = ex.map_names(["Person", "honda crv", "deer", "Animal", "dog"])
    assert m == {0: 0, 1: 2, 2: "ignore", 3: "ambiguous", 4: 8}


def test_unique_images_keeps_one_per_roboflow_original():
    paths = ["a/x_jpg.rf.abc123.jpg", "a/x_jpg.rf.def456.jpg", "a/y_png.rf.0f0f.jpg", "a/plain.jpg"]
    assert len(ex.unique_images(paths)) == 3
    assert ex.original_stem("a/x_jpg.rf.abc123.jpg") == "x"


def test_label_path_swaps_the_last_images_dir():
    assert ex.label_path("C:/d/images/train/cam/f.jpg").replace("\\", "/") == "C:/d/labels/train/cam/f.txt"
    assert ex.label_path("C:/images/set/train/images/f.png").replace("\\", "/") == "C:/images/set/train/labels/f.txt"


def test_check_frame_leaves_out_untagged_objects_only():
    mapping = {0: 0, 1: "ignore", 2: "ambiguous"}
    person = (0, 0.5, 0.5, 0.2, 0.4)
    det_person = [0, 0.9, 0.4, 0.3, 0.6, 0.7]
    # tagged person, detector agrees
    assert ex.check_frame([person], mapping, [det_person]) == (None, [person])
    # detector sees a person nobody tagged
    assert ex.check_frame([], mapping, [det_person])[0] == "untagged_person"
    # below the confidence bar: kept as background
    assert ex.check_frame([], mapping, [[0, 0.5, 0.4, 0.3, 0.6, 0.7]]) == (None, [])
    # a deer box covers the detector's "dog" on it; the deer box itself is dropped
    deer = (1, 0.5, 0.5, 0.2, 0.4)
    assert ex.check_frame([deer], mapping, [[8, 0.8, 0.4, 0.3, 0.6, 0.7]]) == (None, [])
    # generic "Animal" may be a dog: frame left out
    assert ex.check_frame([(2, 0.5, 0.5, 0.1, 0.1)], mapping, [])[0] == "ambiguous_class"
    # labelled car, detector says truck: same family, kept
    car = (0, 0.5, 0.5, 0.2, 0.4)
    assert ex.check_frame([car], {0: 2}, [[5, 0.9, 0.4, 0.3, 0.6, 0.7]])[0] is None


def test_build_writes_ext_sources_and_a_newhouse_list(tmp_path):
    yolo = tmp_path / "dataset" / "yolo"
    _img(str(yolo / "images" / "cam" / "own.jpg"))
    (yolo / "frames.txt").write_text("images/cam/own.jpg\n")
    sets = []
    for name in ("houseA", "houseB"):
        d = tmp_path / "raw" / name
        d.mkdir(parents=True)
        (d / "data.yaml").write_text(yaml.safe_dump({"names": ["person", "raccoon"]}))
        _img(str(d / "train" / "images" / "a_jpg.rf.1.jpg"))
        _img(str(d / "train" / "images" / "a_jpg.rf.2.jpg"))  # augmented copy of a
        _lbl(str(d / "train" / "labels" / "a_jpg.rf.1.txt"), [[0, 0.5, 0.5, 0.2, 0.2], [1, 0.1, 0.1, 0.1, 0.1]])
        _img(str(d / "valid" / "images" / "b_jpg.rf.3.jpg"))
        _lbl(str(d / "valid" / "labels" / "b_jpg.rf.3.txt"), [])
        sets.append(str(d))
    dets = {ex.image_key(d, p): [] for d in sets for p in ex.set_images(d)}
    assert "houseA/valid/images/b_jpg.rf.3.jpg" in dets
    dets["houseA/valid/images/b_jpg.rf.3.jpg"] = [[0, 0.95, 0.1, 0.1, 0.3, 0.5]]  # a person nobody tagged

    rep = ex.build(str(yolo), sets, dets, eval_set="houseB")

    a = rep["sets"]["houseA"]
    assert (a["images"], a["kept"], dict(a["left_out"])) == (2, 1, {"untagged_person": 1})
    assert rep["sets"]["houseB"]["kept"] == 2
    lab = (yolo / "labels" / "ext_houseA" / "a.txt").read_text().split()
    assert lab[0] == "0" and len(lab) == 5  # raccoon dropped, person kept
    assert (yolo / "external_frames.txt").read_text().split() == ["images/ext_houseA/a.jpg"]
    assert sorted((yolo / "newhouse_eval.txt").read_text().split()) == ["images/ext_houseB/a.jpg",
                                                                       "images/ext_houseB/b.jpg"]
    assert (yolo / "frames.txt").read_text() == "images/cam/own.jpg\n"  # the dataset's own list untouched
    assert json.loads((yolo / "external_report.json").read_text())["newhouse_eval"] == 2


def test_is_gray():
    img = np.zeros((32, 32, 3), np.uint8)
    img[:] = (90, 90, 90)
    assert ex.is_gray(img)
    img[:] = (0, 0, 255)
    assert not ex.is_gray(img)
