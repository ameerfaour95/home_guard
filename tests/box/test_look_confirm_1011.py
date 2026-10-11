"""2026-10-10 23:54, the pergola at night: "מה קורה בחוץ יש משהו מעניין" got "בפרגולה יש אדם ליד נגרר" and a photo with an
English caption. Nobody was there: the detector gave stacked chairs "person" 0.52 on the 960-wide picture, and the
vision model was told it as a fact. The owner: the detector looks at the camera's full resolution, and a person it
is not sure of is no assumption at all - no "maybe a person", no photo. And the caption is in his language."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from home_guard_project.box import live_view
from home_guard_project.box.brain import grounded_look as gl
from home_guard_project.box.brain.memory import ChatState
from home_guard_project.box.brain.receipts import ReceiptBook
from home_guard_project.box.brain.registry import CameraState, HouseSnapshot
from home_guard_project.box.brain.tools import Services, ToolContext, caption_words, check_camera, look_around

NOW = 1_791_665_691.0
CH3 = "ameer_v2_ch3"
CHAIRS = {"label": "person", "conf": 0.52, "box": [0.803, 0.3975, 0.8416, 0.5482]}     # the 23:54 "person"
SURE = {"label": "person", "conf": 0.91, "box": [0.40, 0.30, 0.48, 0.70]}
SAID_EN = "A person stands near a trailer under the pergola. The scene is quiet."
SAID_HE = "אדם עומד ליד נגרר מתחת לפרגולה. הסצנה שקטה."


class ZoomDetector:
    """The look detector: *found* on the whole picture, *second* on any second look (None: it cannot run)."""

    def __init__(self, found: List[Dict[str, Any]], second: Optional[float] = 0.0) -> None:
        self.found, self.second = found, second
        self.paths: List[str] = []
        self.zooms: List[Any] = []

    def __call__(self, path: str) -> List[Dict[str, Any]]:
        self.paths.append(path)
        return list(self.found)

    def zoom(self, path: str, box: Any) -> Optional[float]:
        self.zooms.append((path, tuple(box)))
        return self.second


class DetectTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.photo = os.path.join(self.root, "photo.jpg")
        self.full = os.path.join(self.root, "photo_full.jpg")
        for p in (self.photo, self.full):
            with open(p, "wb") as f:
                f.write(b"\xff\xd8jpeg")

    def test_the_chairs_are_no_person_when_the_second_look_finds_none(self) -> None:
        det = ZoomDetector([CHAIRS], second=0.0)
        doubtful: List[gl.Seen] = []
        source, seen = gl.detect(CH3, self.photo, NOW, det, full_path=self.full, doubtful=doubtful)
        self.assertEqual((source, seen), ("detector", []))
        self.assertEqual([d.conf for d in doubtful], [0.52])
        self.assertEqual(det.paths, [self.full])                     # the full resolution, not the 960 picture
        self.assertEqual(det.zooms, [(self.full, tuple(CHAIRS["box"]))])

    def test_a_person_the_second_look_confirms_stays(self) -> None:
        source, seen = gl.detect(CH3, self.photo, NOW, ZoomDetector([CHAIRS], second=gl.ZOOM_CONF + 0.05),
                                 full_path=self.full)
        self.assertEqual([s.label for s in seen], ["person"])

    def test_a_second_look_just_below_the_bar_is_not_enough(self) -> None:
        _source, seen = gl.detect(CH3, self.photo, NOW, ZoomDetector([CHAIRS], second=gl.ZOOM_CONF - 0.01),
                                  full_path=self.full)
        self.assertEqual(seen, [])

    def test_without_the_full_resolution_the_bar_is_higher(self) -> None:
        # A garden lamp read 0.77 on the second look at a 704-wide picture, 0.00 at full resolution.
        lamp = ZoomDetector([CHAIRS], second=0.77)
        self.assertEqual(gl.detect(CH3, self.photo, NOW, lamp)[1], [])
        self.assertEqual(len(gl.detect(CH3, self.photo, NOW, lamp, full_path=self.full)[1]), 1)
        self.assertEqual(len(gl.detect(CH3, self.photo, NOW, ZoomDetector([CHAIRS], second=gl.ZOOM_CONF_SMALL))[1]), 1)

    def test_no_second_look_means_not_confirmed(self) -> None:
        self.assertEqual(gl.detect(CH3, self.photo, NOW, ZoomDetector([CHAIRS], second=None))[1], [])
        self.assertEqual(gl.detect(CH3, self.photo, NOW, lambda _p: [CHAIRS])[1], [])      # a detector with no zoom

    def test_a_sure_person_needs_no_second_look(self) -> None:
        det = ZoomDetector([SURE], second=0.0)
        self.assertEqual(len(gl.detect(CH3, self.photo, NOW, det)[1]), 1)
        self.assertEqual(det.zooms, [])

    def test_at_most_max_zooms_second_looks_the_surest_first(self) -> None:
        lamps = [dict(CHAIRS, conf=0.40 + i / 100) for i in range(gl.MAX_ZOOMS + 2)]
        det = ZoomDetector(lamps, second=0.9)
        _source, seen = gl.detect(CH3, self.photo, NOW, det, full_path=self.full)
        self.assertEqual(len(det.zooms), gl.MAX_ZOOMS)
        self.assertEqual(sorted((s.conf for s in seen), reverse=True), [s.conf for s in seen])
        self.assertEqual(len(seen), gl.MAX_ZOOMS)                    # the rest: not confirmed, not told

    def test_without_a_full_copy_the_photo_itself(self) -> None:
        det = ZoomDetector([SURE])
        gl.detect(CH3, self.photo, NOW, det, full_path=os.path.join(self.root, "gone_full.jpg"))
        self.assertEqual(det.paths, [self.photo])

    def test_facts_keep_the_doubtful_out_of_the_prompt(self) -> None:
        facts = gl.grounded_facts(CH3, self.photo, NOW, detector=ZoomDetector([CHAIRS]), full_path=self.full)
        self.assertEqual(facts.count(gl.PERSON), 0)
        self.assertIn("- people: 0", gl.facts_text(facts))
        self.assertNotIn("0.52", gl.facts_text(facts))
        self.assertEqual(len(facts.record()["doubtful_people"]), 1)

    def test_the_zoom_crop(self) -> None:
        img = np.zeros((1520, 2592, 3), np.uint8)
        crop, (cx, cy) = gl.zoom_crop(img, CHAIRS["box"])
        h, w = crop.shape[:2]
        self.assertEqual(w, h)                                       # square around the box
        self.assertGreaterEqual(w, gl.ZOOM_MIN_SIDE)
        self.assertTrue(0 < cx < w and 0 < cy < h)                    # the candidate's centre is inside
        edge, (ex, ey) = gl.zoom_crop(img, [0.995, 0.99, 1.0, 1.0])      # at the corner: kept inside the picture
        self.assertEqual(edge.shape[:2], (gl.ZOOM_MIN_SIDE, gl.ZOOM_MIN_SIDE))
        self.assertTrue(0 < ex <= gl.ZOOM_MIN_SIDE and 0 < ey <= gl.ZOOM_MIN_SIDE)


class FakeVision:
    def __init__(self, said: str, people: int = 1) -> None:
        self.said, self.people = said, people

    def look(self, camera, images, guard=False, question="", what="a live photo", facts=""):
        return {"ok": True, "description": self.said, "quality": "clear", "people": self.people, "label": "normal",
                "why": ""}


class Deliver:
    def __init__(self) -> None:
        self.photos: List[Any] = []

    def photo(self, chat_id, path, caption=""):
        self.photos.append((path, caption))
        return {"ok": True, "message_id": len(self.photos)}


class LookTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.photo = os.path.join(self.root, "photo.jpg")
        self.full = os.path.join(self.root, "photo_full.jpg")
        for p in (self.photo, self.full):
            with open(p, "wb") as f:
                f.write(b"\xff\xd8jpeg")

    def ctx(self, text: str, vision: Any, detect: Any, lang: str = "he") -> ToolContext:
        self.deliver = Deliver()
        services = Services(roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"),
                            feedback_dir=self.root, work_dir=self.root, mute=None, deliver=self.deliver,
                            now=lambda: NOW, grab_photo=lambda cam: {"image": self.photo, "full": self.full},
                            vision=vision, detect=detect, zones_path=os.path.join(self.root, "zones.yaml"))
        snap = HouseSnapshot(now=NOW, mode="guard", mode_ends=NOW + 3600, mode_started=NOW - 3600, start_hour=0,
                             end_hour=0, cameras=(CameraState(CH3, True, ("פרגולה",), live=True),))
        return ToolContext(turn_id="-5:1", chat_id="-5", speaker={"user_id": 1, "name": "Ameer"}, text=text,
                           lang=lang, mode="guard", snapshot=snap, state=ChatState(), services=services,
                           book=ReceiptBook(os.path.join(self.root, ".r"), now=lambda: NOW))

    def test_the_2354_look_tells_no_person_and_sends_no_photo(self) -> None:
        det = ZoomDetector([CHAIRS], second=0.0)
        out = look_around(self.ctx("מה קורה בחוץ יש משהו מעניין", FakeVision(SAID_EN), det), {})
        row = out["cameras"][0]
        self.assertEqual(row["people"], 0)
        self.assertEqual(self.deliver.photos, [])
        self.assertNotIn("person", row["description"].lower().replace("no people", ""))
        self.assertNotIn("may be", row["description"])
        self.assertEqual(det.paths, [self.full])

    def test_check_camera_with_the_chairs(self) -> None:
        out = check_camera(self.ctx("מה קורה בפרגולה", FakeVision(SAID_HE), ZoomDetector([CHAIRS])), {"camera": CH3})
        self.assertEqual(out["people"], 0)
        self.assertNotIn("אדם עומד", out["description"])
        self.assertNotIn("ייתכן", out["description"])
        self.assertEqual(out["detector"]["people"], 0)

    def test_a_confirmed_person_is_shown_with_a_hebrew_caption(self) -> None:
        look_around(self.ctx("יש מישהו בחוץ?", FakeVision(SAID_EN), ZoomDetector([SURE])), {})
        self.assertEqual(len(self.deliver.photos), 1)
        caption = self.deliver.photos[0][1]
        self.assertTrue(caption.startswith("פרגולה · "), caption)
        self.assertNotRegex(caption, r"[A-Za-z]")                    # no English sentence for a Hebrew owner

    def test_a_hebrew_description_is_kept_in_the_caption(self) -> None:
        look_around(self.ctx("יש מישהו בחוץ?", FakeVision(SAID_HE), ZoomDetector([SURE])), {})
        self.assertIn("אדם עומד ליד נגרר מתחת לפרגולה", self.deliver.photos[0][1])

    def test_caption_words_follow_the_owners_language(self) -> None:
        he = self.ctx("x", None, None)
        en = self.ctx("x", None, None, lang="en")
        self.assertEqual(caption_words(he, SAID_EN), "")
        self.assertEqual(caption_words(he, SAID_HE), "אדם עומד ליד נגרר מתחת לפרגולה")
        self.assertEqual(caption_words(en, SAID_EN), "A person stands near a trailer under the pergola")
        self.assertEqual(caption_words(en, SAID_HE), "")


class GrabFullTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.cameras = os.path.join(self.dir, "cameras.yaml")
        with open(self.cameras, "w", encoding="utf-8") as f:
            yaml.safe_dump({"cameras": {"gate": "rtsp://x"}}, f)
        self.zones = os.path.join(self.dir, "zones.yaml")
        self.out = os.path.join(self.dir, "live")

    def grab_of(self, width: int, height: int):
        def grab(url: str, path: str) -> bool:
            import cv2

            return bool(cv2.imwrite(path, np.full((height, width, 3), 200, np.uint8)))
        return grab

    def test_a_main_stream_picture_keeps_its_full_copy_and_the_look_stays_960(self) -> None:
        import cv2

        shot = live_view.grab_masked("gate", self.cameras, self.out, now=lambda: NOW, grab=self.grab_of(2592, 1520),
                                     zones_path=self.zones, keep_full=True)
        self.assertEqual(cv2.imread(shot["image"]).shape[:2], (562, 960))
        self.assertEqual(cv2.imread(shot["full"]).shape[:2], (1520, 2592))
        self.assertTrue(shot["full"].endswith("_full.jpg"))
        self.assertEqual([n for n in os.listdir(self.out) if ".tmp" in n], [])

    def test_the_full_copy_is_masked_too(self) -> None:
        import cv2

        from home_guard_project.data_collection import zones

        zones.write_scene_maps({"gate": {"areas": [
            {"name": "w", "kind": "black", "points": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]]}]}},
            zones.scene_maps_path_for(self.zones))
        shot = live_view.grab_masked("gate", self.cameras, self.out, now=lambda: NOW, grab=self.grab_of(2592, 1520),
                                     zones_path=self.zones, keep_full=True)
        for key in ("image", "full"):
            img = cv2.imread(shot[key])
            w = img.shape[1]
            self.assertLess(int(img[:, int(w * 0.6):].max()), 30, key)
            self.assertGreater(int(img[:, : int(w * 0.4)].min()), 150, key)

    def test_a_small_grab_has_no_full_copy(self) -> None:
        shot = live_view.grab_masked("gate", self.cameras, self.out, now=lambda: NOW, grab=self.grab_of(704, 576),
                                     zones_path=self.zones, keep_full=True)
        self.assertNotIn("full", shot)
        self.assertTrue(os.path.isfile(shot["image"]))

    def test_without_keep_full_nothing_changes(self) -> None:
        import cv2

        shot = live_view.grab_masked("gate", self.cameras, self.out, now=lambda: NOW, grab=self.grab_of(2592, 1520),
                                     zones_path=self.zones)
        self.assertNotIn("full", shot)
        self.assertEqual(cv2.imread(shot["image"]).shape[1], 2592)   # the grab is the caller's business

    def test_old_full_copies_are_removed(self) -> None:
        os.makedirs(self.out)
        old = os.path.join(self.out, "gate_1_aa_full.jpg")
        keep = os.path.join(self.out, "gate_1_aa.jpg")
        for p in (old, keep):
            with open(p, "wb") as f:
                f.write(b"x")
            os.utime(p, (NOW - 7200, NOW - 7200))
        live_view.prune_full_copies(self.out, NOW)
        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.exists(keep))                          # only the full copies


if __name__ == "__main__":
    unittest.main()
