from __future__ import annotations

import json
import os
import tempfile
import unittest

from home_guard_project.box import ground_replay as rp
from home_guard_project.box import scene_map as sm
from home_guard_project.data_collection import zones as z

CAM = "ameer_week_0_1_ch1"
LEFT = ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))
RIGHT = ((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0))
SCENE = sm.SceneMap(CAM, areas=(sm.Area("yard", sm.MINE, "yard", LEFT),
                                sm.Area("their side", sm.WATCH, "yard", RIGHT, owner="neighbour")))


def meta(ts, label, why="", summary="a man"):
    stem = f"{CAM}_{int(ts)}_alert"
    return {"camera_name": CAM, "clip_path": f"clips/{CAM}/{stem}.mp4", "clip_start_ts": ts - 5, "clip_end_ts": ts + 5,
            "clip_start_local": "2026-10-07T10:00:00", "trigger_ts": ts,
            "alert": {"label": label, "why": why, "summary": summary, "people": 1}}


def looks_at(xs, ts):
    return {"camera": CAM, "t0": ts - 5, "t1": ts + 5,
            "looks": [[ts - 5 + i * 0.5, [[0, 0.9, x - 0.05, 0.2, x + 0.05, 0.6]]] for i, x in enumerate(xs)]}


class ReplayTest(unittest.TestCase):
    def test_suppressed_and_added(self):
        metas = [meta(1000.0, "suspicious", why="walks around at night"),      # on their side: suppressed
                 meta(5000.0, "normal")]                                          # comes in: added
        looks = {rp.stem_of(metas[0]): looks_at([0.8] * 8, 1000.0),
                 rp.stem_of(metas[1]): looks_at([0.8 - 0.04 * i for i in range(14)], 5000.0)}
        rows = rp.replay(metas, looks, lambda cam: SCENE)
        self.assertEqual([(r["base"]["sent"], r["with_ground"]["sent"]) for r in rows], [(True, False), (False, True)])
        total = rp.summarize(rows)["total"]
        self.assertEqual((total["suppressed"], total["added"]), (1, 1))
        self.assertIn("not ours", rows[0]["with_ground"]["reason"])

    def test_without_a_map_both_runs_agree(self):
        metas = [meta(1000.0, "suspicious", why="walks around at night")]
        rows = rp.replay(metas, {rp.stem_of(metas[0]): looks_at([0.8] * 8, 1000.0)}, lambda cam: None)
        self.assertEqual(rows[0]["base"]["sent"], rows[0]["with_ground"]["sent"])

    def test_maps_from_a_scene_maps_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene_maps.yaml")
            z.write_scene_maps({CAM: SCENE.stored()}, path)
            self.assertEqual(rp.maps_from_file(path)(CAM).areas, SCENE.areas)

    def test_load_metas_keeps_the_dates_asked_for(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "meta", CAM))
            for i, day in enumerate(("2026-10-06", "2026-10-07")):
                m = meta(1000.0 + i, "normal")
                m["clip_start_local"] = f"{day}T10:00:00"
                with open(os.path.join(tmp, "meta", CAM, f"{i}.meta.json"), "w", encoding="utf-8") as f:
                    json.dump(m, f)
            self.assertEqual(len(rp.load_metas(tmp)), 2)
            self.assertEqual(len(rp.load_metas(tmp, ["2026-10-07"])), 1)


if __name__ == "__main__":
    unittest.main()
