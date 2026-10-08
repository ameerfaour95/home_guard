"""The camera map editor without Qt: the map dict it builds, and its transport to the box (local and ssh)."""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from home_guard_project.box import scene_interview as si
from home_guard_project.box import scene_map as sm
from home_guard_project.box.app import scene_backend as sb
from home_guard_project.box.app import scene_model as model

LAWN = ((0., .5), (.5, .5), (.5, 1.), (0., 1.))
HOUSE = ((.5, .1), (1., .1), (1., .5), (.5, .5))
STREET = ((0., 0.), (.5, 0.), (.5, .5), (0., .5))
REGIONS = (model.Region(1, LAWN, .25), model.Region(2, HOUSE, .2), model.Region(3, STREET, .25))


class BuildMapTest(unittest.TestCase):
    def test_answers_hand_areas_and_lines_become_a_valid_scene_map(self) -> None:
        hands = [model.HandArea([[.6, .6], [.9, .6], [.9, .9]], "hide", "החלון"),
                 model.HandArea([[.1, .1], [.2, .1], [.2, .2]], None, "no answer yet")]
        line = model.boundary_toward((.5, 0.), (.5, 1.), (.25, .75), "המעקה")
        scene = model.build_map("front", REGIONS, {1: "mine", 2: "neighbour", 3: model.SKIP},
                                {1: "the lawn", 2: ""}, hands, [line])
        self.assertEqual([(a["name"], a["kind"], a.get("owner"), a["zone"]) for a in scene["areas"]],
                         [("the lawn", "mine", None, "yard"), ("area 2", "watch_no_alert", "neighbour", "other"),
                          ("החלון", "black", None, "window")])
        self.assertEqual(scene["areas"][0]["points"], [list(p) for p in LAWN])
        self.assertEqual(scene["lines"], [{"name": "המעקה", "a": [.5, 0.], "b": [.5, 1.], "inward": "right"}])
        parsed = sm.SceneMap.from_dict("front", scene)                # the engine accepts it as it is
        self.assertEqual([a.ground for a in parsed.areas], ["mine", "neighbour", ""])
        self.assertEqual(parsed.lines[0].crossing((.75, .5), (.25, .5)), "in")       # towards the lawn: ours
        self.assertEqual(model.counts(scene), {"mine": 1, "neighbour": 1, "public": 0, "hide": 1, "lines": 1})

    def test_a_boundary_place_is_ours_and_its_line_comes_with_it(self) -> None:
        line = model.Boundary((.5, .1), (.5, .5), "left", "המעקה")
        scene = model.build_map("front", REGIONS, {2: model.BOUNDARY}, {2: "המעקה"}, lines=[line])
        self.assertEqual([(a["name"], a["kind"], a["zone"]) for a in scene["areas"]], [("המעקה", "mine", "fence")])
        self.assertEqual(scene["lines"][0]["inward"], "left")
        self.assertEqual(model.counts(scene)["mine"], 1)

    def test_unanswered_and_unknown_regions_are_left_out(self) -> None:
        scene = model.build_map("front", REGIONS, {2: model.UNKNOWN})
        self.assertEqual(scene, {"camera": "front", "areas": [], "lines": []})
        self.assertEqual(model.counts(scene)["lines"], 0)

    def test_the_rest_of_the_picture_is_carried_over(self) -> None:
        scene = model.build_map("front", REGIONS, {1: "public"}, rest=sm.WATCH, rest_owner=sm.NEIGHBOUR)
        self.assertEqual((scene["rest"], scene["rest_owner"]), (sm.WATCH, sm.NEIGHBOUR))
        self.assertEqual(scene["areas"][0]["owner"], "public")
        self.assertEqual(model.rest_after({}, scene), sm.NEIGHBOUR)
        self.assertEqual(model.rest_after({"watched": [[0, 0], [1, 0], [1, 1]]}, model.build_map("front")), sm.NEIGHBOUR)
        self.assertEqual(model.rest_after({}, model.build_map("front")), "unmapped")

    def test_the_map_survives_the_command_line_round_trip(self) -> None:
        scene = model.build_map("front", REGIONS, {1: "mine", 2: "hide"}, {1: "הדשא 'שלנו'"})
        encoded = model.encode_map(scene)
        self.assertRegex(encoded, r"^[A-Za-z0-9+/=]+$")                 # no spaces or quotes on the command line
        self.assertEqual(si.decode_map_b64(encoded), json.loads(json.dumps(scene)))
        again = sm.SceneMap.from_dict("front", si.decode_map_b64(encoded))
        self.assertEqual(again.to_dict()["areas"], sm.SceneMap.from_dict("front", scene).to_dict()["areas"])

    def test_restart_only_when_the_frame_mask_changes(self) -> None:
        hidden = model.build_map("front", REGIONS, {2: "hide"})
        self.assertTrue(model.restart_expected({}, hidden))
        self.assertFalse(model.restart_expected({"areas": hidden["areas"]}, hidden))
        self.assertFalse(model.restart_expected({}, model.build_map("front", REGIONS, {1: "mine"})))
        self.assertTrue(model.restart_expected({"watched": [[0, 0], [1, 0], [1, 1]]}, model.build_map("front")))

    def test_the_cameras_map_now_becomes_editable_areas_and_lines(self) -> None:
        current = {"watched": [[0, 0], [.5, 0], [.5, 1]],
                   "areas": [{"name": "gate", "kind": "watch_no_alert", "owner": "neighbour", "zone": "gate",
                              "points": [[.6, .6], [.9, .6], [.9, .9]]}],
                   "lines": [{"name": "railing", "a": [.5, 0], "b": [.5, 1], "inward": "right"}]}
        hands, lines = model.from_current(current)
        self.assertEqual([(h.choice, h.name, h.zone) for h in hands],
                         [("neighbour", "gate", "gate"), ("mine", sm.WATCHED_NAME, "other")])
        self.assertEqual((lines[0].name, lines[0].inward), ("railing", "right"))
        scene = model.build_map("front", (), {}, {}, hands, lines)
        self.assertEqual(scene["areas"][0]["zone"], "gate")                # a stored zone is kept
        self.assertEqual(scene["areas"][1]["name"], sm.WATCHED_NAME)

    def test_our_side_of_a_line(self) -> None:
        line = model.boundary_from_areas((.5, 0.), (.5, 1.), [LAWN], "railing")
        self.assertEqual(line.inward, model.boundary_toward((.5, 0.), (.5, 1.), (.2, .8)).inward)
        self.assertEqual(line.flipped().inward, "right" if line.inward == "left" else "left")
        dx, dy = model.inward_vector(line)
        self.assertLess(dx, 0)                                              # the lawn is to the left on the picture
        with self.assertRaises(ValueError):
            model.boundary_from_areas((.5, 0.), (.5, 1.), [], "railing")
        with self.assertRaises(ValueError):
            model.boundary_toward((0., 0.), (1., 1.), (.5, .5))             # the tap is on the line itself

    def test_a_number_sits_inside_its_place_and_off_smaller_ones(self) -> None:
        big = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
        small = ((.3, .3), (.7, .3), (.7, .7), (.3, .7))
        x, y = model.label_point(big, [small])
        self.assertTrue(model.inside((x, y), big))
        self.assertFalse(model.inside((x, y), small))
        self.assertTrue(model.inside(model.label_point(small), small))


class Result(SimpleNamespace):
    pass


def answer(data, code=0, before="INFO loading FastSAM-s.pt\n"):
    return Result(returncode=code, stdout=before + json.dumps(data) + "\n")


def proposal_json(camera="front"):
    return {"camera": camera, "method": "sam", "picture_b64": "/9j/AA==", "image_b64": "/9j/AA==",
            "regions": [{"number": 1, "area": .25, "points": [list(p) for p in LAWN]}],
            "current_map": {"camera": camera, "areas": [], "lines": [], "watched": None}}


class BackendTest(unittest.TestCase):
    def test_operations(self) -> None:
        self.assertEqual(sb.operation("propose", "front"), ["propose", "--camera", "front", "--json", "--embed"])
        self.assertEqual(sb.operation("propose", "front", grid=True)[-1], "--grid")
        args = sb.operation("confirm", "front", scene={"camera": "front", "areas": [], "lines": []}, restart=False)
        self.assertEqual(args[:5], ["confirm", "--camera", "front", "--json", "--map-b64"])
        self.assertEqual(args[-1], "--no-restart")
        self.assertEqual(si.decode_map_b64(args[5]), {"camera": "front", "areas": [], "lines": []})
        for bad in ("Front Door", "x;del", ""):
            with self.assertRaises(ValueError):
                sb.operation("propose", bad)
        with self.assertRaises(ValueError):
            sb.operation("confirm", "front")

    def test_local_command_line_runs_the_boxs_python(self) -> None:
        backend = sb.SceneBackend(python="C:/home_guard/.venv/Scripts/python.exe")
        self.assertEqual(backend.command_line(["propose", "--camera", "front", "--json", "--embed"]),
                         ["C:/home_guard/.venv/Scripts/python.exe", "-m", "home_guard_project.box.scene_interview",
                          "propose", "--camera", "front", "--json", "--embed"])

    def test_ssh_command_line(self) -> None:
        backend = sb.SceneBackend("admin@box", key="C:/keys/homeguard_box")
        line = backend.command_line(sb.operation("propose", "front"))
        self.assertEqual(line[:6], ["ssh.exe", "-i", str(sb.Path("C:/keys/homeguard_box")), "-o", "LogLevel=ERROR",
                                    "admin@box"])
        self.assertEqual(line[6], r"cd /d C:\home_guard && .venv\Scripts\python.exe -m "
                                  r"home_guard_project.box.scene_interview propose --camera front --json --embed")
        scene = model.build_map("front", REGIONS, {1: "mine"}, {1: "it's ours"})
        remote = backend.command_line(sb.operation("confirm", "front", scene=scene))[6]
        self.assertNotIn("'", remote); self.assertNotIn('"', remote)
        self.assertLess(len(remote), sb.REMOTE_LIMIT)

    def test_a_map_too_big_for_the_command_line_is_refused_in_plain_words(self) -> None:
        import random
        rnd = random.Random(1)
        hands = [model.HandArea([[rnd.random(), rnd.random()] for _ in range(32)], "mine", f"area {i}")
                 for i in range(150)]
        scene = model.build_map("front", (), {}, {}, hands)
        with self.assertRaises(sb.SceneError) as caught:
            sb.SceneBackend("admin@box").command_line(sb.operation("confirm", "front", scene=scene))
        self.assertEqual(caught.exception.kind, "too_detailed")

    def test_a_detailed_map_is_simplified_until_it_fits_the_command_line(self) -> None:
        import math
        count, scene = 0, None
        while scene is None or len(model.encode_map(scene)) <= sb.MAP_LIMIT:      # just over the limit
            count += 4
            hands = [model.HandArea([[round(.5 + .3 * math.cos(k / 32 * 6.2832) + i / 1000, 4),
                                      round(.5 + .3 * math.sin(k / 32 * 6.2832), 4)] for k in range(32)], "mine")
                     for i in range(count)]
            scene = model.build_map("front", (), {}, {}, hands)
        argument = sb.map_argument(scene)
        self.assertLessEqual(len(argument), sb.MAP_LIMIT)
        shrunk = si.decode_map_b64(argument)
        self.assertEqual(len(shrunk["areas"]), count)                      # every area kept, with fewer corners
        self.assertTrue(all(3 <= len(a["points"]) < 32 for a in shrunk["areas"]))
        sm.SceneMap.from_dict("front", shrunk)                              # and still a valid map
        remote = sb.SceneBackend("admin@box").command_line(sb.operation("confirm", "front", scene=scene))[6]
        self.assertLess(len(remote), sb.REMOTE_LIMIT)

    def test_simplify_keeps_the_shape_within_the_engines_corner_limit(self) -> None:
        import math
        ring = [[.5 + .4 * math.cos(k / 50 * 6.2832), .5 + .4 * math.sin(k / 50 * 6.2832)] for k in range(50)]
        out = model.simplify(ring)
        self.assertLessEqual(len(out), model.MAX_CORNERS)
        self.assertGreater(len(out), 8)
        self.assertLess(len(model.simplify(ring, tolerance=.02)), len(out))
        self.assertEqual(model.simplify([[0, 0], [1, 0], [1, 1], [0, 1]]), [[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertEqual(len(model.simplify([[0, 0], [1, 0], [.5, 1]], tolerance=.5)), 3)
        self.assertAlmostEqual(model.polygon_area(out), model.polygon_area(ring), delta=.02)

    def test_twelve_detailed_places_still_fit(self) -> None:
        import math
        regions = tuple(model.Region(i + 1, tuple((round(.5 + .4 * math.cos(k / 5 + i), 4),
                                                   round(.5 + .4 * math.sin(k / 5 + i), 4)) for k in range(32)), .1)
                        for i in range(12))
        scene = model.build_map("front", regions, {r.number: "mine" for r in regions})
        remote = sb.SceneBackend("admin@box").command_line(sb.operation("confirm", "front", scene=scene))[6]
        self.assertLess(len(remote), sb.REMOTE_LIMIT)

    def test_propose_over_ssh_reads_the_json_after_log_lines(self) -> None:
        runner = mock.Mock()
        runner.run.return_value = answer(proposal_json())
        backend = sb.SceneBackend("admin@box", runner=runner)
        proposal = backend.propose("front")
        self.assertEqual(runner.run.call_args[0][0][0], "ssh.exe")
        self.assertEqual((proposal.method, proposal.picture, proposal.regions[0].number), ("sam", b"\xff\xd8\xff\x00", 1))
        self.assertEqual(proposal.regions[0].points, LAWN)
        self.assertEqual(proposal.current["watched"], None)

    def test_confirm_locally(self) -> None:
        calls = []

        def run(line, **kwargs):
            calls.append(line)
            return answer({"camera": "front", "map": {"areas": []}, "restart_needed": True, "stale_removed": []})
        saved = sb.SceneBackend(runner=run, python="py").confirm("front", {"camera": "front", "areas": [], "lines": []})
        self.assertEqual(calls[0][:4], ["py", "-m", "home_guard_project.box.scene_interview", "confirm"])
        self.assertTrue(saved.restart_needed)

    def test_box_errors_are_plain_kinds_with_what_the_box_said(self) -> None:
        def failing(data, code):
            return lambda line, **kw: answer(data, code)
        cases = [(failing({"error": "could not get a picture from front right now (is it online?)"}, 1), "box_refused"),
                 (lambda line, **kw: Result(returncode=1, stdout="Traceback: boom"), "no_answer"),
                 (failing({"camera": "other", "map": {}}, 0), "bad_answer")]
        for runner, kind in cases:
            with self.assertRaises(sb.SceneError) as caught:
                sb.SceneBackend(runner=runner, python="py").confirm("front", {"areas": [], "lines": []})
            self.assertEqual(caught.exception.kind, kind)

        def offline(line, **kw):
            raise OSError("ssh.exe not found")
        with self.assertRaises(sb.SceneError) as caught:
            sb.SceneBackend(runner=offline, python="py").propose("front")
        self.assertEqual((caught.exception.kind, caught.exception.detail), ("unreachable", "ssh.exe not found"))
        with self.assertRaises(sb.SceneError) as caught:
            sb.SceneBackend(runner=failing({"error": "unknown camera: 'front'"}, 1), python="py").propose("front")
        self.assertEqual(caught.exception.detail, "unknown camera: 'front'")

    def test_the_box_names_its_cameras(self) -> None:
        self.assertEqual(sb.operation("names"), ["names", "--json", "--lang", "he"])
        self.assertEqual(sb.operation("names", lang="en")[-1], "en")
        backend = sb.SceneBackend(runner=lambda line, **kw: answer(
            {"names": {"ameer_week_0_1_ch3": "\u05de\u05e6\u05dc\u05de\u05d4 3", "front": None}}), python="py")
        self.assertEqual(backend.names(), {"ameer_week_0_1_ch3": "מצלמה 3", "front": ""})
        data = dict(proposal_json(), display_name="הכניסה", display_name_en="Front door")
        proposal = sb.proposal_from("front", data)
        self.assertEqual(proposal.names, {"he": "הכניסה", "en": "Front door"})
        self.assertEqual(sb.proposal_from("front", proposal_json()).names, {"he": "", "en": ""})
        saved = sb.saved_from("front", {"camera": "front", "map": {}, "display_name": "הכניסה"})
        self.assertEqual(saved.names["he"], "הכניסה")
        with self.assertRaises(sb.SceneError):
            sb.SceneBackend(runner=lambda line, **kw: answer({"oops": 1}), python="py").names()

    def test_the_app_never_names_a_camera_itself(self) -> None:
        import pathlib
        app = pathlib.Path(sb.__file__).parent
        for path in app.glob("*.py"):
            self.assertNotRegex(path.read_text(encoding="utf-8"), r"(?:from|import)\s+[.\w]*camera_names\b", path.name)

    def test_the_previous_map_check_and_restore(self) -> None:
        self.assertEqual(sb.operation("restore", "front", check=True), ["restore", "--camera", "front", "--json", "--check"])
        self.assertEqual(sb.operation("restore", "front"), ["restore", "--camera", "front", "--json"])
        self.assertEqual(sb.operation("restore", "front", restart=False)[-1], "--no-restart")
        calls = []
        replies = [{"camera": "front", "has_previous": True, "saved_at": 1791480000.5},
                   {"camera": "front", "restart_needed": True, "saved_at": 1791480000.5,
                    "map": {"camera": "front", "areas": [], "lines": [], "watched": [[0, 0], [1, 0], [1, 1]]}}]

        def run(line, **kwargs):
            calls.append(line)
            return answer(replies[len(calls) - 1])
        backend = sb.SceneBackend(runner=run, python="py")
        self.assertEqual(backend.previous("front"), sb.Previous(True, 1791480000.5))
        restored = backend.restore("front")
        self.assertEqual(calls[1][3:], ["restore", "--camera", "front", "--json"])
        self.assertTrue(restored.restored and restored.restart_needed)
        self.assertEqual(restored.map["watched"], [[0, 0], [1, 0], [1, 1]])
        with self.assertRaises(sb.SceneError):
            sb.previous_from("front", {"camera": "front"})

    def test_leaving_a_camera_cancels_its_command_and_the_next_one_runs(self) -> None:
        backend = sb.SceneBackend("admin@box")
        first = mock.Mock()
        backend.runner = first
        backend.cancel()
        first.cancel.assert_called_once()
        self.assertIsNone(backend.runner)                  # the next camera gets a new runner, not a cancelled one
        injected = mock.Mock()
        kept = sb.SceneBackend("admin@box", runner=injected)
        kept.cancel()
        self.assertIs(kept.runner, injected)

    def test_backend_for_a_camera_page(self) -> None:
        demo = SimpleNamespace(box=SimpleNamespace(demo=True))
        remote = SimpleNamespace(box=SimpleNamespace(demo=False), remote=True, target="admin@box", key="k")
        local = SimpleNamespace(box=SimpleNamespace(demo=False))
        self.assertIsInstance(sb.scene_backend_for(demo), sb.DemoSceneBackend)
        self.assertEqual(sb.scene_backend_for(remote).target, "admin@box")
        self.assertIsNone(sb.scene_backend_for(local).target)


if __name__ == "__main__":
    unittest.main()
