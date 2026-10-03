# tests/box/test_brain_alert_types.py
from __future__ import annotations

import datetime as dt
import os
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from functools import partial

from test_brain_agent import FakeRegistry, Scripted, call, reply
from test_brain_tools_read import snapshot

from home_guard_project.box.brain.agent import OwnerAgentV2, _effective
from home_guard_project.box.brain.memory import ChatMemory, ChatState
from home_guard_project.box.brain.receipts import DONE, FAILED, ReceiptBook
from home_guard_project.box.brain.render import receipt_line
from home_guard_project.box.brain.tools import (
    TOOLS,
    Services,
    ToolContext,
    get_alert_settings,
    set_alert_types,
    set_sensitivity,
)

NOW = dt.datetime(2026, 10, 3, 12, 0).timestamp()


class FakeAlertSettings:
    """Stands in for home_guard_project.box.alert_settings: a house default plus per-camera overrides."""

    def __init__(self):
        self.house = {"alert_on": ["person"], "sensitivity": {"person": 0.4, "vehicle": 0.5, "animal": 0.5}}
        self.own_types, self.own_sens = {}, {}

    def _row(self, name):
        return {"camera": name, "alert_on": list(self.own_types.get(name, self.house["alert_on"])),
                "own_alert_on": self.own_types.get(name), "sensitivity": dict(self.own_sens.get(name, self.house["sensitivity"])),
                "own_sensitivity": self.own_sens.get(name)}

    def get_alert_settings(self, camera=None, **_):
        if camera:
            return self._row(camera)
        return {"house": {"alert_on": list(self.house["alert_on"]), "sensitivity": dict(self.house["sensitivity"])},
                "cameras": [self._row(n) for n in ("main_entrance", "back_door", "front_side")]}

    def set_alert_types(self, camera, types, **_):
        words = [str(t).strip().lower() for t in types]
        current = list(self.own_types.get(camera, self.house["alert_on"])) if camera else list(self.house["alert_on"])
        if words and all(w[:1] in "+-" for w in words):
            for w in words:
                if w[0] == "+" and w[1:] not in current:
                    current.append(w[1:])
                if w[0] == "-" and w[1:] in current:
                    current.remove(w[1:])
            words = current
        if not camera:
            if not words or any(w == "default" for w in words):
                raise ValueError("the house default needs real types: person, vehicle or animal")
            self.house["alert_on"] = words
            return self.get_alert_settings(None)
        if words == ["default"]:
            self.own_types.pop(camera, None)
        else:
            if not words:
                raise ValueError("at least one type")
            self.own_types[camera] = words
        return self._row(camera)

    def set_sensitivity(self, camera, values, **_):
        if values == "default":
            self.own_sens.pop(camera, None)
            return self._row(camera)
        values = {k: (v / 100 if v > 1 else v) for k, v in dict(values).items()}
        if not camera:
            self.house["sensitivity"].update(values)
            return self.get_alert_settings(None)
        self.own_sens[camera] = {**self.own_sens.get(camera, {}), **values}
        return self._row(camera)


class AlertTypesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.fake = FakeAlertSettings()

    def ctx(self, text) -> ToolContext:
        services = Services(roots=lambda: [], desc_dir=self.dir, feedback_dir=self.dir, work_dir=self.dir, mute=None,
                            deliver=None, alert_settings=self.fake, now=lambda: NOW)
        return ToolContext(turn_id="t1", chat_id="-5", speaker={}, text=text, lang="en", mode="assistant",
                           snapshot=snapshot("assistant"), state=ChatState(), services=services,
                           book=ReceiptBook(os.path.join(self.dir, "r"), now=lambda: NOW))

    def test_registered(self) -> None:
        for name in ("get_alert_settings", "set_alert_types", "set_sensitivity"):
            self.assertIn(name, TOOLS)

    def test_read(self) -> None:
        out = get_alert_settings(self.ctx("what do you alert on?"), {"camera": "entrance"})
        self.assertEqual((out["ok"], out["camera"], out["alert_on"]), (True, "main_entrance", ["person"]))
        self.assertIn("house", get_alert_settings(self.ctx("and the house?"), {}))

    def test_add_vehicles_on_one_camera_with_a_receipt(self) -> None:
        ctx = self.ctx("alert me about cars on the entrance too")
        self.assertFalse(set_alert_types(ctx, {"camera": "entrance", "types": ["+vehicle"],
                                               "owner_words": "please"})["ok"])          # not the owner's words
        out = set_alert_types(ctx, {"camera": "entrance", "types": ["+vehicle"], "owner_words": "cars on the entrance"})
        self.assertEqual(out["status"], DONE)
        d = ctx.receipts[-1].detail
        self.assertEqual((d["camera"], d["old"], d["new"], d["own_before"]),
                         ("main_entrance", ["person"], ["person", "vehicle"], None))
        self.assertEqual(receipt_line(ctx.receipts[-1], "en"),
                         "✓ main_entrance alerts on: people → people, vehicles")

    def test_house_and_refusals(self) -> None:
        ctx = self.ctx("for the whole house, animals too")
        out = set_alert_types(ctx, {"camera": "house", "types": ["+animal"], "owner_words": "animals too"})
        self.assertEqual(ctx.receipts[-1].detail["new"], ["person", "animal"])
        self.assertEqual(out["status"], DONE)
        bad = set_alert_types(ctx, {"camera": "house", "types": ["default"], "owner_words": "animals too"})
        self.assertEqual(bad["status"], FAILED)
        self.assertIn("garage", set_alert_types(ctx, {"camera": "garage", "types": ["person"],
                                                      "owner_words": "animals too"})["error"])

    def test_sensitivity(self) -> None:
        ctx = self.ctx("make people 30% on the back camera")
        out = set_sensitivity(ctx, {"camera": "back", "values": {"person": 30}, "owner_words": "people 30%"})
        self.assertEqual(out["status"], DONE)
        self.assertEqual(ctx.receipts[-1].detail["new"]["person"], 0.3)
        self.assertIn("back_door", receipt_line(ctx.receipts[-1], "en"))

    def test_undo_puts_back_the_exact_previous_choice(self) -> None:
        services = Services(roots=lambda: [], desc_dir=self.dir, feedback_dir=self.dir, work_dir=self.dir, mute=None,
                            deliver=None, alert_settings=self.fake, now=lambda: NOW)
        agent = OwnerAgentV2(Scripted([call("set_alert_types", camera="entrance", types=["+vehicle"],
                                            owner_words="cars on the entrance"),
                                       call("set_sensitivity", camera="house", values={"person": 30},
                                            owner_words="people at 30"),
                                       reply("")]),
                             FakeRegistry("assistant"), ChatMemory(os.path.join(self.dir, "c")),
                             ReceiptBook(os.path.join(self.dir, "u"), now=lambda: NOW), services, now=lambda: NOW)
        first = agent.handle("alert on cars on the entrance and people at 30 for the house", "-5", {})
        self.assertTrue(first.undo_token)
        agent.undo_turn("-5", first.undo_token, {})
        self.assertNotIn("main_entrance", self.fake.own_types)          # back to following the house
        self.assertEqual(self.fake.house["sensitivity"]["person"], 0.4)

    def agent(self, tool, camera, **args):
        ctx = self.ctx("change alerts please")
        agent = OwnerAgentV2(Scripted([call(tool, camera=camera, owner_words=ctx.text, **args), reply("")]),
                             FakeRegistry("assistant"), ChatMemory(os.path.join(self.dir, "chat")), ctx.book,
                             ctx.services, now=lambda: NOW)
        return agent, agent.handle(ctx.text, "-5")

    def test_effective_normalizes_types_and_values(self):
        ctx = self.ctx("change alerts please")
        for tool, first, second in (
            ("set_alert_types", {"types": ["VEHICLE", "person"]}, {"types": " person, vehicle "}),
            ("set_alert_types", {"types": ["+VEHICLE", "-animal"]}, {"types": "-ANIMAL,+vehicle"}),
            ("set_sensitivity", {"values": {"PERSON": 30, "animal": "50%"}},
             {"values": {"animal": 0.5, "person": 0.3}}),
        ):
            with self.subTest(tool=tool):
                self.assertEqual(_effective(ctx, tool, dict(camera="entrance", **first)),
                                 _effective(ctx, tool, dict(camera="main_entrance", **second)))
        self.assertEqual(_effective(ctx, "set_alert_types", {"camera": " HOUSE ", "types": ["person"]}),
                         _effective(ctx, "set_alert_types", {"types": "person"}))

    def test_equivalent_calls_run_once(self):
        for tool, first, second in (
            ("set_alert_types", {"types": ["VEHICLE", "person"]}, {"types": "person,vehicle"}),
            ("set_sensitivity", {"values": {"PERSON": 30}}, {"values": {"person": 0.3}}),
        ):
            with self.subTest(tool=tool):
                agent, _ = self.agent(tool, "entrance", **first)
                agent.model = Scripted([call(tool, camera="entrance", owner_words="change alerts", **first),
                                        call(tool, camera="main_entrance", owner_words="alerts please", **second),
                                        reply("")])
                agent._now = lambda: NOW + 1
                with patch.object(self.fake, tool, wraps=getattr(self.fake, tool)) as setter:
                    result = agent.handle("change alerts please", "-5")
                self.assertEqual(setter.call_count, 1)
                self.assertEqual(len(result.receipts), 1)

    def test_changed_state_or_override_ownership_blocks_undo(self):
        for tool, args, field in (("set_alert_types", {"types": ["person", "vehicle"]}, "own_types"),
                                  ("set_sensitivity", {"values": {"person": 30}}, "own_sens")):
            for change in ("effective", "ownership"):
                with self.subTest(tool=tool, change=change):
                    self.fake = FakeAlertSettings()
                    agent, first = self.agent(tool, "entrance", **args)
                    if change == "ownership":
                        key = "alert_on" if field == "own_types" else "sensitivity"
                        self.fake.house[key] = first.receipts[0].detail["new"]
                        getattr(self.fake, field).clear()
                    else:
                        getattr(self.fake, tool)("main_entrance", ["animal"] if field == "own_types" else {"person": 80})
                    before = self.fake.get_alert_settings("main_entrance")
                    undone = agent.undo_turn("-5", first.undo_token)
                    self.assertIn("Changed since", undone.text)
                    self.assertEqual(self.fake.get_alert_settings("main_entrance"), before)
                    self.assertEqual(agent.book.turn_receipts("-5:" + first.undo_token)[0].status, DONE)

    def test_later_same_target_and_tool_blocks_undo_even_if_value_matches(self):
        for camera in ("house", "entrance"):
            for tool, args in (("set_alert_types", {"types": ["person", "vehicle"]}),
                               ("set_sensitivity", {"values": {"person": 30}})):
                with self.subTest(camera=camera, tool=tool):
                    self.fake = FakeAlertSettings()
                    agent, first = self.agent(tool, camera, **args)
                    agent.book.issue("other:123", tool, DONE, detail=dict(first.receipts[0].detail))
                    self.assertIn("Changed since", agent.undo_turn("-5", first.undo_token).text)

    def test_unrelated_later_changes_allow_undo_and_undo_has_no_button(self):
        self.fake.own_sens["main_entrance"] = {"vehicle": 0.8}
        agent, first = self.agent("set_sensitivity", "entrance", values={"person": 30})
        agent.book.issue("other:123", "set_sensitivity", DONE, detail={"camera": "back_door"})
        agent.book.issue("other:124", "set_alert_types", DONE, detail={"camera": "main_entrance"})
        undone = agent.undo_turn("-5", first.undo_token)
        self.assertEqual(self.fake.own_sens["main_entrance"], {"vehicle": 0.8})
        self.assertEqual(undone.undo_token, "")
        self.assertEqual(undone.receipts[0].detail["undo_of"], "set_sensitivity")
        self.assertIn("nothing left to undo", agent.undo_turn("-5", first.undo_token).text)
        self.assertFalse(agent.undo_turn("-5", first.undo_token + ":undo").receipts)

    def test_undo_setter_failure_is_retryable(self):
        agent, first = self.agent("set_alert_types", "entrance", types=["+vehicle"])
        with patch.object(self.fake, "set_alert_types", side_effect=OSError("disk full")):
            undone = agent.undo_turn("-5", first.undo_token)
        self.assertIn("Could not undo", undone.text)
        self.assertEqual(undone.receipts[0].detail["undo_of"], "set_alert_types")
        self.assertEqual(agent.book.turn_receipts("-5:" + first.undo_token)[0].status, DONE)
        self.assertFalse(agent.undo_turn("-5", first.undo_token).undo_token)
        self.assertNotIn("main_entrance", self.fake.own_types)

    def test_malformed_undo_restore_data_is_refused_before_writing(self):
        for tool, args, bad_values in (
            ("set_alert_types", {"types": ["+vehicle"]}, ("default", ["default"], ["+animal"], [])),
            ("set_sensitivity", {"values": {"person": 30}}, ("default", [], {"person": 50})),
        ):
            for bad in bad_values:
                with self.subTest(tool=tool, bad=bad):
                    self.fake = FakeAlertSettings()
                    agent, first = self.agent(tool, "entrance", **args)
                    receipt = first.receipts[0]
                    receipt.detail["own_before"] = bad
                    agent.book.update(receipt, DONE)
                    before = self.fake.get_alert_settings("main_entrance")
                    with patch.object(self.fake, tool, wraps=getattr(self.fake, tool)) as setter:
                        undone = agent.undo_turn("-5", first.undo_token)
                    self.assertIn("Could not undo", undone.text)
                    self.assertEqual(undone.receipts[0].detail["undo_of"], tool)
                    setter.assert_not_called()
                    self.assertEqual(self.fake.get_alert_settings("main_entrance"), before)

    def test_malformed_tool_arguments_and_service_results_never_raise(self):
        for tool in (get_alert_settings, set_alert_types, set_sensitivity):
            for args in ([], {"camera": object()}, {"values": float("nan")}, {"types": {1}}):
                with self.subTest(tool=tool.__name__, args=args):
                    self.assertFalse(tool(self.ctx("change alerts please"), args)["ok"])
        for malformed in ([], {"house": []}, {"house": {"alert_on": [], "sensitivity": "bad"}},
                          {"house": {"alert_on": ["person"], "sensitivity": {"person": float("inf")}}}):
            with patch.object(self.fake, "get_alert_settings", return_value=malformed):
                self.assertFalse(get_alert_settings(self.ctx("read alerts"), {})["ok"])
                for tool, args in ((set_alert_types, {"types": ["person"]}),
                                   (set_sensitivity, {"values": {"person": 30}})):
                    ctx = self.ctx("change alerts please")
                    self.assertFalse(tool(ctx, dict(owner_words=ctx.text, **args))["ok"])
                    self.assertEqual(ctx.receipts[-1].status, FAILED)
        for value in ("bad", "nan", "inf", [], True, object()):
            self.assertFalse(set_sensitivity(self.ctx("change alerts please"),
                             {"values": {"person": value}, "owner_words": "change alerts"})["ok"])

    def test_malformed_camera_never_becomes_a_house_change(self):
        for camera in ([], {}, False, 0):
            for tool, args in ((get_alert_settings, {}),
                               (set_alert_types, {"types": ["+vehicle"]}),
                               (set_sensitivity, {"values": {"person": 30}})):
                with self.subTest(camera=camera, tool=tool.__name__):
                    ctx = self.ctx("change alerts please")
                    before = self.fake.get_alert_settings()
                    out = tool(ctx, dict(camera=camera, owner_words=ctx.text, **args))
                    self.assertFalse(out["ok"])
                    self.assertEqual(self.fake.get_alert_settings(), before)

    def test_new_receipt_rendering_is_localized_and_contains_bad_values(self):
        for tool, args in ((set_alert_types, {"types": ["+animal"]}),
                           (set_sensitivity, {"values": {"person": 30}})):
            ctx = self.ctx("change alerts please")
            tool(ctx, dict(camera="house", owner_words=ctx.text, **args))
            receipt = ctx.receipts[-1]
            for lang in ("en", "he", "ar"):
                self.assertTrue(receipt_line(receipt, lang).startswith("✓"))
            for malformed in (42, [object()], {"person": "nan"}):
                receipt.detail["new"] = malformed
                self.assertEqual(receipt_line(receipt, "en"), "")

    def test_real_api_on_temporary_files_and_invalid_utf8(self):
        from home_guard_project.box import alert_settings
        paths = {name: os.path.join(self.dir, name + ".yaml") for name in
                 ("cameras_path", "alerts_path", "box_path")}
        with open(paths["cameras_path"], "w", encoding="utf-8") as f:
            f.write("cameras:\n  main_entrance: rtsp://unused\n  back_door: rtsp://unused\n")
        with open(paths["box_path"], "w", encoding="utf-8") as f:
            f.write("alert_on: person\nconf_person: 0.4\n")
        self.fake = SimpleNamespace(**{name: partial(getattr(alert_settings, name), **paths) for name in
                                      ("get_alert_settings", "set_alert_types", "set_sensitivity")})
        agent, first = self.agent("set_sensitivity", "entrance", values={"person": 30})
        self.assertEqual(first.receipts[0].status, DONE)
        self.assertEqual(first.receipts[0].detail["new"]["person"], 0.3)
        agent.undo_turn("-5", first.undo_token)
        self.assertIsNone(self.fake.get_alert_settings("main_entrance")["own_sensitivity"])
        with open(paths["cameras_path"], "wb") as f:
            f.write(b"\xff")
        ctx = self.ctx("change alerts please")
        self.assertFalse(get_alert_settings(ctx, {"camera": "entrance"})["ok"])
        self.assertEqual(set_alert_types(ctx, {"camera": "entrance", "types": ["person"],
                                              "owner_words": ctx.text})["status"], FAILED)

    # ---- Fix 1 ----------------------------------------------------------------------------------------------

    def real_api(self, box_text):
        import yaml
        from home_guard_project.box import alert_settings
        paths = {name: os.path.join(self.dir, name + ".yaml") for name in
                 ("cameras_path", "alerts_path", "box_path")}
        with open(paths["cameras_path"], "w", encoding="utf-8") as f:
            f.write("cameras:\n  main_entrance: rtsp://unused\n  back_door: rtsp://unused\n")
        with open(paths["box_path"], "w", encoding="utf-8") as f:
            f.write(box_text)
        self.fake = SimpleNamespace(**{name: partial(getattr(alert_settings, name), **paths) for name in
                                      ("get_alert_settings", "set_alert_types", "set_sensitivity")})
        return lambda: yaml.safe_load(open(paths["box_path"], encoding="utf-8")) or {}

    def test_house_sensitivity_undo_restores_only_the_changed_type(self):
        box = self.real_api("inference_conf: 0.4\n")
        agent, first = self.agent("set_sensitivity", "house", values={"person": 30})
        self.assertEqual(first.receipts[0].status, DONE)
        self.assertEqual(box()["conf_person"], 0.3)
        undone = agent.undo_turn("-5", first.undo_token)
        self.assertTrue(undone.text.startswith("✓"), undone.text)
        self.assertNotIn("conf_vehicle", box())
        self.assertNotIn("conf_animal", box())
        self.assertEqual(box()["conf_person"], 0.4)

    def undo_calls(self, own_before, values):
        self.fake.own_sens.pop("main_entrance", None)
        if own_before:
            self.fake.own_sens["main_entrance"] = dict(own_before)
        agent, first = self.agent("set_sensitivity", "entrance", values=values)
        after = dict(self.fake.own_sens["main_entrance"])
        return agent, first, after

    def test_camera_sensitivity_undo_is_one_call_when_it_can_be(self):
        for own_before, values, expected in (
            (None, {"person": 30}, [("main_entrance", "default")]),
            ({"person": 0.2}, {"person": 30}, [("main_entrance", {"person": 0.2})]),
        ):
            with self.subTest(own_before=own_before):
                self.fake = FakeAlertSettings()
                agent, first, _ = self.undo_calls(own_before, values)
                with patch.object(self.fake, "set_sensitivity", wraps=self.fake.set_sensitivity) as setter:
                    undone = agent.undo_turn("-5", first.undo_token)
                self.assertTrue(undone.text.startswith("✓"), undone.text)
                self.assertEqual([tuple(c.args) for c in setter.call_args_list], expected)
                self.assertEqual(self.fake.own_sens.get("main_entrance"), own_before)

    def test_camera_sensitivity_undo_that_adds_keys_clears_then_sets(self):
        agent, first, after = self.undo_calls({"vehicle": 0.8}, {"person": 30})
        with patch.object(self.fake, "set_sensitivity", wraps=self.fake.set_sensitivity) as setter:
            agent.undo_turn("-5", first.undo_token)
        self.assertEqual([tuple(c.args) for c in setter.call_args_list],
                         [("main_entrance", "default"), ("main_entrance", {"vehicle": 0.8})])
        self.assertEqual(self.fake.own_sens["main_entrance"], {"vehicle": 0.8})

    def test_failed_second_undo_call_puts_the_camera_back_and_retry_works(self):
        agent, first, after = self.undo_calls({"vehicle": 0.8}, {"person": 30})
        real = self.fake.set_sensitivity
        calls = []

        def flaky(camera, values, **kw):
            calls.append(values)
            if len(calls) == 2:
                raise OSError("disk full")
            return real(camera, values, **kw)

        with patch.object(self.fake, "set_sensitivity", side_effect=flaky):
            undone = agent.undo_turn("-5", first.undo_token)
        self.assertIn("Could not undo", undone.text)
        self.assertEqual(calls, ["default", {"vehicle": 0.8}, after])
        self.assertEqual(self.fake.own_sens["main_entrance"], after)       # not stuck on the house values
        again = agent.undo_turn("-5", first.undo_token)
        self.assertNotIn("Changed since", again.text)
        self.assertEqual(self.fake.own_sens["main_entrance"], {"vehicle": 0.8})

    def test_setters_need_a_camera_or_house(self):
        for tool, args in (("set_alert_types", {"types": ["+vehicle"]}), ("set_sensitivity", {"values": {"person": 30}})):
            fn = {"set_alert_types": set_alert_types, "set_sensitivity": set_sensitivity}[tool]
            for camera in (None, "", "   "):
                with self.subTest(tool=tool, camera=camera):
                    ctx = self.ctx("change alerts please")
                    before = self.fake.get_alert_settings()
                    given = dict(args, owner_words=ctx.text)
                    if camera is not None:
                        given["camera"] = camera
                    out = fn(ctx, given)
                    self.assertEqual((out["ok"], out["status"]), (False, FAILED))
                    self.assertIn("say which camera, or 'house'", receipt_line(ctx.receipts[-1], "en"))
                    self.assertEqual(self.fake.get_alert_settings(), before)
            with self.subTest(tool=tool, camera="default"):
                ctx = self.ctx("change alerts please")
                before = self.fake.get_alert_settings()
                out = fn(ctx, dict(args, camera="default", owner_words=ctx.text))
                self.assertFalse(out["ok"])
                self.assertEqual(self.fake.get_alert_settings(), before)
            for camera in ("house", "הבית", "المنزل"):
                with self.subTest(tool=tool, camera=camera):
                    ctx = self.ctx("change alerts please")
                    self.assertEqual(fn(ctx, dict(args, camera=camera, owner_words=ctx.text))["status"], DONE)
                    self.assertEqual(ctx.receipts[-1].detail["camera"], "")

    def test_failed_read_back_after_a_saved_change_is_done_with_the_intended_values(self):
        for tool, camera, args, check in (
            ("set_sensitivity", "house", {"values": {"person": 30}},
             lambda d: self.assertEqual(d["new"], {"person": 0.3, "vehicle": 0.5, "animal": 0.5})),
            ("set_sensitivity", "entrance", {"values": {"person": 30}},
             lambda d: (self.assertEqual(d["new"]["person"], 0.3), self.assertEqual(d["own_after"], {"person": 0.3}))),
            ("set_alert_types", "house", {"types": ["+animal"]},
             lambda d: self.assertEqual(d["new"], ["person", "animal"])),
            ("set_alert_types", "entrance", {"types": ["+vehicle"]},
             lambda d: (self.assertEqual(d["new"], ["person", "vehicle"]),
                        self.assertEqual(d["own_after"], ["person", "vehicle"]))),
        ):
            with self.subTest(tool=tool, camera=camera):
                self.fake = FakeAlertSettings()
                real = getattr(self.fake, tool)
                ctx = self.ctx("change alerts please")
                with patch.object(self.fake, tool, side_effect=lambda c, v, **k: (real(c, v), {"bad": 1})[1]):
                    out = {"set_alert_types": set_alert_types, "set_sensitivity": set_sensitivity}[tool](
                        ctx, dict(args, camera=camera, owner_words=ctx.text))
                self.assertEqual(out["status"], DONE)
                check(ctx.receipts[-1].detail)

    def test_failed_reason_has_no_file_paths(self):
        for message in ("cannot write C:\\Users\\x\\box.yaml now", "cannot write /data/box/box.yaml now",
                        "cannot write 'C:/home guard/box.yaml' now"):
            with self.subTest(message=message):
                ctx = self.ctx("change alerts please")
                with patch.object(self.fake, "set_sensitivity", side_effect=ValueError(message)):
                    out = set_sensitivity(ctx, {"camera": "house", "values": {"person": 30}, "owner_words": ctx.text})
                self.assertEqual(out["status"], FAILED)
                for shown in (out.get("reason", ""), receipt_line(ctx.receipts[-1], "en")):
                    self.assertNotIn("box.yaml", shown)
                    self.assertNotIn("Users", shown)
                    self.assertIn("cannot write", shown)
        ctx = self.ctx("change alerts please")
        with patch.object(self.fake, "set_alert_types", side_effect=OSError(28, "No space left", "C:\\x\\box.yaml")):
            set_alert_types(ctx, {"camera": "house", "types": ["+animal"], "owner_words": ctx.text})
        self.assertNotIn("box.yaml", receipt_line(ctx.receipts[-1], "en"))

    def line(self, tool, lang, **detail):
        from home_guard_project.box.brain.receipts import Receipt
        return receipt_line(Receipt(id="R1", turn="t", tool=tool, status=DONE, target="x", detail=detail, reason=""), lang)

    def test_sensitivity_line_shows_only_what_changed_in_order(self):
        old = {"animal": 0.5, "vehicle": 0.5, "person": 0.4}
        new = {"animal": 0.7, "vehicle": 0.5, "person": 0.3}
        self.assertEqual(self.line("set_sensitivity", "en", camera="", old=old, new=new),
                         "✓ The house: how sure before alerting - people 40% → 30%, animals 50% → 70%")
        self.assertEqual(self.line("set_sensitivity", "en", camera="back_door", old=old, new=new),
                         "✓ back_door: how sure before alerting - people 40% → 30%, animals 50% → 70%")
        he = self.line("set_sensitivity", "he", camera="", old=old, new=new)
        self.assertIn("אנשים 40% ← 30%", he)
        self.assertNotIn("רכבים", he)
        self.assertLess(he.index("אנשים"), he.index("בעלי חיים"))

    def test_alert_type_lists_are_in_people_vehicles_animals_order(self):
        self.assertEqual(self.line("set_alert_types", "en", camera="back_door",
                                   old=["animal", "person"], new=["animal", "vehicle", "person"]),
                         "✓ back_door alerts on: people, animals → people, vehicles, animals")

    def test_hebrew_house_uses_the_masculine_verb(self):
        he = self.line("set_alert_types", "he", camera="", old=["person"], new=["person", "animal"])
        self.assertIn("הבית מתריע על", he)
        self.assertNotIn("מתריעה", he)
        self.assertEqual(self.line("set_alert_types", "en", camera="", old=["person"], new=["person", "animal"]),
                         "✓ The house alerts on: people → people, animals")

    def test_changed_since_names_the_camera_or_house(self):
        from home_guard_project.box.brain.i18n import t
        from home_guard_project.box.brain.render import undo_what
        for tool, camera, en in (("set_alert_types", "back_door", "the alert types of back_door"),
                                 ("set_sensitivity", "", "the sensitivity of the house")):
            what = undo_what(tool, {"camera": camera}, camera or "house", "en")
            self.assertEqual(what, en)
            self.assertEqual(t("undo_changed_since", "en", what=what),
                             f"Changed since - nothing to undo for {en}.")
            for lang in ("he", "ar"):
                self.assertNotEqual(undo_what(tool, {"camera": camera}, "x", lang), tool)
        self.assertIn("הבית", undo_what("set_sensitivity", {"camera": ""}, "house", "he"))


if __name__ == "__main__":
    unittest.main()
