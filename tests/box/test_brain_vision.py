# tests/box/test_brain_vision.py
from __future__ import annotations

import json
import datetime as dt
import os
import tempfile
import unittest
from unittest import mock

from home_guard_project.box.brain.vision import BudgetedVision, Vision, VisionRefused, look_prompt, look_schema
from home_guard_project.box.brain.vision import make_vision
from home_guard_project.box.inference import LABEL_RULES, build_prompt


class VisionTest(unittest.TestCase):
    def test_label_rules_are_shared_with_the_guard_prompt(self) -> None:
        self.assertIn('- "escalation":', LABEL_RULES)
        self.assertIn(LABEL_RULES, build_prompt("cam", 0, "23:00:00", 22, 6))
        self.assertIn(LABEL_RULES, look_prompt("cam", guard=True))
        self.assertNotIn(LABEL_RULES, look_prompt("cam", guard=False))

    def test_quality_is_about_the_picture_not_activity(self) -> None:
        prompt = look_prompt("cam", guard=False)
        self.assertIn("NOT about whether anything is happening", prompt)
        self.assertNotIn("looks clear", prompt)

    def test_question_goes_into_the_assistant_prompt(self) -> None:
        self.assertIn("what was he holding?", look_prompt("cam", guard=False, question="what was he holding?"))

    def test_schemas(self) -> None:
        self.assertEqual(set(look_schema(False)["required"]), {"description", "quality", "people"})
        self.assertEqual(set(look_schema(True)["required"]), {"description", "quality", "people", "label", "why"})

    def test_look_parses_and_cleans_the_answer(self) -> None:
        seen = {}

        def complete(prompt, images, schema):
            seen.update(prompt=prompt, images=images, schema=schema)
            return json.dumps({"description": "A man tries the door.", "quality": "fuzzy", "people": 1,
                               "label": "suspicious", "why": "tries the door"})

        out = Vision(complete).look("main_entrance", [b"jpg"], guard=True)
        self.assertEqual(out, {"ok": True, "description": "A man tries the door.", "quality": "clear", "people": 1,
                               "label": "suspicious", "why": "tries the door"})
        self.assertEqual(seen["images"], [b"jpg"])

    def test_refusal_and_errors_are_explicit(self) -> None:
        def refuse(*_):
            raise VisionRefused("no")

        def broken(*_):
            raise ConnectionError("offline")

        self.assertEqual(Vision(refuse).look("c", [b"x"], guard=True), {"ok": False, "refused": True,
                                                                         "error": "refused"})
        self.assertEqual(Vision(broken).look("c", [b"x"], guard=False)["refused"], False)
        self.assertEqual(Vision(lambda *_: "not json").look("c", [b"x"], guard=False)["ok"], False)
        self.assertEqual(Vision(lambda *_: "{}").look("c", [], guard=False)["error"], "no_pictures")

    def test_daily_budget(self) -> None:
        import tempfile, os  # noqa: E401
        path = os.path.join(tempfile.mkdtemp(), "vision_budget.json")
        ok = Vision(lambda *_: json.dumps({"description": "x", "quality": "clear", "people": 0}))
        clock = {"t": 1_790_000_000.0}
        budget = BudgetedVision(ok, limit_per_day=2, path=path, now=lambda: clock["t"])
        self.assertTrue(budget.look("c", [b"x"], guard=False)["ok"])
        self.assertTrue(budget.look("c", [b"x"], guard=False)["ok"])
        self.assertEqual(budget.look("c", [b"x"], guard=False)["error"], "daily_budget")
        self.assertEqual(BudgetedVision(ok, 2, path, now=lambda: clock["t"]).look("c", [b"x"], guard=False)["error"],
                         "daily_budget")                      # survives a restart
        clock["t"] += 86400
        self.assertTrue(budget.look("c", [b"x"], guard=False)["ok"])


class VisionRobustnessTest(unittest.TestCase):
    def test_factory_handles_bad_configuration_and_client_failure(self) -> None:
        for env in ([], None, {"OPENAI_API_KEY": []}, {"OPENAI_API_KEY": {"bad": "key"}}):
            with self.subTest(env=env):
                self.assertIsNone(make_vision(env))
        with mock.patch("openai.OpenAI", side_effect=ValueError("bad client")), \
                mock.patch("httpx.Client"):
            self.assertIsNone(make_vision({"OPENAI_API_KEY": "fake"}))

    def test_factory_adapter_schema_images_and_refusal_without_network(self) -> None:
        client = mock.Mock()
        message = mock.Mock(refusal=None, content='{"description":"x","people":0}')
        client.chat.completions.create.return_value.choices = [mock.Mock(message=message)]
        with mock.patch("openai.OpenAI", return_value=client), mock.patch("httpx.Client"):
            vision = make_vision({"OPENAI_API_KEY": "fake"}, "test-model")
            self.assertTrue(vision.look("c", [b"jpg"], False)["ok"])
            kwargs = client.chat.completions.create.call_args.kwargs
            self.assertEqual(kwargs["model"], "test-model")
            self.assertEqual(kwargs["response_format"]["json_schema"]["schema"], look_schema(False))
            self.assertEqual(kwargs["messages"][0]["content"][1]["image_url"]["url"],
                             "data:image/jpeg;base64,anBn")
            message.refusal = "no"
            self.assertTrue(vision.look("c", [b"jpg"], False)["refused"])

    def test_guard_look_without_a_valid_label_is_no_answer(self) -> None:
        def vision_for(answer):
            return Vision(lambda prompt, images, schema: json.dumps(answer))

        base = {"description": "A man at the door.", "quality": "clear", "people": 1, "why": ""}
        for label in ({}, {"label": ""}, {"label": None}, {"label": "maybe"}):
            with self.subTest(label=label):
                out = vision_for({**base, **label}).look("c", [b"jpg"], True)
                self.assertEqual(out, {"ok": False, "refused": False, "error": "no_answer"})
        out = vision_for({**base, "label": "Suspicious", "why": "tries the handle"}).look("c", [b"jpg"], True)
        self.assertTrue(out["ok"])
        self.assertEqual(out["label"], "suspicious")
        self.assertTrue(vision_for(base).look("c", [b"jpg"], False)["ok"])

    def test_factory_client_has_a_timeout_and_one_retry(self) -> None:
        with mock.patch("openai.OpenAI", return_value=mock.Mock()) as ctor, mock.patch("httpx.Client"):
            self.assertIsNotNone(make_vision({"OPENAI_API_KEY": "fake"}))
        self.assertEqual(ctor.call_args.kwargs.get("timeout"), 30.0)
        self.assertEqual(ctor.call_args.kwargs.get("max_retries"), 1)

    def test_malformed_model_answers_and_people(self) -> None:
        for raw in ([], {"description": "x"}, 123, b"\xff", '[]', '{"description": []}'):
            with self.subTest(raw=raw):
                self.assertFalse(Vision(lambda *_: raw).look("c", [b"x"], False)["ok"])
        for people in ("bad", [], float("nan"), float("inf")):
            with self.subTest(people=people):
                raw = json.dumps({"description": "x", "quality": "clear", "people": people,
                                  "label": "normal"})
                self.assertEqual(Vision(lambda *_: raw).look("c", [b"x"], True)["people"], 0)

    def test_budget_malformed_files_and_invalid_limits(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "budget.json")
            now = 1_790_000_000.0
            day = dt.datetime.fromtimestamp(now).strftime("%Y-%m-%d")
            vision = Vision(lambda *_: '{"description":"x","people":0}')
            for payload in (b"\xff", b"[]", *(
                    json.dumps({"day": day, "count": value}).encode()
                    for value in ("bad", [], float("nan"), float("inf"), -2))):
                with self.subTest(payload=payload):
                    with open(path, "wb") as f:
                        f.write(payload)
                    self.assertTrue(BudgetedVision(vision, 2, path, now=lambda: now).look("c", [b"x"], False)["ok"])
            for limit in ([], "bad", float("nan"), float("inf")):
                with self.subTest(limit=limit):
                    self.assertEqual(BudgetedVision(vision, limit, path).look("c", [b"x"], False)["error"],
                                     "daily_budget")

    def test_budget_write_failure_keeps_an_in_memory_cap_and_catches_backend(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            vision = Vision(lambda *_: '{"description":"x","people":0}')
            budget = BudgetedVision(vision, 1, os.path.join(root, "b.json"))
            with mock.patch("home_guard_project.box.brain.vision.json.dump", side_effect=TypeError("bad value")):
                self.assertTrue(budget.look("c", [b"x"], False)["ok"])
                self.assertEqual(budget.look("c", [b"x"], False)["error"], "daily_budget")
            broken = mock.Mock()
            broken.look.side_effect = ValueError("bad backend")
            budget = BudgetedVision(broken, 2, os.path.join(root, "other.json"))
            self.assertFalse(budget.look("c", [b"x"], False)["ok"])
            budget = BudgetedVision(vision, 2, os.path.join(root, "b.json"), now=lambda: float("inf"))
            self.assertFalse(budget.look("c", [b"x"], False)["ok"])


if __name__ == "__main__":
    unittest.main()
