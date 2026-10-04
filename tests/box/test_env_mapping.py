from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from home_guard_project.box import live_view
from home_guard_project.box.brain import vision


class EnvironmentMappingTest(unittest.TestCase):
    """The box passes os.environ (an os._Environ, a Mapping but not a dict) to these builders."""

    def test_live_look_accepts_os_environ(self) -> None:
        with tempfile.TemporaryDirectory() as out, mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            self.assertIsNotNone(live_view.make_look_now("cameras.yaml", os.environ, out))

    def test_live_look_still_refuses_a_non_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            self.assertIsNone(live_view.make_look_now("cameras.yaml", ["OPENAI_API_KEY"], out))

    def test_vision_accepts_os_environ(self) -> None:
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            self.assertIsNotNone(vision.make_vision(os.environ))

    def test_vision_still_refuses_a_non_mapping(self) -> None:
        self.assertIsNone(vision.make_vision(["OPENAI_API_KEY"]))
