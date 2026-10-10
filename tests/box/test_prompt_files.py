"""The prompts folder: every prompt file is used, every name the code asks for exists, the loader is strict."""
import os
import re
import tempfile
import unittest
from unittest import mock

from home_guard_project import prompts

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_NAME = re.compile(r"[\"']([A-Za-z0-9_]+\.(?:system_prompt|prompt))[\"']")


def _code_files():
    for top in ("home_guard_project", "run_with_gpt.py"):
        start = os.path.join(REPO, top)
        if os.path.isfile(start):
            yield start
            continue
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in filenames:
                if name.endswith(".py") and os.path.join(dirpath, name) != os.path.join(prompts.PROMPTS_DIR, "__init__.py"):
                    yield os.path.join(dirpath, name)


def _referenced():
    names = set()
    for path in _code_files():
        with open(path, encoding="utf-8", errors="replace") as f:
            names.update(_NAME.findall(f.read()))
    return names


class PromptFolderTest(unittest.TestCase):
    def test_every_prompt_file_is_used_by_the_code(self) -> None:
        unused = set(prompts.names()) - _referenced()
        self.assertFalse(unused, f"prompt files nothing loads: {sorted(unused)}")

    def test_every_prompt_the_code_names_exists(self) -> None:
        missing = _referenced() - set(prompts.names())
        self.assertFalse(missing, f"code names prompt files that do not exist: {sorted(missing)}")

    def test_only_prompt_files_and_the_loader_live_in_the_folder(self) -> None:
        others = [n for n in os.listdir(prompts.PROMPTS_DIR)
                  if not n.endswith(prompts.SUFFIXES) and n not in ("__init__.py", "__pycache__")]
        self.assertFalse(others, others)

    def test_files_end_with_one_newline(self) -> None:
        for name in prompts.names():
            with open(prompts.path(name), "rb") as f:
                raw = f.read()
            self.assertTrue(raw.endswith(b"\n"), name)


class LoaderTest(unittest.TestCase):
    def _folder(self, files):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        for name, raw in files.items():
            with open(os.path.join(directory.name, name), "wb") as f:
                f.write(raw)
        patcher = mock.patch.object(prompts, "PROMPTS_DIR", directory.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        prompts.load.cache_clear()
        self.addCleanup(prompts.load.cache_clear)

    def test_the_final_newline_is_not_part_of_the_prompt(self) -> None:
        self._folder({"a.prompt": b"Line one.\nLine two.\n", "b.prompt": b"Ends with a blank line.\n\n"})
        self.assertEqual(prompts.load("a.prompt"), "Line one.\nLine two.")
        self.assertEqual(prompts.load("b.prompt"), "Ends with a blank line.\n")

    def test_a_crlf_checkout_sends_the_same_text(self) -> None:
        self._folder({"lf.prompt": b"one\ntwo {\"x\": 1}\n", "crlf.prompt": b"one\r\ntwo {\"x\": 1}\r\n"})
        self.assertEqual(prompts.load("lf.prompt"), prompts.load("crlf.prompt"))

    def test_render_fills_placeholders_and_keeps_single_braces(self) -> None:
        self._folder({"t.system_prompt": b'Camera "{{camera}}" at {{time}}. Reply {"ok": true}.\n'})
        self.assertEqual(prompts.render("t.system_prompt", camera="ch6", time="21:00"),
                         'Camera "ch6" at 21:00. Reply {"ok": true}.')

    def test_render_refuses_a_missing_or_an_unused_value(self) -> None:
        self._folder({"t.prompt": b"Hello {{name}}.\n"})
        with self.assertRaises(KeyError):
            prompts.render("t.prompt")
        with self.assertRaises(KeyError):
            prompts.render("t.prompt", name="x", other="y")

    def test_a_placeholder_may_be_called_name_or_text(self) -> None:
        self._folder({"t.prompt": b"{{name}} / {{text}}\n"})
        self.assertEqual(prompts.render("t.prompt", name="a", text="b"), "a / b")

    def test_only_prompt_file_names(self) -> None:
        for bad in ("a.txt", "../a.prompt", "sub/a.prompt"):
            with self.assertRaises(ValueError):
                prompts.path(bad)


if __name__ == "__main__":
    unittest.main()
