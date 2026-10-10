"""2026-10-06 and again 2026-10-10: setup wrote a fresh api_key.env with only OPENAI_API_KEY and
TELEGRAM_BOT_TOKEN, so OPENROUTER_API_KEY was gone and the box ran without its vision model, translator and
describer. Setup now hands the box its keys and the box merges them: a key setup sends is set, every other key
the box already had stays."""

from __future__ import annotations

import os
import tempfile
import unittest

from home_guard_project.box import secrets_file as sf


class MergeTextTest(unittest.TestCase):
    def test_sent_keys_are_set_and_every_other_key_and_comment_stays(self) -> None:
        old = "# box secrets\nOPENAI_API_KEY=old\nOPENROUTER_API_KEY=or-key\n"
        new = "OPENAI_API_KEY=new\nTELEGRAM_BOT_TOKEN=tg\n"
        self.assertEqual(sf.merge_text(old, new),
                         "# box secrets\nOPENAI_API_KEY=new\nOPENROUTER_API_KEY=or-key\nTELEGRAM_BOT_TOKEN=tg\n")

    def test_an_empty_value_never_wipes_a_key(self) -> None:
        self.assertEqual(sf.merge_text("OPENROUTER_API_KEY=or-key\n", "OPENROUTER_API_KEY=\n"),
                         "OPENROUTER_API_KEY=or-key\n")

    def test_values_with_equals_signs_and_windows_line_ends_survive(self) -> None:
        self.assertEqual(sf.merge_text("A=x=y\r\n", "B=p=q==\r\n"), "A=x=y\nB=p=q==\n")


class MergeFileTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.target = os.path.join(tmp.name, "api_key.env")
        self.incoming = os.path.join(tmp.name, "api_key.env.incoming")

    def _write(self, path: str, text: str) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)

    def test_merge_keeps_the_box_key_and_removes_the_incoming_file(self) -> None:
        self._write(self.target, "OPENROUTER_API_KEY=or-key\n")
        self._write(self.incoming, "OPENAI_API_KEY=oai\nTELEGRAM_BOT_TOKEN=tg\n")
        result = sf.merge_file(self.incoming, self.target)
        with open(self.target, encoding="utf-8") as f:
            self.assertEqual(f.read(), "OPENROUTER_API_KEY=or-key\nOPENAI_API_KEY=oai\nTELEGRAM_BOT_TOKEN=tg\n")
        self.assertFalse(os.path.exists(self.incoming))
        self.assertEqual(result, {"set": ["OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN"], "kept": ["OPENROUTER_API_KEY"]})

    def test_a_box_without_a_key_file_gets_one(self) -> None:
        self._write(self.incoming, "OPENAI_API_KEY=oai\n")
        sf.merge_file(self.incoming, self.target)
        with open(self.target, encoding="utf-8") as f:
            self.assertEqual(f.read(), "OPENAI_API_KEY=oai\n")

    def test_the_result_names_keys_but_never_their_values(self) -> None:
        self._write(self.incoming, "OPENAI_API_KEY=secret-value\n")
        self.assertNotIn("secret-value", repr(sf.merge_file(self.incoming, self.target)))


if __name__ == "__main__":
    unittest.main()
