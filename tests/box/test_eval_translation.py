"""Tests for box/eval_translation.py, offline (--fake): inputs read, CSV written side by side."""
from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest

from home_guard_project.box import eval_translation as ev


class EvalTranslationTest(unittest.TestCase):
    def test_fake_run_writes_a_side_by_side_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = {"camera_name": "Cam 2", "alert": {"summary": "A man passes Cam 2 at 03:10.", "why": "lingers",
                                                      "summary_owner": "גבר עובר."}}
            with open(os.path.join(tmp, "a.meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
            results = [{"ai_summary": "A woman at the door.", "camera": "door",
                        "raw": json.dumps({"summary": "A woman at the door.", "why": ""})},
                       {"ai_summary": "A woman at the door.", "camera": "door", "raw": "not json"},   # a repeat
                       {"ai_summary": "", "camera": "door", "raw": ""}]                              # nothing to say
            with open(os.path.join(tmp, "run.jsonl"), "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(r) for r in results))
            out = os.path.join(tmp, "out.csv")

            code = ev.main([tmp, "--fake", "--out", out, "--model", "openrouter:google/gemini-3.1-flash-lite",
                            "--model", "openrouter:openai/gpt-6-luna:batch"])

            self.assertEqual(code, 0)
            with open(out, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual([r["summary_en"] for r in rows], ["A man passes Cam 2 at 03:10.", "A woman at the door."])
        first = rows[0]
        self.assertEqual(first["eye_summary_owner"], "גבר עובר.")
        for tag in ("openrouter:google/gemini-3.1-flash-lite", "openrouter:openai/gpt-6-luna:batch"):
            self.assertEqual(first[f"{tag} source"], "translator")
            self.assertEqual(first[f"{tag} summary"], "[תרגום] A man passes Cam 2 at 03:10.")
        self.assertIn("owner_pick", first)

    def test_a_bad_model_spec_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            ev.parse_model("google/gemini-3.1-flash-lite")


if __name__ == "__main__":
    unittest.main()
