# tests/box/test_brain_profiles.py
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from home_guard_project.box.brain import profiles
from home_guard_project.box.brain.profiles import load_schemas, needs_big, system_prompt, tool_names, tools_for
from home_guard_project.box.brain.tools import TOOLS


class ProfilesTest(unittest.TestCase):
    def test_every_tool_has_a_schema_and_every_schema_a_tool(self) -> None:
        schemas = load_schemas()
        self.assertEqual(len(schemas), len(TOOLS) + 2)
        self.assertEqual(set(schemas) - {"reply", "hand_off"}, set(TOOLS))
        for name, schema in schemas.items():
            self.assertEqual(schema["function"]["name"], name)
            self.assertEqual(schema["function"]["parameters"]["type"], "object")

    def test_modes_have_different_tools(self) -> None:
        guard, assistant = tool_names("guard"), tool_names("assistant")
        self.assertIn("assess_event", guard)
        self.assertNotIn("describe_event", guard)
        self.assertIn("describe_event", assistant)
        self.assertNotIn("assess_event", assistant)
        self.assertIn("reply", guard)
        self.assertNotIn("hand_off", guard)
        self.assertIn("hand_off", tool_names("guard", tier="fast"))
        self.assertNotIn("pause_alerts", tool_names("guard", tier="fast"))
        self.assertNotIn("record_verdict", tool_names("assistant", tier="fast"))
        self.assertIn("send_media", tool_names("assistant", tier="fast"))
        self.assertEqual([t["function"]["name"] for t in tools_for("assistant")], assistant)

    def test_routing_in_code(self) -> None:
        self.assertTrue(needs_big("תשתיק את המצלמה עד שש"))
        self.assertTrue(needs_big("it's me, stop until six"))
        self.assertTrue(needs_big("switch the AI to Hebrew"))
        self.assertTrue(needs_big("anything", threaded=True))
        self.assertFalse(needs_big("send me a picture of the gate"))
        self.assertFalse(needs_big("מה קורה בכניסה עכשיו"))

    def test_set_only_routes_as_a_whole_word(self) -> None:
        for text in ("settle", "reset", "sunset", "settings", "a settled question"):
            with self.subTest(text=text):
                self.assertFalse(needs_big(text))
        for text in ("set", "SET the alert hours", "please set, thanks"):
            with self.subTest(text=text):
                self.assertTrue(needs_big(text))

    def test_prompts_differ_by_mode_and_tier(self) -> None:
        guard, assistant = system_prompt("guard", 14), system_prompt("assistant", 14)
        self.assertIn("MODE: GUARD", guard)
        self.assertIn("MODE: ASSISTANT", assistant)
        self.assertIn("kept 14 days", guard)
        self.assertIn("Never describe an action", guard)
        self.assertNotIn("hand_off", guard)
        self.assertIn("hand_off", system_prompt("guard", 14, tier="fast"))
        self.assertNotIn("{", guard.replace("{}", ""))     # every placeholder was filled

    def test_fast_tier_excludes_every_state_tool_in_both_modes(self) -> None:
        for mode in ("guard", "assistant"):
            with self.subTest(mode=mode):
                fast = tool_names(mode, "fast")
                self.assertFalse(set(fast) & set(profiles.STATE_TOOLS))
                self.assertTrue(set(profiles.STATE_TOOLS) <= set(tool_names(mode)))
                self.assertEqual(fast[-1], "hand_off")
                self.assertEqual([t["function"]["name"] for t in tools_for(mode, "fast")], fast)

    def test_schema_file_has_one_unique_definition_per_tool_plus_two(self) -> None:
        items = json.loads(Path(profiles.TOOLS_PATH).read_text(encoding="utf-8"))
        self.assertEqual(len(items), len(TOOLS) + 2)
        self.assertEqual({item["function"]["name"] for item in items}, set(TOOLS) | {"reply", "hand_off"})

    def test_malformed_message_routes_to_big_without_raising(self) -> None:
        for text in (42, float("nan"), [], {"text": "stop"}, b"pause", object()):
            with self.subTest(text=text), self.assertLogs("box.brain.profiles", level="WARNING") as logs:
                self.assertTrue(needs_big(text))
            self.assertEqual(len(logs.output), 1)
        self.assertFalse(needs_big(None))
        self.assertTrue(needs_big([], threaded=True))

    def test_schema_loader_raises_on_bad_files_and_wrong_root_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tools.json"
            for payload in (b"{", b"{}", b"null", b'"tools"', b"42", b"[\xff]"):
                path.write_bytes(payload)
                with self.subTest(payload=payload), self.assertRaises(RuntimeError) as ctx:
                    load_schemas(path)
                self.assertIn(str(path), str(ctx.exception))
            path.unlink()
            with self.assertRaises(RuntimeError) as ctx:
                load_schemas(path)
            self.assertIn(str(path), str(ctx.exception))

    def test_broken_schema_file_fails_tools_for_but_not_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tools.json"
            path.write_text("{", encoding="utf-8")
            with patch.object(profiles, "TOOLS_PATH", str(path)), patch.object(profiles, "_SCHEMAS", None):
                with self.assertRaises(RuntimeError):
                    tools_for("guard")
                with self.assertRaises(RuntimeError):
                    profiles._schemas()
        self.assertTrue(tools_for("guard"))

    def test_hebrew_state_changes_route_to_big(self) -> None:
        for text in ("תפעיל את המצלמה", "כבה את הכניסה", "הדלק את המצלמה האחורית", "השתק לשעה",
                     "הפסק את ההתראות", "תגדיר שעות מ-23"):
            with self.subTest(text=text):
                self.assertTrue(needs_big(text))
        for text in ("מה קורה בכניסה עכשיו", "יש מישהו בחצר?"):
            with self.subTest(text=text):
                self.assertFalse(needs_big(text))

    def test_english_state_changes_route_to_big(self) -> None:
        for text in ("turn the camera off", "turn the cameras off until 17:00", "unmute", "shut off the porch",
                     "shut down alerts", "disarm", "arm the cameras", "snooze", "that was me", "it was us",
                     "that's me", "was me", "it’s me", "doesn’t work"):
            with self.subTest(text=text):
                self.assertTrue(needs_big(text))
        for text in ("send me a picture of the gate", "what happened today"):
            with self.subTest(text=text):
                self.assertFalse(needs_big(text))

    def test_schema_loader_skips_bad_entries_and_logs_once(self) -> None:
        good = load_schemas()["reply"]
        bad = [None, [], "bad", 5, {}, {"function": []}, {"function": {"name": []}},
               {"type": "function", "function": {"name": "bad", "parameters": []}},
               {"type": "function", "function": {"name": "bad", "parameters": {"type": "array"}}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tools.json"
            path.write_text(json.dumps([good, *bad]), encoding="utf-8")
            with self.assertLogs("box.brain.profiles", level="WARNING") as logs:
                self.assertEqual(load_schemas(path), {"reply": good})
            self.assertEqual(len(logs.output), 1)

    def test_malformed_mode_and_tier_keep_assistant_big_fallback(self) -> None:
        self.assertEqual(tool_names([], tier={}), tool_names("assistant"))
        self.assertEqual(tools_for({}, tier=[]), tools_for("assistant"))
        self.assertEqual(system_prompt([], 14, tier={}), system_prompt("assistant", 14))

    def test_tools_for_skips_missing_schemas_and_logs_once(self) -> None:
        reply = load_schemas()["reply"]
        with patch.object(profiles, "_SCHEMAS", {"reply": reply}):
            with self.assertLogs("box.brain.profiles", level="WARNING") as logs:
                self.assertEqual(tools_for("guard"), [reply])
            self.assertEqual(len(logs.output), 1)

    def test_bad_retention_uses_existing_fourteen_day_default(self) -> None:
        for days in (None, "bad", [], {}, True, float("nan"), float("inf"), -float("inf")):
            with self.subTest(days=days), self.assertLogs("box.brain.profiles", level="WARNING") as logs:
                self.assertIn("kept 14 days", system_prompt("guard", days))
            self.assertEqual(len(logs.output), 1)
        self.assertIn("kept 7 days", system_prompt("guard", "7"))
        self.assertIn("kept 3 days", system_prompt("guard", 3.5))

    def test_prompt_files_with_invalid_utf8_or_braces_do_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(profiles, "PROMPTS_DIR", directory):
            root = Path(directory)
            (root / "common.txt").write_bytes(b"kept {retention_days} days \xff {unknown}")
            (root / "guard.txt").write_text("MODE: GUARD", encoding="utf-8")
            with self.assertLogs("box.brain.profiles", level="WARNING") as logs:
                prompt = system_prompt("guard", 14)
            self.assertEqual(len(logs.output), 1)
            self.assertIn("kept 14 days", prompt)
            self.assertIn("\ufffd", prompt)
            self.assertIn("MODE: GUARD", prompt)

    def test_unreadable_prompt_stops_the_turn(self) -> None:
        with patch.object(profiles, "_read", side_effect=OSError("unreadable")):
            with self.assertRaises(RuntimeError):
                system_prompt("guard", 14)
        with tempfile.TemporaryDirectory() as directory, patch.object(profiles, "PROMPTS_DIR", directory):
            with self.assertRaises(RuntimeError) as ctx:
                system_prompt("guard", 14)
            self.assertIn(directory, str(ctx.exception))



class HebrewCancelTest(unittest.TestCase):
    def test_cancel_with_prefixes_and_suffixes(self):
        for text in ("לבטל את ההשתקה", "בטלו את זה", "תבטלי בבקשה", "בטל", "ובטל!", "(בטל)", "תבטל", "שבטל",
                     "הבטל", "מבטל", "כבטל", "ולבטל", "בטלה", "בטלי"):
            self.assertTrue(needs_big(text), text)

    def test_phone_words_do_not_match(self):
        for text in ("תשלח לי לטלפון", "בטלפון שלי", "ובטלפון", "לבטלפון", "בטלוויזיה"):
            self.assertFalse(needs_big(text), text)

if __name__ == "__main__":
    unittest.main()
