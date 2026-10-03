# tests/box/test_brain_i18n.py
from __future__ import annotations

import string
import unittest

from home_guard_project.box.brain.i18n import LANGS, SUPPORTED_LANGS, TEMPLATES, detect_language, language_override, t


class I18nTest(unittest.TestCase):
    def test_every_template_has_all_languages_with_the_same_placeholders(self) -> None:
        fields = lambda s: sorted(f for _, f, _, _ in string.Formatter().parse(s) if f)  # noqa: E731
        for key, entry in TEMPLATES.items():
            self.assertEqual(set(entry), set(LANGS), key)
            self.assertEqual(fields(entry["he"]), fields(entry["en"]), key)
            self.assertEqual(fields(entry["ar"]), fields(entry["en"]), key)

    def test_t_formats_in_the_asked_language_and_falls_back_to_english(self) -> None:
        self.assertEqual(t("sent_photo", "en", camera="main_entrance"), "✓ Photo sent (main_entrance)")
        self.assertIn("main_entrance", t("sent_photo", "he", camera="main_entrance"))
        self.assertEqual(t("sent_photo", "fr", camera="x"), "✓ Photo sent (x)")

    def test_detect_language_by_letters(self) -> None:
        self.assertEqual(detect_language("תכבה את המצלמה הקדמית"), "he")
        self.assertEqual(detect_language("ما الذي يحدث عند الباب"), "ar")
        self.assertEqual(detect_language("what is happening at the gate"), "en")
        self.assertEqual(detect_language("תראה לי main_entrance עכשיו"), "he")
        self.assertIsNone(detect_language("8"))
        self.assertIsNone(detect_language("👍"))

    def test_detect_language_ignores_digits_and_punctuation_in_script(self) -> None:
        self.assertIsNone(detect_language("٨"))
        self.assertIsNone(detect_language("،"))
        self.assertIsNone(detect_language("٨ ؟"))
        self.assertIsNone(detect_language("״"))
        self.assertEqual(detect_language("שָׁלוֹם"), "he")

    def test_english_and_hebrew_are_spoken_today(self) -> None:
        self.assertEqual(SUPPORTED_LANGS, ("en", "he"))

    def test_language_override(self) -> None:
        self.assertEqual(language_override("please answer in English"), "en")
        self.assertEqual(language_override("תענה בערבית"), "ar")
        self.assertEqual(language_override("أجب بالعبرية"), "he")
        self.assertIsNone(language_override("what happened today"))


if __name__ == "__main__":
    unittest.main()
