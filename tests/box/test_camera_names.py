import unittest

from home_guard_project.box.camera_names import channel_of, display_name, family_names, replace_ids


class DisplayNameTest(unittest.TestCase):
    def test_newest_family_name_wins(self):
        aliases = {"ameer_week_0_1_ch3": ["back yard", "פרגולה"]}
        self.assertEqual(display_name("ameer_week_0_1_ch3", "he", aliases), "פרגולה")

    def test_name_survives_site_rename(self):
        # 2026-10-06 23:51: "כניסה ראשית" was saved on ameer_tes2_ch6; the site was renamed to ameer_week_0_1.
        aliases = {"ameer_tes2_ch6": ["כניסה ראשית"]}
        self.assertEqual(display_name("ameer_week_0_1_ch6", "he", aliases), "כניסה ראשית")

    def test_two_old_sites_with_the_channel_is_ambiguous(self):
        aliases = {"a_ch6": ["x"], "b_ch6": ["y"]}
        self.assertEqual(family_names("c_ch6", aliases), [])
        self.assertEqual(display_name("c_ch6", "he", aliases), "מצלמה 6")

    def test_no_name_never_shows_the_id(self):
        self.assertEqual(display_name("ameer_week_0_1_ch8", "he", {}), "מצלמה 8")
        self.assertEqual(display_name("ameer_week_0_1_ch8", "en", {}), "Camera 8")
        self.assertEqual(display_name("back_door", "en", {}), "back door")

    def test_channel(self):
        self.assertEqual(channel_of("ameer_week_0_1_ch12"), "12")
        self.assertIsNone(channel_of("front_side"))

    def test_replace_ids(self):
        aliases = {"ameer_week_0_1_ch6": ["כניסה ראשית"]}
        text = "🟡 חשוד · ameer_week_0_1_ch6 | אדם ליד ameer_week_0_1_ch8"
        out = replace_ids(text, ["ameer_week_0_1_ch6", "ameer_week_0_1_ch8"], "he", aliases)
        self.assertEqual(out, "🟡 חשוד · כניסה ראשית | אדם ליד מצלמה 8")


if __name__ == "__main__":
    unittest.main()
