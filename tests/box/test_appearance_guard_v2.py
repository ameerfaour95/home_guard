"""appearance_only, deny by default (2026-10-08). home-guard-32 replayed the first version on eval_set_v2: it lowered
41 REAL alerts whose action was in words outside the action list. Any word that is not looks, filler or harmless
presence now keeps the "suspicious"."""

import unittest

from home_guard_project.box.alert_guards import appearance_only

REAL_ALERTS_KEPT = [  # eval_set_v2 real alerts the first version lowered
    "Two people in dark clothing are loading items into a parked vehicle",
    "Person with obscured face and dark clothing carrying a large bag",
    "Two people with obscured faces are interacting with a parked car",
    "Person with a hooded jacket reaching for items at the counter",
    "person with obscured face moving around property",
    "Person with obscured face loading items into a truck",
    "התקרבות של אדם לבוש קפוצ'ון כהה לרכב",
    "אדם עם פנים מכוסות מעמיס פריטים לטנדר",
]
LOOKS_ONLY_LOWERED = [  # the box's own Oct 6-8 reasons that were only about looks
    "person with covered face",
    "התנהגות חשודה: אדם עם פנים מוסתרות",
    "פנים מוסתרות במסכה",
    "התנהגות חשודה: מסכה על הפנים",
    "פנים מוסתרות על ידי כובעים או מסכות",
    "הופעת אדם עם פנים מוסתרות",
    "הופעת אדם עם כיסוי ראש",
    "A person with a covered face appears to be looking at a phone while standing near a wall.",
]


class AppearanceGuardTest(unittest.TestCase):
    def test_real_alerts_keep_their_label(self):
        for why in REAL_ALERTS_KEPT:
            self.assertFalse(appearance_only(why), why)

    def test_looks_only_is_lowered(self):
        for why in LOOKS_ONLY_LOWERED:
            self.assertTrue(appearance_only(why), why)

    def test_no_appearance_or_empty(self):
        for why in ("", "a man walks by", "אדם עומד ליד השער ומסתכל סביב"):
            self.assertFalse(appearance_only(why), why)


if __name__ == "__main__":
    unittest.main()
