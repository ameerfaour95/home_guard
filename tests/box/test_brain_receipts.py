from __future__ import annotations

import json
import os
import tempfile
import unittest

from home_guard_project.box.brain.receipts import DONE, FAILED, REQUESTED, ReceiptBook

NOW = 1_790_000_000.0


class ReceiptBookTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.book = ReceiptBook(self.dir, now=lambda: NOW)

    def _lines(self):
        out = []
        for name in sorted(os.listdir(self.dir)):
            with open(os.path.join(self.dir, name), encoding="utf-8") as f:
                out += [json.loads(line) for line in f if line.strip()]
        return out

    def test_ids_count_per_turn(self) -> None:
        a = self.book.issue("t1", "send_media", DONE, target="E1")
        b = self.book.issue("t1", "pause_alerts", DONE, target="all")
        c = self.book.issue("t2", "send_media", FAILED, target="E2", reason="not_on_box")
        self.assertEqual((a.id, b.id, c.id), ("R1", "R2", "R1"))
        self.assertEqual(c.summary(), "R1 send_media E2 failed (not_on_box)")
        self.assertEqual(len(self._lines()), 3)

    def test_the_same_key_returns_the_same_receipt(self) -> None:
        a = self.book.issue("t1", "send_media", DONE, target="E1", key="t1:send_media:E1")
        b = self.book.issue("t1", "send_media", DONE, target="E1", key="t1:send_media:E1")
        self.assertIs(a, b)
        self.assertIs(self.book.find("t1:send_media:E1"), a)
        self.assertEqual(len(self._lines()), 1)

    def test_update_appends_and_open_receipts_survive_a_restart(self) -> None:
        r = self.book.issue("t1", "set_camera_active", REQUESTED, target="back_door",
                            detail={"chat_id": "-5", "active": False, "lang": "he"})
        self.book.issue("t1", "set_camera_active", REQUESTED, target="front_side", detail={"chat_id": "-5"})
        self.book.update(r, DONE)
        fresh = ReceiptBook(self.dir, now=lambda: NOW + 30)
        still_open = fresh.open_receipts("set_camera_active")
        self.assertEqual([o.target for o in still_open], ["front_side"])
        self.assertEqual(still_open[0].detail, {"chat_id": "-5"})

    def test_a_disk_error_never_raises(self) -> None:
        blocked = os.path.join(self.dir, "file")
        with open(blocked, "w", encoding="utf-8") as f:
            f.write("x")
        book = ReceiptBook(os.path.join(blocked, "receipts"), now=lambda: NOW)
        self.assertEqual(book.issue("t1", "send_media", DONE).status, DONE)


if __name__ == "__main__":
    unittest.main()
