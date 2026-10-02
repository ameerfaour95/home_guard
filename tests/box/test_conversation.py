from __future__ import annotations

import os
import tempfile
import unittest

from home_guard_project.box.conversation import ConversationStore


class ConversationStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = os.path.join(tempfile.mkdtemp(), "conversations")  # does not exist yet

    def test_history_is_empty_before_anything_is_said(self) -> None:
        self.assertEqual(ConversationStore(self.dir).history("-1001"), [])

    def test_a_turn_is_persisted_and_read_back_as_chat_messages(self) -> None:
        store = ConversationStore(self.dir)
        store.append("-1001", "no there was nothing", "Thanks, marked as a false alarm.")
        self.assertEqual(store.history("-1001"), [
            {"role": "user", "content": "no there was nothing"},
            {"role": "assistant", "content": "Thanks, marked as a false alarm."},
        ])

    def test_history_survives_a_restart(self) -> None:
        ConversationStore(self.dir).append("-1001", "hi", "hello")
        fresh = ConversationStore(self.dir)  # a new process after a settings restart
        self.assertEqual(fresh.history("-1001"), [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])

    def test_separate_chats_do_not_mix(self) -> None:
        store = ConversationStore(self.dir)
        store.append("-1001", "family group msg", "ok")
        store.append("555", "dm msg", "sure")
        self.assertEqual([m["content"] for m in store.history("-1001")], ["family group msg", "ok"])
        self.assertEqual([m["content"] for m in store.history("555")], ["dm msg", "sure"])

    def test_the_model_only_sees_the_recent_window(self) -> None:
        store = ConversationStore(self.dir, window=4)
        for i in range(5):
            store.append("-1001", f"q{i}", f"a{i}")
        shown = store.history("-1001")
        self.assertEqual(len(shown), 4)                       # last 2 turns only
        self.assertEqual(shown[0]["content"], "q3")

    def test_the_file_is_capped_so_it_cannot_grow_without_bound(self) -> None:
        store = ConversationStore(self.dir, window=100, max_persist=6)
        for i in range(10):
            store.append("-1001", f"q{i}", f"a{i}")
        kept = store.history("-1001")
        self.assertEqual(len(kept), 6)                        # 3 most recent turns
        self.assertEqual(kept[0]["content"], "q7")

    def test_a_damaged_file_reads_as_empty(self) -> None:
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, "-1001.json"), "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(ConversationStore(self.dir).history("-1001"), [])


if __name__ == "__main__":
    unittest.main()
