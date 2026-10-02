from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from home_guard_project.box.embeddings import Embedder, cosine, make_embedder


def _fake_fetch(texts):
    """A deterministic vector per text, so distinct texts embed differently and no network is used."""
    return [[float(len(t)), 1.0, 0.0] for t in texts]


class CosineTest(unittest.TestCase):
    def test_identical_vectors_score_one(self) -> None:
        self.assertAlmostEqual(cosine([1, 2, 3], [1, 2, 3]), 1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)

    def test_empty_mismatched_or_zero_vectors_score_zero(self) -> None:
        self.assertEqual(cosine([], [1.0]), 0.0)
        self.assertEqual(cosine([1.0, 2.0], [1.0]), 0.0)
        self.assertEqual(cosine([0.0, 0.0], [0.0, 0.0]), 0.0)
        self.assertEqual(cosine(None, [1.0]), 0.0)


class EmbedderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.cache = os.path.join(self.dir, "emb.json")

    def test_a_text_is_fetched_once_then_served_from_cache(self) -> None:
        emb = Embedder("key", self.cache)
        with mock.patch.object(emb, "_fetch", side_effect=_fake_fetch) as fetch:
            first = emb.embed(["a person at the door"])
            second = emb.embed(["a person at the door"])
        self.assertEqual(first, second)
        fetch.assert_called_once()

    def test_only_the_missing_texts_are_fetched(self) -> None:
        emb = Embedder("key", self.cache)
        with mock.patch.object(emb, "_fetch", side_effect=_fake_fetch) as fetch:
            emb.embed(["one"])
            emb.embed(["one", "two"])
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(fetch.call_args_list[1].args[0], ["two"])

    def test_embed_returns_none_when_the_model_cannot_be_reached(self) -> None:
        emb = Embedder("key", self.cache)
        with mock.patch.object(emb, "_fetch", return_value=None):
            self.assertIsNone(emb.embed(["anything"]))

    def test_the_cache_survives_a_restart(self) -> None:
        emb = Embedder("key", self.cache)
        with mock.patch.object(emb, "_fetch", side_effect=_fake_fetch):
            emb.embed(["remember me"])
        self.assertTrue(os.path.isfile(self.cache))
        fresh = Embedder("key", self.cache)
        with mock.patch.object(fresh, "_fetch", side_effect=AssertionError("should not fetch a cached text")):
            self.assertIsNotNone(fresh.embed_one("remember me"))


class MakeEmbedderTest(unittest.TestCase):
    def test_none_without_a_key_or_a_cache_path(self) -> None:
        self.assertIsNone(make_embedder({"OPENAI_API_KEY": ""}, os.path.join(tempfile.mkdtemp(), "c.json")))
        self.assertIsNone(make_embedder({"OPENAI_API_KEY": "key"}, ""))

    def test_built_when_a_key_and_a_path_are_present(self) -> None:
        emb = make_embedder({"OPENAI_API_KEY": "key"}, os.path.join(tempfile.mkdtemp(), "c.json"))
        self.assertIsInstance(emb, Embedder)


if __name__ == "__main__":
    unittest.main()
