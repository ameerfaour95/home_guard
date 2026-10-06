"""Semantic embeddings for RAG over the box's saved alerts.

The owner asks the assistant things like "did anyone come to the door this
week" or "show me when the delivery came". The words they use rarely match the
VLM's summary word for word, so a keyword search misses. This module turns the
query and each alert summary into vectors with OpenAI's small embedding model
and ranks by cosine similarity, so a close-in-meaning summary is found even
when no word is shared.

It degrades safely. With no API key, or when the call fails (the box may be
offline), the embedder is absent or ``embed`` returns ``None``, and the caller
falls back to the plain time + keyword search. Vectors are cached on disk by a
hash of the text, so each alert summary is embedded once, not on every
question; the cache is bounded because the alert corpus only spans the
retention window.

Only the standard library plus httpx (already a dependency), and it uses the OS
trust store like the VLM and chat backends, so it works where TLS is
intercepted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import ssl
from typing import Dict, List, Optional, Sequence

from . import providers

log = logging.getLogger("box.embeddings")

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMENSIONS = 256          # shortened vectors: plenty for one-sentence summaries, small on disk
MAX_CACHE = 5000                # bounded; the live alert corpus only spans the retention window


def _key(text: str) -> str:
    return hashlib.sha1((text or "").strip().lower().encode("utf-8")).hexdigest()


def cosine(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 if either is empty, zero or mismatched."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class Embedder:
    """Embeds short texts with OpenAI's small model, caching vectors on disk.

    Never raises: ``embed`` returns ``None`` when the model cannot be reached,
    so the caller can fall back to a non-semantic search.
    """

    def __init__(self, api_key: str, cache_path: str, model: str = EMBED_MODEL,
                 dimensions: int = EMBED_DIMENSIONS, timeout: float = 20.0) -> None:
        self._api_key = api_key
        self._cache_path = cache_path
        self._model = model
        self._dim = dimensions
        self._timeout = timeout
        self._cache: Dict[str, List[float]] = self._load_cache()

    def _load_cache(self) -> Dict[str, List[float]]:
        try:
            with open(self._cache_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if isinstance(v, list)}
        except (OSError, ValueError):
            pass
        return {}

    def _save_cache(self) -> None:
        if len(self._cache) > MAX_CACHE:                 # keep the most recently added
            self._cache = dict(list(self._cache.items())[-MAX_CACHE:])
        tmp = self._cache_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
            os.replace(tmp, self._cache_path)
        except OSError as exc:
            log.warning("Could not save embedding cache: %s", exc)

    def _fetch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """One embeddings call for *texts*, or None on any failure."""
        import httpx  # noqa: PLC0415

        payload = {"model": self._model, "input": texts, "dimensions": self._dim}
        try:
            resp = httpx.post(
                providers.openai_url("/embeddings"), json=payload, timeout=self._timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
                verify=ssl.create_default_context(),
            )
            resp.raise_for_status()
            rows = resp.json().get("data", [])
            vecs = [row.get("embedding") for row in rows]
            if len(vecs) == len(texts) and all(isinstance(v, list) for v in vecs):
                return vecs
            log.warning("Embeddings response had %d vectors for %d inputs", len(vecs), len(texts))
        except Exception as exc:  # noqa: BLE001 - offline, TLS, rate limit: fall back to keyword search
            log.warning("Embeddings call failed: %s", exc)
        return None

    def embed(self, texts: Sequence[str]) -> Optional[List[Optional[List[float]]]]:
        """Vectors for each text, cache first and one call for the misses.

        Returns a list the same length as *texts* (an entry is ``None`` only if
        that text could not be embedded), or ``None`` when nothing could be
        embedded at all so the caller falls back.
        """
        wanted = [t or "" for t in texts]
        missing = [t for t in wanted if _key(t) not in self._cache]
        if missing:
            uniq = list(dict.fromkeys(missing))          # de-dup, preserve order
            fetched = self._fetch(uniq)
            if fetched is not None:
                for t, vec in zip(uniq, fetched):
                    self._cache[_key(t)] = vec
                self._save_cache()
        out = [self._cache.get(_key(t)) for t in wanted]
        if all(v is None for v in out):
            return None
        return out

    def embed_one(self, text: str) -> Optional[List[float]]:
        got = self.embed([text])
        return got[0] if got else None


def make_embedder(env: Optional[Dict[str, str]] = None, cache_path: str = "") -> Optional[Embedder]:
    """An :class:`Embedder` when a key and a cache path are available, else ``None``."""
    env = dict(os.environ if env is None else env)
    key = env.get("OPENAI_API_KEY", "")
    if not key or not cache_path:
        return None
    return Embedder(key, cache_path)
