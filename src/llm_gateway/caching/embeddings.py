"""Embedding backends for semantic cache.

Two implementations:

  * HashingEmbedder — pure NumPy. Hashes character n-grams into a fixed-
    dim vector and L2-normalizes. Deterministic, fast, and requires no
    network or API key. Suitable for dev and for detecting near-exact
    prompt reuse (which is the bulk of real-world cache hits).

  * OpenAIEmbedder — calls the OpenAI embeddings API. Higher semantic
    quality; used in production for diverse prompts.

Both implement ``embed(text) -> list[float]`` returning unit vectors.
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    async def embed(self, text: str) -> list[float]: ...


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


class HashingEmbedder:
    """Feature-hashing embedder over character n-grams.

    Produces deterministic unit vectors suitable for cosine similarity.
    """

    def __init__(self, dim: int = 256, ngram_range: tuple[int, int] = (3, 5)) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._ngram_range = ngram_range

    def _hash_to_index(self, token: str) -> tuple[int, int]:
        # Use two separate hashes: one for bucket, one for sign (+1/-1).
        # This reduces collision bias ("signed feature hashing").
        h1 = int.from_bytes(
            hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        h2 = int.from_bytes(
            hashlib.blake2b(token.encode("utf-8"), digest_size=8, key=b"sign").digest(),
            "big",
        )
        return h1 % self.dim, 1 if (h2 & 1) else -1

    def _tokens(self, text: str):
        text = text.lower()
        lo, hi = self._ngram_range
        for n in range(lo, hi + 1):
            if len(text) < n:
                continue
            for i in range(len(text) - n + 1):
                yield text[i : i + n]
        # Also yield whole words as unigram features
        for word in text.split():
            if word:
                yield f"w:{word}"

    async def embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec.tolist()

        for tok in self._tokens(text):
            idx, sign = self._hash_to_index(tok)
            vec[idx] += sign

        # Sub-linear TF scaling to dampen long-prompt dominance.
        vec = np.sign(vec) * np.log1p(np.abs(vec))
        vec = _l2_normalize(vec)
        return vec.tolist()


class OpenAIEmbedder:
    """OpenAI embeddings API adapter. Uses text-embedding-3-small by default."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        timeout_seconds: float = 30.0,
    ) -> None:
        import httpx  # lazy

        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
        )
        self._model = model
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        # OpenAI returns already-normalized embeddings for 3-series models.
        resp = await self._client.post(
            "/embeddings",
            json={"input": text, "model": self._model},
        )
        resp.raise_for_status()
        data = resp.json()
        vec = data["data"][0]["embedding"]
        # Defensive normalization
        arr = np.array(vec, dtype=np.float32)
        return _l2_normalize(arr).tolist()

    async def aclose(self) -> None:
        await self._client.aclose()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity for unit vectors == dot product. Safe for any vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
