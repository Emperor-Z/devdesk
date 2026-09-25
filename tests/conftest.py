"""Shared test fixtures — no network/API calls anywhere in the unit suite."""

from __future__ import annotations

import math

import pytest

_HASH_DIM = 64


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder: no network, no model
    weights, and a fixed dimension across calls (unlike a per-call
    vocabulary, which would produce mismatched vector lengths between the
    documents-embedding call and the later query-embedding call).

    Cosine similarity between two texts tracks their shared-word overlap
    closely enough to exercise ranking and the relevance cutoff.
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        vec = [0.0] * _HASH_DIM
        for w in text.lower().split():
            vec[hash(w) % _HASH_DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
