"""Embedding backends: Gemini (primary) with a local MiniLM fallback."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from devdesk import config
from devdesk.llm_client import with_retry


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class GeminiEmbedder:
    """Primary backend: gemini-embedding-001, free-tier via AI Studio."""

    def __init__(self, model: str = config.GEMINI_EMBEDDING_MODEL) -> None:
        from google import genai

        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)
        self._model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        def _call() -> list[list[float]]:
            result = self._client.models.embed_content(model=self._model, contents=texts)
            return [e.values for e in result.embeddings]

        return with_retry(_call)


class MiniLMEmbedder:
    """Offline fallback: local sentence-transformers, no API key needed."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "No GOOGLE_API_KEY set and the offline embedder isn't installed: "
                "set the key, or `pip install -e '.[offline]'`"
            ) from exc

        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=False)
        return [v.tolist() for v in vectors]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if config.use_gemini_embeddings():
        return GeminiEmbedder()
    return MiniLMEmbedder()
