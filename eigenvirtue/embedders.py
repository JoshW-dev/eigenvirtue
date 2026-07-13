"""Embedding backends. Every embedder maps a list of texts to a (n, d) array
of L2-normalized vectors, so axes and scores are comparable across backends."""

from __future__ import annotations

import os

import numpy as np


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class LocalEmbedder:
    """Any sentence-transformers model, e.g. 'all-MiniLM-L6-v2',
    'all-mpnet-base-v2', 'BAAI/bge-base-en-v1.5'. Runs on CPU/MPS."""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalize(np.asarray(vectors, dtype=np.float64))


class OpenAIEmbedder:
    """Optional API backend for cross-model comparison. Requires OPENAI_API_KEY."""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        from openai import OpenAI

        self.name = f"openai/{model_name}"
        self._model_name = model_name
        self._client = OpenAI()

    def encode(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self._model_name, input=texts)
        vectors = np.array([item.embedding for item in response.data], dtype=np.float64)
        return _normalize(vectors)


def get_embedder(spec: str):
    """'openai:<model>' selects the API backend; anything else is a local
    sentence-transformers model name."""
    if spec.startswith("openai:"):
        return OpenAIEmbedder(spec.split(":", 1)[1])
    return LocalEmbedder(spec)
