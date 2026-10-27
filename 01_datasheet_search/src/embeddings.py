"""OpenAI embedding client and document/query embedding helpers."""

from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, OPENAI_API_KEY


class OpenAIEmbedder:
    """Generate document and query vectors using the same embedding model."""

    def __init__(
        self,
        api_key: str | None = OPENAI_API_KEY,
        model: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        client: OpenAI | None = None,
    ) -> None:
        if client is None and not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Copy .env.example to .env.")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.client = client or OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding input cannot contain empty text")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self.client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        return vectors

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        return self.embed_documents([query])[0]
