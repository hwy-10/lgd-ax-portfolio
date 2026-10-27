"""Replaceable retrieval interface and ChromaDB vector retriever."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from config import COLLECTION_NAME, SIMILARITY_THRESHOLD, TOP_K, VECTOR_DB_DIR


@dataclass(frozen=True)
class SearchResult:
    text: str
    source: str
    page: int
    section: str
    similarity: float
    metadata: dict[str, Any]


class Retriever(Protocol):
    def search(self, query: str, top_k: int = TOP_K) -> list[SearchResult]: ...


class ChromaVectorRetriever:
    """Vector-only Phase 1 retriever. 선택 경로이며 기본 검색은 src.pipeline 이다."""

    def __init__(
        self,
        embedder: Any | None = None,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        import chromadb
        from chromadb.errors import NotFoundError

        from src.embeddings import OpenAIEmbedder

        self.threshold = threshold
        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        try:
            self.collection = client.get_collection(COLLECTION_NAME)
        except NotFoundError as error:
            raise RuntimeError(
                "Vector DB collection not found. Run `python -m src.ingest` first."
            ) from error
        self.embedder = embedder or OpenAIEmbedder()

    def search(self, query: str, top_k: int = TOP_K) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        response = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        results: list[SearchResult] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            metadata = dict(metadata or {})
            similarity = max(-1.0, min(1.0, 1.0 - float(distance)))
            if similarity < self.threshold:
                continue
            results.append(
                SearchResult(
                    text=text,
                    source=str(metadata.get("source", "Unknown")),
                    page=int(metadata.get("page", 0)),
                    section=str(metadata.get("section", "Unknown")),
                    similarity=similarity,
                    metadata=metadata,
                )
            )
        return results
