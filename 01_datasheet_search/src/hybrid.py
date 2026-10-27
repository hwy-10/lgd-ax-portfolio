"""Dense + Lexical 하이브리드 검색기 (Phase 2).

검색 순서
  1. 질의에서 대상 품번을 뽑는다                        variants.query_part
  2. 의미 검색과 문자 일치 검색을 각각 돌린다             dense / BM25
  3. 다른 변종 전용 청크를 떨어뜨린다                    변종 게이트
  4. 두 순위를 RRF 로 합친다                            점수 스케일이 달라 순위만 쓴다
  5. 근거에 해당 품번이 실제로 있는지 확인한다            검증 게이트

부품 스펙은 틀린 답보다 "모른다"가 낫다. 5단에서 근거를 못 찾으면 빈 결과를
돌려주고, query.py 가 "확인할 수 없습니다"로 응답한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.lexical import BM25Index
from src.retriever import SearchResult
from src.variants import PartNumber, RegexRules, VariantRules

RRF_K = 60


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion.

    score(d) = Σ 1/(k + rank_i(d))
    두 검색기의 점수 체계(코사인 유사도 vs BM25)가 달라 정규화가 어려우므로
    점수 대신 순위만 쓴다.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_index in enumerate(ranking, start=1):
            scores[doc_index] = scores.get(doc_index, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return fused


@dataclass
class Document:
    text: str
    metadata: dict[str, Any]


class HybridIndex:
    """Chroma 없이도 동작하는 순수 로직. 테스트가 이 클래스를 직접 쓴다."""

    def __init__(self, documents: list[Document], rules: VariantRules | None = None) -> None:
        self.documents = documents
        self.bm25 = BM25Index([document.text for document in documents])
        # 품번을 어떻게 읽을지. 기본은 CH334 한 권에 맞춘 정규식, 여러 문서에서는 문서별 등록 규칙
        self.rules = rules or RegexRules()

    def gate(self, target: PartNumber | None) -> list[int]:
        """변종 게이트를 통과한 문서 인덱스."""
        if target is None:
            return list(range(len(self.documents)))
        return [
            index
            for index, document in enumerate(self.documents)
            if self.rules.applies(document.text, target)
        ]

    def search(
        self,
        question: str,
        dense_ranking: Sequence[int] | None = None,
        top_k: int = 5,
        require_part_evidence: bool = True,
        lexical_query: str | None = None,
    ) -> list[int]:
        """lexical_query 는 BM25 에만 쓰는 질의다 (한국어 용어를 영문으로 확장한 것).
        품번 판정은 항상 원래 질문으로 한다."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        target = self.rules.primary(question)
        allowed = set(self.gate(target))

        lexical_ranking = [
            index
            for index, _ in self.bm25.rank(lexical_query or question)
            if index in allowed
        ]
        rankings: list[Sequence[int]] = [lexical_ranking]
        if dense_ranking is not None:
            rankings.append([index for index in dense_ranking if index in allowed])

        fused = [index for index, _ in rrf_fuse(rankings)]

        if target is not None and target.is_specific and require_part_evidence:
            # 검색 결과가 아니라 게이트를 통과한 전체에서 찾는다.
            # 순위용 질의에서 품번을 빼면 품번 명시 청크가 순위에 안 잡힐 수 있다.
            explicit = [
                index
                for index in sorted(allowed)
                if self.rules.names(self.documents[index].text, target)
            ]
            # 품번을 명시한 근거가 하나도 없으면 답을 만들지 않는다
            if not explicit:
                return []
            # 품번을 명시한 근거 하나는 반드시 남기고 나머지는 검색 순위를 따른다.
            # 명시 청크를 전부 앞세우면 핀 표가 상위를 채워, 계열 공통인
            # 전기적 특성 표(품번 미기재)가 밀려난다.
            top = fused[:top_k]
            explicit_set = set(explicit)
            if not any(index in explicit_set for index in top):
                content = lexical_query or question
                best = max(
                    explicit,
                    key=lambda i: (self.bm25.score(content, i), self.bm25.score(question, i)),
                )
                top = top[: top_k - 1] + [best]
            return top

        return fused[:top_k]


class HybridRetriever:
    """src.retriever.Retriever Protocol 구현체. query.py 가 그대로 쓴다."""

    def __init__(
        self,
        embedder: Any | None = None,
        collection: Any | None = None,
        threshold: float = 0.0,
    ) -> None:
        if collection is None:
            import chromadb
            from chromadb.errors import NotFoundError

            from config import COLLECTION_NAME, VECTOR_DB_DIR

            client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
            try:
                collection = client.get_collection(COLLECTION_NAME)
            except NotFoundError as error:
                raise RuntimeError(
                    "Vector DB collection not found. Run `python -m src.ingest` first."
                ) from error

        self.collection = collection
        self.threshold = threshold
        payload = collection.get(include=["documents", "metadatas"])
        texts = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []
        self.index = HybridIndex(
            [
                Document(text=text, metadata=dict(metadata or {}))
                for text, metadata in zip(texts, metadatas)
            ]
        )
        self._embedder = embedder
        self._dense_available = embedder is not None

    def _dense_ranking(self, question: str, pool: int) -> list[int] | None:
        """의미 검색 순위를 문서 인덱스 목록으로 돌려준다."""
        if not self._dense_available:
            return None
        response = self.collection.query(
            query_embeddings=[self._embedder.embed_query(question)],
            n_results=min(pool, len(self.index.documents)),
            include=["documents"],
        )
        documents = (response.get("documents") or [[]])[0]
        lookup = {
            document.text: index
            for index, document in enumerate(self.index.documents)
        }
        return [lookup[text] for text in documents if text in lookup]

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        dense = self._dense_ranking(query, pool=max(top_k * 6, 30))
        indices = self.index.search(query, dense_ranking=dense, top_k=top_k)
        results: list[SearchResult] = []
        for rank, index in enumerate(indices, start=1):
            document = self.index.documents[index]
            metadata = document.metadata
            results.append(
                SearchResult(
                    text=document.text,
                    source=str(metadata.get("source", "Unknown")),
                    page=int(metadata.get("page", 0)),
                    section=str(metadata.get("section", "Unknown")),
                    similarity=1.0 / rank,  # 융합 순위. 코사인 유사도가 아니다
                    metadata=metadata,
                )
            )
        return results
