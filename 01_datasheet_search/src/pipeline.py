"""질문부터 답변까지. 벡터 DB 없이 PDF -> 청크 -> BM25 로 돈다.

  1. 질문에서 품번을 뽑는다                         variants.query_part
  2. 한국어 용어를 영문으로 바꾼다                   glossary.expand
  3. 다른 변종 전용 청크를 뺀다                      변종 게이트
  4. BM25 로 순위를 매긴다
  5. 질문한 품번이 문서에 실제로 있는지 본다           검증 게이트
  6. 근거가 충분한지 본다                           evidence.judge
  7. 통과한 경우에만 모델을 부른다
  8. 답변을 근거 원문과 대조한다                     answer_check

단계마다 판정 결과를 Trace 에 남긴다. 시연 화면과 평가 스크립트가 같은 Trace 를 쓴다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_METADATA,
    EVIDENCE_MIN_COVERAGE,
    EVIDENCE_MIN_SCORE,
    MIN_CHUNK_SIZE,
    PDF_PATH,
    TOP_K,
)
from src.answer_check import AnswerCheck, check_answer
from src.chunker import create_chunks
from src.evidence import Sufficiency, judge
from src.glossary import Concept, expand
from src.hybrid import Document, HybridIndex
from src.ingest import extract_pages
from src.lexical import word_tokenize
from src.llm import Completion, complete
from src.prompt import NOT_FOUND_ANSWER, SYSTEM_PROMPT, build_input, system_prompt
from src.retriever import SearchResult


def build_index_from_pdf(pdf_path=PDF_PATH) -> HybridIndex:
    pages = extract_pages(pdf_path)
    metadata = {**DOCUMENT_METADATA, "source": pdf_path.name}
    chunks = create_chunks(pages, metadata, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE)
    return HybridIndex([Document(text=c.text, metadata=dict(c.metadata)) for c in chunks])


@lru_cache(maxsize=1)
def default_index() -> HybridIndex:
    return build_index_from_pdf()


def content_query(query: str, part_tokens: set[str]) -> str:
    """순위용 질의에서 품번을 뺀다.

    품번 n-gram 은 BM25 점수를 크게 끌어올려, 품번을 적은 핀 표가 묻는 내용과
    상관없이 상위를 차지한다 ("CH334U 동작 전압" 에 전기적 특성 표가 안 나옴).
    어느 품번의 근거인지는 변종 게이트와 검증 게이트가 보므로 순위는 내용으로 매긴다.
    품번 말고 남는 말이 없으면 원래 질의를 쓴다.
    """
    stripped = query
    # 긴 것부터 지운다. 계열명(TPS7A20)을 먼저 지우면 품번(TPS7A2033)의 꼬리 "33" 이 남는다.
    # 집합을 그대로 돌면 순서가 실행마다 달라 같은 질문의 검색 결과가 바뀌었다
    for raw in sorted(part_tokens, key=lambda t: (-len(t), t)):
        # 품번에 붙은 조사도 함께 뗀다 ("CH334U는" -> "")
        stripped = re.sub(re.escape(raw) + r"(?:은|는|이|가|의|을|를|에|와|과)?", " ",
                          stripped, flags=re.IGNORECASE)
    # 문서가 영문이라 한글 토큰만 남으면 내용으로 순위를 매길 수 없다
    has_latin = any(re.search(r"[a-z0-9]", t) for t in word_tokenize(stripped))
    return stripped if has_latin else query


@dataclass(frozen=True)
class Evidence:
    index: int
    text: str
    source: str
    page: int
    section: str
    score: float
    names_target: bool  # 질문한 품번을 직접 명시한 청크인가


@dataclass
class Trace:
    question: str
    lexical_query: str
    concepts: tuple[Concept, ...]
    target: str | None
    target_specific: bool
    total_chunks: int
    gate_allowed: int
    verification_failed: bool
    evidence: list[Evidence]
    sufficiency: Sufficiency
    model: str | None = None
    answer: str = ""
    completion: Completion | None = None
    check: AnswerCheck | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def gate_excluded(self) -> int:
        return self.total_chunks - self.gate_allowed

    @property
    def llm_called(self) -> bool:
        return self.completion is not None

    @property
    def blocked(self) -> bool:
        return not self.sufficiency.passed

    def search_results(self) -> list[SearchResult]:
        return [
            SearchResult(
                text=e.text,
                source=e.source,
                page=e.page,
                section=e.section,
                similarity=e.score,
                metadata={"bm25": e.score},
            )
            for e in self.evidence
        ]


class DatasheetSearch:
    def __init__(
        self,
        index: HybridIndex | None = None,
        min_score: float = EVIDENCE_MIN_SCORE,
        min_coverage: float = EVIDENCE_MIN_COVERAGE,
        top_k: int = TOP_K,
        document: str | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self.index = index or default_index()
        self.min_score = min_score
        self.min_coverage = min_coverage
        self.top_k = top_k
        self.instructions = system_prompt(document) if document else SYSTEM_PROMPT

    def retrieve(self, question: str) -> Trace:
        """모델 호출 전까지. API 키 없이 돈다."""
        if not question.strip():
            raise ValueError("질문이 비어 있습니다")
        rules = self.index.rules
        target = rules.primary(question)
        part_tokens = rules.part_strings(question)
        # 특정 변종이 둘 이상이면 비교 질문이다. 게이트가 한 품번만 보므로 답하지 않는다
        compared = sorted({p.raw for p in rules.specific_targets(question)})
        expansion = expand(question, part_tokens)
        ranking_query = content_query(expansion.lexical_query, part_tokens)

        allowed = self.index.gate(target)
        indices = self.index.search(
            question,
            top_k=self.top_k,
            lexical_query=ranking_query,
        )
        specific = target is not None and target.is_specific
        # 검증 게이트: 질문한 변종을 명시한 청크가 문서에 하나라도 있는가
        verification_failed = specific and not any(
            rules.names(self.index.documents[i].text, target) for i in allowed
        )

        evidence = []
        for position in indices:
            document = self.index.documents[position]
            metadata = document.metadata
            evidence.append(
                Evidence(
                    index=position,
                    text=document.text,
                    source=str(metadata.get("source", "Unknown")),
                    page=int(metadata.get("page", 0)),
                    section=str(metadata.get("section", "Unknown")),
                    score=self.index.bm25.score(ranking_query, position),
                    names_target=bool(target and rules.names(document.text, target)),
                )
            )

        sufficiency = judge(
            expansion.concepts,
            [e.text for e in evidence],
            top_score=max((e.score for e in evidence), default=0.0),
            min_score=self.min_score,
            min_coverage=self.min_coverage,
        )
        if verification_failed:
            sufficiency = Sufficiency(
                passed=False,
                top_score=0.0,
                coverage=0.0,
                missing=(target.raw,),
                min_score=self.min_score,
                min_coverage=self.min_coverage,
                reason=f"문서에 {target.raw} 를 명시한 근거가 없음 (검증 게이트)",
            )
        elif len(compared) > 1:
            sufficiency = Sufficiency(
                passed=False,
                top_score=sufficiency.top_score,
                coverage=sufficiency.coverage,
                found=sufficiency.found,
                missing=sufficiency.missing,
                min_score=self.min_score,
                min_coverage=self.min_coverage,
                reason=f"품번 {', '.join(compared)} 를 함께 묻는 비교 질문은 지원하지 않음 (한 품번씩 물어 주세요)",
            )

        return Trace(
            question=question,
            lexical_query=ranking_query,
            concepts=expansion.concepts,
            target=target.raw if target else None,
            target_specific=specific,
            total_chunks=len(self.index.documents),
            gate_allowed=len(allowed),
            verification_failed=verification_failed,
            evidence=evidence,
            sufficiency=sufficiency,
        )

    def answer(self, question: str, model: str, generate=complete) -> Trace:
        trace = self.retrieve(question)
        trace.model = model
        if trace.blocked:
            # 근거가 부족하면 모델을 부르지 않는다
            trace.answer = NOT_FOUND_ANSWER
            return trace
        try:
            completion = generate(model, self.instructions, build_input(question, trace.evidence))
        except Exception as error:  # 시연 중 네트워크 오류도 화면에 그대로 보여 준다
            trace.error = f"{type(error).__name__}: {error}"
            trace.answer = "(모델 호출 오류로 답변을 만들지 못했습니다. 검색 결과와 판정은 위와 같습니다)"
            return trace
        trace.completion = completion
        if completion.stop_reason == "refusal":
            trace.notes.append("모델 안전 분류기가 응답을 거절함")
        trace.answer = completion.text or NOT_FOUND_ANSWER
        trace.check = check_answer(
            trace.answer,
            [e.page for e in trace.evidence],
            [e.text for e in trace.evidence],
        )
        return trace


class LexicalRetriever:
    """src.retriever.Retriever Protocol 구현. 근거 부족이면 빈 목록을 돌려준다."""

    def __init__(self, search: DatasheetSearch | None = None) -> None:
        self.search_engine = search or DatasheetSearch()

    def search(self, query: str, top_k: int = TOP_K) -> list[SearchResult]:
        trace = self.search_engine.retrieve(query)
        if trace.blocked:
            return []
        return trace.search_results()[:top_k]
