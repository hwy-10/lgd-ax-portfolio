"""모델에 넘기는 지시문과 근거 블록."""

from __future__ import annotations

from typing import Iterable, Protocol

NOT_FOUND_ANSWER = "제공된 데이터시트에서 해당 내용을 확인할 수 없습니다."

def system_prompt(document: str = "WCH CH334/CH335 데이터시트") -> str:
    """문서 이름만 바꾼 지시문. CH334 한 권일 때의 문장은 그대로 둔다 (평가 재현)."""
    return SYSTEM_TEMPLATE.replace("{document}", document)


SYSTEM_TEMPLATE = f"""너는 {{document}} 검색 도구의 답변 생성기다.
규칙
1. 아래 CONTEXT 에 적힌 내용만 근거로 답한다. 내부 지식으로 보충하거나 추측하지 않는다.
2. 질문한 품번과 다른 품번의 값을 섞지 않는다. CONTEXT 에 품번별로 값이 다르면 질문한 품번의 값만 쓴다.
3. 근거가 부족하면 정확히 다음 문장만 답한다: "{NOT_FOUND_ANSWER}"
4. 수치는 단위와 함께 쓰고, 문장 끝에 근거 페이지를 [p.N] 형식으로 붙인다. N 은 CONTEXT 의 Page 값이다.
5. 한국어로 3문장 이내로 답한다."""

SYSTEM_PROMPT = system_prompt()


class _Evidence(Protocol):
    text: str
    source: str
    page: int
    section: str


def build_context(results: Iterable[_Evidence]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        blocks.append(
            f"[CONTEXT {index}]\n"
            f"Source: {result.source}\n"
            f"Page: {result.page}\n"
            f"Section: {result.section}\n"
            f"Text:\n{result.text}"
        )
    return "\n\n".join(blocks)


def build_input(question: str, results: Iterable[_Evidence]) -> str:
    return f"CONTEXT:\n{build_context(results)}\n\nQUESTION:\n{question}"
