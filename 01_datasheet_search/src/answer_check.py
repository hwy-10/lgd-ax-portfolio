"""생성된 답변을 넘긴 근거 원문과 대조한다.

모델이 근거 밖의 정보를 썼는지는 모델에게 묻지 않고 문자열로 확인한다.

  인용 검사   답변이 붙인 [p.N] 이 실제로 넘긴 근거 페이지인가
  수치 검사   단위가 붙은 수치가 근거 원문에 있는가
  거절 검사   거절 문구로 답했는가

한계: 수치 검사는 숫자가 근거에 '있는지' 만 본다. 그 숫자가 질문한 항목의
값인지(같은 줄의 다른 항목 값인지)까지는 판정하지 않는다. 그래서 평가에서는
정답 사실(evaluation_questions.json 의 must_include)과 함께 본다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from src.prompt import NOT_FOUND_ANSWER

_CITATION = re.compile(
    r"(?:\bp\.?\s*|\bpage\s*)(\d{1,3})|(\d{1,3})\s*(?:페이지|쪽)", re.IGNORECASE
)
_UNITS = (
    r"k?Ω|KΩ|kHz|KHz|MHz|GHz|Hz|Mbps|Gbps|mA|uA|µA|μA|mV|kV|KV|V|"
    r"mS|ms|uS|us|µs|μs|℃|°C|mm|mil|uF|µF|μF|nF|pF|mW|W|A"
)
# 4.5~5.25V 같은 범위는 두 수치 모두 검사한다. 부호를 지키므로 -40 과 +40 은 다른 값이다.
# 부호는 숫자 바로 앞에 붙은 것만 본다 ("3.2-3.5" 의 - 는 범위 구분자).
_SIGNED = r"[-–−+]?\d+(?:\.\d+)?"
_NUMBER_WITH_UNIT = re.compile(
    rf"(?<![\d.])({_SIGNED})(?:\s*(?:~|to|–|-|−)\s*({_SIGNED}))?\s*(?:{_UNITS}|%)(?![A-Za-z])"
)
# 근거 쪽: 앞이 숫자나 영문이 아닐 때만 부호로 읽는다 ("VDD33-0.6" 은 0.6)
_CONTEXT_NUMBER = re.compile(r"(?<![\d.A-Za-z])([-–−]?)(\d+(?:\.\d+)?)")


def _signed_value(token: str) -> float:
    token = token.replace("–", "-").replace("−", "-").lstrip("+")
    return float(token)


@dataclass(frozen=True)
class AnswerCheck:
    refused: bool
    cited_pages: tuple[int, ...] = field(default_factory=tuple)
    invalid_citations: tuple[int, ...] = field(default_factory=tuple)
    numbers: tuple[str, ...] = field(default_factory=tuple)
    unsupported_numbers: tuple[str, ...] = field(default_factory=tuple)
    mixed_refusal: bool = False  # 답을 하고 거절 문장도 덧붙임 (읽는 사람이 어느 쪽인지 모른다)

    @property
    def has_citation(self) -> bool:
        return bool(self.cited_pages)

    @property
    def passed(self) -> bool:
        if self.refused:
            return True
        return (
            self.has_citation
            and not self.invalid_citations
            and not self.unsupported_numbers
            and not self.mixed_refusal
        )

    def summary(self) -> str:
        if self.refused:
            return "거절 문구로 응답"
        parts = [
            f"인용 {len(self.cited_pages)}건"
            + (f" (근거 밖 페이지 {list(self.invalid_citations)})" if self.invalid_citations else ""),
            f"수치 {len(self.numbers)}건"
            + (f" (원문에 없음 {list(self.unsupported_numbers)})" if self.unsupported_numbers else ""),
        ]
        if not self.has_citation:
            parts.append("인용 없음")
        if self.mixed_refusal:
            parts.append("답과 거절 문장이 섞임")
        return ", ".join(parts)


def _without_refusal(answer: str) -> str:
    """거절 문장과 인용 표기를 지우고 남는 내용 (글자, 숫자만)."""
    body = answer.replace(" ", "").replace(NOT_FOUND_ANSWER.replace(" ", ""), "")
    body = _CITATION.sub("", body)
    return re.sub(r"[^0-9A-Za-z가-힣]", "", body)


def mentions_refusal(answer: str) -> bool:
    return NOT_FOUND_ANSWER.replace(" ", "") in answer.replace(" ", "")


def is_refusal(answer: str) -> bool:
    """거절 문장만으로 된 답. 내용을 답하고 거절 문장을 덧붙인 경우는 거절이 아니다."""
    return mentions_refusal(answer) and not _without_refusal(answer)


def cited_pages(answer: str) -> list[int]:
    pages: list[int] = []
    for match in _CITATION.finditer(answer):
        value = int(match.group(1) or match.group(2))
        if value not in pages:
            pages.append(value)
    return pages


def numbers_with_units(answer: str) -> list[str]:
    found: list[str] = []
    for match in _NUMBER_WITH_UNIT.finditer(answer):
        for value in match.groups():
            if value and value not in found:
                found.append(value)
    return found


def _context_numbers(texts: Iterable[str]) -> set[float]:
    values: set[float] = set()
    for text in texts:
        for sign, number in _CONTEXT_NUMBER.findall(text):
            values.add(-float(number) if sign else float(number))
    return values


def check_answer(answer: str, evidence_pages: Iterable[int], evidence_texts: Iterable[str]) -> AnswerCheck:
    if is_refusal(answer):
        return AnswerCheck(refused=True)
    allowed_pages = set(evidence_pages)
    pages = cited_pages(answer)
    invalid = [page for page in pages if page not in allowed_pages]

    # 인용 표기 안의 숫자는 수치 검사에서 뺀다
    body = _CITATION.sub(" ", answer)
    numbers = numbers_with_units(body)
    context = _context_numbers(evidence_texts)
    unsupported = [number for number in numbers if _signed_value(number) not in context]
    return AnswerCheck(
        refused=False,
        cited_pages=tuple(pages),
        invalid_citations=tuple(invalid),
        numbers=tuple(numbers),
        unsupported_numbers=tuple(unsupported),
        mixed_refusal=mentions_refusal(answer),
    )
