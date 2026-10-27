"""부품 변종(품번) 추출과 적용 범위 판정.

데이터시트 한 권에 CH334U · CH334R 처럼 접미사만 다른 변종이 함께 들어 있다.
CH334_CH335_V2.5.pdf 한 파일에 11종이 섞여 있고, 핀·패키지 비교표 페이지에는
11종이 한 페이지에 모두 등장한다.

따라서 config.DOCUMENT_METADATA 의 문서 단위 part_number 로는 변종을 구분할 수 없다.
청크 텍스트에서 직접 뽑아 청크 단위로 판정한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

# 품번 패턴: 영문 2~5자 + 숫자 2자 이상 + 접미사 0~2자
#   CH334U, CH334R, CH335P, TUSB2046B, TPS2552 ...
#   숫자 2자 이상을 요구해 USB2 · INT8 같은 일반 토큰을 걸러낸다
#  는 한글 조사 앞에서 깨진다. "CH334R는" 에서 R 과 는 이 모두 단어 문자라
# 경계가 생기지 않아 품번이 추출되지 않는다. 라틴 문자·숫자만 배제하는 전후방 탐색을 쓴다.
PART_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])([A-Z]{2,5}\d{2,6})([A-Z]{0,2})(?![A-Za-z0-9])"
)


@dataclass(frozen=True)
class PartNumber:
    """품번 하나. family 는 접미사를 뗀 계열명이다."""

    raw: str
    family: str
    suffix: str

    @property
    def is_specific(self) -> bool:
        """접미사가 있으면 특정 변종, 없으면 계열 전체를 가리킨다."""
        return bool(self.suffix)


def parse_part(token: str) -> PartNumber | None:
    match = PART_PATTERN.fullmatch(token.strip().upper())
    if not match:
        return None
    family, suffix = match.group(1), match.group(2)
    return PartNumber(raw=family + suffix, family=family, suffix=suffix)


def extract_parts(text: str) -> list[PartNumber]:
    """텍스트에 등장하는 품번을 등장 순서대로, 중복 없이 돌려준다."""
    seen: set[str] = set()
    parts: list[PartNumber] = []
    for family, suffix in PART_PATTERN.findall(text.upper()):
        raw = family + suffix
        if raw in seen:
            continue
        seen.add(raw)
        parts.append(PartNumber(raw=raw, family=family, suffix=suffix))
    return parts


def query_part(question: str) -> PartNumber | None:
    """질문에서 대상 품번 하나를 고른다. 특정 변종을 계열명보다 우선한다."""
    parts = extract_parts(question)
    if not parts:
        return None
    for part in parts:
        if part.is_specific:
            return part
    return parts[0]


def chunk_applies_to(chunk_text: str, target: PartNumber) -> bool:
    """이 청크가 target 품번에 적용되는 내용인지 판정한다.

    포함   ① target 을 명시한 청크                     (변종 특정 내용)
          ② 같은 계열만 언급하고 변종을 특정하지 않은 청크 (계열 공통 내용)
          ③ 품번이 전혀 없는 청크                      (일반 설명)
    제외   ④ 다른 변종만 명시한 청크                    ← CH334U 질의에 CH334R 표가 오는 경우
    """
    parts = extract_parts(chunk_text)
    if not parts:
        return True

    specific = [p for p in parts if p.is_specific]
    if any(p.raw == target.raw for p in specific):
        return True
    if not specific:
        return True
    return False


def mentions_exactly(chunk_text: str, target: PartNumber) -> bool:
    """청크 안에 target 품번 문자열이 실제로 있는지 확인한다 (검증 게이트용)."""
    return any(p.raw == target.raw for p in extract_parts(chunk_text))


class VariantRules(Protocol):
    """품번을 읽는 방식. 게이트와 검증 게이트, 순위 질의 정리가 이 인터페이스만 쓴다."""

    def primary(self, question: str) -> PartNumber | None: ...

    def specific_targets(self, question: str) -> list[PartNumber]: ...

    def part_strings(self, question: str) -> set[str]: ...

    def applies(self, chunk_text: str, target: PartNumber) -> bool: ...

    def names(self, chunk_text: str, target: PartNumber) -> bool: ...


class RegexRules:
    """CH334/CH335 한 권에 맞춘 기본 규칙. 영문 2~5자 + 숫자 + 접미사 정규식 하나로 읽는다."""

    def primary(self, question: str) -> PartNumber | None:
        return query_part(question)

    def specific_targets(self, question: str) -> list[PartNumber]:
        return [p for p in extract_parts(question) if p.is_specific]

    def part_strings(self, question: str) -> set[str]:
        return {p.raw for p in extract_parts(question)}

    def applies(self, chunk_text: str, target: PartNumber) -> bool:
        return chunk_applies_to(chunk_text, target)

    def names(self, chunk_text: str, target: PartNumber) -> bool:
        return mentions_exactly(chunk_text, target)
