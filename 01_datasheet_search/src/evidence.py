"""근거 충분성 판정 — 모델을 부를지 말지 정한다.

검색은 항상 무언가를 돌려준다. 관련이 낮은 근거를 그대로 넘기면 모델은
인접한 내용을 엮어 그럴듯한 답을 만든다. 그래서 모델 호출 전에 두 조건을 본다.

  ① BM25 1위 점수 ≥ min_score      질문과 겹치는 근거가 있기는 한가
  ② 용어 등장 비율 ≥ min_coverage   질문이 묻는 대상이 근거에 실제로 있는가

점수만으로는 막지 못한다. "CH334U 는 HDMI 2.1 을 지원하는가" 는 품번이 맞아
점수가 높게 나오지만 HDMI 는 문서 어디에도 없다. ② 가 이 경우를 잡는다.

두 기준값은 평가셋으로 정한다 (calibrate.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from src.glossary import Concept

_TOKEN = re.compile(r"[0-9a-z]+")


@dataclass(frozen=True)
class Sufficiency:
    passed: bool
    top_score: float
    coverage: float
    found: tuple[str, ...] = field(default_factory=tuple)
    missing: tuple[str, ...] = field(default_factory=tuple)
    min_score: float = 0.0
    min_coverage: float = 0.0
    reason: str = ""


def term_coverage(
    concepts: Iterable[Concept], texts: Iterable[str]
) -> tuple[float, list[str], list[str]]:
    """판정 용어 중 근거에 등장한 비율.

    용어가 하나도 없으면 0.0 이다. 무엇을 묻는지 판정할 수 없는 질문("CH334U의 색상은?")을
    통과시키면 모델이 근거와 무관한 답을 만든다. 판정할 수 없으면 답하지 않는다.
    """
    vocabulary: set[str] = set()
    for text in texts:
        vocabulary.update(_TOKEN.findall(text.lower()))
    found: list[str] = []
    missing: list[str] = []
    for concept in concepts:
        if all(any(word in vocabulary for word in group) for group in concept.required):
            found.append(concept.label)
        else:
            missing.append(concept.label)
    total = len(found) + len(missing)
    return (len(found) / total if total else 0.0), found, missing


def judge(
    concepts: Iterable[Concept],
    texts: list[str],
    top_score: float,
    min_score: float,
    min_coverage: float,
) -> Sufficiency:
    concepts = list(concepts)
    if not texts:
        return Sufficiency(
            passed=False,
            top_score=0.0,
            coverage=0.0,
            missing=tuple(c.label for c in concepts),
            min_score=min_score,
            min_coverage=min_coverage,
            reason="게이트를 통과한 근거가 없음",
        )
    coverage, found, missing = term_coverage(concepts, texts)
    reasons = []
    if not concepts:
        reasons.append("질문에서 판정할 용어를 찾지 못함 (용어표에 없는 표현)")
    if top_score < min_score:
        reasons.append(f"1위 점수 {top_score:.2f} < 기준 {min_score:.2f}")
    if concepts and coverage < min_coverage:
        reasons.append(
            f"용어 등장 비율 {coverage:.0%} < 기준 {min_coverage:.0%}"
            + (f" (없음: {', '.join(missing)})" if missing else "")
        )
    return Sufficiency(
        passed=not reasons,
        top_score=top_score,
        coverage=coverage,
        found=tuple(found),
        missing=tuple(missing),
        min_score=min_score,
        min_coverage=min_coverage,
        reason="; ".join(reasons) if reasons else "기준 충족",
    )
