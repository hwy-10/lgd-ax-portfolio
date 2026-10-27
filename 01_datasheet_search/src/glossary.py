"""한국어 기술 용어를 데이터시트 영문 용어로 바꾼다.

데이터시트는 영문이고 질문은 한국어로 들어온다. BM25 는 글자가 같아야 맞으므로
"동작 전압" 은 "operating voltage" 로 바꿔 줘야 검색된다.

번역 모델을 쓰지 않고 표를 쓴 이유
  같은 질문은 항상 같은 질의가 되어야 평가가 재현된다.
  어떤 단어가 어떤 영문으로 바뀌었는지 화면에 그대로 보여 줄 수 있다.

표의 값은 "묶음" 의 나열이다. 묶음 안의 | 는 동의어(하나만 있으면 됨),
묶음끼리는 모두 있어야 한다. "입력 전압" 은 input 과 voltage 가 둘 다 근거에 있어야
찾은 것으로 본다. voltage 하나만으로 "입력 전압" 과 "출력 전압" 이 모두 충족되던 것을 막는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 긴 표현을 먼저 맞춘다 ("동작 전압" 이 "전압" 보다 앞).
GLOSSARY: dict[str, tuple[str, ...]] = {
    "동작 전압": ("operating|supply", "voltage"),
    "동작 전류": ("operating|supply", "current"),
    "동작 온도": ("temperature",),
    "소비 전력": ("power|current",),
    "소비 전류": ("current",),
    "대기 전류": ("sleep|standby|quiescent", "current"),
    "슬립 전류": ("sleep", "current"),
    "정지 전류": ("quiescent|iq",),
    "입력 전압": ("input", "voltage"),
    "출력 전압": ("output", "voltage"),
    "출력 전류": ("output", "current"),
    "부하 전류": ("load", "current"),
    "전류 제한": ("current", "limit|limiting"),
    "스위칭 주파수": ("switching|fsw", "frequency|khz|mhz"),
    "역전압": ("reverse", "voltage"),
    "역전류": ("reverse", "current"),
    "열 차단": ("thermal", "shutdown"),
    "소프트 스타트": ("soft", "start"),
    "드롭아웃": ("dropout",),
    "온저항": ("resistance|rds|ron",),
    "인에이블": ("enable",),
    "극성": ("active|polarity",),
    "효율": ("efficiency",),
    "노이즈": ("noise",),
    "피드백": ("feedback|fb",),
    "출력 커패시터": ("output", "capacitor|capacitance"),
    "커패시터": ("capacitor|capacitance",),
    "저전압": ("low|lvr|undervoltage",),
    "과전류": ("overcurrent",),
    "펄스 폭": ("pulse", "width"),
    "파워온": ("power|por",),
    "전원 인가": ("power|por",),
    "자체 전원": ("self", "powered|power"),
    "버스 전원": ("bus", "powered|power"),
    "신규 설계": ("new", "designs|design"),
    "수정 발진자": ("crystal",),
    "크리스탈": ("crystal",),
    "발진자": ("crystal|oscillator",),
    "고속": ("high", "speed"),
    "전속": ("full", "speed"),
    "저속": ("low", "speed"),
    "다운스트림": ("downstream",),
    "업스트림": ("upstream",),
    "풀업": ("pull", "up"),
    "풀다운": ("pull", "down"),
    "라우팅": ("routing",),
    "정전기": ("esd",),
    "임계값": ("threshold",),
    "임계": ("threshold",),
    "검출": ("detection",),
    "패키지": ("package",),
    "배치": ("pin",),
    "핀": ("pin",),
    "포트": ("port|ports",),
    "개수": ("number",),
    "전압": ("voltage",),
    "전류": ("current",),
    "온도": ("temperature",),
    "범위": ("range",),
    "리셋": ("reset",),
    "지연": ("delay",),
    "클럭": ("clock",),
    "주파수": ("frequency|khz|mhz",),
    "저항": ("resistor|resistance",),
    "부하": ("load",),
    "슬립": ("sleep",),
    "기본값": ("default",),
    "기본": ("default",),
    "권장": ("recommended",),
    "내장": ("built|internal",),
    "내부": ("internal|built",),
    # TI 데이터시트 등록 때 보강 (2026.10.26): 첫 측정에서 "최대 연속 전류" 가 막혀 추가
    "최대": ("maximum|max",),
    "연속": ("continuous",),
    "가격": ("price",),
    "단가": ("price",),
    "분해능": ("resolution",),
    "수명": ("lifetime",),
}

_KEYS = sorted(GLOSSARY, key=len, reverse=True)

# 판정 용어에서 뺄 영문 단어. 질문 형식에서 오는 말이라 근거 유무와 관계없다.
STOPWORDS = frozenset(
    """
    a an the is are was were be of for to in on at by with and or what which how
    does do did can could value values about this that it its from as into chip
    """.split()
)

_ENGLISH_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")


@dataclass(frozen=True)
class Concept:
    """판정 단위 하나.

    groups 의 묶음이 모두 근거에 있어야 '찾음' 이다. 묶음 안에서는 하나만 있으면 된다.
    groups 를 주지 않으면 alternatives 전체가 한 묶음이다 (동의어 중 하나).
    """

    label: str
    alternatives: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...] | None = None

    @property
    def required(self) -> tuple[tuple[str, ...], ...]:
        return self.groups if self.groups else (self.alternatives,)


@dataclass(frozen=True)
class Expansion:
    lexical_query: str
    concepts: tuple[Concept, ...]


def _concept(label: str, spec: tuple[str, ...]) -> Concept:
    groups = tuple(tuple(group.split("|")) for group in spec)
    flat = tuple(word for group in groups for word in group)
    return Concept(label=label, alternatives=flat, groups=groups)


def expand(question: str, part_tokens: set[str] | None = None) -> Expansion:
    """질문을 BM25 질의와 판정 용어 목록으로 바꾼다.

    part_tokens 는 품번 문자열이다. 품번은 변종 게이트가 따로 보므로 판정 용어에서 뺀다.
    """
    part_tokens = {token.lower() for token in (part_tokens or set())}
    concepts: list[Concept] = []
    remaining = question
    for key in _KEYS:
        if key in remaining:
            concepts.append(_concept(key, GLOSSARY[key]))
            remaining = remaining.replace(key, " ")

    seen = {concept.label for concept in concepts}
    for word in _ENGLISH_WORD.findall(question):
        lowered = word.lower()
        if lowered in STOPWORDS or lowered in part_tokens or len(lowered) < 2:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        concepts.append(Concept(label=word, alternatives=(lowered,)))

    added = " ".join(" ".join(c.alternatives) for c in concepts if c.label in GLOSSARY)
    lexical_query = f"{question} {added}".strip()
    return Expansion(lexical_query=lexical_query, concepts=tuple(concepts))
