"""프로젝트 부품 전체로 넓힌 검색. 문서를 먼저 고르고, 고른 문서 한 권 안에서 찾는다.

  질문 ─▶ ⓪ 문서 선택 ─▶ (문서 한 권) ① 품번 ② 변종 게이트 ③ BM25 ④ 검증 게이트 ⑤ 근거 충분성 ─▶ ⑥ 모델

⓪ 문서 선택이 하는 일
  - 질문의 품번이나 회로 기호(U5)를 등록 문서의 품번 규칙에 맞춰 본다
  - 회로 기호는 BOM 으로 품번을 찾아 바꾼다 (U5 → LMR33630ADDAR → LMR33630A)
  - 한 문서로 정해지지 않으면 답하지 않는다
      두 부품 이상     비교 질문은 지원하지 않음
      BOM 에는 있음    데이터시트가 등록되지 않은 부품 (ESD, 다이오드, 수동 소자 등)
      BOM 에도 없음    이 프로젝트의 부품이 아님
      부품 없음        어느 문서를 볼지 정할 수 없음

여러 PDF 를 한 BM25 색인에 섞지 않는다. 섞으면 IDF 와 점수 분포가 바뀌어 기준값이
의미를 잃고, 다른 부품의 공통 청크가 근거로 섞인다. 문서마다 색인을 따로 둔다.

품번 규칙은 제조사마다 다르다. WCH 는 접미사 한 글자가 변종이고, TI 는 숫자 끝자리,
-1 같은 꼬리, A/B/C 같은 문자가 변종이며 DBV, DDA 같은 꼬리는 패키지다. 그래서 정규식
하나로 추측하지 않고 문서마다 변종 규칙을 등록한다 (data/catalog.json).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from config import BASE_DIR, EVIDENCE_MIN_COVERAGE, EVIDENCE_MIN_SCORE, TOP_K
from src.hybrid import HybridIndex
from src.llm import complete
from src.pipeline import DatasheetSearch, Trace, build_index_from_pdf
from src.prompt import NOT_FOUND_ANSWER
from src.variants import PartNumber

CATALOG_PATH = BASE_DIR / "data" / "catalog.json"

# BOM 표의 회로 기호 (U5, C28, PORT1, TP6 ...)
_REFDES = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,4}\d{1,3})(?![A-Za-z0-9])")
# 품번처럼 생긴 문자열 (영문 대문자 + 숫자로 시작, 하이픈 허용)
_PART_LIKE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,}[0-9][A-Z0-9]*(?:-[A-Z0-9]+)*)")
_STRONG_PART = re.compile(r"[A-Z]{3,}\d{3,}")
# BOM 값 칸에서 품번을 찾는다 (USBLC6-4SC6-ES, BAT54W, PJ-102AH). "12V", "0.1uF" 같은 값은 품번이 아니다
BOM_PART_NUMBER = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,}[A-Z0-9-]*\d[A-Z0-9-]*)")


@dataclass(frozen=True)
class RegisteredPart(PartNumber):
    """등록 규칙으로 찾은 변종. 접미사가 없어도 특정 변종이다 (TPS2552 와 TPS2553 은 다른 부품)."""

    @property
    def is_specific(self) -> bool:
        return True


@dataclass(frozen=True)
class DocEntry:
    id: str
    file: str
    title: str
    manufacturer: str
    role: str
    variant_patterns: tuple[re.Pattern, ...]
    family_patterns: tuple[re.Pattern, ...]
    note: str = ""

    @property
    def path(self) -> Path:
        return BASE_DIR / self.file


class ManifestRules:
    """문서 하나에 등록된 품번 규칙. src.variants.VariantRules 를 구현한다."""

    def __init__(self, doc: DocEntry) -> None:
        self.doc = doc

    def _variant_matches(self, text: str) -> list[tuple[int, str, str]]:
        out = []
        for pattern in self.doc.variant_patterns:
            for m in pattern.finditer(text):
                out.append((m.start(), m.group(1).replace("–", "-"), m.group(0)))
        return sorted(out)

    def variants_in(self, text: str) -> set[str]:
        return {name for _, name, _ in self._variant_matches(text)}

    def families_in(self, text: str) -> list[str]:
        return [m.group(0) for p in self.doc.family_patterns for m in p.finditer(text)]

    def primary(self, question: str) -> PartNumber | None:
        found = self._variant_matches(question)
        if found:
            name = found[0][1]
            return RegisteredPart(raw=name, family=name, suffix="")
        families = self.families_in(question)
        if families:
            return PartNumber(raw=families[0], family=families[0], suffix="")
        return None

    def specific_targets(self, question: str) -> list[PartNumber]:
        names = list(dict.fromkeys(name for _, name, _ in self._variant_matches(question)))
        return [RegisteredPart(raw=n, family=n, suffix="") for n in names]

    def part_strings(self, question: str) -> set[str]:
        strings = {raw for _, _, raw in self._variant_matches(question)}
        strings.update(self.families_in(question))
        # 주문 품번 꼬리(DBVR, DDAR ...)까지 순위 질의에서 뺀다
        for token in _PART_LIKE.findall(question):
            if self.variants_in(token) or self.families_in(token):
                strings.add(token)
        return strings

    def applies(self, chunk_text: str, target: PartNumber) -> bool:
        if not target.is_specific:
            return True
        found = self.variants_in(chunk_text)
        return not found or target.raw in found

    def names(self, chunk_text: str, target: PartNumber) -> bool:
        return target.raw in self.variants_in(chunk_text)

    def matches(self, text: str) -> bool:
        return bool(self.variants_in(text) or self.families_in(text))


@dataclass(frozen=True)
class BomEntry:
    refs: tuple[str, ...]
    value: str
    footprint: str
    module: str

    @property
    def part(self) -> str:
        """값 칸의 첫 단어 (품번 또는 값)."""
        return self.value.split()[0] if self.value else ""

    @property
    def part_numbers(self) -> list[str]:
        return [p.rstrip("-") for p in BOM_PART_NUMBER.findall(self.value)]


def _expand_refs(cell: str) -> tuple[str, ...]:
    """`C1`~`C4`, `R7`,`R9` 같은 칸을 회로 기호 목록으로 푼다."""
    refs: list[str] = []
    for part in cell.split(","):
        names = re.findall(r"`([A-Za-z]+)(\d+)`", part)
        if len(names) == 2 and "~" in part and names[0][0] == names[1][0]:
            prefix, start, end = names[0][0], int(names[0][1]), int(names[1][1])
            refs.extend(f"{prefix}{n}" for n in range(start, end + 1))
        else:
            refs.extend(f"{p}{n}" for p, n in names)
    return tuple(refs)


def load_bom(path: Path) -> list[BomEntry]:
    """BOM 원가 문서의 표에서 회로 기호와 값을 읽는다. 사람이 다시 옮겨 적지 않는다."""
    entries: list[BomEntry] = []
    module = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^#+\s*(.+)", line)
        if heading:
            module = heading.group(1).strip()
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        refs = _expand_refs(cells[0])
        value = re.sub(r"\*\*", "", cells[1]).strip()
        if refs and value:
            entries.append(BomEntry(refs=refs, value=value, footprint=cells[2].strip("`"), module=module))
    return entries


@dataclass
class Route:
    kind: str                       # doc | off | unregistered | unknown | none | multi
    doc: DocEntry | None = None
    question: str = ""              # 문서 검색에 넘길 질문 (회로 기호를 품번으로 바꾼 것)
    refdes: list[tuple[str, str]] = field(default_factory=list)   # (U5, LMR33630ADDAR)
    targets: list[str] = field(default_factory=list)
    message: str = ""


@dataclass
class CatalogTrace:
    route: Route
    trace: Trace | None = None
    answer: str = ""

    @property
    def blocked(self) -> bool:
        return self.route.kind != "doc" or self.trace is None or self.trace.blocked


class Catalog:
    def __init__(self, path: Path = CATALOG_PATH) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.project = data.get("project", "")
        self.docs = [
            DocEntry(
                id=d["id"],
                file=d["file"],
                title=d["title"],
                manufacturer=d["manufacturer"],
                role=d.get("role", ""),
                variant_patterns=tuple(re.compile(p) for p in d["variant_patterns"]),
                family_patterns=tuple(re.compile(p) for p in d.get("family_patterns", [])),
                note=d.get("note", ""),
            )
            for d in data["documents"]
        ]
        self.rules = {d.id: ManifestRules(d) for d in self.docs}
        bom_source = data.get("bom_source")
        self.bom = load_bom(BASE_DIR / bom_source) if bom_source else []
        self.by_ref = {ref: entry for entry in self.bom for ref in entry.refs}

    # ---------- BOM 과 문서 대응 ----------
    def doc_for_text(self, text: str) -> DocEntry | None:
        hits = [d for d in self.docs if self.rules[d.id].matches(text)]
        return hits[0] if len(hits) == 1 else None

    def coverage(self) -> list[tuple[BomEntry, DocEntry | None, str]]:
        """BOM 항목마다 등록 문서와 변종. 문서 대응은 품번 규칙이 정한다 (손으로 잇지 않는다)."""
        rows = []
        for entry in self.bom:
            doc = self.doc_for_text(entry.value)
            variant = ""
            if doc:
                target = self.rules[doc.id].primary(entry.value)
                variant = target.raw if target else ""
            rows.append((entry, doc, variant))
        return rows

    def bom_part_for_doc(self, doc_id: str) -> BomEntry | None:
        """이 문서에 대응하는 BOM 항목 (실제로 보드에 쓴 주문 품번)."""
        for entry, doc, _ in self.coverage():
            if doc and doc.id == doc_id:
                return entry
        return None

    # ---------- 문서 선택 ----------
    def route(self, question: str, enabled: set[str] | None = None) -> Route:
        """enabled 는 켜 둔 문서 id. None 이면 전부 켠 것과 같다.

        켜 둔 문서는 검색 범위를 정할 뿐, 여러 권을 한 색인으로 합치지 않는다.
          품번이 있는 질문      그 부품 문서가 켜져 있으면 그 한 권, 꺼져 있으면 답하지 않음
          품번이 없는 질문      켜진 문서가 한 권이면 그 한 권 (변종은 BOM 에 쓴 품번 기준)
                               두 권 이상이면 어느 문서인지 정할 수 없어 답하지 않음
        """
        active = [d for d in self.docs if enabled is None or d.id in enabled]
        rewritten = question
        refdes: list[tuple[str, str]] = []
        unregistered: list[str] = []
        for ref in dict.fromkeys(_REFDES.findall(question)):
            entry = self.by_ref.get(ref)
            if not entry:
                continue
            refdes.append((ref, entry.value))
            if self.doc_for_text(entry.value):
                # 회로 기호는 품번으로 바꾼다. 기호를 남기면 근거에서 찾을 용어로 잡혀 판정을 흐린다
                rewritten = re.sub(rf"(?<![A-Za-z0-9]){ref}(?![A-Za-z0-9])", entry.part, rewritten)
            else:
                unregistered.append(f"{ref} = {entry.value}")

        docs = [d for d in self.docs if self.rules[d.id].matches(rewritten)]
        # BOM 에 있지만 문서가 없는 품번 (USBLC6-4SC6-ES, BAT54W ...)
        lowered = rewritten.lower()
        for entry in self.bom:
            if self.doc_for_text(entry.value):
                continue
            for number in entry.part_numbers:
                if re.search(rf"(?<![a-z0-9]){re.escape(number.lower())}(?![a-z0-9])", lowered) \
                        and number not in " ".join(unregistered):
                    unregistered.append(f"{number} ({', '.join(entry.refs)})")
        unknown = [
            token for token in _PART_LIKE.findall(rewritten)
            if not any(self.rules[d.id].matches(token) for d in self.docs)
            and not any(token.lower() in u.lower() for u in unregistered)
            and token not in self.by_ref
        ]
        if docs:
            # 문서가 정해졌으면 핀, 네트 이름(VDD33, VIN)은 다른 부품으로 보지 않는다.
            # 영문 3자 이상 + 숫자 3자 이상처럼 품번 모양이 뚜렷한 것만 두 번째 부품으로 센다
            unknown = [t for t in unknown if _STRONG_PART.match(t)]

        targets = [d.title for d in docs] + unregistered + unknown
        if len(targets) > 1:
            return Route("multi", question=rewritten, refdes=refdes, targets=targets,
                         message=f"부품 둘 이상({', '.join(targets)})을 함께 묻는 질문은 지원하지 않음. 한 부품씩 물어 주세요")
        if docs:
            doc = docs[0]
            if enabled is not None and doc.id not in enabled:
                return Route("off", doc=doc, question=rewritten, refdes=refdes, targets=targets,
                             message=f"{doc.title} 문서가 꺼져 있어 검색하지 않음. 켜면 이 문서에서 찾는다")
            return Route("doc", doc=doc, question=rewritten, refdes=refdes, targets=targets,
                         message=f"{doc.title} ({doc.role})")
        if unregistered:
            return Route("unregistered", question=rewritten, refdes=refdes, targets=targets,
                         message=f"{unregistered[0]}: BOM 에는 있지만 데이터시트가 등록되지 않은 부품")
        if unknown:
            return Route("unknown", question=rewritten, refdes=refdes, targets=targets,
                         message=f"{unknown[0]}: 이 프로젝트 BOM 에 없는 품번")
        if len(active) == 1:
            doc = active[0]
            entry = self.bom_part_for_doc(doc.id)
            # 품번이 없으면 보드에 실제로 쓴 품번을 붙여 변종을 정한다 (TPS2552 와 TPS2553 은 EN 극성이 다르다)
            fixed = f"{entry.part} {rewritten}" if entry else rewritten
            basis = f" 변종은 BOM 값 {entry.part}({', '.join(entry.refs)}) 기준" if entry else ""
            return Route("doc", doc=doc, question=fixed, refdes=refdes,
                         message=f"{doc.title} ({doc.role}). 켜진 문서가 하나라 이 문서에서 찾음.{basis}")
        if not active:
            return Route("none", question=rewritten, refdes=refdes, message="켜진 문서가 없음")
        return Route("none", question=rewritten, refdes=refdes,
                     message=f"부품이 지정되지 않아 문서를 고를 수 없음. 품번을 쓰거나 문서를 하나만 켜 주세요 (켜진 문서 {len(active)}권)")


@lru_cache(maxsize=None)
def _doc_index(path: str, doc_id: str) -> HybridIndex:
    catalog = default_catalog()
    doc = next(d for d in catalog.docs if d.id == doc_id)
    base = build_index_from_pdf(Path(path))
    return HybridIndex(base.documents, rules=catalog.rules[doc.id])


@lru_cache(maxsize=1)
def default_catalog() -> Catalog:
    return Catalog()


class CatalogSearch:
    def __init__(
        self,
        catalog: Catalog | None = None,
        min_score: float = EVIDENCE_MIN_SCORE,
        min_coverage: float = EVIDENCE_MIN_COVERAGE,
        top_k: int = TOP_K,
    ) -> None:
        self.catalog = catalog or default_catalog()
        self.min_score = min_score
        self.min_coverage = min_coverage
        self.top_k = top_k

    def searcher(self, doc: DocEntry) -> DatasheetSearch:
        return DatasheetSearch(
            _doc_index(str(doc.path), doc.id),
            min_score=self.min_score,
            min_coverage=self.min_coverage,
            top_k=self.top_k,
            document=f"{doc.manufacturer} {doc.title}",
        )

    def retrieve(self, question: str, enabled: set[str] | None = None) -> CatalogTrace:
        if not question.strip():
            raise ValueError("질문이 비어 있습니다")
        route = self.catalog.route(question, enabled)
        if route.kind != "doc":
            return CatalogTrace(route=route, answer=NOT_FOUND_ANSWER)
        trace = self.searcher(route.doc).retrieve(route.question)
        return CatalogTrace(route=route, trace=trace)

    def answer(self, question: str, model: str, generate=complete, enabled: set[str] | None = None) -> CatalogTrace:
        route = self.catalog.route(question, enabled)
        if route.kind != "doc":
            return CatalogTrace(route=route, answer=NOT_FOUND_ANSWER)
        trace = self.searcher(route.doc).answer(route.question, model, generate=generate)
        return CatalogTrace(route=route, trace=trace, answer=trace.answer)
