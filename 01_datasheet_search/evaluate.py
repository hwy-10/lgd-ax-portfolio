"""변종 게이트를 실제 데이터시트로 검증한다.

CH334_CH335_V2.5.pdf 한 권에 CH334U · CH334R 등 여러 변종이 함께 들어 있다.
품번은 한 글자만 달라도 다른 부품이므로, 다른 변종 전용 청크가 근거로 오면 안 된다.

임베딩 API 를 호출하지 않는다. PDF -> 청크 -> BM25 만으로 돌아가므로
Vector DB 가 없어도, API 키가 없어도 실행된다.
"""

from __future__ import annotations

import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.hybrid import HybridIndex
from src.pipeline import build_index_from_pdf
from src.variants import extract_parts, parse_part

TOP_K = 5


def specific_variants(text: str) -> set[str]:
    return {part.raw for part in extract_parts(text) if part.is_specific}


def build_index() -> HybridIndex:
    return build_index_from_pdf()


def report_distribution(index: HybridIndex) -> list[str]:
    counter: Counter[str] = Counter()
    unspecified = 0
    for document in index.documents:
        variants = specific_variants(document.text)
        if not variants:
            unspecified += 1
        counter.update(variants)

    print(f"[1] 적재  청크 {len(index.documents)}개")
    print(f"    변종을 특정한 청크 {len(index.documents) - unspecified}개 / "
          f"미특정 {unspecified}개")
    print("    " + ", ".join(f"{v}({c})" for v, c in counter.most_common(8)))
    return [v for v, _ in counter.most_common()]


def check_gate(index: HybridIndex, variants: list[str]) -> int:
    """변종 x 청크 전수 확인 — 다른 변종 전용 청크가 통과하면 위반."""
    checked = violations = 0
    for raw in variants:
        target = parse_part(raw)
        if target is None:
            continue
        for position in index.gate(target):
            checked += 1
            found = specific_variants(index.documents[position].text)
            if found and raw not in found:
                violations += 1
    print(f"\n[2] 변종 게이트 전수 확인  {len(variants)}개 변종 x 청크 = {checked:,}건")
    print(f"    다른 변종 전용 청크의 오통과 : {violations}건")
    return violations


def check_worst_case(index: HybridIndex, variants: list[str]) -> int:
    """임베딩이 다른 변종 청크를 1위로 올린 최악의 순위를 주입해도 막히는가."""
    violations = 0
    for raw in variants:
        target = parse_part(raw)
        if target is None:
            continue
        # 다른 변종만 명시한 청크를 앞세운 적대적 dense 순위
        adversarial = [
            i
            for i, d in enumerate(index.documents)
            if (fv := specific_variants(d.text)) and raw not in fv
        ]
        if not adversarial:
            continue
        ranking = adversarial + [
            i for i in range(len(index.documents)) if i not in set(adversarial)
        ]
        for position in index.search(f"{raw} 사양", dense_ranking=ranking, top_k=TOP_K):
            found = specific_variants(index.documents[position].text)
            if found and raw not in found:
                violations += 1
    print(f"\n[3] 최악 조건  의미 검색이 다른 변종을 1위로 올린 순위를 주입")
    print(f"    결과에 섞인 다른 변종 청크 : {violations}건")
    return violations


def check_evidence_gate(index: HybridIndex) -> bool:
    """문서에 없는 품번을 물으면 답을 만들지 않아야 한다."""
    empty = index.search("CH999Z 동작 전압", top_k=TOP_K)
    print(f"\n[4] 검증 게이트  문서에 없는 CH999Z 질의 -> 결과 {len(empty)}건 "
          f"({'빈 결과 정상' if not empty else '비정상'})")
    return not empty


def main() -> int:
    index = build_index()
    variants = report_distribution(index)
    failures = check_gate(index, variants)
    failures += check_worst_case(index, variants)
    if not check_evidence_gate(index):
        failures += 1
    print("\n" + ("전부 통과" if failures == 0 else f"위반 {failures}건"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
