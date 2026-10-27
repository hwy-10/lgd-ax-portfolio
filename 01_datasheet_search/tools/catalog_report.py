"""BOM 의 각 부품이 어느 데이터시트로 답할 수 있는지 보고서를 만든다.

  python tools/catalog_report.py        → results/catalog_coverage.md

BOM 은 data/modulino_BOM_cost_estimate.md 의 표에서 읽고, 문서 대응은 data/catalog.json 의
품번 규칙이 정한다. 사람이 BOM 과 문서를 손으로 잇지 않는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import RESULTS_DIR  # noqa: E402
from src.catalog import Catalog  # noqa: E402


def main() -> int:
    catalog = Catalog()
    registered, numbered, generic = [], [], []
    for entry, doc, variant in catalog.coverage():
        refs = ", ".join(entry.refs)
        if doc:
            registered.append((refs, entry.value, doc, variant, len(entry.refs)))
        elif entry.part_numbers:
            numbered.append((refs, entry.value, len(entry.refs)))
        else:
            generic.append((refs, entry.value, len(entry.refs)))

    total = sum(len(e.refs) for e in catalog.bom)
    n_reg = sum(r[4] for r in registered)
    n_num = sum(r[2] for r in numbered)
    n_gen = sum(r[2] for r in generic)
    lines = [
        f"# BOM 데이터시트 등록 현황 — {catalog.project}",
        "",
        f"BOM 부품 {total}개 (행 {len(catalog.bom)}개). 등록 문서 {len(catalog.docs)}권.",
        "",
        "| 구분 | 부품 수 | 질문하면 |",
        "|---|---|---|",
        f"| 데이터시트 등록 | {n_reg} | 해당 문서 한 권에서 검색 |",
        f"| 품번은 있으나 데이터시트 미등록 | {n_num} | “등록되지 않은 부품” 으로 답하지 않음 |",
        f"| 수동 소자, 기구 (값이 정본) | {n_gen} | BOM 값을 보여 주고 답하지 않음 |",
        "",
        "## 데이터시트 등록",
        "",
        "| 회로 기호 | BOM 값 | 문서 | 변종 |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r} | {v} | {d.manufacturer} {d.title} | {var} |" for r, v, d, var, _ in registered]
    lines += ["", "## 품번은 있으나 데이터시트 미등록 (다음 등록 대상)", "", "| 회로 기호 | BOM 값 |", "|---|---|"]
    lines += [f"| {r} | {v} |" for r, v, _ in numbered]
    out = RESULTS_DIR / "catalog_coverage.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS_DIR / "catalog_coverage.json").write_text(json.dumps({
        "bom_refs": total, "bom_rows": len(catalog.bom), "documents": len(catalog.docs),
        "registered": n_reg, "numbered_unregistered": n_num, "generic": n_gen,
        "registered_refs": [r for r, *_ in registered],
        "unregistered_parts": [v for _, v, _ in numbered],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines[:10]))
    print(f"\n{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
