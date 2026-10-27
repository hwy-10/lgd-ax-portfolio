"""발표 자료에 들어가는 숫자를 코드로 다시 계산해 results/summary.json 에 쓴다.

자료의 숫자를 손으로 옮기지 않기 위해서다. API 키가 필요 없다.

  python tools/export_summary.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import EVALSET_PATH, EVIDENCE_MIN_COVERAGE, EVIDENCE_MIN_SCORE, RESULTS_DIR  # noqa: E402
from evaluate import check_evidence_gate, check_gate, check_worst_case, specific_variants  # noqa: E402
from src.pipeline import DatasheetSearch, build_index_from_pdf  # noqa: E402
from src.variants import parse_part  # noqa: E402

FAMILY = ("CH334", "CH335")


def main() -> int:
    index = build_index_from_pdf()
    counter: Counter[str] = Counter()
    for document in index.documents:
        counter.update(specific_variants(document.text))
    variants = [v for v, _ in counter.most_common()]

    with contextlib.redirect_stdout(io.StringIO()):
        gate_violations = check_gate(index, variants)
        worst_violations = check_worst_case(index, variants)
        evidence_ok = check_evidence_gate(index)
    checked = sum(len(index.gate(parse_part(v))) for v in variants)

    search = DatasheetSearch(index)
    items = json.loads(EVALSET_PATH.read_text(encoding="utf-8"))
    answerable = [i for i in items if i["type"] != "not_in_doc"]
    absent = [i for i in items if i["type"] == "not_in_doc"]
    hit = passed = blocked = 0
    for item in answerable:
        trace = search.retrieve(item["question"])
        pages = {e.page for e in trace.evidence}
        hit += bool(pages & set(item["expected_pages"]))
        passed += not trace.blocked
    for item in absent:
        blocked += search.retrieve(item["question"]).blocked

    summary = {
        "chunks": len(index.documents),
        "variants_total": len(variants),
        "variants_family": sorted(v for v in variants if v.startswith(FAMILY)),
        "variants_other": sorted(v for v in variants if not v.startswith(FAMILY)),
        "gate_checked": checked,
        "gate_violations": gate_violations,
        "worst_case_violations": worst_violations,
        "unknown_part_empty": evidence_ok,
        "evalset_total": len(items),
        "evalset_answerable": len(answerable),
        "evalset_variant": sum(1 for i in items if i["type"] == "variant"),
        "evalset_absent": len(absent),
        "hit_at_5": hit,
        "answerable_passed": passed,
        "absent_blocked": blocked,
        "min_score": EVIDENCE_MIN_SCORE,
        "min_coverage": EVIDENCE_MIN_COVERAGE,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
