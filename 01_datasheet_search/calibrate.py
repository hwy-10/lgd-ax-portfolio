"""근거 충분성 기준값(min_score, min_coverage)을 평가셋으로 정한다.

모델을 부르지 않는다. 검색 단계까지만 돌려 질문마다 두 값을 잰다.
  top_score   BM25 1위 점수
  coverage    질문 용어가 근거에 등장한 비율

정답이 있는 질문만 넣으면 기준을 낮출수록 점수가 좋아진다. 그래서 문서에 없는
질문을 같이 넣고, 통과율과 차단율을 함께 본다.

  python calibrate.py            결과를 results/ 에 저장
"""

from __future__ import annotations

import csv
import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import EVALSET_PATH, RESULTS_DIR
from src.pipeline import DatasheetSearch

SCORE_GRID = [x / 2 for x in range(0, 31)]          # 0.0 ~ 15.0
COVERAGE_GRID = [0.0, 0.2, 0.25, 0.34, 0.4, 0.5, 0.6, 0.67, 0.75, 0.8, 1.0]


def measure() -> list[dict]:
    # 기준 0 으로 두고 재기만 한다 (게이트와 검증 게이트 차단은 기준과 무관하게 남는다)
    search = DatasheetSearch(min_score=0.0, min_coverage=0.0)
    rows = []
    for item in json.loads(EVALSET_PATH.read_text(encoding="utf-8")):
        trace = search.retrieve(item["question"])
        rows.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "top_score": round(trace.sufficiency.top_score, 3),
                "coverage": round(trace.sufficiency.coverage, 3),
                "gate_blocked": trace.verification_failed or not trace.evidence,
                "missing": list(trace.sufficiency.missing),
            }
        )
    return rows


def passes(row: dict, min_score: float, min_coverage: float) -> bool:
    if row["gate_blocked"]:
        return False
    return row["top_score"] >= min_score and row["coverage"] >= min_coverage


def sweep(rows: list[dict]) -> list[dict]:
    answerable = [r for r in rows if r["type"] != "not_in_doc"]
    absent = [r for r in rows if r["type"] == "not_in_doc"]
    table = []
    for c in COVERAGE_GRID:
        for s in SCORE_GRID:
            passed = sum(passes(r, s, c) for r in answerable)
            blocked = sum(not passes(r, s, c) for r in absent)
            table.append(
                {
                    "min_score": s,
                    "min_coverage": c,
                    "answerable_pass": passed,
                    "answerable_total": len(answerable),
                    "absent_block": blocked,
                    "absent_total": len(absent),
                    "correct": passed + blocked,
                }
            )
    return table


def choose(rows: list[dict], table: list[dict]) -> dict:
    """최적 조합 중 양쪽 여유가 가장 큰 값을 고른다.

    용어 비율  문서에 없는 질문 중 가장 높은 값과 정답 질문 중 가장 낮은 값의 가운데
    점수      정답 질문 중 가장 낮은 1위 점수의 절반 (점수만으로는 둘을 가를 수 없어 보조 기준)
    """
    answerable = [r for r in rows if r["type"] != "not_in_doc" and not r["gate_blocked"]]
    absent = [r for r in rows if r["type"] == "not_in_doc" and not r["gate_blocked"]]
    low_cov = min(r["coverage"] for r in answerable)
    high_cov_absent = max((r["coverage"] for r in absent), default=0.0)
    low_score = min(r["top_score"] for r in answerable)
    high_score_absent = max((r["top_score"] for r in absent), default=0.0)

    target_c = (low_cov + high_cov_absent) / 2
    target_s = low_score / 2
    best_correct = max(t["correct"] for t in table)
    candidates = [t for t in table if t["correct"] == best_correct]
    chosen = min(
        candidates,
        key=lambda t: (abs(t["min_coverage"] - target_c), abs(t["min_score"] - target_s)),
    )
    return {
        **chosen,
        "answerable_min_coverage": low_cov,
        "absent_max_coverage": high_cov_absent,
        "answerable_min_score": low_score,
        "absent_max_score": high_score_absent,
        "score_only_best": max(
            (t for t in table if t["min_coverage"] == 0.0), key=lambda t: t["correct"]
        ),
    }


def main() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = measure()
    table = sweep(rows)
    chosen = choose(rows, table)

    with open(RESULTS_DIR / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (RESULTS_DIR / "threshold_measure.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (RESULTS_DIR / "threshold.json").write_text(
        json.dumps(chosen, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    a = chosen
    print(f"정답 질문의 용어 비율 최저 {a['answerable_min_coverage']:.2f}, "
          f"문서에 없는 질문의 최고 {a['absent_max_coverage']:.2f}")
    print(f"정답 질문의 1위 점수 최저 {a['answerable_min_score']:.1f}, "
          f"문서에 없는 질문의 최고 {a['absent_max_score']:.1f}")
    so = a["score_only_best"]
    print(f"\n점수 기준만 쓸 때 최선  min_score={so['min_score']}  "
          f"통과 {so['answerable_pass']}/{so['answerable_total']}, "
          f"차단 {so['absent_block']}/{so['absent_total']}")
    print(f"선택  min_score={a['min_score']}  min_coverage={a['min_coverage']}  "
          f"통과 {a['answerable_pass']}/{a['answerable_total']}, "
          f"차단 {a['absent_block']}/{a['absent_total']}")
    print("\nconfig.py 의 EVIDENCE_MIN_SCORE, EVIDENCE_MIN_COVERAGE 를 이 값으로 둔다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
