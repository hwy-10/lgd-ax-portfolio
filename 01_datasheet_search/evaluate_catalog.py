"""프로젝트 부품 전체(문서 5권)로 넓힌 검색을 평가한다.

기준값(1위 점수 2.5, 용어 비율 0.67)은 CH334 한 권의 평가셋으로 정한 값을 그대로 쓴다.
나머지 4권(TI)은 기준값을 정할 때 쓰지 않은 문서라, 다시 맞추지 않고 그대로 적용한 결과다.

  python evaluate_catalog.py                    검색 단계까지 (모델 없이)
  python evaluate_catalog.py --verify           정답 사실이 원문 어느 쪽에 있는지 확인
  python evaluate_catalog.py --model ollama:gemma3:4b    답변까지

질문마다 보는 것
  문서 선택    기대한 문서(또는 기대한 차단 사유)로 갔는가
  변종         질문한 변종으로 읽었는가 (U5 → LMR33630A)
  통과/차단    답이 있는 질문은 통과, 없는 질문은 모델 호출 전에 차단
  근거 hit     정답 사실이 상위 5개 근거 원문 안에 실제로 있는가 (페이지가 아니라 문장으로 본다)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import BASE_DIR, EVIDENCE_MIN_COVERAGE, EVIDENCE_MIN_SCORE, RESULTS_DIR
from src.answer_check import is_refusal
from src.catalog import CatalogSearch, default_catalog
from src.ingest import extract_pages

QUESTIONS = BASE_DIR / "tests" / "catalog_questions.json"


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text).lower().replace("–", "-").replace("−", "-")


def has_all(text: str, groups: list[list[str]]) -> bool:
    body = norm(text)
    return all(any(norm(o) in body for o in group) for group in groups)


def grade_answer(answer: str, item: dict) -> bool:
    body = norm(answer)
    if is_refusal(answer):
        return False
    ok = all(any(norm(o) in body for o in g) for g in item.get("answer_must", []))
    return ok and not any(norm(b) in body for b in item.get("answer_must_not", []))


def verify(items: list[dict]) -> int:
    """정답 사실이 해당 문서 원문에 실제로 있는지, 어느 쪽인지 출력한다."""
    catalog = default_catalog()
    docs = {d.id: d for d in catalog.docs}
    pages_cache: dict[str, list] = {}
    problems = 0
    for item in items:
        if "evidence_must" not in item:
            continue
        doc = docs[item["doc"]]
        pages = pages_cache.setdefault(doc.id, extract_pages(doc.path))
        hits = []
        for group in item["evidence_must"]:
            found = [p.page for p in pages if any(norm(o) in norm(p.text) for o in group)]
            hits.append((group[0], found[:6]))
        ok = all(found for _, found in hits)
        problems += not ok
        print(f"{item['id']} {'OK' if ok else 'NG'} {doc.id}  " + "  ".join(f"{w}→p.{f}" for w, f in hits))
    print(f"\n확인 필요 {problems}건")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--model", default="")
    parser.add_argument("--tag", default="", help="결과 파일 이름 꼬리 (예: before)")
    parser.add_argument("--regrade", default="", help="저장된 결과 파일 꼬리. 모델을 부르지 않고 답변만 다시 채점")
    args = parser.parse_args()

    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if args.verify:
        return 1 if verify(items) else 0
    if args.regrade:
        path = RESULTS_DIR / f"catalog_eval_{args.regrade}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        by_id = {i["id"]: i for i in items}
        for row in data["rows"]:
            if "answer" in row:
                row["answer_ok"] = grade_answer(row["answer"], by_id[row["id"]])
        data["summary"]["answer_ok"] = sum(r.get("answer_ok", False) for r in data["rows"])
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"답변 정답 {data['summary']['answer_ok']}/{data['summary']['answer_total']} (다시 채점)")
        return 0

    search = CatalogSearch()
    rows = []
    for item in items:
        ct = search.answer(item["question"], args.model) if args.model else search.retrieve(item["question"])
        tr = ct.trace
        expected_kind = item.get("route", "doc")
        route_ok = ct.route.kind == expected_kind and (
            expected_kind != "doc" or (ct.route.doc and ct.route.doc.id == item["doc"])
        )
        row = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "route": ct.route.kind,
            "doc": ct.route.doc.id if ct.route.doc else None,
            "route_ok": route_ok,
            "message": ct.route.message,
            "blocked": ct.blocked,
            "reason": tr.sufficiency.reason if tr else ct.route.message,
            "target": tr.target if tr else None,
            "pages": [e.page for e in tr.evidence] if tr else [],
        }
        if item["type"] == "blocked":
            row["correct"] = route_ok and ct.blocked
        else:
            row["target_ok"] = item.get("target") in (None, row["target"])
            row["evidence_hit"] = bool(tr) and has_all(" ".join(e.text for e in tr.evidence), item["evidence_must"])
            row["correct"] = route_ok and not ct.blocked and row["target_ok"]
            if args.model and tr:
                row["answer"] = ct.answer
                row["answer_ok"] = grade_answer(ct.answer, item)
                row["check_passed"] = bool(tr.check and tr.check.passed)
        mark = "O" if row["correct"] else "X"
        extra = "" if item["type"] == "blocked" else f" hit={'O' if row['evidence_hit'] else 'X'}"
        if "answer_ok" in row:
            extra += f" 답={'O' if row['answer_ok'] else 'X'}"
        print(f"  {item['id']:<4}{mark} {row['route']:<12}{row['doc'] or '':<10}{extra}  {row['reason'][:60]}")
        rows.append(row)

    answerable = [r for r in rows if r["type"] != "blocked"]
    blocked = [r for r in rows if r["type"] == "blocked"]
    by_doc: dict[str, list] = defaultdict(list)
    for r in answerable:
        by_doc[r["doc"] or "?"].append(r)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "documents": len(default_catalog().docs),
        "min_score": EVIDENCE_MIN_SCORE,
        "min_coverage": EVIDENCE_MIN_COVERAGE,
        "questions": len(rows),
        "answerable": len(answerable),
        "answerable_passed": sum(r["correct"] for r in answerable),
        "evidence_hit": sum(r["evidence_hit"] for r in answerable),
        "blocked_total": len(blocked),
        "blocked_correct": sum(r["correct"] for r in blocked),
        "route_ok": sum(r["route_ok"] for r in rows),
        "by_doc": {
            doc: {"n": len(rs), "passed": sum(r["correct"] for r in rs), "hit": sum(r["evidence_hit"] for r in rs)}
            for doc, rs in sorted(by_doc.items())
        },
    }
    if args.model:
        called = [r for r in answerable if "answer_ok" in r]
        summary["model"] = args.model
        summary["answer_ok"] = sum(r["answer_ok"] for r in called)
        summary["answer_total"] = len(answerable)
    suffix = f"_{args.tag}" if args.tag else ""
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"catalog_eval{suffix}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    s = summary
    print(f"\n문서 {s['documents']}권, 질문 {s['questions']}개 (기준값 {s['min_score']} / {s['min_coverage']} 그대로)")
    print(f"문서 선택 맞음 {s['route_ok']}/{s['questions']}")
    print(f"답이 있는 질문 통과 {s['answerable_passed']}/{s['answerable']}, 근거 hit {s['evidence_hit']}/{s['answerable']}")
    print(f"답하지 말아야 할 질문 차단 {s['blocked_correct']}/{s['blocked_total']}")
    for doc, d in s["by_doc"].items():
        print(f"  {doc:<10} 통과 {d['passed']}/{d['n']}  hit {d['hit']}/{d['n']}")
    if args.model:
        print(f"답변 정답 {s['answer_ok']}/{s['answer_total']} ({args.model})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
