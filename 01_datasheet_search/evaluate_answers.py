"""평가셋으로 답변 모델을 비교한다. 검색과 판정 기준은 고정하고 모델만 바꾼다.

  python evaluate_answers.py                                    .env 의 COMPARE_MODELS
  python evaluate_answers.py --models openai:gpt-4.1-mini,anthropic:claude-fable-5-1

질문마다 남기는 것
  검색     기대 페이지가 상위 5개 안에 있는가 (hit@5), 모델 호출 전에 차단됐는가
  답변     정답 사실 포함 여부 (must_include), 다른 변종 값 혼입 (must_not_include)
  대조     인용 페이지가 넘긴 근거 안인가, 단위 붙은 수치가 원문에 있는가
  비용     입력/출력 토큰, 응답 시간

결과  results/answers_<모델>.json, results/model_comparison.json, results/model_comparison.md
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import COMPARE_MODELS, EVALSET_PATH, EVIDENCE_MIN_COVERAGE, EVIDENCE_MIN_SCORE, RESULTS_DIR
from src.answer_check import check_answer, is_refusal
from src.llm import has_key, parse_spec
from src.pipeline import DatasheetSearch


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def grade(answer: str, item: dict) -> tuple[bool, list[str]]:
    """정답 사실이 모두 들어 있고 다른 변종 값이 없으면 정답."""
    body = normalize(answer)
    problems = []
    for group in item.get("must_include", []):
        if not any(normalize(option) in body for option in group):
            problems.append(f"누락: {'/'.join(group)}")
    for banned in item.get("must_not_include", []):
        if normalize(banned) in body:
            problems.append(f"혼입: {banned}")
    return not problems, problems


def safe_name(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", spec)


def run_model(search: DatasheetSearch, items: list[dict], spec: str) -> list[dict]:
    rows = []
    consecutive_errors = 0
    for item in items:
        if consecutive_errors >= 2:
            # 키·잔액·모델 ID 문제면 나머지도 같은 오류다. 시간과 비용을 쓰지 않고 멈춘다
            raise RuntimeError(f"{spec}: 연속 오류로 중단 ({rows[-1]['error']})")
        trace = search.answer(item["question"], spec)
        consecutive_errors = consecutive_errors + 1 if trace.error else 0
        row = {"llm_called": trace.llm_called, "answer": trace.answer, "error": trace.error}
        if trace.completion:
            c = trace.completion
            row.update(
                served_by=c.model,
                fallback_used=c.fallback_used,
                stop_reason=c.stop_reason,
                input_tokens=c.input_tokens,
                output_tokens=c.output_tokens,
                latency_s=round(c.latency_s, 2),
            )
        rows.append(score(item, trace, row))
    return rows


def score(item: dict, trace, row: dict) -> dict:
    """검색 결과와 답변으로 채점한다. 새로 돌릴 때와 다시 채점할 때 같은 함수를 쓴다."""
    pages = [e.page for e in trace.evidence]
    expected = set(item.get("expected_pages", []))
    row.update(
        id=item["id"],
        type=item["type"],
        question=item["question"],
        retrieved_pages=pages,
        hit=bool(expected & set(pages)) if expected else None,
        blocked=trace.blocked,
        block_reason=trace.sufficiency.reason if trace.blocked else "",
    )
    answer = row["answer"]
    refused = is_refusal(answer)
    row["refused"] = refused
    if item["type"] == "not_in_doc":
        row["correct"] = refused
        row["problems"] = [] if refused else ["문서에 없는 질문에 답을 만듦"]
    else:
        correct, problems = grade(answer, item)
        if refused:
            correct, problems = False, ["답이 있는 질문을 거절함"]
        row["correct"] = correct
        row["problems"] = problems
    for key in ("check_passed", "check", "unsupported_numbers", "invalid_citations", "mixed_refusal"):
        row.pop(key, None)
    if row["llm_called"]:
        check = check_answer(answer, pages, [e.text for e in trace.evidence])
        row["check_passed"] = check.passed
        row["check"] = check.summary()
        row["unsupported_numbers"] = list(check.unsupported_numbers)
        row["invalid_citations"] = list(check.invalid_citations)
        row["mixed_refusal"] = check.mixed_refusal
    mark = "O" if row["correct"] else "X"
    print(f"  {item['id']} {mark} {'차단' if trace.blocked else ''}"
          f"{' 오류 ' + row['error'] if row.get('error') else ''}"
          f"{' ' + ', '.join(row['problems']) if row['problems'] else ''}")
    return row


def regrade(search: DatasheetSearch, items: list[dict], spec: str, name: str) -> list[dict]:
    """저장된 답변을 다시 채점한다. 모델을 부르지 않는다 (검색은 결정적이라 같은 근거가 나온다)."""
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise RuntimeError(f"{spec}: 저장된 답변이 없습니다 ({path.name})")
    saved = {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
    rows = []
    for item in items:
        row = dict(saved[item["id"]])
        rows.append(score(item, search.retrieve(item["question"]), row))
    return rows


def summarize(spec: str, rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["type"] != "not_in_doc"]
    absent = [r for r in rows if r["type"] == "not_in_doc"]
    called = [r for r in rows if r["llm_called"]]
    latencies = [r["latency_s"] for r in called]
    checked = [r for r in called if "check_passed" in r and not r["refused"]]
    return {
        "model": spec,
        "served_by": sorted({r.get("served_by", "") for r in called if r.get("served_by")}),
        "answerable_correct": sum(r["correct"] for r in answerable),
        # 엄격: 정답 사실이 있고 원문 대조(인용 페이지, 수치, 거절 혼재)도 통과.
        # 문자열 채점은 "GANG 만 지원하며, 지원하지 않습니다" 같은 모순된 답도 정답으로 센다
        "answerable_strict": sum(r["correct"] and r.get("check_passed", False) for r in answerable),
        "answerable_total": len(answerable),
        "variant_strict": sum(r["correct"] and r.get("check_passed", False) for r in answerable if r["type"] == "variant"),
        "variant_correct": sum(r["correct"] for r in answerable if r["type"] == "variant"),
        "variant_total": sum(1 for r in answerable if r["type"] == "variant"),
        "wrong_refusals": sum(r["refused"] for r in answerable),
        "absent_correct": sum(r["correct"] for r in absent),
        "absent_total": len(absent),
        "absent_blocked_before_llm": sum(r["blocked"] for r in absent),
        "llm_calls": len(called),
        "errors": sum(1 for r in rows if r["error"]),
        "fallbacks": sum(1 for r in called if r.get("fallback_used")),
        "check_passed": sum(r["check_passed"] for r in checked),
        "check_total": len(checked),
        "unsupported_numbers": sum(len(r.get("unsupported_numbers", [])) for r in checked),
        "mixed_refusals": sum(1 for r in checked if r.get("mixed_refusal")),
        "invalid_citations": sum(len(r.get("invalid_citations", [])) for r in checked),
        "input_tokens_avg": round(statistics.mean(r["input_tokens"] for r in called)) if called else 0,
        "output_tokens_avg": round(statistics.mean(r["output_tokens"] for r in called)) if called else 0,
        "output_tokens_total": sum(r["output_tokens"] for r in called),
        "latency_median_s": round(statistics.median(latencies), 2) if latencies else 0,
        "latency_max_s": round(max(latencies), 2) if latencies else 0,
    }


def write_markdown(summaries: list[dict], retrieval: dict) -> str:
    lines = [
        f"# 모델 비교 ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"평가셋 {retrieval['total']}문항 (정답형 {retrieval['answerable']}, 문서에 없음 {retrieval['absent']})",
        f"기준값 min_score={EVIDENCE_MIN_SCORE}, min_coverage={EVIDENCE_MIN_COVERAGE}",
        "",
        "## 검색 단계 (모델과 무관)",
        "",
        f"- 기대 페이지 hit@5: {retrieval['hit']}/{retrieval['answerable']}",
        f"- 문서에 없는 질문을 모델 호출 전에 차단: {retrieval['blocked']}/{retrieval['absent']}",
        f"- 정답형 질문이 잘못 차단됨: {retrieval['answerable_blocked']}건",
        "",
        "## 답변 단계",
        "",
        "| 모델 | 정답 | 엄격 정답 | 변종 구분 (엄격) | 잘못된 거절 | 문서에 없음 거절 | 원문 대조 통과 | 평균 입력/출력 토큰 | 응답 시간 중앙값 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['model']} | {s['answerable_correct']}/{s['answerable_total']} "
            f"| {s['answerable_strict']}/{s['answerable_total']} "
            f"| {s['variant_correct']}/{s['variant_total']} ({s['variant_strict']}) | {s['wrong_refusals']} "
            f"| {s['absent_correct']}/{s['absent_total']} "
            f"| {s['check_passed']}/{s['check_total']} "
            f"| {s['input_tokens_avg']} / {s['output_tokens_avg']} | {s['latency_median_s']}초 |"
        )
    lines += [
        "",
        "- 정답: 정답 사실이 답에 있고 다른 변종 값이 없다 (문자열 채점).",
        "- 엄격 정답: 정답이면서 원문 대조도 통과. 문자열 채점은 모순된 답도 정답으로 세므로 함께 본다.",
        "- 원문 대조: 인용 페이지가 넘긴 근거 안이고, 단위 붙은 수치가 근거 원문에 있고, 답과 거절 문장이 섞이지 않으면 통과.",
        "- 모델 호출 수는 모델마다 같다. 차단된 질문은 모델을 부르지 않아 토큰이 0 이다.",
    ]
    repeated = [s for s in summaries if len(s.get("runs", [])) > 1]
    if repeated:
        lines += ["", "## 반복 실행 (같은 평가셋, 같은 설정)", ""]
        for s in repeated:
            strict = [r["answerable_strict"] for r in s["runs"]]
            correct = [r["answerable_correct"] for r in s["runs"]]
            variant = [r["variant_correct"] for r in s["runs"]]
            lines.append(
                f"- {s['model']}: {len(s['runs'])}회 정답 {correct}, 엄격 {strict}, 변종 {variant}. "
                f"회차마다 결과가 바뀐 문항 {', '.join(s['unstable']) or '없음'}"
            )
        lines.append("- 로컬 모델은 온도 0, 시드 고정이어도 실행 조건(GPU, 캐시 상태)에 따라 답이 바뀐다.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(COMPARE_MODELS))
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N문항만 (점검용)")
    parser.add_argument("--fresh", action="store_true", help="이전 비교 결과와 합치지 않는다")
    parser.add_argument("--regrade", action="store_true",
                        help="모델을 부르지 않고 저장된 답변(results/answers_*.json)을 다시 채점")
    parser.add_argument("--repeat", type=int, default=1,
                        help="같은 평가셋을 N회 돌려 회차별 결과와 답이 바뀐 문항을 남긴다")
    parser.add_argument("--start", type=int, default=1,
                        help="이 회차부터 모델을 부른다. 앞 회차는 저장된 답변을 다시 채점")
    args = parser.parse_args()

    items = json.loads(EVALSET_PATH.read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]
    specs = [s.strip() for s in args.models.split(",") if s.strip()]
    for spec in specs:
        parse_spec(spec)
    if not args.regrade:
        for spec in specs:
            if not has_key(spec):
                print(f"{spec}: 호출할 수 없어 건너뜁니다 (API 키 또는 Ollama)")
        specs = [s for s in specs if has_key(s)]
    if not specs:
        print("실행할 모델이 없습니다")
        return 1

    RESULTS_DIR.mkdir(exist_ok=True)
    search = DatasheetSearch()
    summaries = []
    retrieval = None
    for spec in specs:
        runs: list[list[dict]] = []
        for k in range(1, args.repeat + 1):
            # 1회차는 answers_<모델>.json, 이후는 answers_<모델>_r<k>.json
            name = f"answers_{safe_name(spec)}" + ("" if k == 1 else f"_r{k}")
            print(f"\n[{spec}] {k}회차")
            try:
                if args.regrade or k < args.start:
                    rows = regrade(search, items, spec, name)
                else:
                    rows = run_model(search, items, spec)
            except RuntimeError as error:
                print(f"  {error}")
                break
            (RESULTS_DIR / f"{name}.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            runs.append(rows)
        if not runs:
            continue
        rows = runs[0]
        summary = summarize(spec, rows)
        summary["runs"] = [
            {
                "answerable_correct": (s := summarize(spec, r))["answerable_correct"],
                "answerable_strict": s["answerable_strict"],
                "variant_correct": s["variant_correct"],
                "variant_strict": s["variant_strict"],
            }
            for r in runs
        ]
        # 회차마다 정답 여부가 바뀐 문항 (로컬 모델은 온도 0 이어도 실행 조건에 따라 답이 바뀐다)
        summary["unstable"] = sorted(
            item["id"]
            for i, item in enumerate(items)
            if len({r[i]["correct"] for r in runs}) > 1
        )
        summaries.append(summary)
        if retrieval is None:
            answerable = [r for r in rows if r["type"] != "not_in_doc"]
            absent = [r for r in rows if r["type"] == "not_in_doc"]
            retrieval = {
                "total": len(rows),
                "answerable": len(answerable),
                "absent": len(absent),
                "hit": sum(1 for r in answerable if r["hit"]),
                "blocked": sum(r["blocked"] for r in absent),
                "answerable_blocked": sum(r["blocked"] for r in answerable),
            }

    if not summaries:
        print("완료된 모델이 없어 결과 파일을 쓰지 않습니다")
        return 1
    # 모델을 따로따로 돌려도 한 표에 모이도록 이전 결과와 합친다 (같은 모델은 새 결과로)
    previous = RESULTS_DIR / "model_comparison.json"
    if previous.exists() and not args.fresh:
        old = json.loads(previous.read_text(encoding="utf-8"))
        same_setting = (old.get("min_score"), old.get("min_coverage"), old.get("retrieval", {}).get("total")) == (
            EVIDENCE_MIN_SCORE, EVIDENCE_MIN_COVERAGE, retrieval["total"])
        if same_setting:
            done = {s["model"] for s in summaries}
            summaries = [m for m in old["models"] if m["model"] not in done] + summaries
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "min_score": EVIDENCE_MIN_SCORE,
        "min_coverage": EVIDENCE_MIN_COVERAGE,
        "retrieval": retrieval,
        "models": summaries,
    }
    (RESULTS_DIR / "model_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    markdown = write_markdown(summaries, retrieval)
    (RESULTS_DIR / "model_comparison.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
