"""설치 없이 여는 시연 기록 docs/demo.html 을 만든다.  python tools/export_demo.py

제출 ZIP 에서는 맨 위의 시연.html 이 된다. Python 이 없는 PC 에서도 브라우저로 연다.
- 검색 단계(품번 추출부터 근거 충분성까지)는 지금 코드로 다시 돌린다
- 모델 답변은 평가 때 저장한 결과(results/)를 그대로 붙인다. 모델을 새로 부르지 않는다
- 원문 대조는 붙인 답변으로 다시 계산한다
- 지금 검색 결과의 페이지가 저장된 결과와 다르면 실패로 끝난다 (기록과 코드가 어긋난 것)
"""

from __future__ import annotations

import base64
import contextlib
import html
import io
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import EVIDENCE_MIN_COVERAGE, EVIDENCE_MIN_SCORE, RESULTS_DIR  # noqa: E402
from src.catalog import CatalogSearch  # noqa: E402
from src.llm import Completion  # noqa: E402
from src.pipeline import DatasheetSearch, Trace, default_index  # noqa: E402
from src.prompt import NOT_FOUND_ANSWER  # noqa: E402
from src.variants import extract_parts  # noqa: E402

OUT = ROOT / "docs" / "demo.html"
SINGLE_RUN = "answers_ollama_gemma3_4b.json"   # gemma3 평가 1회차
CATALOG_RUN = "catalog_eval_gemma.json"

# app.py 의 시나리오와 같다
SCENARIOS = [
    ("single", "① 공통 사양", "CH334U의 동작 전압(VDD33)은?",
     "품번을 적지 않은 전기적 특성 표(p.21)가 근거로 들어오는지"),
    ("single", "② 변종 구분", "CH334R의 패키지는?",
     "같은 표에 있는 CH334U(QSOP28) 값이 섞이지 않는지"),
    ("single", "③ 없는 품번", "CH999Z의 동작 전압은?",
     "문서에 없는 품번이면 모델을 부르지 않는지"),
    ("single", "④ 없는 사양", "CH334U는 HDMI 2.1을 지원하는가?",
     "품번은 맞지만 문서에 없는 내용이면 모델을 부르지 않는지"),
    ("catalog", "① 회로 기호", "U5의 스위칭 주파수는?",
     "BOM 으로 U5 → LMR33630ADDAR → 변종 LMR33630A 를 찾고 그 문서 한 권에서만 찾는지"),
    ("catalog", "② TI 변종", "TPS2553의 EN 핀 극성은?",
     "TPS2552(로우)와 TPS2553(하이)를 문서별 변종 규칙으로 가르는지"),
    ("catalog", "③ 미등록 부품", "U4의 정전기 보호 전압은?",
     "BOM 에는 있지만 데이터시트가 없는 부품이면 답하지 않는지"),
]
NOTES = {
    "CH334U의 동작 전압(VDD33)은?": "이 답은 평가 1회차 기록이다. 2, 3회차에는 같은 질문에 전형값 3.3 V 를 빠뜨려 오답으로 채점됐고, "
                                 "원문 대조는 세 번 모두 통과했다. 원문 대조는 근거 밖 정보를 막을 뿐 정답을 보장하지 않아 정답 사실을 따로 채점한다.",
    "CH334R의 패키지는?": "이 답은 평가 1회차 기록이다. 같은 모델에 이 질문만 따로 물으면 5회 모두 같은 표 옆 칸의 "
                         "CH334U 값(QSOP28)을 냈다. 수치 대조로는 잡히지 않아 평가셋에서 혼입을 따로 채점한다.",
}
ROUTE_LABEL = {
    "doc": ("문서 선택", True),
    "off": ("문서 꺼짐", False),
    "unregistered": ("데이터시트 미등록", False),
    "unknown": ("BOM 에 없는 품번", False),
    "none": ("부품 미지정", False),
    "multi": ("두 부품 이상", False),
}
TYPE_LABEL = {"answerable": "정답형", "variant": "변종 구분", "refdes": "회로 기호",
              "not_in_doc": "문서에 없음", "blocked": "답하지 않아야 함"}


def load(name: str):
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def replay(record: dict):
    """저장된 답변을 모델 호출 자리에 넣는다. 파이프라인은 그대로 돈다."""
    def generate(model, instructions, prompt):
        return Completion(
            text=record["answer"],
            model=record.get("served_by") or model.split(":", 1)[-1],
            input_tokens=record.get("input_tokens") or 0,
            output_tokens=record.get("output_tokens") or 0,
            latency_s=record.get("latency_s") or 0.0,
            stop_reason=record.get("stop_reason") or "",
        )
    return generate


def excluded_variants(search: DatasheetSearch, target_raw: str | None) -> list[str]:
    """app.py 의 excluded_variants 와 같다."""
    if not target_raw:
        return []
    rules = search.index.rules
    target = rules.primary(target_raw)
    if target is None:
        return []
    allowed = set(search.index.gate(target))
    found: set[str] = set()
    for i, document in enumerate(search.index.documents):
        if i not in allowed:
            if hasattr(rules, "variants_in"):
                found.update(rules.variants_in(document.text))
            else:
                found.update(p.raw for p in extract_parts(document.text) if p.is_specific)
    return sorted(found - {target_raw})


def stages(search: DatasheetSearch, t: Trace) -> dict:
    s = t.sufficiency
    return {
        "target": t.target, "specific": t.target_specific,
        "total": t.total_chunks, "allowed": t.gate_allowed, "excluded": t.gate_excluded,
        "excluded_parts": excluded_variants(search, t.target),
        "query": t.lexical_query.strip(),
        "evidence": [{"page": e.page, "section": e.section, "score": round(e.score, 1),
                      "names": e.names_target, "text": e.text} for e in t.evidence],
        "verify_failed": t.verification_failed,
        "suff": {"passed": s.passed, "top": round(s.top_score, 2), "min_score": s.min_score,
                 "cov": s.coverage, "min_cov": s.min_coverage,
                 "found": list(s.found), "missing": list(s.missing)},
    }


def model_part(t: Trace, record: dict | None) -> dict:
    if t.completion:
        c = t.completion
        return {"called": True, "name": c.model, "in": c.input_tokens, "out": c.output_tokens,
                "latency": c.latency_s, "recorded": bool(record and record.get("input_tokens"))}
    return {"called": False}


def check_part(t: Trace) -> dict | None:
    k = t.check
    if not k:
        return None
    return {"passed": k.passed, "cited": list(k.cited_pages), "invalid": list(k.invalid_citations),
            "numbers": list(k.numbers), "unsupported": list(k.unsupported_numbers), "mixed": k.mixed_refusal}


def single_items(problems: list[str]) -> list[dict]:
    search = DatasheetSearch(default_index())
    questions = json.loads((ROOT / "tests" / "evaluation_questions.json").read_text(encoding="utf-8"))
    stored = {r["question"]: r for r in load(SINGLE_RUN)}
    items = []
    for q in questions:
        record = stored[q["question"]]
        t = search.answer(q["question"], "ollama:gemma3:4b", generate=replay(record))
        now = [e.page for e in t.evidence]
        if record.get("retrieved_pages") is not None and now != record["retrieved_pages"]:
            problems.append(f"{q['id']} 검색 페이지가 기록과 다름: 지금 {now}, 기록 {record['retrieved_pages']}")
        if t.llm_called != record["llm_called"]:
            problems.append(f"{q['id']} 모델 호출 여부가 기록과 다름")
        items.append({
            "id": q["id"], "scope": "single", "type": q["type"], "question": q["question"],
            "source": q.get("source", ""),
            "stages": stages(search, t), "model": model_part(t, record),
            "answer": t.answer, "check": check_part(t),
            "grade": {"correct": record.get("correct"), "strict": bool(record.get("correct")) and bool(record.get("check_passed")),
                      "problems": record.get("problems") or []},
        })
    return items


def catalog_items(problems: list[str]) -> list[dict]:
    catalog = CatalogSearch()
    questions = json.loads((ROOT / "tests" / "catalog_questions.json").read_text(encoding="utf-8"))
    stored = {r["question"]: r for r in load(CATALOG_RUN)["rows"]}
    items = []
    for q in questions:
        record = stored[q["question"]]
        ct = catalog.answer(q["question"], "ollama:gemma3:4b", generate=replay(record))
        r = ct.route
        label, ok = ROUTE_LABEL[r.kind]
        item = {
            "id": q["id"], "scope": "catalog", "type": q["type"], "question": q["question"],
            "route": {"kind": r.kind, "label": label, "ok": ok, "message": r.message,
                      "refdes": [list(x) for x in r.refdes], "question": r.question},
            "stages": None, "model": {"called": False}, "answer": ct.answer or NOT_FOUND_ANSWER, "check": None,
            "grade": {"correct": record.get("answer_ok"), "strict": bool(record.get("answer_ok")) and bool(record.get("check_passed")),
                      "problems": []} if record.get("answer") and not record.get("blocked") else None,
        }
        if r.kind != record["route"]:
            problems.append(f"{q['id']} 문서 선택이 기록과 다름: 지금 {r.kind}, 기록 {record['route']}")
        if ct.trace is not None:
            t = ct.trace
            now = [e.page for e in t.evidence]
            if not t.blocked and now != record["pages"]:
                problems.append(f"{q['id']} 검색 페이지가 기록과 다름: 지금 {now}, 기록 {record['pages']}")
            item.update(stages=stages(catalog.searcher(r.doc), t), model=model_part(t, record),
                        answer=t.answer, check=check_part(t))
        items.append(item)
    return items


def gate_text() -> str:
    import evaluate
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        evaluate.main()
    return buffer.getvalue().strip()


def readme_html(path: Path, images: dict[str, Path] | None = None) -> str:
    """README 를 이 화면 안에 그대로 보여 준다. 브라우저는 .md 파일을 원문 텍스트로 열기 때문이다.
    README 가 가리키는 그림은 data URI 로 넣는다."""
    from markdown_it import MarkdownIt

    text = path.read_text(encoding="utf-8")
    for name, image in (images or {}).items():
        text = text.replace(f"]({name})", "](data:image/png;base64," + base64.b64encode(image.read_bytes()).decode() + ")")
    return MarkdownIt("commonmark").enable("table").render(text)


def kpi(big: str, label: str, sub: str, orange: bool = False) -> str:
    return (f'<div class="kpi"><div class="big{" o" if orange else ""}">{html.escape(big)}</div>'
            f'<div class="l">{html.escape(label)}</div><div class="s">{html.escape(sub)}</div></div>')


def results_body(S: dict, T: dict, M: dict, gate: str, chart: str, models_md: str) -> str:
    """검증 결과 탭: 중요한 숫자는 크게, 나머지는 한 줄 설명과 접어 둔 원문으로"""
    part1 = ('<h3>1. 변종 게이트<span class="cmd">python evaluate.py</span></h3><div class="kpis">'
             + kpi(f"{S['worst_case_violations']}건", "최악 조건", "다른 변종을 검색 1위로 넣어도 걸러짐")
             + kpi("빈 결과" if S['unknown_part_empty'] else "실패", "없는 품번 CH999Z", "답을 만들지 않음", orange=True)
             + '</div><details><summary>스크립트 출력 원문</summary><pre>' + html.escape(gate) + '</pre></details>')
    part2 = ('<h3>2. 차단 기준값<span class="cmd">python calibrate.py</span></h3><div class="kpis">'
             + kpi(f"{T['min_coverage']}", "질문 용어 등장 비율",
                   f"정답 질문 최저 {T['answerable_min_coverage']:.2f}, 문서에 없는 질문 최고 {T['absent_max_coverage']:.2f} 의 가운데")
             + kpi(f"{T['min_score']}", "BM25 1위 점수 (보조)", f"점수만으로는 AEC-Q100 질문({T['absent_max_score']:.1f})을 못 막음")
             + kpi(f"{T['correct']}/{T['answerable_total'] + T['absent_total']}", "평가셋 판정 일치",
                   f"정답 질문 {T['answerable_pass']}개 통과, 문서에 없는 질문 {T['absent_block']}개 차단")
             + '</div><details><summary>질문별 분포 차트</summary>'
             + f'<img class="chart" alt="평가셋 질문별 BM25 1위 점수와 용어 등장 비율 분포" src="data:image/png;base64,{chart}"></details>')
    rows = []
    for m in M["models"]:
        runs = m.get("runs") or [m]
        strict = sorted({r["answerable_strict"] for r in runs})
        strict_txt = f"{strict[0]}~{strict[-1]}" if len(strict) > 1 else f"{strict[0]}"
        name = m["model"].split(":", 1)[1].replace("@nothink", "")
        rows.append(f'<tr><td><b>{html.escape(name)}</b></td>'
                    f'<td class="n">{strict_txt}/{m["answerable_total"]}</td>'
                    f'<td class="n hi">{m["variant_correct"]}/{m["variant_total"]}</td>'
                    f'<td class="n">{m["absent_correct"]}/{m["absent_total"]}</td>'
                    f'<td class="n">{m["latency_median_s"]:.1f}초</td></tr>')
    part3 = ('<h3>3. 답변 모델 비교<span class="cmd">python evaluate_answers.py</span></h3>'
             '<table class="cmp"><tr><th>로컬 모델</th><th>엄격 정답 (3회)</th><th>변종 구분</th><th>문서에 없음 거절</th><th>응답 시간</th></tr>'
             + "".join(rows) + '</table>'
             '<p class="cmt">정답률은 nemotron, 변종 구분은 gemma3 가 높습니다. 4B 모델에 같은 표에서 변종 값을 고르는 일은 맡기기 어렵습니다.</p>'
             '<details><summary>전체 표와 반복 실행 결과</summary>' + models_md + '</details>')
    return part1 + part2 + part3


def md_to_html(md: str) -> str:
    """model_comparison.md 정도의 마크다운(제목, 목록, 표, 문단)만 바꾼다."""
    out, table, bullets = [], [], []

    def flush():
        if table:
            rows = [r for r in table if not set(r.replace("|", "").strip()) <= {"-", ":", " "}]
            cells = [[html.escape(c.strip()) for c in r.strip().strip("|").split("|")] for r in rows]
            out.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in cells[0]) + "</tr>"
                       + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in cells[1:]) + "</table>")
            table.clear()
        if bullets:
            out.append("<ul>" + "".join(f"<li>{html.escape(b)}</li>" for b in bullets) + "</ul>")
            bullets.clear()

    for line in md.splitlines():
        if line.startswith("|"):
            table.append(line)
            continue
        if line.startswith("- "):
            bullets.append(line[2:])
            continue
        flush()
        if line.startswith("## "):
            out.append(f"<h4>{html.escape(line[3:])}</h4>")
        elif line.startswith("# "):
            out.append(f"<h3>{html.escape(line[2:])}</h3>")
        elif line.strip():
            out.append(f"<p>{html.escape(line)}</p>")
    flush()
    return "\n".join(out)


def main() -> int:
    problems: list[str] = []
    items = single_items(problems) + catalog_items(problems)
    by_q = {(i["scope"], i["question"]): i for i in items}
    scenarios = []
    for scope, label, question, why in SCENARIOS:
        if (scope, question) not in by_q:
            problems.append(f"시나리오 질문이 평가셋에 없음: {question}")
            continue
        scenarios.append({"scope": scope, "label": label, "question": question, "why": why})
    if problems:
        print("기록과 지금 코드가 어긋납니다. 시연 기록을 만들지 않습니다.\n  " + "\n  ".join(problems))
        return 1

    S = load("summary.json")
    CA = load("catalog_eval_after.json")["summary"]
    chart = base64.b64encode((RESULTS_DIR / "threshold_chart.png").read_bytes()).decode()
    data = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "min_score": EVIDENCE_MIN_SCORE, "min_coverage": EVIDENCE_MIN_COVERAGE,
        "not_found": NOT_FOUND_ANSWER,
        "type_label": TYPE_LABEL, "notes": NOTES,
        "items": items, "scenarios": scenarios,
        "summary": S, "catalog": CA,
    }
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = (TEMPLATE
            .replace("%%DATA%%", blob)
            .replace("%%RESULTS_BODY%%", results_body(S, load("threshold.json"), load("model_comparison.json"), gate_text(), chart,
                                                      md_to_html((RESULTS_DIR / "model_comparison.md").read_text(encoding="utf-8"))))
            .replace("%%IDEC_README%%", readme_html(ROOT / "docs" / "idec_README.md",
                                                   {"workitem_example.png": ROOT / "docs" / "idec_img" / "workitem_example.png"}))
            .replace("%%DOCS_README%%", readme_html(ROOT / "docs" / "docs_README.md"))
            .replace("%%GENERATED%%", data["generated"]))
    OUT.write_text(page, encoding="utf-8")
    single = sum(i["scope"] == "single" for i in items)
    print(f"{OUT}  (질문 {len(items)}개: 한 권 {single}, 부품 전체 {len(items) - single}, "
          f"{OUT.stat().st_size / 1024:.0f} KB). 검색 페이지와 문서 선택이 저장된 평가 기록과 모두 같다")
    return 0


TEMPLATE = (Path(__file__).with_name("demo_template.html")).read_text(encoding="utf-8")

if __name__ == "__main__":
    sys.exit(main())
