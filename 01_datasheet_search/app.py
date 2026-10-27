"""시연 화면.  streamlit run app.py

질문을 넣으면 모델을 부르기 전까지의 판정을 단계별로 보여 주고,
통과한 경우에만 답변을 만든 뒤 근거 원문과 대조한다.
"""

from __future__ import annotations

import contextlib
import io
import json

import altair as alt
import pandas as pd
import streamlit as st

from config import (
    ANSWER_MODEL,
    COMPARE_MODELS,
    EVIDENCE_MIN_COVERAGE,
    EVIDENCE_MIN_SCORE,
    RESULTS_DIR,
)
from src.llm import has_key, parse_spec
from src.catalog import CatalogSearch, CatalogTrace, Route
from src.pipeline import DatasheetSearch, Trace, default_index
from src.prompt import NOT_FOUND_ANSWER
from src.variants import extract_parts

CLOUD_MODELS = ["anthropic:claude-fable-5-1", "openai:gpt-4.1-mini"]

SCENARIOS = [
    ("① 공통 사양", "CH334U의 동작 전압(VDD33)은?",
     "품번을 적지 않은 전기적 특성 표(p.21)가 근거로 들어오는지"),
    ("② 변종 구분", "CH334R의 패키지는?",
     "같은 표에 있는 CH334U(QSOP28) 값이 섞이지 않는지"),
    ("③ 없는 품번", "CH999Z의 동작 전압은?",
     "문서에 없는 품번이면 모델을 부르지 않는지"),
    ("④ 없는 사양", "CH334U는 HDMI 2.1을 지원하는가?",
     "품번은 맞지만 문서에 없는 내용이면 모델을 부르지 않는지"),
]

CATALOG_SCENARIOS = [
    ("① 회로 기호", "U5의 스위칭 주파수는?",
     "BOM 으로 U5 → LMR33630ADDAR → 변종 LMR33630A 를 찾고 그 문서 한 권에서만 찾는지"),
    ("② TI 변종", "TPS2553의 EN 핀 극성은?",
     "TPS2552(로우)와 TPS2553(하이)를 문서별 변종 규칙으로 가르는지"),
    ("③ 미등록 부품", "U4의 정전기 보호 전압은?",
     "BOM 에는 있지만 데이터시트가 없는 부품이면 답하지 않는지"),
    ("④ 두 부품", "TPS2552와 LMR33630의 입력 전압 범위 차이는?",
     "문서가 둘로 갈리면 섞지 않고 답하지 않는지"),
]
SCOPES = {"CH334 한 권": "single", "modulino 부품 전체 (문서 5권)": "catalog"}

st.set_page_config(page_title="데이터시트 검색", page_icon="🔎", layout="wide")


@st.cache_resource
def get_search() -> DatasheetSearch:
    return DatasheetSearch(default_index())


@st.cache_resource
def get_catalog() -> CatalogSearch:
    return CatalogSearch()


@st.cache_data
def gate_evaluation() -> str:
    import evaluate

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        evaluate.main()
    return buffer.getvalue()


def load_json(name: str):
    path = RESULTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def badge(ok: bool, yes: str = "통과", no: str = "차단") -> str:
    return f":green-background[{yes}]" if ok else f":red-background[{no}]"


def step(number: str, title: str, status: str, body: str) -> None:
    with st.container(border=True):
        left, right = st.columns([1, 4])
        left.markdown(f"**{number} {title}**  \n{status}")
        right.markdown(body)


def excluded_variants(search: DatasheetSearch, target_raw: str | None) -> list[str]:
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


ROUTE_LABEL = {
    "doc": ("문서 선택", True),
    "off": ("문서 꺼짐", False),
    "unregistered": ("데이터시트 미등록", False),
    "unknown": ("BOM 에 없는 품번", False),
    "none": ("부품 미지정", False),
    "multi": ("두 부품 이상", False),
}


def show_route(route: Route) -> None:
    label, ok = ROUTE_LABEL[route.kind]
    body = route.message
    if route.refdes:
        body += "  \n회로 기호: " + ", ".join(f"{r} = {v}" for r, v in route.refdes)
    if route.kind == "doc" and route.question:
        body += f"  \n문서 검색에 넘긴 질문: `{route.question}`"
    step("0", "문서 선택", badge(ok, label, label), body)


def show_trace(search: DatasheetSearch, trace: Trace) -> None:
    s = trace.sufficiency

    step("1", "품번 추출",
         f":blue-background[{trace.target or '없음'}]",
         ("특정 변종을 물었습니다. 이 품번에 해당하지 않는 청크는 다음 단계에서 뺍니다."
          if trace.target_specific else
          "특정 변종이 없어 문서 전체를 대상으로 찾습니다."))

    excluded = excluded_variants(search, trace.target)
    step("2", "변종 게이트",
         f"{trace.gate_allowed}/{trace.total_chunks} 통과",
         f"다른 변종만 적힌 청크 **{trace.gate_excluded}개 제외**"
         + (f"  \n제외된 청크의 품번: {', '.join(excluded)}" if excluded else ""))

    if trace.evidence:
        table = pd.DataFrame(
            [
                {
                    "순위": rank,
                    "페이지": e.page,
                    "섹션": e.section,
                    "BM25": round(e.score, 1),
                    "품번 명시": "●" if e.names_target else "",
                }
                for rank, e in enumerate(trace.evidence, start=1)
            ]
        )
        with st.container(border=True):
            st.markdown(f"**3 BM25 검색**  \n검색 질의 `{trace.lexical_query.strip()}`")
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.caption("품번은 순위에서 빼고 내용으로 매깁니다. 품번을 명시한 근거 하나는 항상 남깁니다.")
    else:
        step("3", "BM25 검색", ":gray-background[근거 없음]",
             "질문한 품번을 명시한 청크가 없어 검색 결과를 쓰지 않습니다 (4단계)."
             if trace.verification_failed else "질문 내용과 겹치는 청크가 없습니다.")

    step("4", "검증 게이트",
         badge(not trace.verification_failed),
         (f"문서 어디에도 **{trace.target}** 를 명시한 청크가 없습니다. 답을 만들지 않습니다."
          if trace.verification_failed else
          "질문한 품번이 문서에 실제로 있습니다." if trace.target_specific else "해당 없음"))

    found = ", ".join(s.found) or "-"
    missing = ", ".join(s.missing) or "-"
    step("5", "근거 충분성",
         badge(s.passed),
         f"1위 점수 **{s.top_score:.1f}** (기준 {s.min_score:g}),  "
         f"용어 등장 비율 **{s.coverage:.0%}** (기준 {s.min_coverage:.0%})  \n"
         f"근거에 있음: {found}  /  없음: **{missing}**")

    if trace.completion:
        c = trace.completion
        step("6", "모델 호출", ":blue-background[호출]",
             f"`{c.model}`  입력 {c.input_tokens:,} / 출력 {c.output_tokens:,} 토큰, {c.latency_s:.1f}초"
             + ("  \n안전 분류기 폴백으로 다른 모델이 답함" if c.fallback_used else ""))
    elif trace.blocked:
        step("6", "모델 호출", ":gray-background[호출 안 함]", "근거가 부족해 모델을 부르지 않았습니다. 토큰 0.")
    elif trace.error:
        step("6", "모델 호출", ":red-background[오류]", trace.error)
    else:
        step("6", "모델 호출", ":gray-background[대기]", "검색 단계까지만 실행했습니다.")

    if trace.answer:
        st.markdown("#### 답변")
        if trace.answer == NOT_FOUND_ANSWER:
            st.warning(trace.answer)
        else:
            st.success(trace.answer)
    if trace.check:
        k = trace.check
        st.markdown(
            f"**원문 대조** {badge(k.passed, '통과', '확인 필요')}  "
            f"인용 페이지 {list(k.cited_pages) or '없음'}"
            + (f", 근거 밖 페이지 {list(k.invalid_citations)}" if k.invalid_citations else "")
            + f",  단위 붙은 수치 {list(k.numbers) or '없음'}"
            + (f", 원문에 없는 수치 **{list(k.unsupported_numbers)}**" if k.unsupported_numbers else "")
        )

    if trace.evidence:
        st.markdown("#### 근거 원문")
        for e in trace.evidence:
            with st.expander(f"p.{e.page}  {e.section}  (BM25 {e.score:.1f})"):
                st.text(e.text)


def ask_tab(search: DatasheetSearch, model: str, use_model: bool, scope: str,
            enabled: set[str] | None = None) -> None:
    scenarios = CATALOG_SCENARIOS if scope == "catalog" else SCENARIOS
    st.markdown("##### 시연 시나리오")
    # ?q=질문 으로 열면 그 질문을 바로 실행한다 (발표 자료 캡처용)
    if "question" not in st.session_state and st.query_params.get("q"):
        st.session_state["question"] = st.query_params["q"]
    columns = st.columns(len(scenarios))
    for column, (label, question, why) in zip(columns, scenarios):
        if column.button(label, use_container_width=True, help=why):
            st.session_state["question"] = question
    question = st.text_input("질문", key="question", placeholder="예: CH334U의 동작 온도 범위는?")
    if not question.strip():
        st.caption("시나리오를 누르거나 질문을 입력하세요. 한국어 기술 용어는 영문으로 바꿔 검색합니다.")
        return
    for label, q, why in scenarios:
        if q == question:
            st.caption(f"확인할 것: {why}")
    if scope == "catalog":
        catalog = get_catalog()
        ct: CatalogTrace = (catalog.answer(question, model, enabled=enabled) if use_model
                            else catalog.retrieve(question, enabled=enabled))
        show_route(ct.route)
        if ct.trace is None:
            step("6", "모델 호출", ":gray-background[호출 안 함]", "문서를 한 권으로 정하지 못해 모델을 부르지 않았습니다. 토큰 0.")
            st.markdown("#### 답변")
            st.warning(NOT_FOUND_ANSWER)
            return
        if not use_model:
            ct.trace.answer = NOT_FOUND_ANSWER if ct.trace.blocked else ""
        show_trace(catalog.searcher(ct.route.doc), ct.trace)
        return
    trace = search.answer(question, model) if use_model else search.retrieve(question)
    if not use_model:
        trace.answer = NOT_FOUND_ANSWER if trace.blocked else ""
    show_trace(search, trace)


def results_tab() -> None:
    st.markdown("#### 1. 변종 게이트 전수 검사")
    st.caption("python evaluate.py — 모델을 부르지 않고 PDF, 청크, BM25 만으로 돈다")
    st.code(gate_evaluation(), language=None)

    st.markdown("#### 2. 근거 충분성 기준값")
    chosen = load_json("threshold.json")
    measured = load_json("threshold_measure.json")
    if chosen and measured:
        st.markdown(
            f"정답 질문의 용어 비율은 최저 **{chosen['answerable_min_coverage']:.2f}**, "
            f"문서에 없는 질문은 최고 **{chosen['absent_max_coverage']:.2f}**. "
            f"가운데인 **{chosen['min_coverage']}** 을 기준으로 정했습니다.  \n"
            f"점수 기준만 쓰면 문서에 없는 질문 중 최고 점수가 **{chosen['absent_max_score']:.1f}** 로 "
            f"정답 질문 최저 **{chosen['answerable_min_score']:.1f}** 보다 높아 다 막을 수 없습니다."
        )
        frame = pd.DataFrame(
            [
                {
                    "문항": r["id"],
                    "질문": r["question"],
                    "구분": "문서에 없음" if r["type"] == "not_in_doc" else "정답 있음",
                    "1위 점수": r["top_score"],
                    "용어 비율": r["coverage"],
                }
                for r in measured
                if not r["gate_blocked"]
            ]
        )
        color = alt.Color(
            "구분:N",
            scale=alt.Scale(domain=["정답 있음", "문서에 없음"], range=["#2a78d6", "#eb6834"]),
            legend=alt.Legend(orient="top", title=None),
        )
        shape = alt.Shape("구분:N", scale=alt.Scale(domain=["정답 있음", "문서에 없음"],
                                                   range=["circle", "cross"]), legend=None)
        points = alt.Chart(frame).mark_point(filled=True, size=90, stroke="#fcfcfb", strokeWidth=1.5).encode(
            x=alt.X("용어 비율:Q", scale=alt.Scale(domain=[-0.05, 1.05])),
            y=alt.Y("1위 점수:Q"),
            color=color,
            shape=shape,
            tooltip=["문항", "질문", "1위 점수", "용어 비율"],
        )
        rule_c = alt.Chart(pd.DataFrame({"x": [EVIDENCE_MIN_COVERAGE]})).mark_rule(
            strokeDash=[4, 3], color="#52514e").encode(x="x:Q")
        rule_s = alt.Chart(pd.DataFrame({"y": [EVIDENCE_MIN_SCORE]})).mark_rule(
            strokeDash=[4, 3], color="#52514e").encode(y="y:Q")
        st.altair_chart((points + rule_c + rule_s).properties(height=360), use_container_width=True)
        st.caption("점선이 기준값. 오른쪽 위 영역만 모델을 부른다. 품번 검증 게이트에서 먼저 막힌 문항은 제외.")
    else:
        st.info("python calibrate.py 를 먼저 실행하세요.")

    st.markdown("#### 3. 모델 비교")
    comparison = load_json("model_comparison.json")
    if not comparison:
        st.info("python evaluate_answers.py 실행 결과가 아직 없습니다.")
        return
    r = comparison["retrieval"]
    st.markdown(
        f"검색 단계 (모델과 무관): 기대 페이지 hit@5 **{r['hit']}/{r['answerable']}**, "
        f"문서에 없는 질문을 모델 호출 전에 차단 **{r['blocked']}/{r['absent']}**"
    )
    rows = []
    for m in comparison["models"]:
        rows.append(
            {
                "모델": m["model"],
                "정답": f"{m['answerable_correct']}/{m['answerable_total']}",
                "변종 구분": f"{m['variant_correct']}/{m['variant_total']}",
                "잘못된 거절": m["wrong_refusals"],
                "문서에 없음 거절": f"{m['absent_correct']}/{m['absent_total']}",
                "원문 대조 통과": f"{m['check_passed']}/{m['check_total']}",
                "평균 입력 토큰": m["input_tokens_avg"],
                "평균 출력 토큰": m["output_tokens_avg"],
                "응답 시간 중앙값(초)": m["latency_median_s"],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(f"실행 시각 {comparison['generated_at']}")

    st.markdown("#### 4. 프로젝트 부품 전체로 넓힌 결과")
    before = load_json("catalog_eval_before.json")
    after = load_json("catalog_eval_after.json")
    if after:
        rows = []
        for tag, data in (("문서 등록 직후", before), ("용어표 보강 후", after)):
            if not data:
                continue
            s = data["summary"]
            rows.append({
                "측정": tag,
                "문서 선택": f"{s['route_ok']}/{s['questions']}",
                "답 있는 질문 통과": f"{s['answerable_passed']}/{s['answerable']}",
                "근거에 정답 문장": f"{s['evidence_hit']}/{s['answerable']}",
                "답하지 말 질문 차단": f"{s['blocked_correct']}/{s['blocked_total']}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("기준값(2.5, 0.67)은 CH334 평가셋으로 정한 값을 그대로 썼다. TI 문서 4권은 기준값을 정할 때 쓰지 않았다.")
    coverage = RESULTS_DIR / "catalog_coverage.md"
    if coverage.exists():
        with st.expander("BOM 데이터시트 등록 현황"):
            st.markdown(coverage.read_text(encoding="utf-8"))


def main() -> None:
    search = get_search()
    with st.sidebar:
        st.markdown("### 검색 범위")
        default_scope = 1 if st.query_params.get("scope") == "catalog" else 0
        scope = SCOPES[st.radio("문서", list(SCOPES), index=default_scope, label_visibility="collapsed")]
        st.markdown("### 설정")
        # 클라우드 모델은 .env 에 키를 넣으면 같은 화면, 같은 판정 기준으로 바로 바뀐다
        models = list(dict.fromkeys([ANSWER_MODEL, *COMPARE_MODELS, *CLOUD_MODELS]))
        model = st.selectbox("답변 모델", models)
        available = has_key(model)
        use_model = st.toggle("모델 호출", value=available, disabled=not available)
        if not available:
            provider = parse_spec(model).provider
            reason = "Ollama 가 실행 중이 아니어서" if provider == "ollama" else f"{provider} API 키가 없어"
            st.caption(f"{reason} 검색 단계까지만 실행합니다.")
        elif parse_spec(model).provider == "ollama":
            st.caption("이 PC 에서 도는 로컬 모델 (비용 없음). 첫 답변은 모델을 올리느라 느리다.")
        st.markdown("### 기준값")
        st.markdown(f"- 1위 점수 ≥ **{EVIDENCE_MIN_SCORE:g}**\n- 용어 등장 비율 ≥ **{EVIDENCE_MIN_COVERAGE:.0%}**")
        st.caption("calibrate.py 가 평가셋 38문항으로 정한 값")
        st.markdown("### 문서")
        enabled: set[str] | None = None
        if scope == "catalog":
            st.caption("켠 문서만 검색 범위가 된다. 한 권만 켜면 품번 없이 물어도 그 문서에서 찾는다.")
            enabled = set()
            for doc in get_catalog().catalog.docs:
                entry = get_catalog().catalog.bom_part_for_doc(doc.id)
                where = f" ({', '.join(entry.refs)})" if entry else ""
                if st.checkbox(f"{doc.title.split(' Datasheet')[0]}{where}", value=True, key=f"doc_{doc.id}",
                               help=f"{doc.manufacturer} {doc.title} ({doc.role})"):
                    enabled.add(doc.id)
        else:
            st.caption(f"CH334_CH335_V2.5.pdf, 32쪽, 청크 {len(search.index.documents)}개")

    if scope == "catalog":
        st.title("modulino 부품 데이터시트 검색")
        st.caption("문서를 먼저 고르고 그 한 권 안에서만 찾는다. 회로 기호(U5)는 BOM 으로 품번을 찾는다. "
                   "데이터시트가 등록되지 않은 부품, 두 부품 비교, 부품이 없는 질문은 답하지 않는다.")
    else:
        st.title("CH334/CH335 데이터시트 검색")
        st.caption("답하지 않을 조건을 먼저 정한 검색 도구. 품번 13종이 한 권에 섞인 데이터시트에서 "
                   "질문한 품번의 근거만 찾고, 근거가 부족하면 모델을 부르지 않는다.")

    ask, results = st.tabs(["질문하기", "검증 결과"])
    with ask:
        ask_tab(search, model, use_model, scope, enabled)
    with results:
        results_tab()


main()
