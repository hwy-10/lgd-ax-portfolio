"""터미널에서 질문하기. 단계별 판정을 그대로 출력한다.

  python -m src.query "CH334U 동작 전압"
  python -m src.query --model anthropic:claude-fable-5-1 "CH334R 패키지"
  python -m src.query --no-llm "CH334U HDMI 2.1 지원"     # 검색 단계까지만
"""

from __future__ import annotations

import argparse
import sys

from config import ANSWER_MODEL
from src.pipeline import DatasheetSearch, Trace
from src.prompt import NOT_FOUND_ANSWER, SYSTEM_PROMPT, build_context  # noqa: F401  (재노출)


def print_trace(trace: Trace) -> None:
    s = trace.sufficiency
    print(f"\n[1] 품번        {trace.target or '없음'}"
          + (" (특정 변종)" if trace.target_specific else ""))
    print(f"[2] 변종 게이트  {trace.gate_allowed}/{trace.total_chunks} 청크 통과, "
          f"{trace.gate_excluded} 청크 제외")
    print(f"[3] 검색 질의    {trace.lexical_query.strip()}")
    for rank, e in enumerate(trace.evidence, start=1):
        mark = " *품번 명시" if e.names_target else ""
        print(f"    {rank}. p.{e.page:<3} BM25 {e.score:5.1f}  {e.section[:50]}{mark}")
    print(f"[4] 검증 게이트  {'실패 (문서에 해당 품번 없음)' if trace.verification_failed else '통과'}")
    print(f"[5] 충분성       {'통과' if s.passed else '차단'}  {s.reason}")
    if trace.completion:
        c = trace.completion
        print(f"[6] 모델 호출    {c.model}  입력 {c.input_tokens} / 출력 {c.output_tokens} 토큰, "
              f"{c.latency_s:.1f}초")
    else:
        print("[6] 모델 호출    없음")
    if trace.error:
        print(f"    오류: {trace.error}")
    print("\n[답변]\n" + trace.answer)
    if trace.check:
        print(f"\n[원문 대조] {'통과' if trace.check.passed else '확인 필요'}  {trace.check.summary()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CH334/CH335 데이터시트 검색")
    parser.add_argument("question", nargs="*", help="질문")
    parser.add_argument("--model", default=ANSWER_MODEL, help="제공자:모델ID")
    parser.add_argument("--no-llm", action="store_true", help="모델을 부르지 않고 검색까지만")
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = parse_args()
    question = " ".join(args.question).strip()
    if not question:
        question = input("질문을 입력하세요:\n\n> ").strip()
    search = DatasheetSearch()
    try:
        trace = search.retrieve(question) if args.no_llm else search.answer(question, args.model)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    if args.no_llm and not trace.blocked:
        trace.answer = "(검색 단계까지만 실행)"
    elif args.no_llm:
        trace.answer = NOT_FOUND_ANSWER
    print_trace(trace)


if __name__ == "__main__":
    main()
