"""평가셋 정답을 데이터시트 원문으로 검수한다.

정답형 문항마다
  ① expected_pages 의 원문에 정답 사실(must_include)이 실제로 있는가
  ② 질문한 품번이 그 페이지에 있는가 (변종 문항)
문서에 없음 문항마다
  ③ 질문의 핵심 용어가 문서 전체 어디에도 없는가

모델을 부르지 않는다. 정답을 사람이 원문에서 옮겨 적었으므로, 옮겨 적은 값이
원문과 맞는지를 기계적으로 다시 본다.

  python tools/verify_evalset.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import EVALSET_PATH, PDF_PATH  # noqa: E402
from src.ingest import extract_pages  # noqa: E402
from src.variants import extract_parts  # noqa: E402

# 답변은 한국어로 나오지만 원문은 영문이다. 한국어 정답 표현은 원문 표현으로 바꿔 확인한다.
SOURCE_FORMS = {
    "지원하지 않": "does not support",
    "지원되지 않": "does not support",
    "미지원": "does not support",
    "지원 안": "does not support",
    "없": "does not have",
    "권장하지 않": "not recommended",
    "권장되지 않": "not recommended",
    "비권장": "not recommended",
    "독립": "independent",
    "병렬": "parallel",
    "동시": "concurrent",
    "각 포트": "each",
    "포트마다": "each",
    "포트별": "each",
    "리피터": "repeater",
    "반복기": "repeater",
    "4마이크로": "4us",
    "14밀리": "14ms",
    "1a86": "1ah",
}

# 문서에 없음 문항의 핵심 용어 (원문 전체에 없어야 한다)
ABSENT_TERMS = {
    "N01": ["CH999Z"], "N02": ["HDMI"], "N03": ["Thunderbolt"], "N04": ["price", "$"],
    "N05": ["ADC"], "N06": ["Bluetooth"], "N07": ["MTBF"], "N08": ["PCIe"],
    "N09": ["CH335Z"], "N10": ["AEC", "Q100"],
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower().replace("μ", "u").replace("µ", "u").replace("ω", "")


def main() -> int:
    pages = {p.page: p.text for p in extract_pages(PDF_PATH)}
    whole = "\n".join(pages.values())
    items = json.loads(EVALSET_PATH.read_text(encoding="utf-8"))
    problems = 0
    for item in items:
        if item["type"] == "not_in_doc":
            found = [t for t in ABSENT_TERMS.get(item["id"], []) if t.lower() in whole.lower()]
            ok = not found
            print(f"{item['id']} {'OK ' if ok else 'NG '} 원문에 없음 확인"
                  + (f"  (원문에 있음: {found})" if found else ""))
            problems += not ok
            continue

        text = normalize("\n".join(pages[p] for p in item["expected_pages"]))
        notes = []
        for group in item["must_include"]:
            forms = [normalize(SOURCE_FORMS.get(o, o)) for o in group]
            if not any(form in text for form in forms):
                notes.append(f"원문에 없음: {'/'.join(group)}")
        # 공통 사양 문항(A01, A04 등)은 정답 페이지에 품번이 없는 것이 정상이다. 변종 문항만 본다
        for part in extract_parts(item["question"]) if item["type"] == "variant" else []:
            if part.is_specific and part.raw.lower() not in text:
                # CH334U/S/F/Q 처럼 줄여 쓴 경우는 계열명과 접미사로 다시 본다
                short = f"/{part.suffix.lower()}"
                if not (part.family.lower() in text and short in text):
                    notes.append(f"페이지에 품번 없음: {part.raw}")
        ok = not notes
        problems += not ok
        print(f"{item['id']} {'OK ' if ok else 'NG '} p.{item['expected_pages']}"
              + (f"  {'; '.join(notes)}" if notes else ""))
    print(f"\n{len(items)}문항 중 확인 필요 {problems}건")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
