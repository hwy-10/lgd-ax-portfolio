"""시연 화면 캡처 (발표 자료용). 실행 중인 streamlit 에 접속해 시나리오별로 찍는다.

  streamlit run app.py --server.port 8599
  python tools/screenshot.py

설치된 Chrome 을 쓴다 (playwright 의 브라우저 내려받기 불필요).
"""

from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8599"

SHOTS = [
    ("scenario1_common", "CH334U의 동작 전압(VDD33)은?"),
    ("scenario2_variant", "CH334R의 패키지는?"),
    ("scenario3_unknown_part", "CH999Z의 동작 전압은?"),
    ("scenario4_absent_spec", "CH334U는 HDMI 2.1을 지원하는가?"),
]


def crop(page, out: Path, scale: float = 1.5) -> None:
    """질문 입력칸부터 답변(없으면 6단계)까지만 잘라 발표 자료용으로 남긴다."""
    main = page.locator('[data-testid="stMainBlockContainer"]').bounding_box()
    top = page.get_by_text("질문", exact=True).first.bounding_box()["y"] - 8
    evidence = page.get_by_text("근거 원문", exact=True)
    if evidence.count():
        bottom = evidence.first.bounding_box()["y"] - 12
    else:
        # 근거가 없으면 답변(거절 문장)이 마지막 요소다
        refusal = page.get_by_text("제공된 데이터시트에서 해당 내용을")
        target = refusal.last if refusal.count() else page.get_by_text("모델 호출").last
        box = target.bounding_box()
        bottom = box["y"] + box["height"] + 24
    clip = {"x": main["x"] + 40, "y": top, "width": main["width"] - 80, "height": bottom - top}
    page.screenshot(path=str(out), clip=clip)
    size = {"w": round(clip["width"] * scale), "h": round(clip["height"] * scale)}
    out.with_suffix(".png.json").write_text(json.dumps(size), encoding="utf-8")


def strip(page, out: Path, end_text: str | None, scale: float = 1.5) -> None:
    """'5 근거 충분성' 카드부터 답변 끝까지 가로로 길게 잘라 발표 자료에 크게 넣는다."""
    main = page.locator('[data-testid="stMainBlockContainer"]').bounding_box()
    top = page.get_by_text("5 근거 충분성").first.bounding_box()["y"] - 24
    target = page.get_by_text(end_text).last if end_text and page.get_by_text(end_text).count() else None
    if target is None:
        refusal = page.get_by_text("제공된 데이터시트에서 해당 내용을")
        target = refusal.last if refusal.count() else page.get_by_text("모델 호출").last
    box = target.bounding_box()
    bottom = box["y"] + box["height"] + 24
    clip = {"x": main["x"] + 40, "y": top, "width": main["width"] - 80, "height": bottom - top}
    page.screenshot(path=str(out), clip=clip)
    size = {"w": round(clip["width"] * scale), "h": round(clip["height"] * scale)}
    out.with_suffix(".png.json").write_text(json.dumps(size), encoding="utf-8")


STRIPS = {
    "scenario1_common": ("strip_answer.png", "원문 대조"),
    "scenario4_absent_spec": ("strip_blocked.png", None),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        # streamlit 은 본문이 안쪽 컨테이너에서 스크롤되어 full_page 가 안 먹는다. 창을 길게 연다
        page = browser.new_page(viewport={"width": 1440, "height": 2400}, device_scale_factor=1.5)
        for name, question in SHOTS:
            page.goto(f"{BASE}/?q={urllib.parse.quote(question)}")
            page.get_by_text("근거 충분성").first.wait_for(timeout=60000)
            page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
            crop(page, OUT / f"{name}_crop.png")
            if name in STRIPS:
                file, end_text = STRIPS[name]
                strip(page, OUT / file, end_text)
            print(name)
        page.goto(BASE)
        page.get_by_role("tab", name="검증 결과").click()
        page.get_by_text("전부 통과").first.wait_for(timeout=60000)
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / "results_tab.png"), full_page=True)
        print("results_tab")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
