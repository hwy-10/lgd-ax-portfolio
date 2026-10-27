"""발표자료에 넣을 시연 화면 캡처.  python tools/demo_strips.py

docs/demo.html (제출 ZIP 의 시연.html) 을 열어 시나리오 결과의 일부 단계만 남기고 찍는다.
결과: docs/img/demo_*.png 와 크기를 적은 .json (build_deck.js 가 읽는다)
"""

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"

# (파일, 범위, 시나리오 버튼 순번, 남길 단계 번호, 남길 문단 수)
SHOTS = [
    ("demo_answer.png", "single", 1, ["5", "6"], 2),
    ("demo_blocked.png", "single", 4, ["5", "6"], 0),
    ("demo_catalog_u5.png", "catalog", 1, ["0", "1", "2"], 1),
]

KEEP = """([steps, metas]) => {
  const r = document.getElementById("result");
  let m = 0;
  for (const el of r.children) {
    let keep = false;
    if (el.classList.contains("step")) keep = steps.includes(el.querySelector(".k b").textContent.split(" ")[0]);
    else if (el.classList.contains("ans")) keep = true;
    else if (el.tagName === "P" && el.classList.contains("meta")) keep = m++ < metas;
    el.style.display = keep ? "" : "none";
  }
}"""


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1000, "height": 900}, device_scale_factor=2)
        for name, scope, n, steps, metas in SHOTS:
            page.goto((ROOT / "docs" / "demo.html").as_uri())
            page.click(f"#scope button[data-scope={scope}]")
            page.click(f"#scenarios button:nth-child({n})")
            page.evaluate(KEEP, [steps, metas])
            out = IMG / name
            page.locator("#result").screenshot(path=str(out))
            from PIL import Image
            w, h = Image.open(out).size
            (IMG / (name + ".json")).write_text(json.dumps({"w": w, "h": h}), encoding="utf-8")
            print(out, w, h)
        browser.close()


if __name__ == "__main__":
    main()
