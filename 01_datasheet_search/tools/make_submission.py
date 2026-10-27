"""제출용 ZIP 을 만든다. 비밀값이 들어가면 만들지 않고 실패한다.

  python tools/make_submission.py [ZIP 파일 이름]

ZIP 구조
  <TOP>/README.md                       docs/submission_README.md
  <TOP>/시연.html                        docs/demo.html (tools/export_demo.py 로 새로 만든다)
  <TOP>/발표자료.pptx, 발표자료.pdf        docs/datasheet_search.pptx, .pdf
  <TOP>/01_datasheet_search/...         이 저장소
  <TOP>/02_idec_project/README.md       docs/idec_README.md 와 그림 한 장
  <TOP>/03_local_docs/index.html        홈, 캘린더, IDEC, 강의정리를 묶은 한 파일 (tools/export_docs.py)
"""

from __future__ import annotations

import base64
import io
import re
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
DOCS = Path("C:/modulino_hub/schedule")
SINGLE = ROOT / "build" / "docs_single" / "index.html"
# 02 README. 작업 항목 그림을 README 안에 넣은 것을 build/ 에 만들어 담는다
IDEC_README = ROOT / "docs" / "idec_README.md"
IDEC_IMG = ROOT / "docs" / "idec_img" / "workitem_example.png"
IDEC_README_OUT = ROOT / "build" / "idec_README.md"
# 3부 한 파일을 만드는 스크립트. 개인 자료 경로를 읽으므로 ZIP 에는 넣지 않는다
PART3_BUILDERS = ["tools/export_study.py", "tools/export_docs.py"]
# 제출물을 만드는 데만 쓰는 스크립트 (02 README 그림)
NOT_SHIPPED = ["tools/make_workitem_png.py"]
OUT_DIR = ROOT / "submission"
DEFAULT_NAME = "LGD_AX_허윤.zip"
TOP = "LGD_AX_허윤"
PART1 = "01_datasheet_search"
PART2 = "02_idec_project"
PART3 = "03_local_docs"
MAX_MB = 200

INCLUDE = [
    "README.md",
    "app.py",
    "config.py",
    "evaluate.py",
    "calibrate.py",
    "evaluate_answers.py",
    "requirements.txt",
    "requirements-optional.txt",
    "run_demo.bat",
    ".env.example",
    ".streamlit/config.toml",
    "data/README.md",
    "data/CH334_CH335_V2.5.pdf",
    "data/datasheets/*.pdf",
    "data/catalog.json",
    "data/modulino_BOM_cost_estimate.md",
    "src/*.py",
    "tests/*.py",
    "tests/*.json",
    "tools/*.py",
    "results/*",
    "docs/img/*",
    "docs/EXTENDING.md",
    "evaluate_catalog.py",
    "tools/demo_template.html",
]

FORBIDDEN_NAMES = {".env"}
FORBIDDEN_PARTS = {".venv", ".git", "__pycache__", "vector_db", ".pytest_cache", "submission", "node_modules"}
SECRET_PATTERNS = [
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),        # OpenAI
    re.compile(rb"sk-ant-[A-Za-z0-9_-]{20,}"),    # Anthropic
]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".toml", ".bat", ".example", ".net", ".asc", ".js", ".html", ""}


def glob(base: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(p for p in base.glob(pattern) if p.is_file())
        if not matches and "*" not in pattern:
            raise SystemExit(f"필수 파일이 없습니다: {base / pattern}")
        files.extend(matches)
    return list(dict.fromkeys(files))


def entries() -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = [
        (ROOT / "docs" / "submission_README.md", f"{TOP}/README.md"),
        (ROOT / "docs" / "demo.html", f"{TOP}/시연.html"),
        (ROOT / "docs" / "datasheet_search.pptx", f"{TOP}/발표자료.pptx"),
        (ROOT / "docs" / "datasheet_search.pdf", f"{TOP}/발표자료.pdf"),
        (IDEC_README_OUT, f"{TOP}/{PART2}/README.md"),
        (ROOT / "docs" / "docs_README.md", f"{TOP}/{PART3}/README.md"),
    ]
    for path, _ in items:
        if not path.exists():
            raise SystemExit(f"필수 파일이 없습니다: {path}")
    items += [(p, f"{TOP}/{PART1}/{p.relative_to(ROOT).as_posix()}") for p in glob(ROOT, INCLUDE)
              if p.relative_to(ROOT).as_posix() not in PART3_BUILDERS + NOT_SHIPPED]
    # 3부는 홈, 캘린더, IDEC, 강의정리를 묶은 HTML 한 파일 (tools/export_docs.py)
    items.append((SINGLE, f"{TOP}/{PART3}/index.html"))
    return items


def build_docs_excerpt() -> None:
    """3부 발췌본을 새로 뽑는다. 빼기로 한 내용이 남으면 build_ax.py 가 실패하고 ZIP 도 만들지 않는다."""
    # -I 는 PYTHON* 환경변수를 무시하므로 출력 인코딩은 -X utf8 로 정한다
    result = subprocess.run([sys.executable, "-X", "utf8", "-I", str(DOCS / "build_ax.py")], capture_output=True)
    print(result.stdout.decode("utf-8", "replace").strip())
    if result.returncode:
        raise SystemExit("3부 발췌본 검사 실패. ZIP 을 만들지 않습니다.")


def build_docs_single() -> None:
    """3부 한 파일을 새로 만든다. 강의정리 제출본을 뽑고 홈, 캘린더, IDEC 과 함께 묶는다. 하나라도 실패하면 ZIP 을 만들지 않는다."""
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    python = str(venv) if venv.exists() else sys.executable
    for script in PART3_BUILDERS:
        result = subprocess.run([python, "-X", "utf8", str(ROOT / script)], capture_output=True, cwd=ROOT)
        print(result.stdout.decode("utf-8", "replace").strip())
        if result.returncode:
            raise SystemExit(f"{script} 실패. ZIP 을 만들지 않습니다.")


def build_idec_readme() -> None:
    """02 README 에 그림을 넣는다. 그림 파일을 따로 열지 않아도 README 에서 보이도록
    data URI 로 바꿔 파일 맨 아래 참조로 붙인다 (본문 중간에 긴 문자열이 끼지 않게)."""
    text = IDEC_README.read_text(encoding="utf-8")
    if "](workitem_example.png)" not in text:
        raise SystemExit("02 README 의 그림 자리를 찾지 못했습니다")
    data = IDEC_IMG.read_bytes()
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(data)).convert("RGB")
        # 2배로 찍은 그림이라 반으로 줄인다. 그림 크기를 맞춰 주지 않는 뷰어에서도 화면 폭에 들어가게
        image = image.resize((image.width // 2, image.height // 2), Image.LANCZOS).quantize(colors=128)
        buffer = io.BytesIO()
        image.save(buffer, "PNG", optimize=True)
        data = buffer.getvalue()
    except ImportError:
        pass
    text = text.replace("](workitem_example.png)", "][workitem]")
    text = text.rstrip("\n") + "\n\n[workitem]: data:image/png;base64," + base64.b64encode(data).decode() + "\n"
    IDEC_README_OUT.parent.mkdir(exist_ok=True)
    IDEC_README_OUT.write_text(text, encoding="utf-8")
    print(f"{IDEC_README_OUT}  (그림 {len(data) / 1024:.0f} KB 를 README 안에 넣음)")


def build_demo() -> None:
    """시연.html 을 새로 만든다. 검색 결과가 저장된 평가 기록과 어긋나면 export_demo.py 가 실패하고 ZIP 도 만들지 않는다."""
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    python = str(venv) if venv.exists() else sys.executable
    result = subprocess.run([python, "-X", "utf8", str(ROOT / "tools" / "export_demo.py")], capture_output=True, cwd=ROOT)
    print(result.stdout.decode("utf-8", "replace").strip())
    if result.returncode:
        print(result.stderr.decode("utf-8", "replace").strip())
        raise SystemExit("시연 기록 생성 실패. ZIP 을 만들지 않습니다.")


def check(items: list[tuple[Path, str]]) -> None:
    problems = []
    for path, arc in items:
        if path.name in FORBIDDEN_NAMES or FORBIDDEN_PARTS & set(Path(arc).parts):
            problems.append(f"제외 대상 포함: {arc}")
            continue
        if path.suffix in TEXT_SUFFIXES or path.name == ".env.example":
            data = path.read_bytes()
            for pattern in SECRET_PATTERNS:
                if pattern.search(data):
                    problems.append(f"API 키로 보이는 문자열: {arc}")
    if problems:
        raise SystemExit("ZIP 을 만들지 않습니다.\n  " + "\n  ".join(problems))


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NAME
    build_docs_excerpt()
    build_demo()
    build_docs_single()
    build_idec_readme()
    items = entries()
    check(items)
    OUT_DIR.mkdir(exist_ok=True)
    target = OUT_DIR / name
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, arc in items:
            archive.write(path, arc)
    size = target.stat().st_size / 1024 / 1024
    print(f"{target}  ({len(items)}개 파일, {size:.1f} MB)")
    if size > MAX_MB:
        raise SystemExit(f"제출 한도 {MAX_MB} MB 를 넘습니다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
