"""제출용 ZIP 을 만든다. 비밀값이 들어가면 만들지 않고 실패한다.

  python tools/make_submission.py [ZIP 파일 이름] [--full]

ZIP 구조
  <TOP>/README.md                       docs/submission_README.md
  <TOP>/시연.html                        docs/demo.html (tools/export_demo.py 로 새로 만든다)
  <TOP>/01_datasheet_search/...         핵심 코드만 (PART1_CORE). 실행은 GitHub 저장소에서
                                        --full 이면 이 저장소 전체 (INCLUDE, GitHub 공개본용)
  <TOP>/02_idec_project/README.md       docs/idec_README.md 와 그림 한 장
  <TOP>/03_local_docs/index.html        홈, 캘린더, IDEC, 강의정리를 묶은 한 파일 (tools/export_docs.py)
"""

from __future__ import annotations

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
# 02 README 와 작업 항목 그림
IDEC_README = ROOT / "docs" / "idec_README.md"
IDEC_IMG = ROOT / "docs" / "idec_img" / "workitem_example.png"
# 3부 한 파일을 만드는 스크립트. 개인 자료 경로를 읽으므로 ZIP 에는 넣지 않는다
PART3_BUILDERS = ["tools/export_study.py", "tools/export_docs.py"]
# 제출물을 만드는 데만 쓰는 스크립트 (02 README 그림)
NOT_SHIPPED = ["tools/make_workitem_png.py", "tools/demo_strips.py"]   # demo_strips 는 발표자료용이었다
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

# 제출 ZIP 의 01 은 판정 로직만 둔다. 전체 코드와 실행은 GitHub 저장소로 안내한다
PART1_README = ROOT / "docs" / "part1_README.md"
PART1_CORE = ["config.py", "evaluate.py", "src/*.py"]
# --full (GitHub 공개본) 에만 들어가는 파일: 맨 위 실행 파일, 데이터시트 출처 안내
ROOT_BAT = ROOT / "docs" / "root_run_demo.bat"
REPO_DATA_README = ROOT / "docs" / "repo_data_README.md"

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


def entries(full: bool = False) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = [
        (ROOT / "docs" / "submission_README.md", f"{TOP}/README.md"),
        (ROOT / "docs" / "demo.html", f"{TOP}/시연.html"),
        # 02 README 와 그 그림. 그림은 README 옆 파일로 둔다 (README 안에 글자로 넣으면
        # 마크다운을 그리지 않는 뷰어에서 긴 문자열로 보인다)
        (IDEC_README, f"{TOP}/{PART2}/README.md"),
        (IDEC_IMG, f"{TOP}/{PART2}/workitem_example.png"),
        (ROOT / "docs" / "docs_README.md", f"{TOP}/{PART3}/README.md"),
    ]
    if full:
        items += [(ROOT_BAT, f"{TOP}/run_demo.bat"), (REPO_DATA_README, f"{TOP}/{PART1}/data/README.md")]
    else:
        items.append((PART1_README, f"{TOP}/{PART1}/README.md"))
    for path, _ in items:
        if not path.exists():
            raise SystemExit(f"필수 파일이 없습니다: {path}")
    replaced = {arc for _, arc in items}
    items += [(p, arc) for p in glob(ROOT, INCLUDE if full else PART1_CORE)
              if p.relative_to(ROOT).as_posix() not in PART3_BUILDERS + NOT_SHIPPED
              and (arc := f"{TOP}/{PART1}/{p.relative_to(ROOT).as_posix()}") not in replaced]
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
    args = [a for a in sys.argv[1:] if a != "--full"]
    full = "--full" in sys.argv[1:]
    name = args[0] if args else DEFAULT_NAME
    build_docs_excerpt()
    build_demo()
    build_docs_single()
    items = entries(full)
    check(items)
    OUT_DIR.mkdir(exist_ok=True)
    target = OUT_DIR / name
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, arc in items:
            if full and arc == f"{TOP}/README.md":
                # 공개본에는 01 전체가 있으므로 구성 표의 설명만 바꾼다
                text = path.read_text(encoding="utf-8")
                old = "데이터시트 검색 도구의 핵심 코드 (전체 코드와 실행은 GitHub)"
                if old not in text:
                    raise SystemExit("README 구성 표 문구가 바뀌었습니다. make_submission.py 를 고쳐 주세요")
                archive.writestr(arc, text.replace(old, "데이터시트 검색 도구 (실행은 맨 위 `run_demo.bat`)"))
                continue
            archive.write(path, arc)
    size = target.stat().st_size / 1024 / 1024
    print(f"{target}  ({len(items)}개 파일, {size:.1f} MB)")
    if size > MAX_MB:
        raise SystemExit(f"제출 한도 {MAX_MB} MB 를 넘습니다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
