"""발표용 차트. results/threshold_measure.json 을 읽어 PNG 를 만든다.

  python tools/make_charts.py

한 그림에 두 기준을 나란히 놓는다. 왼쪽은 BM25 1위 점수, 오른쪽은 용어 등장 비율.
점수로는 두 집단이 겹치고 용어 비율로는 갈린다는 것을 보여 주는 것이 목적이다.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import EVIDENCE_MIN_COVERAGE, EVIDENCE_MIN_SCORE, RESULTS_DIR  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e0"
ANSWERABLE = "#2a78d6"   # 범주 1
ABSENT = "#eb6834"       # 범주 2

plt.rcParams.update(
    {
        "font.family": "Malgun Gothic",
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "font.size": 12,
    }
)


def strip(ax, rows, key, threshold, xlabel, xlim, annotate):
    rng = random.Random(7)
    groups = [("정답이 있는 질문", "answerable", ANSWERABLE, "o", 1.0),
              ("문서에 없는 질문", "absent", ABSENT, "X", 0.0)]
    for label, kind, color, marker, y in groups:
        values = [r[key] for r in rows if r["kind"] == kind]
        ys = [y + rng.uniform(-0.18, 0.18) for _ in values]
        ax.scatter(values, ys, s=70, c=color, marker=marker, edgecolors=SURFACE,
                   linewidths=1.5, zorder=3, label=f"{label} ({len(values)})")
    ax.axvline(threshold, color=TEXT, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
    ax.text(threshold, 1.42, f" 기준 {threshold:g}", color=TEXT, fontsize=11, va="bottom")
    for row, text, dx, dy in annotate:
        ax.annotate(text, (row[key], 0.0), xytext=(row[key] + dx, dy), fontsize=10.5,
                    color=TEXT_2, arrowprops={"arrowstyle": "-", "color": TEXT_2, "lw": 0.8})
    ax.set_yticks([0, 1], ["문서에 없음", "정답 있음"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> int:
    measured = json.loads((RESULTS_DIR / "threshold_measure.json").read_text(encoding="utf-8"))
    rows = []
    for r in measured:
        if r["gate_blocked"]:
            continue  # 품번 검증 게이트에서 먼저 막힌 질문은 기준값과 무관
        rows.append({**r, "kind": "absent" if r["type"] == "not_in_doc" else "answerable"})
    gate_blocked = sum(1 for r in measured if r["gate_blocked"])
    by_id = {r["id"]: r for r in rows}

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"wspace": 0.18})
    strip(left, rows, "top_score", EVIDENCE_MIN_SCORE, "BM25 1위 점수", (-1, 55),
          [(by_id["N10"], "AEC-Q100 인증 (12.5)", 6, -0.5)] if "N10" in by_id else [])
    strip(right, rows, "coverage", EVIDENCE_MIN_COVERAGE, "질문 용어가 근거에 등장한 비율",
          (-0.05, 1.08),
          [(by_id["N05"], "내장 ADC 분해능 (0.33)", -0.28, -0.5)] if "N05" in by_id else [])
    right.set_yticklabels([])
    left.set_title("점수로는 겹친다", loc="left", color=TEXT, fontsize=14, fontweight="bold")
    right.set_title("용어 비율로는 갈린다", loc="left", color=TEXT, fontsize=14, fontweight="bold")
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2, fontsize=11,
               labelcolor=TEXT_2)
    fig.text(0.125, -0.04,
             f"평가셋 38문항. 품번 검증 게이트에서 먼저 막힌 {gate_blocked}문항(CH999Z, CH335Z)은 제외. "
             "같은 값의 점은 세로로 흩어 표시.",
             fontsize=10, color=TEXT_2)
    out = RESULTS_DIR / "threshold_chart.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
