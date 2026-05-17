"""평가 결과 리포트 생성기.

``evals/history/*.csv`` 를 읽어 카테고리별 평균/요약과 직전 실행 대비
diff를 Markdown 리포트로 만든다. CI에 붙이지 않더라도, 사람 손으로
PR 본문에 첨부하기 좋게 가공하는 것이 목적이다.

Examples:
    가장 최근 CSV 자동 선택::

        $ uv run python -m evals.report

    특정 두 CSV 비교::

        $ uv run python -m evals.report \\
            --current evals/history/with_rerank.csv \\
            --baseline evals/history/baseline.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

# 카테고리별 평균을 낼 메트릭 컬럼들. 존재하지 않으면 자동으로 건너뛴다.
METRIC_COLUMNS: tuple[str, ...] = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "precision_at_5",
    "mrr",
    "source_recall",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "latency_search_ms",
    "latency_answer_ms",
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """CSV를 dict 리스트로 읽는다.

    Args:
        path: 입력 CSV 경로.

    Returns:
        헤더를 키로 갖는 dict 리스트.

    Raises:
        FileNotFoundError: ``path`` 가 존재하지 않을 때.
    """
    if not path.exists():
        raise FileNotFoundError(f"csv not found: {path}")
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_float(value: Any) -> float | None:
    """문자열/None을 float로 변환한다. 변환 불가 시 ``None``.

    Args:
        value: 원본 값.

    Returns:
        ``float`` 또는 ``None``. 빈 문자열, ``"nan"``도 ``None`` 처리.
    """
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x


def _mean(values: Iterable[float | None]) -> float | None:
    """``None``을 제외한 평균. 모두 ``None``이면 ``None``.

    Args:
        values: 숫자 또는 ``None`` iterable.

    Returns:
        평균값 또는 ``None``.
    """
    nums = [v for v in values if v is not None]
    return statistics.fmean(nums) if nums else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """전체 평균과 카테고리별 평균을 계산한다.

    Args:
        rows: CSV에서 읽은 dict 리스트.

    Returns:
        ``{"overall": {metric: mean}, "by_category": {category: {metric: mean}},
        "n": 총 케이스 수}`` 구조의 dict.
    """
    available = [m for m in METRIC_COLUMNS if rows and m in rows[0]]

    overall = {m: _mean(_as_float(r.get(m)) for r in rows) for m in available}

    by_category: dict[str, dict[str, float | None]] = {}
    categories = {r.get("category", "") for r in rows}
    for cat in sorted(c for c in categories if c is not None):
        subset = [r for r in rows if r.get("category", "") == cat]
        by_category[cat or "(unset)"] = {
            m: _mean(_as_float(r.get(m)) for r in subset) for m in available
        }

    return {"overall": overall, "by_category": by_category, "n": len(rows)}


def _fmt(value: float | None) -> str:
    """리포트용 숫자 포맷. ``None``은 ``"-"``.

    Args:
        value: 표시할 값.

    Returns:
        문자열 표현. 메트릭은 소수 셋째 자리, latency는 정수.
    """
    if value is None:
        return "-"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _diff(curr: float | None, base: float | None) -> str:
    """현재 vs 베이스라인 차이를 부호 포함 문자열로.

    Args:
        curr: 현재 값.
        base: 비교 대상(베이스라인) 값.

    Returns:
        예: ``"+0.042"``, ``"-0.010"``, ``"-"``.
    """
    if curr is None or base is None:
        return "-"
    delta = curr - base
    sign = "+" if delta >= 0 else ""
    if abs(delta) >= 100:
        return f"{sign}{delta:.1f}"
    return f"{sign}{delta:.3f}"


def render_markdown(
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
    current_path: Path,
    baseline_path: Path | None,
) -> str:
    """집계 결과를 Markdown 리포트로 렌더링한다.

    Args:
        current: ``aggregate()`` 결과 (현재 실행).
        baseline: 비교 대상의 집계 결과. ``None``이면 비교 섹션 생략.
        current_path: 현재 CSV 경로(헤더 표기용).
        baseline_path: 베이스라인 CSV 경로.

    Returns:
        Markdown 문자열.
    """
    available = list(current["overall"].keys())
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    lines.append(f"# RAG 평가 리포트 — {now}")
    lines.append("")
    lines.append(f"- Current: `{current_path}` (n={current['n']})")
    if baseline_path:
        lines.append(f"- Baseline: `{baseline_path}` (n={baseline['n'] if baseline else 0})")
    lines.append("")

    # Overall
    lines.append("## Overall")
    lines.append("")
    if baseline:
        lines.append("| metric | current | baseline | Δ |")
        lines.append("|---|---|---|---|")
        for m in available:
            c = current["overall"].get(m)
            b = baseline["overall"].get(m)
            lines.append(f"| {m} | {_fmt(c)} | {_fmt(b)} | {_diff(c, b)} |")
    else:
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for m in available:
            lines.append(f"| {m} | {_fmt(current['overall'].get(m))} |")
    lines.append("")

    # By category
    lines.append("## By category")
    lines.append("")
    cats = sorted(current["by_category"].keys())
    if not cats:
        lines.append("_no rows_")
        return "\n".join(lines)

    header = ["category"] + available
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for cat in cats:
        row = [cat]
        for m in available:
            row.append(_fmt(current["by_category"][cat].get(m)))
        lines.append("| " + " | ".join(row) + " |")

    if baseline:
        lines.append("")
        lines.append("### Δ vs baseline (per category)")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for cat in cats:
            row = [cat]
            base_cat = baseline["by_category"].get(cat, {})
            for m in available:
                row.append(_diff(current["by_category"][cat].get(m), base_cat.get(m)))
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def _find_latest(history_dir: Path) -> Path | None:
    """``history_dir`` 에서 가장 최근 CSV 경로를 반환한다.

    Args:
        history_dir: ``evals/history`` 같은 디렉터리.

    Returns:
        mtime 기준 가장 최근 ``.csv`` 경로. 없으면 ``None``.
    """
    if not history_dir.exists():
        return None
    csvs = sorted(history_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    return csvs[-1] if csvs else None


def _find_previous(history_dir: Path, current: Path) -> Path | None:
    """``current`` 직전의 CSV 를 찾는다.

    Args:
        history_dir: 디렉터리.
        current: 기준 파일.

    Returns:
        직전 파일 또는 ``None``.
    """
    if not history_dir.exists():
        return None
    csvs = sorted(history_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    try:
        idx = csvs.index(current)
    except ValueError:
        return csvs[-1] if csvs and csvs[-1] != current else None
    return csvs[idx - 1] if idx > 0 else None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI 인자 파서.

    Args:
        argv: 인자 리스트.

    Returns:
        파싱된 ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        prog="evals.report",
        description="Render markdown report from evaluation CSVs.",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=None,
        help="현재 실행 CSV. 미지정 시 evals/history 내 최신 파일 자동 선택",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="비교 대상 CSV. 미지정 시 history 내 직전 파일 자동 선택",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="리포트 출력 경로 (.md). 미지정 시 evals/reports/{timestamp}.md",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("evals/history"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 엔트리포인트.

    Args:
        argv: 인자 리스트.

    Returns:
        성공 ``0``, 입력 누락 등 사용자 오류 ``2``.
    """
    args = _parse_args(argv)

    current = args.current or _find_latest(args.history_dir)
    if current is None:
        print(
            f"no CSV found in {args.history_dir}. run evals.ragas_runner first.",
            file=__import__("sys").stderr,
        )
        return 2

    baseline = args.baseline or _find_previous(args.history_dir, current)

    current_rows = _read_csv(current)
    baseline_rows = _read_csv(baseline) if baseline else None

    current_agg = aggregate(current_rows)
    baseline_agg = aggregate(baseline_rows) if baseline_rows is not None else None

    md = render_markdown(current_agg, baseline_agg, current, baseline)

    out_path = args.out
    if out_path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = Path("evals/reports") / f"{stamp}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote report → {out_path}")
    # 표준출력에도 즉시 확인 가능하도록 dump
    print()
    print(md)
    # json 평균 요약(머신 친화)도 stderr 로
    print(
        json.dumps(
            {"overall": current_agg["overall"]},
            ensure_ascii=False,
            indent=2,
        ),
        file=__import__("sys").stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
