#!/usr/bin/env bash
# RAG 평가 실행 헬퍼.
#
# 사용법:
#   ./scripts/run_evaluation.sh               # 검색 메트릭만 빠르게
#   ./scripts/run_evaluation.sh --rerank      # reranker 켠 채로
#   ./scripts/run_evaluation.sh --full        # 답변 생성 + RAGAS까지
#
# 결과:
#   evals/history/{timestamp}.csv (러너 산출)
#   evals/reports/{timestamp}.md  (리포트 산출)
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d_%H%M%S)"
CSV_PATH="evals/history/${STAMP}.csv"
REPORT_PATH="evals/reports/${STAMP}.md"

mkdir -p evals/history evals/reports

RUNNER_ARGS=(
  "--golden" "evals/golden_set.jsonl"
  "--out" "${CSV_PATH}"
)

for arg in "$@"; do
  case "$arg" in
    --full)
      RUNNER_ARGS+=("--with-answer" "--with-ragas")
      ;;
    *)
      RUNNER_ARGS+=("$arg")
      ;;
  esac
done

uv run python -m evals.ragas_runner "${RUNNER_ARGS[@]}"
uv run python -m evals.report --current "${CSV_PATH}" --out "${REPORT_PATH}"

echo
echo "csv:    ${CSV_PATH}"
echo "report: ${REPORT_PATH}"
