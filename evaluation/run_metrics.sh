#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 PREDICTIONS.jsonl [PREDICTIONS.jsonl ...]" >&2
  echo "Set LLM_JUDGE=1 to enable the OpenAI judge." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METRICS_DIR="${METRICS_DIR:-metrics}"
JUDGED_DIR="${JUDGED_DIR:-predictions_judged}"
LLM_JUDGE="${LLM_JUDGE:-0}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-nano}"

mkdir -p "$METRICS_DIR"
if [[ "$LLM_JUDGE" == "1" ]]; then
  mkdir -p "$JUDGED_DIR"
fi

for pred_jsonl in "$@"; do
  if [[ ! -f "$pred_jsonl" ]]; then
    echo "[warn] missing file, skipping: $pred_jsonl" >&2
    continue
  fi

  base="$(basename "$pred_jsonl" .jsonl)"
  base="${base%.merged}"
  summary_json="$METRICS_DIR/${base}_metrics.json"

  args=(
    --pred_jsonl "$pred_jsonl"
    --save_summary_json "$summary_json"
    --group_by category broad_category open_split merged_specialty specialty surgery_type question_category
  )

  if [[ "$LLM_JUDGE" == "1" ]]; then
    args+=(
      --llm_judge
      --judge_model "$JUDGE_MODEL"
      --judged_jsonl "$JUDGED_DIR/${base}_judged.jsonl"
    )
  fi

  echo "Computing metrics for $pred_jsonl"
  python "$SCRIPT_DIR/compute_eval_metrics.py" "${args[@]}"
done
