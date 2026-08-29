#!/usr/bin/env bash
#SBATCH --job-name=ref-mismatch
#SBATCH --partition=mlgpu_medium
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-29%2
#SBATCH --output=value_alignment/slurm_logs/%x_%A_%a.out
#SBATCH --error=value_alignment/slurm_logs/%x_%A_%a.err

set -Eeuo pipefail

MODELS=(qwen3-8b falcon3-7b llama3.1-8b)
VALUES=(self_direction stimulation hedonism achievement power security conformity tradition benevolence universalism)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_COUNT=$((${#MODELS[@]} * ${#VALUES[@]}))

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
# shellcheck disable=SC1091
source "$REPO_ROOT/value_alignment/slurm/_common.sh"
validate_array_index "$TASK_ID" "$TASK_COUNT"
setup_value_alignment_env

TASK_INDEX=$((10#$TASK_ID))
VALUE_INDEX=$((TASK_INDEX % ${#VALUES[@]}))
MODEL_INDEX=$((TASK_INDEX / ${#VALUES[@]}))
MODEL="${MODEL_OVERRIDE:-${MODELS[$MODEL_INDEX]}}"
TARGET="${VALUE_OVERRIDE:-${VALUES[$VALUE_INDEX]}}"
DATA_DIR="${DATA_ROOT:-value_alignment/data/paper_preferences}/$TARGET/down"
OUTPUT_DIR="${OUTPUT_ROOT:-value_alignment/results/paper/reference_mismatch}/$MODEL"

for FILE in "$DATA_DIR/train.jsonl" "$DATA_DIR/eval.jsonl"; do
  if [[ ! -f "$FILE" ]]; then
    echo "Missing preference dataset: $FILE" >&2
    exit 1
  fi
done
mkdir -p "$OUTPUT_DIR"

echo "Reference mismatch: model=$MODEL target=$TARGET"
check_gpu_environment

CMD=(
  python -u -m value_alignment.evaluation.analyze_reference_mismatch
  --model "$MODEL"
  --inputs "$DATA_DIR/train.jsonl" "$DATA_DIR/eval.jsonl"
  --output-json "$OUTPUT_DIR/$TARGET.json"
  --output-csv "$OUTPUT_DIR/$TARGET.csv"
  --batch-size "${BATCH_SIZE:-8}"
  --max-length "${MAX_LENGTH:-512}"
  --max-prompt-length "${MAX_PROMPT_LENGTH:-384}"
)
if [[ "${QLORA:-0}" == "1" ]]; then
  CMD+=(--load-in-4bit)
fi
"${CMD[@]}"
