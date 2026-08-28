#!/usr/bin/env bash
#SBATCH --job-name=aita-eval
#SBATCH --partition=mlgpu_medium
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-89%2
#SBATCH --output=value_alignment/slurm_logs/%x_%A_%a.out
#SBATCH --error=value_alignment/slurm_logs/%x_%A_%a.err

set -Eeuo pipefail

MODELS=(qwen3-8b falcon3-7b llama3.1-8b)
METHODS=(sft dpo hypo)
VALUES=(self_direction stimulation hedonism achievement power security conformity tradition benevolence universalism)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_COUNT=$((${#MODELS[@]} * ${#METHODS[@]} * ${#VALUES[@]}))

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
# shellcheck disable=SC1091
source "$REPO_ROOT/value_alignment/slurm/_common.sh"
validate_array_index "$TASK_ID" "$TASK_COUNT"
setup_value_alignment_env

TASK_INDEX=$((10#$TASK_ID))
VALUE_INDEX=$((TASK_INDEX % ${#VALUES[@]}))
METHOD_INDEX=$(((TASK_INDEX / ${#VALUES[@]}) % ${#METHODS[@]}))
MODEL_INDEX=$((TASK_INDEX / (${#VALUES[@]} * ${#METHODS[@]})))
MODEL="${MODEL_OVERRIDE:-${MODELS[$MODEL_INDEX]}}"
METHOD="${METHOD_OVERRIDE:-${METHODS[$METHOD_INDEX]}}"
TARGET="${VALUE_OVERRIDE:-${VALUES[$VALUE_INDEX]}}"

BASE_ADAPTER=""
if [[ "$METHOD" == "sft" ]]; then
  BASE_ADAPTER="${SFT_CHECKPOINT_ROOT:-value_alignment/checkpoints/paper_sft}/$MODEL/baseline/final"
  CONDITIONED_ADAPTER="${SFT_CHECKPOINT_ROOT:-value_alignment/checkpoints/paper_sft}/$MODEL/$TARGET/final"
else
  CONDITIONED_ADAPTER="${PREFERENCE_CHECKPOINT_ROOT:-value_alignment/checkpoints/paper_preference}/$MODEL/$TARGET/$METHOD/final"
fi

TEST_FILE="${TEST_FILE:-value_alignment/data/aita_eval/test.jsonl}"
OUTPUT_DIR="${OUTPUT_ROOT:-value_alignment/results/paper/aita}/$MODEL/$METHOD"
if [[ ! -f "$TEST_FILE" ]]; then
  echo "Missing AITA test file: $TEST_FILE" >&2
  echo "Run python -m value_alignment.prepare_aita_eval first." >&2
  exit 1
fi
for ADAPTER in "$BASE_ADAPTER" "$CONDITIONED_ADAPTER"; do
  if [[ -n "$ADAPTER" && ! -f "$ADAPTER/adapter_config.json" ]]; then
    echo "Missing PEFT adapter: $ADAPTER/adapter_config.json" >&2
    exit 1
  fi
done
mkdir -p "$OUTPUT_DIR"

echo "Evaluating AITA: model=$MODEL method=$METHOD target=$TARGET"
check_gpu_environment

CMD=(
  python -u -m value_alignment.evaluation.evaluate_aita_probability_gain
  --base-model "$MODEL"
  --conditioned-model "$MODEL"
  --conditioned-adapter "$CONDITIONED_ADAPTER"
  --target-value "$TARGET"
  --test-file "$TEST_FILE"
  --output-json "$OUTPUT_DIR/$TARGET.json"
  --output-csv "$OUTPUT_DIR/$TARGET.csv"
)
if [[ -n "$BASE_ADAPTER" ]]; then
  CMD+=(--base-adapter "$BASE_ADAPTER")
fi
"${CMD[@]}"
