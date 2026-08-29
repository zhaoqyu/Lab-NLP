#!/usr/bin/env bash
#SBATCH --job-name=aita-up-eval
#SBATCH --partition=mlgpu_medium
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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
CHECKPOINT_TAG="${CHECKPOINT_TAG:-${RUN_TAG:-}}"
RESULT_TAG="${RESULT_TAG:-${RUN_TAG:-}}"
validate_run_tag "$CHECKPOINT_TAG"
validate_run_tag "$RESULT_TAG"

BASE_ROOT="${SFT_CHECKPOINT_ROOT:-value_alignment/checkpoints/paper_sft}/$MODEL/baseline"
CONDITIONED_ROOT="${UP_CHECKPOINT_ROOT:-value_alignment/checkpoints/paper_sft_up}/$MODEL/$TARGET"
BASE_ADAPTER="$(tagged_run_path "$BASE_ROOT" "$CHECKPOINT_TAG")/final"
CONDITIONED_ADAPTER="$(tagged_run_path "$CONDITIONED_ROOT" "$CHECKPOINT_TAG")/final"
TEST_FILE="${TEST_FILE:-value_alignment/data/aita_eval/test.jsonl}"
OUTPUT_DIR="${OUTPUT_ROOT:-value_alignment/results/paper/aita_up}/$MODEL/sft_up"
OUTPUT_DIR="$(tagged_run_path "$OUTPUT_DIR" "$RESULT_TAG")"

for ADAPTER in "$BASE_ADAPTER" "$CONDITIONED_ADAPTER"; do
  if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
    echo "Missing PEFT adapter: $ADAPTER/adapter_config.json" >&2
    exit 1
  fi
done
if [[ ! -f "$TEST_FILE" ]]; then
  echo "Missing AITA test file: $TEST_FILE" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"

echo "Evaluating AITA up-regulation: model=$MODEL target=$TARGET"
check_gpu_environment
python -u -m value_alignment.evaluation.evaluate_aita_probability_gain \
  --base-model "$MODEL" \
  --conditioned-model "$MODEL" \
  --base-adapter "$BASE_ADAPTER" \
  --conditioned-adapter "$CONDITIONED_ADAPTER" \
  --target-value "$TARGET" \
  --test-file "$TEST_FILE" \
  --expected-direction less \
  --output-json "$OUTPUT_DIR/$TARGET.json" \
  --output-csv "$OUTPUT_DIR/$TARGET.csv"
