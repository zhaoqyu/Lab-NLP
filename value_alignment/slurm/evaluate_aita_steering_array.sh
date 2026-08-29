#!/usr/bin/env bash
#SBATCH --job-name=aita-steer
#SBATCH --partition=mlgpu_medium
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-59%2
#SBATCH --output=value_alignment/slurm_logs/%x_%A_%a.out
#SBATCH --error=value_alignment/slurm_logs/%x_%A_%a.err

set -Eeuo pipefail

MODELS=(qwen3-8b falcon3-7b llama3.1-8b)
SITES=(attn block)
VALUES=(self_direction stimulation hedonism achievement power security conformity tradition benevolence universalism)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_COUNT=$((${#MODELS[@]} * ${#SITES[@]} * ${#VALUES[@]}))
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
# shellcheck disable=SC1091
source "$REPO_ROOT/value_alignment/slurm/_common.sh"
validate_array_index "$TASK_ID" "$TASK_COUNT"
setup_value_alignment_env

TASK_INDEX=$((10#$TASK_ID))
VALUE_INDEX=$((TASK_INDEX % ${#VALUES[@]}))
SITE_INDEX=$(((TASK_INDEX / ${#VALUES[@]}) % ${#SITES[@]}))
MODEL_INDEX=$((TASK_INDEX / (${#VALUES[@]} * ${#SITES[@]})))
MODEL="${MODEL_OVERRIDE:-${MODELS[$MODEL_INDEX]}}"
SITE="${SITE_OVERRIDE:-${SITES[$SITE_INDEX]}}"
TARGET="${VALUE_OVERRIDE:-${VALUES[$VALUE_INDEX]}}"
SELECTION="${SELECTION_ROOT:-value_alignment/results/paper/steering_selection}/$MODEL/$TARGET/$SITE.json"
TEST_FILE="${TEST_FILE:-value_alignment/data/aita_eval/test.jsonl}"
OUTPUT_DIR="${OUTPUT_ROOT:-value_alignment/results/paper/aita}/$MODEL/steering_$SITE"

for FILE in "$SELECTION" "$TEST_FILE"; do
  if [[ ! -f "$FILE" ]]; then
    echo "Missing steering evaluation input: $FILE" >&2
    exit 1
  fi
done
mkdir -p "$OUTPUT_DIR"
echo "Evaluating AITA steering: model=$MODEL target=$TARGET site=$SITE"
check_gpu_environment
python -u -m value_alignment.evaluation.evaluate_aita_probability_gain \
  --base-model "$MODEL" \
  --conditioned-model "$MODEL" \
  --conditioned-steering-selection "$SELECTION" \
  --target-value "$TARGET" \
  --test-file "$TEST_FILE" \
  --output-json "$OUTPUT_DIR/$TARGET.json" \
  --output-csv "$OUTPUT_DIR/$TARGET.csv"
