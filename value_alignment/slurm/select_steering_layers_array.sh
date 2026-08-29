#!/usr/bin/env bash
#SBATCH --job-name=steer-select
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
VECTOR_FILE="${VECTOR_ROOT:-value_alignment/results/paper/steering_vectors}/$MODEL/$TARGET/vectors.pt"
OUTPUT="${OUTPUT_ROOT:-value_alignment/results/paper/steering_selection}/$MODEL/$TARGET/$SITE.json"

for FILE in "$VECTOR_FILE" "${EVAL_FILE:-value_alignment/data/kvs_survey/test.jsonl}"; do
  if [[ ! -f "$FILE" ]]; then
    echo "Missing steering prerequisite: $FILE" >&2
    exit 1
  fi
done
mkdir -p "$(dirname "$OUTPUT")"
echo "Selecting steering layer: model=$MODEL target=$TARGET site=$SITE"
check_gpu_environment

CMD=(
  python -u -m value_alignment.select_steering_layer
  --model "$MODEL"
  --target-value "$TARGET"
  --site "$SITE"
  --vector-file "$VECTOR_FILE"
  --eval-file "${EVAL_FILE:-value_alignment/data/kvs_survey/test.jsonl}"
  --output "$OUTPUT"
  --strength "${STRENGTH:-1.0}"
  --num-runs "${NUM_RUNS:-1}"
  --temperature "${TEMPERATURE:-0.5}"
  --batch-size "${BATCH_SIZE:-16}"
  --seed "${EVAL_SEED:-42}"
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi
if [[ -n "${LAYERS:-}" ]]; then
  read -r -a LAYER_LIST <<< "$LAYERS"
  CMD+=(--layers "${LAYER_LIST[@]}")
fi
"${CMD[@]}"
