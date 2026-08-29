#!/usr/bin/env bash
#SBATCH --job-name=kvs-up-eval
#SBATCH --partition=mlgpu_medium
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-32%2
#SBATCH --output=value_alignment/slurm_logs/%x_%A_%a.out
#SBATCH --error=value_alignment/slurm_logs/%x_%A_%a.err

set -Eeuo pipefail

MODELS=(qwen3-8b falcon3-7b llama3.1-8b)
VALUES=(self_direction stimulation hedonism achievement power security conformity tradition benevolence universalism)
VARIANTS_PER_MODEL=11
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_COUNT=$((${#MODELS[@]} * VARIANTS_PER_MODEL))

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
# shellcheck disable=SC1091
source "$REPO_ROOT/value_alignment/slurm/_common.sh"
validate_array_index "$TASK_ID" "$TASK_COUNT"
setup_value_alignment_env

TASK_INDEX=$((10#$TASK_ID))
MODEL_INDEX=$((TASK_INDEX / VARIANTS_PER_MODEL))
VARIANT_INDEX=$((TASK_INDEX % VARIANTS_PER_MODEL))
MODEL="${MODEL_OVERRIDE:-${MODELS[$MODEL_INDEX]}}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-${RUN_TAG:-}}"
RESULT_TAG="${RESULT_TAG:-${RUN_TAG:-}}"
validate_run_tag "$CHECKPOINT_TAG"
validate_run_tag "$RESULT_TAG"

if (( VARIANT_INDEX == 0 )); then
  RUN_NAME="sft_baseline"
  ADAPTER_ROOT="${SFT_CHECKPOINT_ROOT:-value_alignment/checkpoints/paper_sft}/$MODEL/baseline"
else
  TARGET="${VALUES[$((VARIANT_INDEX - 1))]}"
  RUN_NAME="sft_up_$TARGET"
  ADAPTER_ROOT="${UP_CHECKPOINT_ROOT:-value_alignment/checkpoints/paper_sft_up}/$MODEL/$TARGET"
fi
ADAPTER="$(tagged_run_path "$ADAPTER_ROOT" "$CHECKPOINT_TAG")/final"
EVAL_FILE="${EVAL_FILE:-value_alignment/data/kvs_survey/test.jsonl}"
OUTPUT_DIR="${OUTPUT_ROOT:-value_alignment/results/paper/kvs_up}/$MODEL"
OUTPUT_DIR="$(tagged_run_path "$OUTPUT_DIR" "$RESULT_TAG")"

if [[ ! -f "$EVAL_FILE" ]]; then
  echo "Missing KVS survey file: $EVAL_FILE" >&2
  exit 1
fi
if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
  echo "Missing PEFT adapter: $ADAPTER/adapter_config.json" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"

echo "Evaluating KVS up-regulation: model=$MODEL run=$RUN_NAME"
check_gpu_environment
python -u -m value_alignment.evaluate_kvs_survey \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --eval-file "$EVAL_FILE" \
  --output "$OUTPUT_DIR/$RUN_NAME.json" \
  --output-csv "$OUTPUT_DIR/$RUN_NAME.csv" \
  --num-runs "${NUM_RUNS:-3}" \
  --temperature "${TEMPERATURE:-0.5}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --seed "${EVAL_SEED:-42}"
