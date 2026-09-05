#!/usr/bin/env bash
#SBATCH --job-name=cap-eval
#SBATCH --partition=mlgpu_medium
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-95%2
#SBATCH --output=value_alignment/slurm_logs/%x_%A_%a.out
#SBATCH --error=value_alignment/slurm_logs/%x_%A_%a.err

set -Eeuo pipefail

MODELS=(qwen3-8b falcon3-7b llama3.1-8b)
VALUES=(self_direction stimulation hedonism achievement power security conformity tradition benevolence universalism)
VARIANTS_PER_MODEL=32
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
ADAPTER=""

if (( VARIANT_INDEX == 0 )); then
  RUN_NAME="base"
elif (( VARIANT_INDEX == 1 )); then
  RUN_NAME="sft_baseline"
  ADAPTER_ROOT="${SFT_CHECKPOINT_ROOT:-value_alignment/checkpoints/paper_sft}/$MODEL/baseline"
  ADAPTER="$(tagged_run_path "$ADAPTER_ROOT" "$CHECKPOINT_TAG")/final"
elif (( VARIANT_INDEX < 12 )); then
  TARGET="${VALUES[$((VARIANT_INDEX - 2))]}"
  RUN_NAME="sft_$TARGET"
  ADAPTER_ROOT="${SFT_CHECKPOINT_ROOT:-value_alignment/checkpoints/paper_sft}/$MODEL/$TARGET"
  ADAPTER="$(tagged_run_path "$ADAPTER_ROOT" "$CHECKPOINT_TAG")/final"
elif (( VARIANT_INDEX < 22 )); then
  TARGET="${VALUES[$((VARIANT_INDEX - 12))]}"
  RUN_NAME="dpo_$TARGET"
  ADAPTER_ROOT="${PREFERENCE_CHECKPOINT_ROOT:-value_alignment/checkpoints/paper_preference}/$MODEL/$TARGET/dpo"
  ADAPTER="$(tagged_run_path "$ADAPTER_ROOT" "$CHECKPOINT_TAG")/final"
else
  TARGET="${VALUES[$((VARIANT_INDEX - 22))]}"
  RUN_NAME="hypo_$TARGET"
  ADAPTER_ROOT="${PREFERENCE_CHECKPOINT_ROOT:-value_alignment/checkpoints/paper_preference}/$MODEL/$TARGET/hypo"
  ADAPTER="$(tagged_run_path "$ADAPTER_ROOT" "$CHECKPOINT_TAG")/final"
fi

MODEL_OUTPUT_ROOT="${OUTPUT_ROOT:-value_alignment/results/paper/capabilities}/$MODEL"
MODEL_OUTPUT_ROOT="$(tagged_run_path "$MODEL_OUTPUT_ROOT" "$RESULT_TAG")"
OUTPUT_DIR="$MODEL_OUTPUT_ROOT/$RUN_NAME"
if [[ -n "$ADAPTER" && ! -f "$ADAPTER/adapter_config.json" ]]; then
  echo "Missing PEFT adapter: $ADAPTER/adapter_config.json" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"

echo "Capability evaluation: model=$MODEL run=$RUN_NAME checkpoint_tag=${CHECKPOINT_TAG:-legacy} result_tag=${RESULT_TAG:-legacy}"
check_gpu_environment
python -c 'import lm_eval' || {
  echo "lm-evaluation-harness is missing; install value_alignment/requirements.txt." >&2
  exit 1
}

read -r -a TASK_LIST <<< "${TASKS:-mmlu gsm8k}"
CMD=(
  python -u -m value_alignment.evaluation.evaluate_capabilities
  --model "$MODEL"
  --tasks "${TASK_LIST[@]}"
  --output-dir "$OUTPUT_DIR"
  --batch-size "${BATCH_SIZE:-auto}"
  --device "${DEVICE:-cuda:0}"
  --dtype "${DTYPE:-bfloat16}"
  --seed "${EVAL_SEED:-42}"
)
if [[ -n "$ADAPTER" ]]; then
  CMD+=(--adapter "$ADAPTER")
fi
if [[ -n "${LIMIT:-}" ]]; then
  CMD+=(--limit "$LIMIT")
fi
if [[ "${APPLY_CHAT_TEMPLATE:-0}" == "1" ]]; then
  CMD+=(--apply-chat-template)
fi
if [[ "${LOG_SAMPLES:-0}" == "1" ]]; then
  CMD+=(--log-samples)
fi
if [[ "${QLORA:-0}" == "1" ]]; then
  CMD+=(--load-in-4bit)
fi
"${CMD[@]}"
