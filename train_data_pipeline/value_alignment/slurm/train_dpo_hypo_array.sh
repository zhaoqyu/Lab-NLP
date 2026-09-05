#!/usr/bin/env bash
#SBATCH --job-name=value-pref
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
METHODS=(dpo hypo)
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
VALUE="${VALUE_OVERRIDE:-${VALUES[$VALUE_INDEX]}}"
RUN_TAG="${RUN_TAG:-}"
validate_run_tag "$RUN_TAG"
DATA_DIR="${DATA_ROOT:-value_alignment/data/paper_preferences}/$VALUE/down"
OUTPUT_DIR="${OUTPUT_ROOT:-value_alignment/checkpoints/paper_preference}/$MODEL/$VALUE"
EFFECTIVE_OUTPUT_DIR="$(tagged_run_path "$OUTPUT_DIR/$METHOD" "$RUN_TAG")"
HYPO_REPO="${HYPO_REPO:-$REPO_ROOT/third_party/2026_ICLR_HyPO}"
export HYPO_REPO

for FILE in "$DATA_DIR/train.jsonl" "$DATA_DIR/eval.jsonl"; do
  if [[ ! -f "$FILE" ]]; then
    echo "Missing preference dataset: $FILE" >&2
    echo "Run value_alignment/prepare_kvs_dpo.py before submitting." >&2
    exit 1
  fi
done
for FILE in "$HYPO_REPO/hypo_config.py" "$HYPO_REPO/hypo_trainer.py"; do
  if [[ ! -f "$FILE" ]]; then
    echo "Missing official HyPO file: $FILE" >&2
    exit 1
  fi
done

echo "Training preference model: model=$MODEL method=$METHOD target=$VALUE seed=${SEED:-42} run_tag=${RUN_TAG:-legacy}"
echo "Data: $DATA_DIR"
echo "Output: $EFFECTIVE_OUTPUT_DIR"
check_gpu_environment

CMD=(
  python -u -m value_alignment.train_with_official_hypo
  --method "$METHOD"
  --model "$MODEL"
  --train-file "$DATA_DIR/train.jsonl"
  --eval-file "$DATA_DIR/eval.jsonl"
  --output-dir "$OUTPUT_DIR"
  --beta "${BETA:-0.1}"
  --gamma "${GAMMA:-0.0}"
  --tau "${TAU:-0.0}"
  --epochs "${EPOCHS:-10}"
  --lr "${LEARNING_RATE:-1e-4}"
  --warmup-ratio "${WARMUP_RATIO:-0.15}"
  --batch-size "${BATCH_SIZE:-1}"
  --grad-accum "${GRAD_ACCUM:-16}"
  --lora-r "${LORA_R:-256}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-2}"
  --early-stopping-threshold "${EARLY_STOPPING_THRESHOLD:-0.01}"
  --seed "${SEED:-42}"
  --gradient-checkpointing
)
if [[ "${QLORA:-0}" == "1" ]]; then
  CMD+=(--load-in-4bit)
fi
if [[ -n "$RUN_TAG" ]]; then
  CMD+=(--run-tag "$RUN_TAG")
fi
"${CMD[@]}"
