#!/usr/bin/env bash
#SBATCH --job-name=value-sft
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
VARIANTS=(baseline self_direction stimulation hedonism achievement power security conformity tradition benevolence universalism)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_COUNT=$((${#MODELS[@]} * ${#VARIANTS[@]}))

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
# shellcheck disable=SC1091
source "$REPO_ROOT/value_alignment/slurm/_common.sh"
validate_array_index "$TASK_ID" "$TASK_COUNT"
setup_value_alignment_env

TASK_INDEX=$((10#$TASK_ID))
MODEL_INDEX=$((TASK_INDEX / ${#VARIANTS[@]}))
VARIANT_INDEX=$((TASK_INDEX % ${#VARIANTS[@]}))
MODEL="${MODEL_OVERRIDE:-${MODELS[$MODEL_INDEX]}}"
VARIANT="${VARIANT_OVERRIDE:-${VARIANTS[$VARIANT_INDEX]}}"

DATA_ROOT="${DATA_ROOT:-value_alignment/data/paper_sft/$MODEL}"
if [[ "$VARIANT" == "baseline" ]]; then
  DATA_DIR="$DATA_ROOT/baseline"
else
  DATA_DIR="$DATA_ROOT/$VARIANT/down"
fi
OUTPUT_DIR="${OUTPUT_ROOT:-value_alignment/checkpoints/paper_sft}/$MODEL/$VARIANT"

for FILE in "$DATA_DIR/train.jsonl" "$DATA_DIR/eval.jsonl"; do
  if [[ ! -f "$FILE" ]]; then
    echo "Missing SFT dataset: $FILE" >&2
    echo "Run prepare_kvs_sft.py with this model's baseline-ratings JSON first." >&2
    exit 1
  fi
done

echo "Training paper SFT: model=$MODEL variant=$VARIANT"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
check_gpu_environment

CMD=(
  python -u -m value_alignment.train_survey_sft
  --model "$MODEL"
  --train-file "$DATA_DIR/train.jsonl"
  --eval-file "$DATA_DIR/eval.jsonl"
  --output-dir "$OUTPUT_DIR"
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
"${CMD[@]}"
