#!/usr/bin/env bash
#SBATCH --job-name=value-align
#SBATCH --partition=mlgpu_medium
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5%2
#SBATCH --output=value_alignment/slurm_logs/%x_%A_%a.out
#SBATCH --error=value_alignment/slurm_logs/%x_%A_%a.err

set -Eeuo pipefail

# Array IDs: 0/1 Qwen DPO/HyPO, 2/3 Mistral DPO/HyPO,
# and 4/5 Llama DPO/HyPO.
EXPERIMENTS=(
  "qwen2.5-7b|dpo"
  "qwen2.5-7b|hypo"
  "mistral-7b|dpo"
  "mistral-7b|hypo"
  "llama3.1-8b|dpo"
  "llama3.1-8b|hypo"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ! "$TASK_ID" =~ ^[0-9]+$ ]]; then
  echo "SLURM_ARRAY_TASK_ID must be a non-negative integer; got: $TASK_ID" >&2
  exit 2
fi
TASK_INDEX=$((10#$TASK_ID))
LAST_TASK_INDEX=$((${#EXPERIMENTS[@]} - 1))
if (( TASK_INDEX > LAST_TASK_INDEX )); then
  echo "Array index $TASK_INDEX is out of range (expected 0-$LAST_TASK_INDEX)." >&2
  exit 2
fi

IFS='|' read -r DEFAULT_MODEL DEFAULT_METHOD <<< "${EXPERIMENTS[$TASK_INDEX]}"
MODEL="${MODEL_OVERRIDE:-$DEFAULT_MODEL}"
METHOD="${METHOD_OVERRIDE:-$DEFAULT_METHOD}"
if [[ "$METHOD" != "dpo" && "$METHOD" != "hypo" ]]; then
  echo "METHOD_OVERRIDE must be 'dpo' or 'hypo'; got: $METHOD" >&2
  exit 2
fi

# Submit from the repository root. REPO_ROOT can be exported when submitting
# from another directory.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$REPO_ROOT/value_alignment/train_with_official_hypo.py" ]]; then
  echo "Could not find the Lab-NLP repository at: $REPO_ROOT" >&2
  echo "Run sbatch from the repository root or export REPO_ROOT=/path/to/Lab-NLP." >&2
  exit 1
fi
cd "$REPO_ROOT"
REPO_ROOT="$PWD"

if [[ -n "${MODULES_TO_LOAD:-}" ]]; then
  if ! type module >/dev/null 2>&1; then
    echo "MODULES_TO_LOAD was set, but the cluster 'module' command is unavailable." >&2
    exit 1
  fi
  read -r -a MODULE_LIST <<< "$MODULES_TO_LOAD"
  module load "${MODULE_LIST[@]}"
fi

VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
if [[ "${SKIP_ENV_ACTIVATION:-0}" == "1" ]]; then
  echo "Using Python already available on PATH (SKIP_ENV_ACTIVATION=1)."
elif [[ -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
elif [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
  echo "Using the already active Python environment."
else
  echo "No Python environment found at: $VENV_PATH" >&2
  echo "Create it, set VENV_PATH, or submit with SKIP_ENV_ACTIVATION=1." >&2
  exit 1
fi

TRAIN_FILE="${TRAIN_FILE:-value_alignment/data/aita_dpo/train.jsonl}"
EVAL_FILE="${EVAL_FILE:-value_alignment/data/aita_dpo/eval.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-value_alignment/checkpoints/slurm}"
HYPO_REPO="${HYPO_REPO:-$REPO_ROOT/third_party/2026_ICLR_HyPO}"

BETA="${BETA:-0.1}"
GAMMA="${GAMMA:-0.0}"
TAU="${TAU:-0.0}"
EPOCHS="${EPOCHS:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"

export HYPO_REPO
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for REQUIRED_FILE in \
  "$TRAIN_FILE" \
  "$EVAL_FILE" \
  "$HYPO_REPO/hypo_config.py" \
  "$HYPO_REPO/hypo_trainer.py"; do
  if [[ ! -f "$REQUIRED_FILE" ]]; then
    echo "Required file not found: $REQUIRED_FILE" >&2
    exit 1
  fi
done

MODEL_SLUG="${MODEL//\//_}"
RUN_TAG="${RUN_TAG:-}"
RUN_SUFFIX="${RUN_TAG:+-$RUN_TAG}"
MODEL_OUTPUT_DIR="$OUTPUT_ROOT/${MODEL_SLUG}${RUN_SUFFIX}"
FINAL_MODEL_DIR="$MODEL_OUTPUT_DIR/$METHOD/final"

ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
JOB_ID="${SLURM_JOB_ID:-local}"
RUN_LOG="value_alignment/slurm_logs/${MODEL_SLUG}_${METHOD}_${ARRAY_JOB_ID}_${TASK_INDEX}.log"

mkdir -p value_alignment/slurm_logs "$MODEL_OUTPUT_DIR" "$HF_HOME"

if [[ -d "$FINAL_MODEL_DIR" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
  echo "Final model already exists: $FINAL_MODEL_DIR" >&2
  echo "Set RUN_TAG for a new run or ALLOW_OVERWRITE=1 to replace it." >&2
  exit 1
fi

finish() {
  local status=$?
  echo "Finished at: $(date --iso-8601=seconds 2>/dev/null || date)"
  echo "Exit status: $status"
}
trap finish EXIT

echo "=========================================="
echo "SLURM_JOB_ID: $JOB_ID"
echo "SLURM_ARRAY_JOB_ID: $ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $TASK_INDEX"
echo "Experiment: $MODEL / $METHOD"
echo "Repository: $REPO_ROOT"
echo "Train file: $TRAIN_FILE"
echo "Eval file: $EVAL_FILE"
echo "Output: $FINAL_MODEL_DIR"
echo "Run log: $RUN_LOG"
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Started at: $(date --iso-8601=seconds 2>/dev/null || date)"
echo "=========================================="

nvidia-smi || true

echo "python: $(command -v python)"
python - <<'PY'
import datasets
import peft
import torch
import transformers
import trl

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("peft:", peft.__version__)
print("trl:", trl.__version__)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; this training job requires a GPU.")
print("gpu:", torch.cuda.get_device_name(0))
print("bf16_supported:", torch.cuda.is_bf16_supported())
PY

TRAIN_CMD=(
  python -u value_alignment/train_with_official_hypo.py
  --method "$METHOD"
  --model "$MODEL"
  --train-file "$TRAIN_FILE"
  --eval-file "$EVAL_FILE"
  --output-dir "$MODEL_OUTPUT_DIR"
  --beta "$BETA"
  --gamma "$GAMMA"
  --tau "$TAU"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --lr "$LEARNING_RATE"
  --max-length "$MAX_LENGTH"
  --max-prompt-length "$MAX_PROMPT_LENGTH"
)

if [[ "${NO_LORA:-0}" == "1" ]]; then
  TRAIN_CMD+=(--no-lora)
fi

echo "Running command:"
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'

"${TRAIN_CMD[@]}" 2>&1 | tee "$RUN_LOG"

echo "Model saved to: $FINAL_MODEL_DIR"
