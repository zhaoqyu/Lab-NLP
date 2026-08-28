#!/usr/bin/env bash

setup_value_alignment_env() {
  REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
  if [[ ! -f "$REPO_ROOT/value_alignment/model_utils.py" ]]; then
    echo "Could not find the Lab-NLP repository at: $REPO_ROOT" >&2
    echo "Submit from the repository root or export REPO_ROOT=/path/to/Lab-NLP." >&2
    return 1
  fi
  cd "$REPO_ROOT"
  REPO_ROOT="$PWD"

  if [[ -n "${MODULES_TO_LOAD:-}" ]]; then
    if ! type module >/dev/null 2>&1; then
      echo "MODULES_TO_LOAD was set, but the cluster module command is unavailable." >&2
      return 1
    fi
    read -r -a MODULE_LIST <<< "$MODULES_TO_LOAD"
    module load "${MODULE_LIST[@]}"
  fi

  VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
  if [[ "${SKIP_ENV_ACTIVATION:-0}" == "1" ]]; then
    echo "Using Python already available on PATH."
  elif [[ -f "$VENV_PATH/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/activate"
  elif [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
    echo "Using the already active Python environment."
  else
    echo "No Python environment found at: $VENV_PATH" >&2
    return 1
  fi

  export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/.cache/huggingface}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  mkdir -p "$HF_HOME" value_alignment/slurm_logs
}

check_gpu_environment() {
  nvidia-smi || true
  python - <<'PY'
import torch
import transformers

print("python packages:")
print("  torch:", torch.__version__)
print("  transformers:", transformers.__version__)
print("  cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; this job requires a GPU.")
print("  gpu:", torch.cuda.get_device_name(0))
print("  bf16_supported:", torch.cuda.is_bf16_supported())
PY
}

validate_array_index() {
  local task_id="$1"
  local task_count="$2"
  if [[ ! "$task_id" =~ ^[0-9]+$ ]] || (( 10#$task_id >= task_count )); then
    echo "Array index must be between 0 and $((task_count - 1)); got: $task_id" >&2
    return 2
  fi
}
