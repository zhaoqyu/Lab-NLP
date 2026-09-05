#!/usr/bin/env bash
#SBATCH --job-name=kvs-baseline
#SBATCH --partition=mlgpu_medium
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2%2
#SBATCH --output=value_alignment/slurm_logs/%x_%A_%a.out
#SBATCH --error=value_alignment/slurm_logs/%x_%A_%a.err

set -Eeuo pipefail

MODELS=(qwen3-8b falcon3-7b llama3.1-8b)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
# shellcheck disable=SC1091
source "$REPO_ROOT/value_alignment/slurm/_common.sh"
validate_array_index "$TASK_ID" "${#MODELS[@]}"
setup_value_alignment_env

MODEL="${MODEL_OVERRIDE:-${MODELS[$((10#$TASK_ID))]}}"
OUTPUT="${OUTPUT:-value_alignment/data/baseline_ratings/${MODEL}.json}"
mkdir -p "$(dirname "$OUTPUT")"

echo "Collecting KVS baseline ratings for $MODEL"
echo "Output: $OUTPUT"
check_gpu_environment

python -u -m value_alignment.collect_kvs_baseline_ratings \
  --model "$MODEL" \
  --output "$OUTPUT" \
  --batch-size "${BATCH_SIZE:-16}" \
  --temperature "${TEMPERATURE:-0.5}" \
  --seed "${SEED:-42}"
