#!/usr/bin/env bash
# One-shot, non-SLURM pipeline for the crunch week (single model, reduced value
# subset, one seed). Trains SFT baseline+targets, DPO, HyPO, SimPO, KTO for a
# handful of Schwartz values, evaluates all of them on KVS + AITA, and writes
# the final aggregate tables. No sbatch/SLURM required -- this runs everything
# sequentially in one process so it works on any single-GPU box.
#
# Usage, from the repository root (the folder that contains value_alignment/):
#   nohup bash value_alignment/run_priority_pipeline.sh > priority_run.log 2>&1 &
#   tail -f priority_run.log
#
# Safe to re-run / resume: every stage writes a ".done" marker under
# value_alignment/slurm_logs/priority_run/ and is skipped on the next run if
# its marker exists. Delete a marker (or the whole priority_run/ directory) to
# force that stage to redo. A failed stage is logged and skipped -- it does
# NOT stop the rest of the pipeline, so you always get as much done as
# possible in one unattended pass.
#
# Override any of these with environment variables before running, e.g.:
#   MODEL=qwen3-8b VALUES_STR="security benevolence" bash value_alignment/run_priority_pipeline.sh
set -uo pipefail

MODEL="${MODEL:-qwen2.5-7b}"
IFS=' ' read -r -a VALUES <<< "${VALUES_STR:-universalism security benevolence self_direction power}"
SEED="${SEED:-42}"
SFT_EPOCHS="${SFT_EPOCHS:-5}"
PREF_EPOCHS="${PREF_EPOCHS:-3}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
QLORA="${QLORA:-1}"
EVAL_NUM_RUNS="${EVAL_NUM_RUNS:-3}"
PREF_METHODS=(dpo hypo simpo kto)

REPO_ROOT="${REPO_ROOT:-$PWD}"
# shellcheck disable=SC1091
source "$REPO_ROOT/value_alignment/slurm/_common.sh"
if ! setup_value_alignment_env; then
  echo "FATAL: environment setup failed (see messages above). Fix and re-run." >&2
  exit 1
fi
check_gpu_environment || echo "WARNING: GPU check reported a problem above; continuing anyway." >&2

RUN_LOG_DIR="value_alignment/slurm_logs/priority_run"
mkdir -p "$RUN_LOG_DIR"
STATUS_FILE="$RUN_LOG_DIR/status.tsv"
touch "$STATUS_FILE"

EXTRA_4BIT_FLAG=()
if [[ "$QLORA" == "1" ]]; then
  EXTRA_4BIT_FLAG=(--load-in-4bit)
fi

echo "==================================================================="
echo " Priority run started: $(date -u +%FT%TZ)"
echo " Model:  $MODEL"
echo " Values: ${VALUES[*]}"
echo " Seed:   $SEED   QLoRA: $QLORA   LoRA r=$LORA_R alpha=$LORA_ALPHA"
echo "==================================================================="

run_step() {
  local name="$1"; shift
  local marker="$RUN_LOG_DIR/${name}.done"
  local logfile="$RUN_LOG_DIR/${name}.log"
  if [[ -f "$marker" ]]; then
    echo "[SKIP] $name (already done; rm $marker to redo)"
    printf 'SKIP\t%s\t%s\n' "$name" "$(date -u +%FT%TZ)" >> "$STATUS_FILE"
    return 0
  fi
  echo "[RUN ] $name"
  if "$@" > "$logfile" 2>&1; then
    touch "$marker"
    echo "[ OK ] $name"
    printf 'OK\t%s\t%s\n' "$name" "$(date -u +%FT%TZ)" >> "$STATUS_FILE"
  else
    echo "[FAIL] $name -- see $logfile"
    printf 'FAIL\t%s\t%s\n' "$name" "$(date -u +%FT%TZ)" >> "$STATUS_FILE"
  fi
}

clone_hypo_repo() {
  [[ -f third_party/2026_ICLR_HyPO/hypo_config.py ]] || \
    git clone https://github.com/tmllab/2026_ICLR_HyPO.git third_party/2026_ICLR_HyPO
}

# ---------------------------------------------------------------- Stage 0 --
run_step "00_regression_tests" python -m unittest discover -s value_alignment/tests -v

# ---------------------------------------------------------------- Stage 1 --
# Static data prep (CPU only, fast). Values are restricted to the reduced set.
run_step "01_clone_hypo_repo" clone_hypo_repo
run_step "02_prepare_kvs_dpo" python -m value_alignment.prepare_kvs_dpo \
  --target-values "${VALUES[@]}" --seed "$SEED"
run_step "03_prepare_kvs_eval" python -m value_alignment.prepare_kvs_eval --split test
run_step "04_prepare_aita_eval" python -m value_alignment.prepare_aita_eval \
  --values "${VALUES[@]}" --seed "$SEED"

# ---------------------------------------------------------------- Stage 2 --
# Model-specific baseline ratings (needs the GPU; required before SFT data).
BASELINE_RATINGS="value_alignment/data/baseline_ratings/$MODEL.json"
run_step "05_collect_baseline_ratings" python -m value_alignment.collect_kvs_baseline_ratings \
  --model "$MODEL" --output "$BASELINE_RATINGS" --seed "$SEED"

run_step "06_prepare_kvs_sft" python -m value_alignment.prepare_kvs_sft \
  --baseline-ratings "$BASELINE_RATINGS" \
  --output-root "value_alignment/data/paper_sft/$MODEL" \
  --target-values "${VALUES[@]}"

# ---------------------------------------------------------------- Stage 3 --
# SFT: matched baseline control + one adapter per target value.
SFT_DATA_ROOT="value_alignment/data/paper_sft/$MODEL"
SFT_CKPT_ROOT="value_alignment/checkpoints/paper_sft/$MODEL"

run_step "10_train_sft_baseline" python -m value_alignment.train_survey_sft \
  --model "$MODEL" \
  --train-file "$SFT_DATA_ROOT/baseline/train.jsonl" \
  --eval-file "$SFT_DATA_ROOT/baseline/eval.jsonl" \
  --output-dir "$SFT_CKPT_ROOT/baseline" \
  --epochs "$SFT_EPOCHS" --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" \
  --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" --seed "$SEED" \
  --gradient-checkpointing "${EXTRA_4BIT_FLAG[@]}"

for VALUE in "${VALUES[@]}"; do
  run_step "10_train_sft_${VALUE}" python -m value_alignment.train_survey_sft \
    --model "$MODEL" \
    --train-file "$SFT_DATA_ROOT/${VALUE}/down/train.jsonl" \
    --eval-file "$SFT_DATA_ROOT/${VALUE}/down/eval.jsonl" \
    --output-dir "$SFT_CKPT_ROOT/${VALUE}" \
    --epochs "$SFT_EPOCHS" --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" \
    --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" --seed "$SEED" \
    --gradient-checkpointing "${EXTRA_4BIT_FLAG[@]}"
done

# ---------------------------------------------------------------- Stage 4 --
# Preference methods: DPO, HyPO (official trainer), SimPO, KTO (TRL trainers).
PREF_DATA_ROOT="value_alignment/data/paper_preferences"
PREF_CKPT_ROOT="value_alignment/checkpoints/paper_preference/$MODEL"

for VALUE in "${VALUES[@]}"; do
  for METHOD in dpo hypo; do
    run_step "20_train_${METHOD}_${VALUE}" python -m value_alignment.train_with_official_hypo \
      --method "$METHOD" --model "$MODEL" \
      --train-file "$PREF_DATA_ROOT/${VALUE}/down/train.jsonl" \
      --eval-file "$PREF_DATA_ROOT/${VALUE}/down/eval.jsonl" \
      --output-dir "$PREF_CKPT_ROOT/${VALUE}" \
      --epochs "$PREF_EPOCHS" --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" \
      --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" --seed "$SEED" \
      --gradient-checkpointing "${EXTRA_4BIT_FLAG[@]}"
  done
  for METHOD in simpo kto; do
    run_step "20_train_${METHOD}_${VALUE}" python -m value_alignment.train_extra_methods \
      --method "$METHOD" --model "$MODEL" \
      --train-file "$PREF_DATA_ROOT/${VALUE}/down/train.jsonl" \
      --eval-file "$PREF_DATA_ROOT/${VALUE}/down/eval.jsonl" \
      --output-dir "$PREF_CKPT_ROOT/${VALUE}" \
      --epochs "$PREF_EPOCHS" --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" \
      --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" --seed "$SEED" \
      --gradient-checkpointing "${EXTRA_4BIT_FLAG[@]}"
  done
done

# ---------------------------------------------------------------- Stage 5 --
# KVS intrinsic evaluation: base model, sft_baseline, then every adapter.
KVS_EVAL_FILE="value_alignment/data/kvs_survey/test.jsonl"
KVS_OUT_DIR="value_alignment/results/paper/kvs/$MODEL"
mkdir -p "$KVS_OUT_DIR"

eval_kvs() {
  local run_name="$1" adapter="$2"
  local cmd=(python -m value_alignment.evaluate_kvs_survey \
    --model "$MODEL" --eval-file "$KVS_EVAL_FILE" \
    --output "$KVS_OUT_DIR/${run_name}.json" --output-csv "$KVS_OUT_DIR/${run_name}.csv" \
    --num-runs "$EVAL_NUM_RUNS" --seed "$SEED")
  if [[ -n "$adapter" ]]; then
    cmd+=(--adapter "$adapter")
  fi
  "${cmd[@]}"
}

run_step "30_eval_kvs_base" eval_kvs "base" ""
run_step "30_eval_kvs_sft_baseline" eval_kvs "sft_baseline" "$SFT_CKPT_ROOT/baseline/final"
for VALUE in "${VALUES[@]}"; do
  run_step "30_eval_kvs_sft_${VALUE}" eval_kvs "sft_${VALUE}" "$SFT_CKPT_ROOT/${VALUE}/final"
  for METHOD in "${PREF_METHODS[@]}"; do
    run_step "30_eval_kvs_${METHOD}_${VALUE}" eval_kvs "${METHOD}_${VALUE}" "$PREF_CKPT_ROOT/${VALUE}/${METHOD}/final"
  done
done

# ---------------------------------------------------------------- Stage 6 --
# AITA out-of-domain behavioral evaluation.
AITA_TEST_FILE="value_alignment/data/aita_eval/test.jsonl"

eval_aita() {
  local method="$1" value="$2" base_adapter="$3" conditioned_adapter="$4"
  local out_dir="value_alignment/results/paper/aita/$MODEL/${method}"
  mkdir -p "$out_dir"
  local cmd=(python -m value_alignment.evaluation.evaluate_aita_probability_gain \
    --base-model "$MODEL" --conditioned-model "$MODEL" \
    --conditioned-adapter "$conditioned_adapter" \
    --target-value "$value" --test-file "$AITA_TEST_FILE" \
    --output-json "$out_dir/${value}.json" --output-csv "$out_dir/${value}.csv")
  if [[ -n "$base_adapter" ]]; then
    cmd+=(--base-adapter "$base_adapter")
  fi
  "${cmd[@]}"
}

for VALUE in "${VALUES[@]}"; do
  run_step "40_eval_aita_sft_${VALUE}" eval_aita "sft" "$VALUE" \
    "$SFT_CKPT_ROOT/baseline/final" "$SFT_CKPT_ROOT/${VALUE}/final"
  for METHOD in "${PREF_METHODS[@]}"; do
    run_step "40_eval_aita_${METHOD}_${VALUE}" eval_aita "$METHOD" "$VALUE" \
      "" "$PREF_CKPT_ROOT/${VALUE}/${METHOD}/final"
  done
done

# ---------------------------------------------------------------- Stage 7 --
# Aggregate. --allow-missing so a partial matrix (some failed stages above)
# still produces a usable table instead of hard-erroring.
run_step "50_summarize_kvs" python -m value_alignment.evaluation.summarize_kvs_experiments \
  --models "$MODEL" --methods sft dpo hypo simpo kto --target-values "${VALUES[@]}" --allow-missing
run_step "50_summarize_aita" python -m value_alignment.evaluation.summarize_aita_experiments \
  --models "$MODEL" --methods sft dpo hypo simpo kto --target-values "${VALUES[@]}" --allow-missing

echo "==================================================================="
echo " Priority run finished: $(date -u +%FT%TZ)"
FAIL_COUNT=$(grep -c '^FAIL' "$STATUS_FILE" || true)
OK_COUNT=$(grep -c '^OK' "$STATUS_FILE" || true)
SKIP_COUNT=$(grep -c '^SKIP' "$STATUS_FILE" || true)
echo " OK: $OK_COUNT   SKIPPED: $SKIP_COUNT   FAILED: $FAIL_COUNT"
if [[ "$FAIL_COUNT" -gt 0 ]]; then
  echo " Failed stages (see $RUN_LOG_DIR/<stage>.log for each):"
  grep '^FAIL' "$STATUS_FILE" | cut -f2 | sed 's/^/   - /'
fi
echo " Results:"
echo "   value_alignment/results/paper/kvs_summary.csv"
echo "   value_alignment/results/paper/aita_summary.csv"
echo "==================================================================="
