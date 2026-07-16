#!/bin/bash
# Unified Pipeline for Value Alignment in LLMs (HyDPO)
# This script orchestrates Data Generation, Model Training, and Evaluation.

set -e

echo "=========================================================="
echo "    Value Alignment Unified Pipeline (HyDPO)             "
echo "=========================================================="

# 1. Data Preparation
echo "[1/3] Starting Data Preparation Pipeline..."
python3 src/data/generate_synthetic_preferences.py
python3 src/data/prepare_aita_dpo.py
python3 src/data/prepare_kvs_eval.py
python3 src/data/validate_preference_pairs.py
echo "Data preparation complete."

# 2. Model Training
echo "[2/3] Starting Model Training Pipeline..."
echo "Submitting training job to SLURM cluster..."
python3 src/training/train_with_official_hypo.py --config configs/qwen_hypo_smoketest.json
echo "Training complete. Checkpoints saved."

# 3. Evaluation Framework
echo "[3/3] Starting Evaluation Framework..."
echo "-> Intrinsic Evaluation (KVS)..."
python3 src/evaluation/evaluate_kvs_survey.py --model qwen2.5-7b --eval-file dataset/kvs_data_new.json --output results/kvs_hypo_scores.json
python3 src/evaluation/compare_kvs_results.py --trained results/kvs_hypo_scores.json --target-values Security_personal

echo "-> Extrinsic Evaluation (AITA)..."
python3 src/evaluation/evaluate_aita_probability_gain.py --base-model qwen2.5-7b --trained-model checkpoints/hypo/final --test-file dataset/aita_dataset_reduced.json

echo "Pipeline execution finished successfully."
