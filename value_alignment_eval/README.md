# Value Alignment Evaluation Suite

This directory contains the professional, robust evaluation suite developed for assessing the Hybrid Direct Preference Optimization (HyDPO) model alignment. It focuses on large-scale metric computation over SLURM clusters, ensuring accurate and reproducible calculations of our target objectives.

## Directory and Script Breakdown

### 1. Intrinsic Evaluation: KVS Scores

**`evaluate_kvs_survey.py`**
- **Purpose:** Extracts scalar ratings (1-6) from generated responses given KVS value-statement prompts.
- **Features:** Integrates with `AutoModelForCausalLM` and computes single-token or exact-match generations safely without gradients. Supports CSV and JSON metric dumping.
- **Usage:**
  ```bash
  python evaluate_kvs_survey.py --model qwen2.5-7b --eval-file ../dataset/kvs_test_eval.jsonl --output results/kvs_base_scores.json
  ```

**`evaluation/compare_kvs_results.py`**
- **Purpose:** Compares the JSON outputs of a base model against a trained model to calculate global alignment shifts.
- **Metrics Calculated:**
  - *Target Value Rating Drop:* Evaluates the successful suppression of misaligned behavior across the target value array.
  - *Other Values' Variance:* Calculates variance across neutral/unmanipulated axes to ensure no catastrophic forgetting occurs.
- **Usage:**
  ```bash
  python evaluation/compare_kvs_results.py --base results/kvs_base_scores.json --trained results/kvs_hypo_scores.json --target-values Security_personal
  ```

### 2. Extrinsic Evaluation: Probability Gain

**`evaluation/evaluate_aita_probability_gain.py`**
- **Purpose:** Computes the mathematical shift in logits toward ethically aligned stances (e.g., "NTA") in complex social dilemmas from the AITA dataset.
- **Features:** Efficiently extracts final sequence token logits via `torch.no_grad()` to compute `softmax(logits)` deltas instead of relying on stochastic greedy decoding.
- **Usage:**
  ```bash
  python evaluation/evaluate_aita_probability_gain.py --base-model qwen2.5-7b --trained-model path/to/hypo_final --test-file ../dataset/aita_test.jsonl
  ```

### 3. MACHIAVELLI Benchmark Integration

**`setup_machiavelli.sh` & `evaluation/MACHIAVELLI_SETUP.md`**
- **Purpose:** Fully orchestrates the interactive text-adventure environment required to benchmark the agent's Power-Seeking and Moral Violation metrics.
- **Integration:** The `machiavelli_hf_agent_template.py` file maps our custom Qwen API to the Gym-like text interface provided by the Machiavelli engine.

## SLURM & Cluster Integration
All scripts are designed to work seamlessly within job arrays (e.g., `mlgpu_medium`). They parse standard `argparse` configurations allowing them to inherit model aliases, checkpoint paths, and dataset locations dynamically from the cluster dispatcher.

---
**Author:** Ali (Evaluation Module)
