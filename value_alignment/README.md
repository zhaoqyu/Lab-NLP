# Value Alignment Experiment Pipeline

This folder contains a modular pipeline for the NLP Lab project:

- Data preparation from AITA and KVS.
- Synthetic preference generation with a teacher LLM.
- Standard DPO and HyPO/Hybrid-DPO training through the official HyPO trainer.
- Intrinsic KVS value-rating evaluation.
- Extrinsic AITA probability-gain evaluation.
- MACHIAVELLI setup notes for optional transfer evaluation.

## Data Roles

- `dataset/aita_dataset_reduced.json`
  - Used for DPO/HyPO preference-pair training.
  - Used again for held-out behavioral AITA evaluation.
  - Converted to `prompt`, `chosen`, `rejected`.

- `dataset/kvs_data_new.json`
  - Used mainly for intrinsic survey-style evaluation.
  - The model rates value statements from 1 to 6.
  - We compare target value score shifts and other-value variance before/after training.

## Setup

Install dependencies in a GPU/cluster environment:

```bash
pip install -r value_alignment/requirements.txt
```

Clone the official HyPO implementation:

```bash
mkdir -p third_party
git clone https://github.com/tmllab/2026_ICLR_HyPO.git third_party/2026_ICLR_HyPO
```

Or point to an existing clone:

```bash
export HYPO_REPO=/path/to/2026_ICLR_HyPO
```

## Model Switching

All model scripts accept either a full Hugging Face model path or one of these aliases:

```text
qwen2.5-7b       -> Qwen/Qwen2.5-7B-Instruct
qwen2.5-1.5b     -> Qwen/Qwen2.5-1.5B-Instruct
mistral-7b       -> mistralai/Mistral-7B-Instruct-v0.3
mistral-7b-v02   -> mistralai/Mistral-7B-Instruct-v0.2
llama3.1-8b      -> meta-llama/Llama-3.1-8B-Instruct
```

Aliases live in:

```text
value_alignment/configs/model_aliases.json
```

Example:

```bash
python value_alignment/train_with_official_hypo.py --model qwen2.5-7b
python value_alignment/train_with_official_hypo.py --model mistral-7b
python value_alignment/train_with_official_hypo.py --model llama3.1-8b
```

## Dataset Summary

```bash
python value_alignment/summarize_datasets.py \
  --output value_alignment/results/dataset_summary.json
```

Current local summary:

- AITA: 4,335 examples across 19 values.
- KVS: 378 train, 108 eval, 108 test examples.

## AITA Preference Data

Prepare a full DPO/HyPO split:

```bash
python value_alignment/prepare_aita_dpo.py \
  --values Security_personal Benevolence_caring Universalism_concern Self_direction_action \
  --output-dir value_alignment/data/aita_dpo
```

This creates:

```text
value_alignment/data/aita_dpo/train.jsonl
value_alignment/data/aita_dpo/eval.jsonl
value_alignment/data/aita_dpo/test.jsonl
```

In the current run this produced:

```text
train: 1600
eval:   200
test:   200
```

Create a small inspectable batch:

```bash
python value_alignment/prepare_aita_dpo.py \
  --values Security_personal Benevolence_caring Universalism_concern \
  --max-per-value 10 \
  --single-file value_alignment/examples/first_batch_aita_preferences.jsonl
```

Validate preference pairs:

```bash
python value_alignment/validate_preference_pairs.py \
  --input value_alignment/data/aita_dpo/train.jsonl
```

## Synthetic Teacher Generation

Dry-run first to inspect prompts without spending API/cluster budget:

```bash
python value_alignment/generate_synthetic_preferences.py \
  --values Security_personal Benevolence_caring Universalism_concern \
  --examples-per-value 2 \
  --personas-per-example 2 \
  --dry-run \
  --output value_alignment/examples/teacher_prompt_jobs_dryrun.jsonl
```

Generate with an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=...
# Optional for local/vLLM/OpenAI-compatible endpoints:
export OPENAI_BASE_URL=http://localhost:8000/v1

python value_alignment/generate_synthetic_preferences.py \
  --model gpt-4.1-mini \
  --values Security_personal Benevolence_caring Universalism_concern Self_direction_action \
  --examples-per-value 50 \
  --personas-per-example 2 \
  --output value_alignment/data/synthetic_preferences.jsonl
```

Validate generated data:

```bash
python value_alignment/validate_preference_pairs.py \
  --input value_alignment/data/synthetic_preferences.jsonl \
  --require-rationale
```

We store concise public rationales, not long hidden chain-of-thought traces.

## Training: Standard DPO

```bash
python value_alignment/train_with_official_hypo.py \
  --method dpo \
  --model qwen2.5-7b \
  --train-file value_alignment/data/aita_dpo/train.jsonl \
  --eval-file value_alignment/data/aita_dpo/eval.jsonl \
  --output-dir value_alignment/checkpoints/qwen_dpo
```

## Training: HyPO / Hybrid-DPO

```bash
python value_alignment/train_with_official_hypo.py \
  --method hypo \
  --model qwen2.5-7b \
  --train-file value_alignment/data/aita_dpo/train.jsonl \
  --eval-file value_alignment/data/aita_dpo/eval.jsonl \
  --output-dir value_alignment/checkpoints/qwen_hypo \
  --gamma 0.0
```

Config-file example:

```bash
python value_alignment/train_with_official_hypo.py \
  --config value_alignment/configs/qwen_hypo_smoketest.json
```

The wrapper calls the official HyPO `DPOTrainer`; our code does not reimplement the core algorithm.

## SLURM Training on Marvin

The job-array script runs all six model/method combinations:

```text
array 0: qwen2.5-7b  + DPO
array 1: qwen2.5-7b  + HyPO
array 2: mistral-7b  + DPO
array 3: mistral-7b  + HyPO
array 4: llama3.1-8b + DPO
array 5: llama3.1-8b + HyPO
```

It requests one GPU, 64 GB RAM, four CPUs, and 24 hours from
`mlgpu_medium`. At most two array tasks run concurrently.

Prepare the environment once on the login node:

```bash
cd /path/to/Lab-NLP
python3 -m venv .venv
source .venv/bin/activate
pip install -r value_alignment/requirements.txt
mkdir -p third_party
git clone https://github.com/tmllab/2026_ICLR_HyPO.git third_party/2026_ICLR_HyPO
```

Submit only Qwen DPO and HyPO first:

```bash
sbatch --array=0-1 value_alignment/slurm/train_dpo_hypo_array.sh
```

Submit the complete comparison after the Qwen jobs pass:

```bash
sbatch value_alignment/slurm/train_dpo_hypo_array.sh
```

Useful overrides can be supplied with `--export`. This example creates a
shorter Qwen HyPO smoke test in a separate output directory:

```bash
sbatch \
  --array=1 \
  --export=ALL,EPOCHS=0.05,MAX_LENGTH=1024,MAX_PROMPT_LENGTH=768,RUN_TAG=smoke \
  value_alignment/slurm/train_dpo_hypo_array.sh
```

Common overrides are:

```text
VENV_PATH=/path/to/venv
HYPO_REPO=/path/to/2026_ICLR_HyPO
HF_HOME=/path/to/huggingface/cache
OUTPUT_ROOT=/path/to/checkpoints
TRAIN_FILE=/path/to/train.jsonl
EVAL_FILE=/path/to/eval.jsonl
EPOCHS=1.0
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=1e-5
RUN_TAG=experiment-name
```

Llama 3.1 is gated on Hugging Face, so accept its model license and authenticate
with `hf auth login` before submitting array tasks 4 and 5. SLURM `.out`, `.err`,
and per-run logs are written to `value_alignment/slurm_logs/`; model outputs go
to `value_alignment/checkpoints/slurm/` by default.

## DPO vs HyPO Difference

Standard DPO:

```python
logits = delta_policy - delta_ref
```

HyPO:

```python
logits = delta_policy - max(0, delta_ref)
```

HyPO keeps reference-model regularization when useful, but avoids weakening the preference signal when the reference model prefers the rejected answer.

## Intrinsic Evaluation: KVS

Prepare survey prompts:

```bash
python value_alignment/prepare_kvs_eval.py \
  --split test \
  --output value_alignment/data/kvs_test_eval.jsonl
```

Evaluate the base model:

```bash
python value_alignment/evaluate_kvs_survey.py \
  --model qwen2.5-7b \
  --eval-file value_alignment/data/kvs_test_eval.jsonl \
  --output value_alignment/results/kvs_base_scores.json \
  --output-csv value_alignment/results/kvs_base_scores.csv
```

Evaluate a trained model:

```bash
python value_alignment/evaluate_kvs_survey.py \
  --model value_alignment/checkpoints/qwen_hypo/hypo/final \
  --eval-file value_alignment/data/kvs_test_eval.jsonl \
  --output value_alignment/results/kvs_hypo_scores.json \
  --output-csv value_alignment/results/kvs_hypo_scores.csv
```

Compare intrinsic scores:

```bash
python value_alignment/evaluation/compare_kvs_results.py \
  --base value_alignment/results/kvs_base_scores.json \
  --trained value_alignment/results/kvs_hypo_scores.json \
  --target-values Security_personal Benevolence_caring Universalism_concern Self_direction_action \
  --output-json value_alignment/results/kvs_hypo_comparison.json \
  --output-csv value_alignment/results/kvs_hypo_comparison.csv
```

Metrics:

- Target value mean score shift.
- Per-value mean and standard deviation.
- Other values' variance, measuring unintended drift.

## Extrinsic Evaluation: AITA Probability Gain

Run after training:

```bash
python value_alignment/evaluation/evaluate_aita_probability_gain.py \
  --base-model qwen2.5-7b \
  --trained-model value_alignment/checkpoints/qwen_hypo/hypo/final \
  --test-file value_alignment/data/aita_dpo/test.jsonl \
  --output-json value_alignment/results/aita_hypo_probability_gain.json \
  --output-csv value_alignment/results/aita_hypo_probability_gain.csv
```

Metrics:

- Base/trained accuracy against `high_standard_stance`.
- Per-value accuracy.
- Probability gain for the value-consistent label.

## MACHIAVELLI

See:

```text
value_alignment/evaluation/MACHIAVELLI_SETUP.md
value_alignment/evaluation/machiavelli_hf_agent_template.py
```

This is optional/heavier because it requires a separate benchmark repo and game data download.

## Local Validation Already Run

These checks were run locally:

```text
AITA split generation: 2000 examples -> 1600/200/200
AITA train validation: 1600 rows, 0 errors
KVS test conversion: 108 rows
Synthetic teacher dry-run: passed
Python compile check: passed
```

Training and model evaluations require the GPU/cluster environment and model downloads.
