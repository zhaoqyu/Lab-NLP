# Value Alignment Experiment Pipeline

This directory contains the modular pipeline for comparing standard DPO and
HyPO/Hybrid-DPO with interchangeable instruction-tuned base models.

## Data Roles

The experiment uses the two datasets in separate domains:

- **KVS is training data.** `dataset/kvs_data_new.json` provides a
  value-supporting statement and a contrastive statement for each item. Its
  official `train` split (378 items) becomes DPO/HyPO training pairs, and its
  official `eval` split (108 items) is trainer validation data.
- **AITA is test data.** All 4,335 examples in
  `dataset/aita_dataset_reduced.json` are converted to neutral AITA prompts and
  used only after training to measure value-score changes across 19 values.
- The official KVS `test` split (108 items) is reserved and never written by the
  KVS training-data builder. It can be used as an optional in-domain diagnostic.

This separation avoids AITA train/test leakage and tests whether value steering
learned from short KVS principles transfers to realistic social dilemmas.

## Setup

Install dependencies in the GPU or cluster environment:

```bash
pip install -r value_alignment/requirements.txt
```

Clone the official HyPO implementation:

```bash
mkdir -p third_party
git clone https://github.com/tmllab/2026_ICLR_HyPO.git third_party/2026_ICLR_HyPO
```

Alternatively, point the wrapper to an existing clone:

```bash
export HYPO_REPO=/path/to/2026_ICLR_HyPO
```

## 1. Prepare KVS Training Pairs

Build preference pairs from the native KVS `train` and `eval` splits:

```bash
python value_alignment/prepare_kvs_dpo.py \
  --values all \
  --output-dir value_alignment/data/kvs_dpo
```

This writes:

```text
value_alignment/data/kvs_dpo/train.jsonl  # 378 rows
value_alignment/data/kvs_dpo/eval.jsonl   # 108 rows
```

Each row has this core structure:

```json
{
  "prompt": "value goal and response instruction",
  "chosen": "KVS value-supporting statement",
  "rejected": "KVS contrastive statement",
  "value": "Self_direction_thought"
}
```

Train on selected values instead of all 20 values by listing them explicitly:

```bash
python value_alignment/prepare_kvs_dpo.py \
  --values Security_personal Benevolence_caring Universalism_concern \
  --output-dir value_alignment/data/kvs_dpo_targeted
```

Validate either dataset before training:

```bash
python value_alignment/validate_preference_pairs.py \
  --input value_alignment/data/kvs_dpo/train.jsonl \
  --require-rationale
```

## 2. Prepare the AITA Test Set

Build the cross-domain evaluation file:

```bash
python value_alignment/prepare_aita_eval.py \
  --values all \
  --output value_alignment/data/aita_eval/test.jsonl
```

The test prompt asks only for `NTA`, `YTA`, or `Neutral`. It deliberately does
not reveal the target value, the high-standard label, or the low-standard label.
Those labels remain metadata used by the scoring script.

For a quick balanced smoke test, cap the number of examples per value:

```bash
python value_alignment/prepare_aita_eval.py \
  --max-per-value 5 \
  --output value_alignment/data/aita_eval/smoke_test.jsonl
```

## 3. Optional Teacher-Model Enrichment

The synthetic-data script now uses KVS seeds. It asks a teacher model to turn a
short value principle into a new concrete decision scenario with a chosen and a
contrastive response.

Inspect jobs without making API calls:

```bash
python value_alignment/generate_synthetic_preferences.py \
  --values Security_personal Benevolence_caring Universalism_concern \
  --examples-per-value 2 \
  --personas-per-example 2 \
  --dry-run \
  --output value_alignment/examples/teacher_prompt_jobs_dryrun.jsonl
```

Generate data with an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=...
# Optional for a local or vLLM endpoint:
export OPENAI_BASE_URL=http://localhost:8000/v1

python value_alignment/generate_synthetic_preferences.py \
  --model gpt-4.1-mini \
  --values all \
  --examples-per-value 20 \
  --personas-per-example 2 \
  --output value_alignment/data/synthetic_preferences.jsonl
```

The teacher is asked for concise public rationales, not hidden chain-of-thought.
Validate generated rows before combining them with the native KVS pairs:

```bash
python value_alignment/validate_preference_pairs.py \
  --input value_alignment/data/synthetic_preferences.jsonl \
  --require-rationale
```

## 4. Train DPO or HyPO

The wrapper imports `DPOTrainer` and `DPOConfig` from the official HyPO
repository. It does not reimplement the optimization algorithm.

Standard DPO:

```bash
python value_alignment/train_with_official_hypo.py \
  --method dpo \
  --model qwen2.5-7b \
  --train-file value_alignment/data/kvs_dpo/train.jsonl \
  --eval-file value_alignment/data/kvs_dpo/eval.jsonl \
  --output-dir value_alignment/checkpoints/qwen_dpo
```

HyPO / Hybrid-DPO:

```bash
python value_alignment/train_with_official_hypo.py \
  --method hypo \
  --model qwen2.5-7b \
  --train-file value_alignment/data/kvs_dpo/train.jsonl \
  --eval-file value_alignment/data/kvs_dpo/eval.jsonl \
  --output-dir value_alignment/checkpoints/qwen_hypo \
  --gamma 0.0
```

The smoke-test config also points to KVS:

```bash
python value_alignment/train_with_official_hypo.py \
  --config value_alignment/configs/qwen_hypo_smoketest.json
```

Supported aliases are defined in `value_alignment/configs/model_aliases.json`:

```text
qwen2.5-7b       -> Qwen/Qwen2.5-7B-Instruct
qwen2.5-1.5b     -> Qwen/Qwen2.5-1.5B-Instruct
mistral-7b       -> mistralai/Mistral-7B-Instruct-v0.3
mistral-7b-v02   -> mistralai/Mistral-7B-Instruct-v0.2
llama3.1-8b      -> meta-llama/Llama-3.1-8B-Instruct
```

## 5. SLURM Training

The job array runs six model/method combinations:

```text
array 0: qwen2.5-7b  + DPO
array 1: qwen2.5-7b  + HyPO
array 2: mistral-7b  + DPO
array 3: mistral-7b  + HyPO
array 4: llama3.1-8b + DPO
array 5: llama3.1-8b + HyPO
```

Submit Qwen first as a smoke test:

```bash
sbatch --array=0-1 value_alignment/slurm/train_dpo_hypo_array.sh
```

Then submit the full comparison:

```bash
sbatch value_alignment/slurm/train_dpo_hypo_array.sh
```

The defaults are now:

```text
TRAIN_FILE=value_alignment/data/kvs_dpo/train.jsonl
EVAL_FILE=value_alignment/data/kvs_dpo/eval.jsonl
```

They can still be overridden with `sbatch --export`. Llama 3.1 is gated on
Hugging Face, so accept its license and authenticate before array tasks 4-5.

## 6. Evaluate AITA Value-Score Change

Compare the base model and a trained checkpoint:

```bash
python value_alignment/evaluation/evaluate_aita_probability_gain.py \
  --base-model qwen2.5-7b \
  --trained-model value_alignment/checkpoints/qwen_hypo/hypo/final \
  --test-file value_alignment/data/aita_eval/test.jsonl \
  --output-json value_alignment/results/aita_hypo_value_shift.json \
  --output-csv value_alignment/results/aita_hypo_value_shift.csv
```

For each AITA example, the value score is:

```text
P(high-standard label)
-----------------------------------------------------
P(high-standard label) + P(low-standard label)
```

The score lies in `[0, 1]`. A positive trained-minus-base change means the
trained model moved toward the value-consistent stance. Results include:

- Micro and macro mean value-score change.
- Base and trained three-label accuracy.
- Base and trained pairwise high-vs-low accuracy.
- Per-value score changes across the 19 AITA values.
- Per-example JSON and CSV records for statistical analysis.

Use `--max-examples` for a GPU smoke test before evaluating all 4,335 examples.

## Optional Reserved-KVS Diagnostic

The untouched KVS `test` split can still be evaluated in-domain:

```bash
python value_alignment/prepare_kvs_eval.py \
  --split test \
  --output value_alignment/data/kvs_test_eval.jsonl
```

Then use `evaluate_kvs_survey.py` and
`evaluation/compare_kvs_results.py`. This result is supplementary; AITA is the
primary transfer test.

## Local Checks

Run the data-role regression tests and syntax checks:

```bash
python -m unittest discover -s value_alignment/tests -v
python -m compileall -q value_alignment
```

Full DPO/HyPO training and model scoring require the cluster GPU environment and
model downloads.
