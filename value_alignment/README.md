# Value Alignment Experiment Pipeline

This package reproduces the core experiments from *From Value Conditioning to
Behavioral Shift: Lightweight Value Alignment of LLMs* and adds DPO and HyPO as
isolated comparison methods.

## Experiment Matrix

The main paper experiment trains one target-specific LoRA adapter for each of
the ten basic Schwartz values and each base model:

| Role | Methods | Models |
|---|---|---|
| Paper reproduction | survey SFT, matched baseline SFT, DPO | Qwen3-8B, Falcon3-7B-Instruct, Llama-3.1-8B-Instruct |
| Project extension | HyPO using the official trainer | same three models |

All model names are aliases in `value_alignment/configs/model_aliases.json`.
Mistral and Qwen2.5 aliases remain available for additional experiments.

## Data Roles

- `dataset/kvs_data_new.json` is the training and in-domain evaluation source.
  It contains 378 train, 108 validation, and 108 held-out test descriptions.
- Three task prompts and nine response templates expand every KVS description
  into 27 survey variants. This gives 10,206 SFT train rows, 2,916 validation
  rows, and 2,916 held-out test prompts.
- `dataset/aita_dataset_reduced.json` is evaluation-only. Refined annotations
  are mapped to ten basic values and capped at 500 examples per value, producing
  the paper's 2,902-example cross-domain test set.
- The KVS test split and all AITA examples are excluded from training.

The shared mapping from 20 refined labels to ten basic values is defined once in
`value_alignment/value_taxonomy.py`. In particular, `Face` maps to `Power` and
`Humility` maps to `Tradition`.

## Setup

Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r value_alignment/requirements.txt
```

Clone the official HyPO implementation used by the preference trainer:

```bash
mkdir -p third_party
git clone https://github.com/tmllab/2026_ICLR_HyPO.git third_party/2026_ICLR_HyPO
```

An existing clone can instead be selected with `HYPO_REPO=/path/to/repo`.
Llama 3.1 is gated on Hugging Face, so accept its license and authenticate on
the cluster before starting those jobs.

## 1. Prepare Static Data

These commands do not need a GPU:

```bash
python -m value_alignment.prepare_kvs_dpo
python -m value_alignment.prepare_kvs_eval
python -m value_alignment.prepare_aita_eval
```

They write:

```text
value_alignment/data/paper_preferences/<target>/down/{train,eval}.jsonl
value_alignment/data/kvs_survey/test.jsonl
value_alignment/data/aita_eval/test.jsonl
```

For a down-regulated target, DPO/HyPO target rows use the opposing KVS sentence
as `chosen` and the positive sentence as `rejected`. Non-target rows retain the
positive ordering as anchors. Every target dataset contains 378 train and 108
validation pairs.

## 2. Collect Model Baseline Ratings

Survey SFT needs model-specific baseline ratings for all non-target values. Run
all three models on SLURM:

```bash
sbatch value_alignment/slurm/collect_kvs_baselines_array.sh
```

For one local or interactive GPU run:

```bash
python -m value_alignment.collect_kvs_baseline_ratings \
  --model qwen3-8b \
  --output value_alignment/data/baseline_ratings/qwen3-8b.json
```

The collector evaluates all 27 prompt/template variants per description at
temperature 0.5 and assigns the model-specific baseline by majority vote.

## 3. Build Survey-SFT Data

Build a matched baseline dataset plus ten target datasets for each model after
its baseline-rating job finishes:

```bash
python -m value_alignment.prepare_kvs_sft \
  --baseline-ratings value_alignment/data/baseline_ratings/qwen3-8b.json \
  --output-root value_alignment/data/paper_sft/qwen3-8b
```

Repeat with `falcon3-7b` and `llama3.1-8b`. Target examples receive rating 1;
all non-target examples keep that model's measured baseline rating. The
`baseline/` control keeps baseline ratings for every value but has exactly the
same prompts and number of optimization examples.

## 4. Train SFT, DPO, and HyPO

The paper settings are the defaults: LoRA rank 256, learning rate `1e-4`, ten
epochs, warmup ratio 0.15, and early stopping with patience 2 and threshold
0.01. LoRA alpha is 1,024 for Falcon3 and 512 for Qwen/Llama.

Smoke-test Qwen baseline SFT and Security SFT:

```bash
sbatch --array=0,6 value_alignment/slurm/train_sft_array.sh
```

Run all 33 SFT jobs:

```bash
sbatch value_alignment/slurm/train_sft_array.sh
```

Smoke-test Qwen Security DPO and HyPO:

```bash
sbatch --array=5,15 value_alignment/slurm/train_dpo_hypo_array.sh
```

Run all 60 preference jobs:

```bash
sbatch value_alignment/slurm/train_dpo_hypo_array.sh
```

The default is the paper's bf16 LoRA setup. On a smaller GPU, enable NF4 QLoRA
for either training array without changing its dataset or output layout:

```bash
sbatch --export=ALL,QLORA=1 value_alignment/slurm/train_sft_array.sh
sbatch --export=ALL,QLORA=1 value_alignment/slurm/train_dpo_hypo_array.sh
```

`train_with_official_hypo.py` imports `DPOTrainer` and `DPOConfig` directly
from the official HyPO repository. Setting `--method dpo` disables the HyPO
reference-margin clipping; `--method hypo` enables it. Both methods consume the
same target-specific KVS pairs.

## 5. KVS Intrinsic Evaluation

The evaluator uses all 2,916 test prompts, temperature 0.5, and three seeded
stochastic runs. It supports an unmodified model or a PEFT adapter.

Smoke-test all Qwen Security conditions and their controls:

```bash
sbatch --array=0,1,7,17,27 value_alignment/slurm/evaluate_kvs_array.sh
```

Run all 96 model conditions:

```bash
sbatch value_alignment/slurm/evaluate_kvs_array.sh
```

After evaluation, create the complete comparison table:

```bash
python -m value_alignment.evaluation.summarize_kvs_experiments
```

For every model, method, and target this reports:

```text
Target Value Rating Drop = mean(base rating - conditioned rating)
Other Values' Fluctuation = mean(abs(conditioned rating - base rating))
```

Both metrics and their valid-example counts are computed separately in each of
the three stochastic runs, then reported as mean and sample standard deviation.
The per-prompt majority vote remains in the raw result JSON as a diagnostic and
is not used to collapse the three reported runs.

SFT targets are paired with the matched `sft_baseline` control. DPO and HyPO
targets are paired with the original base model.

## 6. AITA Behavioral Evaluation

AITA scoring normalizes sequence probabilities over `NTA`, `Neutral`, and
`YTA`. For each example:

```text
PG = delta(low-standard) - delta(high-standard) + 0.5 * delta(unused)
```

Positive Probability Gain means that down-regulating the target value moved
probability in the expected behavioral direction. The evaluator also reports a
strict version with unused weight zero and a one-sided t-test for mean PG > 0.

Smoke-test Qwen Security for SFT, DPO, and HyPO:

```bash
sbatch --array=5,15,25 value_alignment/slurm/evaluate_aita_array.sh
```

Run the complete 90-job evaluation:

```bash
sbatch value_alignment/slurm/evaluate_aita_array.sh
```

Aggregate the ten targets for one model/method:

```bash
python -m value_alignment.evaluation.summarize_aita_results \
  --inputs value_alignment/results/paper/aita/qwen3-8b/sft/*.json \
  --output-json value_alignment/results/paper/aita/qwen3-8b/sft_summary.json \
  --output-csv value_alignment/results/paper/aita/qwen3-8b/sft_summary.csv
```

The aggregate is weighted by the number of AITA examples in each value group,
matching the paper's uneven 2,902-example distribution.

## 7. Optional Extensions

- `generate_synthetic_preferences.py` creates persona-diverse preference pairs
  from KVS seeds using an OpenAI-compatible teacher endpoint. It requests a
  concise public rationale, not hidden chain-of-thought. Synthetic data is not
  part of the paper-reproduction condition and should be reported separately.
- `evaluation/machiavelli_hf_agent_template.py` provides a neutral Hugging Face
  agent template with the paper sampling settings: temperature 0.6, top-p 0.9,
  and top-k 20.
- Mistral-7B-Instruct and Qwen2.5-7B-Instruct can replace a paper model through
  the same `--model` interface without changing data code.

## Verification

Run CPU-side regression tests and syntax checks:

```bash
python -m unittest discover -s value_alignment/tests -v
PYTHONPYCACHEPREFIX=/tmp/lab-nlp-pycache python -m compileall -q value_alignment
bash -n value_alignment/slurm/*.sh
```

Full training and model scoring require the cluster GPU environment and model
downloads; the data builders and pure metric tests run locally.
