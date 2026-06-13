# Value Alignment DPO / HyPO

This folder adds project code for comparing standard DPO and HyPO/Hybrid-DPO on the lab value-alignment data.

## Data Roles

- `dataset/aita_dataset_reduced.json`: converted into `prompt/chosen/rejected` preference pairs for DPO/HyPO training.
- `dataset/kvs_data_new.json`: converted into 1-6 survey prompts for value evaluation before and after training.

## Setup

Install Python dependencies in a GPU environment:

```bash
pip install -r value_alignment/requirements.txt
```

The training wrapper uses the official HyPO implementation. Clone it inside the repo:

```bash
mkdir -p third_party
git clone https://github.com/tmllab/2026_ICLR_HyPO.git third_party/2026_ICLR_HyPO
```

Alternatively:

```bash
export HYPO_REPO=/path/to/2026_ICLR_HyPO
```

## Prepare AITA Preference Data

```bash
python value_alignment/prepare_aita_dpo.py \
  --values Security_personal Benevolence_caring Universalism_concern \
  --output-dir value_alignment/data/aita_dpo
```

This creates:

```text
value_alignment/data/aita_dpo/train.jsonl
value_alignment/data/aita_dpo/eval.jsonl
value_alignment/data/aita_dpo/test.jsonl
```

## Prepare KVS Survey Evaluation Data

```bash
python value_alignment/prepare_kvs_eval.py \
  --split test \
  --output value_alignment/data/kvs_test_eval.jsonl
```

## Train Standard DPO

```bash
python value_alignment/train_with_official_hypo.py \
  --method dpo \
  --model Qwen/Qwen2.5-1.5B-Instruct
```

## Train HyPO / Hybrid-DPO

```bash
python value_alignment/train_with_official_hypo.py \
  --method hypo \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --gamma 0.0
```

## Evaluate On KVS

Base model:

```bash
python value_alignment/evaluate_kvs_survey.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --eval-file value_alignment/data/kvs_test_eval.jsonl \
  --output value_alignment/results/kvs_base_scores.json
```

Trained model:

```bash
python value_alignment/evaluate_kvs_survey.py \
  --model value_alignment/checkpoints/hypo/final \
  --eval-file value_alignment/data/kvs_test_eval.jsonl \
  --output value_alignment/results/kvs_hypo_scores.json
```

## DPO vs HyPO Difference

Standard DPO uses:

```python
logits = delta_policy - delta_ref
```

HyPO uses:

```python
logits = delta_policy - max(0, delta_ref)
```

This keeps reference-model regularization when the reference already prefers the chosen response, but avoids weakening the learning signal when the reference prefers the rejected response.
