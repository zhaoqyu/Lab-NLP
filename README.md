# Lab NLP: Value Alignment Evaluation

This repository contains the evaluation environment and scripts for the value alignment project.

## Directory Structure

- `data/`: Contains the datasets used for evaluation.
  - `aita_dataset_reduced.json`: Used for extrinsic evaluation.
  - `kvs_data_new.json`: Used for intrinsic evaluation.
- `src/`: Contains the evaluation scripts.
  - `intrinsic_eval.py`: Evaluates the "Target Value Rating Drop" and "Other Values' Variance".
  - `extrinsic_eval.py`: Evaluates "Probability Gain" and downstream behavioral changes using Causal LM.
- `setup_machiavelli.sh`: Script to clone and setup the MACHIAVELLI benchmark.
- `requirements.txt`: Python dependencies required for the project.

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup MACHIAVELLI Benchmark:**
   ```bash
   ./setup_machiavelli.sh
   ```

## Usage

**Intrinsic Evaluation:**
```bash
python src/intrinsic_eval.py
```

**Extrinsic Evaluation:**
```bash
python src/extrinsic_eval.py
```
