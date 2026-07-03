# Value Alignment Evaluation

This directory contains the specific evaluation environment and scripts for the value alignment project, kept separate from the main repository files.

## Directory Structure

- `src/`: Contains the evaluation scripts.
  - `intrinsic_eval.py`: Evaluates the "Target Value Rating Drop" and "Other Values' Variance".
  - `extrinsic_eval.py`: Evaluates "Probability Gain" and downstream behavioral changes using Causal LM.
- `setup_machiavelli.sh`: Script to clone and setup the MACHIAVELLI benchmark.
- `requirements.txt`: Python dependencies required for the project.

**Note:** The evaluation scripts expect the datasets to be located in the `dataset/` directory at the root of the repository (i.e. `../dataset/`).

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
