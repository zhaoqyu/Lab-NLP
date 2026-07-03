# Value Alignment Evaluation

This directory isolates the evaluation scripts and environments developed for the Value Alignment project. It focuses strictly on assessing both the intrinsic shifts in moral representation and the extrinsic behavioral changes in the aligned model.

## Directory and File Breakdown

Here is a highly detailed breakdown of what each file in this folder does and how to execute it.

### 1. `src/intrinsic_eval.py`
**Purpose:** Calculates internal shifts in the model's value representation.
- **What it runs:** It parses the JSON representation of model outputs from the `kvs_data_new.json` dataset.
- **Metrics Calculated:** 
  - *Target Value Rating Drop:* How much the model decreases its agreement with negative or manipulated values.
  - *Other Values' Variance:* A stability check ensuring that positive values are not forgotten.
- **How to run:** `python src/intrinsic_eval.py`

### 2. `src/extrinsic_eval.py`
**Purpose:** Evaluates the model's behavior in complex, real-world social dilemmas.
- **What it runs:** It loads the language model (e.g., `Qwen2.5-7B-Instruct`), dynamically tokenizes responses, and computes forward passes (without gradients) on scenarios from the `aita_dataset_reduced.json` dataset.
- **Metrics Calculated:**
  - *Probability Gain:* The mathematical shift in probability mass towards the ethical or value-aligned stance (e.g., predicting "NTA" instead of "YTA").
- **How to run:** `python src/extrinsic_eval.py`

### 3. `setup_machiavelli.sh`
**Purpose:** Prepares the advanced MACHIAVELLI behavioral benchmark.
- **What it runs:** It executes a series of bash commands to clone the official MACHIAVELLI repository (`aypan17/machiavelli`), initializes an isolated Python virtual environment, and installs the required dependencies to run the text-based adventure games. 
- **Metrics Evaluated via Benchmark:** Power-Seeking Score, Moral Violations Score, Disutility Score.
- **How to run:** `./setup_machiavelli.sh`

### 4. `simulate_result/` (Directory)
**Purpose:** Contains demonstration outputs proving the pipeline works.
- **Details:** Since running the 7B parameter model requires significant GPU VRAM, this folder stores output logs generated using a smaller fallback model (`gpt2`) on the CPU with a truncated sample size. Check the `simulation_info.md` inside for more details.

## General Setup Instructions

To properly configure the environment before running the evaluation scripts:
```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install all core required packages
pip install -r requirements.txt

# 3. Setup the Machiavelli environment separately
./setup_machiavelli.sh
```

**Note:** The evaluation scripts assume the datasets are located in the main repository's root `dataset/` directory (e.g., `../dataset/aita_dataset_reduced.json`).
