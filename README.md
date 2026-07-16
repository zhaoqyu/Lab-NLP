# Value Alignment in Large Language Models using HyDPO

This repository contains the unified, end-to-end pipeline for generating, training, and rigorously evaluating human-aligned value models using Hybrid Direct Preference Optimization (HyDPO). 

## Architecture

The project has been refactored from isolated components into a single, cohesive `src/` directory to facilitate seamless SLURM cluster orchestration:

- **`src/data/`**: Pipeline for synthetic data generation and preference pair structuring (AITA scenarios, KVS).
- **`src/training/`**: SLURM-compatible training modules orchestrating the HyDPO base algorithms.
- **`src/evaluation/`**: Mathematical evaluation scripts computing Intrinsic (Target Value Rating Drop) and Extrinsic (Probability Gain) metrics over pre-generated logs.

## Quick Start

You can run the entire sequential pipeline using the master orchestration script:

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```
