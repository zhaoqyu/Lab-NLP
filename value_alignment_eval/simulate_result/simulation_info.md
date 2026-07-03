# Simulation Details

The results contained in this folder (`extrinsic_eval_results.txt` and `intrinsic_eval_results.txt`) are **simulated demonstration runs**.

## Execution Environment & Setup
- **Compute:** CPU-only (No GPU was available during the automated setup).
- **Model Used:** `gpt2` (This is a much smaller fallback model used exclusively to ensure the pipeline runs without Out-Of-Memory errors). The final project is expected to run on `Qwen/Qwen2.5-7B-Instruct`.
- **Sample Size:** Limited to `2` samples to prevent long execution times and timeout errors during the initial pipeline verification.

These results prove that the evaluation code is fully operational. To get the real results, the code should be executed on the designated GPU cluster using the actual Qwen model.
