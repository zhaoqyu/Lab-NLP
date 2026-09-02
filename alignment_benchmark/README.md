# KVS Value-Alignment Benchmark

This package runs a controlled comparison of lightweight value-alignment methods. Every training and validation
example is derived from the existing KVS survey; AITA is used only for the final behavioral evaluation. The main
experiment uses one frozen base checkpoint, `Qwen/Qwen2.5-7B-Instruct`, so the independent variable is the alignment
method rather than model family.

## Compared Methods

| Family | Method | Learned object | Weight update | Matched control |
|---|---|---|---|---|
| Supervised | SFT | Scalar KVS rating | QLoRA | Base-model rating imitation |
| Reference preference | DPO | Chosen over rejected response | QLoRA | Affirming pairs |
| Mismatch-aware preference | HyPO | DPO with clipped reference margin | QLoRA | Affirming pairs |
| Reference preference | IPO | Squared preference-margin target | QLoRA | Affirming pairs |
| Reference-free preference | SimPO | Length-normalized preference margin | QLoRA | Affirming pairs |
| Monolithic preference | ORPO | SFT likelihood plus odds-ratio preference | QLoRA | Affirming pairs |
| Representation intervention | CAA block | Residual-stream contrast vector | None | Frozen base model |
| Representation intervention | CAA attention | Attention-output contrast vector | None | Frozen base model |

HyPO imports the official trainer from a pinned commit of
[`tmllab/2026_ICLR_HyPO`](https://github.com/tmllab/2026_ICLR_HyPO). DPO and IPO use the matching TRL `0.9.6`
implementation. They intentionally use the same base reference model here: this is a method comparison, not a
reproduction of HyPO's stronger-reference experiments.

## Fairness Contract

- KVS has immutable source splits: 378 train, 108 validation, and 108 test records.
- One Teacher call produces one canonical pair for each original KVS source. The Teacher may paraphrase but cannot
  invent a scenario, person, fact, or moral conflict.
- SFT, every preference method, and both steering variants receive exactly the same source IDs in each split.
- All hyperparameter and steering layer/strength decisions use KVS validation only.
- KVS test and AITA are locked until evaluation. AITA never enters training, pair construction, early stopping, or
  steering selection.
- Each trainable intervention is compared against a same-method, same-seed control adapter. Steering is compared
  against the same frozen base model.
- Main metrics macro-average refined values before averaging the ten basic values. Micro averages are retained in
  result files as sensitivity analyses.

The fixed Schwartz mapping and registered protocol are documented in
[`docs/experimental_protocol.md`](docs/experimental_protocol.md).

## Google Colab Workflow

Use a Colab GPU runtime and run the notebooks in order:

1. [`notebooks/01_build_data.ipynb`](notebooks/01_build_data.ipynb): install, mount Drive, generate Teacher data,
   audit it, collect frozen-model baselines, and build all method views.
2. [`notebooks/02_train_one_run.ipynb`](notebooks/02_train_one_run.ipynb): train one selected adapter. Repeat across
   Colab sessions, or use `valuebench run-next`.
3. [`notebooks/03_steer_and_evaluate.ipynb`](notebooks/03_steer_and_evaluate.ipynb): build CAA vectors and evaluate
   completed interventions.
4. [`notebooks/04_analyze_results.ipynb`](notebooks/04_analyze_results.ipynb): verify completeness, bootstrap the
   registered metrics, and export publication tables and figures.

Store `OPENROUTER_API_KEY` in Colab's Secrets panel. Never paste a key into a notebook, source file, command, output,
or Git commit. If a key has appeared in a chat or notebook, revoke it and create a replacement before continuing.

The Teacher defaults to OpenRouter model `openai/gpt-5.6-sol` with maximal reasoning, strict JSON Schema output, and
provider data collection disabled. The pipeline stores a concise public rationale, never private chain-of-thought.
Generation is incremental and atomic, so interrupted calls resume without repeating completed rows.

## Run Matrix

The paper configuration registers:

- 198 adapter fits: 6 methods x 11 targets (10 interventions + control) x 3 seeds.
- 20 CAA searches: 10 target values x 2 hook sites.
- 200 locked evaluations: 180 adapter interventions + 20 CAA interventions.

This is intentionally larger than a single Colab session. Set `VALUEBENCH_OUTPUT_ROOT` to a Google Drive directory;
every command writes a `DONE` marker and resumes safely. `valuebench status` reports progress, while `valuebench
run-next` executes one ready evaluation or one pending construction run.

## Command Reference

```bash
valuebench doctor
valuebench teacher
valuebench audit-data --fail-on-quality
valuebench prepare-aita
valuebench collect-baselines
valuebench summarize-mismatch
valuebench build-views
valuebench validate-data
valuebench make-plan

valuebench train --method dpo --target Security --seed 13
valuebench build-steering --target Security --site block
valuebench evaluate --method dpo --target Security --seed 13

valuebench status
valuebench run-next
valuebench aggregate-results
valuebench paper-artifacts
```

All commands accept `--config`. Outputs are rooted at `VALUEBENCH_OUTPUT_ROOT` when that environment variable is set,
otherwise at `alignment_benchmark/artifacts/paper`.

## Outputs

The pipeline produces canonical JSONL and quality reports, immutable method views and checksums, QLoRA adapters,
steering vectors and validation grids, raw per-example Parquet scores, aggregate CSV tables, bootstrap confidence
intervals, FDR-adjusted per-value tests, PNG/PDF figures, LaTeX tables, efficiency summaries, and a final SHA-256
manifest. See [`docs/output_schema.md`](docs/output_schema.md) for the full layout.

Actual claims, numbers, and plots are created only after the registered GPU runs finish. The code never substitutes
mock values for missing experiments; final aggregation fails on an incomplete matrix unless explicitly invoked with
`--allow-incomplete` for debugging.
