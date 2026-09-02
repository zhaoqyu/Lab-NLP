# Output Schema

`VALUEBENCH_OUTPUT_ROOT` contains the complete experiment state:

```text
data/
  canonical/{train,eval,test}.jsonl
  teacher_jobs.jsonl
  quality/teacher_audit.csv
  quality/teacher_audit_summary.json
  aita.jsonl
  views/
    sft/<target>/{train,eval,test}.jsonl
    preference/<target>/{train,eval,test}.jsonl
    steering/{train,eval,test}.jsonl
    manifest.json
baselines/
  kvs_ratings.{csv,parquet}
  reference_margins.{csv,parquet}
  reference_mismatch.parquet
checkpoints/<method>/<target>/seed-<seed>/
  final/
  trainer_state.json
  manifest.json
  DONE
steering/<site>/<target>/
  vector_layer_<layer>.pt
  selection_grid.csv
  selected_vector.pt
  selected.json
  DONE
results/
  raw/<method>/<target>/seed-<seed>/{kvs,aita}.parquet
  raw/<method>/<target>/seed-<seed>/summary.json
  aggregate/{per_run,per_target,method_summary,efficiency}.csv
  aggregate/reference_mismatch.csv
  completeness.{csv,json}
paper/
  table_*.{csv,tex}
  figure_*.{png,pdf}
  paper_context.json
  manifest.json
experiment_plan.csv
experiment_plan_summary.json
```

Raw KVS Parquet rows contain control/intervention rating distributions, expected ratings, drop, drift, target flag,
method, target, seed, basic value, refined value, and source ID. Raw AITA rows contain both label distributions,
high/low-standard stances, primary/strict probability gains, and the same provenance keys.

Every model run manifest records package versions, hardware, hashes of train/eval views, elapsed time, peak allocated
GPU memory, trainable parameter count, metrics, and the pinned HyPO commit where applicable. The final paper manifest
adds SHA-256 and byte size for every exported table and figure.
