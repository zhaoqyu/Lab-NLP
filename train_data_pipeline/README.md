# train_data_pipeline

**A complete, standalone, run-and-forget pipeline: data -> training -> evaluation ->
result tables.** Everything it needs is inside this one folder — no other part of the
repo has to be checked out or understood first. Assume zero results exist; this is what
gets us from nothing to the two CSVs the report and the presentation are waiting on.

## TL;DR

```bash
cd train_data_pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r value_alignment/requirements.txt
nohup python3 run_priority_pipeline.py > run.log 2>&1 &
tail -f run.log
```

Walk away. Come back in 6-12 hours (see the timing table below) to
`value_alignment/results/paper/kvs_summary.csv` and
`value_alignment/results/paper/aita_summary.csv`.

**Two interchangeable ways to run this** — same stages, same output, same `.done`
markers, pick whichever actually runs on your server:
- `python3 run_priority_pipeline.py` — **use this one if in doubt.** Needs nothing but
  the Python interpreter you already have (the whole pipeline is Python anyway). No
  shell scripting involved, so it doesn't care whether bash is installed, whether the
  script file has its execute bit set, or whether the filesystem is mounted `noexec` —
  all common reasons a `.sh` file refuses to run on a locked-down shared/HPC server with
  no root access.
- `bash run_priority_pipeline.sh` — the original shell version. Identical behavior,
  use it if your server is a normal box where bash scripts just work.

Both write to the exact same `value_alignment/slurm_logs/priority_run/` markers, so you
can even start with one and resume with the other if you switch servers mid-run.

## Everything in this folder

```
train_data_pipeline/
  README.md                        <- this file
  run_priority_pipeline.py         <- RECOMMENDED entry point. Pure Python, no shell
                                       needed at all -- run it with `python3
                                       run_priority_pipeline.py`.
  run_priority_pipeline.sh         <- the same pipeline as a bash script, for servers
                                       where bash scripts run normally. Functionally
                                       identical to the .py version above; pick one.
  dataset/
    kvs_data_new.json              <- KVS value survey: 378/108/108 train/eval/test
    aita_dataset_reduced.json      <- AITA moral-dilemma posts, evaluation-only
  value_alignment/                 <- full copy of the project's Python package
    model_utils.py                 <- model name aliases (qwen2.5-7b -> HF repo id)
    value_taxonomy.py               <- 20 refined -> 10 basic Schwartz value mapping
    survey_data.py                  <- KVS survey-variant expansion (27 templates)
    experiment_utils.py             <- run manifests, reproducibility metadata
    prepare_kvs_dpo.py               <- builds DPO/HyPO/SimPO/KTO preference pairs
    prepare_kvs_eval.py              <- builds the held-out KVS survey eval set
    prepare_aita_eval.py             <- builds the neutral AITA eval set
    prepare_kvs_sft.py                <- builds SFT baseline-control + target data
    collect_kvs_baseline_ratings.py   <- gets the base model's own 1-6 ratings
    train_survey_sft.py               <- SFT trainer
    train_with_official_hypo.py       <- DPO / HyPO trainer (official HyPO loss)
    train_extra_methods.py            <- SimPO / KTO trainer (new, added for this run)
    evaluate_kvs_survey.py            <- intrinsic KVS evaluation (any base/adapter)
    activation_steering.py            <- imported by the evaluator; steering itself
                                          is not part of this particular run
    evaluation/
      evaluate_aita_probability_gain.py  <- extrinsic AITA evaluation
      summarize_kvs_experiments.py       <- aggregates all KVS results into one CSV
      summarize_aita_experiments.py      <- aggregates all AITA results into one CSV
      compare_kvs_results.py, aita_metrics.py, statistics_utils.py, ...  <- metric code
    configs/model_aliases.json         <- model alias -> Hugging Face repo id
    slurm/_common.sh                    <- environment-setup helpers (no SLURM needed;
                                            run_priority_pipeline.sh just reuses these
                                            same helper functions directly)
    tests/                               <- 21 regression tests, run as stage 0
    requirements.txt                     <- exact pinned dependencies
```

This is a full copy of the `value_alignment/` package as it exists on branch
`mike-colab-valuebench` of this repo, plus three additions: `train_extra_methods.py`
(SimPO + KTO, did not exist before) and the two orchestration scripts,
`run_priority_pipeline.py` and `run_priority_pipeline.sh` (both new, functionally
identical, pick whichever runs on your server). Two files got a one-line edit each
(`evaluation/summarize_kvs_experiments.py`, `evaluation/summarize_aita_experiments.py`)
so their `--methods` flag recognizes `simpo`/`kto` in addition to the methods that were
already there. Nothing else was touched. Confirmed by running, from inside this exact
folder: `python -m compileall value_alignment` and
`python -m unittest discover -s value_alignment/tests` (21/21 pass).

## What this run actually covers

- **Model**: `Qwen/Qwen2.5-7B-Instruct` (alias `qwen2.5-7b`) — one model.
- **Values** (5 of 10 Schwartz values — the ones with the most KVS/AITA data, so the
  numbers aren't noise): Universalism, Security, Benevolence, Self-direction, Power.
- **Methods**: SFT (a matched baseline-control adapter + one target adapter per value),
  DPO, HyPO, SimPO, KTO. One seed (42).
- **SimPO** = TRL's `CPOTrainer` with `loss_type="simpo"`, `cpo_alpha=0` (pure SimPO, no
  BC regularizer). Reference-free — no second model copy is ever loaded.
- **KTO** = TRL's `KTOTrainer`, fed unpaired `(prompt, completion, label)` rows built by
  exploding the existing chosen/rejected preference pairs into one desirable + one
  undesirable row each. With a PEFT adapter and no explicit `--ref-model`, it reuses the
  same "policy with the adapter disabled" trick DPO/HyPO already use for the reference
  model, so this doesn't load a second model copy either.
- This intentionally skips Falcon3-7B/Llama-3.1-8B and the other 5 values (part of the
  older 3-model/10-value/3-seed paper-reproduction plan) — this scope is what can
  realistically finish and be trustworthy within a week. Every choice above is a
  command-line flag or environment variable, not a hardcoded constant, so widening scope
  later re-uses the exact same script (see "Adjusting scope" below).

## Before you start: verify the prerequisites

Run these on the server *before* kicking off a run that might take half a day, so a
missing prerequisite fails in 10 seconds instead of 5 hours in:

```bash
nvidia-smi                                     # confirms a CUDA GPU is visible
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python3 -c "import bitsandbytes"                # only needed if QLORA=1 (the default)
curl -sI https://huggingface.co | head -1       # must NOT be blocked
curl -sI https://github.com | head -1           # must NOT be blocked (HyPO repo clone)
df -h .                                          # need ~25-30 GB free
```

If `huggingface.co` or `github.com` is blocked from this server the same way it was
blocked from Ali's own connected desktop sandbox (a 403 from a proxy), the script will
hang or fail at the model-download or HyPO-clone step and there is no workaround from
inside the script — that has to be fixed at the network/proxy level first.

## Exact steps

```bash
cd train_data_pipeline

python3 -m venv .venv
source .venv/bin/activate
pip install -r value_alignment/requirements.txt

# recommended: a ~10-15 minute smoke test on 1 value / 1 epoch before the full run,
# so a bug shows up in minutes instead of after hours of real training
VALUES_STR="security" PREF_EPOCHS=1 SFT_EPOCHS=1 EVAL_NUM_RUNS=1 \
  python3 run_priority_pipeline.py
cat value_alignment/slurm_logs/priority_run/status.tsv   # everything should say OK

# if the smoke test is all OK, wipe its markers/outputs and do the real run
rm -rf value_alignment/slurm_logs/priority_run value_alignment/checkpoints \
       value_alignment/results value_alignment/data/paper_sft \
       value_alignment/data/baseline_ratings

nohup python3 run_priority_pipeline.py > run.log 2>&1 &
tail -f run.log
```

`nohup ... &` keeps it running if the SSH session drops. That's the whole job — start it
and walk away. (If bash scripts run fine on your server and you'd rather use the shell
version, swap `python3 run_priority_pipeline.py` for `bash run_priority_pipeline.sh`
everywhere above — both understand the same environment variables and write to the same
`.done` markers.)

## What happens, in order

1. `00_regression_tests` — the 21 existing unit tests, as a sanity check.
2. `01_clone_hypo_repo` — clones `tmllab/2026_ICLR_HyPO` into `third_party/` (needed by
   DPO/HyPO; skipped if already present).
3. `02_prepare_kvs_dpo`, `03_prepare_kvs_eval`, `04_prepare_aita_eval` — CPU-only, fast
   (seconds). Already smoke-tested for real on Ali's machine with these exact values.
4. `05_collect_baseline_ratings` — runs the base model on all 594 KVS descriptions x 27
   survey variants to get its own 1-6 rating for each (first GPU-heavy step).
5. `06_prepare_kvs_sft` — turns those baseline ratings into SFT training data.
6. `10_train_sft_baseline`, `10_train_sft_<value>` x 5 — SFT training, 6 adapters.
7. `20_train_{dpo,hypo,simpo,kto}_<value>` x 5 values — 20 preference-method adapters.
8. `30_eval_kvs_*` — KVS intrinsic evaluation for the base model + all 26 adapters.
9. `40_eval_aita_*` — AITA behavioral evaluation for all 26 adapters.
10. `50_summarize_kvs`, `50_summarize_aita` — the two final aggregate tables.

Every one of these is a separate stage with its own `.done` marker under
`value_alignment/slurm_logs/priority_run/<stage>.done` and its own log at
`value_alignment/slurm_logs/priority_run/<stage>.log`. **Re-running the exact same
command (`python3 run_priority_pipeline.py` or `bash run_priority_pipeline.sh`, either
one) resumes from wherever it left off** instead of starting over — safe after a disconnect, an OOM on one job, or a reboot. A stage that
fails is logged and skipped, not fatal to the rest of the run: check
`value_alignment/slurm_logs/priority_run/status.tsv` (one line per stage: `OK`, `FAIL`,
or `SKIP`) to see the full picture at a glance, and the matching `.log` file for any
`FAIL` to see exactly what went wrong.

## What you get when it's done

**`value_alignment/results/paper/kvs_summary.csv`** — one row per (model, method,
target value), columns include:
`target_value_rating_drop`, `target_value_rating_drop_ci_95_low/high`,
`other_values_mean_absolute_fluctuation` (+ its CI), `target_source_cluster_count`, and
the same numbers aggregated across training runs. This is Target Value Rating Drop vs.
Other Values' Fluctuation — the pair that has to be read together (a big drop with a big
fluctuation is collateral damage, not precise value control). Feeds
`Final_Report/sections/06_results.tex` table `tab:kvs-main` and the presentation's
"Results — Intrinsic Value Shift" slide.

**`value_alignment/results/paper/aita_summary.csv`** — one row per (model, method),
columns include: `weighted_average_probability_gain` (+ 95% bootstrap CI),
`macro_average_probability_gain`, `strict_weighted_average_probability_gain` (the
sensitivity variant), `weighted_expected_direction_effect`, and `fdr_significant_values`
(how many of the 5 target values survive Benjamini-Hochberg correction). Feeds table
`tab:aita-main` and the "Results — Out-of-Domain Behaviour" slide.

**Everything behind those two numbers is also there and traceable:**
- `value_alignment/checkpoints/paper_sft/qwen2.5-7b/{baseline,<value>}/final/` — 6 SFT
  LoRA adapters.
- `value_alignment/checkpoints/paper_preference/qwen2.5-7b/<value>/{dpo,hypo,simpo,kto}/final/`
  — 20 preference-method LoRA adapters.
- A `run_manifest.json` next to every adapter: exact CLI arguments, git commit,
  installed package versions, SHA-256 hashes of the training data, GPU/hostname, elapsed
  time — enough to answer "what exactly produced this number" for any cell in the table.
- `value_alignment/results/paper/kvs/qwen2.5-7b/*.json` and
  `.../aita/qwen2.5-7b/*/*.json` — the raw per-condition scores the two summary CSVs were
  built from, in case a different aggregation or a spot-check is ever needed.

At that point the project has real numbers instead of `\num{--}` placeholders, and the
next step is genuinely just: read the two CSVs, fill in `06_results.tex` and the results
slides, and write the analysis section — no more pipeline work should be needed unless
the report calls for a wider run (more values, more seeds, the other two models).

## How long this takes

Nobody has run this exact matrix before — this is an estimate, not a measurement, and it
depends heavily on the GPU and on how fast the one-time model download goes:

| Stage | Rough time |
|---|---|
| Qwen2.5-7B-Instruct download (first run only, ~15 GB) | 10-30 min, network-dependent |
| Baseline KVS rating collection (594 descriptions x 27 variants) | 30-60 min |
| SFT training (6 adapters: baseline control + 5 targets) | ~1-1.5 h |
| Preference training (20 adapters: 4 methods x 5 values, 378-756 pairs each) | ~2-4 h |
| KVS evaluation (27 conditions x 2,916 held-out prompts x 3 stochastic runs) | ~2-4 h |
| AITA evaluation (26 conditions, up to 500 examples/value, deterministic) | ~30-60 min |
| Aggregation | seconds |
| **Total** | **~6-12 hours of continuous single-GPU time** |

That comfortably fits in a day if started right away, leaving the rest of the week for
writing up results and a second pass if anything needs re-running. The smoke test above
(1 value, 1 epoch) should take well under 20 minutes and is the fastest way to catch an
environment problem before committing to the full run.

## Adjusting scope

Every knob is an environment variable, read the same way by both entry points:

```bash
VALUES_STR="universalism security benevolence self_direction power tradition" \
MODEL=qwen3-8b SEED=43 \
  python3 run_priority_pipeline.py
```

Available: `MODEL`, `VALUES_STR` (space-separated basic Schwartz values, lowercase with
underscores), `SEED`, `SFT_EPOCHS`, `PREF_EPOCHS`, `LORA_R`, `LORA_ALPHA`, `BATCH_SIZE`,
`GRAD_ACCUM`, `QLORA` (`0` to disable 4-bit and use full bf16 LoRA), `EVAL_NUM_RUNS`.

## Troubleshooting

- **The `.sh` script won't run at all, even with `bash run_priority_pipeline.sh`** —
  don't debug it, just switch to `python3 run_priority_pipeline.py` instead (same
  environment variables, same output, same `.done` markers). This is exactly what it's
  for: a shared/HPC server with no root access, a `noexec`-mounted home or scratch
  partition, or no bash installed will all block a `.sh` file in ways a plain `python3
  script.py` invocation never hits, since nothing is trying to execute the file itself
  as a program — only the Python interpreter is.
- **CUDA out of memory during training** — set `QLORA=1` if it isn't already (it's the
  default), lower `LORA_R` (e.g. `32`), or lower `BATCH_SIZE`/raise `GRAD_ACCUM` to keep
  the same effective batch size with less peak memory.
- **`bitsandbytes` import/build errors** — usually a CUDA-toolkit/driver mismatch; try
  `QLORA=0` (full bf16 LoRA, needs more VRAM but sidesteps bitsandbytes) as a fallback
  while that's investigated separately.
- **Stuck / very slow at `01_clone_hypo_repo` or the first model load** — almost always
  network access to `github.com` or `huggingface.co` is blocked or throttled; verify with
  the `curl` commands above rather than waiting it out.
- **One stage's `.log` shows a real error and you fixed the underlying cause** — just
  delete that one `value_alignment/slurm_logs/priority_run/<stage>.done` marker (or the
  whole `priority_run/` directory to redo everything) and re-run the same command; it
  will only redo what's missing.
- **Need to hand this off mid-run** — `value_alignment/slurm_logs/priority_run/status.tsv`
  and the `run.log` from `nohup` together show exactly where things stand for whoever
  picks it up next.

## Context

This folder is a deliberately self-contained duplicate of `value_alignment/` from
branch `mike-colab-valuebench` of this repo (plus the two new files and two one-line
patches described above) — everything needed to run lives here so nobody has to
cross-reference the rest of the repo structure to use it. The canonical, ongoing
development copy of the pipeline still lives at `value_alignment/` on that branch;
this folder is the run-this-and-get-results package, not a fork of the project.
