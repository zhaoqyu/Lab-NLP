#!/usr/bin/env python3
"""One-shot, pure-Python pipeline for the crunch week (single model, reduced
value subset, one seed). Trains SFT baseline+targets, DPO, HyPO, SimPO, KTO
for a handful of Schwartz values, evaluates all of them on KVS + AITA, and
writes the final aggregate tables.

This is a straight Python re-implementation of run_priority_pipeline.sh, for
servers where an ordinary user cannot execute a shell script at all (no
execute bit even after chmod +x, a noexec-mounted filesystem, bash missing or
restricted, etc.). It needs nothing but a Python interpreter that can already
import the packages in value_alignment/requirements.txt -- no bash, no sh, no
sbatch. Every external command below is invoked as a subprocess, exactly the
same commands the shell version ran.

Usage, from inside this folder (train_data_pipeline/), with your virtualenv
already activated:

    python3 run_priority_pipeline.py

or, so it survives a dropped SSH session:

    nohup python3 run_priority_pipeline.py > run.log 2>&1 &
    tail -f run.log

Safe to re-run / resume: every stage writes a ".done" marker under
value_alignment/slurm_logs/priority_run/ and is skipped on the next run if
its marker exists. Delete a marker (or the whole priority_run/ directory) to
force that stage to redo. A failed stage is logged and skipped -- it does
NOT stop the rest of the pipeline.

Override any of these with environment variables before running, e.g.:
    MODEL=qwen3-8b VALUES_STR="security benevolence" python3 run_priority_pipeline.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path.cwd()))

MODEL = os.environ.get("MODEL", "qwen2.5-7b")
VALUES = os.environ.get(
    "VALUES_STR", "universalism security benevolence self_direction power"
).split()
SEED = os.environ.get("SEED", "42")
SFT_EPOCHS = os.environ.get("SFT_EPOCHS", "5")
PREF_EPOCHS = os.environ.get("PREF_EPOCHS", "3")
LORA_R = os.environ.get("LORA_R", "64")
LORA_ALPHA = os.environ.get("LORA_ALPHA", "128")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "1")
GRAD_ACCUM = os.environ.get("GRAD_ACCUM", "16")
QLORA = os.environ.get("QLORA", "1")
EVAL_NUM_RUNS = os.environ.get("EVAL_NUM_RUNS", "3")
PREF_METHODS = ["dpo", "hypo", "simpo", "kto"]

EXTRA_4BIT_FLAG = ["--load-in-4bit"] if QLORA == "1" else []

RUN_LOG_DIR = REPO_ROOT / "value_alignment" / "slurm_logs" / "priority_run"
STATUS_FILE = RUN_LOG_DIR / "status.tsv"


def setup_env() -> None:
    if not (REPO_ROOT / "value_alignment" / "model_utils.py").exists():
        raise SystemExit(
            f"Could not find value_alignment/model_utils.py under {REPO_ROOT}. "
            "Run this from inside the train_data_pipeline/ folder, or set "
            "REPO_ROOT=/path/to/train_data_pipeline."
        )
    os.environ.setdefault(
        "HF_HOME", str(Path(os.environ.get("SCRATCH", Path.home())) / ".cache" / "huggingface")
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.touch(exist_ok=True)


def check_gpu() -> None:
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("WARNING: nvidia-smi not found on PATH.")
    try:
        import torch

        print("python packages:")
        print("  torch:", torch.__version__)
        print("  cuda_available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("  gpu:", torch.cuda.get_device_name(0))
            print("  bf16_supported:", torch.cuda.is_bf16_supported())
        else:
            print("WARNING: CUDA is unavailable; training/evaluation will be very slow or will fail.")
    except ImportError as exc:
        print(f"WARNING: could not import torch to check the GPU ({exc}).")


def log_status(status: str, name: str) -> None:
    with STATUS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"{status}\t{name}\t{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")


def run_step(name: str, argv: list[str]) -> None:
    marker = RUN_LOG_DIR / f"{name}.done"
    logfile = RUN_LOG_DIR / f"{name}.log"
    if marker.exists():
        print(f"[SKIP] {name} (already done; rm {marker} to redo)")
        log_status("SKIP", name)
        return
    print(f"[RUN ] {name}")
    print(f"        {' '.join(argv)}")
    with logfile.open("w", encoding="utf-8") as handle:
        result = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
    if result.returncode == 0:
        marker.touch()
        print(f"[ OK ] {name}")
        log_status("OK", name)
    else:
        print(f"[FAIL] {name} -- see {logfile}")
        log_status("FAIL", name)


def clone_hypo_repo() -> None:
    dest = REPO_ROOT / "third_party" / "2026_ICLR_HyPO"
    if (dest / "hypo_config.py").exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/tmllab/2026_ICLR_HyPO.git", str(dest)],
        check=True,
    )


def run_step_callable(name: str, func) -> None:
    marker = RUN_LOG_DIR / f"{name}.done"
    logfile = RUN_LOG_DIR / f"{name}.log"
    if marker.exists():
        print(f"[SKIP] {name} (already done; rm {marker} to redo)")
        log_status("SKIP", name)
        return
    print(f"[RUN ] {name}")
    try:
        with logfile.open("w", encoding="utf-8") as handle:
            import contextlib

            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                func()
        marker.touch()
        print(f"[ OK ] {name}")
        log_status("OK", name)
    except Exception as exc:  # noqa: BLE001 -- log and continue, never abort the whole run
        with logfile.open("a", encoding="utf-8") as handle:
            handle.write(f"\nEXCEPTION: {exc!r}\n")
        print(f"[FAIL] {name} -- see {logfile}")
        log_status("FAIL", name)


def py(module: str, *args: str) -> list[str]:
    return [sys.executable, "-u", "-m", module, *args]


def main() -> None:
    setup_env()
    check_gpu()

    print("=" * 70)
    print(f" Priority run started: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f" Model:  {MODEL}")
    print(f" Values: {' '.join(VALUES)}")
    print(f" Seed:   {SEED}   QLoRA: {QLORA}   LoRA r={LORA_R} alpha={LORA_ALPHA}")
    print("=" * 70)

    # ------------------------------------------------------------ Stage 0 --
    run_step("00_regression_tests", [sys.executable, "-m", "unittest", "discover", "-s", "value_alignment/tests", "-v"])

    # ------------------------------------------------------------ Stage 1 --
    run_step_callable("01_clone_hypo_repo", clone_hypo_repo)
    run_step("02_prepare_kvs_dpo", py("value_alignment.prepare_kvs_dpo", "--target-values", *VALUES, "--seed", SEED))
    run_step("03_prepare_kvs_eval", py("value_alignment.prepare_kvs_eval", "--split", "test"))
    run_step("04_prepare_aita_eval", py("value_alignment.prepare_aita_eval", "--values", *VALUES, "--seed", SEED))

    # ------------------------------------------------------------ Stage 2 --
    baseline_ratings = f"value_alignment/data/baseline_ratings/{MODEL}.json"
    run_step(
        "05_collect_baseline_ratings",
        py("value_alignment.collect_kvs_baseline_ratings", "--model", MODEL, "--output", baseline_ratings, "--seed", SEED),
    )
    sft_data_root = f"value_alignment/data/paper_sft/{MODEL}"
    run_step(
        "06_prepare_kvs_sft",
        py(
            "value_alignment.prepare_kvs_sft",
            "--baseline-ratings", baseline_ratings,
            "--output-root", sft_data_root,
            "--target-values", *VALUES,
        ),
    )

    # ------------------------------------------------------------ Stage 3 --
    sft_ckpt_root = f"value_alignment/checkpoints/paper_sft/{MODEL}"

    run_step(
        "10_train_sft_baseline",
        py(
            "value_alignment.train_survey_sft",
            "--model", MODEL,
            "--train-file", f"{sft_data_root}/baseline/train.jsonl",
            "--eval-file", f"{sft_data_root}/baseline/eval.jsonl",
            "--output-dir", f"{sft_ckpt_root}/baseline",
            "--epochs", SFT_EPOCHS, "--lora-r", LORA_R, "--lora-alpha", LORA_ALPHA,
            "--batch-size", BATCH_SIZE, "--grad-accum", GRAD_ACCUM, "--seed", SEED,
            "--gradient-checkpointing", *EXTRA_4BIT_FLAG,
        ),
    )
    for value in VALUES:
        run_step(
            f"10_train_sft_{value}",
            py(
                "value_alignment.train_survey_sft",
                "--model", MODEL,
                "--train-file", f"{sft_data_root}/{value}/down/train.jsonl",
                "--eval-file", f"{sft_data_root}/{value}/down/eval.jsonl",
                "--output-dir", f"{sft_ckpt_root}/{value}",
                "--epochs", SFT_EPOCHS, "--lora-r", LORA_R, "--lora-alpha", LORA_ALPHA,
                "--batch-size", BATCH_SIZE, "--grad-accum", GRAD_ACCUM, "--seed", SEED,
                "--gradient-checkpointing", *EXTRA_4BIT_FLAG,
            ),
        )

    # ------------------------------------------------------------ Stage 4 --
    pref_data_root = "value_alignment/data/paper_preferences"
    pref_ckpt_root = f"value_alignment/checkpoints/paper_preference/{MODEL}"

    for value in VALUES:
        for method in ["dpo", "hypo"]:
            run_step(
                f"20_train_{method}_{value}",
                py(
                    "value_alignment.train_with_official_hypo",
                    "--method", method, "--model", MODEL,
                    "--train-file", f"{pref_data_root}/{value}/down/train.jsonl",
                    "--eval-file", f"{pref_data_root}/{value}/down/eval.jsonl",
                    "--output-dir", f"{pref_ckpt_root}/{value}",
                    "--epochs", PREF_EPOCHS, "--lora-r", LORA_R, "--lora-alpha", LORA_ALPHA,
                    "--batch-size", BATCH_SIZE, "--grad-accum", GRAD_ACCUM, "--seed", SEED,
                    "--gradient-checkpointing", *EXTRA_4BIT_FLAG,
                ),
            )
        for method in ["simpo", "kto"]:
            run_step(
                f"20_train_{method}_{value}",
                py(
                    "value_alignment.train_extra_methods",
                    "--method", method, "--model", MODEL,
                    "--train-file", f"{pref_data_root}/{value}/down/train.jsonl",
                    "--eval-file", f"{pref_data_root}/{value}/down/eval.jsonl",
                    "--output-dir", f"{pref_ckpt_root}/{value}",
                    "--epochs", PREF_EPOCHS, "--lora-r", LORA_R, "--lora-alpha", LORA_ALPHA,
                    "--batch-size", BATCH_SIZE, "--grad-accum", GRAD_ACCUM, "--seed", SEED,
                    "--gradient-checkpointing", *EXTRA_4BIT_FLAG,
                ),
            )

    # ------------------------------------------------------------ Stage 5 --
    kvs_eval_file = "value_alignment/data/kvs_survey/test.jsonl"
    kvs_out_dir = f"value_alignment/results/paper/kvs/{MODEL}"
    (REPO_ROOT / kvs_out_dir).mkdir(parents=True, exist_ok=True)

    def eval_kvs_argv(run_name: str, adapter: str | None) -> list[str]:
        argv = py(
            "value_alignment.evaluate_kvs_survey",
            "--model", MODEL, "--eval-file", kvs_eval_file,
            "--output", f"{kvs_out_dir}/{run_name}.json",
            "--output-csv", f"{kvs_out_dir}/{run_name}.csv",
            "--num-runs", EVAL_NUM_RUNS, "--seed", SEED,
        )
        if adapter:
            argv += ["--adapter", adapter]
        return argv

    run_step("30_eval_kvs_base", eval_kvs_argv("base", None))
    run_step("30_eval_kvs_sft_baseline", eval_kvs_argv("sft_baseline", f"{sft_ckpt_root}/baseline/final"))
    for value in VALUES:
        run_step(f"30_eval_kvs_sft_{value}", eval_kvs_argv(f"sft_{value}", f"{sft_ckpt_root}/{value}/final"))
        for method in PREF_METHODS:
            run_step(
                f"30_eval_kvs_{method}_{value}",
                eval_kvs_argv(f"{method}_{value}", f"{pref_ckpt_root}/{value}/{method}/final"),
            )

    # ------------------------------------------------------------ Stage 6 --
    aita_test_file = "value_alignment/data/aita_eval/test.jsonl"

    def eval_aita_argv(method: str, value: str, base_adapter: str | None, conditioned_adapter: str) -> list[str]:
        out_dir = f"value_alignment/results/paper/aita/{MODEL}/{method}"
        (REPO_ROOT / out_dir).mkdir(parents=True, exist_ok=True)
        argv = py(
            "value_alignment.evaluation.evaluate_aita_probability_gain",
            "--base-model", MODEL, "--conditioned-model", MODEL,
            "--conditioned-adapter", conditioned_adapter,
            "--target-value", value, "--test-file", aita_test_file,
            "--output-json", f"{out_dir}/{value}.json",
            "--output-csv", f"{out_dir}/{value}.csv",
        )
        if base_adapter:
            argv += ["--base-adapter", base_adapter]
        return argv

    for value in VALUES:
        run_step(
            f"40_eval_aita_sft_{value}",
            eval_aita_argv("sft", value, f"{sft_ckpt_root}/baseline/final", f"{sft_ckpt_root}/{value}/final"),
        )
        for method in PREF_METHODS:
            run_step(
                f"40_eval_aita_{method}_{value}",
                eval_aita_argv(method, value, None, f"{pref_ckpt_root}/{value}/{method}/final"),
            )

    # ------------------------------------------------------------ Stage 7 --
    run_step(
        "50_summarize_kvs",
        py(
            "value_alignment.evaluation.summarize_kvs_experiments",
            "--models", MODEL, "--methods", "sft", "dpo", "hypo", "simpo", "kto",
            "--target-values", *VALUES, "--allow-missing",
        ),
    )
    run_step(
        "50_summarize_aita",
        py(
            "value_alignment.evaluation.summarize_aita_experiments",
            "--models", MODEL, "--methods", "sft", "dpo", "hypo", "simpo", "kto",
            "--target-values", *VALUES, "--allow-missing",
        ),
    )

    print("=" * 70)
    print(f" Priority run finished: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    lines = STATUS_FILE.read_text(encoding="utf-8").splitlines() if STATUS_FILE.exists() else []
    ok = sum(1 for line in lines if line.startswith("OK"))
    skip = sum(1 for line in lines if line.startswith("SKIP"))
    fail = sum(1 for line in lines if line.startswith("FAIL"))
    print(f" OK: {ok}   SKIPPED: {skip}   FAILED: {fail}")
    if fail:
        print(" Failed stages (see value_alignment/slurm_logs/priority_run/<stage>.log for each):")
        for line in lines:
            if line.startswith("FAIL"):
                print("   -", line.split("\t")[1])
    print(" Results:")
    print("   value_alignment/results/paper/kvs_summary.csv")
    print("   value_alignment/results/paper/aita_summary.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
