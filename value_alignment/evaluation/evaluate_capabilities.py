#!/usr/bin/env python3
"""Run MMLU and GSM8K through the official lm-evaluation-harness."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from value_alignment.experiment_utils import write_run_manifest
from value_alignment.model_utils import resolve_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=["mmlu", "gsm8k"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-aliases", type=Path, default=Path("value_alignment/configs/model_aliases.json"))
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        default=None,
        help="Smoke tests only; accepts a count or fraction and must not be used for final numbers.",
    )
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_lm_eval_command(args: argparse.Namespace, resolved_model: str) -> list[str]:
    model_args = [f"pretrained={resolved_model}", f"dtype={args.dtype}"]
    if args.adapter is not None:
        model_args.append(f"peft={args.adapter}")
    if args.load_in_4bit:
        model_args.append("load_in_4bit=True")
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        ",".join(model_args),
        "--tasks",
        ",".join(args.tasks),
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--output_path",
        str(args.output_dir),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.apply_chat_template:
        command.append("--apply_chat_template")
    if args.log_samples:
        command.append("--log_samples")
    return command


def _find_metric(metrics: dict, preferred: list[str], contains: list[str]) -> float | None:
    for key in preferred:
        if isinstance(metrics.get(key), (int, float)):
            return float(metrics[key])
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and all(token in key for token in contains):
            return float(value)
    return None


def extract_capability_metrics(payload: dict) -> dict[str, float | None]:
    results = payload.get("results", {})
    groups = payload.get("groups", {})
    mmlu_metrics = groups.get("mmlu") or results.get("mmlu") or {}
    mmlu_accuracy = _find_metric(
        mmlu_metrics,
        ["acc,none", "acc"],
        ["acc"],
    )
    if mmlu_accuracy is None:
        subject_scores = []
        for task, metrics in results.items():
            if task.startswith("mmlu_"):
                score = _find_metric(metrics, ["acc,none", "acc"], ["acc"])
                if score is not None:
                    subject_scores.append(score)
        if subject_scores:
            mmlu_accuracy = sum(subject_scores) / len(subject_scores)

    gsm8k_metrics = results.get("gsm8k", {})
    gsm8k_flexible = _find_metric(
        gsm8k_metrics,
        ["exact_match,flexible-extract", "exact_match_flexible-extract"],
        ["exact_match", "flexible"],
    )
    gsm8k_strict = _find_metric(
        gsm8k_metrics,
        ["exact_match,strict-match", "exact_match_strict-match"],
        ["exact_match", "strict"],
    )
    return {
        "mmlu_accuracy": mmlu_accuracy,
        "gsm8k_flexible_extract_accuracy": gsm8k_flexible,
        "gsm8k_strict_match_accuracy": gsm8k_strict,
    }


def find_harness_result(output_dir: Path) -> Path:
    candidates = [
        path
        for path in output_dir.rglob("*.json")
        if path.name.startswith("results") and path.name != "capability_summary.json"
    ]
    if not candidates:
        raise FileNotFoundError(f"lm-evaluation-harness wrote no results JSON below {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main() -> None:
    args = parse_args()
    resolved_model = resolve_model_name(args.model, args.model_aliases)
    command = build_lm_eval_command(args, resolved_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "evaluator": "EleutherAI/lm-evaluation-harness",
        "resolved_model": resolved_model,
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "command": command,
        "task_defaults": {
            "mmlu": "harness task default (accuracy)",
            "gsm8k": "harness task default (5-shot; flexible-extract reported)",
        },
    }
    input_files = {"model_aliases": args.model_aliases}
    if args.adapter is not None:
        input_files["adapter_config"] = args.adapter / "adapter_config.json"
    write_run_manifest(args.output_dir, args, input_files, metadata=metadata)

    if args.dry_run:
        write_run_manifest(
            args.output_dir,
            args,
            input_files,
            metadata={**metadata, "dry_run": True},
            status="completed",
        )
        print(json.dumps({"command": command}, indent=2))
        return
    if importlib.util.find_spec("lm_eval") is None:
        raise SystemExit(
            "lm-evaluation-harness is not installed. Install value_alignment/requirements.txt first."
        )
    subprocess.run(command, check=True)

    harness_result = find_harness_result(args.output_dir)
    payload = json.loads(harness_result.read_text(encoding="utf-8"))
    summary = {
        "model": resolved_model,
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "tasks": args.tasks,
        "metrics": extract_capability_metrics(payload),
        "harness_result": str(harness_result),
    }
    (args.output_dir / "capability_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_run_manifest(
        args.output_dir,
        args,
        input_files,
        metadata={**metadata, **summary},
        status="completed",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
