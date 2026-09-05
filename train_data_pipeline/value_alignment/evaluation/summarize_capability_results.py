#!/usr/bin/env python3
"""Summarize capability retention for all target-specific checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from value_alignment.experiment_utils import tagged_run_dir
from value_alignment.value_taxonomy import selected_basic_values, value_slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("value_alignment/results/paper/capabilities"),
    )
    parser.add_argument("--models", nargs="+", default=["qwen3-8b", "falcon3-7b", "llama3.1-8b"])
    parser.add_argument("--methods", nargs="+", choices=["sft", "dpo", "hypo"], default=["sft", "dpo", "hypo"])
    parser.add_argument("--target-values", nargs="+", default=["all"])
    parser.add_argument("--run-tags", nargs="+", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("value_alignment/results/paper/capability_summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("value_alignment/results/paper/capability_summary.csv"),
    )
    parser.add_argument(
        "--output-aggregate-csv",
        type=Path,
        default=Path("value_alignment/results/paper/capability_method_averages.csv"),
    )
    return parser.parse_args()


def load_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["metrics"]


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    targets = selected_basic_values(args.target_values)
    run_tags = args.run_tags or [""]
    rows = []
    missing = []
    for run_tag in run_tags:
        for model in args.models:
            model_dir = tagged_run_dir(args.results_root / model, run_tag)
            base_path = model_dir / "base" / "capability_summary.json"
            if not base_path.exists():
                missing.append(str(base_path))
                continue
            base = load_metrics(base_path)
            for method in args.methods:
                for target in targets:
                    path = model_dir / f"{method}_{value_slug(target)}" / "capability_summary.json"
                    if not path.exists():
                        missing.append(str(path))
                        continue
                    metrics = load_metrics(path)
                    row = {
                        "model": model,
                        "method": method,
                        "target_value": target,
                        "run_tag": run_tag or "legacy",
                        **metrics,
                        "base_mmlu_accuracy": base.get("mmlu_accuracy"),
                        "base_gsm8k_flexible_extract_accuracy": base.get(
                            "gsm8k_flexible_extract_accuracy"
                        ),
                        "source": str(path),
                    }
                    for metric in ("mmlu_accuracy", "gsm8k_flexible_extract_accuracy"):
                        if metrics.get(metric) is not None and base.get(metric) is not None:
                            row[f"{metric}_delta_from_base"] = metrics[metric] - base[metric]
                        else:
                            row[f"{metric}_delta_from_base"] = None
                    rows.append(row)

    if missing and not args.allow_missing:
        raise FileNotFoundError(
            f"Missing {len(set(missing))} capability result files; first: {missing[0]}"
        )

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["method"])].append(row)
    aggregates = []
    for (model, method), method_rows in sorted(grouped.items()):
        aggregate = {
            "model": model,
            "method": method,
            "checkpoint_count": len(method_rows),
            "training_run_count": len({row["run_tag"] for row in method_rows}),
        }
        for metric in ("mmlu_accuracy", "gsm8k_flexible_extract_accuracy"):
            values = [float(row[metric]) for row in method_rows if row.get(metric) is not None]
            deltas = [
                float(row[f"{metric}_delta_from_base"])
                for row in method_rows
                if row.get(f"{metric}_delta_from_base") is not None
            ]
            aggregate[f"{metric}_mean"] = statistics.fmean(values) if values else None
            aggregate[f"{metric}_sample_std"] = sample_std(values)
            aggregate[f"{metric}_delta_from_base_mean"] = (
                statistics.fmean(deltas) if deltas else None
            )
        aggregates.append(aggregate)

    result = {
        "rows": rows,
        "method_averages": aggregates,
        "missing_files": sorted(set(missing)),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    row_fields = list(rows[0]) if rows else ["model", "method", "target_value"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregate_fields = list(aggregates[0]) if aggregates else ["model", "method"]
    with args.output_aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregates)
    print(json.dumps({"checkpoints": len(rows), "method_averages": len(aggregates)}, indent=2))


if __name__ == "__main__":
    main()
