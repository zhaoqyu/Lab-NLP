#!/usr/bin/env python3
"""Aggregate all paper and HyPO KVS comparisons into one table."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from value_alignment.evaluation.compare_kvs_results import compare_results, load_rows
from value_alignment.experiment_utils import tagged_run_dir
from value_alignment.value_taxonomy import selected_basic_values, value_slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("value_alignment/results/paper/kvs"),
    )
    parser.add_argument("--models", nargs="+", default=["qwen3-8b", "falcon3-7b", "llama3.1-8b"])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["sft", "sft_up", "dpo", "hypo", "simpo", "kto", "steering_attn", "steering_block"],
        default=["sft", "dpo", "hypo"],
    )
    parser.add_argument("--direction", choices=["down", "up"], default="down")
    parser.add_argument("--target-values", nargs="+", default=["all"])
    parser.add_argument(
        "--run-tags",
        nargs="+",
        default=None,
        help="Optional tagged training/evaluation runs, for example seed42 seed43 seed44.",
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("value_alignment/results/paper/kvs_summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("value_alignment/results/paper/kvs_summary.csv"),
    )
    parser.add_argument(
        "--output-aggregate-csv",
        type=Path,
        default=Path("value_alignment/results/paper/kvs_seed_aggregates.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = selected_basic_values(args.target_values)
    run_tags = args.run_tags or [""]
    summary_rows = []
    missing = []
    for run_tag in run_tags:
        for model in args.models:
            model_dir = tagged_run_dir(args.results_root / model, run_tag)
            for method in args.methods:
                base_path = model_dir / (
                    "sft_baseline.json" if method.startswith("sft") else "base.json"
                )
                for target in targets:
                    conditioned_path = model_dir / f"{method}_{value_slug(target)}.json"
                    absent = [path for path in (base_path, conditioned_path) if not path.exists()]
                    if absent:
                        missing.extend(str(path) for path in absent)
                        continue
                    comparison = compare_results(
                        load_rows(base_path),
                        load_rows(conditioned_path),
                        target,
                        args.bootstrap_replicates,
                        args.bootstrap_seed,
                    )
                    raw_ci_low = comparison[
                        "target_value_rating_drop_cluster_bootstrap_ci_95_low"
                    ]
                    raw_ci_high = comparison[
                        "target_value_rating_drop_cluster_bootstrap_ci_95_high"
                    ]
                    if args.direction == "down":
                        expected_ci_low, expected_ci_high = raw_ci_low, raw_ci_high
                    elif raw_ci_low is None or raw_ci_high is None:
                        expected_ci_low, expected_ci_high = None, None
                    else:
                        expected_ci_low, expected_ci_high = -raw_ci_high, -raw_ci_low
                    summary_rows.append(
                        {
                            "model": model,
                            "method": method,
                            "target_value": comparison["target_value"],
                            "run_tag": run_tag or "legacy",
                            "direction": args.direction,
                            "target_value_rating_drop": comparison[
                                "target_value_rating_drop"
                            ],
                            "target_value_expected_direction_shift": (
                                comparison["target_value_rating_drop"]
                                if args.direction == "down"
                                else -comparison["target_value_rating_drop"]
                            ),
                            "target_value_rating_drop_sample_std": comparison[
                                "target_value_rating_drop_sample_std"
                            ],
                            "target_value_rating_drop_ci_95_low": comparison[
                                "target_value_rating_drop_cluster_bootstrap_ci_95_low"
                            ],
                            "target_value_rating_drop_ci_95_high": comparison[
                                "target_value_rating_drop_cluster_bootstrap_ci_95_high"
                            ],
                            "target_value_expected_direction_shift_ci_95_low": expected_ci_low,
                            "target_value_expected_direction_shift_ci_95_high": expected_ci_high,
                            "target_source_cluster_count": comparison[
                                "target_source_cluster_count"
                            ],
                            "other_values_mean_absolute_fluctuation": comparison[
                                "other_values_mean_absolute_fluctuation"
                            ],
                            "other_values_mean_absolute_fluctuation_sample_std": comparison[
                                "other_values_mean_absolute_fluctuation_sample_std"
                            ],
                            "other_values_fluctuation_ci_95_low": comparison[
                                "other_values_fluctuation_cluster_bootstrap_ci_95_low"
                            ],
                            "other_values_fluctuation_ci_95_high": comparison[
                                "other_values_fluctuation_cluster_bootstrap_ci_95_high"
                            ],
                            "other_values_source_cluster_count": comparison[
                                "other_values_source_cluster_count"
                            ],
                            "target_paired_count_mean": comparison[
                                "target_paired_count_mean"
                            ],
                            "target_paired_count_sample_std": comparison[
                                "target_paired_count_sample_std"
                            ],
                            "other_values_paired_count_mean": comparison[
                                "other_values_paired_count_mean"
                            ],
                            "other_values_paired_count_sample_std": comparison[
                                "other_values_paired_count_sample_std"
                            ],
                            "base_result": str(base_path),
                            "conditioned_result": str(conditioned_path),
                        }
                    )

    if missing and not args.allow_missing:
        unique_missing = list(dict.fromkeys(missing))
        raise FileNotFoundError(
            f"Missing {len(unique_missing)} required KVS result files; first: {unique_missing[0]}"
        )
    grouped_rows: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in summary_rows:
        grouped_rows[(row["model"], row["method"], row["target_value"])].append(row)
    seed_aggregates = []
    for (model, method, target), grouped in sorted(grouped_rows.items()):
        target_metrics = [row["target_value_rating_drop"] for row in grouped]
        direction_metrics = [
            row["target_value_expected_direction_shift"] for row in grouped
        ]
        other_metrics = [row["other_values_mean_absolute_fluctuation"] for row in grouped]
        seed_aggregates.append(
            {
                "model": model,
                "method": method,
                "target_value": target,
                "training_run_count": len(grouped),
                "run_tags": ",".join(row["run_tag"] for row in grouped),
                "target_value_rating_drop_mean_across_training_runs": statistics.fmean(target_metrics),
                "target_value_rating_drop_sample_std_across_training_runs": (
                    statistics.stdev(target_metrics) if len(target_metrics) > 1 else 0.0
                ),
                "target_expected_direction_shift_mean_across_training_runs": statistics.fmean(
                    direction_metrics
                ),
                "target_expected_direction_shift_sample_std_across_training_runs": (
                    statistics.stdev(direction_metrics) if len(direction_metrics) > 1 else 0.0
                ),
                "other_values_fluctuation_mean_across_training_runs": statistics.fmean(other_metrics),
                "other_values_fluctuation_sample_std_across_training_runs": (
                    statistics.stdev(other_metrics) if len(other_metrics) > 1 else 0.0
                ),
            }
        )

    result = {
        "comparisons": len(summary_rows),
        "missing_files": list(dict.fromkeys(missing)),
        "rows": summary_rows,
        "seed_aggregates": seed_aggregates,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = list(summary_rows[0]) if summary_rows else ["model", "method", "target_value"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    args.output_aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
    aggregate_fields = (
        list(seed_aggregates[0])
        if seed_aggregates
        else ["model", "method", "target_value"]
    )
    with args.output_aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(seed_aggregates)
    print(
        json.dumps(
            {
                "comparisons": len(summary_rows),
                "missing_files": len(result["missing_files"]),
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
                "output_aggregate_csv": str(args.output_aggregate_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
