#!/usr/bin/env python3
"""Aggregate the complete AITA matrix, including repeated training runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from value_alignment.evaluation.summarize_aita_results import summarize_result_files
from value_alignment.experiment_utils import tagged_run_dir
from value_alignment.value_taxonomy import selected_basic_values, value_slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("value_alignment/results/paper/aita"),
    )
    parser.add_argument("--models", nargs="+", default=["qwen3-8b", "falcon3-7b", "llama3.1-8b"])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["sft", "sft_up", "dpo", "hypo", "simpo", "kto", "steering_attn", "steering_block"],
        default=["sft", "dpo", "hypo"],
    )
    parser.add_argument("--target-values", nargs="+", default=["all"])
    parser.add_argument("--run-tags", nargs="+", default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--expected-direction", choices=["greater", "less"], default="greater")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("value_alignment/results/paper/aita_summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("value_alignment/results/paper/aita_summary.csv"),
    )
    parser.add_argument(
        "--output-aggregate-csv",
        type=Path,
        default=Path("value_alignment/results/paper/aita_seed_aggregates.csv"),
    )
    return parser.parse_args()


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    targets = selected_basic_values(args.target_values)
    run_tags = args.run_tags or [""]
    rows = []
    detailed = []
    missing = []
    for run_tag in run_tags:
        for model in args.models:
            for method in args.methods:
                result_dir = tagged_run_dir(args.results_root / model / method, run_tag)
                paths = [result_dir / f"{value_slug(target)}.json" for target in targets]
                absent = [path for path in paths if not path.exists()]
                if absent:
                    missing.extend(str(path) for path in absent)
                    continue
                summary = summarize_result_files(
                    paths,
                    args.bootstrap_replicates,
                    args.bootstrap_seed,
                    args.expected_direction,
                )
                rows.append(
                    {
                        "model": model,
                        "method": method,
                        "run_tag": run_tag or "legacy",
                        "count": summary["count"],
                        "weighted_average_probability_gain": summary[
                            "weighted_average_probability_gain"
                        ],
                        "weighted_bootstrap_ci_95_low": summary[
                            "weighted_bootstrap_ci_95_low"
                        ],
                        "weighted_bootstrap_ci_95_high": summary[
                            "weighted_bootstrap_ci_95_high"
                        ],
                        "macro_average_probability_gain": summary[
                            "macro_average_probability_gain"
                        ],
                        "strict_weighted_average_probability_gain": summary[
                            "strict_weighted_average_probability_gain"
                        ],
                        "weighted_expected_direction_effect": summary[
                            "weighted_expected_direction_effect"
                        ],
                        "weighted_expected_direction_bootstrap_ci_95_low": summary[
                            "weighted_expected_direction_bootstrap_ci_95_low"
                        ],
                        "weighted_expected_direction_bootstrap_ci_95_high": summary[
                            "weighted_expected_direction_bootstrap_ci_95_high"
                        ],
                        "fdr_significant_values": summary["multiple_testing"][
                            "significant_values"
                        ],
                    }
                )
                detailed.append(
                    {
                        "model": model,
                        "method": method,
                        "run_tag": run_tag or "legacy",
                        "summary": summary,
                    }
                )

    if missing and not args.allow_missing:
        raise FileNotFoundError(
            f"Missing {len(set(missing))} AITA result files; first: {missing[0]}"
        )

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["method"])].append(row)
    aggregates = []
    for (model, method), method_rows in sorted(grouped.items()):
        weighted = [row["weighted_average_probability_gain"] for row in method_rows]
        macro = [row["macro_average_probability_gain"] for row in method_rows]
        strict = [row["strict_weighted_average_probability_gain"] for row in method_rows]
        expected_effect = [
            row["weighted_expected_direction_effect"] for row in method_rows
        ]
        aggregates.append(
            {
                "model": model,
                "method": method,
                "training_run_count": len(method_rows),
                "run_tags": ",".join(row["run_tag"] for row in method_rows),
                "weighted_probability_gain_mean_across_training_runs": statistics.fmean(weighted),
                "weighted_probability_gain_sample_std_across_training_runs": sample_std(weighted),
                "macro_probability_gain_mean_across_training_runs": statistics.fmean(macro),
                "macro_probability_gain_sample_std_across_training_runs": sample_std(macro),
                "strict_probability_gain_mean_across_training_runs": statistics.fmean(strict),
                "strict_probability_gain_sample_std_across_training_runs": sample_std(strict),
                "expected_direction_effect_mean_across_training_runs": statistics.fmean(
                    expected_effect
                ),
                "expected_direction_effect_sample_std_across_training_runs": sample_std(
                    expected_effect
                ),
            }
        )

    result = {
        "rows": rows,
        "training_run_aggregates": aggregates,
        "details": detailed,
        "missing_files": sorted(set(missing)),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    row_fields = list(rows[0]) if rows else ["model", "method", "run_tag"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregate_fields = list(aggregates[0]) if aggregates else ["model", "method"]
    with args.output_aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregates)
    print(json.dumps({"runs": len(rows), "aggregates": len(aggregates)}, indent=2))


if __name__ == "__main__":
    main()
