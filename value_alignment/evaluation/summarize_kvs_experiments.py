#!/usr/bin/env python3
"""Aggregate all paper and HyPO KVS comparisons into one table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from value_alignment.evaluation.compare_kvs_results import compare_results, load_rows
from value_alignment.value_taxonomy import selected_basic_values, value_slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("value_alignment/results/paper/kvs"),
    )
    parser.add_argument("--models", nargs="+", default=["qwen3-8b", "falcon3-7b", "llama3.1-8b"])
    parser.add_argument("--methods", nargs="+", choices=["sft", "dpo", "hypo"], default=["sft", "dpo", "hypo"])
    parser.add_argument("--target-values", nargs="+", default=["all"])
    parser.add_argument("--allow-missing", action="store_true")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = selected_basic_values(args.target_values)
    summary_rows = []
    missing = []
    for model in args.models:
        model_dir = args.results_root / model
        for method in args.methods:
            base_path = model_dir / ("sft_baseline.json" if method == "sft" else "base.json")
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
                )
                summary_rows.append(
                    {
                        "model": model,
                        "method": method,
                        "target_value": comparison["target_value"],
                        "target_value_rating_drop": comparison["target_value_rating_drop"],
                        "target_value_rating_drop_sample_std": comparison[
                            "target_value_rating_drop_sample_std"
                        ],
                        "other_values_mean_absolute_fluctuation": comparison[
                            "other_values_mean_absolute_fluctuation"
                        ],
                        "other_values_mean_absolute_fluctuation_sample_std": comparison[
                            "other_values_mean_absolute_fluctuation_sample_std"
                        ],
                        "target_paired_count_mean": comparison["target_paired_count_mean"],
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
    result = {
        "comparisons": len(summary_rows),
        "missing_files": list(dict.fromkeys(missing)),
        "rows": summary_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = list(summary_rows[0]) if summary_rows else ["model", "method", "target_value"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(
        json.dumps(
            {
                "comparisons": len(summary_rows),
                "missing_files": len(result["missing_files"]),
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
