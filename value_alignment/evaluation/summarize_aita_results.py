#!/usr/bin/env python3
"""Aggregate target-specific AITA Probability Gain result files."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from value_alignment.evaluation.aita_metrics import summarize
from value_alignment.evaluation.statistics_utils import benjamini_hochberg, bootstrap_mean_interval
from value_alignment.value_taxonomy import canonical_basic_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--expected-direction", choices=["greater", "less"], default="greater")
    return parser.parse_args()


def summarize_result_files(
    paths: list[Path],
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 42,
    expected_direction: str = "greater",
) -> dict:
    rows = []
    all_example_rows = []
    seen = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        target_value = canonical_basic_value(payload["target_value"])
        if target_value in seen:
            raise ValueError(f"Duplicate AITA result for {target_value}")
        seen.add(target_value)
        example_rows = payload.get("rows", [])
        if not example_rows:
            raise ValueError(f"AITA result has no per-example rows: {path}")
        if any(canonical_basic_value(row["value"]) != target_value for row in example_rows):
            raise ValueError(f"AITA result mixes target values: {path}")
        all_example_rows.extend(example_rows)
        summary = summarize(
            example_rows,
            bootstrap_replicates,
            bootstrap_seed,
            expected_direction,
        )["overall"]
        rows.append(
            {
                "target_value": target_value,
                "count": summary["count"],
                "mean_probability_gain": summary["mean_probability_gain"],
                "mean_percentage_points": summary["mean_probability_gain_percentage_points"],
                "expected_direction_effect": (
                    summary["mean_probability_gain"]
                    if expected_direction == "greater"
                    else -summary["mean_probability_gain"]
                ),
                "expected_direction_success_fraction": summary[
                    "expected_direction_success_fraction"
                ],
                "expected_direction_bootstrap_ci_95_low": summary[
                    "expected_direction_bootstrap_ci_95_low"
                ],
                "expected_direction_bootstrap_ci_95_high": summary[
                    "expected_direction_bootstrap_ci_95_high"
                ],
                "one_sided_p_value": summary["one_sided_p_value"],
                "bootstrap_ci_95_low": summary["bootstrap_ci_95_low"],
                "bootstrap_ci_95_high": summary["bootstrap_ci_95_high"],
                "strict_mean_probability_gain": summary["strict_mean_probability_gain"],
                "strict_mean_percentage_points": summary["strict_mean_percentage_points"],
                "source": str(path),
            }
        )

    q_values = benjamini_hochberg([row["one_sided_p_value"] for row in rows])
    for row, q_value in zip(rows, q_values, strict=True):
        row["one_sided_fdr_q_value"] = q_value
        row["fdr_significant_0_05"] = q_value is not None and q_value < 0.05

    total = sum(row["count"] for row in rows)
    if total == 0:
        raise ValueError("No AITA examples were available for aggregation.")
    weighted_gain = sum(row["count"] * row["mean_probability_gain"] for row in rows) / total
    strict_weighted_gain = sum(row["count"] * row["strict_mean_probability_gain"] for row in rows) / total
    combined = summarize(
        all_example_rows,
        bootstrap_replicates,
        bootstrap_seed,
        expected_direction,
    )["overall"]
    target_means = [row["mean_probability_gain"] for row in rows]
    macro_gain = statistics.fmean(target_means)
    macro_ci_low, macro_ci_high = bootstrap_mean_interval(
        target_means,
        bootstrap_replicates,
        f"{bootstrap_seed}:macro-targets",
    )
    return {
        "count": total,
        "weighted_average_probability_gain": weighted_gain,
        "weighted_average_percentage_points": 100 * weighted_gain,
        "expected_direction": expected_direction,
        "weighted_expected_direction_effect": (
            weighted_gain if expected_direction == "greater" else -weighted_gain
        ),
        "weighted_expected_direction_bootstrap_ci_95_low": combined[
            "expected_direction_bootstrap_ci_95_low"
        ],
        "weighted_expected_direction_bootstrap_ci_95_high": combined[
            "expected_direction_bootstrap_ci_95_high"
        ],
        "weighted_bootstrap_ci_95_low": combined["bootstrap_ci_95_low"],
        "weighted_bootstrap_ci_95_high": combined["bootstrap_ci_95_high"],
        "macro_average_probability_gain": macro_gain,
        "macro_average_percentage_points": 100 * macro_gain,
        "macro_target_bootstrap_ci_95_low": macro_ci_low,
        "macro_target_bootstrap_ci_95_high": macro_ci_high,
        "strict_weighted_average_probability_gain": strict_weighted_gain,
        "strict_weighted_average_percentage_points": 100 * strict_weighted_gain,
        "multiple_testing": {
            "method": "Benjamini-Hochberg",
            "family_size": len([value for value in q_values if value is not None]),
            "fdr_threshold": 0.05,
            "significant_values": sum(row["fdr_significant_0_05"] for row in rows),
        },
        "per_value": rows,
    }


def write_summary(result: dict, output_json: Path, output_csv: Path) -> None:
    rows = result["per_value"]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["target_value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    result = summarize_result_files(
        args.inputs,
        args.bootstrap_replicates,
        args.bootstrap_seed,
        args.expected_direction,
    )
    write_summary(result, args.output_json, args.output_csv)
    print(json.dumps({key: value for key, value in result.items() if key != "per_value"}, indent=2))


if __name__ == "__main__":
    main()
