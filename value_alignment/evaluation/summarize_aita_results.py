#!/usr/bin/env python3
"""Aggregate target-specific AITA Probability Gain result files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from value_alignment.value_taxonomy import canonical_basic_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    seen = set()
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        target_value = canonical_basic_value(payload["target_value"])
        if target_value in seen:
            raise ValueError(f"Duplicate AITA result for {target_value}")
        seen.add(target_value)
        summary = payload["summary"]["overall"]
        rows.append(
            {
                "target_value": target_value,
                "count": summary["count"],
                "mean_probability_gain": summary["mean_probability_gain"],
                "mean_percentage_points": summary["mean_probability_gain_percentage_points"],
                "one_sided_p_value": summary["one_sided_p_value"],
                "strict_mean_probability_gain": summary["strict_mean_probability_gain"],
                "strict_mean_percentage_points": summary["strict_mean_percentage_points"],
                "source": str(path),
            }
        )

    total = sum(row["count"] for row in rows)
    weighted_gain = sum(row["count"] * row["mean_probability_gain"] for row in rows) / total
    strict_weighted_gain = sum(row["count"] * row["strict_mean_probability_gain"] for row in rows) / total
    result = {
        "count": total,
        "weighted_average_probability_gain": weighted_gain,
        "weighted_average_percentage_points": 100 * weighted_gain,
        "strict_weighted_average_probability_gain": strict_weighted_gain,
        "strict_weighted_average_percentage_points": 100 * strict_weighted_gain,
        "per_value": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["target_value"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in result.items() if key != "per_value"}, indent=2))


if __name__ == "__main__":
    main()
