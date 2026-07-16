#!/usr/bin/env python3
"""Compare KVS intrinsic value scores across model runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="Base-model KVS score JSON.")
    parser.add_argument("--trained", type=Path, required=True, help="Trained-model KVS score JSON.")
    parser.add_argument("--target-values", nargs="+", default=[])
    parser.add_argument("--output-json", type=Path, default=Path("value_alignment/results/kvs_comparison.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("value_alignment/results/kvs_comparison.csv"))
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("summary", data)


def metric(summary: dict, value: str, field: str) -> float | None:
    value_summary = summary.get(value, {})
    result = value_summary.get(field)
    return None if result is None else float(result)


def main() -> None:
    args = parse_args()
    base = load_summary(args.base)
    trained = load_summary(args.trained)
    values = sorted(set(base) | set(trained))
    rows = []

    for value in values:
        base_mean = metric(base, value, "mean_rating")
        trained_mean = metric(trained, value, "mean_rating")
        delta = None if base_mean is None or trained_mean is None else trained_mean - base_mean
        rows.append(
            {
                "value": value,
                "base_mean": base_mean,
                "trained_mean": trained_mean,
                "delta": delta,
                "base_std": metric(base, value, "std_rating"),
                "trained_std": metric(trained, value, "std_rating"),
                "base_count": base.get(value, {}).get("count", 0),
                "trained_count": trained.get(value, {}).get("count", 0),
                "is_target": value in args.target_values,
            }
        )

    other_deltas = [row["delta"] for row in rows if row["delta"] is not None and not row["is_target"]]
    target_deltas = [row["delta"] for row in rows if row["delta"] is not None and row["is_target"]]
    comparison = {
        "target_values": args.target_values,
        "target_mean_delta": sum(target_deltas) / len(target_deltas) if target_deltas else None,
        "other_values_mean_delta": sum(other_deltas) / len(other_deltas) if other_deltas else None,
        "other_values_variance": (
            sum((delta - (sum(other_deltas) / len(other_deltas))) ** 2 for delta in other_deltas) / len(other_deltas)
            if other_deltas
            else None
        ),
        "per_value": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["value"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in comparison.items() if k != "per_value"}, indent=2))


if __name__ == "__main__":
    main()
