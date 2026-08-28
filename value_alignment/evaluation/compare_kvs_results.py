#!/usr/bin/env python3
"""Compute paper-aligned intrinsic KVS metrics from paired model runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from value_alignment.value_taxonomy import canonical_basic_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="Base-model KVS result JSON.")
    parser.add_argument(
        "--conditioned",
        "--trained",
        dest="conditioned",
        type=Path,
        required=True,
        help="Conditioned-model KVS result JSON.",
    )
    parser.add_argument("--target-value", required=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("value_alignment/results/kvs_comparison.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("value_alignment/results/kvs_comparison.csv"),
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        raise ValueError(f"Expected a KVS result object with a rows list: {path}")
    return data["rows"]


def _index_rows(rows: list[dict], label: str) -> dict[str, dict]:
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"The {label} KVS result contains duplicate row IDs.")
    return indexed


def _rating_runs(row: dict) -> list[int | float | None]:
    ratings = row.get("ratings")
    if isinstance(ratings, list) and ratings:
        return ratings
    return [row.get("rating")]


def _paired_indexes(
    base_rows: list[dict],
    conditioned_rows: list[dict],
) -> tuple[dict[str, dict], dict[str, dict], int]:
    base_by_id = _index_rows(base_rows, "base")
    conditioned_by_id = _index_rows(conditioned_rows, "conditioned")
    missing_conditioned = sorted(set(base_by_id) - set(conditioned_by_id))
    missing_base = sorted(set(conditioned_by_id) - set(base_by_id))
    if missing_conditioned or missing_base:
        raise ValueError(
            "KVS result row IDs do not match: "
            f"missing conditioned={len(missing_conditioned)}, missing base={len(missing_base)}"
        )

    run_counts = {
        len(_rating_runs(row))
        for row in [*base_by_id.values(), *conditioned_by_id.values()]
    }
    if len(run_counts) != 1:
        raise ValueError(f"KVS rows do not share one run count: {sorted(run_counts)}")
    return base_by_id, conditioned_by_id, run_counts.pop()


def pair_rows_for_run(
    base_by_id: dict[str, dict],
    conditioned_by_id: dict[str, dict],
    run_index: int,
) -> list[dict]:
    pairs = []
    for row_id, base_row in base_by_id.items():
        conditioned_row = conditioned_by_id[row_id]
        if base_row["value"] != conditioned_row["value"]:
            raise ValueError(f"Value mismatch for row {row_id}.")
        base_rating = _rating_runs(base_row)[run_index]
        conditioned_rating = _rating_runs(conditioned_row)[run_index]
        if base_rating is None or conditioned_rating is None:
            continue
        if not 1 <= float(base_rating) <= 6 or not 1 <= float(conditioned_rating) <= 6:
            raise ValueError(f"KVS ratings must be in 1-6 for row {row_id}.")
        delta = float(conditioned_rating) - float(base_rating)
        pairs.append(
            {
                "id": row_id,
                "source_id": base_row.get("source_id"),
                "value": base_row["value"],
                "base_rating": float(base_rating),
                "conditioned_rating": float(conditioned_rating),
                "delta": delta,
                "rating_drop": -delta,
                "absolute_change": abs(delta),
            }
        )
    return pairs


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def compare_results(
    base_rows: list[dict],
    conditioned_rows: list[dict],
    target_value: str,
) -> dict:
    target = canonical_basic_value(target_value)
    base_by_id, conditioned_by_id, run_count = _paired_indexes(base_rows, conditioned_rows)
    run_results = []
    per_value_runs: dict[str, list[dict]] = defaultdict(list)

    for run_index in range(run_count):
        pairs = pair_rows_for_run(base_by_id, conditioned_by_id, run_index)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for pair in pairs:
            grouped[pair["value"]].append(pair)

        for value, value_pairs in grouped.items():
            deltas = [pair["delta"] for pair in value_pairs]
            per_value_runs[value].append(
                {
                    "run": run_index,
                    "paired_count": len(value_pairs),
                    "base_mean": statistics.fmean(pair["base_rating"] for pair in value_pairs),
                    "conditioned_mean": statistics.fmean(
                        pair["conditioned_rating"] for pair in value_pairs
                    ),
                    "mean_delta": statistics.fmean(deltas),
                    "rating_drop": -statistics.fmean(deltas),
                    "mean_absolute_change": statistics.fmean(
                        pair["absolute_change"] for pair in value_pairs
                    ),
                }
            )

        target_drops = [pair["rating_drop"] for pair in pairs if pair["value"] == target]
        other_fluctuations = [
            pair["absolute_change"] for pair in pairs if pair["value"] != target
        ]
        if not target_drops:
            raise ValueError(
                f"No valid paired KVS ratings found for target {target!r} in run {run_index}."
            )
        run_results.append(
            {
                "run": run_index,
                "paired_count": len(pairs),
                "target_paired_count": len(target_drops),
                "other_values_paired_count": len(other_fluctuations),
                "target_value_rating_drop": statistics.fmean(target_drops),
                "other_values_mean_absolute_fluctuation": statistics.fmean(other_fluctuations),
            }
        )

    per_value = []
    for value, value_runs in sorted(per_value_runs.items()):
        per_value.append(
            {
                "value": value,
                "is_target": value == target,
                "paired_count_mean": statistics.fmean(row["paired_count"] for row in value_runs),
                "paired_count_sample_std": sample_std(
                    [float(row["paired_count"]) for row in value_runs]
                ),
                "base_mean": statistics.fmean(row["base_mean"] for row in value_runs),
                "conditioned_mean": statistics.fmean(
                    row["conditioned_mean"] for row in value_runs
                ),
                "mean_delta": statistics.fmean(row["mean_delta"] for row in value_runs),
                "rating_drop": statistics.fmean(row["rating_drop"] for row in value_runs),
                "rating_drop_sample_std": sample_std(
                    [row["rating_drop"] for row in value_runs]
                ),
                "mean_absolute_change": statistics.fmean(
                    row["mean_absolute_change"] for row in value_runs
                ),
                "mean_absolute_change_sample_std": sample_std(
                    [row["mean_absolute_change"] for row in value_runs]
                ),
                "runs": value_runs,
            }
        )

    target_drops = [row["target_value_rating_drop"] for row in run_results]
    other_fluctuations = [
        row["other_values_mean_absolute_fluctuation"] for row in run_results
    ]
    target_counts = [float(row["target_paired_count"]) for row in run_results]
    other_counts = [float(row["other_values_paired_count"]) for row in run_results]
    return {
        "target_value": target,
        "num_runs": run_count,
        "paired_count_mean": statistics.fmean(row["paired_count"] for row in run_results),
        "target_paired_count_mean": statistics.fmean(target_counts),
        "target_paired_count_sample_std": sample_std(target_counts),
        "other_values_paired_count_mean": statistics.fmean(other_counts),
        "other_values_paired_count_sample_std": sample_std(other_counts),
        "target_value_rating_drop": statistics.fmean(target_drops),
        "target_value_rating_drop_sample_std": sample_std(target_drops),
        "other_values_mean_absolute_fluctuation": statistics.fmean(other_fluctuations),
        "other_values_mean_absolute_fluctuation_sample_std": sample_std(other_fluctuations),
        "runs": run_results,
        "per_value": per_value,
    }


def main() -> None:
    args = parse_args()
    comparison = compare_results(
        load_rows(args.base),
        load_rows(args.conditioned),
        args.target_value,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = comparison["per_value"]
    csv_rows = [{key: value for key, value in row.items() if key != "runs"} for row in rows]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]) if csv_rows else ["value"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(
        json.dumps(
            {
                key: value
                for key, value in comparison.items()
                if key not in {"runs", "per_value"}
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
