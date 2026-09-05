#!/usr/bin/env python3
"""Relate HyPO's empirical advantage to frozen-reference mismatch severity."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from value_alignment.value_taxonomy import canonical_basic_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mismatch-summary",
        type=Path,
        default=Path("value_alignment/results/paper/reference_mismatch_summary.json"),
    )
    parser.add_argument(
        "--kvs-summary",
        type=Path,
        default=Path("value_alignment/results/paper/kvs_summary.json"),
    )
    parser.add_argument("--aita-summary", type=Path, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("value_alignment/results/paper/hypo_advantage_analysis.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("value_alignment/results/paper/hypo_advantage_analysis.csv"),
    )
    return parser.parse_args()


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[indexed[position][0]] = average_rank
        start = end
    return ranks


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values):
        raise ValueError("Correlation inputs must have equal lengths.")
    if len(x_values) < 2:
        return None
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
    if x_scale == 0 or y_scale == 0:
        return None
    return numerator / (x_scale * y_scale)


def correlation_summary(rows: list[dict], x_key: str, y_key: str) -> dict:
    paired = [
        (float(row[x_key]), float(row[y_key]))
        for row in rows
        if row.get(x_key) is not None and row.get(y_key) is not None
    ]
    x_values = [pair[0] for pair in paired]
    y_values = [pair[1] for pair in paired]
    pearson = pearson_correlation(x_values, y_values)
    spearman = pearson_correlation(_ranks(x_values), _ranks(y_values)) if paired else None
    result = {
        "count": len(paired),
        "pearson_r": pearson,
        "spearman_rho": spearman,
    }
    if len(paired) >= 3:
        try:
            from scipy.stats import pearsonr, spearmanr

            result["pearson_p_value"] = (
                float(pearsonr(x_values, y_values).pvalue)
                if pearson is not None
                else None
            )
            result["spearman_p_value"] = (
                float(spearmanr(x_values, y_values).pvalue)
                if spearman is not None
                else None
            )
        except ImportError:
            result["pearson_p_value"] = None
            result["spearman_p_value"] = None
    else:
        result["pearson_p_value"] = None
        result["spearman_p_value"] = None
    return result


def _mean_by_method(rows: list[dict], metric: str) -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        key = (
            str(row["model"]),
            canonical_basic_value(row["target_value"]),
            str(row["method"]),
        )
        grouped[key].append(float(value))
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def load_aita_target_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for detail in payload.get("details", []):
        for target in detail["summary"]["per_value"]:
            rows.append(
                {
                    "model": detail["model"],
                    "method": detail["method"],
                    "target_value": target["target_value"],
                    "mean_probability_gain": target["mean_probability_gain"],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    mismatch_payload = json.loads(args.mismatch_summary.read_text(encoding="utf-8"))
    kvs_payload = json.loads(args.kvs_summary.read_text(encoding="utf-8"))
    target_drop = _mean_by_method(kvs_payload["rows"], "target_value_rating_drop")
    other_fluctuation = _mean_by_method(
        kvs_payload["rows"],
        "other_values_mean_absolute_fluctuation",
    )
    aita_gain = {}
    if args.aita_summary is not None:
        aita_gain = _mean_by_method(
            load_aita_target_rows(args.aita_summary),
            "mean_probability_gain",
        )

    rows = []
    for mismatch in mismatch_payload["rows"]:
        model = str(mismatch["model"])
        target = canonical_basic_value(mismatch["target_value"])
        dpo_key = (model, target, "dpo")
        hypo_key = (model, target, "hypo")
        if dpo_key not in target_drop or hypo_key not in target_drop:
            continue
        row = {
            "model": model,
            "target_value": target,
            "train_target_mismatch_rate": mismatch.get("train_target_mismatch_rate"),
            "train_target_mean_reference_margin": mismatch.get(
                "train_target_mean_reference_margin"
            ),
            "train_target_mean_removed_pessimistic_bonus": mismatch.get(
                "train_target_mean_removed_pessimistic_bonus"
            ),
            "kvs_hypo_minus_dpo_target_drop": target_drop[hypo_key] - target_drop[dpo_key],
            "kvs_dpo_minus_hypo_other_fluctuation": (
                other_fluctuation[dpo_key] - other_fluctuation[hypo_key]
            ),
        }
        if dpo_key in aita_gain and hypo_key in aita_gain:
            row["aita_hypo_minus_dpo_probability_gain"] = (
                aita_gain[hypo_key] - aita_gain[dpo_key]
            )
        rows.append(row)

    outcome_metrics = [
        "kvs_hypo_minus_dpo_target_drop",
        "kvs_dpo_minus_hypo_other_fluctuation",
    ]
    if any("aita_hypo_minus_dpo_probability_gain" in row for row in rows):
        outcome_metrics.append("aita_hypo_minus_dpo_probability_gain")
    predictors = [
        "train_target_mismatch_rate",
        "train_target_mean_removed_pessimistic_bonus",
    ]

    correlations = {}
    for predictor in predictors:
        correlations[predictor] = {
            outcome: correlation_summary(rows, predictor, outcome)
            for outcome in outcome_metrics
        }
    per_model = {}
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        per_model[model] = {
            predictor: {
                outcome: correlation_summary(model_rows, predictor, outcome)
                for outcome in outcome_metrics
            }
            for predictor in predictors
        }

    result = {
        "interpretation": {
            "positive_target_drop_advantage": "HyPO produces a larger intended target-value drop than DPO.",
            "positive_preservation_advantage": "HyPO produces less non-target fluctuation than DPO.",
            "positive_aita_advantage": "HyPO produces a larger AITA Probability Gain than DPO.",
            "caution": "Correlation is diagnostic, not causal; model/value cells are not fully independent.",
        },
        "rows": rows,
        "overall_correlations": correlations,
        "per_model_correlations": per_model,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = list(rows[0]) if rows else ["model", "target_value"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(correlations, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
