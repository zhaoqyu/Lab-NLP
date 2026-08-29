"""Pure summaries for HyPO reference-mismatch diagnostics."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict


def wilson_interval(
    successes: int,
    count: int,
    z: float = 1.959963984540054,
) -> tuple[float | None, float | None]:
    if count == 0:
        return None, None
    proportion = successes / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / count + z * z / (4 * count * count)
        )
        / denominator
    )
    return center - radius, center + radius


def summarize_group(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    margins = [float(row["reference_margin"]) for row in rows]
    normalized_margins = [float(row["mean_token_reference_margin"]) for row in rows]
    mismatch_count = sum(bool(row["reference_mismatch"]) for row in rows)
    ci_low, ci_high = wilson_interval(mismatch_count, len(rows))
    return {
        "count": len(rows),
        "mismatch_count": mismatch_count,
        "reference_mismatch_rate": mismatch_count / len(rows),
        "reference_mismatch_rate_ci_95_low": ci_low,
        "reference_mismatch_rate_ci_95_high": ci_high,
        "mean_reference_margin": statistics.fmean(margins),
        "median_reference_margin": statistics.median(margins),
        "reference_margin_sample_std": statistics.stdev(margins) if len(margins) > 1 else 0.0,
        "mean_token_reference_margin": statistics.fmean(normalized_margins),
        "mean_hypo_removed_pessimistic_bonus": statistics.fmean(
            float(row["hypo_removed_pessimistic_bonus"]) for row in rows
        ),
    }


def summarize_margins(rows: list[dict]) -> dict:
    def split_summary(split_rows: list[dict]) -> dict:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in split_rows:
            grouped[str(row["value"])].append(row)
        return {
            "overall": summarize_group(split_rows),
            "target": summarize_group([row for row in split_rows if row["is_target"]]),
            "anchors": summarize_group([row for row in split_rows if not row["is_target"]]),
            "per_value": {
                value: summarize_group(value_rows)
                for value, value_rows in sorted(grouped.items())
            },
        }

    grouped_by_split: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped_by_split[str(row.get("source_split") or "unknown")].append(row)
    return {
        **split_summary(rows),
        "by_split": {
            split: split_summary(split_rows)
            for split, split_rows in sorted(grouped_by_split.items())
        },
    }
