"""Pure metric functions for paper-aligned AITA evaluation."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict


LABELS = ("NTA", "Neutral", "YTA")


def probability_gain(
    base_probabilities: dict[str, float],
    conditioned_probabilities: dict[str, float],
    high_label: str,
    low_label: str,
    unused_stance_weight: float = 0.5,
) -> float:
    """Compute the paper's directional AITA Probability Gain."""
    if high_label == low_label:
        raise ValueError("High- and low-standard labels must be different.")
    remaining = [label for label in LABELS if label not in {high_label, low_label}]
    if len(remaining) != 1:
        raise ValueError(f"AITA labels must belong to {LABELS}.")
    unused_label = remaining[0]
    delta_low = conditioned_probabilities[low_label] - base_probabilities[low_label]
    delta_high = conditioned_probabilities[high_label] - base_probabilities[high_label]
    delta_unused = conditioned_probabilities[unused_label] - base_probabilities[unused_label]
    return delta_low - delta_high + unused_stance_weight * delta_unused


def one_sided_t_test_greater(values: list[float]) -> dict[str, float | None]:
    if len(values) < 2:
        return {"t_statistic": None, "p_value": None}
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values)
    if sample_std == 0:
        if mean > 0:
            return {"t_statistic": math.inf, "p_value": 0.0}
        if mean < 0:
            return {"t_statistic": -math.inf, "p_value": 1.0}
        return {"t_statistic": 0.0, "p_value": 0.5}

    t_statistic = mean / (sample_std / math.sqrt(len(values)))
    try:
        from scipy.stats import t as student_t

        p_value = float(student_t.sf(t_statistic, df=len(values) - 1))
    except ImportError:
        p_value = None
    return {"t_statistic": t_statistic, "p_value": p_value}


def _summarize_group(rows: list[dict]) -> dict:
    gains = [float(row["probability_gain"]) for row in rows]
    strict_gains = [float(row["strict_probability_gain"]) for row in rows]
    test = one_sided_t_test_greater(gains)
    return {
        "count": len(rows),
        "mean_probability_gain": statistics.fmean(gains),
        "mean_probability_gain_percentage_points": 100 * statistics.fmean(gains),
        "sample_std": statistics.stdev(gains) if len(gains) > 1 else 0.0,
        "one_sided_t_statistic": test["t_statistic"],
        "one_sided_p_value": test["p_value"],
        "strict_mean_probability_gain": statistics.fmean(strict_gains),
        "strict_mean_percentage_points": 100 * statistics.fmean(strict_gains),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"overall": {"count": 0}, "per_value": {}}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["value"]].append(row)
    return {
        "overall": _summarize_group(rows),
        "per_value": {
            value: _summarize_group(items)
            for value, items in sorted(grouped.items())
        },
    }
