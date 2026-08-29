"""Pure metric functions for paper-aligned AITA evaluation."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from value_alignment.evaluation.statistics_utils import bootstrap_mean_interval


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


def one_sided_t_test(
    values: list[float],
    alternative: str = "greater",
) -> dict[str, float | None]:
    if alternative not in {"greater", "less"}:
        raise ValueError("One-sided alternative must be 'greater' or 'less'.")
    if len(values) < 2:
        return {"t_statistic": None, "p_value": None}
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values)
    if sample_std == 0:
        if mean > 0:
            return {
                "t_statistic": math.inf,
                "p_value": 0.0 if alternative == "greater" else 1.0,
            }
        if mean < 0:
            return {
                "t_statistic": -math.inf,
                "p_value": 1.0 if alternative == "greater" else 0.0,
            }
        return {"t_statistic": 0.0, "p_value": 0.5}

    t_statistic = mean / (sample_std / math.sqrt(len(values)))
    try:
        from scipy.stats import t as student_t

        if alternative == "greater":
            p_value = float(student_t.sf(t_statistic, df=len(values) - 1))
        else:
            p_value = float(student_t.cdf(t_statistic, df=len(values) - 1))
    except ImportError:
        p_value = None
    return {"t_statistic": t_statistic, "p_value": p_value}


def one_sided_t_test_greater(values: list[float]) -> dict[str, float | None]:
    return one_sided_t_test(values, "greater")


def _summarize_group(
    rows: list[dict],
    bootstrap_replicates: int,
    bootstrap_seed: int | str,
    expected_direction: str,
) -> dict:
    gains = [float(row["probability_gain"]) for row in rows]
    strict_gains = [float(row["strict_probability_gain"]) for row in rows]
    test = one_sided_t_test(gains, expected_direction)
    ci_low, ci_high = bootstrap_mean_interval(
        gains,
        bootstrap_replicates,
        bootstrap_seed,
    )
    strict_ci_low, strict_ci_high = bootstrap_mean_interval(
        strict_gains,
        bootstrap_replicates,
        f"{bootstrap_seed}:strict",
    )
    expected_effects = [
        gain if expected_direction == "greater" else -gain for gain in gains
    ]
    expected_ci_low, expected_ci_high = bootstrap_mean_interval(
        expected_effects,
        bootstrap_replicates,
        f"{bootstrap_seed}:expected-direction",
    )
    return {
        "count": len(rows),
        "mean_probability_gain": statistics.fmean(gains),
        "mean_probability_gain_percentage_points": 100 * statistics.fmean(gains),
        "sample_std": statistics.stdev(gains) if len(gains) > 1 else 0.0,
        "standard_error": (
            statistics.stdev(gains) / math.sqrt(len(gains)) if len(gains) > 1 else 0.0
        ),
        "bootstrap_ci_95_low": ci_low,
        "bootstrap_ci_95_high": ci_high,
        "positive_gain_fraction": sum(gain > 0 for gain in gains) / len(gains),
        "expected_direction_mean_effect": statistics.fmean(expected_effects),
        "expected_direction_success_fraction": (
            sum(effect > 0 for effect in expected_effects) / len(expected_effects)
        ),
        "expected_direction_bootstrap_ci_95_low": expected_ci_low,
        "expected_direction_bootstrap_ci_95_high": expected_ci_high,
        "one_sided_t_statistic": test["t_statistic"],
        "one_sided_p_value": test["p_value"],
        "one_sided_alternative": expected_direction,
        "strict_mean_probability_gain": statistics.fmean(strict_gains),
        "strict_mean_percentage_points": 100 * statistics.fmean(strict_gains),
        "strict_bootstrap_ci_95_low": strict_ci_low,
        "strict_bootstrap_ci_95_high": strict_ci_high,
    }


def summarize(
    rows: list[dict],
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 42,
    expected_direction: str = "greater",
) -> dict:
    if not rows:
        return {"overall": {"count": 0}, "per_value": {}}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["value"]].append(row)
    return {
        "overall": _summarize_group(
            rows,
            bootstrap_replicates,
            bootstrap_seed,
            expected_direction,
        ),
        "per_value": {
            value: _summarize_group(
                items,
                bootstrap_replicates,
                f"{bootstrap_seed}:{value}",
                expected_direction,
            )
            for value, items in sorted(grouped.items())
        },
    }
