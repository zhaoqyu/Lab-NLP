"""Small deterministic statistical helpers for experiment summaries."""

from __future__ import annotations

import random
import statistics
from typing import Sequence


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    if not 0 <= probability <= 1:
        raise ValueError("Percentile probability must be between 0 and 1.")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_mean_interval(
    values: Sequence[float],
    replicates: int = 2000,
    seed: int | str = 42,
    confidence: float = 0.95,
) -> tuple[float | None, float | None]:
    if not values or replicates <= 0:
        return None, None
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be strictly between 0 and 1.")
    numeric_values = [float(value) for value in values]
    rng = random.Random(seed)
    count = len(numeric_values)
    bootstrap_means = [
        statistics.fmean(rng.choice(numeric_values) for _ in range(count))
        for _ in range(replicates)
    ]
    alpha = (1 - confidence) / 2
    return percentile(bootstrap_means, alpha), percentile(bootstrap_means, 1 - alpha)


def benjamini_hochberg(p_values: Sequence[float | None]) -> list[float | None]:
    """Return monotone Benjamini-Hochberg adjusted p-values."""
    valid = sorted(
        (float(p_value), index)
        for index, p_value in enumerate(p_values)
        if p_value is not None
    )
    adjusted: list[float | None] = [None] * len(p_values)
    if not valid:
        return adjusted

    running_min = 1.0
    test_count = len(valid)
    for rank_index in range(test_count - 1, -1, -1):
        p_value, original_index = valid[rank_index]
        rank = rank_index + 1
        running_min = min(running_min, p_value * test_count / rank)
        adjusted[original_index] = min(1.0, running_min)
    return adjusted
