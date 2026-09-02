from __future__ import annotations

import numpy as np
import pandas as pd

from valuebench.evaluation import _gain
from valuebench.statistics import fdr_bh, hierarchical_stratified_mean_ci


def test_probability_gain_rewards_low_standard_shift():
    labels = ["NTA", "NEUTRAL", "YTA"]
    control = np.array([0.7, 0.2, 0.1])
    intervention = np.array([0.2, 0.2, 0.6])
    assert _gain(control, intervention, labels, "NTA", "YTA", 0.5) == 1.0
    assert _gain(control, intervention, labels, "NTA", "YTA", 0.0) == 1.0


def test_stratified_bootstrap_equal_weights_sparse_strata():
    rows = []
    for seed in (13, 42, 97):
        rows.append({"seed": seed, "fine": "small", "source_id": f"s-{seed}", "score": 1.0})
        rows.extend(
            {"seed": seed, "fine": "large", "source_id": f"l-{seed}-{index}", "score": 0.0}
            for index in range(20)
        )
    result = hierarchical_stratified_mean_ci(
        pd.DataFrame(rows),
        "score",
        stratum_column="fine",
        samples=500,
        confidence=0.95,
        seed_key="unit-test",
    )
    assert result["mean"] == 0.5
    assert result["ci_low"] == 0.5
    assert result["ci_high"] == 0.5


def test_fdr_preserves_order_and_bounds():
    adjusted = fdr_bh([0.001, 0.02, 0.5])
    assert adjusted == sorted(adjusted)
    assert all(0 <= value <= 1 for value in adjusted)
