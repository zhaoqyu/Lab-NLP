from __future__ import annotations

import pandas as pd

from valuebench.plan import build_experiment_plan


def test_registered_matrix_has_expected_size(config):
    summary = build_experiment_plan(config)
    frame = pd.read_csv(summary["output"], keep_default_na=False)
    assert summary["training_runs"] == 198
    assert summary["steering_runs"] == 20
    assert summary["rows"] == 218
    assert (frame["evaluation_done_file"] != "").sum() == 200
