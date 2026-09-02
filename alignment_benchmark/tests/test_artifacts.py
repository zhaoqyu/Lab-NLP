from __future__ import annotations

import pandas as pd

from valuebench.artifacts import build_paper_artifacts


def test_paper_artifact_smoke(config):
    root = config.paths.output_root / "results" / "aggregate"
    root.mkdir(parents=True, exist_ok=True)
    methods = ("dpo", "hypo")
    targets = ("Security", "Power", "Universalism")
    method_rows = []
    for method_index, method in enumerate(methods):
        center = 0.02 + method_index * 0.01
        row = {"method": method, "targets": 3, "seeds": 3}
        for metric, value in (
            ("target_rating_drop", 0.4 + method_index * 0.1),
            ("non_target_drift", 0.08 + method_index * 0.01),
            ("aita_probability_gain", center),
            ("aita_strict_probability_gain", center * 0.8),
        ):
            row[metric] = value
            row[f"{metric}_ci_low"] = value - 0.005
            row[f"{metric}_ci_high"] = value + 0.005
        method_rows.append(row)
    pd.DataFrame(method_rows).to_csv(root / "method_summary.csv", index=False)

    target_rows = []
    run_rows = []
    for method_index, method in enumerate(methods):
        for target_index, target in enumerate(targets):
            gain = 0.01 * (method_index + target_index + 1)
            target_rows.append(
                {
                    "method": method,
                    "target": target,
                    "target_rating_drop": 0.25 + 0.05 * target_index,
                    "non_target_drift": 0.04 + 0.01 * target_index,
                    "aita_probability_gain": gain,
                    "aita_n": 100,
                    "aita_min_fine_n": 50,
                }
            )
            for seed in (13, 42, 97):
                run_rows.append(
                    {
                        "method": method,
                        "target": target,
                        "seed": seed,
                        "aita_probability_gain": gain + (seed % 10) / 1000,
                    }
                )
    pd.DataFrame(target_rows).to_csv(root / "per_target.csv", index=False)
    pd.DataFrame(run_rows).to_csv(root / "per_run.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": method,
                "mean_train_hours": 0.5 + index,
                "mean_peak_gpu_memory_gb": 8 + index,
                "trainable_parameters": 10_000,
            }
            for index, method in enumerate(methods)
        ]
    ).to_csv(root / "efficiency.csv", index=False)
    pd.DataFrame().to_csv(root / "reference_mismatch.csv", index=False)

    manifest = build_paper_artifacts(config)
    created = {entry["path"] for entry in manifest["files"]}
    assert str(config.paths.output_root / "paper" / "table_main_method_comparison.tex") in created
    assert str(config.paths.output_root / "paper" / "figure_aita_method_comparison.pdf") in created
    assert len(created) >= 14
