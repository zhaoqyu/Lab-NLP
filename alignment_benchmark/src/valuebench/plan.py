"""Create a resumable experiment matrix for repeated Colab sessions."""

from __future__ import annotations

from .config import ProjectConfig, config_fingerprint
from .io import write_json
from .taxonomy import value_slug


def build_experiment_plan(config: ProjectConfig) -> dict:
    import pandas as pd

    rows = []
    for method in config.experiment.trainable_methods:
        targets = (
            ["control", *config.experiment.values]
            if config.experiment.train_method_controls
            else config.experiment.values
        )
        for target in targets:
            for seed in config.training.seeds:
                target_slug = target if target == "control" else value_slug(target)
                run_dir = config.paths.output_root / "checkpoints" / method / target_slug / f"seed-{seed}"
                rows.append(
                    {
                        "kind": "train",
                        "method": method,
                        "target": target,
                        "seed": seed,
                        "site": "",
                        "run_dir": str(run_dir),
                        "done_file": str(run_dir / "DONE"),
                        "evaluation_done_file": (
                            ""
                            if target == "control"
                            else str(
                                config.paths.output_root
                                / "results"
                                / "raw"
                                / method
                                / target_slug
                                / f"seed-{seed}"
                                / "DONE"
                            )
                        ),
                    }
                )
    for site in config.evaluation.steering_sites:
        for target in config.experiment.values:
            run_dir = config.paths.output_root / "steering" / site / value_slug(target)
            rows.append(
                {
                    "kind": "steering",
                    "method": f"steering_{site}",
                    "target": target,
                    "seed": -1,
                    "site": site,
                    "run_dir": str(run_dir),
                    "done_file": str(run_dir / "DONE"),
                    "evaluation_done_file": str(
                        config.paths.output_root
                        / "results"
                        / "raw"
                        / f"steering_{site}"
                        / value_slug(target)
                        / "seed--1"
                        / "DONE"
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    output = config.paths.output_root / "experiment_plan.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = {
        "rows": len(frame),
        "training_runs": int((frame["kind"] == "train").sum()),
        "steering_runs": int((frame["kind"] == "steering").sum()),
        "methods": sorted(frame["method"].unique().tolist()),
        "values": config.experiment.values,
        "seeds": config.training.seeds,
        "config_sha256": config_fingerprint(config),
        "output": str(output),
    }
    write_json(config.paths.output_root / "experiment_plan_summary.json", summary)
    return summary
