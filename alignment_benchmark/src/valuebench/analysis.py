"""Aggregate raw evaluations, quantify uncertainty, and audit matrix completeness."""

from __future__ import annotations

from pathlib import Path

from .config import ProjectConfig
from .io import read_json, write_json
from .statistics import clustered_mean_ci, fdr_bh, hierarchical_stratified_mean_ci


def _raw_files(config: ProjectConfig, name: str) -> list[Path]:
    return sorted((config.paths.output_root / "results" / "raw").glob(f"**/{name}.parquet"))


def _read_frames(paths: list[Path]):
    import pandas as pd

    if not paths:
        raise RuntimeError("No raw evaluation files found. Complete evaluation runs first.")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def validate_completeness(config: ProjectConfig, *, require_all: bool = True) -> dict:
    import pandas as pd

    plan_path = config.paths.output_root / "experiment_plan.csv"
    if not plan_path.exists():
        raise FileNotFoundError("experiment_plan.csv is missing; run `valuebench make-plan`")
    plan = pd.read_csv(plan_path)
    plan["training_done"] = plan["done_file"].map(lambda path: Path(path).exists())
    plan["evaluation_required"] = ~((plan["kind"] == "train") & (plan["target"] == "control"))

    def evaluation_done(row) -> bool:
        if row["kind"] == "train" and row["target"] == "control":
            return True
        seed = int(row["seed"])
        result = (
            config.paths.output_root
            / "results"
            / "raw"
            / row["method"]
            / str(row["target"]).lower()
            / f"seed-{seed}"
            / "DONE"
        )
        return result.exists()

    plan["evaluation_done"] = plan.apply(evaluation_done, axis=1)
    required = plan["evaluation_required"]
    report_path = config.paths.output_root / "results" / "completeness.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(report_path, index=False)
    summary = {
        "planned": len(plan),
        "training_complete": int(plan["training_done"].sum()),
        "evaluations_planned": int(required.sum()),
        "evaluation_complete": int((required & plan["evaluation_done"]).sum()),
        "all_training_complete": bool(plan["training_done"].all()),
        "all_evaluation_complete": bool(plan.loc[required, "evaluation_done"].all()),
        "report": str(report_path),
    }
    write_json(config.paths.output_root / "results" / "completeness.json", summary)
    if require_all and not (summary["all_training_complete"] and summary["all_evaluation_complete"]):
        raise RuntimeError("Experiment matrix is incomplete; inspect results/completeness.csv")
    return summary


def aggregate_results(config: ProjectConfig, *, require_complete: bool = True) -> dict:
    import pandas as pd

    validate_completeness(config, require_all=require_complete)
    kvs = _read_frames(_raw_files(config, "kvs"))
    aita = _read_frames(_raw_files(config, "aita"))
    plan = pd.read_csv(config.paths.output_root / "experiment_plan.csv")
    expected = plan.loc[
        ~((plan["kind"] == "train") & (plan["target"] == "control")),
        ["method", "target", "seed"],
    ].drop_duplicates()
    kvs = kvs.merge(expected, on=["method", "target", "seed"], how="inner", validate="many_to_one")
    aita = aita.merge(expected, on=["method", "target", "seed"], how="inner", validate="many_to_one")
    output = config.paths.output_root / "results" / "aggregate"
    output.mkdir(parents=True, exist_ok=True)

    run_rows = []
    keys = ["method", "target", "seed"]
    for (method, target, seed), group in kvs.groupby(keys):
        target_rows = group[group["is_target"]]
        non_target_rows = group[~group["is_target"]]
        aita_group = aita[(aita["method"] == method) & (aita["target"] == target) & (aita["seed"] == seed)]
        run_rows.append(
            {
                "method": method,
                "target": target,
                "seed": seed,
                "target_rating_drop": target_rows.groupby("fine_value")["rating_drop"].mean().mean(),
                "non_target_drift": non_target_rows.groupby("value")["absolute_drift"].mean().mean(),
                "aita_probability_gain": aita_group.groupby("fine_value")["probability_gain"].mean().mean(),
                "aita_strict_probability_gain": (
                    aita_group.groupby("fine_value")["strict_probability_gain"].mean().mean()
                ),
                "micro_target_rating_drop": target_rows["rating_drop"].mean(),
                "micro_non_target_drift": non_target_rows["absolute_drift"].mean(),
                "micro_aita_probability_gain": aita_group["probability_gain"].mean(),
                "kvs_target_n": len(target_rows),
                "kvs_non_target_n": len(non_target_rows),
                "aita_n": len(aita_group),
            }
        )
    per_run = pd.DataFrame(run_rows)
    per_run.to_csv(output / "per_run.csv", index=False)

    target_rows = []
    for (method, target), group in kvs.groupby(["method", "target"]):
        target_frame = group[group["is_target"]]
        non_target_frame = group[~group["is_target"]]
        aita_frame = aita[(aita["method"] == method) & (aita["target"] == target)]
        drop = hierarchical_stratified_mean_ci(
            target_frame,
            "rating_drop",
            stratum_column="fine_value",
            samples=config.evaluation.bootstrap_samples,
            confidence=config.evaluation.confidence_level,
            seed_key=f"drop:{method}:{target}",
        )
        drift = hierarchical_stratified_mean_ci(
            non_target_frame,
            "absolute_drift",
            stratum_column="value",
            samples=config.evaluation.bootstrap_samples,
            confidence=config.evaluation.confidence_level,
            seed_key=f"drift:{method}:{target}",
        )
        gain = hierarchical_stratified_mean_ci(
            aita_frame,
            "probability_gain",
            stratum_column="fine_value",
            samples=config.evaluation.bootstrap_samples,
            confidence=config.evaluation.confidence_level,
            seed_key=f"gain:{method}:{target}",
        )
        strict_gain = hierarchical_stratified_mean_ci(
            aita_frame,
            "strict_probability_gain",
            stratum_column="fine_value",
            samples=config.evaluation.bootstrap_samples,
            confidence=config.evaluation.confidence_level,
            seed_key=f"strict:{method}:{target}",
        )
        per_seed_fine_counts = aita_frame.groupby(["seed", "fine_value"]).size()
        target_rows.append(
            {
                "method": method,
                "target": target,
                "target_rating_drop": drop["mean"],
                "target_drop_ci_low": drop["ci_low"],
                "target_drop_ci_high": drop["ci_high"],
                "target_drop_p_positive": drop["p_positive"],
                "non_target_drift": drift["mean"],
                "drift_ci_low": drift["ci_low"],
                "drift_ci_high": drift["ci_high"],
                "aita_probability_gain": gain["mean"],
                "aita_gain_ci_low": gain["ci_low"],
                "aita_gain_ci_high": gain["ci_high"],
                "aita_gain_p_positive": gain["p_positive"],
                "aita_strict_probability_gain": strict_gain["mean"],
                "micro_target_rating_drop": target_frame["rating_drop"].mean(),
                "micro_non_target_drift": non_target_frame["absolute_drift"].mean(),
                "micro_aita_probability_gain": aita_frame["probability_gain"].mean(),
                "aita_n": len(aita_frame) // max(1, aita_frame["seed"].nunique()),
                "aita_fine_values": aita_frame["fine_value"].nunique(),
                "aita_min_fine_n": int(per_seed_fine_counts.min()),
            }
        )
    per_target = pd.DataFrame(target_rows)
    per_target["aita_gain_q_bh"] = fdr_bh(per_target["aita_gain_p_positive"].tolist())
    per_target.to_csv(output / "per_target.csv", index=False)

    method_rows = []
    for method, group in per_run.groupby("method"):
        row = {"method": method, "targets": group["target"].nunique(), "seeds": group["seed"].nunique()}
        for metric in (
            "target_rating_drop",
            "non_target_drift",
            "aita_probability_gain",
            "aita_strict_probability_gain",
        ):
            stats = clustered_mean_ci(
                group,
                metric,
                cluster_column="target",
                samples=config.evaluation.bootstrap_samples,
                confidence=config.evaluation.confidence_level,
                seed_key=f"method:{method}:{metric}",
            )
            row[metric] = stats["mean"]
            row[f"{metric}_ci_low"] = stats["ci_low"]
            row[f"{metric}_ci_high"] = stats["ci_high"]
        method_rows.append(row)
    method_summary = pd.DataFrame(method_rows).sort_values("aita_probability_gain", ascending=False)
    method_summary.to_csv(output / "method_summary.csv", index=False)

    efficiency = _aggregate_efficiency(config)
    efficiency.to_csv(output / "efficiency.csv", index=False)
    mismatch = _aggregate_mismatch(config, per_target)
    mismatch.to_csv(output / "reference_mismatch.csv", index=False)
    summary = {
        "kvs_rows": len(kvs),
        "aita_rows": len(aita),
        "runs": len(per_run),
        "methods": sorted(per_run["method"].unique().tolist()),
        "targets": sorted(per_run["target"].unique().tolist()),
        "output": str(output),
    }
    write_json(output / "summary.json", summary)
    return summary


def _aggregate_efficiency(config: ProjectConfig):
    import numpy as np
    import pandas as pd

    rows = []
    for path in sorted((config.paths.output_root / "checkpoints").glob("**/manifest.json")):
        manifest = read_json(path)
        if manifest.get("status") != "completed":
            continue
        rows.append(
            {
                "method": manifest["method"],
                "target": manifest["target"],
                "seed": manifest["seed"],
                "elapsed_seconds": manifest["elapsed_seconds"],
                "peak_gpu_memory_gb": manifest.get("peak_gpu_memory_bytes", 0) / 2**30,
                "trainable_parameters": manifest.get("parameters", {}).get("trainable", np.nan),
            }
        )
    for path in sorted((config.paths.output_root / "steering").glob("*/*/selected.json")):
        selected = read_json(path)
        rows.append(
            {
                "method": f"steering_{selected['site']}",
                "target": selected["target"],
                "seed": -1,
                "elapsed_seconds": selected.get("elapsed_seconds", np.nan),
                "peak_gpu_memory_gb": selected.get("peak_gpu_memory_bytes", 0) / 2**30,
                "trainable_parameters": 0,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=["method", "mean_train_hours", "mean_peak_gpu_memory_gb", "trainable_parameters"]
        )
    target_runs = frame[frame["target"] != "control"]
    return target_runs.groupby("method", as_index=False).agg(
        mean_train_hours=("elapsed_seconds", lambda values: values.mean() / 3600),
        mean_peak_gpu_memory_gb=("peak_gpu_memory_gb", "mean"),
        trainable_parameters=("trainable_parameters", "median"),
    )


def _aggregate_mismatch(config: ProjectConfig, per_target):
    import pandas as pd

    path = config.paths.output_root / "baselines" / "reference_mismatch.parquet"
    if not path.exists():
        return pd.DataFrame()
    mismatch = pd.read_parquet(path)
    rates = (
        mismatch[(mismatch["split"] == "train") & mismatch["is_target"]]
        .groupby("target", as_index=False)
        .agg(
            reference_mismatch_rate=("mismatch", "mean"),
            mean_reference_margin=("reference_margin", "mean"),
        )
    )
    effects = per_target.pivot(index="target", columns="method", values="aita_probability_gain")
    if {"hypo", "dpo"}.issubset(effects.columns):
        effects["hypo_minus_dpo_gain"] = effects["hypo"] - effects["dpo"]
    return rates.merge(effects.reset_index(), on="target", how="left")
