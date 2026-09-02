"""Generate camera-ready tables and figures from registered aggregate results."""

from __future__ import annotations

from pathlib import Path

from .config import ProjectConfig, config_fingerprint
from .io import read_json, sha256_file, utc_now, write_json

METHOD_ORDER = (
    "sft",
    "dpo",
    "hypo",
    "ipo",
    "simpo",
    "orpo",
    "steering_block",
    "steering_attn",
)

METHOD_LABELS = {
    "sft": "SFT",
    "dpo": "DPO",
    "hypo": "HyPO",
    "ipo": "IPO",
    "simpo": "SimPO",
    "orpo": "ORPO",
    "steering_block": "CAA (block)",
    "steering_attn": "CAA (attention)",
}


def _read_aggregates(config: ProjectConfig) -> dict:
    import pandas as pd

    root = config.paths.output_root / "results" / "aggregate"
    required = ("method_summary", "per_target", "per_run", "efficiency")
    frames = {}
    for name in required:
        path = root / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run `valuebench aggregate-results` first")
        frames[name] = pd.read_csv(path)
    mismatch = root / "reference_mismatch.csv"
    try:
        frames["reference_mismatch"] = pd.read_csv(mismatch) if mismatch.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        frames["reference_mismatch"] = pd.DataFrame()
    return frames


def _ordered(frame):
    import pandas as pd

    result = frame.copy()
    result["method"] = pd.Categorical(result["method"], categories=METHOD_ORDER, ordered=True)
    return result.sort_values("method")


def _style():
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.05)
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(figure, root: Path, name: str) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = root / f"{name}.{suffix}"
        figure.savefig(path, bbox_inches="tight")
        paths.append(path)
    return paths


def _ci_text(row, metric: str) -> str:
    return f"{row[metric]:.3f} [{row[f'{metric}_ci_low']:.3f}, {row[f'{metric}_ci_high']:.3f}]"


def _write_tables(config: ProjectConfig, frames: dict, root: Path) -> list[Path]:
    method_summary = _ordered(frames["method_summary"])
    per_target = _ordered(frames["per_target"])
    efficiency = _ordered(frames["efficiency"])
    table_paths = []

    main = method_summary[["method", "targets", "seeds"]].copy()
    main["method"] = main["method"].astype(str).map(METHOD_LABELS)
    for metric, label in (
        ("target_rating_drop", "KVS target rating drop"),
        ("non_target_drift", "KVS non-target drift"),
        ("aita_probability_gain", "AITA probability gain"),
        ("aita_strict_probability_gain", "AITA strict gain"),
    ):
        main[label] = method_summary.apply(lambda row: _ci_text(row, metric), axis=1)
    main_path = root / "table_main_method_comparison.csv"
    main.to_csv(main_path, index=False)
    table_paths.append(main_path)

    target_table = per_target.copy()
    target_table["method"] = target_table["method"].astype(str).map(METHOD_LABELS)
    target_table["aita_power_flag"] = target_table["aita_min_fine_n"].map(
        lambda count: "main" if count >= config.evaluation.min_aita_main_table_n else "underpowered"
    )
    target_path = root / "table_per_value_results.csv"
    target_table.to_csv(target_path, index=False)
    table_paths.append(target_path)

    if not efficiency.empty:
        efficiency["method"] = efficiency["method"].astype(str).map(METHOD_LABELS)
        efficiency_path = root / "table_efficiency.csv"
        efficiency.to_csv(efficiency_path, index=False)
        table_paths.append(efficiency_path)

    latex = main.rename(
        columns={
            "method": "Method",
            "targets": "Targets",
            "seeds": "Seeds",
        }
    )
    latex_path = root / "table_main_method_comparison.tex"
    latex.to_latex(latex_path, index=False, escape=True)
    table_paths.append(latex_path)
    return table_paths


def _dataset_artifacts(config: ProjectConfig, root: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from .taxonomy import to_basic
    from .teacher import raw_kvs_rows

    records = [
        {
            "dataset": "KVS",
            "split": row["split"],
            "value": row["value"],
            "fine_value": row["fine_value"],
            "source_id": row["source_id"],
        }
        for row in raw_kvs_rows(config)
    ]
    raw_aita = read_json(config.paths.aita)
    seen = set()
    for fine_value, examples in raw_aita.items():
        for example in examples:
            key = (fine_value, example["post"].strip())
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "dataset": "AITA",
                    "split": "test",
                    "value": to_basic(fine_value),
                    "fine_value": fine_value,
                    "source_id": "",
                }
            )
    frame = pd.DataFrame(records)
    composition = (
        frame.groupby(["dataset", "split", "value", "fine_value"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )
    table_path = root / "table_dataset_composition.csv"
    composition.to_csv(table_path, index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    kvs = (
        composition[composition["dataset"] == "KVS"].groupby(["split", "value"], as_index=False)["rows"].sum()
    )
    aita = composition[composition["dataset"] == "AITA"].groupby("value", as_index=False)["rows"].sum()
    sns.barplot(data=kvs, x="value", y="rows", hue="split", ax=axes[0])
    sns.barplot(data=aita, x="value", y="rows", color=sns.color_palette("colorblind")[2], ax=axes[1])
    axes[0].set_title("KVS composition by registered split")
    axes[1].set_title("Unique AITA evaluation scenarios")
    for axis in axes:
        axis.set_xlabel("")
        axis.set_ylabel("Rows")
        axis.tick_params(axis="x", rotation=40)
    figure_paths = _save_figure(figure, root, "figure_dataset_composition")

    created = [table_path, *figure_paths]
    quality_path = config.paths.output_root / "data" / "quality" / "teacher_audit.csv"
    if quality_path.exists():
        quality = pd.read_csv(quality_path)
        quality_table = quality.groupby("value", as_index=False).agg(
            rows=("source_id", "size"),
            quality_pass_rate=("quality_pass", "mean"),
            mean_affirming_similarity=("affirming_similarity", "mean"),
            mean_opposing_similarity=("opposing_similarity", "mean"),
        )
        quality_output = root / "table_teacher_quality.csv"
        quality_table.to_csv(quality_output, index=False)
        created.append(quality_output)
    return created


def _training_curve_artifacts(config: ProjectConfig, root: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    records = []
    for path in sorted((config.paths.output_root / "checkpoints").glob("*/*/seed-*/trainer_state.json")):
        state = read_json(path)
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        for entry in state.get("log_history", []):
            for source_key, metric in (("loss", "train_loss"), ("eval_loss", "eval_loss")):
                if source_key in entry:
                    records.append(
                        {
                            "method": manifest["method"],
                            "target": manifest["target"],
                            "seed": manifest["seed"],
                            "epoch": entry.get("epoch"),
                            "step": entry.get("step"),
                            "metric": metric,
                            "value": entry[source_key],
                        }
                    )
    if not records:
        return []
    frame = pd.DataFrame(records).dropna(subset=["epoch"])
    frame["Method"] = frame["method"].map(METHOD_LABELS)
    table_path = root / "table_training_curves.csv"
    frame.to_csv(table_path, index=False)
    metrics = [metric for metric in ("train_loss", "eval_loss") if metric in set(frame["metric"])]
    figure, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 4.6), squeeze=False)
    for axis, metric in zip(axes[0], metrics, strict=True):
        subset = frame[frame["metric"] == metric]
        sns.lineplot(
            data=subset,
            x="epoch",
            y="value",
            hue="Method",
            estimator="mean",
            errorbar="sd",
            ax=axis,
        )
        axis.set_title(metric.replace("_", " ").title())
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
    return [table_path, *_save_figure(figure, root, "figure_training_curves")]


def _main_gain_figure(frames: dict, root: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    frame = _ordered(frames["method_summary"])
    x = np.arange(len(frame))
    values = frame["aita_probability_gain"].to_numpy() * 100
    lower = values - frame["aita_probability_gain_ci_low"].to_numpy() * 100
    upper = frame["aita_probability_gain_ci_high"].to_numpy() * 100 - values
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    colors = sns.color_palette("colorblind", len(frame))
    axis.bar(x, values, color=colors, width=0.72)
    axis.errorbar(x, values, yerr=np.vstack([lower, upper]), fmt="none", color="black", capsize=3)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, [METHOD_LABELS[str(method)] for method in frame["method"]], rotation=25, ha="right")
    axis.set_ylabel("AITA probability gain (percentage points)")
    axis.set_title("Behavioral value shift on held-out AITA scenarios")
    return _save_figure(figure, root, "figure_aita_method_comparison")


def _selectivity_figure(frames: dict, root: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    frame = frames["per_target"].copy()
    frame["Method"] = frame["method"].map(METHOD_LABELS)
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    sns.scatterplot(
        data=frame,
        x="non_target_drift",
        y="target_rating_drop",
        hue="Method",
        style="Method",
        s=68,
        alpha=0.82,
        ax=axis,
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Mean absolute drift on non-target KVS values")
    axis.set_ylabel("Target KVS rating drop")
    axis.set_title("Intervention strength versus collateral value drift")
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    return _save_figure(figure, root, "figure_kvs_selectivity")


def _heatmap_figure(frames: dict, root: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    frame = _ordered(frames["per_target"])
    matrix = frame.pivot(index="method", columns="target", values="aita_probability_gain") * 100
    matrix.index = [METHOD_LABELS[str(method)] for method in matrix.index]
    figure, axis = plt.subplots(figsize=(11.0, 4.8))
    sns.heatmap(matrix, center=0, cmap="vlag", annot=True, fmt=".1f", linewidths=0.35, ax=axis)
    axis.set_xlabel("Target value")
    axis.set_ylabel("")
    axis.set_title("AITA probability gain by method and target value")
    axis.tick_params(axis="x", rotation=35)
    return _save_figure(figure, root, "figure_aita_value_heatmap")


def _transfer_figure(frames: dict, root: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    frame = frames["per_target"].copy()
    frame["Method"] = frame["method"].map(METHOD_LABELS)
    frame["aita_gain_pp"] = frame["aita_probability_gain"] * 100
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    sns.regplot(
        data=frame,
        x="target_rating_drop",
        y="aita_gain_pp",
        scatter=False,
        color="0.35",
        ci=95,
        ax=axis,
    )
    sns.scatterplot(
        data=frame,
        x="target_rating_drop",
        y="aita_gain_pp",
        hue="Method",
        style="Method",
        s=68,
        alpha=0.82,
        ax=axis,
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("KVS target rating drop")
    axis.set_ylabel("AITA probability gain (percentage points)")
    axis.set_title("Transfer from value ratings to moral judgments")
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    return _save_figure(figure, root, "figure_kvs_aita_transfer")


def _stability_figure(frames: dict, root: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    frame = _ordered(frames["per_run"])
    frame["Method"] = frame["method"].astype(str).map(METHOD_LABELS)
    frame["aita_gain_pp"] = frame["aita_probability_gain"] * 100
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    sns.boxplot(data=frame, x="Method", y="aita_gain_pp", color="white", fliersize=0, ax=axis)
    sns.stripplot(
        data=frame,
        x="Method",
        y="aita_gain_pp",
        hue="seed",
        dodge=False,
        alpha=0.58,
        size=3.5,
        palette="colorblind",
        ax=axis,
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("")
    axis.set_ylabel("AITA probability gain (percentage points)")
    axis.set_title("Variation across target values and random seeds")
    axis.tick_params(axis="x", rotation=25)
    axis.legend(title="Seed", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    return _save_figure(figure, root, "figure_seed_stability")


def _efficiency_figure(frames: dict, root: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    efficiency = frames["efficiency"]
    if efficiency.empty:
        return []
    performance = frames["method_summary"][["method", "aita_probability_gain"]]
    frame = efficiency.merge(performance, on="method", how="inner")
    if frame.empty:
        return []
    frame["Method"] = frame["method"].map(METHOD_LABELS)
    frame["aita_gain_pp"] = frame["aita_probability_gain"] * 100
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    sns.scatterplot(
        data=frame,
        x="mean_train_hours",
        y="aita_gain_pp",
        hue="Method",
        style="Method",
        size="mean_peak_gpu_memory_gb",
        sizes=(70, 240),
        ax=axis,
    )
    axis.set_xlabel("Mean training / construction time (hours)")
    axis.set_ylabel("AITA probability gain (percentage points)")
    axis.set_title("Alignment effectiveness and computational cost")
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    return _save_figure(figure, root, "figure_performance_efficiency")


def _mismatch_figure(frames: dict, root: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    frame = frames["reference_mismatch"]
    if frame.empty or "hypo_minus_dpo_gain" not in frame:
        return []
    frame = frame.dropna(subset=["reference_mismatch_rate", "hypo_minus_dpo_gain"]).copy()
    if frame.empty:
        return []
    frame["gain_difference_pp"] = frame["hypo_minus_dpo_gain"] * 100
    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    sns.regplot(
        data=frame,
        x="reference_mismatch_rate",
        y="gain_difference_pp",
        scatter_kws={"s": 70},
        ax=axis,
    )
    for row in frame.itertuples():
        axis.annotate(
            row.target,
            (row.reference_mismatch_rate, row.gain_difference_pp),
            xytext=(4, 4),
            textcoords="offset points",
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Reference mismatch rate in KVS training pairs")
    axis.set_ylabel("HyPO minus DPO AITA gain (percentage points)")
    axis.set_title("Does HyPO help where reference mismatch is greater?")
    return _save_figure(figure, root, "figure_hypo_reference_mismatch")


def _steering_grid_figure(config: ProjectConfig, root: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    paths = sorted((config.paths.output_root / "steering").glob("*/*/selection_grid.csv"))
    if not paths:
        return []
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    sites = sorted(frame["site"].unique())
    figure, axes = plt.subplots(1, len(sites), figsize=(6.2 * len(sites), 4.8), squeeze=False)
    for axis, site in zip(axes[0], sites, strict=True):
        subset = frame[frame["site"] == site]
        matrix = subset.pivot_table(
            index="layer", columns="coefficient", values="selection_objective", aggfunc="mean"
        )
        sns.heatmap(matrix, center=0, cmap="vlag", annot=True, fmt=".2f", ax=axis)
        axis.set_title(f"CAA {site}: mean validation objective")
        axis.set_xlabel("Steering coefficient")
        axis.set_ylabel("Layer")
    return _save_figure(figure, root, "figure_steering_selection")


def build_paper_artifacts(config: ProjectConfig) -> dict:
    """Create deterministic tables, plots, and a checksum manifest."""
    import matplotlib.pyplot as plt

    _style()
    frames = _read_aggregates(config)
    root = config.paths.output_root / "paper"
    root.mkdir(parents=True, exist_ok=True)
    created = _write_tables(config, frames, root)
    created.extend(_dataset_artifacts(config, root))
    plt.close("all")
    created.extend(_training_curve_artifacts(config, root))
    plt.close("all")
    builders = (
        _main_gain_figure,
        _selectivity_figure,
        _heatmap_figure,
        _transfer_figure,
        _stability_figure,
        _efficiency_figure,
        _mismatch_figure,
    )
    for builder in builders:
        created.extend(builder(frames, root))
        plt.close("all")
    created.extend(_steering_grid_figure(config, root))
    plt.close("all")

    context = {}
    for name, path in (
        ("teacher_quality", config.paths.output_root / "data" / "quality" / "teacher_audit_summary.json"),
        ("aita", config.paths.output_root / "data" / "aita_summary.json"),
        ("experiment_plan", config.paths.output_root / "experiment_plan_summary.json"),
        ("aggregate", config.paths.output_root / "results" / "aggregate" / "summary.json"),
    ):
        if path.exists():
            context[name] = read_json(path)
    write_json(root / "paper_context.json", context)
    created.append(root / "paper_context.json")

    manifest = {
        "generated_at_utc": utc_now(),
        "base_model": config.model.id,
        "config_sha256": config_fingerprint(config),
        "files": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(set(created))
        ],
    }
    write_json(root / "manifest.json", manifest)
    manifest["manifest"] = str(root / "manifest.json")
    return manifest
