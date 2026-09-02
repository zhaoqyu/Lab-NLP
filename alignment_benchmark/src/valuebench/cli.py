"""Typer command-line interface for Colab and local orchestration."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
from pathlib import Path

import typer
from rich.console import Console

from .config import ProjectConfig, load_config

DEFAULT_CONFIG = Path("alignment_benchmark/configs/paper.yaml")
PAPER_STACK = {
    "transformers": "4.45.2",
    "trl": "0.9.6",
    "peft": "0.13.2",
    "accelerate": "1.6.0",
    "bitsandbytes": "0.45.5",
    "datasets": "3.2.0",
}
app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
console = Console()


def _config(path: Path) -> ProjectConfig:
    return load_config(path)


def _show(value) -> None:
    console.print_json(json.dumps(value, default=str))


@app.command()
def doctor(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    strict: bool = typer.Option(False, help="Fail unless the registered package stack and CUDA are ready."),
) -> None:
    """Check data paths, runtime packages, GPU visibility, and secret presence."""
    cfg = _config(config)
    packages = (
        "torch",
        "transformers",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
        "datasets",
        "openai",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    cuda = {"available": False}
    try:
        import torch

        cuda["available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            cuda.update(
                {
                    "device": torch.cuda.get_device_name(0),
                    "bf16": torch.cuda.is_bf16_supported(),
                    "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                }
            )
    except ImportError:
        pass
    stack_ok = all(versions.get(package) == expected for package, expected in PAPER_STACK.items())
    result = {
        "python": platform.python_version(),
        "packages": versions,
        "paper_stack": PAPER_STACK,
        "paper_stack_ok": stack_ok,
        "cuda": cuda,
        "kvs": str(cfg.paths.kvs),
        "aita": str(cfg.paths.aita),
        "output_root": str(cfg.paths.output_root),
        "openrouter_key_present": bool(os.environ.get("OPENROUTER_API_KEY")),
    }
    _show(result)
    if strict and not (stack_ok and cuda["available"]):
        raise typer.Exit(code=1)


@app.command("teacher")
def teacher_command(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    splits: str = typer.Option("train,eval,test", help="Comma-separated KVS splits."),
    limit: int = typer.Option(0, min=0, help="Generate only this many selected records; 0 means all."),
    dry_run: bool = typer.Option(False, help="Write prompts without calling OpenRouter."),
    overwrite: bool = typer.Option(False, help="Regenerate only the selected records."),
) -> None:
    """Generate or resume canonical KVS pairs through OpenRouter."""
    from .teacher import generate_canonical

    selected = tuple(split.strip() for split in splits.split(",") if split.strip())
    unknown = set(selected) - {"train", "eval", "test"}
    if unknown:
        raise typer.BadParameter(f"Unknown split(s): {sorted(unknown)}")
    _show(
        generate_canonical(
            _config(config),
            splits=selected,
            limit=limit,
            dry_run=dry_run,
            overwrite=overwrite,
        )
    )


@app.command("audit-data")
def audit_data(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    embeddings: bool = typer.Option(True, "--embeddings/--no-embeddings"),
    fail_on_quality: bool = typer.Option(False),
) -> None:
    """Audit generated pairs for balance, leakage, duplicates, and fidelity."""
    from .quality import audit_canonical

    _show(audit_canonical(_config(config), embeddings=embeddings, fail_on_quality=fail_on_quality))


@app.command("prepare-aita")
def prepare_aita_command(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
) -> None:
    """Validate and map AITA categories into the fixed ten-value taxonomy."""
    from .data import prepare_aita

    _show(prepare_aita(_config(config)))


@app.command("collect-baselines")
def collect_baselines_command(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
) -> None:
    """Score the frozen base model on KVS ratings and reference preferences."""
    from .baselines import collect_baselines

    _show(collect_baselines(_config(config)))


@app.command("summarize-mismatch")
def summarize_mismatch_command(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
) -> None:
    """Measure DPO reference mismatch for every target intervention."""
    from .baselines import summarize_reference_mismatch

    _show(summarize_reference_mismatch(_config(config)))


@app.command("build-views")
def build_views_command(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
) -> None:
    """Derive equal-cardinality SFT, preference, and steering views from KVS."""
    from .data import build_views

    result = build_views(_config(config))
    _show({"views": len(result["views"]), "expected_counts": result["expected_counts"]})


@app.command("validate-data")
def validate_data_command(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
) -> None:
    """Verify identical source IDs and row counts across method views."""
    from .data import validate_fairness

    result = validate_fairness(_config(config))
    _show({"all_pass": result["all_pass"], "checks": len(result["checks"])})


@app.command("make-plan")
def make_plan_command(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
) -> None:
    """Register the complete resumable experiment matrix."""
    from .plan import build_experiment_plan

    _show(build_experiment_plan(_config(config)))


@app.command("train")
def train_command(
    method: str = typer.Option(...),
    target: str = typer.Option(...),
    seed: int = typer.Option(...),
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    force: bool = typer.Option(False),
) -> None:
    """Train one registered QLoRA adapter and resume interrupted checkpoints."""
    from .training import train_run

    _show(train_run(_config(config), method=method, target=target, seed=seed, force=force))


@app.command("build-steering")
def build_steering_command(
    target: str = typer.Option(...),
    site: str = typer.Option(..., help="block or attn"),
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    force: bool = typer.Option(False),
) -> None:
    """Build one CAA vector and select layer/strength on KVS eval only."""
    from .steering import build_steering

    _show(build_steering(_config(config), target=target, site=site, force=force))


@app.command("evaluate")
def evaluate_command(
    method: str = typer.Option(...),
    target: str = typer.Option(...),
    seed: int = typer.Option(...),
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    force: bool = typer.Option(False),
) -> None:
    """Evaluate one intervention against its matched control on locked test data."""
    from .evaluation import evaluate_run

    _show(evaluate_run(_config(config), method=method, target=target, seed=seed, force=force))


def _plan_frame(cfg: ProjectConfig):
    import pandas as pd

    path = cfg.paths.output_root / "experiment_plan.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `valuebench make-plan`")
    frame = pd.read_csv(path, keep_default_na=False)
    frame["construction_done"] = frame["done_file"].map(lambda path: Path(path).exists())
    frame["evaluation_required"] = frame["evaluation_done_file"].astype(bool)
    frame["evaluation_done"] = frame["evaluation_done_file"].map(lambda path: not path or Path(path).exists())
    return frame


@app.command("status")
def status_command(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
) -> None:
    """Show construction and evaluation progress for the registered matrix."""
    frame = _plan_frame(_config(config))
    required = frame["evaluation_required"]
    _show(
        {
            "registered_rows": len(frame),
            "construction_complete": int(frame["construction_done"].sum()),
            "evaluations_registered": int(required.sum()),
            "evaluation_complete": int((required & frame["evaluation_done"]).sum()),
            "next_construction": frame.loc[~frame["construction_done"]].head(1).to_dict("records"),
            "next_evaluation": frame.loc[frame["construction_done"] & ~frame["evaluation_done"]]
            .head(1)
            .to_dict("records"),
        }
    )


@app.command("run-next")
def run_next_command(
    phase: str = typer.Option("auto", help="auto, construct, or evaluate"),
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
) -> None:
    """Run exactly one pending construction or evaluation job for Colab sessions."""
    if phase not in {"auto", "construct", "evaluate"}:
        raise typer.BadParameter("phase must be auto, construct, or evaluate")
    cfg = _config(config)
    frame = _plan_frame(cfg)
    construction = frame.loc[~frame["construction_done"]]
    evaluation = frame.loc[frame["construction_done"] & ~frame["evaluation_done"]]
    if phase in {"auto", "evaluate"} and not evaluation.empty:
        from .evaluation import evaluate_run

        row = evaluation.iloc[0]
        result = evaluate_run(
            cfg,
            method=row["method"],
            target=row["target"],
            seed=int(row["seed"]),
        )
        _show({"phase": "evaluate", "run": row.to_dict(), "result": result})
        return
    if phase in {"auto", "construct"} and not construction.empty:
        row = construction.iloc[0]
        if row["kind"] == "train":
            from .training import train_run

            result = train_run(cfg, method=row["method"], target=row["target"], seed=int(row["seed"]))
        else:
            from .steering import build_steering

            result = build_steering(cfg, target=row["target"], site=row["site"])
        _show({"phase": "construct", "run": row.to_dict(), "result": result})
        return
    _show({"status": "no pending run for selected phase"})


@app.command("aggregate-results")
def aggregate_results_command(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    allow_incomplete: bool = typer.Option(False),
) -> None:
    """Aggregate raw evaluations with cluster-aware bootstrap intervals."""
    from .analysis import aggregate_results

    _show(aggregate_results(_config(config), require_complete=not allow_incomplete))


@app.command("paper-artifacts")
def paper_artifacts_command(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
) -> None:
    """Generate publication tables, figures, and checksum manifests."""
    from .artifacts import build_paper_artifacts

    _show(build_paper_artifacts(_config(config)))
