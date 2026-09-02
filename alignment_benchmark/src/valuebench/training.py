"""Unified QLoRA training for SFT and offline preference objectives."""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

from .config import ProjectConfig, config_fingerprint
from .data import validate_fairness
from .io import read_jsonl, sha256_file, utc_now, write_json
from .modeling import (
    format_chat_prompt,
    hardware_info,
    load_base_model,
    load_tokenizer,
    release_model,
    trainable_parameters,
)
from .taxonomy import to_basic, value_slug

OFFICIAL_HYPO_URL = "https://github.com/tmllab/2026_ICLR_HyPO.git"
OFFICIAL_HYPO_COMMIT = "e552477308a3f5ac518a46b7f56fe77f5a3a994f"
TRAINABLE_METHODS = {"sft", "dpo", "hypo", "ipo", "simpo", "orpo"}


def ensure_official_hypo(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path / ".git").exists():
        subprocess.run(["git", "clone", OFFICIAL_HYPO_URL, str(path)], check=True)
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()
    if current != OFFICIAL_HYPO_COMMIT:
        subprocess.run(["git", "fetch", "origin", OFFICIAL_HYPO_COMMIT], cwd=path, check=True)
        subprocess.run(["git", "checkout", "--detach", OFFICIAL_HYPO_COMMIT], cwd=path, check=True)
    return path


def checkpoint_dir(config: ProjectConfig, method: str, target: str, seed: int) -> Path:
    slug = "control" if target == "control" else value_slug(target)
    return config.paths.output_root / "checkpoints" / method / slug / f"seed-{seed}"


def _view_paths(config: ProjectConfig, method: str, target: str) -> tuple[Path, Path]:
    kind = "sft" if method == "sft" else "preference"
    slug = "control" if target == "control" else value_slug(target)
    root = config.paths.output_root / "data" / "views" / kind / slug
    return root / "train.jsonl", root / "eval.jsonl"


def _versions() -> dict:
    packages = ("torch", "transformers", "trl", "peft", "accelerate", "bitsandbytes", "datasets")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _dataset(config: ProjectConfig, method: str, target: str, tokenizer):
    from datasets import Dataset

    train_path, eval_path = _view_paths(config, method, target)
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError("Method views are missing. Run `valuebench build-views` first.")

    def format_rows(path: Path) -> list[dict]:
        rows = []
        for source in read_jsonl(path):
            prompt = format_chat_prompt(
                tokenizer,
                config.model.system_prompt,
                source["prompt"],
                max_user_tokens=config.model.max_prompt_length - 64,
            )
            if method == "sft":
                rows.append({"text": prompt + source["completion"] + tokenizer.eos_token})
            else:
                rows.append(
                    {
                        "prompt": prompt,
                        "chosen": source["chosen"].strip(),
                        "rejected": source["rejected"].strip(),
                    }
                )
        return rows

    return (
        Dataset.from_list(format_rows(train_path)),
        Dataset.from_list(format_rows(eval_path)),
        train_path,
        eval_path,
    )


def _lora(config: ProjectConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=config.training.lora_r,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


def _precision_kwargs() -> dict:
    import torch

    return {
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    }


def _strategy_key(config_class) -> str:
    parameters = inspect.signature(config_class.__init__).parameters
    return "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"


def _common_arguments(config: ProjectConfig, output: Path, seed: int, learning_rate: float) -> dict:
    kwargs = {
        "output_dir": str(output),
        "num_train_epochs": config.training.epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": config.training.train_batch_size,
        "per_device_eval_batch_size": config.training.eval_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "warmup_ratio": config.training.warmup_ratio,
        "weight_decay": config.training.weight_decay,
        "optim": config.training.optimizer,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": config.training.save_total_limit,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": ["tensorboard"],
        "logging_dir": str(output / "tensorboard"),
        "seed": seed,
        "data_seed": seed,
        "remove_unused_columns": False,
        **_precision_kwargs(),
    }
    return kwargs


def _early_stopping(config: ProjectConfig):
    from transformers import EarlyStoppingCallback

    return EarlyStoppingCallback(early_stopping_patience=config.training.early_stopping_patience)


def _build_trainer(
    config: ProjectConfig, method: str, model, tokenizer, train_data, eval_data, output: Path, seed: int
):
    common = _common_arguments(
        config,
        output,
        seed,
        config.training.sft_learning_rate if method == "sft" else config.training.preference_learning_rate,
    )
    callbacks = [_early_stopping(config)]
    peft_config = _lora(config)

    if method == "sft":
        from trl import SFTConfig, SFTTrainer

        common["remove_unused_columns"] = True
        common[_strategy_key(SFTConfig)] = "epoch"
        args = SFTConfig(
            **common,
            dataset_text_field="text",
            packing=False,
            max_seq_length=config.model.max_length,
        )
        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            peft_config=peft_config,
            callbacks=callbacks,
        )

    if method == "hypo":
        official = ensure_official_hypo(config.paths.official_hypo)
        if str(official) not in sys.path:
            sys.path.insert(0, str(official))
        from hypo_config import DPOConfig
        from hypo_trainer import DPOTrainer

        common[_strategy_key(DPOConfig)] = "epoch"
        args = DPOConfig(
            **common,
            beta=config.training.dpo_beta,
            max_length=config.model.max_length,
            max_prompt_length=config.model.max_prompt_length,
            precompute_ref_log_probs=True,
        )
        return DPOTrainer(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            peft_config=peft_config,
            callbacks=callbacks,
        )

    if method in {"dpo", "ipo"}:
        from trl import DPOConfig, DPOTrainer

        common[_strategy_key(DPOConfig)] = "epoch"
        args = DPOConfig(
            **common,
            beta=config.training.ipo_beta if method == "ipo" else config.training.dpo_beta,
            loss_type="ipo" if method == "ipo" else "sigmoid",
            max_length=config.model.max_length,
            max_prompt_length=config.model.max_prompt_length,
            precompute_ref_log_probs=True,
        )
        return DPOTrainer(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            peft_config=peft_config,
            callbacks=callbacks,
        )

    if method == "simpo":
        from trl import CPOConfig, CPOTrainer

        common[_strategy_key(CPOConfig)] = "epoch"
        args = CPOConfig(
            **common,
            beta=config.training.simpo_beta,
            simpo_gamma=config.training.simpo_gamma,
            cpo_alpha=0.0,
            loss_type="simpo",
            max_length=config.model.max_length,
            max_prompt_length=config.model.max_prompt_length,
        )
        return CPOTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            peft_config=peft_config,
            callbacks=callbacks,
        )

    if method == "orpo":
        from trl import ORPOConfig, ORPOTrainer

        common[_strategy_key(ORPOConfig)] = "epoch"
        args = ORPOConfig(
            **common,
            beta=config.training.orpo_beta,
            max_length=config.model.max_length,
            max_prompt_length=config.model.max_prompt_length,
        )
        return ORPOTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            peft_config=peft_config,
            callbacks=callbacks,
        )
    raise ValueError(f"Unsupported method: {method}")


def train_run(
    config: ProjectConfig,
    *,
    method: str,
    target: str,
    seed: int,
    force: bool = False,
) -> dict:
    import torch
    from transformers.trainer_utils import get_last_checkpoint

    method = method.lower()
    if method not in TRAINABLE_METHODS:
        raise ValueError(f"method must be one of {sorted(TRAINABLE_METHODS)}")
    if target != "control":
        target = to_basic(target)
    if seed not in config.training.seeds:
        raise ValueError(f"Seed {seed} is not in the registered paper seeds {config.training.seeds}")
    validate_fairness(config)

    output = checkpoint_dir(config, method, target, seed)
    done = output / "DONE"
    if done.exists() and not force:
        return json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(config)
    train_data, eval_data, train_path, eval_path = _dataset(config, method, target, tokenizer)
    manifest = {
        "status": "running",
        "method": method,
        "target": target,
        "seed": seed,
        "base_model": config.model.id,
        "config_sha256": config_fingerprint(config),
        "started_at_utc": utc_now(),
        "train_rows": len(train_data),
        "eval_rows": len(eval_data),
        "train_sha256": sha256_file(train_path),
        "eval_sha256": sha256_file(eval_path),
        "versions": _versions(),
        "hardware": hardware_info(),
        "official_hypo_commit": OFFICIAL_HYPO_COMMIT if method == "hypo" else None,
        "trainer_source": "official_hypo" if method == "hypo" else "trl",
    }
    write_json(output / "manifest.json", manifest)

    model = load_base_model(config)
    trainer = _build_trainer(config, method, model, tokenizer, train_data, eval_data, output, seed)
    manifest["parameters"] = trainable_parameters(trainer.model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    resume = None if force else get_last_checkpoint(str(output))
    result = trainer.train(resume_from_checkpoint=resume)
    elapsed = time.perf_counter() - started
    final_dir = output / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    trainer.state.save_to_json(str(output / "trainer_state.json"))

    manifest.update(
        {
            "status": "completed",
            "completed_at_utc": utc_now(),
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "train_metrics": result.metrics,
            "final_dir": str(final_dir),
        }
    )
    write_json(output / "manifest.json", manifest)
    done.write_text("completed\n", encoding="utf-8")
    release_model(trainer, model, tokenizer)
    return manifest
