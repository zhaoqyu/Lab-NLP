#!/usr/bin/env python3
"""Train standard DPO or HyPO using the official HyPO trainer code.

This script imports `DPOTrainer` and `DPOConfig` from the official repository
cloned at `third_party/2026_ICLR_HyPO`. It avoids the Alignment Handbook launcher so
we can train on the local KVS preference JSONL files used in this lab project.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback

from value_alignment.experiment_utils import git_metadata, tagged_run_dir, write_run_manifest
from value_alignment.model_utils import resolve_model_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_HYPO_DIR = Path(os.environ.get("HYPO_REPO", PROJECT_ROOT / "third_party" / "2026_ICLR_HyPO"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file. CLI args override config values.")
    parser.add_argument("--method", choices=["dpo", "hypo"], default="hypo")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--ref-model", default=None)
    parser.add_argument("--model-aliases", type=Path, default=Path("value_alignment/configs/model_aliases.json"))
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("value_alignment/data/paper_preferences/security/down/train.jsonl"),
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("value_alignment/data/paper_preferences/security/down/eval.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("value_alignment/checkpoints"))
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional isolated run name; outputs go under <method>/runs/<run-tag>.",
    )
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.15)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--lora-r", type=int, default=256)
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=0,
        help="0 selects 1024 for Falcon3 and 512 for other paper models.",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use NF4 QLoRA.")
    parser.add_argument("--no-lora", action="store_true")
    args = parser.parse_args()
    if args.config is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        defaults = vars(parser.parse_args([]))
        for key, value in config.items():
            attr = key.replace("-", "_")
            if attr == "lora":
                if getattr(args, "no_lora") == defaults.get("no_lora"):
                    args.no_lora = not bool(value)
                continue
            if hasattr(args, attr) and getattr(args, attr) == defaults.get(attr):
                setattr(args, attr, value)
    for path_attr in ["train_file", "eval_file", "output_dir", "model_aliases"]:
        setattr(args, path_attr, Path(getattr(args, path_attr)))
    return args


def default_lora_alpha(model_name: str) -> int:
    return 1024 if "falcon3" in model_name.lower() else 512


def main() -> None:
    args = parse_args()
    if not OFFICIAL_HYPO_DIR.exists():
        raise SystemExit(
            "Official HyPO repo not found. Clone it with:\n"
            "  git clone https://github.com/tmllab/2026_ICLR_HyPO.git third_party/2026_ICLR_HyPO\n"
            "or set HYPO_REPO=/path/to/2026_ICLR_HyPO"
        )
    sys.path.insert(0, str(OFFICIAL_HYPO_DIR))
    from hypo_config import DPOConfig
    from hypo_trainer import DPOTrainer

    dataset = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "eval": str(args.eval_file)},
    )

    model_name = resolve_model_name(args.model, args.model_aliases)
    ref_model_name = resolve_model_name(args.ref_model, args.model_aliases) if args.ref_model else None
    method_dir = tagged_run_dir(args.output_dir / args.method, args.run_tag)
    lora_alpha = args.lora_alpha or default_lora_alpha(model_name)
    manifest_inputs = {
        "train_data": args.train_file,
        "eval_data": args.eval_file,
        "model_aliases": args.model_aliases,
        "official_hypo_config": OFFICIAL_HYPO_DIR / "hypo_config.py",
        "official_hypo_trainer": OFFICIAL_HYPO_DIR / "hypo_trainer.py",
    }
    manifest_metadata = {
        "trainer": "official_hypo",
        "method": args.method,
        "resolved_model": model_name,
        "resolved_reference_model": ref_model_name or model_name,
        "reference_implementation": git_metadata(OFFICIAL_HYPO_DIR),
        "effective_lora_alpha": None if args.no_lora else lora_alpha,
        "effective_output_dir": str(method_dir),
    }
    write_run_manifest(
        method_dir,
        args,
        manifest_inputs,
        metadata=manifest_metadata,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {"torch_dtype": dtype}
    if args.load_in_4bit:
        if args.no_lora:
            raise ValueError("--load-in-4bit requires LoRA; remove --no-lora.")
        if not torch.cuda.is_available():
            raise RuntimeError("--load-in-4bit requires a CUDA GPU and bitsandbytes.")
        model_kwargs.update(
            {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                ),
                "device_map": {"": torch.cuda.current_device()},
            }
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if args.gradient_checkpointing:
        model.config.use_cache = False

    ref_model = None
    if ref_model_name and not args.no_lora:
        raise ValueError(
            "The official trainer uses the policy with its LoRA adapter disabled as the frozen "
            "reference. Use --no-lora before supplying a separate --ref-model."
        )
    if ref_model_name:
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name, **model_kwargs)

    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    training_kwargs = {
        "output_dir": str(method_dir),
        "beta": args.beta,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_rewards/margins",
        "greater_is_better": True,
        "save_total_limit": 2,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "report_to": [],
        "seed": args.seed,
        "im_enable": args.method == "hypo",
        "im_gamma": args.gamma,
        "im_tau": args.tau,
    }
    config_parameters = inspect.signature(DPOConfig.__init__).parameters
    strategy_key = (
        "eval_strategy" if "eval_strategy" in config_parameters else "evaluation_strategy"
    )
    training_kwargs[strategy_key] = "epoch"
    training_args = DPOConfig(**training_kwargs)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        tokenizer=tokenizer,
        peft_config=peft_config,
        loss_type="sigmoid",
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        ],
    )

    trainer.train()
    final_dir = method_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    write_run_manifest(
        method_dir,
        args,
        manifest_inputs,
        metadata={**manifest_metadata, "final_model_dir": str(final_dir)},
        status="completed",
    )


if __name__ == "__main__":
    main()
