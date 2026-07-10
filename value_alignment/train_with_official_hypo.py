#!/usr/bin/env python3
"""Train standard DPO or HyPO using the official HyPO trainer code.

This script imports `DPOTrainer` and `DPOConfig` from the official repository
cloned at `third_party/2026_ICLR_HyPO`. It avoids the Alignment Handbook launcher so
we can train on the local AITA JSONL files used in this lab project.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from value_alignment.model_utils import resolve_model_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_HYPO_DIR = Path(os.environ.get("HYPO_REPO", PROJECT_ROOT / "third_party" / "2026_ICLR_HyPO"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file. CLI args override config values.")
    parser.add_argument("--method", choices=["dpo", "hypo"], default="hypo")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--ref-model", default=None)
    parser.add_argument("--model-aliases", type=Path, default=Path("value_alignment/configs/model_aliases.json"))
    parser.add_argument("--train-file", type=Path, default=Path("value_alignment/data/aita_dpo/train.jsonl"))
    parser.add_argument("--eval-file", type=Path, default=Path("value_alignment/data/aita_dpo/eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("value_alignment/checkpoints"))
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    ref_model = None
    if ref_model_name:
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name, **model_kwargs)

    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    method_dir = args.output_dir / args.method
    training_args = DPOConfig(
        output_dir=str(method_dir),
        beta=args.beta,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        evaluation_strategy="steps",
        save_strategy="steps",
        remove_unused_columns=False,
        report_to=[],
        im_enable=args.method == "hypo",
        im_gamma=args.gamma,
        im_tau=args.tau,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        tokenizer=tokenizer,
        peft_config=peft_config,
        loss_type="sigmoid",
    )

    trainer.train()
    trainer.save_model(str(method_dir / "final"))
    tokenizer.save_pretrained(str(method_dir / "final"))


if __name__ == "__main__":
    main()
