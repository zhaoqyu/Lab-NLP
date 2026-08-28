#!/usr/bin/env python3
"""LoRA survey-SFT training for baseline and target-specific KVS datasets."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from value_alignment.model_utils import resolve_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-aliases", type=Path, default=Path("value_alignment/configs/model_aliases.json"))
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=256)
    parser.add_argument("--lora-alpha", type=int, default=0, help="0 selects the paper setting by model family.")
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use NF4 QLoRA instead of bf16 LoRA.")
    return parser.parse_args()


def default_lora_alpha(model_name: str) -> int:
    return 1024 if "falcon3" in model_name.lower() else 512


def tokenize_example(row: dict, tokenizer, max_length: int) -> dict:
    prompt_ids = tokenizer(row["prompt"], add_special_tokens=True, truncation=False)["input_ids"]
    response_text = row["response"] + (tokenizer.eos_token or "")
    response_ids = tokenizer(response_text, add_special_tokens=False, truncation=False)["input_ids"]
    if len(response_ids) >= max_length:
        raise ValueError(f"Survey response is longer than --max-length for row {row.get('id')}")
    prompt_ids = prompt_ids[: max_length - len(response_ids)]
    input_ids = prompt_ids + response_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + response_ids,
    }


def make_training_args(args: argparse.Namespace) -> TrainingArguments:
    kwargs = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": [],
        "remove_unused_columns": False,
        "seed": args.seed,
    }
    parameter_names = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in parameter_names else "evaluation_strategy"
    kwargs[strategy_key] = "epoch"
    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    resolved_model = resolve_model_name(args.model, args.model_aliases)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if args.load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError("--load-in-4bit requires a CUDA GPU and bitsandbytes.")
    model_kwargs = {"torch_dtype": dtype}
    if args.load_in_4bit:
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
    model = AutoModelForCausalLM.from_pretrained(resolved_model, **model_kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_alpha = args.lora_alpha or default_lora_alpha(resolved_model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    dataset = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "eval": str(args.eval_file)},
    )
    tokenized = dataset.map(
        lambda row: tokenize_example(row, tokenizer, args.max_length),
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing survey SFT data",
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    trainer_kwargs = {
        "model": model,
        "args": make_training_args(args),
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["eval"],
        "data_collator": collator,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        ],
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in trainer_parameters else "tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()
