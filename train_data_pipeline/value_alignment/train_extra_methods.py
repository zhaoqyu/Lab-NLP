#!/usr/bin/env python3
"""Train SimPO or KTO adapters on the same target-specific KVS preference data
used by ``train_with_official_hypo.py`` (DPO/HyPO).

Both methods are implemented with the stock TRL 0.9.6 trainers so no extra
third-party repository is required:

- SimPO is CPOTrainer with ``loss_type="simpo"`` and ``cpo_alpha=0`` (pure
  SimPO, no BC/NLL regularizer). It is reference-free: no reference model is
  ever loaded, matching the paper description of SimPO.
- KTO is KTOTrainer. It consumes unpaired (prompt, completion, label) rows,
  so this script explodes each chosen/rejected pair from the DPO-format
  JSONL into one desirable row (label=True, the chosen completion) and one
  undesirable row (label=False, the rejected completion). With PEFT and no
  explicit --ref-model, KTOTrainer (like DPOTrainer) uses the policy with its
  LoRA adapter disabled as the frozen reference, so no second copy of the
  model is loaded here either.

Output layout, run manifests, and CLI conventions mirror
``train_with_official_hypo.py`` so this drops into the same pipeline
(evaluation scripts only look for ``<output_dir>/<method>/.../final``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback

from value_alignment.experiment_utils import tagged_run_dir, write_run_manifest
from value_alignment.model_utils import resolve_model_name


DEFAULT_BETA = {"simpo": 2.0, "kto": 0.1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file. CLI args override config values.")
    parser.add_argument("--method", choices=["simpo", "kto"], required=True)
    parser.add_argument("--model", default="qwen2.5-7b")
    parser.add_argument("--model-aliases", type=Path, default=Path("value_alignment/configs/model_aliases.json"))
    parser.add_argument("--train-file", type=Path, required=True, help="DPO-format JSONL with prompt/chosen/rejected.")
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("value_alignment/checkpoints/paper_preference"))
    parser.add_argument("--run-tag", default="", help="Optional isolated run name; outputs go under <method>/runs/<run-tag>.")
    parser.add_argument("--beta", type=float, default=None, help="Defaults to 2.0 for simpo, 0.1 for kto.")
    parser.add_argument("--simpo-gamma", type=float, default=1.0, help="SimPO target reward margin (simpo only).")
    parser.add_argument("--cpo-alpha", type=float, default=0.0, help="BC/NLL regularizer weight; 0 = pure SimPO (simpo only).")
    parser.add_argument("--desirable-weight", type=float, default=1.0, help="KTO only.")
    parser.add_argument("--undesirable-weight", type=float, default=1.0, help="KTO only.")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use NF4 QLoRA.")
    args = parser.parse_args()
    if args.config is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        defaults = vars(parser.parse_args(["--method", args.method, "--train-file", str(args.train_file), "--eval-file", str(args.eval_file)]))
        for key, value in config.items():
            attr = key.replace("-", "_")
            if hasattr(args, attr) and getattr(args, attr) == defaults.get(attr):
                setattr(args, attr, value)
    if args.beta is None:
        args.beta = DEFAULT_BETA[args.method]
    for path_attr in ["train_file", "eval_file", "output_dir", "model_aliases"]:
        setattr(args, path_attr, Path(getattr(args, path_attr)))
    return args


def load_dpo_dataset(train_file: Path, eval_file: Path):
    return load_dataset("json", data_files={"train": str(train_file), "eval": str(eval_file)})


def to_kto_dataset(dpo_dataset):
    """Explode prompt/chosen/rejected rows into unpaired prompt/completion/label rows."""

    def explode(batch):
        prompts, completions, labels = [], [], []
        for prompt, chosen, rejected in zip(batch["prompt"], batch["chosen"], batch["rejected"]):
            prompts.extend([prompt, prompt])
            completions.extend([chosen, rejected])
            labels.extend([True, False])
        return {"prompt": prompts, "completion": completions, "label": labels}

    keep_columns = {"prompt", "chosen", "rejected"}
    return dpo_dataset.map(
        explode,
        batched=True,
        remove_columns=[c for c in dpo_dataset["train"].column_names if c in keep_columns or c not in ("prompt", "completion", "label")],
    )


def main() -> None:
    args = parse_args()
    from trl import CPOConfig, CPOTrainer, KTOConfig, KTOTrainer

    dataset = load_dpo_dataset(args.train_file, args.eval_file)
    if args.method == "kto":
        dataset = to_kto_dataset(dataset)

    model_name = resolve_model_name(args.model, args.model_aliases)
    method_dir = tagged_run_dir(args.output_dir / args.method, args.run_tag)
    manifest_inputs = {
        "train_data": args.train_file,
        "eval_data": args.eval_file,
        "model_aliases": args.model_aliases,
    }
    manifest_metadata = {
        "trainer": f"trl_{args.method}",
        "method": args.method,
        "resolved_model": model_name,
        "effective_lora_alpha": args.lora_alpha,
        "effective_output_dir": str(method_dir),
    }
    write_run_manifest(method_dir, args, manifest_inputs, metadata=manifest_metadata)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {"torch_dtype": dtype}
    if args.load_in_4bit:
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

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    common_kwargs = {
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
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "save_total_limit": 2,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "report_to": [],
        "seed": args.seed,
    }

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        )
    ]

    if args.method == "simpo":
        training_args = CPOConfig(
            **common_kwargs,
            loss_type="simpo",
            cpo_alpha=args.cpo_alpha,
            simpo_gamma=args.simpo_gamma,
            metric_for_best_model="eval_rewards/margins",
            greater_is_better=True,
        )
        trainer = CPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            tokenizer=tokenizer,
            peft_config=peft_config,
            callbacks=callbacks,
        )
    else:
        training_args = KTOConfig(
            **common_kwargs,
            desirable_weight=args.desirable_weight,
            undesirable_weight=args.undesirable_weight,
            metric_for_best_model="eval_rewards/margins",
            greater_is_better=True,
        )
        trainer = KTOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            tokenizer=tokenizer,
            peft_config=peft_config,
            callbacks=callbacks,
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
