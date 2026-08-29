#!/usr/bin/env python3
"""Evaluate one target-specific checkpoint with the paper's AITA metric."""

from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from value_alignment.activation_steering import SteeringHook, load_steering_selection
from value_alignment.evaluation.aita_metrics import LABELS, probability_gain, summarize
from value_alignment.model_utils import resolve_model_name
from value_alignment.value_taxonomy import canonical_basic_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="Matched baseline-tuned checkpoint or base model.")
    parser.add_argument("--conditioned-model", required=True)
    parser.add_argument(
        "--base-adapter",
        type=Path,
        default=None,
        help="Optional baseline-control PEFT adapter loaded on --base-model.",
    )
    parser.add_argument(
        "--conditioned-adapter",
        type=Path,
        default=None,
        help="Optional target-specific PEFT adapter loaded on --conditioned-model.",
    )
    parser.add_argument(
        "--conditioned-steering-selection",
        type=Path,
        default=None,
        help="Optional JSON produced by select_steering_layer.py.",
    )
    parser.add_argument("--target-value", required=True)
    parser.add_argument("--test-file", type=Path, default=Path("value_alignment/data/aita_eval/test.jsonl"))
    parser.add_argument("--model-aliases", type=Path, default=Path("value_alignment/configs/model_aliases.json"))
    parser.add_argument("--output-json", type=Path, default=Path("value_alignment/results/aita_probability_gain.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("value_alignment/results/aita_probability_gain.csv"))
    parser.add_argument("--unused-stance-weight", type=float, default=0.5)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--expected-direction",
        choices=["greater", "less"],
        default="greater",
        help="Use less for up-regulation because Probability Gain remains down-oriented.",
    )
    return parser.parse_args()


def load_model(model_name: str, aliases_path: Path, adapter: Path | None = None):
    resolved = resolve_model_name(model_name, aliases_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tokenizer, resolved


def label_logprob(
    model,
    tokenizer,
    prompt: str,
    label: str,
    max_length: int,
    steering: SteeringHook | None = None,
) -> float:
    label_ids = tokenizer(" " + label, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    prompt_budget = max_length - label_ids.shape[1]
    if prompt_budget < 1:
        raise ValueError("--max-length is too small to hold a label completion.")
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_budget,
    ).input_ids.to(model.device)
    input_ids = torch.cat([prompt_ids, label_ids], dim=1)
    if steering is not None:
        steering.set_start_positions([prompt_ids.shape[1] - 1])
    with torch.no_grad():
        logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    start = prompt_ids.shape[1] - 1
    return sum(float(log_probs[0, start + offset, token_id]) for offset, token_id in enumerate(label_ids[0]))


def label_distribution(
    model,
    tokenizer,
    prompt: str,
    max_length: int,
    steering: SteeringHook | None = None,
) -> dict[str, float]:
    logps = {
        label: label_logprob(model, tokenizer, prompt, label, max_length, steering)
        for label in LABELS
    }
    max_logp = max(logps.values())
    unnormalized = {label: math.exp(value - max_logp) for label, value in logps.items()}
    denominator = sum(unnormalized.values())
    return {label: value / denominator for label, value in unnormalized.items()}


def main() -> None:
    args = parse_args()
    target_value = canonical_basic_value(args.target_value)
    examples = [
        row
        for row in (json.loads(line) for line in args.test_file.open(encoding="utf-8"))
        if canonical_basic_value(row["value"]) == target_value
    ]
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    if not examples:
        raise SystemExit(f"No AITA examples found for target value {target_value}.")

    base_model, base_tokenizer, resolved_base = load_model(
        args.base_model,
        args.model_aliases,
        args.base_adapter,
    )
    conditioned_model, conditioned_tokenizer, resolved_conditioned = load_model(
        args.conditioned_model,
        args.model_aliases,
        args.conditioned_adapter,
    )
    conditioned_steering = None
    steering_metadata = None
    if args.conditioned_steering_selection is not None:
        if args.conditioned_adapter is not None:
            raise ValueError(
                "Do not combine --conditioned-adapter with --conditioned-steering-selection."
            )
        steering_metadata = load_steering_selection(args.conditioned_steering_selection)
        conditioned_steering = SteeringHook(
            conditioned_model,
            steering_metadata["site"],
            steering_metadata["selected_layer"],
            steering_metadata["vector"],
            steering_metadata["strength"],
        )

    rows = []
    with conditioned_steering if conditioned_steering is not None else nullcontext():
        for index, row in enumerate(examples):
            high_label = row["high_standard_stance"]
            low_label = row["low_standard_stance"]
            prompt = row["prompt"] + "\n\nAnswer:"
            base_probs = label_distribution(base_model, base_tokenizer, prompt, args.max_length)
            conditioned_probs = label_distribution(
                conditioned_model,
                conditioned_tokenizer,
                prompt,
                args.max_length,
                conditioned_steering,
            )
            gain = probability_gain(
                base_probs,
                conditioned_probs,
                high_label,
                low_label,
                args.unused_stance_weight,
            )
            strict_gain = probability_gain(
                base_probs,
                conditioned_probs,
                high_label,
                low_label,
                0.0,
            )
            unused_label = next(
                label for label in LABELS if label not in {high_label, low_label}
            )
            rows.append(
                {
                    "idx": index,
                    "id": row.get("id", index),
                    "value": target_value,
                    "fine_value": row.get("fine_value"),
                    "high_standard_label": high_label,
                    "low_standard_label": low_label,
                    "unused_label": unused_label,
                    "base_high_prob": base_probs[high_label],
                    "conditioned_high_prob": conditioned_probs[high_label],
                    "base_low_prob": base_probs[low_label],
                    "conditioned_low_prob": conditioned_probs[low_label],
                    "base_unused_prob": base_probs[unused_label],
                    "conditioned_unused_prob": conditioned_probs[unused_label],
                    "probability_gain": gain,
                    "strict_probability_gain": strict_gain,
                }
            )

    result = {
        "metric": {
            "name": "AITA Probability Gain",
            "definition": "delta(low) - delta(high) + unused_weight * delta(unused)",
            "unused_stance_weight": args.unused_stance_weight,
            "successful_raw_direction": (
                "positive Probability Gain"
                if args.expected_direction == "greater"
                else "negative Probability Gain"
            ),
            "expected_direction": args.expected_direction,
        },
        "target_value": target_value,
        "base_model": resolved_base,
        "base_adapter": str(args.base_adapter) if args.base_adapter is not None else None,
        "conditioned_model": resolved_conditioned,
        "conditioned_adapter": (
            str(args.conditioned_adapter) if args.conditioned_adapter is not None else None
        ),
        "conditioned_steering": (
            {
                key: value
                for key, value in steering_metadata.items()
                if key != "vector"
            }
            if steering_metadata is not None
            else None
        ),
        "test_file": str(args.test_file),
        "summary": summarize(
            rows,
            args.bootstrap_replicates,
            args.bootstrap_seed,
            args.expected_direction,
        ),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
