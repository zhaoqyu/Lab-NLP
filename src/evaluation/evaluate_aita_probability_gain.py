#!/usr/bin/env python3
"""AITA extrinsic evaluation: accuracy and probability gain for value-consistent labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from value_alignment.model_utils import resolve_model_name


LABELS = ["NTA", "YTA", "Neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--trained-model", required=True)
    parser.add_argument("--test-file", type=Path, default=Path("value_alignment/data/aita_dpo/test.jsonl"))
    parser.add_argument("--model-aliases", type=Path, default=Path("value_alignment/configs/model_aliases.json"))
    parser.add_argument("--output-json", type=Path, default=Path("value_alignment/results/aita_probability_gain.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("value_alignment/results/aita_probability_gain.csv"))
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    return parser.parse_args()


def load_model(model_name: str, aliases_path: Path):
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
    model.eval()
    return model, tokenizer


def label_logprob(model, tokenizer, prompt: str, label: str, max_length: int) -> float:
    label_text = " " + label
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(model.device)
    label_ids = tokenizer(label_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    input_ids = torch.cat([prompt_ids, label_ids], dim=1)
    with torch.no_grad():
        logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    start = prompt_ids.shape[1] - 1
    total = 0.0
    for i, token_id in enumerate(label_ids[0]):
        total += float(log_probs[0, start + i, token_id])
    return total


def label_distribution(model, tokenizer, prompt: str, max_length: int) -> dict[str, float]:
    logps = {label: label_logprob(model, tokenizer, prompt, label, max_length) for label in LABELS}
    max_logp = max(logps.values())
    probs = {label: math.exp(value - max_logp) for label, value in logps.items()}
    total = sum(probs.values())
    return {label: prob / total for label, prob in probs.items()}


def expected_label(row: dict) -> str:
    if "high_standard_stance" in row:
        return row["high_standard_stance"]
    chosen = row["chosen"].strip()
    for label in LABELS:
        if chosen.startswith(label):
            return label
    raise ValueError("Could not infer expected high-standard label.")


def summarize(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["value"]].append(row)
    per_value = {}
    for value, items in sorted(grouped.items()):
        per_value[value] = {
            "count": len(items),
            "base_accuracy": sum(item["base_correct"] for item in items) / len(items),
            "trained_accuracy": sum(item["trained_correct"] for item in items) / len(items),
            "accuracy_gain": (sum(item["trained_correct"] for item in items) - sum(item["base_correct"] for item in items)) / len(items),
            "mean_probability_gain": sum(item["probability_gain"] for item in items) / len(items),
        }
    return {
        "overall": {
            "count": len(rows),
            "base_accuracy": sum(item["base_correct"] for item in rows) / len(rows) if rows else None,
            "trained_accuracy": sum(item["trained_correct"] for item in rows) / len(rows) if rows else None,
            "mean_probability_gain": sum(item["probability_gain"] for item in rows) / len(rows) if rows else None,
        },
        "per_value": per_value,
    }


def main() -> None:
    args = parse_args()
    examples = [json.loads(line) for line in args.test_file.open(encoding="utf-8")]
    if args.max_examples > 0:
        examples = examples[: args.max_examples]

    base_model, base_tokenizer = load_model(args.base_model, args.model_aliases)
    trained_model, trained_tokenizer = load_model(args.trained_model, args.model_aliases)

    rows = []
    for idx, row in enumerate(examples):
        high_label = expected_label(row)
        prompt = row["prompt"] + "\n\nAnswer:"
        base_probs = label_distribution(base_model, base_tokenizer, prompt, args.max_length)
        trained_probs = label_distribution(trained_model, trained_tokenizer, prompt, args.max_length)
        base_pred = max(base_probs, key=base_probs.get)
        trained_pred = max(trained_probs, key=trained_probs.get)
        rows.append(
            {
                "idx": idx,
                "value": row.get("value", "unknown"),
                "expected_label": high_label,
                "base_pred": base_pred,
                "trained_pred": trained_pred,
                "base_high_prob": base_probs[high_label],
                "trained_high_prob": trained_probs[high_label],
                "probability_gain": trained_probs[high_label] - base_probs[high_label],
                "base_correct": int(base_pred == high_label),
                "trained_correct": int(trained_pred == high_label),
            }
        )

    result = {"summary": summarize(rows), "rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["idx"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
