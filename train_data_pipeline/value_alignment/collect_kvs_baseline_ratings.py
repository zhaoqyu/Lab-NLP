#!/usr/bin/env python3
"""Estimate model-specific KVS baseline ratings by template majority vote."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from value_alignment.model_utils import resolve_model_name
from value_alignment.survey_data import extract_rating, iter_survey_variants, kvs_item_id, majority_rating
from value_alignment.value_taxonomy import basic_value_for_fine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, default=Path("dataset/kvs_data_new.json"))
    parser.add_argument("--model-aliases", type=Path, default=Path("value_alignment/configs/model_aliases.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-items", type=int, default=0, help="Optional global item cap for smoke tests.")
    return parser.parse_args()


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_items < 0:
        raise SystemExit("--batch-size must be positive and --max-items must be non-negative.")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    examples = []
    item_metadata = {}
    item_count = 0
    for split in ("train", "eval", "test"):
        for index, item in enumerate(data[split]):
            if args.max_items and item_count >= args.max_items:
                break
            source_id = kvs_item_id(split, index)
            item_metadata[source_id] = {
                "source_split": split,
                "fine_value": item["level2"][0],
                "value": basic_value_for_fine(item["level2"][0]),
            }
            examples.extend(
                iter_survey_variants(
                    item,
                    split,
                    index,
                    data["tasks"],
                    data["response_template"],
                )
            )
            item_count += 1
        if args.max_items and item_count >= args.max_items:
            break

    resolved_model = resolve_model_name(args.model, args.model_aliases)
    model, tokenizer = load_model(resolved_model)
    set_seed(args.seed)
    votes: dict[str, list[int]] = defaultdict(list)
    raw_generations: dict[str, list[str]] = defaultdict(list)

    for start in range(0, len(examples), args.batch_size):
        batch = examples[start : start + args.batch_size]
        prompts = [row["prompt"] for row in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
        prompt_length = inputs["input_ids"].shape[1]
        generations = tokenizer.batch_decode(output_ids[:, prompt_length:], skip_special_tokens=True)
        for row, generated in zip(batch, generations, strict=True):
            source_id = row["source_id"]
            raw_generations[source_id].append(generated.strip())
            rating = extract_rating(generated)
            if rating is not None:
                votes[source_id].append(rating)

    missing = sorted(set(item_metadata) - set(votes))
    if missing:
        raise RuntimeError(f"No valid 1-6 rating was generated for {len(missing)} items; first: {missing[0]}")

    ratings = {source_id: majority_rating(item_votes) for source_id, item_votes in votes.items()}
    details = {
        source_id: {
            **item_metadata[source_id],
            "rating": ratings[source_id],
            "votes": votes[source_id],
            "valid_votes": len(votes[source_id]),
            "total_templates": len(raw_generations[source_id]),
            "generations": raw_generations[source_id],
        }
        for source_id in sorted(item_metadata)
    }
    result = {
        "model": resolved_model,
        "input": str(args.input),
        "templates_per_description": len(data["tasks"]) * len(data["response_template"]),
        "temperature": args.temperature,
        "seed": args.seed,
        "ratings": ratings,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "model": resolved_model,
                "items": len(ratings),
                "template_generations": len(examples),
                "rating_distribution": dict(sorted(Counter(ratings.values()).items())),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
