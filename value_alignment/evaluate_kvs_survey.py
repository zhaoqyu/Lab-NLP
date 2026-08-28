#!/usr/bin/env python3
"""Evaluate a base model or LoRA adapter on the held-out KVS survey."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from value_alignment.model_utils import resolve_model_name
from value_alignment.survey_data import extract_rating, majority_rating


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model alias, Hub ID, or local model path.")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Optional PEFT adapter. When set, --model is the adapter's base model.",
    )
    parser.add_argument(
        "--model-aliases",
        type=Path,
        default=Path("value_alignment/configs/model_aliases.json"),
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("value_alignment/data/kvs_survey/test.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("value_alignment/results/kvs_scores.json"),
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test limit.")
    return parser.parse_args()


def load_model(model_name: str, adapter: Path | None):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model


def batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def generate_run(
    model,
    tokenizer,
    rows: list[dict],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[list[int | None], list[str]]:
    ratings: list[int | None] = []
    generations: list[str] = []
    for batch in batched(rows, batch_size):
        encoded = tokenizer(
            [row["prompt"] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        with torch.inference_mode():
            output_ids = model.generate(**encoded, **generation_kwargs)

        prompt_width = encoded["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(
            output_ids[:, prompt_width:],
            skip_special_tokens=True,
        )
        generations.extend(text.strip() for text in decoded)
        ratings.extend(extract_rating(text) for text in decoded)
    return ratings, generations


def summarize(scored_rows: list[dict], run_count: int) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in scored_rows:
        grouped[row["value"]].append(row)

    summary = {}
    for value, value_rows in sorted(grouped.items()):
        final_ratings = [row["rating"] for row in value_rows if row["rating"] is not None]
        run_means = []
        for run_index in range(run_count):
            run_ratings = [
                row["ratings"][run_index]
                for row in value_rows
                if row["ratings"][run_index] is not None
            ]
            run_means.append(sum(run_ratings) / len(run_ratings) if run_ratings else None)
        valid_run_means = [mean for mean in run_means if mean is not None]
        summary[value] = {
            "count": len(value_rows),
            "valid_count": len(final_ratings),
            "invalid_count": len(value_rows) - len(final_ratings),
            "mean_rating": sum(final_ratings) / len(final_ratings) if final_ratings else None,
            "run_means": run_means,
            "run_mean_sample_std": (
                statistics.stdev(valid_run_means) if len(valid_run_means) > 1 else 0.0
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    if args.num_runs < 1:
        raise ValueError("--num-runs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    model_name = resolve_model_name(args.model, args.model_aliases)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(model_name, args.adapter)

    with args.eval_file.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    all_ratings: list[list[int | None]] = [[] for _ in rows]
    all_generations: list[list[str]] = [[] for _ in rows]
    for run_index in range(args.num_runs):
        set_seed(args.seed + run_index)
        ratings, generations = generate_run(
            model,
            tokenizer,
            rows,
            args.batch_size,
            args.max_new_tokens,
            args.temperature,
        )
        for index, (rating, generated) in enumerate(zip(ratings, generations, strict=True)):
            all_ratings[index].append(rating)
            all_generations[index].append(generated)

    scored_rows = []
    for row, ratings, generations in zip(rows, all_ratings, all_generations, strict=True):
        valid_ratings = [rating for rating in ratings if rating is not None]
        final_rating = majority_rating(valid_ratings) if valid_ratings else None
        scored_rows.append(
            {
                **row,
                "ratings": ratings,
                "rating": final_rating,
                "generations": generations,
            }
        )

    result = {
        "model": model_name,
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "settings": {
            "num_runs": args.num_runs,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "summary": summarize(scored_rows, args.num_runs),
        "rows": scored_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["id", "source_id", "value", "fine_value", "rating", "ratings", "generations"]
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in scored_rows:
                writer.writerow(
                    {
                        field: json.dumps(row.get(field), ensure_ascii=False)
                        if isinstance(row.get(field), list)
                        else row.get(field)
                        for field in fields
                    }
                )

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
